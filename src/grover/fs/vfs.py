"""VFS — mount router with routing, permissions, events, capabilities."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from grover.events import EventBus, EventType, FileEvent

from .exceptions import AuthenticationRequiredError, CapabilityNotSupportedError, MountNotFoundError
from .permissions import Permission
from .protocol import SupportsReconcile, SupportsTrash, SupportsVersions
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    GlobResult,
    GrepMatch,
    GrepResult,
    ListResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    TreeResult,
    WriteResult,
)
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from .mounts import MountConfig, MountRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VFS:
    """Routes operations to backends via mount registry.

    Presents a single namespace to callers while delegating to the
    appropriate backend based on the path prefix. Enforces permissions,
    handles cross-mount copy/move, and provides capability gating.

    Session lifecycle is per-operation only.  No transaction mode.
    """

    def __init__(self, registry: MountRegistry, event_bus: EventBus | None = None) -> None:
        self._registry = registry
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    def _get_capability(self, backend: Any, protocol: type[T]) -> T | None:
        if isinstance(backend, protocol):
            return backend
        return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def _emit(self, event: FileEvent) -> None:
        if self._event_bus is not None:
            await self._event_bus.emit(event)

    # ------------------------------------------------------------------
    # Session Management (per-operation only)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _session_for(self, mount: MountConfig) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or None for non-SQL."""
        if not mount.has_session_factory:
            yield None
            return

        assert mount.session_factory is not None
        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all backends."""
        for mount in self._registry.list_mounts():
            if hasattr(mount.backend, "close"):
                try:
                    await mount.backend.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.mount_path, exc_info=True)

    # ------------------------------------------------------------------
    # Path Helpers
    # ------------------------------------------------------------------

    def _prefix_path(self, path: str | None, mount_path: str) -> str | None:
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def _prefix_file_info(self, info: FileInfo, mount: MountConfig) -> FileInfo:
        prefixed_path = self._prefix_path(info.path, mount.mount_path) or info.path
        info.path = prefixed_path
        info.mount_type = mount.mount_type
        info.permission = self._registry.get_permission(prefixed_path).value
        return info

    def _check_writable(self, virtual_path: str) -> None:
        perm = self._registry.get_permission(virtual_path)
        if perm == Permission.READ_ONLY:
            raise PermissionError(f"Cannot write to read-only path: {virtual_path}")

    # ------------------------------------------------------------------
    # User-scoped path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _is_shared_access(rel_path: str) -> tuple[bool, str | None, str | None]:
        """Parse ``/@shared/{owner}/{rest}`` from a relative path.

        Returns ``(is_shared, owner, rest_path)``.
        """
        segments = rel_path.strip("/").split("/")
        if len(segments) >= 2 and segments[0] == "@shared":
            owner = segments[1]
            rest = "/" + "/".join(segments[2:]) if len(segments) > 2 else "/"
            return True, owner, rest
        return False, None, None

    def _resolve_user_path(
        self, mount: MountConfig, rel_path: str, user_id: str | None
    ) -> str:
        """Rewrite *rel_path* for an authenticated mount.

        - Regular mounts: pass through unchanged.
        - Authenticated mounts: prepend ``/{user_id}/`` to the path.
        - ``@shared/{owner}/`` paths: resolve to ``/{owner}/``.
        - Missing ``user_id`` on authenticated mount: raise.
        """
        if not mount.authenticated:
            return rel_path

        if not user_id:
            raise AuthenticationRequiredError(
                "user_id is required for authenticated mount"
            )

        is_shared, owner, rest = self._is_shared_access(rel_path)
        if is_shared and owner is not None and rest is not None:
            return f"/{owner}{rest}" if rest != "/" else f"/{owner}"

        if rel_path == "/":
            return f"/{user_id}"
        return f"/{user_id}{rel_path}"

    @staticmethod
    def _strip_user_prefix(path: str, user_id: str) -> str:
        """Remove ``/{user_id}`` prefix from a backend path."""
        prefix = f"/{user_id}/"
        if path.startswith(prefix):
            return "/" + path[len(prefix):]
        if path == f"/{user_id}":
            return "/"
        return path

    def _restore_user_path(
        self, stored_path: str | None, mount: MountConfig, user_id: str | None
    ) -> str | None:
        """Convert a stored backend path back to a user-facing virtual path.

        Strips the ``/{user_id}/`` prefix for authenticated mounts and
        re-applies the mount prefix.
        """
        if stored_path is None:
            return None
        if not mount.authenticated or user_id is None:
            return self._prefix_path(stored_path, mount.mount_path)
        stripped = self._strip_user_prefix(stored_path, user_id)
        return self._prefix_path(stripped, mount.mount_path)

    def _restore_file_info(
        self, info: FileInfo, mount: MountConfig, user_id: str | None
    ) -> FileInfo:
        """Like ``_prefix_file_info`` but strips user prefix for authenticated mounts."""
        if not mount.authenticated or user_id is None:
            return self._prefix_file_info(info, mount)
        stripped = self._strip_user_prefix(info.path, user_id)
        info.path = self._prefix_path(stripped, mount.mount_path) or info.path
        info.mount_type = mount.mount_type
        info.permission = self._registry.get_permission(mount.mount_path).value
        return info

    async def _check_share_access(
        self,
        session: AsyncSession | None,
        mount: MountConfig,
        stored_path: str,
        user_id: str,
        required: str = "read",
    ) -> None:
        """Verify that *user_id* has shared access to *stored_path*.

        Raises ``PermissionError`` if the mount has a SharingService and
        the user lacks the required permission.  No-op if no SharingService
        is configured.
        """
        if mount.sharing is None or session is None:
            return
        has_access = await mount.sharing.check_permission(
            session, stored_path, user_id, required=required,
        )
        if not has_access:
            raise PermissionError(
                f"Access denied: {user_id!r} does not have {required!r} "
                f"permission on shared path"
            )

    # ------------------------------------------------------------------
    # Read Operations
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        user_id: str | None = None,
    ) -> ReadResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "read")
            result = await mount.backend.read(rel_path, offset, limit, session=sess)
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        return result

    async def list_dir(
        self, path: str = "/", *, user_id: str | None = None
    ) -> ListResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, original_rel = self._registry.resolve(path)

        # Handle @shared virtual directories on authenticated mounts
        if mount.authenticated and user_id:
            segments = original_rel.strip("/").split("/")
            if segments[0] == "@shared":
                return await self._list_shared_dir(mount, segments, user_id)

        rel_path = self._resolve_user_path(mount, original_rel, user_id)
        async with self._session_for(mount) as sess:
            result = await mount.backend.list_dir(rel_path, session=sess)

        result.path = self._restore_user_path(result.path, mount, user_id) or path
        result.entries = [
            self._restore_file_info(entry, mount, user_id)
            for entry in result.entries
        ]

        # On authenticated mount root, add virtual @shared/ entry
        if mount.authenticated and user_id is not None and original_rel == "/":
            result.entries.append(
                FileInfo(
                    path=mount.mount_path + "/@shared",
                    name="@shared",
                    is_directory=True,
                    permission=mount.permission.value,
                    mount_type=mount.mount_type,
                )
            )

        return result

    async def _list_shared_dir(
        self,
        mount: MountConfig,
        segments: list[str],
        user_id: str,
    ) -> ListResult:
        """List virtual ``@shared/`` directories.

        - ``/@shared`` → list distinct owners who shared with *user_id*
        - ``/@shared/{owner}`` → list that owner's shared files accessible to *user_id*
        - ``/@shared/{owner}/sub/...`` → list_dir on the owner's sub-path (permission-checked)
        """
        if mount.sharing is None:
            return ListResult(
                success=True,
                message="No sharing configured",
                entries=[],
                path=mount.mount_path + "/@shared",
            )

        if len(segments) == 1:
            # /@shared — list distinct owners
            async with self._session_for(mount) as sess:
                assert sess is not None
                shares = await mount.sharing.list_shared_with(sess, user_id)
            # Extract distinct owners from share paths
            owners: set[str] = set()
            for share in shares:
                # Share paths are stored as /{owner}/... — extract owner
                parts = share.path.strip("/").split("/")
                if parts:
                    owners.add(parts[0])
            entries = [
                FileInfo(
                    path=f"{mount.mount_path}/@shared/{owner}",
                    name=owner,
                    is_directory=True,
                    permission=mount.permission.value,
                    mount_type=mount.mount_type,
                )
                for owner in sorted(owners)
            ]
            return ListResult(
                success=True,
                message=f"Found {len(entries)} shared owner(s)",
                entries=entries,
                path=mount.mount_path + "/@shared",
            )

        # /@shared/{owner}/... — resolve to /{owner}/... and list
        owner = segments[1]
        sub_path = "/" + "/".join(segments[2:]) if len(segments) > 2 else "/"
        stored_path = f"/{owner}{sub_path}" if sub_path != "/" else f"/{owner}"

        async with self._session_for(mount) as sess:
            assert sess is not None
            # Check that user has read access to this owner's path
            await self._check_share_access(sess, mount, stored_path, user_id, "read")
            result = await mount.backend.list_dir(stored_path, session=sess)

        # Rewrite paths: /{owner}/x → {mount}/@shared/{owner}/x
        shared_prefix = f"{mount.mount_path}/@shared/{owner}"
        result.path = shared_prefix + sub_path if sub_path != "/" else shared_prefix
        for entry in result.entries:
            # Strip /{owner} prefix and prepend @shared/{owner}
            stripped = self._strip_user_prefix(entry.path, owner)
            entry.path = shared_prefix + stripped if stripped != "/" else shared_prefix
            entry.mount_type = mount.mount_type
            entry.permission = mount.permission.value
        return result

    def _list_root(self) -> ListResult:
        entries: list[FileInfo] = []
        for mount in self._registry.list_visible_mounts():
            name = mount.mount_path.lstrip("/")
            entries.append(
                FileInfo(
                    path=mount.mount_path,
                    name=name,
                    is_directory=True,
                    permission=mount.permission.value,
                    mount_type=mount.mount_type,
                )
            )
        return ListResult(
            success=True,
            message=f"Found {len(entries)} mount(s)",
            entries=entries,
            path="/",
        )

    async def exists(
        self, path: str, *, user_id: str | None = None
    ) -> bool:
        path = normalize_path(path)

        if path == "/":
            return True

        if self._registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return False

        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                try:
                    await self._check_share_access(sess, mount, rel_path, user_id, "read")
                except PermissionError:
                    return False
            return await mount.backend.exists(rel_path, session=sess)

    async def get_info(
        self, path: str, *, user_id: str | None = None
    ) -> FileInfo | None:
        path = normalize_path(path)

        if self._registry.has_mount(path):
            for mount in self._registry.list_mounts():
                if mount.mount_path == path:
                    name = mount.mount_path.lstrip("/")
                    return FileInfo(
                        path=mount.mount_path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                try:
                    await self._check_share_access(sess, mount, rel_path, user_id, "read")
                except PermissionError:
                    return None
            info = await mount.backend.get_info(rel_path, session=sess)
        if info is not None:
            info = self._restore_file_info(info, mount, user_id)
        return info

    def get_permission_info(self, path: str) -> tuple[str, bool]:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)

        permission = self._registry.get_permission(path)

        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths

        return permission.value, is_override

    # ------------------------------------------------------------------
    # Search / Query Operations
    # ------------------------------------------------------------------

    async def glob(
        self, pattern: str, path: str = "/", *, user_id: str | None = None
    ) -> GlobResult:
        path = normalize_path(path)

        if path == "/":
            # Aggregate across all visible mounts (exclude hidden)
            all_entries: list[FileInfo] = []
            for mount in self._registry.list_visible_mounts():
                glob_path = "/"
                if mount.authenticated and user_id:
                    glob_path = f"/{user_id}"
                async with self._session_for(mount) as sess:
                    result = await mount.backend.glob(pattern, glob_path, session=sess)
                if result.success:
                    all_entries.extend(
                        self._restore_file_info(e, mount, user_id)
                        for e in result.entries
                    )
            return GlobResult(
                success=True,
                message=f"Found {len(all_entries)} match(es)",
                entries=all_entries,
                pattern=pattern,
                path=path,
            )

        mount, rel_path = self._registry.resolve(path)
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            result = await mount.backend.glob(pattern, rel_path, session=sess)
        result.path = self._restore_user_path(result.path, mount, user_id) or path
        result.entries = [
            self._restore_file_info(e, mount, user_id) for e in result.entries
        ]
        return result

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        user_id: str | None = None,
    ) -> GrepResult:
        path = normalize_path(path)

        if path == "/":
            all_matches: list[GrepMatch] = []
            total_searched = 0
            total_matched = 0
            truncated = False

            # Don't pass count_only to backends — we need actual matches
            # to aggregate correctly. We apply count_only at VFS level.

            for mount in self._registry.list_visible_mounts():
                remaining = max_results - len(all_matches) if max_results > 0 else max_results
                if max_results > 0 and remaining <= 0:
                    truncated = True
                    break
                grep_path = "/"
                if mount.authenticated and user_id:
                    grep_path = f"/{user_id}"
                async with self._session_for(mount) as sess:
                    result = await mount.backend.grep(
                        pattern,
                        grep_path,
                        session=sess,
                        glob_filter=glob_filter,
                        case_sensitive=case_sensitive,
                        fixed_string=fixed_string,
                        invert=invert,
                        word_match=word_match,
                        context_lines=context_lines,
                        max_results=remaining,
                        max_results_per_file=max_results_per_file,
                        count_only=False,
                        files_only=files_only,
                    )
                if result.success:
                    for m in result.matches:
                        restored = self._restore_user_path(m.file_path, mount, user_id)
                        m.file_path = restored or m.file_path
                    all_matches.extend(result.matches)
                    total_searched += result.files_searched
                    total_matched += result.files_matched
                    if result.truncated:
                        truncated = True

            if count_only:
                total = total_matched if files_only else len(all_matches)
                return GrepResult(
                    success=True,
                    message=f"Count: {total}",
                    matches=[],
                    pattern=pattern,
                    path=path,
                    files_searched=total_searched,
                    files_matched=total_matched,
                    truncated=truncated,
                )

            return GrepResult(
                success=True,
                message=f"Found {len(all_matches)} match(es) in {total_matched} file(s)",
                matches=all_matches,
                pattern=pattern,
                path=path,
                files_searched=total_searched,
                files_matched=total_matched,
                truncated=truncated,
            )

        mount, rel_path = self._registry.resolve(path)
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            result = await mount.backend.grep(
                pattern,
                rel_path,
                session=sess,
                glob_filter=glob_filter,
                case_sensitive=case_sensitive,
                fixed_string=fixed_string,
                invert=invert,
                word_match=word_match,
                context_lines=context_lines,
                max_results=max_results,
                max_results_per_file=max_results_per_file,
                count_only=count_only,
                files_only=files_only,
            )
        result.path = self._restore_user_path(result.path, mount, user_id) or path
        for m in result.matches:
            restored = self._restore_user_path(m.file_path, mount, user_id)
            m.file_path = restored or m.file_path
        return result

    async def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        path = normalize_path(path)

        if path == "/":
            all_entries: list[FileInfo] = []
            total_files = 0
            total_dirs = 0

            # Include mount roots themselves
            for mount in self._registry.list_visible_mounts():
                name = mount.mount_path.lstrip("/")
                all_entries.append(
                    FileInfo(
                        path=mount.mount_path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )
                )
                total_dirs += 1

            # Backends count depth from their own root, so pass max_depth
            # unchanged — the mount roots are added above at VFS level.
            if max_depth is None or max_depth > 0:
                for mount in self._registry.list_visible_mounts():
                    tree_path = "/"
                    if mount.authenticated and user_id:
                        tree_path = f"/{user_id}"
                    async with self._session_for(mount) as sess:
                        result = await mount.backend.tree(
                            tree_path,
                            max_depth=max_depth,
                            session=sess,
                        )
                    if result.success:
                        all_entries.extend(
                            self._restore_file_info(e, mount, user_id)
                            for e in result.entries
                        )
                        total_files += result.total_files
                        total_dirs += result.total_dirs

            all_entries.sort(key=lambda e: e.path)
            return TreeResult(
                success=True,
                message=f"{total_dirs} directories, {total_files} files",
                entries=all_entries,
                path="/",
                total_files=total_files,
                total_dirs=total_dirs,
            )

        mount, rel_path = self._registry.resolve(path)
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            result = await mount.backend.tree(rel_path, max_depth=max_depth, session=sess)
        result.path = self._restore_user_path(result.path, mount, user_id) or path
        result.entries = [
            self._restore_file_info(e, mount, user_id) for e in result.entries
        ]
        return result

    # ------------------------------------------------------------------
    # Write Operations (permission-checked)
    # ------------------------------------------------------------------

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "write")
            write_kwargs: dict[str, Any] = {"overwrite": overwrite, "session": sess}
            if mount.authenticated and user_id is not None:
                write_kwargs["owner_id"] = user_id
            result = await mount.backend.write(
                rel_path, content, created_by, **write_kwargs
            )
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=path, content=content)
            )
        return result

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        user_id: str | None = None,
    ) -> EditResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return EditResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "write")
            result = await mount.backend.edit(
                rel_path, old_string, new_string, replace_all, created_by, session=sess
            )
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        if result.success:
            await self._emit(FileEvent(event_type=EventType.FILE_WRITTEN, path=path))
        return result

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        user_id: str | None = None,
    ) -> DeleteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path, user_id)

        # If backend doesn't support trash and permanent=False, explicit failure
        if not permanent and not self._get_capability(mount.backend, SupportsTrash):
            return DeleteResult(
                success=False,
                message="Trash not supported on this mount. "
                "Use permanent=True to delete permanently.",
            )

        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "write")
            result = await mount.backend.delete(rel_path, permanent, session=sess)
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        if result.success:
            await self._emit(FileEvent(event_type=EventType.FILE_DELETED, path=path))
        return result

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        user_id: str | None = None,
    ) -> MkdirResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        async with self._session_for(mount) as sess:
            result = await mount.backend.mkdir(rel_path, parents, session=sess)
        result.path = self._restore_user_path(result.path, mount, user_id)
        result.created_dirs = [
            self._restore_user_path(d, mount, user_id) or d
            for d in result.created_dirs
        ]
        return result

    async def move(
        self,
        src: str,
        dest: str,
        *,
        user_id: str | None = None,
        follow: bool = False,
    ) -> MoveResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(src)
            self._check_writable(dest)
        except PermissionError as e:
            return MoveResult(success=False, message=str(e))

        src_mount, src_rel_orig = self._registry.resolve(src)
        dest_mount, dest_rel_orig = self._registry.resolve(dest)

        src_shared = self._is_shared_access(src_rel_orig)[0] if src_mount.authenticated else False
        dest_shared = (
            self._is_shared_access(dest_rel_orig)[0] if dest_mount.authenticated else False
        )

        src_rel = self._resolve_user_path(src_mount, src_rel_orig, user_id)
        dest_rel = self._resolve_user_path(dest_mount, dest_rel_orig, user_id)

        if src_mount is dest_mount:
            async with self._session_for(src_mount) as sess:
                if src_shared and user_id:
                    await self._check_share_access(sess, src_mount, src_rel, user_id, "write")
                if dest_shared and user_id:
                    await self._check_share_access(sess, dest_mount, dest_rel, user_id, "write")
                result = await src_mount.backend.move(
                    src_rel, dest_rel, session=sess,
                    follow=follow, sharing=src_mount.sharing,
                )
            result.old_path = self._restore_user_path(result.old_path, src_mount, user_id)
            result.new_path = self._restore_user_path(result.new_path, dest_mount, user_id)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src)
                )
            return result

        # Cross-mount move: read → write → delete (non-atomic).
        async with self._session_for(src_mount) as src_sess:
            read_result = await src_mount.backend.read(src_rel, session=src_sess)
        if not read_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot read source for cross-mount move: {read_result.message}",
            )

        if read_result.content is None:
            return MoveResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        async with self._session_for(dest_mount) as dest_sess:
            write_result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess
            )
        if not write_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot write to destination for cross-mount move: {write_result.message}",
            )

        async with self._session_for(src_mount) as src_sess:
            delete_result = await src_mount.backend.delete(
                src_rel, permanent=False, session=src_sess
            )
        if not delete_result.success:
            return MoveResult(
                success=False,
                message=f"Copied but failed to delete source: {delete_result.message}",
            )

        await self._emit(FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src))
        return MoveResult(
            success=True,
            message=f"Moved {src} -> {dest} (cross-mount)",
            old_path=src,
            new_path=dest,
        )

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        user_id: str | None = None,
    ) -> WriteResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        src_mount, src_rel_orig = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        src_shared = self._is_shared_access(src_rel_orig)[0] if src_mount.authenticated else False
        src_rel = self._resolve_user_path(src_mount, src_rel_orig, user_id)
        dest_rel = self._resolve_user_path(dest_mount, dest_rel, user_id)

        if src_mount is dest_mount:
            async with self._session_for(src_mount) as sess:
                if src_shared and user_id:
                    await self._check_share_access(sess, src_mount, src_rel, user_id, "read")
                result = await src_mount.backend.copy(src_rel, dest_rel, session=sess)
            result.file_path = self._restore_user_path(result.file_path, dest_mount, user_id)
            if result.success:
                await self._emit(FileEvent(event_type=EventType.FILE_WRITTEN, path=dest))
            return result

        # Cross-mount copy: read → write
        async with self._session_for(src_mount) as src_sess:
            if src_shared and user_id:
                await self._check_share_access(src_sess, src_mount, src_rel, user_id, "read")
            read_result = await src_mount.backend.read(src_rel, session=src_sess)
        if not read_result.success:
            return WriteResult(
                success=False,
                message=f"Cannot read source for cross-mount copy: {read_result.message}",
            )

        if read_result.content is None:
            return WriteResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        async with self._session_for(dest_mount) as dest_sess:
            result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess
            )
        result.file_path = self._restore_user_path(result.file_path, dest_mount, user_id)
        if result.success:
            await self._emit(FileEvent(event_type=EventType.FILE_WRITTEN, path=dest))
        return result

    # ------------------------------------------------------------------
    # Version Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(
        self, path: str, *, user_id: str | None = None
    ) -> ListVersionsResult:
        path = normalize_path(path)
        mount, rel_path_orig = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path_orig)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path_orig, user_id)
        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "read")
            return await cap.list_versions(rel_path, session=sess)

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        mount, rel_path_orig = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path_orig)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path_orig, user_id)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "write")
            result = await cap.restore_version(rel_path, version, session=sess)
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        if result.success:
            await self._emit(FileEvent(event_type=EventType.FILE_RESTORED, path=path))
        return result

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        mount, rel_path_orig = self._registry.resolve(path)
        is_shared = self._is_shared_access(rel_path_orig)[0] if mount.authenticated else False
        rel_path = self._resolve_user_path(mount, rel_path_orig, user_id)
        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            if is_shared and user_id:
                await self._check_share_access(sess, mount, rel_path, user_id, "read")
            return await cap.get_version_content(rel_path, version, session=sess)

    # ------------------------------------------------------------------
    # Trash Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> ListResult:
        """List all items in trash across all mounts (skips unsupported)."""
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self._session_for(mount) as sess:
                result = await cap.list_trash(session=sess)
            if result.success:
                prefixed_entries = [
                    self._restore_file_info(entry, mount, user_id)
                    for entry in result.entries
                ]
                all_entries.extend(prefixed_entries)

        return ListResult(
            success=True,
            message=f"Found {len(all_entries)} item(s) in trash",
            entries=all_entries,
            path="/__trash__",
        )

    async def restore_from_trash(
        self, path: str, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        rel_path = self._resolve_user_path(mount, rel_path, user_id)
        cap = self._get_capability(mount.backend, SupportsTrash)
        if cap is None:
            raise CapabilityNotSupportedError(f"Mount at {mount.mount_path} does not support trash")
        async with self._session_for(mount) as sess:
            result = await cap.restore_from_trash(rel_path, session=sess)
        result.file_path = self._restore_user_path(result.file_path, mount, user_id)
        if result.success:
            await self._emit(FileEvent(event_type=EventType.FILE_RESTORED, path=path))
        return result

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts (skips unsupported)."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self._session_for(mount) as sess:
                result = await cap.empty_trash(session=sess)
            if not result.success:
                return result
            total_deleted += result.total_deleted or 0
            mounts_processed += 1

        return DeleteResult(
            success=True,
            message=f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)",
            total_deleted=total_deleted,
            permanent=True,
        )

    # ------------------------------------------------------------------
    # Reconciliation (capability-gated)
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        """Reconcile disk ↔ DB for capable mounts."""
        total = {"created": 0, "updated": 0, "deleted": 0}
        mounts = self._registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.mount_path == mount_path]

        for mount in mounts:
            cap = self._get_capability(mount.backend, SupportsReconcile)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            for k in total:
                total[k] += stats.get(k, 0)

        return total

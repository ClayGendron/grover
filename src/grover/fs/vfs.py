"""VFS — mount router with routing, permissions, events, capabilities."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from grover.events import EventBus, EventType, FileEvent

from .exceptions import CapabilityNotSupportedError, MountNotFoundError
from .permissions import Permission
from .protocol import SupportsReconcile, SupportsSearch, SupportsTrash, SupportsVersions
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

    All user-scoping (per-user path namespacing, ``@shared`` resolution,
    share permission checks) is handled by the backend — typically
    ``UserScopedFileSystem``.  VFS just passes ``user_id`` through.

    Session lifecycle is per-operation only.  No transaction mode.
    """

    def __init__(self, registry: MountRegistry, event_bus: EventBus | None = None) -> None:
        self._registry = registry
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    def get_capability(self, backend: Any, protocol: type[T]) -> T | None:
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
    async def session_for(self, mount: MountConfig) -> AsyncGenerator[AsyncSession | None]:
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
        async with self.session_for(mount) as sess:
            result = await mount.backend.read(
                rel_path, offset, limit, session=sess, user_id=user_id
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def list_dir(self, path: str = "/", *, user_id: str | None = None) -> ListResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        async with self.session_for(mount) as sess:
            result = await mount.backend.list_dir(rel_path, session=sess, user_id=user_id)

        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [self._prefix_file_info(entry, mount) for entry in result.entries]
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

    async def exists(self, path: str, *, user_id: str | None = None) -> bool:
        path = normalize_path(path)

        if path == "/":
            return True

        if self._registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return False

        async with self.session_for(mount) as sess:
            return await mount.backend.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileInfo | None:
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

        async with self.session_for(mount) as sess:
            info = await mount.backend.get_info(rel_path, session=sess, user_id=user_id)
        if info is not None:
            info = self._prefix_file_info(info, mount)
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
                async with self.session_for(mount) as sess:
                    result = await mount.backend.glob(pattern, "/", session=sess, user_id=user_id)
                if result.success:
                    all_entries.extend(self._prefix_file_info(e, mount) for e in result.entries)
            return GlobResult(
                success=True,
                message=f"Found {len(all_entries)} match(es)",
                entries=all_entries,
                pattern=pattern,
                path=path,
            )

        mount, rel_path = self._registry.resolve(path)
        async with self.session_for(mount) as sess:
            result = await mount.backend.glob(pattern, rel_path, session=sess, user_id=user_id)
        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [self._prefix_file_info(e, mount) for e in result.entries]
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
                async with self.session_for(mount) as sess:
                    result = await mount.backend.grep(
                        pattern,
                        "/",
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
                        user_id=user_id,
                    )
                if result.success:
                    for m in result.matches:
                        m.file_path = (
                            self._prefix_path(m.file_path, mount.mount_path) or m.file_path
                        )
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
        async with self.session_for(mount) as sess:
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
                user_id=user_id,
            )
        result.path = self._prefix_path(result.path, mount.mount_path) or path
        for m in result.matches:
            m.file_path = self._prefix_path(m.file_path, mount.mount_path) or m.file_path
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
                    async with self.session_for(mount) as sess:
                        result = await mount.backend.tree(
                            "/",
                            max_depth=max_depth,
                            session=sess,
                            user_id=user_id,
                        )
                    if result.success:
                        all_entries.extend(self._prefix_file_info(e, mount) for e in result.entries)
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
        async with self.session_for(mount) as sess:
            result = await mount.backend.tree(
                rel_path, max_depth=max_depth, session=sess, user_id=user_id
            )
        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [self._prefix_file_info(e, mount) for e in result.entries]
        return result

    async def search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        user_id: str | None = None,
    ) -> list:
        """Route semantic search to backend(s), aggregate across mounts.

        Returns ``list[SearchResult]`` — the internal Grover search type.
        """
        path = normalize_path(path)

        if path == "/":
            all_results: list = []
            for mount in self._registry.list_visible_mounts():
                cap = self.get_capability(mount.backend, SupportsSearch)
                if cap is None:
                    continue
                async with self.session_for(mount) as sess:
                    results = await cap.search(query, k, path="/", session=sess, user_id=user_id)
                all_results.extend(results)
            # Sort by score descending, truncate to k
            all_results.sort(key=lambda r: r.score, reverse=True)
            return all_results[:k]

        mount, rel_path = self._registry.resolve(path)
        cap = self.get_capability(mount.backend, SupportsSearch)
        if cap is None:
            return []
        async with self.session_for(mount) as sess:
            return await cap.search(query, k, path=rel_path, session=sess, user_id=user_id)

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
        async with self.session_for(mount) as sess:
            result = await mount.backend.write(
                rel_path,
                content,
                created_by,
                overwrite=overwrite,
                session=sess,
                user_id=user_id,
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(
                    event_type=EventType.FILE_WRITTEN,
                    path=path,
                    content=content,
                    user_id=user_id,
                )
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
        async with self.session_for(mount) as sess:
            result = await mount.backend.edit(
                rel_path,
                old_string,
                new_string,
                replace_all,
                created_by,
                session=sess,
                user_id=user_id,
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=path, user_id=user_id)
            )
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

        # If backend doesn't support trash and permanent=False, explicit failure
        if not permanent and not self.get_capability(mount.backend, SupportsTrash):
            return DeleteResult(
                success=False,
                message="Trash not supported on this mount. "
                "Use permanent=True to delete permanently.",
            )

        async with self.session_for(mount) as sess:
            result = await mount.backend.delete(rel_path, permanent, session=sess, user_id=user_id)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_DELETED, path=path, user_id=user_id)
            )
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
        async with self.session_for(mount) as sess:
            result = await mount.backend.mkdir(rel_path, parents, session=sess, user_id=user_id)
        result.path = self._prefix_path(result.path, mount.mount_path)
        result.created_dirs = [
            self._prefix_path(d, mount.mount_path) or d for d in result.created_dirs
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

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            async with self.session_for(src_mount) as sess:
                result = await src_mount.backend.move(
                    src_rel,
                    dest_rel,
                    session=sess,
                    follow=follow,
                    user_id=user_id,
                )
            result.old_path = self._prefix_path(result.old_path, src_mount.mount_path)
            result.new_path = self._prefix_path(result.new_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(
                        event_type=EventType.FILE_MOVED,
                        path=dest,
                        old_path=src,
                        user_id=user_id,
                    )
                )
            return result

        # Cross-mount move: read → write → delete (non-atomic).
        async with self.session_for(src_mount) as src_sess:
            read_result = await src_mount.backend.read(src_rel, session=src_sess, user_id=user_id)
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

        async with self.session_for(dest_mount) as dest_sess:
            write_result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess, user_id=user_id
            )
        if not write_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot write to destination for cross-mount move: {write_result.message}",
            )

        async with self.session_for(src_mount) as src_sess:
            delete_result = await src_mount.backend.delete(
                src_rel, permanent=False, session=src_sess, user_id=user_id
            )
        if not delete_result.success:
            return MoveResult(
                success=False,
                message=f"Copied but failed to delete source: {delete_result.message}",
            )

        await self._emit(
            FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src, user_id=user_id)
        )
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

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            async with self.session_for(src_mount) as sess:
                result = await src_mount.backend.copy(
                    src_rel, dest_rel, session=sess, user_id=user_id
                )
            result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                )
            return result

        # Cross-mount copy: read → write
        async with self.session_for(src_mount) as src_sess:
            read_result = await src_mount.backend.read(src_rel, session=src_sess, user_id=user_id)
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

        async with self.session_for(dest_mount) as dest_sess:
            result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess, user_id=user_id
            )
        result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
            )
        return result

    # ------------------------------------------------------------------
    # Version Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> ListVersionsResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        cap = self.get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self.session_for(mount) as sess:
            return await cap.list_versions(rel_path, session=sess, user_id=user_id)

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        cap = self.get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self.session_for(mount) as sess:
            result = await cap.restore_version(rel_path, version, session=sess, user_id=user_id)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
            )
        return result

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        cap = self.get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self.session_for(mount) as sess:
            return await cap.get_version_content(rel_path, version, session=sess, user_id=user_id)

    # ------------------------------------------------------------------
    # Trash Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> ListResult:
        """List all items in trash across all mounts (skips unsupported)."""
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            cap = self.get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self.session_for(mount) as sess:
                result = await cap.list_trash(session=sess, user_id=user_id)
            if result.success:
                all_entries.extend(self._prefix_file_info(entry, mount) for entry in result.entries)

        return ListResult(
            success=True,
            message=f"Found {len(all_entries)} item(s) in trash",
            entries=all_entries,
            path="/__trash__",
        )

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        cap = self.get_capability(mount.backend, SupportsTrash)
        if cap is None:
            raise CapabilityNotSupportedError(f"Mount at {mount.mount_path} does not support trash")
        async with self.session_for(mount) as sess:
            result = await cap.restore_from_trash(rel_path, session=sess, user_id=user_id)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
            )
        return result

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts (skips unsupported)."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            cap = self.get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self.session_for(mount) as sess:
                result = await cap.empty_trash(session=sess, user_id=user_id)
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
            cap = self.get_capability(mount.backend, SupportsReconcile)
            if cap is None:
                continue
            async with self.session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            for k in total:
                total[k] += stats.get(k, 0)

        return total

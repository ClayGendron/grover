"""UserScopedFileSystem — user-scoped DBFS with sharing support.

Subclasses ``DatabaseFileSystem`` and adds per-user path namespacing,
``@shared`` virtual directory resolution, share permission checks,
and owner-scoped trash.  Implements the ``SupportsReBAC`` protocol.

VFS treats this as an opaque ``StorageBackend`` — all user-scoping
logic lives here, not in VFS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .database_fs import DatabaseFileSystem
from .exceptions import AuthenticationRequiredError
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    GlobResult,
    GrepResult,
    ListResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareInfo,
    TreeResult,
    WriteResult,
)
from .utils import normalize_path

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase, FileVersionBase

    from .sharing import SharingService

logger = logging.getLogger(__name__)


class UserScopedFileSystem(DatabaseFileSystem):
    """Database-backed FS with per-user path namespacing and sharing.

    Every method requires ``user_id``.  Paths are automatically prefixed
    with ``/{user_id}/`` for storage isolation.  The ``@shared/{owner}/``
    virtual namespace resolves to another user's files with permission
    checks via a ``SharingService``.

    Implements ``StorageBackend``, ``SupportsVersions``, ``SupportsTrash``,
    and ``SupportsReBAC`` protocols.
    """

    def __init__(
        self,
        sharing: SharingService | None = None,
        *,
        dialect: str = "sqlite",
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        schema: str | None = None,
    ) -> None:
        super().__init__(
            dialect=dialect,
            file_model=file_model,
            file_version_model=file_version_model,
            schema=schema,
        )
        self._sharing = sharing

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_user_id(user_id: str | None) -> str:
        """Raise if *user_id* is missing, empty, or contains unsafe characters."""
        if not user_id:
            raise AuthenticationRequiredError("user_id is required for authenticated mount")
        if "/" in user_id or "\\" in user_id or "\0" in user_id or "@" in user_id:
            raise AuthenticationRequiredError("user_id contains invalid characters")
        if ".." in user_id:
            raise AuthenticationRequiredError("user_id contains invalid characters")
        return user_id

    @staticmethod
    def _is_shared_access(path: str) -> tuple[bool, str | None, str | None]:
        """Parse ``/@shared/{owner}/{rest}`` from a path.

        Returns ``(is_shared, owner, rest_path)``.
        """
        segments = path.strip("/").split("/")
        if len(segments) >= 2 and segments[0] == "@shared":
            owner = segments[1]
            rest = "/" + "/".join(segments[2:]) if len(segments) > 2 else "/"
            return True, owner, rest
        return False, None, None

    @staticmethod
    def _resolve_path(path: str, user_id: str) -> str:
        """Rewrite a user-facing path to a stored path.

        - Regular paths: prepend ``/{user_id}/``.
        - ``@shared/{owner}/rest``: resolve to ``/{owner}/rest``.
        """
        is_shared, owner, rest = UserScopedFileSystem._is_shared_access(path)
        if is_shared and owner is not None and rest is not None:
            return f"/{owner}{rest}" if rest != "/" else f"/{owner}"

        if path == "/":
            return f"/{user_id}"
        return f"/{user_id}{path}"

    @staticmethod
    def _strip_user_prefix(path: str, user_id: str) -> str:
        """Remove ``/{user_id}`` prefix from a stored path."""
        prefix = f"/{user_id}/"
        if path.startswith(prefix):
            return "/" + path[len(prefix) :]
        if path == f"/{user_id}":
            return "/"
        return path

    def _restore_path(
        self,
        stored_path: str | None,
        user_id: str,
        original_path: str | None = None,
    ) -> str | None:
        """Convert a stored path back to a user-facing path.

        If *original_path* is provided (e.g. the ``@shared/...`` path the
        user passed in), it is returned directly — no guessing needed.
        """
        if original_path is not None:
            return original_path
        if stored_path is None:
            return None
        return self._strip_user_prefix(stored_path, user_id)

    def _restore_file_info(self, info: FileInfo, user_id: str) -> FileInfo:
        """Strip user prefix from a FileInfo's path."""
        info.path = self._strip_user_prefix(info.path, user_id)
        return info

    # ------------------------------------------------------------------
    # Share permission checks
    # ------------------------------------------------------------------

    async def _check_share_access(
        self,
        session: AsyncSession,
        stored_path: str,
        user_id: str,
        required: str = "read",
    ) -> None:
        """Verify *user_id* has shared access to *stored_path*.

        Raises ``PermissionError`` if no sharing service is configured or
        the user lacks the required permission.
        """
        if self._sharing is None:
            raise PermissionError("Access denied: sharing is not configured")
        has_access = await self._sharing.check_permission(
            session,
            stored_path,
            user_id,
            required=required,
        )
        if not has_access:
            raise PermissionError(
                f"Access denied: {user_id!r} does not have {required!r} permission on shared path"
            )

    # ------------------------------------------------------------------
    # Shared directory listing
    # ------------------------------------------------------------------

    async def _list_shared_dir(
        self,
        segments: list[str],
        user_id: str,
        session: AsyncSession,
    ) -> ListResult:
        """List virtual ``@shared/`` directories.

        - ``/@shared`` → list distinct owners who shared with *user_id*
        - ``/@shared/{owner}`` → list that owner's shared files
        - ``/@shared/{owner}/sub/...`` → list sub-path (permission-checked)
        """
        if self._sharing is None:
            return ListResult(
                success=True,
                message="No sharing configured",
                entries=[],
                path="/@shared",
            )

        if len(segments) == 1:
            # /@shared — list distinct owners
            shares = await self._sharing.list_shared_with(session, user_id)
            owners: set[str] = set()
            for share in shares:
                parts = share.path.strip("/").split("/")
                if parts:
                    owners.add(parts[0])
            entries = [
                FileInfo(
                    path=f"/@shared/{owner}",
                    name=owner,
                    is_directory=True,
                )
                for owner in sorted(owners)
            ]
            return ListResult(
                success=True,
                message=f"Found {len(entries)} shared owner(s)",
                entries=entries,
                path="/@shared",
            )

        # /@shared/{owner}/... — resolve to /{owner}/... and list
        owner = segments[1]
        sub_path = "/" + "/".join(segments[2:]) if len(segments) > 2 else "/"
        stored_path = f"/{owner}{sub_path}" if sub_path != "/" else f"/{owner}"
        shared_prefix = f"/@shared/{owner}"

        # Fast path: directory-level share covers this path
        try:
            await self._check_share_access(session, stored_path, user_id, "read")
            result = await super().list_dir(stored_path, session=session)

            result.path = shared_prefix + sub_path if sub_path != "/" else shared_prefix
            for entry in result.entries:
                stripped = self._strip_user_prefix(entry.path, owner)
                entry.path = shared_prefix + stripped if stripped != "/" else shared_prefix
            return result

        except PermissionError:
            pass  # Fall through to filtered listing

        # Filtered fallback: show only files/dirs with specific shares
        shares = await self._sharing.list_shares_under_prefix(session, user_id, stored_path)
        if not shares:
            raise PermissionError(
                f"Access denied: {user_id!r} does not have 'read' permission on shared path"
            )

        direct_files: set[str] = set()
        child_dirs: set[str] = set()
        prefix_with_slash = stored_path + "/"
        for share in shares:
            remainder = share.path[len(prefix_with_slash) :]
            if "/" in remainder:
                child_dirs.add(remainder.split("/", 1)[0])
            else:
                direct_files.add(remainder)

        entries: list[FileInfo] = []
        base = shared_prefix if sub_path == "/" else f"{shared_prefix}{sub_path}"

        for name in sorted(direct_files):
            file_stored = f"{stored_path}/{name}"
            try:
                info = await super().get_info(file_stored, session=session)
            except Exception:
                info = None
            if info is not None:
                stripped = self._strip_user_prefix(info.path, owner)
                info.path = shared_prefix + stripped if stripped != "/" else shared_prefix
                entries.append(info)
            else:
                entries.append(
                    FileInfo(
                        path=f"{base}/{name}",
                        name=name,
                        is_directory=False,
                    )
                )

        for name in sorted(child_dirs):
            entries.append(
                FileInfo(
                    path=f"{base}/{name}",
                    name=name,
                    is_directory=True,
                )
            )

        return ListResult(
            success=True,
            message=f"Found {len(entries)} shared item(s)",
            entries=entries,
            path=shared_prefix + sub_path if sub_path != "/" else shared_prefix,
        )

    # ------------------------------------------------------------------
    # Core protocol: StorageBackend (overrides)
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ReadResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        result = await super().read(stored, offset, limit, session=sess)
        orig = path if is_shared else None
        result.file_path = self._restore_path(result.file_path, uid, orig)
        return result

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "write")
        result = await super().write(
            stored,
            content,
            created_by,
            overwrite=overwrite,
            session=sess,
            owner_id=uid,
        )
        orig = path if is_shared else None
        result.file_path = self._restore_path(result.file_path, uid, orig)
        return result

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> EditResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "write")
        result = await super().edit(
            stored,
            old_string,
            new_string,
            replace_all,
            created_by,
            session=sess,
        )
        orig = path if is_shared else None
        result.file_path = self._restore_path(result.file_path, uid, orig)
        return result

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "write")
        result = await super().delete(stored, permanent, session=sess)
        orig = path if is_shared else None
        result.file_path = self._restore_path(result.file_path, uid, orig)
        return result

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        uid = self._require_user_id(user_id)
        is_shared, owner, _rest = self._is_shared_access(path)
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "write")
        created_dirs, error = await self.directories.mkdir(
            sess,
            stored,
            parents,
            self.metadata.get_file,
            owner_id=uid,
        )
        if error is not None:
            return MkdirResult(success=False, message=error)
        stored = normalize_path(stored)
        # For shared paths, display the @shared path; for own paths, strip prefix
        if is_shared:
            display_path = path

            def _display(d: str) -> str:
                assert owner is not None
                stripped = self._strip_user_prefix(d, owner)
                return f"/@shared/{owner}{stripped}" if stripped != "/" else f"/@shared/{owner}"
        else:
            display_path = self._strip_user_prefix(stored, uid)

            def _display(d: str) -> str:
                return self._strip_user_prefix(d, uid)

        if created_dirs:
            return MkdirResult(
                success=True,
                message=f"Created directory: {display_path}",
                path=display_path,
                created_dirs=[_display(d) for d in created_dirs],
            )
        return MkdirResult(
            success=True,
            message=f"Directory already exists: {display_path}",
            path=display_path,
            created_dirs=[],
        )

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListResult:
        uid = self._require_user_id(user_id)
        sess = self._require_session(session)

        # Handle @shared virtual directories
        segments = path.strip("/").split("/")
        if segments[0] == "@shared":
            return await self._list_shared_dir(segments, uid, sess)

        stored = self._resolve_path(path, uid)
        result = await super().list_dir(stored, session=sess)

        result.path = self._restore_path(result.path, uid) or path
        result.entries = [self._restore_file_info(entry, uid) for entry in result.entries]

        # At root, add virtual @shared/ entry
        if path == "/":
            result.entries.append(
                FileInfo(
                    path="/@shared",
                    name="@shared",
                    is_directory=True,
                )
            )

        return result

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> bool:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            try:
                await self._check_share_access(sess, stored, uid, "read")
            except PermissionError:
                return False
        return await super().exists(stored, session=sess)

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfo | None:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            try:
                await self._check_share_access(sess, stored, uid, "read")
            except PermissionError:
                return None
        info = await super().get_info(stored, session=sess)
        if info is not None:
            if is_shared:
                info.path = path
            else:
                self._restore_file_info(info, uid)
        return info

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        follow: bool = False,
        sharing: Any = None,
        user_id: str | None = None,
    ) -> MoveResult:
        uid = self._require_user_id(user_id)
        src_shared = self._is_shared_access(src)[0]
        dest_shared = self._is_shared_access(dest)[0]
        src_stored = self._resolve_path(src, uid)
        dest_stored = self._resolve_path(dest, uid)
        sess = self._require_session(session)
        if src_shared:
            await self._check_share_access(sess, src_stored, uid, "write")
        if dest_shared:
            await self._check_share_access(sess, dest_stored, uid, "write")
        if src_shared and not dest_shared:
            raise PermissionError("Cannot move shared files out of the owner's namespace")
        result = await super().move(
            src_stored,
            dest_stored,
            session=sess,
            follow=follow,
            sharing=self._sharing,
        )
        src_orig = src if src_shared else None
        dest_orig = dest if dest_shared else None
        result.old_path = self._restore_path(result.old_path, uid, src_orig)
        result.new_path = self._restore_path(result.new_path, uid, dest_orig)
        return result

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        uid = self._require_user_id(user_id)
        src_shared = self._is_shared_access(src)[0]
        dest_shared = self._is_shared_access(dest)[0]
        src_stored = self._resolve_path(src, uid)
        dest_stored = self._resolve_path(dest, uid)
        sess = self._require_session(session)
        if src_shared:
            await self._check_share_access(sess, src_stored, uid, "read")
        if dest_shared:
            await self._check_share_access(sess, dest_stored, uid, "write")
        # Call super().write directly to avoid polymorphic dispatch
        # (copy_file calls self.write which would re-enter our override)
        src_file = await self.metadata.get_file(sess, src_stored)
        if not src_file:
            return WriteResult(
                success=False,
                message=f"Source not found: {src}",
            )
        if src_file.is_directory:
            return WriteResult(
                success=False,
                message="Directory copy not yet implemented",
            )
        content = await self._read_content(src_stored, sess)
        if content is None:
            return WriteResult(
                success=False,
                message=f"Source content not found: {src}",
            )
        result = await super().write(
            dest_stored,
            content,
            "copy",
            overwrite=True,
            session=sess,
            owner_id=uid,
        )
        dest_orig = dest if dest_shared else None
        result.file_path = self._restore_path(result.file_path, uid, dest_orig)
        return result

    # ------------------------------------------------------------------
    # Search / Query (overrides)
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GlobResult:
        uid = self._require_user_id(user_id)
        is_shared, owner, _rest = self._is_shared_access(path)
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        result = await super().glob(pattern, stored, session=sess)
        if is_shared and owner is not None:
            shared_prefix = f"/@shared/{owner}"
            result.path = path
            for e in result.entries:
                stripped = self._strip_user_prefix(e.path, owner)
                e.path = shared_prefix + stripped if stripped != "/" else shared_prefix
        else:
            result.path = self._restore_path(result.path, uid) or path
            result.entries = [self._restore_file_info(e, uid) for e in result.entries]
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
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GrepResult:
        uid = self._require_user_id(user_id)
        is_shared, owner, _rest = self._is_shared_access(path)
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        # Resolve glob_filter via super().glob to avoid polymorphic dispatch
        # (super().grep calls self.glob() which would re-enter our override)
        resolved_glob_filter = glob_filter
        if glob_filter is not None:
            glob_result = await super().glob(glob_filter, stored, session=sess)
            if not glob_result.success:
                return GrepResult(
                    success=False,
                    message=glob_result.message,
                    pattern=pattern,
                    path=path,
                )
            # Pass candidate paths as a synthetic glob filter that matches
            # the resolved stored paths. Since super().grep with glob_filter
            # calls self.glob(), we avoid the issue by passing None and
            # pre-filtering.
            resolved_glob_filter = None

            # Get the list of candidate file paths from glob
            candidate_paths = [e.path for e in glob_result.entries if not e.is_directory]
            if not candidate_paths:
                return GrepResult(
                    success=True,
                    message="No files match glob filter",
                    pattern=pattern,
                    path=path,
                )

        result = await super().grep(
            pattern,
            stored,
            glob_filter=resolved_glob_filter,
            case_sensitive=case_sensitive,
            fixed_string=fixed_string,
            invert=invert,
            word_match=word_match,
            context_lines=context_lines,
            max_results=max_results,
            max_results_per_file=max_results_per_file,
            count_only=count_only,
            files_only=files_only,
            session=sess,
        )

        # If we pre-resolved a glob_filter, filter matches to only those
        # files that matched the glob
        if glob_filter is not None and not count_only:
            result.matches = [m for m in result.matches if m.file_path in candidate_paths]

        if is_shared and owner is not None:
            shared_prefix = f"/@shared/{owner}"
            result.path = path
            for m in result.matches:
                stripped = self._strip_user_prefix(m.file_path, owner)
                m.file_path = shared_prefix + stripped if stripped != "/" else shared_prefix
        else:
            result.path = self._restore_path(result.path, uid) or path
            for m in result.matches:
                restored = self._restore_path(m.file_path, uid)
                m.file_path = restored or m.file_path
        return result

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> TreeResult:
        uid = self._require_user_id(user_id)
        is_shared, owner, _rest = self._is_shared_access(path)
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        result = await super().tree(
            stored,
            max_depth=max_depth,
            session=sess,
        )
        if is_shared and owner is not None:
            shared_prefix = f"/@shared/{owner}"
            result.path = path
            for e in result.entries:
                stripped = self._strip_user_prefix(e.path, owner)
                e.path = shared_prefix + stripped if stripped != "/" else shared_prefix
        else:
            result.path = self._restore_path(result.path, uid) or path
            result.entries = [self._restore_file_info(e, uid) for e in result.entries]
        return result

    # ------------------------------------------------------------------
    # Capability: SupportsVersions (overrides)
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListVersionsResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        return await super().list_versions(stored, session=sess)

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GetVersionContentResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "read")
        return await super().get_version_content(stored, version, session=sess)

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> RestoreResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            await self._check_share_access(sess, stored, uid, "write")
        # Call super() methods directly to avoid polymorphic dispatch
        # (restore_version calls self.get_version_content and self.write)
        vc_result = await super().get_version_content(stored, version, session=sess)
        if not vc_result.success or vc_result.content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )
        write_result = await super().write(
            stored,
            vc_result.content,
            created_by="restore",
            session=sess,
        )
        orig = path if is_shared else None
        result = RestoreResult(
            success=True,
            message=f"Restored {path} to version {version}",
            file_path=self._restore_path(stored, uid, orig),
            restored_version=version,
            current_version=write_result.version,
        )
        return result

    # ------------------------------------------------------------------
    # Capability: SupportsTrash (overrides)
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> ListResult:
        uid = self._require_user_id(user_id)
        result = await super().list_trash(
            session=session,
            owner_id=uid,
        )
        result.entries = [self._restore_file_info(entry, uid) for entry in result.entries]
        return result

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> RestoreResult:
        uid = self._require_user_id(user_id)
        stored = self._resolve_path(path, uid)
        result = await super().restore_from_trash(
            stored,
            session=session,
            owner_id=uid,
        )
        result.file_path = self._restore_path(result.file_path, uid)
        return result

    async def empty_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        uid = self._require_user_id(user_id)
        return await super().empty_trash(session=session, owner_id=uid)

    # ------------------------------------------------------------------
    # Capability: SupportsReBAC (share CRUD)
    # ------------------------------------------------------------------

    async def share(
        self,
        path: str,
        grantee_id: str,
        permission: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
        expires_at: datetime | None = None,
    ) -> ShareInfo:
        """Create a share on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._sharing is None:
            raise ValueError("No sharing service configured")
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        share_record = await self._sharing.create_share(
            sess,
            stored,
            grantee_id,
            permission,
            uid,
            expires_at=expires_at,
        )
        return ShareInfo(
            path=path,
            grantee_id=share_record.grantee_id,
            permission=share_record.permission,
            granted_by=share_record.granted_by,
            created_at=share_record.created_at,
            expires_at=share_record.expires_at,
        )

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Remove a share on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._sharing is None:
            raise ValueError("No sharing service configured")
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        return await self._sharing.remove_share(sess, stored, grantee_id)

    async def list_shares_on_path(
        self,
        path: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> list[ShareInfo]:
        """List shares on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._sharing is None:
            return []
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        shares = await self._sharing.list_shares_on_path(sess, stored)
        return [
            ShareInfo(
                path=path,
                grantee_id=s.grantee_id,
                permission=s.permission,
                granted_by=s.granted_by,
                created_at=s.created_at,
                expires_at=s.expires_at,
            )
            for s in shares
        ]

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> list[ShareInfo]:
        """List all shares granted to *user_id*.

        Converts stored ``/{owner}/rest`` paths to ``@shared/{owner}/rest``
        user-facing paths.
        """
        uid = self._require_user_id(user_id)
        if self._sharing is None:
            return []
        sess = self._require_session(session)
        shares = await self._sharing.list_shared_with(sess, uid)
        result: list[ShareInfo] = []
        for s in shares:
            # Convert stored /{owner}/rest to @shared/{owner}/rest
            parts = s.path.strip("/").split("/", 1)
            if len(parts) >= 2:
                owner, rest = parts[0], parts[1]
                user_path = f"/@shared/{owner}/{rest}"
            elif parts:
                user_path = f"/@shared/{parts[0]}"
            else:
                user_path = s.path
            result.append(
                ShareInfo(
                    path=user_path,
                    grantee_id=s.grantee_id,
                    permission=s.permission,
                    granted_by=s.granted_by,
                    created_at=s.created_at,
                    expires_at=s.expires_at,
                )
            )
        return result

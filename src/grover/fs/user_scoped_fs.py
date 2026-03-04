"""UserScopedFileSystem — user-scoped DBFS with sharing support.

Subclasses ``DatabaseFileSystem`` and adds per-user path namespacing,
``@shared`` virtual directory resolution, share permission checks,
and owner-scoped trash.  Implements the ``SupportsReBAC`` protocol.

VFS treats this as an opaque ``GroverFileSystem`` — all user-scoping
logic lives here, not in VFS.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from grover.types.operations import (
    ChunkListResult,
    ChunkResult,
    DeleteResult,
    EditResult,
    ExistsResult,
    FileInfoResult,
    GetVersionContentResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareResult,
    WriteResult,
)
from grover.types.search import (
    FileSearchCandidate,
    GlobResult,
    GrepResult,
    ListDirEvidence,
    ListDirResult,
    ShareEvidence,
    ShareSearchResult,
    TrashResult,
    TreeResult,
    VersionResult,
)

from .database_fs import DatabaseFileSystem
from .exceptions import AuthenticationRequiredError
from .paths import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunk import FileChunkBase
    from grover.models.file import FileBase
    from grover.models.share import FileShareBase
    from grover.models.version import FileVersionBase

    from .providers.chunks.protocol import ChunkProvider
    from .providers.embedding.protocol import EmbeddingProvider
    from .providers.graph.protocol import GraphProvider
    from .providers.search.protocol import SearchProvider
    from .providers.storage.protocol import StorageProvider
    from .providers.versioning.protocol import VersionProvider

logger = logging.getLogger(__name__)


class UserScopedFileSystem(DatabaseFileSystem):
    """Database-backed FS with per-user path namespacing and sharing.

    Every method requires ``user_id``.  Paths are automatically prefixed
    with ``/{user_id}/`` for storage isolation.  The ``@shared/{owner}/``
    virtual namespace resolves to another user's files with permission
    checks via inlined share logic.

    Implements ``GroverFileSystem`` and ``SupportsReBAC`` protocols.
    """

    def __init__(
        self,
        share_model: type[FileShareBase] | None = None,
        *,
        dialect: str = "sqlite",
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
        schema: str | None = None,
        storage_provider: StorageProvider | None = None,
        graph_provider: GraphProvider | None = None,
        search_provider: SearchProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        version_provider: VersionProvider | None = None,
        chunk_provider: ChunkProvider | None = None,
    ) -> None:
        super().__init__(
            dialect=dialect,
            file_model=file_model,
            file_version_model=file_version_model,
            file_chunk_model=file_chunk_model,
            schema=schema,
            storage_provider=storage_provider,
            graph_provider=graph_provider,
            search_provider=search_provider,
            embedding_provider=embedding_provider,
            version_provider=version_provider,
            chunk_provider=chunk_provider,
        )
        self._share_model = share_model

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

        The path is normalized **before** prepending the user namespace to
        prevent ``..`` traversal out of the user's directory.
        """
        path = normalize_path(path)
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

    def _restore_file_info(self, info: FileInfoResult, user_id: str) -> FileInfoResult:
        """Strip user prefix from a FileInfoResult's path."""
        info.path = self._strip_user_prefix(info.path, user_id)
        return info

    # ------------------------------------------------------------------
    # Share logic (inlined from SharingService)
    # ------------------------------------------------------------------

    async def _create_share(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str,
        granted_by: str,
        *,
        expires_at: datetime | None = None,
    ) -> FileShareBase:
        """Create a share record. Flushes but does not commit."""
        if permission not in ("read", "write"):
            raise ValueError(f"Invalid permission: {permission!r}. Must be 'read' or 'write'.")
        assert self._share_model is not None
        path = normalize_path(path)
        share = self._share_model(
            id=str(uuid.uuid4()),
            path=path,
            grantee_id=grantee_id,
            permission=permission,
            granted_by=granted_by,
            expires_at=expires_at,
        )
        session.add(share)
        await session.flush()
        return share

    async def _remove_share(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
    ) -> bool:
        """Remove an exact share match. Returns True if found."""
        assert self._share_model is not None
        path = normalize_path(path)
        model = self._share_model
        result = await session.execute(
            select(model).where(model.path == path, model.grantee_id == grantee_id)
        )
        share = result.scalar_one_or_none()
        if share is None:
            return False
        await session.delete(share)
        await session.flush()
        return True

    async def _list_shares_on_path(
        self,
        session: AsyncSession,
        path: str,
    ) -> list[FileShareBase]:
        """List all shares for a given path."""
        assert self._share_model is not None
        path = normalize_path(path)
        model = self._share_model
        result = await session.execute(select(model).where(model.path == path))
        return list(result.scalars().all())

    async def _list_shared_with(
        self,
        session: AsyncSession,
        grantee_id: str,
    ) -> list[FileShareBase]:
        """List all shares granted to a grantee."""
        assert self._share_model is not None
        model = self._share_model
        result = await session.execute(select(model).where(model.grantee_id == grantee_id))
        return list(result.scalars().all())

    async def _list_shares_under_prefix(
        self,
        session: AsyncSession,
        grantee_id: str,
        prefix: str,
    ) -> list[FileShareBase]:
        """List non-expired shares for *grantee_id* strictly under *prefix*."""
        assert self._share_model is not None
        prefix = normalize_path(prefix)
        model = self._share_model
        now = datetime.now(UTC)
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = escaped + "/%"
        prefix_slash = prefix + "/"
        result = await session.execute(
            select(model).where(
                model.grantee_id == grantee_id,
                model.path.like(like_pattern, escape="\\"),  # type: ignore[union-attr]
            )
        )
        shares = result.scalars().all()
        active: list[FileShareBase] = []
        for share in shares:
            if not share.path.startswith(prefix_slash):
                continue
            if share.expires_at is not None:
                exp = share.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                if exp <= now:
                    continue
            active.append(share)
        return active

    async def _check_permission(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        required: str = "read",
    ) -> bool:
        """Check if *grantee_id* has *required* permission on *path*."""
        assert self._share_model is not None
        path = normalize_path(path)
        ancestors = []
        current = path
        while True:
            ancestors.append(current)
            if current == "/":
                break
            parent = current.rsplit("/", 1)[0] or "/"
            current = parent
        model = self._share_model
        now = datetime.now(UTC)
        result = await session.execute(
            select(model).where(
                model.grantee_id == grantee_id,
                model.path.in_(ancestors),  # type: ignore[union-attr]
            )
        )
        shares = result.scalars().all()
        for share in shares:
            if share.expires_at is not None:
                exp = share.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                if exp <= now:
                    continue
            if required == "read":
                return True
            if required == "write" and share.permission == "write":
                return True
        return False

    async def _update_share_paths(
        self,
        session: AsyncSession,
        old_prefix: str,
        new_prefix: str,
    ) -> int:
        """Bulk-update share paths when a file/directory is renamed."""
        assert self._share_model is not None
        old_prefix = normalize_path(old_prefix)
        new_prefix = normalize_path(new_prefix)
        model = self._share_model
        escaped = old_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        result = await session.execute(
            select(model).where(
                model.path.like(escaped + "%", escape="\\"),  # type: ignore[union-attr]
            )
        )
        shares = list(result.scalars().all())
        count = 0
        for share in shares:
            if share.path == old_prefix or share.path.startswith(old_prefix + "/"):
                share.path = new_prefix + share.path[len(old_prefix) :]
                count += 1
        if count > 0:
            await session.flush()
        return count

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
        if self._share_model is None:
            raise PermissionError("Access denied: sharing is not configured")
        has_access = await self._check_permission(
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
    ) -> ListDirResult:
        """List virtual ``@shared/`` directories.

        - ``/@shared`` → list distinct owners who shared with *user_id*
        - ``/@shared/{owner}`` → list that owner's shared files
        - ``/@shared/{owner}/sub/...`` → list sub-path (permission-checked)
        """
        if self._share_model is None:
            return ListDirResult(
                success=True,
                message="No sharing configured",
            )

        if len(segments) == 1:
            # /@shared — list distinct owners
            shares = await self._list_shared_with(session, user_id)
            owners: set[str] = set()
            for share in shares:
                parts = share.path.strip("/").split("/")
                if parts:
                    owners.add(parts[0])
            result_candidates: list[FileSearchCandidate] = []
            for owner in sorted(owners):
                entry_path = f"/@shared/{owner}"
                result_candidates.append(
                    FileSearchCandidate(
                        path=entry_path,
                        evidence=[
                            ListDirEvidence(
                                strategy="list_dir",
                                path=entry_path,
                                is_directory=True,
                            )
                        ],
                    )
                )
            return ListDirResult(
                success=True,
                message=f"Found {len(result_candidates)} shared owner(s)",
                candidates=result_candidates,
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
            result = result.remap_paths(
                lambda p: (
                    shared_prefix + (self._strip_user_prefix(p, owner) or "")
                    if self._strip_user_prefix(p, owner) != "/"
                    else shared_prefix
                )
            )
            return result

        except PermissionError:
            pass  # Fall through to filtered listing

        # Filtered fallback: show only files/dirs with specific shares
        shares = await self._list_shares_under_prefix(session, user_id, stored_path)
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

        result_candidates: list[FileSearchCandidate] = []
        base = shared_prefix if sub_path == "/" else f"{shared_prefix}{sub_path}"

        for name in sorted(direct_files):
            file_stored = f"{stored_path}/{name}"
            size_bytes: int | None = None
            try:
                info = await super().get_info(file_stored, session=session)
                if info.success:
                    size_bytes = info.size_bytes
            except Exception:
                pass
            entry_path = f"{base}/{name}"
            result_candidates.append(
                FileSearchCandidate(
                    path=entry_path,
                    evidence=[
                        ListDirEvidence(
                            strategy="list_dir",
                            path=entry_path,
                            is_directory=False,
                            size_bytes=size_bytes,
                        )
                    ],
                )
            )

        for name in sorted(child_dirs):
            entry_path = f"{base}/{name}"
            result_candidates.append(
                FileSearchCandidate(
                    path=entry_path,
                    evidence=[
                        ListDirEvidence(
                            strategy="list_dir",
                            path=entry_path,
                            is_directory=True,
                        )
                    ],
                )
            )

        return ListDirResult(
            success=True,
            message=f"Found {len(result_candidates)} shared item(s)",
            candidates=result_candidates,
        )

    # ------------------------------------------------------------------
    # Core protocol: GroverFileSystem (overrides)
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
        result.path = self._restore_path(result.path, uid, orig) or ""
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
        result.path = self._restore_path(result.path, uid, orig) or ""
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
        result.path = self._restore_path(result.path, uid, orig) or ""
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
        result.path = self._restore_path(result.path, uid, orig) or ""
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
        created_dirs, error = await self._mkdir_impl(
            sess,
            stored,
            parents,
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
    ) -> ListDirResult:
        uid = self._require_user_id(user_id)
        sess = self._require_session(session)

        # Handle @shared virtual directories
        segments = path.strip("/").split("/")
        if segments[0] == "@shared":
            return await self._list_shared_dir(segments, uid, sess)

        stored = self._resolve_path(path, uid)
        result = await super().list_dir(stored, session=sess)

        # Remap stored paths back to user-facing paths
        result = result.remap_paths(lambda p: self._restore_path(p, uid) or p)

        # At root, add virtual @shared/ entry
        if path == "/":
            result.candidates.append(
                FileSearchCandidate(
                    path="/@shared",
                    evidence=[
                        ListDirEvidence(
                            strategy="list_dir",
                            path="/@shared",
                            is_directory=True,
                            size_bytes=None,
                        )
                    ],
                )
            )

        return result

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ExistsResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            try:
                await self._check_share_access(sess, stored, uid, "read")
            except PermissionError:
                return ExistsResult(exists=False, path=path)
        return await super().exists(stored, session=sess)

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfoResult:
        uid = self._require_user_id(user_id)
        is_shared = self._is_shared_access(path)[0]
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        if is_shared:
            try:
                await self._check_share_access(sess, stored, uid, "read")
            except PermissionError:
                return FileInfoResult(success=False, message="Access denied", path=path)
        info = await super().get_info(stored, session=sess)
        if info.success:
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
        )
        # Update share paths after successful follow-move
        if result.success and follow and self._share_model is not None:
            await self._update_share_paths(sess, src_stored, dest_stored)
        src_orig = src if src_shared else None
        dest_orig = dest if dest_shared else None
        result.old_path = self._restore_path(result.old_path, uid, src_orig) or result.old_path
        result.new_path = self._restore_path(result.new_path, uid, dest_orig) or result.new_path
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
        src_file = await self._get_file_record(sess, src_stored)
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
        result.path = self._restore_path(result.path, uid, dest_orig) or ""
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
            result = result.remap_paths(
                lambda p: (
                    shared_prefix + (self._strip_user_prefix(p, owner) or "")
                    if self._strip_user_prefix(p, owner) != "/"
                    else shared_prefix
                )
            )
        else:
            result = result.remap_paths(lambda p: self._restore_path(p, uid) or p)
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
        candidate_paths: set[str] | None = None
        if glob_filter is not None:
            glob_result = await super().glob(glob_filter, stored, session=sess)
            if not glob_result.success:
                return GrepResult(
                    success=False,
                    message=glob_result.message,
                    pattern=pattern,
                )
            # Pass candidate paths as a synthetic glob filter that matches
            # the resolved stored paths. Since super().grep with glob_filter
            # calls self.glob(), we avoid the issue by passing None and
            # pre-filtering.
            resolved_glob_filter = None

            # Get the set of candidate file paths from glob
            candidate_paths = set(glob_result.files())
            if not candidate_paths:
                return GrepResult(
                    success=True,
                    message="No files match glob filter",
                    pattern=pattern,
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

        # If we pre-resolved a glob_filter, filter candidates to only those
        # files that matched the glob
        if candidate_paths is not None and not count_only:
            import copy as _copy

            result = _copy.copy(result)
            result.candidates = [c for c in result.candidates if c.path in candidate_paths]

        if is_shared and owner is not None:
            shared_prefix = f"/@shared/{owner}"
            result = result.remap_paths(
                lambda p: (
                    shared_prefix + (self._strip_user_prefix(p, owner) or "")
                    if self._strip_user_prefix(p, owner) != "/"
                    else shared_prefix
                )
            )
        else:
            result = result.remap_paths(lambda p: self._restore_path(p, uid) or p)
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
            result = result.remap_paths(
                lambda p: (
                    shared_prefix + (self._strip_user_prefix(p, owner) or "")
                    if self._strip_user_prefix(p, owner) != "/"
                    else shared_prefix
                )
            )
        else:
            result = result.remap_paths(lambda p: self._restore_path(p, uid) or p)
        return result

    # ------------------------------------------------------------------
    # Version operations (overrides)
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VersionResult:
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
            path=self._restore_path(stored, uid, orig) or "",
            restored_version=version,
            version=write_result.version,
        )
        return result

    # ------------------------------------------------------------------
    # Trash operations (overrides)
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> TrashResult:
        uid = self._require_user_id(user_id)
        result = await super().list_trash(
            session=session,
            owner_id=uid,
        )
        # Remap candidate paths to strip user prefix
        result = result.remap_paths(lambda p: self._strip_user_prefix(p, uid))
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
        result.path = self._restore_path(result.path, uid) or ""
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
    ) -> ShareResult:
        """Create a share on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._share_model is None:
            raise ValueError("No sharing service configured")
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        share_record = await self._create_share(
            sess,
            stored,
            grantee_id,
            permission,
            uid,
            expires_at=expires_at,
        )
        return ShareResult(
            success=True,
            message=f"Shared {path} with {grantee_id}",
            path=path,
            grantee_id=share_record.grantee_id,
            permission=share_record.permission,
            granted_by=share_record.granted_by,
        )

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareResult:
        """Remove a share on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._share_model is None:
            raise ValueError("No sharing service configured")
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        removed = await self._remove_share(sess, stored, grantee_id)
        if removed:
            return ShareResult(
                success=True,
                message=f"Removed share on {path} for {grantee_id}",
                path=path,
                grantee_id=grantee_id,
            )
        return ShareResult(
            success=False,
            message=f"No share found on {path} for {grantee_id}",
            path=path,
            grantee_id=grantee_id,
        )

    async def list_shares_on_path(
        self,
        path: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareSearchResult:
        """List shares on a user-facing path."""
        uid = self._require_user_id(user_id)
        if self._is_shared_access(path)[0]:
            raise PermissionError("Cannot manage shares on paths you do not own")
        if self._share_model is None:
            return ShareSearchResult(
                success=True,
                message="No sharing configured",
            )
        stored = self._resolve_path(path, uid)
        sess = self._require_session(session)
        shares = await self._list_shares_on_path(sess, stored)
        candidates = [
            FileSearchCandidate(
                path=path,
                evidence=[
                    ShareEvidence(
                        strategy="share",
                        path=path,
                        grantee_id=s.grantee_id,
                        permission=s.permission,
                        granted_by=s.granted_by,
                        expires_at=s.expires_at,
                    )
                ],
            )
            for s in shares
        ]
        return ShareSearchResult(
            success=True,
            message=f"Found {len(shares)} share(s)",
            candidates=candidates,
        )

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareSearchResult:
        """List all shares granted to *user_id*.

        Converts stored ``/{owner}/rest`` paths to ``@shared/{owner}/rest``
        user-facing paths.
        """
        uid = self._require_user_id(user_id)
        if self._share_model is None:
            return ShareSearchResult(
                success=True,
                message="No sharing configured",
            )
        sess = self._require_session(session)
        shares = await self._list_shared_with(sess, uid)
        candidates: list[FileSearchCandidate] = []
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
            candidates.append(
                FileSearchCandidate(
                    path=user_path,
                    evidence=[
                        ShareEvidence(
                            strategy="share",
                            path=user_path,
                            grantee_id=s.grantee_id,
                            permission=s.permission,
                            granted_by=s.granted_by,
                            expires_at=s.expires_at,
                        )
                    ],
                )
            )
        return ShareSearchResult(
            success=True,
            message=f"Found {len(shares)} share(s)",
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # File chunk operations (overrides with user scoping)
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult:
        # No user scoping on replace — stored path should already be resolved
        return await super().replace_file_chunks(file_path, chunks, session=session)

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult:
        # No user scoping on delete — stored path should already be resolved
        return await super().delete_file_chunks(file_path, session=session)

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkListResult:
        # No user scoping on list — stored path should already be resolved
        return await super().list_file_chunks(file_path, session=session)

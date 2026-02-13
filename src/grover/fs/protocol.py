"""StorageBackend protocol — runtime-checkable interfaces.

Split into a core protocol and opt-in capability protocols so that
non-SQL backends can implement just the core without being forced
to provide versioning, trash, or reconciliation.

The shared services (MetadataService, VersioningService, DirectoryService,
TrashService) and the orchestration functions in ``operations.py`` are
built on SQLAlchemy and are intended for SQL-backed backends only.
Non-SQL backends implement the StorageBackend protocol directly without
using these shared modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

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
        TreeResult,
        WriteResult,
    )


@runtime_checkable
class StorageBackend(Protocol):
    """Core interface every backend must implement.

    ``session`` is optional on all methods.  SQL backends should
    fail fast if ``session is None``.  Non-SQL backends ignore it.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Called at mount time.  No-op if not needed."""
        ...

    async def close(self) -> None:
        """Called on unmount / shutdown."""
        ...

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ReadResult: ...

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListResult: ...

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> bool: ...

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfo | None: ...

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

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
    ) -> WriteResult: ...

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
    ) -> EditResult: ...

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> DeleteResult: ...

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult: ...

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        follow: bool = False,
        sharing: Any = None,
        user_id: str | None = None,
    ) -> MoveResult: ...

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> WriteResult: ...

    # ------------------------------------------------------------------
    # Search / Query
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GlobResult: ...

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
    ) -> GrepResult: ...

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> TreeResult: ...


@runtime_checkable
class SupportsVersions(Protocol):
    """Opt-in: version listing, content retrieval, restore."""

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListVersionsResult: ...

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GetVersionContentResult: ...

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> RestoreResult: ...


@runtime_checkable
class SupportsTrash(Protocol):
    """Opt-in: soft-delete trash management."""

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> ListResult: ...

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> RestoreResult: ...

    async def empty_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> DeleteResult: ...


@runtime_checkable
class SupportsReBAC(Protocol):
    """Opt-in: relationship-based access control (user-scoped paths + sharing).

    Backends implementing this protocol handle per-user path namespacing,
    ``@shared`` virtual directory resolution, and share CRUD.  VFS delegates
    all user-scoping to backends that satisfy this protocol.
    """

    async def share(
        self,
        path: str,
        grantee_id: str,
        permission: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
        expires_at: Any | None = None,
    ) -> Any: ...

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> bool: ...

    async def list_shares_on_path(
        self,
        path: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> list[Any]: ...

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> list[Any]: ...


@runtime_checkable
class SupportsReconcile(Protocol):
    """Opt-in: disk ↔ DB reconciliation."""

    async def reconcile(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> dict[str, int]: ...

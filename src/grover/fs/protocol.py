"""StorageBackend protocol — runtime-checkable interfaces.

Split into a core protocol and opt-in capability protocols so that
non-SQL backends can implement just the core without being forced
to provide versioning, trash, or reconciliation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from .types import (
        DeleteResult,
        EditResult,
        FileInfo,
        GetVersionContentResult,
        ListResult,
        ListVersionsResult,
        MkdirResult,
        MoveResult,
        ReadResult,
        RestoreResult,
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
        self, path: str, offset: int = 0, limit: int = 2000,
        *, session: AsyncSession | None = None,
    ) -> ReadResult: ...

    async def list_dir(
        self, path: str = "/",
        *, session: AsyncSession | None = None,
    ) -> ListResult: ...

    async def exists(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> bool: ...

    async def get_info(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> FileInfo | None: ...

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write(
        self, path: str, content: str, created_by: str = "agent",
        *, overwrite: bool = True,
        session: AsyncSession | None = None,
    ) -> WriteResult: ...

    async def edit(
        self, path: str, old_string: str, new_string: str,
        replace_all: bool = False, created_by: str = "agent",
        *, session: AsyncSession | None = None,
    ) -> EditResult: ...

    async def delete(
        self, path: str, permanent: bool = False,
        *, session: AsyncSession | None = None,
    ) -> DeleteResult: ...

    async def mkdir(
        self, path: str, parents: bool = True,
        *, session: AsyncSession | None = None,
    ) -> MkdirResult: ...

    async def move(
        self, src: str, dest: str,
        *, session: AsyncSession | None = None,
    ) -> MoveResult: ...

    async def copy(
        self, src: str, dest: str,
        *, session: AsyncSession | None = None,
    ) -> WriteResult: ...


@runtime_checkable
class SupportsVersions(Protocol):
    """Opt-in: version listing, content retrieval, restore."""

    async def list_versions(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> ListVersionsResult: ...

    async def get_version_content(
        self, path: str, version: int,
        *, session: AsyncSession | None = None,
    ) -> GetVersionContentResult: ...

    async def restore_version(
        self, path: str, version: int,
        *, session: AsyncSession | None = None,
    ) -> RestoreResult: ...


@runtime_checkable
class SupportsTrash(Protocol):
    """Opt-in: soft-delete trash management."""

    async def list_trash(
        self, *, session: AsyncSession | None = None,
    ) -> ListResult: ...

    async def restore_from_trash(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> RestoreResult: ...

    async def empty_trash(
        self, *, session: AsyncSession | None = None,
    ) -> DeleteResult: ...


@runtime_checkable
class SupportsReconcile(Protocol):
    """Opt-in: disk ↔ DB reconciliation."""

    async def reconcile(
        self, *, session: AsyncSession | None = None,
    ) -> dict[str, int]: ...

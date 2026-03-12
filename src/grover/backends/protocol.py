"""GroverFileSystem protocol — runtime-checkable interfaces.

``GroverFileSystem`` is the single protocol that every backend must
implement.  It covers CRUD, queries, versioning, trash, search,
connections, and file chunks.

There are two opt-in capability protocols:

* ``SupportsReBAC`` — relationship-based access control
* ``SupportsReconcile`` — disk ↔ DB reconciliation

The shared services (DefaultVersionProvider, DirectoryService,
TrashService) and the orchestration functions in ``operations.py`` are
built on SQLAlchemy and are intended for SQL-backed backends only.
Non-SQL backends implement the GroverFileSystem protocol directly without
using these shared modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunk import FileChunkBase
    from grover.models.file import FileBase
    from grover.providers.search.extractors import EmbeddableChunk
    from grover.providers.search.types import SearchResult
    from grover.results.operations import (
        BatchChunkResult,
        BatchWriteResult,
        ChunkListResult,
        ChunkResult,
        ConnectionListResult,
        ConnectionResult,
        DeleteResult,
        EditResult,
        ExistsResult,
        FileInfoResult,
        GetVersionContentResult,
        MkdirResult,
        MoveResult,
        ReadResult,
        ReconcileResult,
        RestoreResult,
        ShareResult,
        VerifyVersionResult,
        WriteResult,
    )
    from grover.results.search import (
        GlobResult,
        GrepResult,
        ListDirResult,
        ShareSearchResult,
        TrashResult,
        TreeResult,
        VectorSearchResult,
        VersionResult,
    )


@runtime_checkable
class GroverFileSystem(Protocol):
    """Core interface every backend must implement."""

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
    ) -> ListDirResult: ...

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ExistsResult: ...

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfoResult: ...

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

    async def write_file(
        self,
        file: FileBase,
        *,
        overwrite: bool = True,
        created_by: str = "agent",
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> WriteResult: ...

    async def write_files(
        self,
        files: list[FileBase],
        *,
        overwrite: bool = True,
        created_by: str = "agent",
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> BatchWriteResult: ...

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

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VersionResult: ...

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

    async def verify_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VerifyVersionResult: ...

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> list[VerifyVersionResult]: ...

    # ------------------------------------------------------------------
    # Trash
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> TrashResult: ...

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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession | None = None,
    ) -> None: ...

    async def search_remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None: ...

    async def vector_search(self, query: str, k: int = 10) -> VectorSearchResult: ...

    async def lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SearchResult]: ...

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
        session: AsyncSession | None = None,
    ) -> ConnectionResult: ...

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> ConnectionResult: ...

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> ConnectionListResult: ...

    # ------------------------------------------------------------------
    # File chunks
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult: ...

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult: ...

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkListResult: ...

    async def write_chunk(
        self,
        chunk: FileChunkBase,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult: ...

    async def write_chunks(
        self,
        chunks: list[FileChunkBase],
        *,
        session: AsyncSession | None = None,
    ) -> BatchChunkResult: ...


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
        expires_at: datetime | None = None,
    ) -> ShareResult: ...

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareResult: ...

    async def list_shares_on_path(
        self,
        path: str,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareSearchResult: ...

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
        session: AsyncSession | None = None,
    ) -> ShareSearchResult: ...


@runtime_checkable
class SupportsReconcile(Protocol):
    """Opt-in: disk ↔ DB reconciliation."""

    async def reconcile(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> ReconcileResult: ...

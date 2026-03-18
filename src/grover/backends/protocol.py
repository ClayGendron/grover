"""GroverFileSystem protocol — runtime-checkable interfaces.

``GroverFileSystem`` is the single protocol that every backend must
implement.  It covers CRUD, queries, versioning, trash, search,
connections, file chunks, and graph operations.

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

    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.file import FileModelBase
    from grover.models.internal.results import (
        FileOperationResult,
        FileSearchResult,
        FileSearchSet,
        GroverResult,
    )
    from grover.providers.search.extractors import EmbeddableChunk


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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def list_dir(
        self,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

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
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def write_file(
        self,
        file: FileModelBase,
        *,
        overwrite: bool = True,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def write_files(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool = True,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession,
        follow: bool = False,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    # ------------------------------------------------------------------
    # Search / Query
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

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
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def verify_versions(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> list[FileOperationResult]: ...

    # ------------------------------------------------------------------
    # Trash
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult: ...

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    async def empty_trash(
        self,
        *,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult: ...

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession,
    ) -> None: ...

    async def search_remove_file(
        self,
        path: str,
        *,
        session: AsyncSession,
    ) -> None: ...

    async def vector_search(
        self, query: str, k: int = 10, *, candidates: FileSearchSet | None = None
    ) -> FileSearchResult: ...

    async def lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession,
    ) -> FileSearchResult: ...

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
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    # ------------------------------------------------------------------
    # FileModel chunks
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def write_chunk(
        self,
        chunk: FileChunkModelBase,
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    async def predecessors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def successors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def ancestors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def descendants(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def neighborhood(
        self,
        candidates: FileSearchSet,
        *,
        max_depth: int = 2,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def min_meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def pagerank(
        self,
        candidates: FileSearchSet,
        *,
        personalization: dict[str, float] | None = None,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def betweenness_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def closeness_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def katz_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def in_degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def out_degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...

    async def hits(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult: ...


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
        session: AsyncSession,
        expires_at: datetime | None = None,
    ) -> FileOperationResult: ...

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
        session: AsyncSession,
    ) -> FileOperationResult: ...

    async def list_shares_on_path(
        self,
        path: str,
        *,
        user_id: str,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
        session: AsyncSession,
    ) -> FileSearchResult: ...


@runtime_checkable
class SupportsReconcile(Protocol):
    """Opt-in: disk ↔ DB reconciliation."""

    async def reconcile(
        self,
        *,
        session: AsyncSession,
    ) -> FileOperationResult: ...

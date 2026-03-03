"""Provider protocols for filesystem-centric architecture.

All protocols are ``@runtime_checkable`` for ``isinstance()`` checks
at mount time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.fs.providers.search.types import DeleteResult as SearchDeleteResult
    from grover.fs.providers.search.types import UpsertResult, VectorEntry
    from grover.ref import Ref
    from grover.types.operations import (
        ChunkListResult,
        ChunkResult,
        FileInfoResult,
        ReconcileResult,
        VerifyVersionResult,
    )
    from grover.types.search import (
        GlobResult,
        GrepResult,
        LexicalSearchResult,
        ListDirResult,
        TreeResult,
        VectorSearchResult,
    )


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async-first protocol for text-to-vector embedding.

    Implementations convert text into fixed-dimension float vectors
    suitable for similarity search.
    """

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...


# ---------------------------------------------------------------------------
# Storage providers
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageProvider(Protocol):
    """Core storage operations — NO session parameters.

    Handles content I/O, file operations, and metadata for an external
    storage backend (e.g. local disk). When ``storage_provider`` is ``None``
    on ``DatabaseFileSystem``, all content lives in the DB content column.
    """

    async def read_content(self, path: str) -> str | None: ...

    async def write_content(self, path: str, content: str) -> None: ...

    async def delete_content(self, path: str) -> None: ...

    async def move_content(self, src: str, dest: str) -> None: ...

    async def copy_content(self, src: str, dest: str) -> None: ...

    async def exists(self, path: str) -> bool: ...

    async def mkdir(self, path: str, parents: bool = True) -> None: ...

    async def get_info(self, path: str) -> FileInfoResult: ...


@runtime_checkable
class SupportsStorageQueries(Protocol):
    """Disk-level glob/grep/tree/list_dir."""

    async def storage_glob(self, pattern: str, path: str = "/") -> GlobResult: ...

    async def storage_grep(self, pattern: str, path: str = "/", **kwargs: Any) -> GrepResult: ...

    async def storage_tree(self, path: str = "/", max_depth: int | None = None) -> TreeResult: ...

    async def storage_list_dir(self, path: str) -> ListDirResult: ...


@runtime_checkable
class SupportsStorageReconcile(Protocol):
    """Sync external storage with DB."""

    async def reconcile(self, **kwargs: Any) -> ReconcileResult: ...


# ---------------------------------------------------------------------------
# Graph provider
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Replaces the former ``GraphStore`` protocol. ``RustworkxGraph``
    implements this plus optional capability protocols.
    """

    # Node operations
    def add_node(self, path: str, **attrs: object) -> None: ...

    def remove_node(self, path: str) -> None: ...

    def has_node(self, path: str) -> bool: ...

    def get_node(self, path: str) -> dict: ...

    def nodes(self) -> list[str]: ...

    # Edge operations
    def add_edge(self, source: str, target: str, edge_type: str, **attrs: object) -> None: ...

    def remove_edge(self, source: str, target: str) -> None: ...

    def has_edge(self, source: str, target: str) -> bool: ...

    def get_edge(self, source: str, target: str) -> dict: ...

    def edges(self) -> list[tuple[str, str, dict]]: ...

    # Queries
    def dependents(self, path: str) -> list[Ref]: ...

    def dependencies(self, path: str) -> list[Ref]: ...

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]: ...

    def path_between(self, source: str, target: str) -> list[Ref] | None: ...

    def contains(self, path: str) -> list[Ref]: ...

    def by_parent(self, parent_path: str) -> list[Ref]: ...

    def remove_file_subgraph(self, path: str) -> list[str]: ...

    # Graph-level properties
    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    def is_dag(self) -> bool: ...


# ---------------------------------------------------------------------------
# Search provider
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchProvider(Protocol):
    """Search interface — vector + lexical.

    Existing stores (``LocalVectorStore``, ``PineconeVectorStore``,
    ``DatabricksVectorStore``) implement this directly. Replaces
    ``VectorStore`` as the filesystem-level type.
    """

    # Vector operations
    async def upsert(
        self,
        entries: list[VectorEntry],
        *,
        namespace: str | None = None,
    ) -> UpsertResult: ...

    async def vector_search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        namespace: str | None = None,
        filter: Any = None,  # noqa: A002
        include_metadata: bool = True,
        score_threshold: float | None = None,
    ) -> VectorSearchResult: ...

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> SearchDeleteResult: ...

    async def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> list[VectorEntry | None]: ...

    # Lexical search (stores that don't support it return empty result)
    async def lexical_search(self, query: str, *, k: int = 10) -> LexicalSearchResult: ...

    # Lifecycle
    async def connect(self) -> None: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Version provider
# ---------------------------------------------------------------------------


@runtime_checkable
class VersionProvider(Protocol):
    """Version storage — diff-based with periodic snapshots."""

    async def save_version(
        self,
        session: Any,
        file: Any,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None: ...

    async def delete_versions(self, session: Any, file_id: str) -> None: ...

    async def list_versions(self, session: Any, file: Any) -> list: ...

    async def get_version_content(self, session: Any, file: Any, version: int) -> str | None: ...

    async def verify_chain(self, session: Any, file: Any) -> VerifyVersionResult: ...


# ---------------------------------------------------------------------------
# Chunk provider
# ---------------------------------------------------------------------------


@runtime_checkable
class ChunkProvider(Protocol):
    """Chunk storage — file chunk CRUD."""

    async def replace_file_chunks(
        self, session: Any, file_path: str, chunks: list[dict]
    ) -> ChunkResult: ...

    async def delete_file_chunks(self, session: Any, file_path: str) -> ChunkResult: ...

    async def list_file_chunks(self, session: Any, file_path: str) -> ChunkListResult: ...

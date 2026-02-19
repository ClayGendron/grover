"""Search layer protocols — async-first interfaces for embedding and vector storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.search.filters import FilterExpression
    from grover.search.types import (
        DeleteResult,
        IndexConfig,
        IndexInfo,
        SparseVector,
        TextEntry,
        UpsertResult,
        VectorEntry,
        VectorSearchResult,
    )


# ------------------------------------------------------------------
# Core protocols
# ------------------------------------------------------------------


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


@runtime_checkable
class VectorStore(Protocol):
    """Async-first protocol for vector storage and search.

    Core data operations accept optional ``namespace`` and ``filter`` keyword
    arguments.  Stores that do not support them raise ``ValueError`` if a
    non-None value is passed.
    """

    async def upsert(
        self,
        entries: list[VectorEntry],
        *,
        namespace: str | None = None,
    ) -> UpsertResult:
        """Insert or update vector entries."""
        ...

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
        include_metadata: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for the *k* nearest vectors."""
        ...

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> DeleteResult:
        """Delete vectors by their IDs."""
        ...

    async def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> list[VectorEntry | None]:
        """Fetch vectors by their IDs.  Missing IDs return ``None``."""
        ...

    async def connect(self) -> None:
        """Open connection / initialize resources."""
        ...

    async def close(self) -> None:
        """Release connection / clean up resources."""
        ...

    @property
    def index_name(self) -> str:
        """Name of the underlying index."""
        ...


# ------------------------------------------------------------------
# Capability protocols — checked via isinstance() at runtime
# ------------------------------------------------------------------


@runtime_checkable
class SupportsNamespaces(Protocol):
    """Store supports namespace partitioning.

    Core ``VectorStore`` methods accept the ``namespace=`` kwarg; this
    protocol adds namespace management operations.
    """

    async def list_namespaces(self) -> list[str]:
        """List all namespaces in the index."""
        ...

    async def delete_namespace(self, namespace: str) -> None:
        """Delete an entire namespace and all its vectors."""
        ...


@runtime_checkable
class SupportsMetadataFilter(Protocol):
    """Store supports metadata filtering on ``search()``.

    The ``filter=`` parameter is already on ``VectorStore.search()``; this
    protocol's unique method is ``compile_filter``, which converts a
    provider-agnostic ``FilterExpression`` into the store's native format.
    """

    def compile_filter(self, expr: FilterExpression) -> Any:
        """Compile a ``FilterExpression`` to the store's native filter format."""
        ...


@runtime_checkable
class SupportsIndexLifecycle(Protocol):
    """Store supports programmatic index create/delete/list."""

    async def create_index(self, config: IndexConfig) -> None:
        """Create a new vector index."""
        ...

    async def delete_index(self, name: str) -> None:
        """Delete an existing vector index."""
        ...

    async def list_indexes(self) -> list[IndexInfo]:
        """List all available indexes."""
        ...


@runtime_checkable
class SupportsHybridSearch(Protocol):
    """Store supports hybrid (sparse+dense or keyword+vector) search."""

    async def hybrid_search(
        self,
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: SparseVector | None = None,
        query_text: str | None = None,
        k: int = 10,
        alpha: float = 0.5,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
    ) -> list[VectorSearchResult]:
        """Run a hybrid search combining dense, sparse, and/or keyword signals."""
        ...


@runtime_checkable
class SupportsReranking(Protocol):
    """Store supports server-side reranking of search results."""

    async def reranked_search(
        self,
        vector: list[float],
        query_text: str,
        *,
        k: int = 10,
        rerank_model: str | None = None,
        rerank_top_n: int | None = None,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
    ) -> list[VectorSearchResult]:
        """Search with server-side reranking applied to the results."""
        ...


@runtime_checkable
class SupportsTextSearch(Protocol):
    """Store embeds query text internally — no external EmbeddingProvider needed."""

    async def text_search(
        self,
        query: str,
        *,
        k: int = 10,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
    ) -> list[VectorSearchResult]:
        """Search using raw text (the store handles embedding internally)."""
        ...


@runtime_checkable
class SupportsTextIngest(Protocol):
    """Store embeds document text on ingest — no external EmbeddingProvider needed."""

    async def text_upsert(
        self,
        entries: list[TextEntry],
        *,
        namespace: str | None = None,
    ) -> UpsertResult:
        """Upsert text entries (the store handles embedding internally)."""
        ...

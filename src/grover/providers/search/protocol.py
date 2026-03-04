"""Search layer protocols — vector storage and capability interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.providers.search.filters import FilterExpression
    from grover.providers.search.types import (
        DeleteResult,
        IndexConfig,
        IndexInfo,
        SparseVector,
        UpsertResult,
        VectorEntry,
        VectorHit,
    )
    from grover.results.search import LexicalSearchResult, VectorSearchResult


# ---------------------------------------------------------------------------
# Core search protocol (filesystem-level)
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchProvider(Protocol):
    """Search interface — vector + lexical.

    Existing stores (``LocalVectorStore``, ``PineconeVectorStore``,
    ``DatabricksVectorStore``) implement this directly.
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
        filter: Any = None,
        include_metadata: bool = True,
        score_threshold: float | None = None,
    ) -> VectorSearchResult: ...

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> DeleteResult: ...

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


# ------------------------------------------------------------------
# Capability protocols — checked via isinstance() at runtime
# ------------------------------------------------------------------


@runtime_checkable
class SupportsNamespaces(Protocol):
    """Store supports namespace partitioning."""

    async def list_namespaces(self) -> list[str]: ...

    async def delete_namespace(self, namespace: str) -> None: ...


@runtime_checkable
class SupportsMetadataFilter(Protocol):
    """Store supports metadata filtering on ``search()``."""

    def compile_filter(self, expr: FilterExpression) -> object: ...


@runtime_checkable
class SupportsIndexLifecycle(Protocol):
    """Store supports programmatic index create/delete/list."""

    async def create_index(self, config: IndexConfig) -> None: ...

    async def delete_index(self, name: str) -> None: ...

    async def list_indexes(self) -> list[IndexInfo]: ...


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
    ) -> list[VectorHit]: ...


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
    ) -> list[VectorHit]: ...

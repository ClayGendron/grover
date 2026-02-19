"""PineconeVectorStore — Pinecone vector database backend."""

from __future__ import annotations

import logging
import os
from typing import Any

from grover.search.filters import FilterExpression, compile_pinecone
from grover.search.types import (
    DeleteResult,
    IndexConfig,
    IndexInfo,
    SparseVector,
    UpsertResult,
    VectorEntry,
    VectorSearchResult,
)

try:
    from pinecone import PineconeAsyncio, ServerlessSpec

    _HAS_PINECONE = True
except ImportError:  # pragma: no cover
    PineconeAsyncio = None  # type: ignore[assignment,misc]
    ServerlessSpec = None  # type: ignore[assignment,misc]
    _HAS_PINECONE = False

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 1000


class PineconeVectorStore:
    """Pinecone vector store with full capability support.

    Implements ``VectorStore``, ``SupportsNamespaces``,
    ``SupportsMetadataFilter``, ``SupportsIndexLifecycle``,
    ``SupportsHybridSearch``, and ``SupportsReranking``.

    Usage::

        store = PineconeVectorStore(index_name="my-index")
        await store.connect()
        await store.upsert([VectorEntry(id="a", vector=[0.1, ...], metadata={})])
        results = await store.search([0.1, ...], k=5)
        await store.close()
    """

    def __init__(
        self,
        *,
        index_name: str,
        api_key: str | None = None,
        namespace: str = "",
    ) -> None:
        if not _HAS_PINECONE:
            msg = (
                "pinecone is required for PineconeVectorStore. "
                "Install it with: pip install grover[pinecone]"
            )
            raise ImportError(msg)

        self._index_name = index_name
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self._default_namespace = namespace
        self._client: Any = None
        self._index: Any = None

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def upsert(
        self,
        entries: list[VectorEntry],
        *,
        namespace: str | None = None,
    ) -> UpsertResult:
        """Batch upsert vectors. Chunks at 1000 vectors per API call."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()

        total = 0
        for i in range(0, len(entries), _UPSERT_BATCH_SIZE):
            batch = entries[i : i + _UPSERT_BATCH_SIZE]
            vectors = [
                {
                    "id": e.id,
                    "values": e.vector,
                    "metadata": e.metadata,
                }
                for e in batch
            ]
            resp = await idx.upsert(vectors=vectors, namespace=ns)
            total += getattr(resp, "upserted_count", len(batch))

        return UpsertResult(upserted_count=total)

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
        """Query the index for nearest vectors."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()

        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": k,
            "namespace": ns,
            "include_metadata": include_metadata,
            "include_values": include_metadata,
        }
        if filter is not None:
            kwargs["filter"] = self.compile_filter(filter)

        resp = await idx.query(**kwargs)

        results: list[VectorSearchResult] = []
        for match in resp.matches:
            score = match.score
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                VectorSearchResult(
                    id=match.id,
                    score=score,
                    metadata=dict(match.metadata) if match.metadata else {},
                    vector=list(match.values) if match.values else None,
                )
            )
        return results

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> DeleteResult:
        """Delete vectors by their IDs."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()
        await idx.delete(ids=ids, namespace=ns)
        # Pinecone delete is fire-and-forget; actual count unknown
        return DeleteResult(deleted_count=len(ids))

    async def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> list[VectorEntry | None]:
        """Fetch vectors by their IDs."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()
        resp = await idx.fetch(ids=ids, namespace=ns)

        results: list[VectorEntry | None] = []
        vectors_map = resp.vectors if resp.vectors else {}
        for entry_id in ids:
            vec = vectors_map.get(entry_id)
            if vec is None:
                results.append(None)
            else:
                results.append(
                    VectorEntry(
                        id=vec.id,
                        vector=list(vec.values) if vec.values else [],
                        metadata=dict(vec.metadata) if vec.metadata else {},
                    )
                )
        return results

    async def connect(self) -> None:
        """Initialize the Pinecone async client and get index handle."""
        self._client = PineconeAsyncio(api_key=self._api_key)
        desc = await self._client.describe_index(self._index_name)
        host = desc.host
        self._index = self._client.IndexAsyncio(host=host)

    async def close(self) -> None:
        """Close the async client."""
        if self._index is not None:
            await self._index.close()
            self._index = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def index_name(self) -> str:
        """Return the index name."""
        return self._index_name

    # ------------------------------------------------------------------
    # SupportsNamespaces
    # ------------------------------------------------------------------

    async def list_namespaces(self) -> list[str]:
        """List all namespaces in the index."""
        idx = self._require_index()
        namespaces: list[str] = []
        async for page in idx.list_namespaces():
            if hasattr(page, "namespaces") and page.namespaces:
                for ns in page.namespaces:
                    name = getattr(ns, "name", str(ns))
                    namespaces.append(name)
        return namespaces

    async def delete_namespace(self, namespace: str) -> None:
        """Delete an entire namespace."""
        idx = self._require_index()
        await idx.delete_namespace(namespace)

    # ------------------------------------------------------------------
    # SupportsMetadataFilter
    # ------------------------------------------------------------------

    def compile_filter(self, expr: FilterExpression) -> dict[str, Any]:
        """Compile a FilterExpression to Pinecone's MongoDB-style filter dict."""
        return compile_pinecone(expr)

    # ------------------------------------------------------------------
    # SupportsIndexLifecycle
    # ------------------------------------------------------------------

    async def create_index(self, config: IndexConfig) -> None:
        """Create a Pinecone index."""
        client = self._require_client()

        spec_config = config.cloud_config
        if "cloud" in spec_config and "region" in spec_config:
            spec = ServerlessSpec(
                cloud=spec_config["cloud"],
                region=spec_config["region"],
            )
        else:
            spec = spec_config.get("spec", ServerlessSpec(cloud="aws", region="us-east-1"))

        await client.create_index(
            name=config.name,
            dimension=config.dimension,
            metric=config.metric,
            spec=spec,
        )

    async def delete_index(self, name: str) -> None:
        """Delete a Pinecone index."""
        client = self._require_client()
        await client.delete_index(name)

    async def list_indexes(self) -> list[IndexInfo]:
        """List all Pinecone indexes."""
        client = self._require_client()
        indexes = await client.list_indexes()
        return [
            IndexInfo(
                name=idx.name,
                dimension=idx.dimension,
                metric=idx.metric,
                vector_count=getattr(idx, "total_vector_count", 0) or 0,
                metadata={
                    "host": getattr(idx, "host", ""),
                    "status": getattr(idx, "status", {}),
                },
            )
            for idx in indexes
        ]

    # ------------------------------------------------------------------
    # SupportsHybridSearch
    # ------------------------------------------------------------------

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
        """Run a hybrid search combining dense and sparse vectors."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()

        kwargs: dict[str, Any] = {
            "top_k": k,
            "namespace": ns,
            "include_metadata": True,
            "include_values": True,
        }

        if dense_vector is not None:
            kwargs["vector"] = dense_vector
        if sparse_vector is not None:
            kwargs["sparse_vector"] = {
                "indices": sparse_vector.indices,
                "values": sparse_vector.values,
            }
        if filter is not None:
            kwargs["filter"] = self.compile_filter(filter)

        resp = await idx.query(**kwargs)

        return [
            VectorSearchResult(
                id=match.id,
                score=match.score,
                metadata=dict(match.metadata) if match.metadata else {},
                vector=list(match.values) if match.values else None,
            )
            for match in resp.matches
        ]

    # ------------------------------------------------------------------
    # SupportsReranking
    # ------------------------------------------------------------------

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
        """Search with server-side reranking via Pinecone Inference."""
        # First, get initial results
        search_results = await self.search(
            vector,
            k=k,
            namespace=namespace,
            filter=filter,
            include_metadata=True,
        )

        if not search_results:
            return []

        # Rerank using Pinecone Inference API
        client = self._require_client()
        documents = [{"id": r.id, "text": r.metadata.get("content", r.id)} for r in search_results]

        model = rerank_model or "bge-reranker-v2-m3"
        top_n = rerank_top_n or len(documents)

        rerank_resp = await client.inference.rerank(
            model=model,
            query=query_text,
            documents=documents,
            top_n=top_n,
            return_documents=True,
        )

        reranked: list[VectorSearchResult] = []
        for item in rerank_resp.data:
            original_idx = item.index
            original = search_results[original_idx]
            reranked.append(
                VectorSearchResult(
                    id=original.id,
                    score=item.score,
                    metadata=original.metadata,
                    vector=original.vector,
                )
            )
        return reranked

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_index(self) -> Any:
        """Return the index handle, raising if not connected."""
        if self._index is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._index

    def _require_client(self) -> Any:
        """Return the client, raising if not connected."""
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

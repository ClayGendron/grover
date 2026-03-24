"""PineconeVectorStore — Pinecone vector database backend."""

from __future__ import annotations

import logging
from typing import Any

from grover.models.internal.evidence import VectorEvidence
from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileOperationResult, FileSearchResult, FileSearchSet
from grover.providers.search.filters import FilterExpression, compile_pinecone
from grover.providers.search.protocol import IndexConfig, parent_path_from_id

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
    """Pinecone vector store.

    Implements ``SearchProvider``.  Also provides Pinecone-specific extras
    (namespaces, hybrid search, reranking, filter compilation) that are
    not part of the core protocol.

    Usage::

        store = PineconeVectorStore(index_name="my-index")
        await store.connect()
        await store.upsert(files=[File(path="/a.py", embedding=[0.1, ...])])
        result = await store.vector_search([0.1, ...], k=5)
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
            msg = "pinecone is required for PineconeVectorStore. Install it with: pip install grover[pinecone]"
            raise ImportError(msg)

        self._index_name = index_name
        self._api_key = api_key or ""
        self._default_namespace = namespace
        self._client: PineconeAsyncio | None = None
        self._index: Any | None = None

    # ------------------------------------------------------------------
    # SearchProvider protocol
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

    async def upsert(self, *, files: list[File], namespace: str | None = None) -> BatchResult:
        """Batch upsert vectors. Chunks at 1000 vectors per API call."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()

        succeeded = 0
        for i in range(0, len(files), _UPSERT_BATCH_SIZE):
            batch = files[i : i + _UPSERT_BATCH_SIZE]
            vectors = [
                {
                    "id": f.path,
                    "values": f.embedding,
                    "metadata": {},
                }
                for f in batch
            ]
            resp = await idx.upsert(vectors=vectors, namespace=ns)
            succeeded += getattr(resp, "upserted_count", len(batch))

        results = [FileOperationResult(file=File(path=f.path), success=True) for f in files]
        return BatchResult(
            results=results,
            succeeded=succeeded,
            failed=0,
            success=True,
            message=f"Upserted {succeeded} entries",
        )

    async def delete(self, *, files: list[str], namespace: str | None = None) -> BatchResult:
        """Delete vectors by their IDs."""
        ns = namespace if namespace is not None else self._default_namespace
        idx = self._require_index()
        await idx.delete(ids=files, namespace=ns)
        results = [FileOperationResult(file=File(path=p), success=True) for p in files]
        return BatchResult(
            results=results,
            succeeded=len(files),
            failed=0,
            success=True,
            message=f"Deleted {len(files)} entries",
        )

    async def vector_search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        candidates: FileSearchSet | None = None,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
        include_metadata: bool = True,
        score_threshold: float | None = None,
    ) -> FileSearchResult:
        """Query the index, returning a ``FileSearchResult``."""
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

        merged: dict[str, list[VectorEvidence]] = {}
        for match in resp.matches:
            score = match.score
            if score_threshold is not None and score < score_threshold:
                continue
            fp = parent_path_from_id(match.id)
            content = ""
            if match.metadata:
                content = match.metadata.get("content", "")
            snippet = content[:200] + ("..." if len(content) > 200 else "") if content else ""
            ev = VectorEvidence(operation="vector_search", snippet=snippet)
            merged.setdefault(fp, []).append(ev)

        result_files = [File(path=p, evidence=list(evs)) for p, evs in merged.items()]
        result = FileSearchResult(
            success=True,
            message=f"Found matches in {len(result_files)} file(s)",
            files=result_files,
        )

        if candidates is not None:
            allowed = set(candidates.paths)
            result.files = [f for f in result.files if f.path in allowed]
            result.message = f"Found {len(result.files)} match(es) (filtered)"

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
    # Pinecone-specific extras (not protocol)
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

    def compile_filter(self, expr: FilterExpression) -> dict[str, Any]:
        """Compile a FilterExpression to Pinecone's MongoDB-style filter dict."""
        return compile_pinecone(expr)

    async def delete_index(self, name: str) -> None:
        """Delete a Pinecone index."""
        client = self._require_client()
        await client.delete_index(name)

    async def list_indexes(self) -> list[dict[str, Any]]:
        """List all Pinecone indexes."""
        client = self._require_client()
        indexes = await client.list_indexes()
        return [
            {
                "name": idx.name,
                "dimension": idx.dimension,
                "metric": idx.metric,
                "vector_count": getattr(idx, "total_vector_count", 0) or 0,
                "host": getattr(idx, "host", ""),
                "status": getattr(idx, "status", {}),
            }
            for idx in indexes
        ]

    async def hybrid_search(
        self,
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: Any | None = None,
        query_text: str | None = None,
        k: int = 10,
        alpha: float = 0.5,
        namespace: str | None = None,
        filter: FilterExpression | None = None,
    ) -> FileSearchResult:
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

        merged: dict[str, list[VectorEvidence]] = {}
        for match in resp.matches:
            fp = parent_path_from_id(match.id)
            ev = VectorEvidence(operation="hybrid_search", snippet="")
            merged.setdefault(fp, []).append(ev)

        result_files = [File(path=p, evidence=list(evs)) for p, evs in merged.items()]
        return FileSearchResult(
            success=True,
            message=f"Found matches in {len(result_files)} file(s)",
            files=result_files,
        )

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
    ) -> FileSearchResult:
        """Search with server-side reranking via Pinecone Inference."""
        search_result = await self.vector_search(
            vector,
            k=k,
            namespace=namespace,
            filter=filter,
        )

        if not search_result.files:
            return search_result

        client = self._require_client()
        documents = [{"id": f.path, "text": f.path} for f in search_result.files]

        model = rerank_model or "bge-reranker-v2-m3"
        top_n = rerank_top_n or len(documents)

        rerank_resp = await client.inference.rerank(
            model=model,
            query=query_text,
            documents=documents,
            top_n=top_n,
            return_documents=True,
        )

        reranked_files: list[File] = []
        for item in rerank_resp.data:
            original = search_result.files[item.index]
            reranked_files.append(original)

        return FileSearchResult(
            success=True,
            message=f"Found matches in {len(reranked_files)} file(s)",
            files=reranked_files,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_index(self) -> Any:
        """Return the index handle, raising if not connected."""
        if self._index is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._index

    def _require_client(self) -> PineconeAsyncio:
        """Return the client, raising if not connected."""
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

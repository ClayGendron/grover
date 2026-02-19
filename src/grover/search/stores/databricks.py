"""DatabricksVectorStore — Databricks Vector Search backend (Direct Vector Access)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from grover.search.filters import FilterExpression, compile_databricks
from grover.search.types import (
    DeleteResult,
    IndexConfig,
    IndexInfo,
    UpsertResult,
    VectorEntry,
    VectorSearchResult,
)

try:
    from databricks.vector_search.client import VectorSearchClient

    _HAS_DATABRICKS = True
except ImportError:  # pragma: no cover
    VectorSearchClient = None  # type: ignore[assignment,misc]
    _HAS_DATABRICKS = False

logger = logging.getLogger(__name__)


class DatabricksVectorStore:
    """Databricks Vector Search store (Direct Vector Access mode).

    Implements ``VectorStore``, ``SupportsMetadataFilter``,
    ``SupportsIndexLifecycle``, and ``SupportsHybridSearch``.

    All SDK calls are wrapped in ``asyncio.to_thread`` because the
    Databricks SDK is synchronous.

    Usage::

        store = DatabricksVectorStore(
            index_name="catalog.schema.my_index",
            endpoint_name="my_endpoint",
        )
        await store.connect()
        await store.upsert([VectorEntry(id="a", vector=[0.1, ...], metadata={})])
        results = await store.search([0.1, ...], k=5)
        await store.close()
    """

    def __init__(
        self,
        *,
        index_name: str,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        embedding_vector_column: str = "vector",
        primary_key_column: str = "id",
    ) -> None:
        if not _HAS_DATABRICKS:
            msg = (
                "databricks-vectorsearch is required for DatabricksVectorStore. "
                "Install it with: pip install grover[databricks]"
            )
            raise ImportError(msg)

        self._index_name = index_name
        self._endpoint_name = endpoint_name
        self._host = host or os.environ.get("DATABRICKS_HOST", "")
        self._token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self._vector_column = embedding_vector_column
        self._pk_column = primary_key_column
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
        """Upsert vectors via asyncio.to_thread (SDK is sync-only)."""
        self._reject_namespace(namespace)
        idx = self._require_index()

        rows = [
            {
                self._pk_column: e.id,
                self._vector_column: e.vector,
                **e.metadata,
            }
            for e in entries
        ]
        await asyncio.to_thread(idx.upsert, rows)
        return UpsertResult(upserted_count=len(entries))

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
        self._reject_namespace(namespace)
        idx = self._require_index()

        kwargs: dict[str, Any] = {
            "query_vector": vector,
            "columns": [self._pk_column],
            "num_results": k,
        }
        if filter is not None:
            kwargs["filters"] = self.compile_filter(filter)
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        resp = await asyncio.to_thread(idx.similarity_search, **kwargs)
        return self._parse_search_response(resp, include_metadata)

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> DeleteResult:
        """Delete vectors by their primary keys."""
        self._reject_namespace(namespace)
        idx = self._require_index()
        await asyncio.to_thread(idx.delete, primary_keys=ids)
        return DeleteResult(deleted_count=len(ids))

    async def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> list[VectorEntry | None]:
        """Fetch vectors by their IDs via scan with primary key filter."""
        self._reject_namespace(namespace)
        idx = self._require_index()

        # Databricks has no direct get-by-ID; use primary key filter
        filter_str = f"{self._pk_column} IN ({', '.join(_quote(i) for i in ids)})"
        resp = await asyncio.to_thread(
            idx.similarity_search,
            query_vector=[0.0],  # dummy — overridden by filter
            columns=[self._pk_column, self._vector_column],
            filters=filter_str,
            num_results=len(ids),
        )

        # Build lookup from response
        found: dict[str, VectorEntry] = {}
        manifest = resp.get("manifest", {})
        columns = [c["name"] for c in manifest.get("columns", [])]
        rows = resp.get("result", {}).get("data_array", [])

        for row in rows:
            row_dict = dict(zip(columns, row, strict=False))
            entry_id = str(row_dict.get(self._pk_column, ""))
            vec = row_dict.get(self._vector_column)
            meta = {
                mk: mv
                for mk, mv in row_dict.items()
                if mk not in (self._pk_column, self._vector_column, "score")
            }
            found[entry_id] = VectorEntry(
                id=entry_id,
                vector=list(vec) if vec else [],
                metadata=meta,
            )

        return [found.get(entry_id) for entry_id in ids]

    async def connect(self) -> None:
        """Initialize the Databricks client and get index handle."""
        kwargs: dict[str, Any] = {}
        if self._host:
            kwargs["workspace_url"] = self._host
        if self._token:
            kwargs["personal_access_token"] = self._token
        self._client = VectorSearchClient(**kwargs)
        self._index = await asyncio.to_thread(
            self._client.get_index,
            endpoint_name=self._endpoint_name,
            index_name=self._index_name,
        )

    async def close(self) -> None:
        """Release references (SDK has no explicit close)."""
        self._index = None
        self._client = None

    @property
    def index_name(self) -> str:
        """Return the index name."""
        return self._index_name

    # ------------------------------------------------------------------
    # SupportsMetadataFilter
    # ------------------------------------------------------------------

    def compile_filter(self, expr: FilterExpression) -> str:
        """Compile a FilterExpression to a Databricks SQL-like filter string."""
        return compile_databricks(expr)

    # ------------------------------------------------------------------
    # SupportsIndexLifecycle
    # ------------------------------------------------------------------

    async def create_index(self, config: IndexConfig) -> None:
        """Create a Direct Vector Access index."""
        client = self._require_client()

        cloud_config = config.cloud_config
        endpoint = cloud_config.get("endpoint_name", self._endpoint_name)
        pk = cloud_config.get("primary_key", self._pk_column)
        vec_col = cloud_config.get("embedding_vector_column", self._vector_column)
        schema = cloud_config.get("schema")

        kwargs: dict[str, Any] = {
            "endpoint_name": endpoint,
            "index_name": config.name,
            "primary_key": pk,
            "embedding_dimension": config.dimension,
            "embedding_vector_column": vec_col,
        }
        if schema is not None:
            kwargs["schema"] = schema

        await asyncio.to_thread(client.create_direct_access_index, **kwargs)

    async def delete_index(self, name: str) -> None:
        """Delete a Databricks vector search index."""
        client = self._require_client()
        await asyncio.to_thread(client.delete_index, index_name=name)

    async def list_indexes(self) -> list[IndexInfo]:
        """List indexes on the configured endpoint."""
        client = self._require_client()
        indexes = await asyncio.to_thread(client.list_indexes, name=self._endpoint_name)

        return [
            IndexInfo(
                name=getattr(idx, "name", str(idx)),
                dimension=getattr(idx, "embedding_dimension", 0) or 0,
                metric="l2",
                vector_count=getattr(idx, "num_vectors", 0) or 0,
                metadata={
                    "endpoint": self._endpoint_name,
                    "status": getattr(idx, "status", {}),
                },
            )
            for idx in (indexes if indexes else [])
        ]

    # ------------------------------------------------------------------
    # SupportsHybridSearch
    # ------------------------------------------------------------------

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
    ) -> list[VectorSearchResult]:
        """Hybrid search combining vector similarity + keyword (BM25)."""
        self._reject_namespace(namespace)
        idx = self._require_index()

        kwargs: dict[str, Any] = {
            "columns": [self._pk_column],
            "num_results": k,
            "query_type": "HYBRID",
        }

        if dense_vector is not None:
            kwargs["query_vector"] = dense_vector
        if query_text is not None:
            kwargs["query_text"] = query_text
        if filter is not None:
            kwargs["filters"] = self.compile_filter(filter)

        resp = await asyncio.to_thread(idx.similarity_search, **kwargs)
        return self._parse_search_response(resp, include_metadata=True)

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

    @staticmethod
    def _reject_namespace(namespace: str | None) -> None:
        """Raise ValueError if a namespace is provided."""
        if namespace is not None:
            msg = "DatabricksVectorStore does not support namespaces"
            raise ValueError(msg)

    def _parse_search_response(
        self,
        resp: dict[str, Any],
        include_metadata: bool,
    ) -> list[VectorSearchResult]:
        """Parse a Databricks similarity_search response into VectorSearchResults."""
        manifest = resp.get("manifest", {})
        columns = [c["name"] for c in manifest.get("columns", [])]
        rows = resp.get("result", {}).get("data_array", [])

        results: list[VectorSearchResult] = []
        for row in rows:
            row_dict = dict(zip(columns, row, strict=False))
            entry_id = str(row_dict.get(self._pk_column, ""))
            score = float(row_dict.get("score", 0.0))

            metadata: dict[str, Any] = {}
            if include_metadata:
                metadata = {
                    mk: mv
                    for mk, mv in row_dict.items()
                    if mk not in (self._pk_column, "score", self._vector_column)
                }

            results.append(
                VectorSearchResult(
                    id=entry_id,
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


def _quote(value: str) -> str:
    """Quote a string value for Databricks SQL-like filters."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"

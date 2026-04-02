"""DatabricksVectorStore — Databricks Vector Search backend.

Implements the ``VectorStore`` protocol using Databricks Direct Vector
Access indexes.  All SDK calls are wrapped in ``asyncio.to_thread``
because the Databricks SDK is synchronous.

Requires ``databricks-vectorsearch``::

    pip install grover[databricks]
"""

from __future__ import annotations

import asyncio
from typing import Any

from grover.vector_store import VectorHit, VectorItem

try:
    from databricks.vector_search.client import VectorSearchClient

    _HAS_DATABRICKS = True
except ImportError:  # pragma: no cover
    VectorSearchClient = None  # ty: ignore[invalid-assignment]
    _HAS_DATABRICKS = False

_UPSERT_BATCH_SIZE = 1000


class DatabricksVectorStore:
    """Databricks Vector Search store (Direct Vector Access mode).

    Usage::

        store = DatabricksVectorStore(
            index_name="catalog.schema.my_index",
            endpoint_name="my_endpoint",
        )
        await store.connect()
        hits = await store.query([0.1, 0.2, ...], k=5)
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
                "databricks-vectorsearch is required for DatabricksVectorStore."
                " Install it with: pip install grover[databricks]"
            )
            raise ImportError(msg)

        self._index_name = index_name
        self._endpoint_name = endpoint_name
        self._host = host or ""
        self._token = token or ""
        self._vector_column = embedding_vector_column
        self._pk_column = primary_key_column
        self._client: VectorSearchClient | None = None
        self._index: Any | None = None

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def query(
        self,
        vector: list[float],
        *,
        k: int = 10,
        paths: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[VectorHit]:
        """Find the *k* nearest vectors via similarity_search."""
        idx = self._require_index()

        filters: dict[str, str] = {}
        if user_id is not None:
            filters["owner_id"] = user_id

        resp: dict[str, Any] = await asyncio.to_thread(
            idx.similarity_search,
            query_vector=vector,
            columns=[self._pk_column],
            num_results=k,
            filters=filters if filters else None,
        )

        manifest = resp.get("manifest", {})
        columns = [c["name"] for c in manifest.get("columns", [])]
        rows = resp.get("result", {}).get("data_array", [])

        score_col = "score"
        score_idx = columns.index(score_col) if score_col in columns else None
        pk_idx = columns.index(self._pk_column) if self._pk_column in columns else 0

        hits: list[VectorHit] = []
        for row in rows:
            path = str(row[pk_idx])
            score = float(row[score_idx]) if score_idx is not None else 0.0
            hits.append(VectorHit(path=path, score=score))

        if paths is not None:
            allowed = set(paths)
            hits = [h for h in hits if h.path in allowed]

        return hits

    async def upsert(self, items: list[VectorItem]) -> None:
        """Insert or update vectors, batching at 1000 per API call."""
        idx = self._require_index()

        for start in range(0, len(items), _UPSERT_BATCH_SIZE):
            batch = items[start : start + _UPSERT_BATCH_SIZE]
            rows = [
                {
                    self._pk_column: item.path,
                    self._vector_column: item.vector,
                }
                for item in batch
            ]
            await asyncio.to_thread(idx.upsert, rows)

    async def delete(self, paths: list[str]) -> None:
        """Remove vectors by their primary keys."""
        idx = self._require_index()
        await asyncio.to_thread(idx.delete, primary_keys=paths)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the Databricks client and get the index handle."""
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_index(self) -> Any:
        """Return the index handle, raising if not connected."""
        if self._index is None:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)
        return self._index

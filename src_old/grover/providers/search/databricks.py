"""DatabricksVectorStore — Databricks Vector Search backend (Direct Vector Access)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from grover.models.internal.evidence import VectorEvidence
from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileOperationResult, FileSearchResult, FileSearchSet
from grover.providers.search.protocol import IndexConfig, parent_path_from_id

try:
    from databricks.vector_search.client import VectorSearchClient

    _HAS_DATABRICKS = True
except ImportError:  # pragma: no cover
    VectorSearchClient = None  # type: ignore[assignment,misc]
    _HAS_DATABRICKS = False

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 1000


class DatabricksVectorStore:
    """Databricks Vector Search store (Direct Vector Access mode).

    Implements ``SearchProvider``.

    All SDK calls are wrapped in ``asyncio.to_thread`` because the
    Databricks SDK is synchronous.

    Usage::

        store = DatabricksVectorStore(
            index_name="catalog.schema.my_index",
            endpoint_name="my_endpoint",
        )
        await store.connect()
        await store.upsert(files=[File(path="/a.py", embedding=[0.1, ...])])
        result = await store.vector_search([0.1, ...], k=5)
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
        self._host = host or ""
        self._token = token or ""
        self._vector_column = embedding_vector_column
        self._pk_column = primary_key_column
        self._client: VectorSearchClient | None = None
        self._index: Any | None = None

    # ------------------------------------------------------------------
    # SearchProvider protocol
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

    async def upsert(self, *, files: list[File]) -> BatchResult:
        """Upsert vectors via asyncio.to_thread (SDK is sync-only)."""
        idx = self._require_index()

        succeeded = 0

        for i in range(0, len(files), _UPSERT_BATCH_SIZE):
            batch = files[i : i + _UPSERT_BATCH_SIZE]
            rows = [
                {
                    self._pk_column: f.path,
                    self._vector_column: f.embedding,
                }
                for f in batch
            ]
            await asyncio.to_thread(idx.upsert, rows)
            succeeded += len(batch)

        results = [FileOperationResult(file=File(path=f.path), success=True) for f in files]

        return BatchResult(
            results=results,
            succeeded=succeeded,
            failed=0,
            success=True,
            message=f"Upserted {succeeded} entries",
        )

    async def delete(self, *, files: list[str]) -> BatchResult:
        """Delete vectors by their primary keys."""
        idx = self._require_index()
        await asyncio.to_thread(idx.delete, primary_keys=files)
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
    ) -> FileSearchResult:
        """Query the index for nearest vectors, returning a ``FileSearchResult``."""
        idx = self._require_index()

        kwargs: dict[str, Any] = {
            "query_vector": vector,
            "columns": [self._pk_column],
            "num_results": k,
        }

        resp = await asyncio.to_thread(idx.similarity_search, **kwargs)

        # Parse response
        manifest = resp.get("manifest", {})
        columns = [c["name"] for c in manifest.get("columns", [])]
        rows = resp.get("result", {}).get("data_array", [])

        merged: dict[str, list[VectorEvidence]] = {}
        for row in rows:
            row_dict = dict(zip(columns, row, strict=False))
            entry_id = str(row_dict.get(self._pk_column, ""))
            fp = parent_path_from_id(entry_id)
            ev = VectorEvidence(operation="vector_search", snippet="")
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
    # Lifecycle (concrete, not protocol)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_index(self) -> Any:
        """Return the index handle, raising if not connected."""
        if self._index is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._index

    def _require_client(self) -> VectorSearchClient:
        """Return the client, raising if not connected."""
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

"""Tests for DatabricksVectorStore — all operations mock the Databricks SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileSearchResult
from grover.providers.search.databricks import DatabricksVectorStore
from grover.providers.search.protocol import IndexConfig

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_search_response(
    rows: list[list],
    columns: list[str] | None = None,
) -> dict:
    """Build a mock Databricks similarity_search response dict."""
    if columns is None:
        columns = ["id", "score"]
    return {
        "manifest": {
            "column_count": len(columns),
            "columns": [{"name": c} for c in columns],
        },
        "result": {
            "row_count": len(rows),
            "data_array": rows,
        },
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_index():
    """A mock VectorSearchIndex."""
    idx = MagicMock()
    idx.upsert = MagicMock()
    idx.similarity_search = MagicMock(return_value=_make_search_response([], ["id", "score"]))
    idx.delete = MagicMock()
    return idx


@pytest.fixture
def mock_client(mock_index):
    """A mock VectorSearchClient."""
    client = MagicMock()
    client.get_index = MagicMock(return_value=mock_index)
    client.create_direct_access_index = MagicMock()
    return client


@pytest.fixture
async def store(mock_client, mock_index):
    """A connected DatabricksVectorStore with mocked SDK."""
    with patch(
        "grover.providers.search.databricks.VectorSearchClient",
        return_value=mock_client,
    ):
        s = DatabricksVectorStore(
            index_name="catalog.schema.test_idx",
            endpoint_name="test-endpoint",
            host="https://test.databricks.net",
            token="fake-token",
        )
        await s.connect()
        yield s
        await s.close()


# ==================================================================
# Connect / Close
# ==================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_creates_client_and_index(self, mock_client, mock_index):
        with patch(
            "grover.providers.search.databricks.VectorSearchClient",
            return_value=mock_client,
        ):
            s = DatabricksVectorStore(
                index_name="catalog.schema.idx",
                endpoint_name="ep",
                host="https://ws.databricks.net",
                token="tok",
            )
            await s.connect()
            mock_client.get_index.assert_called_once_with(
                endpoint_name="ep",
                index_name="catalog.schema.idx",
            )
            await s.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, store):
        await store.close()
        assert store._index is None
        assert store._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, store):
        await store.close()
        await store.close()  # should not raise

    @pytest.mark.asyncio
    async def test_operations_before_connect_raise(self):
        s = DatabricksVectorStore(index_name="x", endpoint_name="e", host="h", token="t")
        with pytest.raises(RuntimeError, match="Not connected"):
            await s.upsert(files=[File(path="/a.py", embedding=[0.1])])


# ==================================================================
# Upsert
# ==================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store, mock_index):
        result = await store.upsert(files=[File(path="/a.py", embedding=[0.1, 0.2])])
        assert isinstance(result, BatchResult)
        assert result.succeeded == 1
        mock_index.upsert.assert_called_once()
        rows = mock_index.upsert.call_args.args[0]
        assert len(rows) == 1
        assert rows[0]["id"] == "/a.py"
        assert rows[0]["vector"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store, mock_index):
        files = [File(path=f"/f{i}.py", embedding=[0.1]) for i in range(5)]
        result = await store.upsert(files=files)
        assert result.succeeded == 5


# ==================================================================
# Delete
# ==================================================================


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_by_ids(self, store, mock_index):
        result = await store.delete(files=["/a.py", "/b.py"])
        assert isinstance(result, BatchResult)
        assert result.succeeded == 2
        mock_index.delete.assert_called_once_with(primary_keys=["/a.py", "/b.py"])


# ==================================================================
# Vector Search
# ==================================================================


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_vector_search_basic(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", 0.95], ["/b.py", 0.80]],
            ["id", "score"],
        )
        result = await store.vector_search([0.1, 0.2], k=5)
        assert isinstance(result, FileSearchResult)
        assert result.success
        assert len(result.files) == 2
        assert result.files[0].path == "/a.py"

    @pytest.mark.asyncio
    async def test_vector_search_groups_by_parent(self, store, mock_index):
        """Chunk IDs like /a.py#foo get grouped under parent /a.py."""
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py#foo", 0.95], ["/a.py#bar", 0.90]],
            ["id", "score"],
        )
        result = await store.vector_search([0.1], k=5)
        assert len(result.files) == 1
        assert result.files[0].path == "/a.py"
        # Should have 2 evidence entries (one per chunk hit)
        assert len(result.files[0].evidence) == 2

    @pytest.mark.asyncio
    async def test_vector_search_empty(self, store, mock_index):
        result = await store.vector_search([0.1], k=5)
        assert result.success
        assert len(result.files) == 0

    @pytest.mark.asyncio
    async def test_vector_search_with_candidates(self, store, mock_index):
        from grover.models.internal.results import FileSearchSet

        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", 0.95], ["/b.py", 0.80]],
            ["id", "score"],
        )
        candidates = FileSearchSet.from_paths(["/a.py"])
        result = await store.vector_search([0.1], k=5, candidates=candidates)
        assert len(result.files) == 1
        assert result.files[0].path == "/a.py"


# ==================================================================
# Index Lifecycle
# ==================================================================


class TestIndexLifecycle:
    @pytest.mark.asyncio
    async def test_create_index(self, store, mock_client):
        config = IndexConfig(
            name="catalog.schema.new_idx",
            dimension=128,
            metric="cosine",
            cloud_config={
                "endpoint_name": "my-ep",
                "primary_key": "pk",
                "embedding_vector_column": "vec",
                "schema": {"pk": "string", "vec": "array<float>"},
            },
        )
        await store.create_index(config)
        mock_client.create_direct_access_index.assert_called_once()
        call_kwargs = mock_client.create_direct_access_index.call_args.kwargs
        assert call_kwargs["index_name"] == "catalog.schema.new_idx"
        assert call_kwargs["embedding_dimension"] == 128
        assert call_kwargs["primary_key"] == "pk"

    @pytest.mark.asyncio
    async def test_create_index_defaults(self, store, mock_client):
        """Uses store's endpoint_name and pk_column when not in cloud_config."""
        config = IndexConfig(name="catalog.schema.idx", dimension=64, metric="cosine")
        await store.create_index(config)
        call_kwargs = mock_client.create_direct_access_index.call_args.kwargs
        assert call_kwargs["endpoint_name"] == "test-endpoint"
        assert call_kwargs["primary_key"] == "id"


# ==================================================================
# Import guard
# ==================================================================


class TestImportGuard:
    def test_import_guard_message(self):
        with (
            patch.dict("sys.modules", {"databricks": None, "databricks.vector_search": None}),
            patch("grover.providers.search.databricks._HAS_DATABRICKS", False),
            pytest.raises(ImportError, match="databricks-vectorsearch is required"),
        ):
            DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")


# ==================================================================
# isinstance checks
# ==================================================================


class TestProtocolConformance:
    def test_satisfies_search_provider(self):
        from grover.providers.search.protocol import SearchProvider

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert isinstance(s, SearchProvider)

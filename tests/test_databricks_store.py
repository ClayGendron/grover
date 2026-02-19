"""Tests for DatabricksVectorStore — all operations mock the Databricks SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grover.search.filters import and_, eq, gt
from grover.search.stores.databricks import DatabricksVectorStore
from grover.search.types import IndexConfig, VectorEntry, VectorSearchResult

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
    client.delete_index = MagicMock()
    client.list_indexes = MagicMock(return_value=[])
    return client


@pytest.fixture
async def store(mock_client, mock_index):
    """A connected DatabricksVectorStore with mocked SDK."""
    with patch(
        "grover.search.stores.databricks.VectorSearchClient",
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
            "grover.search.stores.databricks.VectorSearchClient",
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
            await s.upsert([VectorEntry(id="a", vector=[0.1], metadata={})])


# ==================================================================
# Upsert
# ==================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store, mock_index):
        entry = VectorEntry(id="/a.py", vector=[0.1, 0.2], metadata={"lang": "python"})
        result = await store.upsert([entry])
        assert result.upserted_count == 1
        mock_index.upsert.assert_called_once()
        rows = mock_index.upsert.call_args.args[0]
        assert len(rows) == 1
        assert rows[0]["id"] == "/a.py"
        assert rows[0]["vector"] == [0.1, 0.2]
        assert rows[0]["lang"] == "python"

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store, mock_index):
        entries = [VectorEntry(id=f"/f{i}.py", vector=[0.1], metadata={}) for i in range(5)]
        result = await store.upsert(entries)
        assert result.upserted_count == 5

    @pytest.mark.asyncio
    async def test_upsert_namespace_raises(self, store):
        with pytest.raises(ValueError, match="does not support namespaces"):
            await store.upsert(
                [VectorEntry(id="a", vector=[0.1], metadata={})],
                namespace="ns",
            )


# ==================================================================
# Search
# ==================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_basic(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", 0.95], ["/b.py", 0.80]],
            ["id", "score"],
        )
        results = await store.search([0.1, 0.2], k=5)
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert results[0].id == "/a.py"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response([], ["id", "score"])
        await store.search([0.1], k=5, filter=eq("lang", "python"))
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["filters"] == "lang = 'python'"

    @pytest.mark.asyncio
    async def test_search_with_complex_filter(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response([], ["id", "score"])
        f = and_(eq("lang", "python"), gt("year", 2020))
        await store.search([0.1], k=5, filter=f)
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["filters"] == "(lang = 'python' AND year > 2020)"

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response([], ["id", "score"])
        await store.search([0.1], k=5, score_threshold=0.5)
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_search_namespace_raises(self, store):
        with pytest.raises(ValueError, match="does not support namespaces"):
            await store.search([0.1], k=5, namespace="ns")

    @pytest.mark.asyncio
    async def test_search_with_metadata(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", "python", 0.95]],
            ["id", "lang", "score"],
        )
        results = await store.search([0.1], k=5)
        assert results[0].metadata == {"lang": "python"}

    @pytest.mark.asyncio
    async def test_search_include_metadata_false(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", "python", 0.95]],
            ["id", "lang", "score"],
        )
        results = await store.search([0.1], k=5, include_metadata=False)
        assert results[0].metadata == {}


# ==================================================================
# Delete
# ==================================================================


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_by_ids(self, store, mock_index):
        result = await store.delete(["/a.py", "/b.py"])
        assert result.deleted_count == 2
        mock_index.delete.assert_called_once_with(primary_keys=["/a.py", "/b.py"])

    @pytest.mark.asyncio
    async def test_delete_namespace_raises(self, store):
        with pytest.raises(ValueError, match="does not support namespaces"):
            await store.delete(["/a.py"], namespace="ns")


# ==================================================================
# Fetch
# ==================================================================


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_existing(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", [0.1, 0.2], 0.99]],
            ["id", "vector", "score"],
        )
        results = await store.fetch(["/a.py"])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == "/a.py"
        assert results[0].vector == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_fetch_missing(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [], ["id", "vector", "score"]
        )
        results = await store.fetch(["/missing.py"])
        assert results == [None]

    @pytest.mark.asyncio
    async def test_fetch_mixed(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", [0.1], 0.99]],
            ["id", "vector", "score"],
        )
        results = await store.fetch(["/a.py", "/missing.py"])
        assert results[0] is not None
        assert results[0].id == "/a.py"
        assert results[1] is None

    @pytest.mark.asyncio
    async def test_fetch_namespace_raises(self, store):
        with pytest.raises(ValueError, match="does not support namespaces"):
            await store.fetch(["/a.py"], namespace="ns")


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

    @pytest.mark.asyncio
    async def test_delete_index(self, store, mock_client):
        await store.delete_index("catalog.schema.old")
        mock_client.delete_index.assert_called_once_with(index_name="catalog.schema.old")

    @pytest.mark.asyncio
    async def test_list_indexes(self, store, mock_client):
        idx_mock = MagicMock()
        idx_mock.name = "catalog.schema.idx1"
        idx_mock.embedding_dimension = 128
        idx_mock.num_vectors = 1000
        idx_mock.status = {"ready": True}
        mock_client.list_indexes.return_value = [idx_mock]

        results = await store.list_indexes()
        assert len(results) == 1
        assert results[0].name == "catalog.schema.idx1"
        assert results[0].dimension == 128
        assert results[0].vector_count == 1000

    @pytest.mark.asyncio
    async def test_list_indexes_empty(self, store, mock_client):
        mock_client.list_indexes.return_value = []
        results = await store.list_indexes()
        assert results == []


# ==================================================================
# Hybrid Search
# ==================================================================


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_with_vector(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", 0.9]],
            ["id", "score"],
        )
        results = await store.hybrid_search(dense_vector=[0.1, 0.2], k=5)
        assert len(results) == 1
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["query_vector"] == [0.1, 0.2]
        assert call_kwargs["query_type"] == "HYBRID"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_text(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response(
            [["/a.py", 0.85]],
            ["id", "score"],
        )
        results = await store.hybrid_search(query_text="search query", k=5)
        assert len(results) == 1
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["query_text"] == "search query"
        assert call_kwargs["query_type"] == "HYBRID"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filter(self, store, mock_index):
        mock_index.similarity_search.return_value = _make_search_response([], ["id", "score"])
        await store.hybrid_search(dense_vector=[0.1], k=5, filter=eq("type", "code"))
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["filters"] == "type = 'code'"

    @pytest.mark.asyncio
    async def test_hybrid_search_namespace_raises(self, store):
        with pytest.raises(ValueError, match="does not support namespaces"):
            await store.hybrid_search(dense_vector=[0.1], k=5, namespace="ns")


# ==================================================================
# Properties
# ==================================================================


class TestProperties:
    def test_index_name(self):
        s = DatabricksVectorStore(index_name="catalog.schema.idx", endpoint_name="ep", token="t")
        assert s.index_name == "catalog.schema.idx"


# ==================================================================
# Import guard
# ==================================================================


class TestImportGuard:
    def test_import_guard_message(self):
        with (
            patch.dict("sys.modules", {"databricks": None, "databricks.vector_search": None}),
            patch("grover.search.stores.databricks._HAS_DATABRICKS", False),
            pytest.raises(ImportError, match="databricks-vectorsearch is required"),
        ):
            DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")


# ==================================================================
# isinstance checks
# ==================================================================


class TestProtocolConformance:
    def test_satisfies_vector_store(self):
        from grover.search.protocols import VectorStore

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert isinstance(s, VectorStore)

    def test_satisfies_supports_metadata_filter(self):
        from grover.search.protocols import SupportsMetadataFilter

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert isinstance(s, SupportsMetadataFilter)

    def test_satisfies_supports_index_lifecycle(self):
        from grover.search.protocols import SupportsIndexLifecycle

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert isinstance(s, SupportsIndexLifecycle)

    def test_satisfies_supports_hybrid_search(self):
        from grover.search.protocols import SupportsHybridSearch

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert isinstance(s, SupportsHybridSearch)

    def test_does_not_satisfy_supports_namespaces(self):
        from grover.search.protocols import SupportsNamespaces

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert not isinstance(s, SupportsNamespaces)

    def test_does_not_satisfy_supports_reranking(self):
        from grover.search.protocols import SupportsReranking

        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        assert not isinstance(s, SupportsReranking)


# ==================================================================
# Compile filter
# ==================================================================


class TestCompileFilter:
    def test_compile_eq(self):
        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        result = s.compile_filter(eq("color", "red"))
        assert result == "color = 'red'"

    def test_compile_and(self):
        s = DatabricksVectorStore(index_name="x", endpoint_name="e", token="t")
        result = s.compile_filter(and_(eq("a", "x"), gt("b", 2)))
        assert result == "(a = 'x' AND b > 2)"

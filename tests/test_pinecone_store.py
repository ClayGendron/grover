"""Tests for PineconeVectorStore — all operations mock the Pinecone SDK."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grover.search.filters import and_, eq, gt
from grover.search.stores.pinecone import PineconeVectorStore
from grover.search.types import (
    IndexConfig,
    SparseVector,
    VectorEntry,
    VectorSearchResult,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_match(
    match_id: str, score: float, metadata: dict | None = None, values: list | None = None
):
    """Create a mock Pinecone match object."""
    m = SimpleNamespace()
    m.id = match_id
    m.score = score
    m.metadata = metadata or {}
    m.values = values or []
    return m


def _make_index_model(
    name: str, dimension: int = 128, metric: str = "cosine", host: str = "host.pinecone.io"
):
    """Create a mock Pinecone IndexModel."""
    m = SimpleNamespace()
    m.name = name
    m.dimension = dimension
    m.metric = metric
    m.host = host
    m.total_vector_count = 42
    m.status = {"ready": True}
    return m


def _make_fetch_vector(vec_id: str, values: list[float], metadata: dict | None = None):
    """Create a mock Pinecone fetched vector."""
    v = SimpleNamespace()
    v.id = vec_id
    v.values = values
    v.metadata = metadata or {}
    return v


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_index():
    """A mock IndexAsyncio."""
    idx = AsyncMock()
    idx.upsert = AsyncMock()
    idx.query = AsyncMock()
    idx.delete = AsyncMock()
    idx.fetch = AsyncMock()
    idx.list_namespaces = AsyncMock()
    idx.delete_namespace = AsyncMock()
    idx.close = AsyncMock()
    return idx


@pytest.fixture
def mock_client(mock_index):
    """A mock PineconeAsyncio client."""
    client = AsyncMock()
    client.describe_index = AsyncMock(
        return_value=SimpleNamespace(host="test-host.pinecone.io")
    )
    client.IndexAsyncio = MagicMock(return_value=mock_index)
    client.create_index = AsyncMock()
    client.delete_index = AsyncMock()
    client.list_indexes = AsyncMock(return_value=[])
    client.close = AsyncMock()
    # Inference API
    client.inference = AsyncMock()
    client.inference.rerank = AsyncMock()
    return client


@pytest.fixture
async def store(mock_client, mock_index):
    """A connected PineconeVectorStore with mocked SDK."""
    with patch("grover.search.stores.pinecone.PineconeAsyncio", return_value=mock_client):
        s = PineconeVectorStore(index_name="test-index", api_key="fake-key")
        await s.connect()
        yield s
        await s.close()


# ==================================================================
# Connect / Close
# ==================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_creates_client_and_index(self, mock_client, mock_index):
        with patch("grover.search.stores.pinecone.PineconeAsyncio", return_value=mock_client):
            s = PineconeVectorStore(index_name="test-index", api_key="fake-key")
            await s.connect()
            mock_client.describe_index.assert_called_once_with("test-index")
            mock_client.IndexAsyncio.assert_called_once_with(host="test-host.pinecone.io")
            await s.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, store, mock_index, mock_client):
        await store.close()
        mock_index.close.assert_called()

    @pytest.mark.asyncio
    async def test_operations_before_connect_raise(self):
        s = PineconeVectorStore(index_name="x", api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            await s.upsert([VectorEntry(id="a", vector=[0.1], metadata={})])


# ==================================================================
# Upsert
# ==================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store, mock_index):
        mock_index.upsert.return_value = SimpleNamespace(upserted_count=1)
        entry = VectorEntry(id="/a.py", vector=[0.1, 0.2], metadata={"key": "val"})
        result = await store.upsert([entry])
        assert result.upserted_count == 1
        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args
        vectors = call_kwargs.kwargs["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["id"] == "/a.py"
        assert vectors[0]["values"] == [0.1, 0.2]
        assert vectors[0]["metadata"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_upsert_with_namespace(self, store, mock_index):
        mock_index.upsert.return_value = SimpleNamespace(upserted_count=1)
        entry = VectorEntry(id="/a.py", vector=[0.1], metadata={})
        await store.upsert([entry], namespace="test-ns")
        call_kwargs = mock_index.upsert.call_args
        assert call_kwargs.kwargs["namespace"] == "test-ns"

    @pytest.mark.asyncio
    async def test_upsert_batch_chunks_at_1000(self, store, mock_index):
        mock_index.upsert.return_value = SimpleNamespace(upserted_count=1000)
        entries = [VectorEntry(id=f"/f{i}.py", vector=[0.1], metadata={}) for i in range(2500)]
        result = await store.upsert(entries)
        # Should be called 3 times: 1000, 1000, 500
        assert mock_index.upsert.call_count == 3
        assert result.upserted_count == 3000  # 1000 * 3 (mocked return)

    @pytest.mark.asyncio
    async def test_upsert_default_namespace(self, mock_client, mock_index):
        with patch("grover.search.stores.pinecone.PineconeAsyncio", return_value=mock_client):
            s = PineconeVectorStore(index_name="test-index", api_key="k", namespace="default-ns")
            await s.connect()
            mock_index.upsert.return_value = SimpleNamespace(upserted_count=1)
            await s.upsert([VectorEntry(id="a", vector=[0.1], metadata={})])
            assert mock_index.upsert.call_args.kwargs["namespace"] == "default-ns"
            await s.close()


# ==================================================================
# Search
# ==================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_basic(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(
            matches=[
                _make_match("/a.py", 0.95, {"content": "hello"}, [0.1, 0.2]),
                _make_match("/b.py", 0.80, {"content": "world"}, [0.3, 0.4]),
            ]
        )
        results = await store.search([0.1, 0.2], k=5)
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert results[0].id == "/a.py"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_with_namespace(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        await store.search([0.1], k=5, namespace="my-ns")
        assert mock_index.query.call_args.kwargs["namespace"] == "my-ns"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        await store.search([0.1], k=5, filter=eq("lang", "python"))
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == {"lang": {"$eq": "python"}}

    @pytest.mark.asyncio
    async def test_search_with_complex_filter(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        f = and_(eq("lang", "python"), gt("year", 2020))
        await store.search([0.1], k=5, filter=f)
        call_kwargs = mock_index.query.call_args.kwargs
        expected = {"$and": [{"lang": {"$eq": "python"}}, {"year": {"$gt": 2020}}]}
        assert call_kwargs["filter"] == expected

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(
            matches=[
                _make_match("/a.py", 0.95),
                _make_match("/b.py", 0.30),
            ]
        )
        results = await store.search([0.1], k=5, score_threshold=0.5)
        assert len(results) == 1
        assert results[0].id == "/a.py"

    @pytest.mark.asyncio
    async def test_search_include_metadata_false(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        await store.search([0.1], k=5, include_metadata=False)
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["include_metadata"] is False
        assert call_kwargs["include_values"] is False


# ==================================================================
# Delete
# ==================================================================


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_by_ids(self, store, mock_index):
        result = await store.delete(["/a.py", "/b.py"])
        assert result.deleted_count == 2
        mock_index.delete.assert_called_once_with(ids=["/a.py", "/b.py"], namespace="")

    @pytest.mark.asyncio
    async def test_delete_with_namespace(self, store, mock_index):
        await store.delete(["/a.py"], namespace="ns1")
        mock_index.delete.assert_called_once_with(ids=["/a.py"], namespace="ns1")


# ==================================================================
# Fetch
# ==================================================================


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_existing(self, store, mock_index):
        mock_index.fetch.return_value = SimpleNamespace(
            vectors={
                "/a.py": _make_fetch_vector("/a.py", [0.1, 0.2], {"content": "hello"}),
            }
        )
        results = await store.fetch(["/a.py"])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == "/a.py"
        assert results[0].vector == [0.1, 0.2]
        assert results[0].metadata == {"content": "hello"}

    @pytest.mark.asyncio
    async def test_fetch_missing(self, store, mock_index):
        mock_index.fetch.return_value = SimpleNamespace(vectors={})
        results = await store.fetch(["/missing.py"])
        assert results == [None]

    @pytest.mark.asyncio
    async def test_fetch_mixed(self, store, mock_index):
        mock_index.fetch.return_value = SimpleNamespace(
            vectors={
                "/a.py": _make_fetch_vector("/a.py", [0.1], {}),
            }
        )
        results = await store.fetch(["/a.py", "/missing.py"])
        assert results[0] is not None
        assert results[1] is None


# ==================================================================
# Namespaces
# ==================================================================


class TestNamespaces:
    @pytest.mark.asyncio
    async def test_list_namespaces(self, store, mock_index):
        page = SimpleNamespace(
            namespaces=[SimpleNamespace(name="ns1"), SimpleNamespace(name="ns2")]
        )
        # list_namespaces() returns an async iterator directly, not a coroutine
        store._index.list_namespaces = MagicMock(return_value=_async_iter([page]))
        results = await store.list_namespaces()
        assert results == ["ns1", "ns2"]

    @pytest.mark.asyncio
    async def test_delete_namespace(self, store, mock_index):
        await store.delete_namespace("old-ns")
        mock_index.delete_namespace.assert_called_once_with("old-ns")


# ==================================================================
# Index Lifecycle
# ==================================================================


class TestIndexLifecycle:
    @pytest.mark.asyncio
    async def test_create_index_serverless(self, store, mock_client):
        config = IndexConfig(
            name="new-idx",
            dimension=128,
            metric="cosine",
            cloud_config={"cloud": "aws", "region": "us-east-1"},
        )
        await store.create_index(config)
        mock_client.create_index.assert_called_once()
        call_kwargs = mock_client.create_index.call_args.kwargs
        assert call_kwargs["name"] == "new-idx"
        assert call_kwargs["dimension"] == 128
        assert call_kwargs["metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_create_index_default_spec(self, store, mock_client):
        """When cloud_config lacks cloud/region, falls back to default ServerlessSpec."""
        config = IndexConfig(name="idx", dimension=64, metric="cosine", cloud_config={})
        await store.create_index(config)
        mock_client.create_index.assert_called_once()
        call_kwargs = mock_client.create_index.call_args.kwargs
        assert call_kwargs["name"] == "idx"
        # Spec should be the default ServerlessSpec(cloud="aws", region="us-east-1")
        assert call_kwargs["spec"] is not None

    @pytest.mark.asyncio
    async def test_delete_index(self, store, mock_client):
        await store.delete_index("old-idx")
        mock_client.delete_index.assert_called_once_with("old-idx")

    @pytest.mark.asyncio
    async def test_list_indexes(self, store, mock_client):
        mock_client.list_indexes.return_value = [
            _make_index_model("idx1", 128, "cosine"),
            _make_index_model("idx2", 256, "euclidean"),
        ]
        results = await store.list_indexes()
        assert len(results) == 2
        assert results[0].name == "idx1"
        assert results[0].dimension == 128
        assert results[0].metric == "cosine"
        assert results[0].vector_count == 42


# ==================================================================
# Hybrid Search
# ==================================================================


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(
            matches=[_make_match("/a.py", 0.9, {"content": "x"}, [0.1])]
        )
        results = await store.hybrid_search(dense_vector=[0.1, 0.2], k=5)
        assert len(results) == 1
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["vector"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_hybrid_search_sparse_dense(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(
            matches=[_make_match("/a.py", 0.85)]
        )
        sv = SparseVector(indices=[0, 5, 10], values=[0.1, 0.5, 0.3])
        results = await store.hybrid_search(
            dense_vector=[0.1], sparse_vector=sv, k=5
        )
        assert len(results) == 1
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["sparse_vector"] == {"indices": [0, 5, 10], "values": [0.1, 0.5, 0.3]}

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filter(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        await store.hybrid_search(dense_vector=[0.1], k=5, filter=eq("type", "code"))
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == {"type": {"$eq": "code"}}


# ==================================================================
# Reranking
# ==================================================================


class TestReranking:
    @pytest.mark.asyncio
    async def test_reranked_search(self, store, mock_index, mock_client):
        # Initial search results
        mock_index.query.return_value = SimpleNamespace(
            matches=[
                _make_match("/a.py", 0.90, {"content": "alpha"}),
                _make_match("/b.py", 0.85, {"content": "beta"}),
            ]
        )
        # Rerank response
        mock_client.inference.rerank.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, score=0.95),  # /b.py reranked higher
                SimpleNamespace(index=0, score=0.70),  # /a.py reranked lower
            ]
        )
        results = await store.reranked_search(
            [0.1], "beta query", k=5, rerank_model="test-model", rerank_top_n=2
        )
        assert len(results) == 2
        assert results[0].id == "/b.py"
        assert results[0].score == 0.95
        assert results[1].id == "/a.py"
        assert results[1].score == 0.70

        mock_client.inference.rerank.assert_called_once()
        call_kwargs = mock_client.inference.rerank.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["top_n"] == 2

    @pytest.mark.asyncio
    async def test_reranked_search_empty_results(self, store, mock_index):
        mock_index.query.return_value = SimpleNamespace(matches=[])
        results = await store.reranked_search([0.1], "query", k=5)
        assert results == []


# ==================================================================
# Properties
# ==================================================================


class TestProperties:
    def test_index_name(self):
        s = PineconeVectorStore(index_name="my-index", api_key="k")
        assert s.index_name == "my-index"


# ==================================================================
# Import guard
# ==================================================================


class TestImportGuard:
    def test_import_guard_message(self):
        with (
            patch.dict("sys.modules", {"pinecone": None}),
            patch("grover.search.stores.pinecone._HAS_PINECONE", False),
            pytest.raises(ImportError, match="pinecone is required"),
        ):
            PineconeVectorStore(index_name="x", api_key="k")


# ==================================================================
# isinstance checks
# ==================================================================


class TestProtocolConformance:
    def test_satisfies_vector_store(self):
        from grover.search.protocols import VectorStore

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, VectorStore)

    def test_satisfies_supports_namespaces(self):
        from grover.search.protocols import SupportsNamespaces

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, SupportsNamespaces)

    def test_satisfies_supports_metadata_filter(self):
        from grover.search.protocols import SupportsMetadataFilter

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, SupportsMetadataFilter)

    def test_satisfies_supports_index_lifecycle(self):
        from grover.search.protocols import SupportsIndexLifecycle

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, SupportsIndexLifecycle)

    def test_satisfies_supports_hybrid_search(self):
        from grover.search.protocols import SupportsHybridSearch

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, SupportsHybridSearch)

    def test_satisfies_supports_reranking(self):
        from grover.search.protocols import SupportsReranking

        s = PineconeVectorStore(index_name="x", api_key="k")
        assert isinstance(s, SupportsReranking)


# ==================================================================
# Compile filter
# ==================================================================


class TestCompileFilter:
    def test_compile_eq(self):
        s = PineconeVectorStore(index_name="x", api_key="k")
        result = s.compile_filter(eq("color", "red"))
        assert result == {"color": {"$eq": "red"}}

    def test_compile_and(self):
        s = PineconeVectorStore(index_name="x", api_key="k")
        result = s.compile_filter(and_(eq("a", 1), gt("b", 2)))
        assert result == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}


# ------------------------------------------------------------------
# Async iterator helper for mocking
# ------------------------------------------------------------------


class _AsyncIterator:
    """Helper to create an async iterator from a list."""

    def __init__(self, items: list) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _async_iter(items: list) -> _AsyncIterator:
    return _AsyncIterator(items)

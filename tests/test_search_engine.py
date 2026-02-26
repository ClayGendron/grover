"""Tests for SearchEngine — orchestrator wiring EmbeddingProvider + VectorStore."""

from __future__ import annotations

import hashlib
import math

import pytest

from grover.ref import Ref
from grover.search._engine import SearchEngine, _content_hash
from grover.search.extractors import EmbeddableChunk
from grover.search.stores.local import LocalVectorStore
from grover.search.types import SearchResult

# ------------------------------------------------------------------
# Fake provider (sync — tests the sync/async bridge)
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic sync embedding provider for testing."""

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


class AsyncFakeProvider:
    """Deterministic async embedding provider for testing."""

    async def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "async-fake"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def engine() -> SearchEngine:
    """Engine with sync FakeProvider + LocalVectorStore."""
    store = LocalVectorStore(dimension=_FAKE_DIM)
    return SearchEngine(vector=store, embedding=FakeProvider())


@pytest.fixture
def async_engine() -> SearchEngine:
    """Engine with async FakeProvider + LocalVectorStore."""
    store = LocalVectorStore(dimension=_FAKE_DIM)
    return SearchEngine(vector=store, embedding=AsyncFakeProvider())


@pytest.fixture
def engine_no_provider() -> SearchEngine:
    """Engine with no embedding provider."""
    store = LocalVectorStore(dimension=_FAKE_DIM)
    return SearchEngine(vector=store)


# ==================================================================
# Add and Search
# ==================================================================


class TestAddAndSearch:
    @pytest.mark.asyncio
    async def test_add_and_search(self, engine: SearchEngine):
        await engine.add("/a.py", "def foo(): pass")
        await engine.add("/b.py", "class Bar: pass")
        results = await engine.search("def foo(): pass")
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].ref.path == "/a.py"

    @pytest.mark.asyncio
    async def test_add_with_parent_path(self, engine: SearchEngine):
        await engine.add("/chunk.txt", "def foo(): pass", parent_path="/a.py")
        results = await engine.search("def foo(): pass")
        assert results[0].parent_path == "/a.py"

    @pytest.mark.asyncio
    async def test_add_batch(self, engine: SearchEngine):
        entries = [
            EmbeddableChunk(path="/a.py", content="def foo(): pass"),
            EmbeddableChunk(path="/b.py", content="class Bar: pass"),
            EmbeddableChunk(path="/c.py", content="x = 42", parent_path="/pkg.py"),
        ]
        await engine.add_batch(entries)
        assert len(engine) == 3
        assert engine.has("/a.py")
        assert engine.has("/b.py")
        assert engine.has("/c.py")

    @pytest.mark.asyncio
    async def test_add_batch_empty(self, engine: SearchEngine):
        await engine.add_batch([])
        assert len(engine) == 0

    @pytest.mark.asyncio
    async def test_search_scores_sorted(self, engine: SearchEngine):
        await engine.add("/a.py", "alpha")
        await engine.add("/b.py", "beta")
        await engine.add("/c.py", "gamma")
        results = await engine.search("alpha", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_exact_match(self, engine: SearchEngine):
        await engine.add("/a.py", "unique content here")
        await engine.add("/b.py", "other stuff")
        results = await engine.search("unique content here")
        assert results[0].ref.path == "/a.py"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_search_k_limits(self, engine: SearchEngine):
        for i in range(10):
            await engine.add(f"/f{i}.py", f"content {i}")
        results = await engine.search("content 5", k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_result_ref(self, engine: SearchEngine):
        await engine.add("/a.py", "hello")
        results = await engine.search("hello")
        assert isinstance(results[0].ref, Ref)
        assert results[0].ref.path == "/a.py"

    @pytest.mark.asyncio
    async def test_search_result_content(self, engine: SearchEngine):
        await engine.add("/a.py", "hello world")
        results = await engine.search("hello world")
        assert results[0].content == "hello world"


# ==================================================================
# Async provider
# ==================================================================


class TestAsyncProvider:
    @pytest.mark.asyncio
    async def test_add_and_search_async_provider(self, async_engine: SearchEngine):
        await async_engine.add("/a.py", "async content")
        results = await async_engine.search("async content")
        assert len(results) >= 1
        assert results[0].ref.path == "/a.py"

    @pytest.mark.asyncio
    async def test_add_batch_async_provider(self, async_engine: SearchEngine):
        entries = [
            EmbeddableChunk(path="/a.py", content="alpha"),
            EmbeddableChunk(path="/b.py", content="beta"),
        ]
        await async_engine.add_batch(entries)
        assert len(async_engine) == 2


# ==================================================================
# Remove
# ==================================================================


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove(self, engine: SearchEngine):
        await engine.add("/a.py", "content")
        await engine.add("/b.py", "other")
        await engine.remove("/a.py")
        assert not engine.has("/a.py")
        assert engine.has("/b.py")

    @pytest.mark.asyncio
    async def test_remove_file(self, engine: SearchEngine):
        await engine.add("/a.py", "file content")
        await engine.add("/chunk1", "def foo(): pass", parent_path="/a.py")
        await engine.add("/chunk2", "def bar(): pass", parent_path="/a.py")
        await engine.add("/b.py", "other file")
        assert len(engine) == 4

        await engine.remove_file("/a.py")
        assert len(engine) == 1
        assert not engine.has("/a.py")
        assert not engine.has("/chunk1")
        assert not engine.has("/chunk2")
        assert engine.has("/b.py")

    @pytest.mark.asyncio
    async def test_remove_file_nonexistent(self, engine: SearchEngine):
        await engine.remove_file("/ghost.py")  # should not raise


# ==================================================================
# Passthrough helpers
# ==================================================================


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_has(self, engine: SearchEngine):
        assert not engine.has("/a.py")
        await engine.add("/a.py", "hello")
        assert engine.has("/a.py")

    @pytest.mark.asyncio
    async def test_content_hash(self, engine: SearchEngine):
        await engine.add("/a.py", "hello")
        assert engine.content_hash("/a.py") == _content_hash("hello")

    @pytest.mark.asyncio
    async def test_content_hash_missing(self, engine: SearchEngine):
        assert engine.content_hash("/missing.py") is None

    @pytest.mark.asyncio
    async def test_len(self, engine: SearchEngine):
        assert len(engine) == 0
        await engine.add("/a.py", "content")
        assert len(engine) == 1


# ==================================================================
# No provider
# ==================================================================


class TestNoProvider:
    @pytest.mark.asyncio
    async def test_add_skips_without_provider(self, engine_no_provider: SearchEngine):
        # add() silently skips vector indexing when no embedding provider
        await engine_no_provider.add("/a.py", "content")

    @pytest.mark.asyncio
    async def test_add_batch_skips_without_provider(self, engine_no_provider: SearchEngine):
        entries = [EmbeddableChunk(path="/a.py", content="c")]
        # add_batch() silently skips vector indexing when no embedding provider
        await engine_no_provider.add_batch(entries)

    @pytest.mark.asyncio
    async def test_search_raises(self, engine_no_provider: SearchEngine):
        with pytest.raises(RuntimeError, match="no embedding provider"):
            await engine_no_provider.search("query")


# ==================================================================
# Persistence
# ==================================================================


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_load(self, engine: SearchEngine, tmp_path):
        await engine.add("/a.py", "hello")
        await engine.add("/b.py", "world")

        save_dir = str(tmp_path / "index")
        engine.save(save_dir)

        engine2 = SearchEngine(
            vector=LocalVectorStore(dimension=_FAKE_DIM), embedding=FakeProvider()
        )
        engine2.load(save_dir)

        assert len(engine2) == 2
        assert engine2.has("/a.py")
        assert engine2.has("/b.py")

    @pytest.mark.asyncio
    async def test_search_after_load(self, engine: SearchEngine, tmp_path):
        await engine.add("/a.py", "unique query text")
        await engine.add("/b.py", "other stuff")

        save_dir = str(tmp_path / "index")
        engine.save(save_dir)

        engine2 = SearchEngine(
            vector=LocalVectorStore(dimension=_FAKE_DIM), embedding=FakeProvider()
        )
        engine2.load(save_dir)
        results = await engine2.search("unique query text")
        assert results[0].ref.path == "/a.py"


# ==================================================================
# Lifecycle
# ==================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect(self, engine: SearchEngine):
        await engine.connect()  # should not raise

    @pytest.mark.asyncio
    async def test_close(self, engine: SearchEngine):
        await engine.close()  # should not raise


# ==================================================================
# Properties
# ==================================================================


class TestProperties:
    def test_vector_property(self, engine: SearchEngine):
        assert isinstance(engine.vector, LocalVectorStore)

    def test_embedding_property(self, engine: SearchEngine):
        assert isinstance(engine.embedding, FakeProvider)

    def test_embedding_none(self, engine_no_provider: SearchEngine):
        assert engine_no_provider.embedding is None


# ==================================================================
# Dimension validation
# ==================================================================


class TestDimensionValidation:
    def test_matching_dimensions_accepted(self):
        store = LocalVectorStore(dimension=_FAKE_DIM)
        engine = SearchEngine(vector=store, embedding=FakeProvider())
        assert engine.vector is store

    def test_mismatched_dimensions_rejected(self):
        store = LocalVectorStore(dimension=64)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            SearchEngine(vector=store, embedding=FakeProvider())

    def test_store_without_dimension_skips_check(self):
        class NoDimStore:
            """Mock store without a dimension attribute."""

            async def upsert(self, entries, *, namespace=None):
                pass

            async def search(
                self,
                vector,
                *,
                k=10,
                namespace=None,
                filter=None,  # noqa: A002
                include_metadata=True,
                score_threshold=None,
            ):
                return []

            async def delete(self, ids, *, namespace=None):
                pass

            async def fetch(self, ids, *, namespace=None):
                return []

            async def connect(self):
                pass

            async def close(self):
                pass

            @property
            def index_name(self):
                return "no-dim"

        engine = SearchEngine(vector=NoDimStore(), embedding=FakeProvider())
        assert engine.vector is not None

    def test_no_embedding_skips_check(self):
        store = LocalVectorStore(dimension=64)
        engine = SearchEngine(vector=store)
        assert engine.embedding is None

    def test_no_vector_skips_check(self):
        engine = SearchEngine(embedding=FakeProvider())
        assert engine.vector is None


# ==================================================================
# Model name validation
# ==================================================================


class TestModelNameValidation:
    def test_matching_model_name_accepted(self):
        store = LocalVectorStore(dimension=_FAKE_DIM)
        engine = SearchEngine(vector=store, embedding=FakeProvider(), model_name="fake")
        assert engine.model_name == "fake"

    def test_mismatched_model_name_rejected(self):
        store = LocalVectorStore(dimension=_FAKE_DIM)
        with pytest.raises(ValueError, match="Model mismatch"):
            SearchEngine(vector=store, embedding=FakeProvider(), model_name="wrong")

    def test_model_name_without_embedding_stored(self):
        engine = SearchEngine(model_name="x")
        assert engine.model_name == "x"

    def test_model_name_none_by_default(self):
        engine = SearchEngine()
        assert engine.model_name is None

    def test_model_name_property(self):
        store = LocalVectorStore(dimension=_FAKE_DIM)
        engine = SearchEngine(vector=store, embedding=FakeProvider(), model_name="fake")
        assert engine.model_name == "fake"

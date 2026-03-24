"""Tests for LocalVectorStore — in-process usearch HNSW vector store."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileSearchResult
from grover.providers.search.filters import eq
from grover.providers.search.local import LocalVectorStore

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_DIM = 32


def _hash_vector(text: str) -> list[float]:
    """Deterministic unit vector from text hash."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [float(b) for b in h]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _make_file(entry_id: str, content: str) -> File:
    """Build a File with a deterministic embedding vector."""
    return File(path=entry_id, embedding=_hash_vector(content))


@pytest.fixture
def store() -> LocalVectorStore:
    return LocalVectorStore(dimension=_DIM)


# ==================================================================
# Upsert
# ==================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store: LocalVectorStore):
        result = await store.upsert(files=[_make_file("/a.py", "hello")])
        assert isinstance(result, BatchResult)
        assert result.succeeded == 1
        assert len(store) == 1
        assert store.has("/a.py")

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store: LocalVectorStore):
        files = [_make_file(f"/f{i}.py", f"content {i}") for i in range(5)]
        result = await store.upsert(files=files)
        assert result.succeeded == 5
        assert len(store) == 5

    @pytest.mark.asyncio
    async def test_upsert_deduplicates(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "version 1")])
        await store.upsert(files=[_make_file("/a.py", "version 2")])
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_upsert_tracks_parent_children(self, store: LocalVectorStore):
        parent = _make_file("/a.py", "file")
        child1 = _make_file("/a.py#chunk1", "c1")
        child2 = _make_file("/a.py#chunk2", "c2")
        await store.upsert(files=[parent, child1, child2])
        assert len(store) == 3

        store.remove_file("/a.py")
        assert len(store) == 0


# ==================================================================
# Vector Search
# ==================================================================


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "alpha"), _make_file("/b.py", "beta")])
        result = await store.vector_search(_hash_vector("alpha"), k=2)
        assert isinstance(result, FileSearchResult)
        assert result.success
        assert len(result.files) >= 1

    @pytest.mark.asyncio
    async def test_search_exact_match(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "exact match"), _make_file("/b.py", "other stuff")])
        result = await store.vector_search(_hash_vector("exact match"), k=2)
        assert result.files[0].path == "/a.py"

    @pytest.mark.asyncio
    async def test_search_empty_store(self, store: LocalVectorStore):
        result = await store.vector_search(_hash_vector("anything"), k=5)
        assert result.success
        assert len(result.files) == 0

    @pytest.mark.asyncio
    async def test_search_k_limits_results(self, store: LocalVectorStore):
        files = [_make_file(f"/f{i}.py", f"content {i}") for i in range(10)]
        await store.upsert(files=files)
        result = await store.vector_search(_hash_vector("content 5"), k=3)
        assert len(result.files) <= 3

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "match"), _make_file("/b.py", "totally different")])
        result = await store.vector_search(_hash_vector("match"), k=10, score_threshold=0.99)
        # Only the exact match should survive
        assert len(result.files) == 1
        assert result.files[0].path == "/a.py"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store: LocalVectorStore):
        # Filter test: add metadata via internal store
        f1 = _make_file("/a.py", "alpha")
        f2 = _make_file("/b.go", "beta")
        await store.upsert(files=[f1, f2])
        # Manually inject metadata for filter testing
        for meta in store._key_to_meta.values():
            if meta["id"] == "/a.py":
                meta["lang"] = "python"
            elif meta["id"] == "/b.go":
                meta["lang"] = "go"
        result = await store.vector_search(_hash_vector("alpha"), k=10, filter=eq("lang", "python"))
        assert len(result.files) == 1
        assert result.files[0].path == "/a.py"

    @pytest.mark.asyncio
    async def test_search_with_candidates(self, store: LocalVectorStore):
        from grover.models.internal.results import FileSearchSet

        await store.upsert(files=[_make_file("/a.py", "alpha"), _make_file("/b.py", "beta")])
        candidates = FileSearchSet.from_paths(["/a.py"])
        result = await store.vector_search(_hash_vector("beta"), k=2, candidates=candidates)
        # Only /a.py should be in results since candidates restrict it
        for f in result.files:
            assert f.path == "/a.py"

    @pytest.mark.asyncio
    async def test_search_groups_chunks(self, store: LocalVectorStore):
        """Chunk entries get grouped under parent path."""
        parent = _make_file("/a.py", "file content")
        chunk = _make_file("/a.py#foo", "chunk content")
        await store.upsert(files=[parent, chunk])
        result = await store.vector_search(_hash_vector("chunk content"), k=5)
        # Results should be grouped by parent path /a.py
        paths = [f.path for f in result.files]
        assert "/a.py" in paths


# ==================================================================
# Delete
# ==================================================================


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_by_id(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "hello"), _make_file("/b.py", "world")])
        result = await store.delete(files=["/a.py"])
        assert isinstance(result, BatchResult)
        assert result.succeeded == 1
        assert len(store) == 1
        assert not store.has("/a.py")
        assert store.has("/b.py")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: LocalVectorStore):
        result = await store.delete(files=["/ghost.py"])
        assert result.succeeded == 0

    @pytest.mark.asyncio
    async def test_delete_multiple(self, store: LocalVectorStore):
        files = [_make_file(f"/f{i}.py", f"c{i}") for i in range(5)]
        await store.upsert(files=files)
        result = await store.delete(files=["/f0.py", "/f2.py", "/f4.py"])
        assert result.succeeded == 3
        assert len(store) == 2


# ==================================================================
# Local-specific methods
# ==================================================================


class TestLocalMethods:
    @pytest.mark.asyncio
    async def test_has(self, store: LocalVectorStore):
        assert not store.has("/a.py")
        await store.upsert(files=[_make_file("/a.py", "hello")])
        assert store.has("/a.py")

    @pytest.mark.asyncio
    async def test_remove_file(self, store: LocalVectorStore):
        parent = _make_file("/a.py", "file")
        child1 = _make_file("/a.py#chunk1", "c1")
        child2 = _make_file("/a.py#chunk2", "c2")
        other = _make_file("/b.py", "other")
        await store.upsert(files=[parent, child1, child2, other])
        assert len(store) == 4

        store.remove_file("/a.py")
        assert len(store) == 1
        assert not store.has("/a.py")
        assert not store.has("/a.py#chunk1")
        assert not store.has("/a.py#chunk2")
        assert store.has("/b.py")

    @pytest.mark.asyncio
    async def test_remove_file_nonexistent(self, store: LocalVectorStore):
        store.remove_file("/ghost.py")  # should not raise
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_remove_file_without_children(self, store: LocalVectorStore):
        await store.upsert(files=[_make_file("/a.py", "hello")])
        store.remove_file("/a.py")
        assert len(store) == 0

    def test_len_empty(self, store: LocalVectorStore):
        assert len(store) == 0

    def test_index_name(self, store: LocalVectorStore):
        assert store.index_name == "local"

    def test_dimension_property_returns_constructor_value(self):
        store = LocalVectorStore(dimension=128)
        assert store.dimension == 128


# ==================================================================
# Lifecycle
# ==================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_noop(self, store: LocalVectorStore):
        await store.connect()  # should not raise

    @pytest.mark.asyncio
    async def test_close_noop(self, store: LocalVectorStore):
        await store.close()  # should not raise


# ==================================================================
# Persistence
# ==================================================================


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, store: LocalVectorStore, tmp_path: Path):
        files = [
            _make_file("/a.py", "hello"),
            _make_file("/b.py", "world"),
            _make_file("/a.py#chunk", "piece"),
        ]
        await store.upsert(files=files)

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)

        assert len(store2) == 3
        assert store2.has("/a.py")
        assert store2.has("/b.py")
        assert store2.has("/a.py#chunk")

    @pytest.mark.asyncio
    async def test_search_after_load(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert(files=[_make_file("/a.py", "unique content"), _make_file("/b.py", "other")])

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)
        result = await store2.vector_search(_hash_vector("unique content"), k=2)
        assert result.files[0].path == "/a.py"

    @pytest.mark.asyncio
    async def test_save_creates_files(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert(files=[_make_file("/a.py", "content")])
        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        assert (Path(save_dir) / "search.usearch").exists()
        assert (Path(save_dir) / "search_meta.json").exists()

    @pytest.mark.asyncio
    async def test_parent_tracking_after_load(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert(files=[_make_file("/a.py", "file"), _make_file("/a.py#chunk", "c1")])

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)
        store2.remove_file("/a.py")
        assert len(store2) == 0

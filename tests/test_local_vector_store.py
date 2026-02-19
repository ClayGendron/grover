"""Tests for LocalVectorStore — in-process usearch HNSW vector store."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from grover.search.filters import eq
from grover.search.stores.local import LocalVectorStore
from grover.search.types import DeleteResult, UpsertResult, VectorEntry, VectorSearchResult

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


def _make_entry(entry_id: str, content: str, parent_path: str | None = None) -> VectorEntry:
    """Build a VectorEntry with a deterministic vector."""
    meta: dict = {"content": content, "content_hash": hashlib.sha256(content.encode()).hexdigest()}
    if parent_path is not None:
        meta["parent_path"] = parent_path
    return VectorEntry(id=entry_id, vector=_hash_vector(content), metadata=meta)


@pytest.fixture
def store() -> LocalVectorStore:
    return LocalVectorStore(dimension=_DIM)


# ==================================================================
# Upsert
# ==================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store: LocalVectorStore):
        entry = _make_entry("/a.py", "hello")
        result = await store.upsert([entry])
        assert isinstance(result, UpsertResult)
        assert result.upserted_count == 1
        assert len(store) == 1
        assert store.has("/a.py")

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store: LocalVectorStore):
        entries = [_make_entry(f"/f{i}.py", f"content {i}") for i in range(5)]
        result = await store.upsert(entries)
        assert result.upserted_count == 5
        assert len(store) == 5

    @pytest.mark.asyncio
    async def test_upsert_deduplicates(self, store: LocalVectorStore):
        e1 = _make_entry("/a.py", "version 1")
        e2 = _make_entry("/a.py", "version 2")
        await store.upsert([e1])
        await store.upsert([e2])
        assert len(store) == 1
        assert store.content_hash("/a.py") == hashlib.sha256(b"version 2").hexdigest()

    @pytest.mark.asyncio
    async def test_upsert_tracks_parent_children(self, store: LocalVectorStore):
        parent = _make_entry("/a.py", "file")
        child1 = _make_entry("/chunk1", "c1", parent_path="/a.py")
        child2 = _make_entry("/chunk2", "c2", parent_path="/a.py")
        await store.upsert([parent, child1, child2])
        assert len(store) == 3

        store.remove_file("/a.py")
        assert len(store) == 0


# ==================================================================
# Search
# ==================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, store: LocalVectorStore):
        entries = [_make_entry("/a.py", "alpha"), _make_entry("/b.py", "beta")]
        await store.upsert(entries)
        results = await store.search(_hash_vector("alpha"), k=2)
        assert len(results) >= 1
        assert all(isinstance(r, VectorSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_scores_sorted(self, store: LocalVectorStore):
        entries = [_make_entry(f"/f{i}.py", f"text {i}") for i in range(5)]
        await store.upsert(entries)
        results = await store.search(_hash_vector("text 0"), k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_exact_match(self, store: LocalVectorStore):
        entries = [_make_entry("/a.py", "exact match"), _make_entry("/b.py", "other stuff")]
        await store.upsert(entries)
        results = await store.search(_hash_vector("exact match"), k=2)
        assert results[0].id == "/a.py"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_search_empty_store(self, store: LocalVectorStore):
        results = await store.search(_hash_vector("anything"), k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_k_limits_results(self, store: LocalVectorStore):
        entries = [_make_entry(f"/f{i}.py", f"content {i}") for i in range(10)]
        await store.upsert(entries)
        results = await store.search(_hash_vector("content 5"), k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, store: LocalVectorStore):
        entries = [_make_entry("/a.py", "match"), _make_entry("/b.py", "totally different")]
        await store.upsert(entries)
        results = await store.search(_hash_vector("match"), k=10, score_threshold=0.99)
        # Only the exact match should survive
        assert len(results) == 1
        assert results[0].id == "/a.py"

    @pytest.mark.asyncio
    async def test_search_include_metadata_false(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello")])
        results = await store.search(_hash_vector("hello"), k=1, include_metadata=False)
        assert results[0].metadata == {}
        assert results[0].vector is None

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store: LocalVectorStore):
        e1 = _make_entry("/a.py", "alpha")
        e1_meta = dict(e1.metadata)
        e1_meta["lang"] = "python"
        e1_with = VectorEntry(id=e1.id, vector=e1.vector, metadata=e1_meta)

        e2 = _make_entry("/b.go", "beta")
        e2_meta = dict(e2.metadata)
        e2_meta["lang"] = "go"
        e2_with = VectorEntry(id=e2.id, vector=e2.vector, metadata=e2_meta)

        await store.upsert([e1_with, e2_with])
        results = await store.search(_hash_vector("alpha"), k=10, filter=eq("lang", "python"))
        assert len(results) == 1
        assert results[0].id == "/a.py"


# ==================================================================
# Delete
# ==================================================================


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_by_id(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello"), _make_entry("/b.py", "world")])
        result = await store.delete(["/a.py"])
        assert isinstance(result, DeleteResult)
        assert result.deleted_count == 1
        assert len(store) == 1
        assert not store.has("/a.py")
        assert store.has("/b.py")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: LocalVectorStore):
        result = await store.delete(["/ghost.py"])
        assert result.deleted_count == 0

    @pytest.mark.asyncio
    async def test_delete_multiple(self, store: LocalVectorStore):
        entries = [_make_entry(f"/f{i}.py", f"c{i}") for i in range(5)]
        await store.upsert(entries)
        result = await store.delete(["/f0.py", "/f2.py", "/f4.py"])
        assert result.deleted_count == 3
        assert len(store) == 2


# ==================================================================
# Fetch
# ==================================================================


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_existing(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello")])
        results = await store.fetch(["/a.py"])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == "/a.py"
        assert "content" in results[0].metadata

    @pytest.mark.asyncio
    async def test_fetch_missing(self, store: LocalVectorStore):
        results = await store.fetch(["/missing.py"])
        assert results == [None]

    @pytest.mark.asyncio
    async def test_fetch_mixed(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello")])
        results = await store.fetch(["/a.py", "/missing.py"])
        assert results[0] is not None
        assert results[1] is None


# ==================================================================
# Local-specific methods
# ==================================================================


class TestLocalMethods:
    @pytest.mark.asyncio
    async def test_has(self, store: LocalVectorStore):
        assert not store.has("/a.py")
        await store.upsert([_make_entry("/a.py", "hello")])
        assert store.has("/a.py")

    @pytest.mark.asyncio
    async def test_content_hash(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello")])
        assert store.content_hash("/a.py") == hashlib.sha256(b"hello").hexdigest()

    @pytest.mark.asyncio
    async def test_content_hash_missing(self, store: LocalVectorStore):
        assert store.content_hash("/missing.py") is None

    @pytest.mark.asyncio
    async def test_remove_file(self, store: LocalVectorStore):
        parent = _make_entry("/a.py", "file")
        child1 = _make_entry("/chunk1", "c1", parent_path="/a.py")
        child2 = _make_entry("/chunk2", "c2", parent_path="/a.py")
        other = _make_entry("/b.py", "other")
        await store.upsert([parent, child1, child2, other])
        assert len(store) == 4

        store.remove_file("/a.py")
        assert len(store) == 1
        assert not store.has("/a.py")
        assert not store.has("/chunk1")
        assert not store.has("/chunk2")
        assert store.has("/b.py")

    @pytest.mark.asyncio
    async def test_remove_file_nonexistent(self, store: LocalVectorStore):
        store.remove_file("/ghost.py")  # should not raise
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_remove_file_without_children(self, store: LocalVectorStore):
        await store.upsert([_make_entry("/a.py", "hello")])
        store.remove_file("/a.py")
        assert len(store) == 0

    def test_len_empty(self, store: LocalVectorStore):
        assert len(store) == 0

    def test_index_name(self, store: LocalVectorStore):
        assert store.index_name == "local"


# ==================================================================
# Namespace rejection
# ==================================================================


class TestNamespaceRejection:
    @pytest.mark.asyncio
    async def test_upsert_rejects_namespace(self, store: LocalVectorStore):
        with pytest.raises(ValueError, match="namespaces"):
            await store.upsert([_make_entry("/a.py", "x")], namespace="ns")

    @pytest.mark.asyncio
    async def test_search_rejects_namespace(self, store: LocalVectorStore):
        with pytest.raises(ValueError, match="namespaces"):
            await store.search([0.0] * _DIM, namespace="ns")

    @pytest.mark.asyncio
    async def test_delete_rejects_namespace(self, store: LocalVectorStore):
        with pytest.raises(ValueError, match="namespaces"):
            await store.delete(["/a.py"], namespace="ns")

    @pytest.mark.asyncio
    async def test_fetch_rejects_namespace(self, store: LocalVectorStore):
        with pytest.raises(ValueError, match="namespaces"):
            await store.fetch(["/a.py"], namespace="ns")


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
        entries = [
            _make_entry("/a.py", "hello"),
            _make_entry("/b.py", "world"),
            _make_entry("/chunk", "piece", parent_path="/a.py"),
        ]
        await store.upsert(entries)

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)

        assert len(store2) == 3
        assert store2.has("/a.py")
        assert store2.has("/b.py")
        assert store2.has("/chunk")

    @pytest.mark.asyncio
    async def test_search_after_load(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert([_make_entry("/a.py", "unique content"), _make_entry("/b.py", "other")])

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)
        results = await store2.search(_hash_vector("unique content"), k=2)
        assert results[0].id == "/a.py"

    @pytest.mark.asyncio
    async def test_save_creates_files(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert([_make_entry("/a.py", "content")])
        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        assert (Path(save_dir) / "search.usearch").exists()
        assert (Path(save_dir) / "search_meta.json").exists()

    @pytest.mark.asyncio
    async def test_content_hash_preserved(self, store: LocalVectorStore, tmp_path: Path):
        await store.upsert([_make_entry("/a.py", "hello")])
        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)
        assert store2.content_hash("/a.py") == hashlib.sha256(b"hello").hexdigest()

    @pytest.mark.asyncio
    async def test_parent_tracking_after_load(self, store: LocalVectorStore, tmp_path: Path):
        parent = _make_entry("/a.py", "file")
        child = _make_entry("/chunk", "c1", parent_path="/a.py")
        await store.upsert([parent, child])

        save_dir = str(tmp_path / "index")
        store.save(save_dir)

        store2 = LocalVectorStore(dimension=_DIM)
        store2.load(save_dir)
        store2.remove_file("/a.py")
        assert len(store2) == 0

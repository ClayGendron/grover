"""Tests for GroverStore — LangGraph BaseStore implementation."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

lg = pytest.importorskip("langgraph")

from grover._grover import Grover  # noqa: E402
from grover.fs.local_fs import LocalFileSystem  # noqa: E402
from grover.integrations.langchain._store import GroverStore  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider (deterministic, fast)
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

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
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover(data_dir=str(data))
    g.mount("/data", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def grover_with_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data_search"
    g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
    g.mount("/data", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def store(grover: Grover) -> GroverStore:
    return GroverStore(grover=grover, prefix="/data/store")


# ==================================================================
# Tests
# ==================================================================


class TestStorePutAndGet:
    def test_store_put_and_get(self, store: GroverStore):
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        item = store.get(("users", "alice"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark"}
        assert item.key == "prefs"
        assert item.namespace == ("users", "alice")


class TestStoreGetMissing:
    def test_store_get_missing_returns_none(self, store: GroverStore):
        item = store.get(("nonexistent",), "key")
        assert item is None


class TestStorePutOverwrites:
    def test_store_put_overwrites(self, store: GroverStore):
        store.put(("ns",), "key", {"v": 1})
        store.put(("ns",), "key", {"v": 2})
        item = store.get(("ns",), "key")
        assert item is not None
        assert item.value == {"v": 2}


class TestStoreDelete:
    def test_store_delete(self, store: GroverStore):
        store.put(("ns",), "key", {"v": 1})
        assert store.get(("ns",), "key") is not None
        store.delete(("ns",), "key")
        assert store.get(("ns",), "key") is None


class TestStoreListNamespaces:
    def test_store_list_namespaces(self, store: GroverStore):
        store.put(("users", "alice"), "prefs", {"a": 1})
        store.put(("users", "bob"), "prefs", {"b": 2})
        store.put(("system",), "config", {"c": 3})

        namespaces = store.list_namespaces()
        # Only namespaces that directly contain items are returned
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("system",) in namespaces
        # Parent-only namespaces are NOT included
        assert ("users",) not in namespaces


class TestStoreListNamespacesWithDepth:
    def test_store_list_namespaces_with_depth(self, store: GroverStore):
        store.put(("a", "b", "c"), "key", {"v": 1})
        store.put(("x",), "key", {"v": 2})

        # max_depth truncates (not filters) — per LangGraph spec
        namespaces = store.list_namespaces(max_depth=1)
        assert ("a",) in namespaces  # truncated from ("a", "b", "c")
        assert ("x",) in namespaces
        # Deeper namespaces are truncated, not present at full depth
        assert ("a", "b") not in namespaces
        assert ("a", "b", "c") not in namespaces


class TestStoreSearch:
    def test_store_search(self, grover_with_search: Grover):
        store = GroverStore(grover=grover_with_search, prefix="/data/store")
        store.put(("docs",), "readme", {"content": "Getting started guide"})
        store.put(("docs",), "api", {"content": "API reference documentation"})
        grover_with_search.index()

        results = store.search(("docs",), query="API reference")
        assert isinstance(results, list)
        # Results may or may not find matches depending on embeddings
        # but should not error


class TestStoreSearchNoIndex:
    def test_store_search_no_index(self, store: GroverStore):
        store.put(("docs",), "readme", {"content": "hello"})
        store.put(("docs",), "api", {"content": "api docs"})

        # Without search query, should fall back to listing
        results = store.search(("docs",))
        assert isinstance(results, list)
        assert len(results) == 2
        keys = {r.key for r in results}
        assert "readme" in keys
        assert "api" in keys


class TestStoreNamespaceIsolation:
    def test_store_namespace_isolation(self, store: GroverStore):
        store.put(("ns1",), "key", {"v": "one"})
        store.put(("ns2",), "key", {"v": "two"})

        item1 = store.get(("ns1",), "key")
        item2 = store.get(("ns2",), "key")
        assert item1 is not None
        assert item2 is not None
        assert item1.value["v"] == "one"
        assert item2.value["v"] == "two"


class TestStoreBatchMultipleOps:
    def test_store_batch_multiple_ops(self, store: GroverStore):
        from langgraph.store.base import GetOp, PutOp

        results = store.batch(
            [
                PutOp(("batch",), "k1", {"x": 1}),
                PutOp(("batch",), "k2", {"x": 2}),
                GetOp(("batch",), "k1"),
                GetOp(("batch",), "k2"),
            ]
        )
        assert len(results) == 4
        # First two are puts (return None)
        assert results[0] is None
        assert results[1] is None
        # Last two are gets (return Items)
        assert results[2] is not None
        assert results[2].value == {"x": 1}
        assert results[3] is not None
        assert results[3].value == {"x": 2}


class TestStoreAsyncBatch:
    async def test_store_async_batch(self, store: GroverStore):
        from langgraph.store.base import GetOp, PutOp

        results = await store.abatch(
            [
                PutOp(("async",), "key", {"v": 42}),
            ]
        )
        assert len(results) == 1
        assert results[0] is None

        results = await store.abatch(
            [
                GetOp(("async",), "key"),
            ]
        )
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].value == {"v": 42}

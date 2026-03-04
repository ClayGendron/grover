"""Tests for GroverStore — LangGraph BaseStore implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from _helpers import FakeProvider

lg = pytest.importorskip("langgraph")

from grover.backends.local import LocalFileSystem  # noqa: E402
from grover.client import (  # noqa: E402
    Grover,
    GroverAsync,
)
from grover.integrations.langchain.store import GroverStore  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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
    g = Grover()
    g.add_mount("/data", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def grover_with_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data_search"
    g = Grover()
    g.add_mount(
        "/data",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g
    g.close()


@pytest.fixture
def store(grover: Grover) -> GroverStore:
    return GroverStore(grover=grover, prefix="/data/store")


@pytest.fixture
async def grover_async(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data_async"
    g = GroverAsync()
    await g.add_mount("/data", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def store_async(grover_async: GroverAsync) -> GroverStore:
    return GroverStore(grover=grover_async, prefix="/data/store")


# ==================================================================
# Sync tests (Grover)
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


# ==================================================================
# is_async flag
# ==================================================================


class TestIsAsyncFlag:
    def test_is_async_false_with_grover(self, store: GroverStore):
        assert store._is_async is False

    async def test_is_async_true_with_grover_async(self, store_async: GroverStore):
        assert store_async._is_async is True


# ==================================================================
# Async native tests (GroverAsync)
# ==================================================================


class TestStoreAsyncBatch:
    async def test_abatch_put_and_get(self, store_async: GroverStore):
        from langgraph.store.base import GetOp, PutOp

        results = await store_async.abatch([PutOp(("async",), "key", {"v": 42})])
        assert len(results) == 1
        assert results[0] is None

        results = await store_async.abatch([GetOp(("async",), "key")])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].value == {"v": 42}

    async def test_abatch_search(self, store_async: GroverStore, grover_async: GroverAsync):
        from langgraph.store.base import PutOp, SearchOp

        await store_async.abatch(
            [
                PutOp(("docs",), "readme", {"content": "hello"}),
                PutOp(("docs",), "api", {"content": "api docs"}),
            ]
        )

        results = await store_async.abatch([SearchOp(("docs",))])
        assert len(results) == 1
        items = results[0]
        assert isinstance(items, list)
        assert len(items) == 2
        keys = {item.key for item in items}
        assert "readme" in keys
        assert "api" in keys

    async def test_abatch_list_namespaces(self, store_async: GroverStore):
        from langgraph.store.base import ListNamespacesOp, PutOp

        await store_async.abatch(
            [
                PutOp(("users", "alice"), "prefs", {"a": 1}),
                PutOp(("system",), "config", {"c": 3}),
            ]
        )

        results = await store_async.abatch([ListNamespacesOp()])
        assert len(results) == 1
        namespaces = results[0]
        assert ("users", "alice") in namespaces
        assert ("system",) in namespaces


# ==================================================================
# TypeError when calling async with sync Grover
# ==================================================================


class TestStoreAsyncTypeError:
    async def test_abatch_raises_type_error(self, store: GroverStore):
        from langgraph.store.base import GetOp

        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await store.abatch([GetOp(("ns",), "key")])


# ==================================================================
# Sync wrapper tests (GroverAsync store, sync methods via asyncio.run)
# ==================================================================


def _make_sync_store(tmp_path: Path) -> tuple[GroverStore, GroverAsync]:
    """Create a GroverAsync-backed store outside an event loop."""
    data = tmp_path / "grover_data_sync_store"
    ws = tmp_path / "workspace_sync_store"
    ws.mkdir(exist_ok=True)

    async def _setup() -> GroverAsync:
        g = GroverAsync()
        await g.add_mount("/data", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
        return g

    ga = asyncio.run(_setup())
    return GroverStore(grover=ga, prefix="/data/store"), ga


class TestStoreSyncWrapper:
    def test_put_and_get_sync_wrapper(self, tmp_path: Path):
        store, ga = _make_sync_store(tmp_path)
        try:
            store.put(("ns",), "key", {"v": 42})
            item = store.get(("ns",), "key")
            assert item is not None
            assert item.value == {"v": 42}
        finally:
            asyncio.run(ga.close())

    def test_list_namespaces_sync_wrapper(self, tmp_path: Path):
        store, ga = _make_sync_store(tmp_path)
        try:
            store.put(("users", "alice"), "prefs", {"a": 1})
            namespaces = store.list_namespaces()
            assert ("users", "alice") in namespaces
        finally:
            asyncio.run(ga.close())

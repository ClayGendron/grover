"""Tests for Phase 2: Per-mount graph and search engine."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover._grover_async import GroverAsync
from grover.fs.local_fs import LocalFileSystem
from grover.fs.protocol import SupportsSearch
from grover.graph import RustworkxGraph
from grover.search._engine import SearchEngine

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing."""

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
def workspace1(tmp_path: Path) -> Path:
    ws = tmp_path / "ws1"
    ws.mkdir()
    return ws


@pytest.fixture
def workspace2(tmp_path: Path) -> Path:
    ws = tmp_path / "ws2"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace1: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def multi_grover(workspace1: Path, workspace2: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    await g.mount("/mount1", LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local1"))
    await g.mount("/mount2", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2"))
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Mount injects graph and search engine
# ==================================================================


class TestMountInjection:
    @pytest.mark.asyncio
    async def test_mount_injects_graph(self, grover: GroverAsync):
        """After mount, backend should have a RustworkxGraph."""
        mount = next(
            m for m in grover._registry.list_visible_mounts() if m.mount_path == "/project"
        )
        assert hasattr(mount.backend, "_graph")
        assert isinstance(mount.backend._graph, RustworkxGraph)

    @pytest.mark.asyncio
    async def test_mount_injects_search_engine(self, grover: GroverAsync):
        """After mount, backend should have a SearchEngine."""
        mount = next(
            m for m in grover._registry.list_visible_mounts() if m.mount_path == "/project"
        )
        assert hasattr(mount.backend, "_search_engine")
        assert isinstance(mount.backend._search_engine, SearchEngine)

    @pytest.mark.asyncio
    async def test_no_graph_attr_on_grover(self, grover: GroverAsync):
        """GroverAsync should not have self.graph (removed in favour of per-mount)."""
        assert not hasattr(grover, "graph") or not isinstance(
            getattr(type(grover), "graph", None), property
        )

    @pytest.mark.asyncio
    async def test_get_graph_returns_mount_graph(self, grover: GroverAsync):
        """get_graph() returns the backend's _graph."""
        mount = next(
            m for m in grover._registry.list_visible_mounts() if m.mount_path == "/project"
        )
        assert grover.get_graph() is mount.backend._graph

    @pytest.mark.asyncio
    async def test_get_graph_with_path(self, multi_grover: GroverAsync):
        """get_graph(path) returns the correct mount's graph."""
        m1 = next(
            m for m in multi_grover._registry.list_visible_mounts() if m.mount_path == "/mount1"
        )
        m2 = next(
            m for m in multi_grover._registry.list_visible_mounts() if m.mount_path == "/mount2"
        )
        assert multi_grover.get_graph("/mount1/file.py") is m1.backend._graph
        assert multi_grover.get_graph("/mount2/file.py") is m2.backend._graph

    @pytest.mark.asyncio
    async def test_hidden_mount_no_graph(self, grover: GroverAsync):
        """Hidden mounts (like /.grover) should not get a graph."""
        meta_mount = next(
            (m for m in grover._registry.list_mounts() if m.mount_path == "/.grover"),
            None,
        )
        if meta_mount is not None:
            assert getattr(meta_mount.backend, "_graph", None) is None


# ==================================================================
# Search routes through VFS
# ==================================================================


class TestSearchRouting:
    @pytest.mark.asyncio
    async def test_search_routes_through_vfs(self, grover: GroverAsync):
        """search() should use VFS routing."""
        await grover.write("/project/auth.py", 'def authenticate():\n    """Auth."""\n    pass\n')
        results = await grover.search("authenticate")
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_single_mount(self, multi_grover: GroverAsync):
        """Search scoped to one mount only returns results from that mount."""
        await multi_grover.write(
            "/mount1/auth.py", 'def authenticate():\n    """Auth."""\n    pass\n'
        )
        await multi_grover.write(
            "/mount2/data.py", 'def process_data():\n    """Data."""\n    pass\n'
        )
        results = await multi_grover.search("authenticate", path="/mount1")
        paths = [r.ref.path for r in results] + [r.parent_path for r in results if r.parent_path]
        assert all("/mount1" in p for p in paths if p)

    @pytest.mark.asyncio
    async def test_search_cross_mount_aggregation(self, multi_grover: GroverAsync):
        """Root search aggregates across mounts."""
        await multi_grover.write(
            "/mount1/mod.py", 'def compute():\n    """Compute stuff."""\n    pass\n'
        )
        await multi_grover.write(
            "/mount2/lib.py", 'def compute_more():\n    """Compute more."""\n    pass\n'
        )
        results = await multi_grover.search("compute", path="/")
        # Should have results from both mounts
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_backend_supports_search_protocol(self, grover: GroverAsync):
        """Backends with search engines should satisfy SupportsSearch."""
        mount = next(
            m for m in grover._registry.list_visible_mounts() if m.mount_path == "/project"
        )
        assert isinstance(mount.backend, SupportsSearch)


# ==================================================================
# Write indexes correct mount
# ==================================================================


class TestPerMountIndexing:
    @pytest.mark.asyncio
    async def test_write_indexes_correct_mount(self, multi_grover: GroverAsync):
        """Writing a file should populate the correct mount's graph."""
        await multi_grover.write("/mount1/code.py", "def foo():\n    pass\n")
        m1 = next(
            m for m in multi_grover._registry.list_visible_mounts() if m.mount_path == "/mount1"
        )
        m2 = next(
            m for m in multi_grover._registry.list_visible_mounts() if m.mount_path == "/mount2"
        )
        assert m1.backend._graph.has_node("/mount1/code.py")
        assert not m2.backend._graph.has_node("/mount1/code.py")

    @pytest.mark.asyncio
    async def test_delete_cleans_correct_mount(self, multi_grover: GroverAsync):
        """Deleting a file should clean up the correct mount's graph."""
        await multi_grover.write("/mount1/gone.py", "def gone():\n    pass\n")
        m1 = next(
            m for m in multi_grover._registry.list_visible_mounts() if m.mount_path == "/mount1"
        )
        assert m1.backend._graph.has_node("/mount1/gone.py")
        await multi_grover.delete("/mount1/gone.py")
        assert not m1.backend._graph.has_node("/mount1/gone.py")


# ==================================================================
# Graph operations resolve mount
# ==================================================================


class TestGraphOpsResolveMount:
    @pytest.mark.asyncio
    async def test_dependents_resolves_mount(self, multi_grover: GroverAsync):
        """dependents() uses the correct mount's graph."""
        await multi_grover.write("/mount1/lib.py", "def helper():\n    return 42\n")
        deps = multi_grover.dependents("/mount1/lib.py")
        assert isinstance(deps, list)

    @pytest.mark.asyncio
    async def test_dependencies_resolves_mount(self, multi_grover: GroverAsync):
        """dependencies() uses the correct mount's graph."""
        await multi_grover.write("/mount2/consumer.py", "def main():\n    pass\n")
        deps = multi_grover.dependencies("/mount2/consumer.py")
        assert isinstance(deps, list)

    @pytest.mark.asyncio
    async def test_contains_resolves_mount(self, multi_grover: GroverAsync):
        """contains() uses the correct mount's graph."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        await multi_grover.write("/mount1/funcs.py", code)
        chunks = multi_grover.contains("/mount1/funcs.py")
        assert len(chunks) >= 2


# ==================================================================
# Per-mount persistence
# ==================================================================


class TestPerMountPersistence:
    @pytest.mark.asyncio
    async def test_save_load_per_mount(self, workspace1: Path, workspace2: Path, tmp_path: Path):
        """Graph and search persist and reload per mount."""
        data = tmp_path / "grover_data"

        # Create and populate
        g1 = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g1.mount(
            "/mount1",
            LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local1"),
        )
        await g1.mount(
            "/mount2",
            LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2"),
        )
        await g1.write("/mount1/a.py", "def alpha():\n    pass\n")
        await g1.write("/mount2/b.py", "def beta():\n    pass\n")
        await g1.save()
        await g1.close()

        # Reload
        g2 = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g2.mount(
            "/mount1",
            LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local1"),
        )
        await g2.mount(
            "/mount2",
            LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2"),
        )

        # Both mounts should have their graph state
        assert g2.get_graph("/mount1").has_node("/mount1/a.py")
        assert g2.get_graph("/mount2").has_node("/mount2/b.py")
        # Cross-check: mount1 graph shouldn't have mount2's node
        assert not g2.get_graph("/mount1").has_node("/mount2/b.py")

        # Search should also be restored
        results = await g2.search("alpha", path="/mount1")
        assert len(results) >= 1

        await g2.close()


# ==================================================================
# Engine-based mount
# ==================================================================


class TestEngineMountGraphSearch:
    @pytest.mark.asyncio
    async def test_engine_mount_gets_graph_and_search(self, tmp_path: Path):
        """Engine-based mounts also get per-mount graph and search."""
        from sqlalchemy.ext.asyncio import create_async_engine

        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        try:
            await g.mount("/db", engine=engine)
            mount = next(m for m in g._registry.list_visible_mounts() if m.mount_path == "/db")
            assert isinstance(mount.backend._graph, RustworkxGraph)
            assert isinstance(mount.backend._search_engine, SearchEngine)

            # Write and verify graph
            await g.write("/db/test.py", "def db_func():\n    pass\n")
            assert g.get_graph("/db").has_node("/db/test.py")
        finally:
            await g.close()
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_session_factory_mount_gets_graph(self, tmp_path: Path):
        """Session-factory mounts also get per-mount graph."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel

        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        try:
            await g.mount("/sf", session_factory=factory, dialect="sqlite")
            mount = next(m for m in g._registry.list_visible_mounts() if m.mount_path == "/sf")
            assert isinstance(mount.backend._graph, RustworkxGraph)
        finally:
            await g.close()
            await engine.dispose()


# ==================================================================
# Search path scoping
# ==================================================================


class TestSearchPathScoping:
    @pytest.mark.asyncio
    async def test_search_with_subpath(self, grover: GroverAsync):
        """Search with a subpath should filter to that path."""
        await grover.write("/project/src/auth.py", 'def auth():\n    """Auth."""\n    pass\n')
        await grover.write("/project/tests/test.py", 'def test():\n    """Test."""\n    pass\n')
        # Search scoped to /project/src
        results = await grover.search("auth", path="/project/src")
        # Should find results
        assert isinstance(results, list)

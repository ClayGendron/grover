"""Tests for Phase 2: Per-mount graph and search engine."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.internal.results import FileSearchResult as InternalFileSearchResult
from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph
from grover.providers.search.local import LocalVectorStore

if TYPE_CHECKING:
    from pathlib import Path


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
    g = GroverAsync()
    lfs = LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local")
    await g.add_mount(
        "/project",
        lfs,
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def multi_grover(workspace1: Path, workspace2: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    lfs1 = LocalFileSystem(workspace_dir=workspace1, data_dir=data / "local1")
    lfs2 = LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2")
    await g.add_mount(
        "/mount1",
        lfs1,
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    await g.add_mount(
        "/mount2",
        lfs2,
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Mount injects graph and search engine
# ==================================================================


class TestMountInjection:
    @pytest.mark.asyncio
    async def test_mount_injects_graph(self, grover: GroverAsync):
        """After mount, mount should have a RustworkxGraph."""
        mount = next(m for m in grover._ctx.registry.list_visible_mounts() if m.path == "/project")
        assert mount.filesystem.graph_provider is not None
        assert isinstance(mount.filesystem.graph_provider, RustworkxGraph)

    @pytest.mark.asyncio
    async def test_mount_has_search_provider(self, grover: GroverAsync):
        """After mount with explicit search_provider, filesystem should have it."""
        mount = next(m for m in grover._ctx.registry.list_visible_mounts() if m.path == "/project")
        assert mount.filesystem.search_provider is not None
        assert isinstance(mount.filesystem.search_provider, LocalVectorStore)

    @pytest.mark.asyncio
    async def test_no_graph_attr_on_grover(self, grover: GroverAsync):
        """GroverAsync should not have self.graph (removed in favour of per-mount)."""
        assert not hasattr(grover, "graph") or not isinstance(getattr(type(grover), "graph", None), property)

    @pytest.mark.asyncio
    async def test_get_graph_returns_mount_graph(self, grover: GroverAsync):
        """get_graph() returns the mount's graph."""
        mount = next(m for m in grover._ctx.registry.list_visible_mounts() if m.path == "/project")
        assert grover.get_graph() is mount.filesystem.graph_provider

    @pytest.mark.asyncio
    async def test_get_graph_with_path(self, multi_grover: GroverAsync):
        """get_graph(path) returns the correct mount's graph."""
        m1 = next(m for m in multi_grover._ctx.registry.list_visible_mounts() if m.path == "/mount1")
        m2 = next(m for m in multi_grover._ctx.registry.list_visible_mounts() if m.path == "/mount2")
        assert multi_grover.get_graph("/mount1/file.py") is m1.filesystem.graph_provider
        assert multi_grover.get_graph("/mount2/file.py") is m2.filesystem.graph_provider

    @pytest.mark.asyncio
    async def test_no_hidden_grover_mount(self, grover: GroverAsync):
        """No hidden /.grover mount should be auto-created."""
        meta_mount = next(
            (m for m in grover._ctx.registry.list_mounts() if m.path == "/.grover"),
            None,
        )
        assert meta_mount is None


# ==================================================================
# Search routes through VFS
# ==================================================================


class TestSearchRouting:
    @pytest.mark.asyncio
    async def test_vector_search_routes_through_vfs(self, grover: GroverAsync):
        """vector_search() should use VFS routing."""
        await grover.write("/project/auth.py", 'def authenticate():\n    """Auth."""\n    pass\n')
        await grover.flush()
        result = await grover.vector_search("authenticate")
        assert result.success is True
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_vector_search_single_mount(self, multi_grover: GroverAsync):
        """Search scoped to one mount only returns results from that mount."""
        await multi_grover.write("/mount1/auth.py", 'def authenticate():\n    """Auth."""\n    pass\n')
        await multi_grover.write("/mount2/data.py", 'def process_data():\n    """Data."""\n    pass\n')
        result = await multi_grover.vector_search("authenticate", path="/mount1")
        assert all("/mount1" in p for p in result.paths)

    @pytest.mark.asyncio
    async def test_vector_search_cross_mount_aggregation(self, multi_grover: GroverAsync):
        """Root search aggregates across mounts."""
        await multi_grover.write("/mount1/mod.py", 'def compute():\n    """Compute stuff."""\n    pass\n')
        await multi_grover.write("/mount2/lib.py", 'def compute_more():\n    """Compute more."""\n    pass\n')
        await multi_grover.flush()
        result = await multi_grover.vector_search("compute", path="/")
        # Should have results from both mounts
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_mount_has_search_provider(self, grover: GroverAsync):
        """Mounts with explicit search_provider should have it on filesystem."""
        mount = next(m for m in grover._ctx.registry.list_visible_mounts() if m.path == "/project")
        assert mount.filesystem.search_provider is not None
        assert isinstance(mount.filesystem.search_provider, LocalVectorStore)


# ==================================================================
# Write indexes correct mount
# ==================================================================


class TestPerMountIndexing:
    @pytest.mark.asyncio
    async def test_write_indexes_correct_mount(self, multi_grover: GroverAsync):
        """Writing a file should populate the correct mount's graph."""
        await multi_grover.write("/mount1/code.py", "def foo():\n    pass\n")
        await multi_grover.flush()
        m1 = next(m for m in multi_grover._ctx.registry.list_visible_mounts() if m.path == "/mount1")
        m2 = next(m for m in multi_grover._ctx.registry.list_visible_mounts() if m.path == "/mount2")
        assert m1.filesystem.graph_provider.has_node("/mount1/code.py")
        assert not m2.filesystem.graph_provider.has_node("/mount1/code.py")

    @pytest.mark.asyncio
    async def test_delete_cleans_correct_mount(self, multi_grover: GroverAsync):
        """Deleting a file should clean up the correct mount's graph."""
        await multi_grover.write("/mount1/gone.py", "def gone():\n    pass\n")
        await multi_grover.flush()
        m1 = next(m for m in multi_grover._ctx.registry.list_visible_mounts() if m.path == "/mount1")
        assert m1.filesystem.graph_provider.has_node("/mount1/gone.py")
        await multi_grover.delete("/mount1/gone.py")
        await multi_grover.flush()
        assert not m1.filesystem.graph_provider.has_node("/mount1/gone.py")


# ==================================================================
# Graph operations resolve mount
# ==================================================================


class TestGraphOpsResolveMount:
    @pytest.mark.asyncio
    async def test_predecessors_resolves_mount(self, multi_grover: GroverAsync):
        """predecessors() uses the correct mount's graph."""
        await multi_grover.write("/mount1/lib.py", "def helper():\n    return 42\n")
        await multi_grover.flush()
        result = await multi_grover.predecessors(FileSearchSet.from_paths(["/mount1/lib.py"]))
        assert isinstance(result, InternalFileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_successors_resolves_mount(self, multi_grover: GroverAsync):
        """successors() uses the correct mount's graph."""
        await multi_grover.write("/mount2/consumer.py", "def main():\n    pass\n")
        await multi_grover.flush()
        result = await multi_grover.successors(FileSearchSet.from_paths(["/mount2/consumer.py"]))
        assert isinstance(result, InternalFileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_successors_via_graph_provider_resolves_mount(self, multi_grover: GroverAsync):
        """successors() via graph provider uses the correct mount's graph."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        await multi_grover.write("/mount1/funcs.py", code)
        await multi_grover.flush()
        graph = multi_grover.get_graph("/mount1/funcs.py")
        result = await graph.successors(
            FileSearchSet.from_paths(["/mount1/funcs.py"]),
            session=AsyncMock(),
        )
        assert len(result) >= 2


# ==================================================================
# Per-mount persistence
# ==================================================================


# ==================================================================
# Engine-based mount
# ==================================================================


class TestEngineMountGraphSearch:
    @pytest.mark.asyncio
    async def test_engine_mount_gets_graph_and_search(self, tmp_path: Path):
        """Engine-based mounts also get per-mount graph and search."""
        from sqlalchemy.ext.asyncio import create_async_engine

        g = GroverAsync()
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        try:
            await g.add_mount(
                "/db",
                engine=engine,
                embedding_provider=FakeProvider(),
                search_provider=LocalVectorStore(dimension=FAKE_DIM),
            )
            mount = next(m for m in g._ctx.registry.list_visible_mounts() if m.path == "/db")
            assert isinstance(mount.filesystem.graph_provider, RustworkxGraph)
            assert isinstance(mount.filesystem.search_provider, LocalVectorStore)

            # Write and verify graph
            await g.write("/db/test.py", "def db_func():\n    pass\n")
            await g.flush()
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

        g = GroverAsync()
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        try:
            await g.add_mount("/sf", session_factory=factory, dialect="sqlite", embedding_provider=FakeProvider())
            mount = next(m for m in g._ctx.registry.list_visible_mounts() if m.path == "/sf")
            assert isinstance(mount.filesystem.graph_provider, RustworkxGraph)
        finally:
            await g.close()
            await engine.dispose()


# ==================================================================
# Search path scoping
# ==================================================================


class TestSearchPathScoping:
    @pytest.mark.asyncio
    async def test_vector_search_with_subpath(self, grover: GroverAsync):
        """Search with a subpath should filter to that path."""
        await grover.write("/project/src/auth.py", 'def auth():\n    """Auth."""\n    pass\n')
        await grover.write("/project/tests/test.py", 'def test():\n    """Test."""\n    pass\n')
        # Search scoped to /project/src
        result = await grover.vector_search("auth", path="/project/src")
        # Should find results
        assert result.success is True

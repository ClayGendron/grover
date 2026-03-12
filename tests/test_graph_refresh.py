"""Tests for graph refresh — lazy loading, TTL, and mount wiring."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from grover.client import GroverAsync
from grover.models.connection import FileConnection
from grover.models.file import File
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class TestMountWiring:
    """Verify configure_refresh is wired during add_mount."""

    async def test_configure_refresh_wired_on_mount(self, async_engine: AsyncEngine):
        """After add_mount(), the graph provider has refresh params configured."""
        g = GroverAsync()
        await g.add_mount("/data", engine=async_engine)
        gp = g.get_graph("/data/foo.py")
        assert isinstance(gp, RustworkxGraph)
        assert gp._refresh_file_model is not None
        assert gp._refresh_path_prefix == "/data"
        await g.close()

    async def test_configure_refresh_not_wired_for_hidden_mount(self, async_engine: AsyncEngine):
        """Hidden mounts should not get configure_refresh wired."""
        g = GroverAsync()
        graph = RustworkxGraph()
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        from grover.mount import Mount

        mount = Mount(
            path="/__hidden",
            filesystem=fs,
            session_factory=None,
            hidden=True,
        )
        await g.add_mount(mount)
        # Hidden mount — configure_refresh not called
        assert graph._refresh_file_model is None
        await g.close()


class TestLazyLoadViaFacade:
    """End-to-end: facade graph queries trigger lazy load from DB."""

    async def test_lazy_load_on_first_query(self, async_engine: AsyncEngine):
        """Mount with DB data, call pagerank() → graph auto-loads from DB."""
        # Seed DB with data before mounting
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(File(path="/a.py", parent_path="/"))
            session.add(File(path="/b.py", parent_path="/"))
            session.add(
                FileConnection(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        # Mount — graph is empty, but configure_refresh is wired
        g = GroverAsync()
        await g.add_mount("/data", engine=async_engine)
        gp = g.get_graph()
        assert isinstance(gp, RustworkxGraph)
        # Graph is empty — needs_refresh is True
        assert gp.node_count == 0
        assert gp.needs_refresh is True

        # Query via facade triggers lazy load
        result = await g.pagerank(path="/data/a.py")
        assert result.success
        # Graph should now be populated
        assert gp.node_count >= 2
        assert gp.loaded_at is not None
        await g.close()

    async def test_no_auto_refresh_without_ttl(self, async_engine: AsyncEngine):
        """With stale_after=None, no auto-refresh after initial load."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(File(path="/a.py", parent_path="/"))
            await session.commit()

        g = GroverAsync()
        await g.add_mount("/data", engine=async_engine)
        gp = g.get_graph()
        assert isinstance(gp, RustworkxGraph)

        # Trigger initial load
        await g.pagerank(path="/data/a.py")
        assert gp.has_node("/data/a.py")
        first_load = gp.loaded_at

        # Add new data to DB directly (simulating ETL)
        async with sf() as session:
            session.add(File(path="/etl_new.py", parent_path="/"))
            await session.commit()

        # Query again — no TTL, so no refresh
        await g.pagerank(path="/data/a.py")
        assert gp.loaded_at == first_load  # Same load time — no reload
        assert not gp.has_node("/data/etl_new.py")  # Stale data
        await g.close()

    async def test_auto_refresh_with_ttl(self, async_engine: AsyncEngine):
        """With stale_after set, auto-refreshes after TTL expires."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(File(path="/a.py", parent_path="/"))
            await session.commit()

        g = GroverAsync()
        graph = RustworkxGraph(stale_after=0.01)  # 10ms TTL
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        await g.add_mount("/data", fs, engine=async_engine)
        gp = g.get_graph()

        # Trigger initial load
        await g.pagerank(path="/data/a.py")
        assert gp.has_node("/data/a.py")

        # Add new data to DB directly (simulating ETL)
        async with sf() as session:
            session.add(File(path="/etl_new.py", parent_path="/"))
            await session.commit()

        # Force staleness
        gp._loaded_at = time.monotonic() - 1.0
        assert gp.needs_refresh is True

        # Query again — TTL expired, auto-refreshes
        await g.pagerank(path="/data/a.py")
        assert gp.has_node("/data/etl_new.py")  # Fresh data
        await g.close()

    async def test_db_connection_survives_refresh(self, async_engine: AsyncEngine):
        """Connection in DB survives TTL-triggered refresh."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        # Seed DB with files and a connection (relative paths, matching file storage)
        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(File(path="/a.py", parent_path="/"))
            session.add(File(path="/b.py", parent_path="/"))
            session.add(
                FileConnection(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        g = GroverAsync()
        graph = RustworkxGraph(stale_after=0.01)
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        await g.add_mount("/data", fs, engine=async_engine)

        # Trigger initial load
        await g.pagerank(path="/data/a.py")
        assert graph.has_edge("/data/a.py", "/data/b.py")

        # Force staleness and trigger refresh
        graph._loaded_at = time.monotonic() - 1.0
        await g.pagerank(path="/data/a.py")

        # Connection is still there after refresh
        assert graph.has_edge("/data/a.py", "/data/b.py")
        await g.close()

"""Tests for graph refresh — lazy loading, TTL, and mount wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.client import GroverAsync
from grover.models.config import EngineConfig, SessionConfig
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class TestMountWiring:
    """Verify graph provider is wired correctly during add_mount."""

    async def test_graph_provider_is_rustworkx_after_mount(self):
        """After add_mount(), the graph provider is a RustworkxGraph."""
        g = GroverAsync()
        await g.add_mount("data", engine_config=EngineConfig(url="sqlite+aiosqlite://"))
        gp = g.get_graph("/data/foo.py")
        assert isinstance(gp, RustworkxGraph)
        await g.close()

    async def test_hidden_mount_graph_provider_not_wired(self):
        """Hidden mounts should not have configure_refresh called on graph provider."""
        g = GroverAsync()
        graph = RustworkxGraph()
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        from grover.mount import Mount

        mount = Mount(
            name="__hidden",
            filesystem=fs,
            hidden=True,
        )
        await g.add_mount(mount=mount)
        # Hidden mount — graph provider is still a RustworkxGraph (unchanged)
        assert isinstance(graph, RustworkxGraph)
        await g.close()


class TestLazyLoadViaFacade:
    """End-to-end: facade graph queries trigger lazy load from DB."""

    async def test_lazy_load_on_first_query(self, async_engine: AsyncEngine):
        """Mount with DB data, call pagerank() → graph auto-loads from DB."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(
                FileConnectionModel(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        g = GroverAsync()
        await g.add_mount("data", session_config=SessionConfig(session_factory=sf, dialect="sqlite"))
        gp = g.get_graph()
        assert isinstance(gp, RustworkxGraph)
        # Graph is empty — needs_refresh is True
        assert len(gp.nodes) == 0
        assert gp.needs_refresh is True

        # Query via facade triggers lazy load
        result = await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert result.success
        # Graph should now be populated
        assert len(gp.nodes) >= 2
        assert gp.loaded_at is not None
        await g.close()

    async def test_no_auto_refresh_without_ttl(self, async_engine: AsyncEngine):
        """With ttl=-1 (default), no auto-refresh after initial load."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(
                FileConnectionModel(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        g = GroverAsync()
        await g.add_mount("data", session_config=SessionConfig(session_factory=sf, dialect="sqlite"))
        gp = g.get_graph()
        assert isinstance(gp, RustworkxGraph)

        # Trigger initial load
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert gp.has_node("/a.py")
        first_load = gp.loaded_at

        # Add new connection to DB directly (simulating ETL)
        async with sf() as session:
            session.add(
                FileConnectionModel(
                    source_path="/etl_new.py",
                    target_path="/etl_dep.py",
                    type="imports",
                    path="/etl_new.py[imports]/etl_dep.py",
                )
            )
            await session.commit()

        # Query again — no TTL, so no refresh
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert gp.loaded_at == first_load  # Same load time — no reload
        assert not gp.has_node("/etl_new.py")  # Stale data
        await g.close()

    async def test_auto_refresh_with_stale_after(self, async_engine: AsyncEngine):
        """With stale_after set, auto-refreshes after TTL expires."""
        import time

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(
                FileConnectionModel(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        g = GroverAsync()
        graph = RustworkxGraph(stale_after=1)
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        await g.add_mount("data", filesystem=fs, session_config=SessionConfig(session_factory=sf, dialect="sqlite"))
        gp = g.get_graph()

        # Trigger initial load
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert gp.has_node("/a.py")

        # Add new connection to DB directly (simulating ETL)
        async with sf() as session:
            session.add(
                FileConnectionModel(
                    source_path="/etl_new.py",
                    target_path="/etl_dep.py",
                    type="imports",
                    path="/etl_new.py[imports]/etl_dep.py",
                )
            )
            await session.commit()

        # Force staleness: loaded 2 seconds ago, stale_after=1
        gp.loaded_at = time.monotonic() - 2
        assert gp.needs_refresh is True

        # Query again — TTL expired, auto-refreshes
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert gp.has_node("/etl_new.py")  # Fresh data
        await g.close()

    async def test_db_connection_survives_refresh(self, async_engine: AsyncEngine):
        """Connection in DB survives TTL-triggered refresh."""
        import time

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        # Seed DB with files and a connection
        sf = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add(FileModel(path="/a.py", parent_path="/"))
            session.add(FileModel(path="/b.py", parent_path="/"))
            session.add(
                FileConnectionModel(
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    path="/a.py[imports]/b.py",
                )
            )
            await session.commit()

        g = GroverAsync()
        graph = RustworkxGraph(stale_after=1)
        from grover.backends.database import DatabaseFileSystem

        fs = DatabaseFileSystem(graph_provider=graph)
        await g.add_mount("data", filesystem=fs, session_config=SessionConfig(session_factory=sf, dialect="sqlite"))

        # Trigger initial load
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))
        assert graph.has_edge("/a.py", "/b.py")

        # Force staleness: loaded 2 seconds ago, stale_after=1
        graph.loaded_at = time.monotonic() - 2
        await g.pagerank(FileSearchSet.from_paths(["/data/a.py"]))

        # Connection is still there after refresh
        assert graph.has_edge("/a.py", "/b.py")
        await g.close()

"""Tests for connection operations on DatabaseFileSystem and integration through GroverAsync.

This covers:
- DatabaseFileSystem connection unit tests (low-level DB CRUD)
- GroverAsync + DatabaseFileSystem connection integration (graph)
- GroverAsync + LocalFileSystem connection integration
- _analyze_and_integrate edge routing through FS
- Graph projection (in-memory, updated via worker)
- save() no longer writes edges
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.backends.database import DatabaseFileSystem
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.config import SessionConfig
from grover.models.database.connection import FileConnectionModel
from grover.models.internal.results import FileOperationResult
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# =========================================================================
# DFS connection unit tests (low-level, direct DB access)
# =========================================================================


class TestConnectionMethods:
    @pytest.fixture
    async def setup(
        self,
    ) -> tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()
        yield fs, factory, engine  # type: ignore[misc]
        await engine.dispose()

    async def test_add_connection_creates_db_record(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            result = await fs.add_connection("/a.py", "/b.py", "imports", weight=1.0, session=sess)
            await sess.commit()

        assert isinstance(result, FileOperationResult)
        assert result.success
        assert result.file.path == "/a.py[imports]/b.py"
        assert "created" in result.message.lower()

        # Verify the record exists in DB
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].source_path == "/a.py"
            assert records[0].target_path == "/b.py"
            assert records[0].type == "imports"

    async def test_add_connection_computes_path(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            result = await fs.add_connection("/src/a.py", "/src/b.py", "imports", session=sess)
            await sess.commit()

        assert result.file.path == "/src/a.py[imports]/src/b.py"

    async def test_add_connection_upsert(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        """Adding the same connection twice updates instead of creating duplicate."""
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", weight=1.0, session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.add_connection("/a.py", "/b.py", "imports", weight=2.0, session=sess)
            await sess.commit()

        assert result.success
        assert "updated" in result.message.lower()

        # Should still be only 1 record
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].weight == 2.0

    async def test_delete_connection_removes_record(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.delete_connection("/a.py", "/b.py", connection_type="imports", session=sess)
            await sess.commit()

        assert result.success
        assert "deleted" in result.message.lower()

        # Verify gone
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            assert len(list(row.scalars().all())) == 0

    async def test_delete_connection_not_found(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            result = await fs.delete_connection("/a.py", "/b.py", connection_type="imports", session=sess)

        assert not result.success
        assert "not found" in result.message.lower()

    async def test_delete_connection_all_types(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        """Deleting without connection_type removes all edges between source and target."""
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/a.py", "/b.py", "calls", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.delete_connection("/a.py", "/b.py", session=sess)
            await sess.commit()

        assert result.success

        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            assert len(list(row.scalars().all())) == 0

    async def test_delete_connections_for_path(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        """delete_connections_for_path removes all connections where path is source or target."""
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/c.py", "/a.py", "imports", session=sess)
            await fs.add_connection("/x.py", "/y.py", "imports", session=sess)
            await sess.commit()

        async with factory() as sess:
            count = await fs.delete_connections_for_path(sess, "/a.py")
            await sess.commit()

        assert count == 2

        # /x.py -> /y.py should remain
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].source_path == "/x.py"

    async def test_list_connections_both_directions(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/c.py", "/a.py", "calls", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.list_connections("/a.py", direction="both", session=sess)

        assert len(result.connections) == 2

    async def test_list_connections_outgoing(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/c.py", "/a.py", "calls", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.list_connections("/a.py", direction="out", session=sess)

        assert len(result.connections) == 1
        assert result.connections[0].target_path == "/b.py"

    async def test_list_connections_incoming(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/c.py", "/a.py", "calls", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.list_connections("/a.py", direction="in", session=sess)

        assert len(result.connections) == 1
        assert result.connections[0].source_path == "/c.py"

    async def test_list_connections_filter_by_type(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)
            await fs.add_connection("/a.py", "/c.py", "calls", session=sess)
            await sess.commit()

        async with factory() as sess:
            result = await fs.list_connections("/a.py", direction="out", connection_type="imports", session=sess)

        assert len(result.connections) == 1
        assert result.connections[0].target_path == "/b.py"

    async def test_connection_result_shape(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            result = await fs.add_connection("/src/main.py", "/src/utils.py", "imports", weight=0.5, session=sess)
            await sess.commit()

        assert result.success is True
        assert result.file.path == "/src/main.py[imports]/src/utils.py"


# =========================================================================
# Integration: GroverAsync + DatabaseFileSystem + Graph
# =========================================================================


class TestConnectionIntegrationDBFS:
    """Connection operations through GroverAsync with DatabaseFileSystem."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()

        g = GroverAsync()
        sc = SessionConfig(session_factory=factory, dialect="sqlite")
        await g.add_mount("vfs", filesystem=fs, session_config=sc)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_add_connection_returns_result(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        result = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        assert isinstance(result, FileOperationResult)
        assert result.success
        assert result.file.path == "/vfs/a.py[imports]/vfs/b.py"

    async def test_add_connection_updates_graph(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()

        graph = grover.get_graph("/vfs")
        assert isinstance(graph, RustworkxGraph)
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_delete_connection_removes_from_db_and_graph(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()
        assert grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

        result = await grover.delete_connection("/vfs/a.py", "/vfs/b.py", connection_type="imports")
        await grover.flush()
        assert result.success

        # Graph updated via worker
        assert not grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_delete_connection_updates_graph(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()
        assert grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

        await grover.delete_connection("/vfs/a.py", "/vfs/b.py", connection_type="imports")
        await grover.flush()
        assert not grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_add_connection_creates_graph_edges(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.add_connection("/vfs/a.py", "/vfs/c.py", "calls")
        await grover.flush()

        graph = grover.get_graph("/vfs")
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")
        assert graph.has_edge("/vfs/a.py", "/vfs/c.py")

    async def test_failed_add_does_not_update_graph(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        """add_connection to a nonexistent mount should fail without graph update."""
        grover, _engine = setup
        result = await grover.add_connection("/nope/a.py", "/nope/b.py", "imports")
        assert not result.success
        await grover.flush()
        # Graph should not have any edges for the failed path
        graph = grover.get_graph("/vfs")
        assert not graph.has_edge("/nope/a.py", "/nope/b.py")

    async def test_multiple_connection_types_between_same_files(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        r1 = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        r2 = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "calls")
        assert r1.success
        assert r2.success

        # Edge should exist in graph (graph deduplicates by endpoints)
        graph = grover.get_graph("/vfs")
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")


# =========================================================================
# Integration: GroverAsync + LocalFileSystem + Graph
# =========================================================================


class TestConnectionIntegrationLocalFS:
    """Connection operations through GroverAsync with LocalFileSystem."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> GroverAsync:
        data = tmp_path / "grover_data"
        ws = tmp_path / "workspace"
        ws.mkdir()

        g = GroverAsync()
        lfs = LocalFileSystem(workspace_dir=ws, data_dir=data / "local")
        await g.add_mount("local", filesystem=lfs)

        yield g  # type: ignore[misc]
        await g.close()

    async def test_add_connection_through_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        result = await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        assert result.success
        assert result.file.path == "/local/a.py[imports]/local/b.py"
        await grover.flush()

        # Graph updated
        graph = grover.get_graph("/local")
        assert graph.has_edge("/local/a.py", "/local/b.py")

    async def test_delete_connection_through_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        await grover.flush()

        result = await grover.delete_connection("/local/a.py", "/local/b.py", connection_type="imports")
        await grover.flush()
        assert result.success
        assert not grover.get_graph("/local").has_edge("/local/a.py", "/local/b.py")

    async def test_add_connections_creates_graph_edges_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        await grover.add_connection("/local/c.py", "/local/a.py", "calls")
        await grover.flush()

        graph = grover.get_graph("/local")
        assert graph.has_edge("/local/a.py", "/local/b.py")
        assert graph.has_edge("/local/c.py", "/local/a.py")


# =========================================================================
# Analyze-and-integrate: edges through FS
# =========================================================================


class TestAnalyzeIntegrateConnections:
    """Tests for _analyze_and_integrate — deferred (requires background indexing)."""


# =========================================================================
# DFS.delete_outgoing_connections
# =========================================================================


class TestDeleteOutgoingConnections:
    """Verify delete_outgoing_connections only removes source-side edges."""

    @pytest.fixture
    async def setup(
        self,
    ) -> tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()
        yield fs, factory, engine  # type: ignore[misc]
        await engine.dispose()

    async def test_only_deletes_outgoing(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", session=sess)  # outgoing from /a.py
            await fs.add_connection("/c.py", "/a.py", "calls", session=sess)  # incoming to /a.py
            await fs.add_connection("/x.py", "/y.py", "imports", session=sess)  # unrelated
            await sess.commit()

        async with factory() as sess:
            count = await fs.delete_outgoing_connections(sess, "/a.py")
            await sess.commit()

        assert count == 1  # only /a.py -> /b.py

        # /c.py -> /a.py and /x.py -> /y.py should remain
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = list(row.scalars().all())
            assert len(records) == 2
            sources = {r.source_path for r in records}
            assert sources == {"/c.py", "/x.py"}


# =========================================================================
# Graph projection: loaded from DB, updated via events
# =========================================================================


class TestGraphProjection:
    """Verify graph is an in-memory projection updated via events."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()

        g = GroverAsync()
        sc = SessionConfig(session_factory=factory, dialect="sqlite")
        await g.add_mount("vfs", filesystem=fs, session_config=sc)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_connection_adds_nodes_and_edge_to_graph(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()

        graph = grover.get_graph("/vfs")
        assert graph.has_node("/vfs/a.py")
        assert graph.has_node("/vfs/b.py")
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")

    # test_delete_file_cleans_up_connections — deferred (requires background indexing)


# =========================================================================
# Save no longer writes edges
# =========================================================================


class TestSaveNoEdgePersistence:
    """Verify save no longer calls to_sql for edge persistence."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()

        g = GroverAsync()
        sc = SessionConfig(session_factory=factory, dialect="sqlite")
        await g.add_mount("vfs", filesystem=fs, session_config=sc)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_save_does_not_crash(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        """save() still works even though to_sql is no longer called for edges."""
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")

        # save should succeed without errors
        await grover.save()


# =========================================================================
# Batch add_connections
# =========================================================================


class TestBatchAddConnections:
    """Tests for batch add_connections on DatabaseFileSystem and GroverAsync."""

    @pytest.fixture
    async def setup(
        self,
    ) -> tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()
        yield fs, factory, engine  # type: ignore[misc]
        await engine.dispose()

    async def test_batch_add_creates_multiple(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup
        conns = [
            FileConnectionModel.create("/a.py", "/b.py", "imports"),
            FileConnectionModel.create("/a.py", "/c.py", "calls"),
            FileConnectionModel.create("/x.py", "/y.py", "imports"),
        ]

        async with factory() as sess:
            result = await fs.add_connections(conns, session=sess)
            await sess.commit()

        assert result.success
        assert len(result.connections) == 3

        # Verify all in DB
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = list(row.scalars().all())
            assert len(records) == 3

    async def test_batch_add_updates_existing(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        # Create initial connection
        async with factory() as sess:
            await fs.add_connection("/a.py", "/b.py", "imports", weight=1.0, session=sess)
            await sess.commit()

        # Batch add with same path but different weight
        conns = [
            FileConnectionModel.create("/a.py", "/b.py", "imports", weight=5.0),
            FileConnectionModel.create("/a.py", "/c.py", "calls"),
        ]

        async with factory() as sess:
            result = await fs.add_connections(conns, session=sess)
            await sess.commit()

        assert result.success
        assert len(result.connections) == 2

        # Verify: 2 records total, first one updated
        async with factory() as sess:
            row = await sess.execute(select(FileConnectionModel))
            records = {r.path: r for r in row.scalars().all()}
            assert len(records) == 2
            assert records["/a.py[imports]/b.py"].weight == 5.0

    async def test_batch_add_empty_list(
        self, setup: tuple[DatabaseFileSystem, async_sessionmaker[AsyncSession], AsyncEngine]
    ) -> None:
        fs, factory, _engine = setup

        async with factory() as sess:
            result = await fs.add_connections([], session=sess)

        assert result.success
        assert len(result.connections) == 0

    async def test_batch_add_via_facade(self, tmp_path: Path) -> None:
        """Integration test: batch add_connections through GroverAsync."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()

        g = GroverAsync()
        sc = SessionConfig(session_factory=factory, dialect="sqlite")
        await g.add_mount("vfs", filesystem=fs, session_config=sc)

        try:
            conns = [
                FileConnectionModel.create("/vfs/a.py", "/vfs/b.py", "imports"),
                FileConnectionModel.create("/vfs/a.py", "/vfs/c.py", "calls"),
            ]
            result = await g.add_connections(conns)
            assert result.success
            assert len(result.connections) == 2

            # Graph updated after flush
            await g.flush()
            graph = g.get_graph("/vfs")
            assert graph.has_edge("/vfs/a.py", "/vfs/b.py")
            assert graph.has_edge("/vfs/a.py", "/vfs/c.py")
        finally:
            await g.close()
            await engine.dispose()

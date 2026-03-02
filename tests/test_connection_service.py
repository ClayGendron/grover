"""Tests for ConnectionService and connection integration through GroverAsync.

This covers:
- ConnectionService unit tests (low-level DB CRUD)
- GroverAsync + DatabaseFileSystem connection integration (graph)
- GroverAsync + LocalFileSystem connection integration
- _analyze_and_integrate edge routing through FS
- Graph projection (in-memory, updated via worker)
- save() no longer writes edges
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover._grover_async import GroverAsync
from grover.fs.connections import ConnectionService
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_fs import LocalFileSystem
from grover.graph import RustworkxGraph
from grover.models.connections import FileConnection
from grover.types import ConnectionResult

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# =========================================================================
# Helpers
# =========================================================================

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


# =========================================================================
# ConnectionService unit tests (low-level, direct DB access)
# =========================================================================


class TestConnectionService:
    @pytest.fixture
    async def db(self) -> tuple[AsyncEngine, ConnectionService]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        svc = ConnectionService(FileConnection)
        yield engine, svc  # type: ignore[misc]
        await engine.dispose()

    async def test_add_connection_creates_db_record(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            result = await svc.add_connection(sess, "/a.py", "/b.py", "imports", weight=1.0)
            await sess.commit()

        assert isinstance(result, ConnectionResult)
        assert result.success
        assert result.source_path == "/a.py"
        assert result.target_path == "/b.py"
        assert result.connection_type == "imports"
        assert "created" in result.message.lower()

        # Verify the record exists in DB
        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].source_path == "/a.py"
            assert records[0].target_path == "/b.py"
            assert records[0].type == "imports"

    async def test_add_connection_computes_path(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            result = await svc.add_connection(sess, "/src/a.py", "/src/b.py", "imports")
            await sess.commit()

        assert result.path == "/src/a.py[imports]/src/b.py"

    async def test_add_connection_upsert(self, db: tuple[AsyncEngine, ConnectionService]) -> None:
        """Adding the same connection twice updates instead of creating duplicate."""
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports", weight=1.0)
            await sess.commit()

        async with factory() as sess:
            result = await svc.add_connection(sess, "/a.py", "/b.py", "imports", weight=2.0)
            await sess.commit()

        assert result.success
        assert "updated" in result.message.lower()

        # Should still be only 1 record
        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].weight == 2.0

    async def test_delete_connection_removes_record(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await sess.commit()

        async with factory() as sess:
            result = await svc.delete_connection(sess, "/a.py", "/b.py", connection_type="imports")
            await sess.commit()

        assert result.success
        assert "deleted" in result.message.lower()

        # Verify gone
        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
            assert len(list(row.scalars().all())) == 0

    async def test_delete_connection_not_found(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            result = await svc.delete_connection(sess, "/a.py", "/b.py", connection_type="imports")

        assert not result.success
        assert "not found" in result.message.lower()

    async def test_delete_connection_all_types(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        """Deleting without connection_type removes all edges between source and target."""
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/a.py", "/b.py", "calls")
            await sess.commit()

        async with factory() as sess:
            result = await svc.delete_connection(sess, "/a.py", "/b.py")
            await sess.commit()

        assert result.success

        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
            assert len(list(row.scalars().all())) == 0

    async def test_delete_connections_for_path(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        """delete_connections_for_path removes all connections where path is source or target."""
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/c.py", "/a.py", "imports")
            await svc.add_connection(sess, "/x.py", "/y.py", "imports")
            await sess.commit()

        async with factory() as sess:
            count = await svc.delete_connections_for_path(sess, "/a.py")
            await sess.commit()

        assert count == 2

        # /x.py -> /y.py should remain
        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
            records = list(row.scalars().all())
            assert len(records) == 1
            assert records[0].source_path == "/x.py"

    async def test_list_connections_both_directions(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/c.py", "/a.py", "calls")
            await sess.commit()

        async with factory() as sess:
            conns = await svc.list_connections(sess, "/a.py", direction="both")

        assert len(conns) == 2

    async def test_list_connections_outgoing(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/c.py", "/a.py", "calls")
            await sess.commit()

        async with factory() as sess:
            conns = await svc.list_connections(sess, "/a.py", direction="out")

        assert len(conns) == 1
        assert conns[0].target_path == "/b.py"

    async def test_list_connections_incoming(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/c.py", "/a.py", "calls")
            await sess.commit()

        async with factory() as sess:
            conns = await svc.list_connections(sess, "/a.py", direction="in")

        assert len(conns) == 1
        assert conns[0].source_path == "/c.py"

    async def test_list_connections_filter_by_type(
        self, db: tuple[AsyncEngine, ConnectionService]
    ) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")
            await svc.add_connection(sess, "/a.py", "/c.py", "calls")
            await sess.commit()

        async with factory() as sess:
            conns = await svc.list_connections(
                sess, "/a.py", direction="out", connection_type="imports"
            )

        assert len(conns) == 1
        assert conns[0].target_path == "/b.py"

    async def test_connection_result_shape(self, db: tuple[AsyncEngine, ConnectionService]) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            result = await svc.add_connection(
                sess, "/src/main.py", "/src/utils.py", "imports", weight=0.5
            )
            await sess.commit()

        assert result.success is True
        assert result.path == "/src/main.py[imports]/src/utils.py"
        assert result.source_path == "/src/main.py"
        assert result.target_path == "/src/utils.py"
        assert result.connection_type == "imports"


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
        fs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/vfs", fs, session_factory=factory)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_add_connection_returns_result(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        result = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        assert isinstance(result, ConnectionResult)
        assert result.success
        assert result.path == "/vfs/a.py[imports]/vfs/b.py"
        assert result.source_path == "/vfs/a.py"
        assert result.target_path == "/vfs/b.py"
        assert result.connection_type == "imports"

    async def test_add_connection_updates_graph(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()

        graph = grover.get_graph("/vfs")
        assert isinstance(graph, RustworkxGraph)
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_delete_connection_removes_from_db_and_graph(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()
        assert grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

        result = await grover.delete_connection("/vfs/a.py", "/vfs/b.py", connection_type="imports")
        await grover.flush()
        assert result.success

        # Graph updated via worker
        assert not grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_delete_connection_updates_graph(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()
        assert grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

        await grover.delete_connection("/vfs/a.py", "/vfs/b.py", connection_type="imports")
        await grover.flush()
        assert not grover.get_graph("/vfs").has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_list_connections_returns_records(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.add_connection("/vfs/a.py", "/vfs/c.py", "calls")

        conns = await grover.list_connections("/vfs/a.py", direction="out")
        assert len(conns) == 2

    async def test_list_connections_no_mount_returns_empty(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        conns = await grover.list_connections("/nonexistent/path")
        assert conns == []

    async def test_failed_add_does_not_update_graph(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """add_connection to a nonexistent mount should fail without graph update."""
        grover, _engine = setup
        result = await grover.add_connection("/nope/a.py", "/nope/b.py", "imports")
        assert not result.success
        await grover.flush()
        # Graph should not have any edges for the failed path
        graph = grover.get_graph("/vfs")
        assert not graph.has_edge("/nope/a.py", "/nope/b.py")

    async def test_multiple_connection_types_between_same_files(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        r1 = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        r2 = await grover.add_connection("/vfs/a.py", "/vfs/b.py", "calls")
        assert r1.success
        assert r2.success

        # Both edges should exist in graph
        graph = grover.get_graph("/vfs")
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")

        # DB should have 2 records
        conns = await grover.list_connections("/vfs/a.py", direction="out")
        assert len(conns) == 2


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

        g = GroverAsync(data_dir=str(data))
        lfs = LocalFileSystem(workspace_dir=ws, data_dir=data / "local")
        await g.add_mount("/local", lfs)

        yield g  # type: ignore[misc]
        await g.close()

    async def test_add_connection_through_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        result = await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        assert result.success
        assert result.path == "/local/a.py[imports]/local/b.py"
        await grover.flush()

        # Graph updated
        graph = grover.get_graph("/local")
        assert graph.has_edge("/local/a.py", "/local/b.py")

    async def test_delete_connection_through_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        await grover.flush()

        result = await grover.delete_connection(
            "/local/a.py", "/local/b.py", connection_type="imports"
        )
        await grover.flush()
        assert result.success
        assert not grover.get_graph("/local").has_edge("/local/a.py", "/local/b.py")

    async def test_list_connections_through_local_fs(self, setup: GroverAsync) -> None:
        grover = setup
        await grover.add_connection("/local/a.py", "/local/b.py", "imports")
        await grover.add_connection("/local/c.py", "/local/a.py", "calls")

        conns = await grover.list_connections("/local/a.py")
        assert len(conns) == 2


# =========================================================================
# Analyze-and-integrate: edges through FS
# =========================================================================


class TestAnalyzeIntegrateConnections:
    """Verify that _analyze_and_integrate routes dependency edges through FS."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync(
            data_dir=str(tmp_path / "grover_data"),
            embedding_provider=FakeProvider(),
        )
        await g.add_mount("/vfs", fs, session_factory=factory)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_analyze_persists_edges_through_fs(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Writing Python code that imports a module should persist the edge through FS."""
        grover, engine = setup

        # Write a file with an import
        code = "import os\n\ndef hello():\n    pass\n"
        result = await grover.write("/vfs/main.py", code)
        assert result.success
        await grover.flush()

        # Graph should have the file node
        graph = grover.get_graph("/vfs")
        assert graph.has_node("/vfs/main.py")

        # Verify DB has the connection record
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(select(FileConnection))
            records = list(rows.scalars().all())
            # At least one imports edge
            import_records = [r for r in records if r.type == "imports"]
            assert len(import_records) >= 1

    async def test_reanalyze_replaces_stale_edges(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Re-writing a file should replace old edges with new ones."""
        grover, engine = setup

        # Write with one import
        await grover.write("/vfs/mod.py", "import os\n\ndef f():\n    pass\n")
        await grover.flush()

        # Rewrite with different import
        await grover.write("/vfs/mod.py", "import sys\n\ndef g():\n    pass\n")
        await grover.flush()

        # DB should only have the new edge (old ones deleted)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/mod.py")
            )
            records = list(rows.scalars().all())
            # Should have sys import, not os
            targets = [r.target_path for r in records]
            assert any("sys" in t for t in targets)

    async def test_contains_edges_not_persisted_to_db(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Structural 'contains' edges should stay in-memory, not written to DB."""
        grover, engine = setup

        code = "import os\n\ndef hello():\n    pass\n\nclass Foo:\n    pass\n"
        await grover.write("/vfs/mod.py", code)
        await grover.flush()

        # Graph should have 'contains' edges (in-memory)
        graph = grover.get_graph("/vfs")
        assert graph.has_node("/vfs/mod.py")

        # DB should NOT have any 'contains' edges
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(select(FileConnection))
            records = list(rows.scalars().all())
            contains_records = [r for r in records if r.type == "contains"]
            assert len(contains_records) == 0

    async def test_reanalyze_preserves_incoming_edges(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Re-analyzing a file should not delete connections from OTHER files to this one."""
        grover, engine = setup

        # Write target file
        await grover.write("/vfs/b.py", "x = 1\n")

        # Add a user-created connection from c -> b
        await grover.add_connection("/vfs/c.py", "/vfs/b.py", "depends_on")

        # Rewrite b.py — this triggers re-analysis
        await grover.write("/vfs/b.py", "import os\nx = 2\n")
        await grover.flush()

        # The c -> b connection should still exist in DB
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(
                    FileConnection.source_path == "/vfs/c.py",
                    FileConnection.target_path == "/vfs/b.py",
                )
            )
            records = list(rows.scalars().all())
            assert len(records) == 1
            assert records[0].type == "depends_on"


# =========================================================================
# ConnectionService.delete_outgoing_connections
# =========================================================================


class TestDeleteOutgoingConnections:
    """Verify delete_outgoing_connections only removes source-side edges."""

    @pytest.fixture
    async def db(self) -> tuple[AsyncEngine, ConnectionService]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        svc = ConnectionService(FileConnection)
        yield engine, svc  # type: ignore[misc]
        await engine.dispose()

    async def test_only_deletes_outgoing(self, db: tuple[AsyncEngine, ConnectionService]) -> None:
        engine, svc = db
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with factory() as sess:
            await svc.add_connection(sess, "/a.py", "/b.py", "imports")  # outgoing from /a.py
            await svc.add_connection(sess, "/c.py", "/a.py", "calls")  # incoming to /a.py
            await svc.add_connection(sess, "/x.py", "/y.py", "imports")  # unrelated
            await sess.commit()

        async with factory() as sess:
            count = await svc.delete_outgoing_connections(sess, "/a.py")
            await sess.commit()

        assert count == 1  # only /a.py -> /b.py

        # /c.py -> /a.py and /x.py -> /y.py should remain
        async with factory() as sess:
            row = await sess.execute(select(FileConnection))
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
        fs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/vfs", fs, session_factory=factory)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_connection_adds_nodes_and_edge_to_graph(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")
        await grover.flush()

        graph = grover.get_graph("/vfs")
        assert graph.has_node("/vfs/a.py")
        assert graph.has_node("/vfs/b.py")
        assert graph.has_edge("/vfs/a.py", "/vfs/b.py")

    async def test_delete_file_cleans_up_connections(
        self, setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Deleting a file should clean up its connection DB records."""
        grover, engine = setup

        # Write a file and add connections
        await grover.write("/vfs/a.txt", "content")
        await grover.flush()
        await grover.add_connection("/vfs/a.txt", "/vfs/b.txt", "imports")
        await grover.add_connection("/vfs/c.txt", "/vfs/a.txt", "calls")
        await grover.flush()

        # Delete the file
        await grover.delete("/vfs/a.txt", permanent=True)
        await grover.flush()

        # Connection DB records should be cleaned up
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(
                    (FileConnection.source_path == "/vfs/a.txt")
                    | (FileConnection.target_path == "/vfs/a.txt")
                )
            )
            records = list(rows.scalars().all())
            assert len(records) == 0


# =========================================================================
# Save no longer writes edges
# =========================================================================


class TestSaveNoEdgePersistence:
    """Verify _async_save no longer calls to_sql for edge persistence."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, AsyncEngine]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/vfs", fs, session_factory=factory)

        yield g, engine  # type: ignore[misc]
        await g.close()
        await engine.dispose()

    async def test_save_does_not_crash(self, setup: tuple[GroverAsync, AsyncEngine]) -> None:
        """save() still works even though to_sql is no longer called for edges."""
        grover, _engine = setup
        await grover.add_connection("/vfs/a.py", "/vfs/b.py", "imports")

        # save should succeed without errors
        await grover.save()

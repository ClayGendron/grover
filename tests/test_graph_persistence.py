"""Tests for graph persistence via FileConnection (to_sql / from_sql)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import select

from grover.graph import RustworkxGraph
from grover.models.connections import FileConnection
from grover.models.files import File

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TestToSql:
    """Persist in-memory graph to grover_file_connections."""

    async def test_empty_graph(self, async_session: AsyncSession):
        """to_sql on an empty graph should not create any rows."""
        g = RustworkxGraph()
        await g.to_sql(async_session)
        await async_session.flush()

        result = await async_session.execute(select(FileConnection))
        assert result.scalars().all() == []

    async def test_persists_edges(self, async_session: AsyncSession):
        """Edges in memory should be persisted as FileConnection rows."""
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/")
        g.add_node("/b.py", parent_path="/")
        g.add_edge("/a.py", "/b.py", "imports")

        await g.to_sql(async_session)
        await async_session.flush()

        result = await async_session.execute(select(FileConnection))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].source_path == "/a.py"
        assert rows[0].target_path == "/b.py"
        assert rows[0].type == "imports"

    async def test_edge_weight(self, async_session: AsyncSession):
        """Custom weight should round-trip through persistence."""
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/")
        g.add_node("/b.py", parent_path="/")
        g.add_edge("/a.py", "/b.py", "calls", weight=2.5)

        await g.to_sql(async_session)
        await async_session.flush()

        result = await async_session.execute(select(FileConnection))
        row = result.scalars().one()
        assert row.weight == 2.5

    async def test_stale_edge_cleanup(self, async_session: AsyncSession):
        """Edges removed from memory should be deleted from the DB."""
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/")
        g.add_node("/b.py", parent_path="/")
        g.add_node("/c.py", parent_path="/")
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")

        await g.to_sql(async_session)
        await async_session.flush()

        # Verify 2 edges persisted
        result = await async_session.execute(select(FileConnection))
        assert len(result.scalars().all()) == 2

        # Remove one edge in memory
        g.remove_edge("/a.py", "/c.py")
        await g.to_sql(async_session)
        await async_session.flush()

        # Only 1 edge should remain
        result = await async_session.execute(select(FileConnection))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].source_path == "/a.py"
        assert rows[0].target_path == "/b.py"

    async def test_upsert_idempotency(self, async_session: AsyncSession):
        """Calling to_sql twice without changes should not create duplicates."""
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/")
        g.add_node("/b.py", parent_path="/")
        g.add_edge("/a.py", "/b.py", "imports")

        await g.to_sql(async_session)
        await async_session.flush()

        await g.to_sql(async_session)
        await async_session.flush()

        result = await async_session.execute(select(FileConnection))
        assert len(result.scalars().all()) == 1


class TestFromSql:
    """Load graph state from grover_file_connections."""

    async def test_empty_database(self, async_session: AsyncSession):
        """from_sql on empty DB should produce an empty graph."""
        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.node_count == 0
        assert g.edge_count == 0

    async def test_loads_nodes_and_edges(self, async_session: AsyncSession):
        """Files and connections in the DB should be loaded as nodes and edges."""
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py"
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert g.has_edge("/a.py", "/b.py")
        assert g.node_count == 2
        assert g.edge_count == 1

    async def test_skips_deleted_files(self, async_session: AsyncSession):
        """Soft-deleted files should not be loaded as nodes."""
        from datetime import UTC, datetime

        async_session.add(File(path="/active.py", parent_path="/"))
        async_session.add(
            File(
                path="/deleted.py",
                parent_path="/",
                deleted_at=datetime.now(UTC),
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert g.has_node("/active.py")
        assert not g.has_node("/deleted.py")

    async def test_edge_weight_loaded(self, async_session: AsyncSession):
        """Edge weight should be loaded from DB into edge data."""
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                weight=3.0,
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        edge = g.get_edge("/a.py", "/b.py")
        assert edge is not None
        assert edge["weight"] == 3.0

    async def test_dangling_edge_creates_node(self, async_session: AsyncSession):
        """Edges referencing paths not in the files table should auto-create nodes."""
        # Only one file in DB, but edge references two paths
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py",
                target_path="/missing.py",
                type="imports",
                path="/a.py[imports]/missing.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert g.has_node("/a.py")
        assert g.has_node("/missing.py")  # auto-created by add_edge
        assert g.has_edge("/a.py", "/missing.py")

    async def test_replaces_existing_state(self, async_session: AsyncSession):
        """from_sql should replace any existing in-memory graph state."""
        g = RustworkxGraph()
        g.add_node("/old.py", parent_path="/")

        async_session.add(File(path="/new.py", parent_path="/"))
        await async_session.flush()

        await g.from_sql(async_session)

        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")


class TestRoundTrip:
    """Full to_sql → from_sql round-trip."""

    async def test_simple_round_trip(self, async_session: AsyncSession):
        """Graph saved with to_sql should be fully recoverable with from_sql."""
        g1 = RustworkxGraph()
        g1.add_node("/a.py", parent_path="/")
        g1.add_node("/b.py", parent_path="/")
        g1.add_node("/c.py", parent_path="/")
        g1.add_edge("/a.py", "/b.py", "imports")
        g1.add_edge("/b.py", "/c.py", "calls", weight=2.0)

        # Save nodes as File rows (from_sql needs them)
        for path in g1.nodes():
            node = g1.get_node(path)
            async_session.add(
                File(
                    path=path,
                    parent_path=node.get("parent_path", "/"),
                )
            )
        await g1.to_sql(async_session)
        await async_session.flush()

        # Load into a fresh graph
        g2 = RustworkxGraph()
        await g2.from_sql(async_session)

        assert g2.node_count == 3
        assert g2.edge_count == 2
        assert g2.has_edge("/a.py", "/b.py")
        assert g2.has_edge("/b.py", "/c.py")

        edge = g2.get_edge("/b.py", "/c.py")
        assert edge["weight"] == 2.0

    async def test_round_trip_preserves_edge_ids(self, async_session: AsyncSession):
        """Edge IDs should remain stable across save/load cycles."""
        g1 = RustworkxGraph()
        g1.add_node("/a.py", parent_path="/")
        g1.add_node("/b.py", parent_path="/")
        g1.add_edge("/a.py", "/b.py", "imports")

        original_edge = g1.get_edge("/a.py", "/b.py")
        original_id = original_edge["id"]

        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        await g1.to_sql(async_session)
        await async_session.flush()

        g2 = RustworkxGraph()
        await g2.from_sql(async_session)

        loaded_edge = g2.get_edge("/a.py", "/b.py")
        assert loaded_edge["id"] == original_id

    async def test_uses_file_connections_table(self, async_session: AsyncSession):
        """Verify data is written to grover_file_connections."""
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/")
        g.add_node("/b.py", parent_path="/")
        g.add_edge("/a.py", "/b.py", "imports")

        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        await g.to_sql(async_session)
        await async_session.flush()

        # FileConnection table should have the edge
        result = await async_session.execute(select(FileConnection))
        assert len(result.scalars().all()) == 1

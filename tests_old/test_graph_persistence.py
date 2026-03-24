"""Tests for graph persistence via FileConnectionModel (from_sql only — to_sql removed)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TestToSqlRemoved:
    """to_sql was removed — persistence is via ConnectionService."""

    def test_to_sql_not_available(self) -> None:
        g = RustworkxGraph()
        assert not hasattr(g, "to_sql")


class TestFromSql:
    """Load graph state from grover_file_connections."""

    async def test_empty_database(self, async_session: AsyncSession):
        """from_sql on empty DB should produce an empty graph."""
        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    async def test_loads_nodes_and_edges(self, async_session: AsyncSession):
        """Connection endpoints should be loaded as nodes and edges."""
        async_session.add(
            FileConnectionModel(source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py")
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert g.has_edge("/a.py", "/b.py")
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    async def test_dangling_edge_creates_node(self, async_session: AsyncSession):
        """Edges referencing paths not in the files table should auto-create nodes."""
        async_session.add(
            FileConnectionModel(
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
        assert g.has_node("/missing.py")
        assert g.has_edge("/a.py", "/missing.py")

    async def test_replaces_existing_state(self, async_session: AsyncSession):
        """from_sql should replace any existing in-memory graph state."""
        g = RustworkxGraph()
        g.add_node("/old.py")

        async_session.add(
            FileConnectionModel(
                source_path="/new.py",
                target_path="/new2.py",
                type="imports",
                path="/new.py[imports]/new2.py",
            )
        )
        await async_session.flush()

        await g.from_sql(async_session)

        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")
        assert g.has_node("/new2.py")

    async def test_from_sql_atomic_no_empty_intermediate(self, async_session: AsyncSession):
        """from_sql builds new state in local vars, no empty intermediate on self."""
        g = RustworkxGraph()
        g.add_node("/old.py")
        g.add_node("/old2.py")
        g.add_edge("/old.py", "/old2.py", "imports")

        # Set up DB with different data
        async_session.add(
            FileConnectionModel(
                source_path="/new.py",
                target_path="/new2.py",
                type="imports",
                path="/new.py[imports]/new2.py",
            )
        )
        await async_session.flush()

        # After from_sql, old state is fully replaced — no empty intermediate
        await g.from_sql(async_session)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1
        assert g.has_node("/new.py")
        assert g.has_node("/new2.py")
        assert not g.has_node("/old.py")

    async def test_from_sql_records_loaded_at(self, async_session: AsyncSession):
        """After from_sql(), loaded_at is set to a monotonic timestamp (int seconds)."""
        g = RustworkxGraph()
        assert g.loaded_at is None

        before = time.monotonic()
        await g.from_sql(async_session)
        after = time.monotonic()

        assert g.loaded_at is not None
        assert before <= g.loaded_at <= after

    async def test_from_sql_with_stale_after_does_not_immediately_need_refresh(self, async_session: AsyncSession):
        """After from_sql() with stale_after=60, needs_refresh is False immediately."""
        g = RustworkxGraph(stale_after=60)
        await g.from_sql(async_session)

        assert g.loaded_at is not None
        assert g.needs_refresh is False

    async def test_from_sql_no_stale_after_never_needs_refresh(self, async_session: AsyncSession):
        """After from_sql() with no stale_after (default), needs_refresh stays False."""
        g = RustworkxGraph()  # stale_after=None default
        await g.from_sql(async_session)

        assert g.loaded_at is not None
        assert g.needs_refresh is False

    async def test_from_sql_ignores_files_without_connections(self, async_session: AsyncSession):
        """Files in DB but with no connections should not appear in the graph."""
        async_session.add(FileModel(path="/lonely.py", parent_path="/"))
        async_session.add(
            FileConnectionModel(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert not g.has_node("/lonely.py")
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert len(g.nodes) == 2


class TestStaleness:
    """Tests for needs_refresh, stale_after, loaded_at."""

    def test_needs_refresh_true_when_empty_and_never_loaded(self):
        g = RustworkxGraph()
        assert g.needs_refresh is True

    def test_needs_refresh_false_when_warm_from_mutations(self):
        """Graph populated by writes should not trigger refresh."""
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert g.needs_refresh is False

    def test_needs_refresh_false_after_load_no_stale_after(self):
        g = RustworkxGraph()  # stale_after=None default
        g.loaded_at = time.monotonic()
        assert g.needs_refresh is False

    def test_needs_refresh_true_when_stale(self):
        g = RustworkxGraph(stale_after=1)
        g.loaded_at = time.monotonic() - 2  # loaded 2 seconds ago, TTL is 1
        assert g.needs_refresh is True

    def test_needs_refresh_false_when_fresh(self):
        g = RustworkxGraph(stale_after=3600)
        g.loaded_at = time.monotonic()
        assert g.needs_refresh is False

    def test_stale_after_attribute(self):
        g = RustworkxGraph(stale_after=60)
        assert g.stale_after == 60
        g.stale_after = 120
        assert g.stale_after == 120

    def test_loaded_at_none_initially(self):
        g = RustworkxGraph()
        assert g.loaded_at is None


class TestEnsureFresh:
    """Tests for _ensure_fresh() — the self-refresh mechanism."""

    async def test_ensure_fresh_loads_on_first_query(self, async_session: AsyncSession):
        """_ensure_fresh with session triggers from_sql when graph is empty."""
        async_session.add(
            FileConnectionModel(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        assert g.needs_refresh is True

        await g._ensure_fresh(async_session)

        assert g.has_node("/a.py")
        assert g.needs_refresh is False

    async def test_ensure_fresh_noop_when_fresh(self, async_session: AsyncSession):
        """After loading, _ensure_fresh with no stale_after does not re-query."""
        g = RustworkxGraph()
        await g.from_sql(async_session)

        with patch.object(g, "from_sql", new_callable=AsyncMock) as mock_sql:
            await g._ensure_fresh(async_session)
            mock_sql.assert_not_called()

    async def test_query_method_triggers_lazy_load(self, async_session: AsyncSession):
        """Calling predecessors() with session triggers auto-load from DB."""
        async_session.add(
            FileConnectionModel(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()

        # Graph is empty but calling predecessors with session triggers lazy load
        result = await g.predecessors(FileSearchSet.from_paths(["/b.py"]), session=async_session)
        assert result.success
        assert len(result.files) == 1
        assert result.files[0].path == "/a.py"

    async def test_ensure_fresh_reloads_when_stale(self, async_session: AsyncSession):
        """With stale_after set, _ensure_fresh reloads after TTL expires."""
        async_session.add(
            FileConnectionModel(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph(stale_after=1)
        await g.from_sql(async_session)
        assert g.has_node("/a.py")

        # Simulate staleness: loaded 2 seconds ago, TTL is 1
        g.loaded_at = time.monotonic() - 2
        assert g.needs_refresh is True

        # Add new connection to DB
        async_session.add(
            FileConnectionModel(
                source_path="/b.py",
                target_path="/c.py",
                type="imports",
                path="/b.py[imports]/c.py",
            )
        )
        await async_session.flush()

        # _ensure_fresh should reload
        await g._ensure_fresh(async_session)
        assert g.has_node("/c.py")

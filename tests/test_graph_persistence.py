"""Tests for graph persistence via FileConnection (from_sql only — to_sql removed)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from grover.models.connection import FileConnection
from grover.models.file import File
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

    async def test_dangling_edge_creates_node(self, async_session: AsyncSession):
        """Edges referencing paths not in the files table should auto-create nodes."""
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
        assert g.has_node("/missing.py")
        assert g.has_edge("/a.py", "/missing.py")

    async def test_replaces_existing_state(self, async_session: AsyncSession):
        """from_sql should replace any existing in-memory graph state."""
        g = RustworkxGraph()
        g.add_node("/old.py")

        async_session.add(File(path="/new.py", parent_path="/"))
        await async_session.flush()

        await g.from_sql(async_session)

        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")

    async def test_from_sql_atomic_no_empty_intermediate(self, async_session: AsyncSession):
        """from_sql builds new state in local vars, no empty intermediate on self."""
        g = RustworkxGraph()
        g.add_node("/old.py")
        g.add_node("/old2.py")
        g.add_edge("/old.py", "/old2.py", "imports")

        # Set up DB with different data
        async_session.add(File(path="/new.py", parent_path="/"))
        async_session.add(File(path="/new2.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/new.py",
                target_path="/new2.py",
                type="imports",
                path="/new.py[imports]/new2.py",
            )
        )
        await async_session.flush()

        # After from_sql, old state is fully replaced — no empty intermediate
        await g.from_sql(async_session)
        assert g.node_count == 2
        assert g.edge_count == 1
        assert g.has_node("/new.py")
        assert g.has_node("/new2.py")
        assert not g.has_node("/old.py")

    async def test_from_sql_records_loaded_at(self, async_session: AsyncSession):
        """After from_sql(), _loaded_at is set to a monotonic timestamp."""
        g = RustworkxGraph()
        assert g.loaded_at is None

        before = time.monotonic()
        await g.from_sql(async_session)
        after = time.monotonic()

        assert g.loaded_at is not None
        assert before <= g.loaded_at <= after


class TestStaleness:
    """Tests for needs_refresh, stale_after, loaded_at properties."""

    def test_needs_refresh_true_when_empty_and_never_loaded(self):
        g = RustworkxGraph()
        assert g.needs_refresh is True

    def test_needs_refresh_false_when_warm_from_mutations(self):
        """Graph populated by writes should not trigger refresh."""
        g = RustworkxGraph()
        g.add_node("/a.py")
        # Has data from writes but never loaded from SQL
        assert g.needs_refresh is False

    def test_needs_refresh_false_after_load_no_ttl(self):
        g = RustworkxGraph()
        g._loaded_at = time.monotonic()
        # stale_after=None (default) → no auto-refresh
        assert g.needs_refresh is False

    def test_needs_refresh_true_when_stale(self):
        g = RustworkxGraph(stale_after=0.01)
        g._loaded_at = time.monotonic() - 1.0  # 1 second ago, TTL is 0.01s
        assert g.needs_refresh is True

    def test_needs_refresh_false_when_fresh(self):
        g = RustworkxGraph(stale_after=3600)
        g._loaded_at = time.monotonic()
        assert g.needs_refresh is False

    def test_stale_after_property(self):
        g = RustworkxGraph(stale_after=60.0)
        assert g.stale_after == 60.0
        g.stale_after = 120.0
        assert g.stale_after == 120.0

    def test_loaded_at_none_initially(self):
        g = RustworkxGraph()
        assert g.loaded_at is None


class TestConfigureRefresh:
    """Tests for configure_refresh()."""

    def test_configure_refresh_stores_fields(self):
        g = RustworkxGraph()
        g.configure_refresh(file_model=File, path_prefix="/data")
        assert g._refresh_file_model is File
        assert g._refresh_path_prefix == "/data"

    def test_configure_refresh_defaults(self):
        g = RustworkxGraph()
        g.configure_refresh()
        assert g._refresh_file_model is None
        assert g._refresh_path_prefix == ""


class TestEnsureFresh:
    """Tests for _ensure_fresh() — the self-refresh mechanism."""

    async def test_ensure_fresh_loads_on_first_query(self, async_session: AsyncSession):
        """configure_refresh + _ensure_fresh with session triggers from_sql."""
        async_session.add(File(path="/a.py", parent_path="/"))
        await async_session.flush()

        g = RustworkxGraph()
        g.configure_refresh(file_model=File, path_prefix="")
        assert g.needs_refresh is True

        await g._ensure_fresh(async_session)

        assert g.has_node("/a.py")
        assert g.needs_refresh is False

    async def test_ensure_fresh_noop_when_fresh(self, async_session: AsyncSession):
        """After loading, _ensure_fresh with no TTL does not re-query."""
        g = RustworkxGraph()
        g.configure_refresh(file_model=File, path_prefix="")
        await g.from_sql(async_session)

        with patch.object(g, "from_sql", new_callable=AsyncMock) as mock_sql:
            await g._ensure_fresh(async_session)
            mock_sql.assert_not_called()

    async def test_ensure_fresh_skips_without_session(self):
        """No session → no error, stays empty."""
        g = RustworkxGraph()
        await g._ensure_fresh(None)
        assert g.node_count == 0

    async def test_query_method_triggers_lazy_load(self, async_session: AsyncSession):
        """Calling predecessors() with session triggers auto-load from DB."""
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                path="/a.py[imports]/b.py",
            )
        )
        await async_session.flush()

        g = RustworkxGraph()
        g.configure_refresh(file_model=File, path_prefix="")

        # Graph is empty but calling predecessors with session triggers lazy load
        result = await g.predecessors("/b.py", session=async_session)
        assert result.success
        assert len(result.file_candidates) == 1
        assert result.file_candidates[0].path == "/a.py"

    async def test_ensure_fresh_reloads_when_stale(self, async_session: AsyncSession):
        """With stale_after set, _ensure_fresh reloads after TTL expires."""
        async_session.add(File(path="/a.py", parent_path="/"))
        await async_session.flush()

        g = RustworkxGraph(stale_after=0.01)
        g.configure_refresh(file_model=File, path_prefix="")
        await g.from_sql(async_session)
        assert g.has_node("/a.py")

        # Simulate staleness
        g._loaded_at = time.monotonic() - 1.0
        assert g.needs_refresh is True

        # Add new data to DB
        async_session.add(File(path="/b.py", parent_path="/"))
        await async_session.flush()

        # _ensure_fresh should reload
        await g._ensure_fresh(async_session)
        assert g.has_node("/b.py")

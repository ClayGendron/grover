"""Tests for the RustworkxGraph class — knowledge graph over file paths."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_mock_session = AsyncMock()


# ======================================================================
# TestGraphInit
# ======================================================================


class TestGraphInit:
    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert len(g.nodes) == 0
        assert len(g.edges) == 0
        assert g.nodes == set()
        assert g.edges == []

    def test_repr_empty(self) -> None:
        g = RustworkxGraph()
        assert repr(g) == "RustworkxGraph(nodes=0, edges=0)"


# ======================================================================
# TestNodeOperations
# ======================================================================


class TestNodeOperations:
    def test_add_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert g.has_node("/a.py")
        assert len(g.nodes) == 1

    def test_add_node_idempotent(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/a.py")
        assert len(g.nodes) == 1

    def test_add_node_idempotent_with_attrs(self) -> None:
        # Minimal storage — attrs accepted but not stored
        g = RustworkxGraph()
        g.add_node("/a.py", lang="python")
        g.add_node("/a.py", size=42)
        assert len(g.nodes) == 1

    def test_get_node_includes_path(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py", lang="python")
        data = g.get_node("/a.py")
        assert data["path"] == "/a.py"

    def test_get_node_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError, match="Node not found"):
            g.get_node("/missing.py")

    def test_has_node(self) -> None:
        g = RustworkxGraph()
        assert not g.has_node("/a.py")
        g.add_node("/a.py")
        assert g.has_node("/a.py")

    def test_remove_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.remove_node("/a.py")
        assert not g.has_node("/a.py")
        assert len(g.nodes) == 0

    def test_remove_node_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError, match="Node not found"):
            g.remove_node("/missing.py")

    def test_remove_node_cleans_edges(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        g.add_edge("/a.py", "/b.py", "imports")
        assert len(g.edges) == 1
        g.remove_node("/a.py")
        assert len(g.edges) == 0

    def test_nodes_list(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.nodes == {"/a.py", "/b.py"}


# ======================================================================
# TestEdgeOperations
# ======================================================================


class TestEdgeOperations:
    def test_add_edge(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.has_edge("/a.py", "/b.py")
        assert len(g.edges) == 1

    def test_edge_minimal_data(self) -> None:
        # Minimal storage — type, weight, metadata not stored
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        data = g.get_edge("/a.py", "/b.py")
        assert data["source"] == "/a.py"
        assert data["target"] == "/b.py"

    def test_add_edge_auto_creates_nodes(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")

    def test_add_edge_upsert_idempotent(self) -> None:
        # Minimal storage — second add is a no-op
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/b.py", "imports")
        assert len(g.edges) == 1

    def test_get_edge(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        data = g.get_edge("/a.py", "/b.py")
        assert data["source"] == "/a.py"
        assert data["target"] == "/b.py"

    def test_get_edge_not_found(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        with pytest.raises(KeyError, match="No edge"):
            g.get_edge("/a.py", "/b.py")

    def test_remove_edge(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.remove_edge("/a.py", "/b.py")
        assert not g.has_edge("/a.py", "/b.py")
        assert len(g.edges) == 0

    def test_remove_edge_not_found(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        with pytest.raises(KeyError, match="No edge"):
            g.remove_edge("/a.py", "/b.py")

    def test_has_edge_missing_nodes(self) -> None:
        g = RustworkxGraph()
        assert not g.has_edge("/a.py", "/b.py")

    def test_edges_list(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        edges = g.edges
        assert len(edges) == 2
        sources = {e[0] for e in edges}
        assert sources == {"/a.py", "/b.py"}


# ======================================================================
# TestDependentsAndDependencies
# ======================================================================


class TestPredecessorsAndSuccessors:
    async def test_predecessors_incoming(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/c.py", "/b.py", "imports")
        result = await g.predecessors(FileSearchSet.from_paths(["/b.py"]), session=_mock_session)
        assert set(result.paths) == {"/a.py", "/c.py"}

    async def test_successors_outgoing(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        result = await g.successors(FileSearchSet.from_paths(["/a.py"]), session=_mock_session)
        assert set(result.paths) == {"/b.py", "/c.py"}

    async def test_predecessors_empty(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert len(await g.predecessors(FileSearchSet.from_paths(["/a.py"]), session=_mock_session)) == 0

    async def test_successors_empty(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert len(await g.successors(FileSearchSet.from_paths(["/a.py"]), session=_mock_session)) == 0

    async def test_predecessors_unknown_returns_empty(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 1.0  # mark as loaded so _ensure_fresh is a no-op
        result = await g.predecessors(FileSearchSet.from_paths(["/missing.py"]), session=_mock_session)
        assert result.success
        assert len(result) == 0

    async def test_successors_unknown_returns_empty(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 1.0  # mark as loaded so _ensure_fresh is a no-op
        result = await g.successors(FileSearchSet.from_paths(["/missing.py"]), session=_mock_session)
        assert result.success
        assert len(result) == 0

    async def test_returns_typed_result(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = await g.predecessors(FileSearchSet.from_paths(["/b.py"]), session=_mock_session)
        assert len(result) == 1
        assert result.paths[0] == "/a.py"


# ======================================================================
# TestRemoveNode
# ======================================================================


class TestRemoveNode:
    def test_node_removed(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        g.add_node("/other.py")
        g.remove_node("/file.py")
        assert not g.has_node("/file.py")
        assert g.has_node("/other.py")

    def test_edges_cleaned(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/other.py", "/file.py", "imports")
        g.remove_node("/file.py")
        # Only the edge between remaining nodes is removed, but /file.py::Foo
        # and /other.py still exist as nodes. Edges involving /file.py are gone.
        assert not g.has_edge("/file.py", "/file.py::Foo")
        assert not g.has_edge("/other.py", "/file.py")

    def test_returns_none(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        result = g.remove_node("/file.py")
        assert result is None

    def test_node_gone_after_remove(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        g.remove_node("/file.py")
        assert len(g.nodes) == 0

    def test_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.remove_node("/missing.py")


# ======================================================================
# TestGraphLevel
# ======================================================================


class TestGraphLevel:
    def test_node_count(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert len(g.nodes) == 2

    def test_edge_count(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        assert len(g.edges) == 2

    def test_repr(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert repr(g) == "RustworkxGraph(nodes=2, edges=1)"


# ======================================================================
# Persistence tests (async)
# ======================================================================


class TestToSqlRemoved:
    """to_sql was removed — persistence is via ConnectionService."""

    def test_to_sql_not_available(self) -> None:
        g = RustworkxGraph()
        assert not hasattr(g, "to_sql")


class TestFromSql:
    async def test_loads_nodes_from_connections(self, async_session: AsyncSession) -> None:
        async_session.add(
            FileConnectionModel(source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py")
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert len(g.nodes) == 2

    async def test_loads_edges(self, async_session: AsyncSession) -> None:
        async_session.add(FileModel(path="/a.py", parent_path="/"))
        async_session.add(FileModel(path="/b.py", parent_path="/"))
        async_session.add(
            FileConnectionModel(source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py")
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_edge("/a.py", "/b.py")

    async def test_files_without_connections_not_loaded(self, async_session: AsyncSession) -> None:
        async_session.add(FileModel(path="/lonely.py", parent_path="/"))
        async_session.add(
            FileConnectionModel(source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py")
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert not g.has_node("/lonely.py")
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")

    async def test_clears_existing_graph(self, async_session: AsyncSession) -> None:
        g = RustworkxGraph()
        g.add_node("/old.py")
        assert g.has_node("/old.py")

        async_session.add(
            FileConnectionModel(
                source_path="/new.py",
                target_path="/new2.py",
                type="imports",
                path="/new.py[imports]/new2.py",
            )
        )
        await async_session.commit()

        await g.from_sql(async_session)
        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")

    async def test_auto_creates_nodes_for_dangling_edges(self, async_session: AsyncSession) -> None:
        # Edge endpoints not in grover_files — from_sql should still load them
        async_session.add(
            FileConnectionModel(
                source_path="/orphan_a.py",
                target_path="/orphan_b.py",
                type="imports",
                path="/orphan_a.py[imports]/orphan_b.py",
            )
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_node("/orphan_a.py")
        assert g.has_node("/orphan_b.py")
        assert g.has_edge("/orphan_a.py", "/orphan_b.py")


class TestFromSqlRoundTrip:
    async def test_from_sql_loads_topology(self, async_session: AsyncSession) -> None:
        """from_sql loads nodes from files and edges from connections."""
        async_session.add(FileModel(path="/a.py", parent_path="/"))
        async_session.add(FileModel(path="/b.py", parent_path="/"))
        async_session.add(FileModel(path="/c.py", parent_path="/"))
        async_session.add(
            FileConnectionModel(source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py")
        )
        async_session.add(
            FileConnectionModel(source_path="/b.py", target_path="/c.py", type="calls", path="/b.py[calls]/c.py")
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        assert g.has_edge("/a.py", "/b.py")
        assert g.has_edge("/b.py", "/c.py")

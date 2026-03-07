"""Tests for the RustworkxGraph class — knowledge graph over file paths."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from grover.models.connection import FileConnection
from grover.models.file import File
from grover.providers.graph import RustworkxGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.ref import Ref


# ======================================================================
# Helpers
# ======================================================================


def _ref_paths(refs: list[Ref]) -> set[str]:
    """Extract paths from a list of Refs for order-independent comparison."""
    return {r.path for r in refs}


# ======================================================================
# TestGraphInit
# ======================================================================


class TestGraphInit:
    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.nodes() == []
        assert g.edges() == []

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
        assert g.node_count == 1

    def test_add_node_idempotent(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/a.py")
        assert g.node_count == 1

    def test_add_node_idempotent_with_attrs(self) -> None:
        # Minimal storage — attrs accepted but not stored
        g = RustworkxGraph()
        g.add_node("/a.py", lang="python")
        g.add_node("/a.py", size=42)
        assert g.node_count == 1

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
        assert g.node_count == 0

    def test_remove_node_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError, match="Node not found"):
            g.remove_node("/missing.py")

    def test_remove_node_cleans_edges(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.edge_count == 1
        g.remove_node("/a.py")
        assert g.edge_count == 0

    def test_nodes_list(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert set(g.nodes()) == {"/a.py", "/b.py"}


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
        assert g.edge_count == 1

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
        assert g.edge_count == 1

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
        assert g.edge_count == 0

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
        edges = g.edges()
        assert len(edges) == 2
        sources = {e[0] for e in edges}
        assert sources == {"/a.py", "/b.py"}


# ======================================================================
# TestDependentsAndDependencies
# ======================================================================


class TestPredecessorsAndSuccessors:
    def test_predecessors_incoming(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/c.py", "/b.py", "imports")
        result = g.predecessors("/b.py")
        assert set(result.paths) == {"/a.py", "/c.py"}

    def test_successors_outgoing(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        result = g.successors("/a.py")
        assert set(result.paths) == {"/b.py", "/c.py"}

    def test_predecessors_empty(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert len(g.predecessors("/a.py")) == 0

    def test_successors_empty(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        assert len(g.successors("/a.py")) == 0

    def test_predecessors_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.predecessors("/missing.py")

    def test_successors_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.successors("/missing.py")

    def test_returns_typed_result(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = g.predecessors("/b.py")
        assert len(result) == 1
        assert result.paths[0] == "/a.py"


# ======================================================================
# TestPathBetween
# ======================================================================


class TestPathBetween:
    def test_direct(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = g.path_between("/a.py", "/b.py")
        assert result
        assert list(result.paths) == ["/a.py", "/b.py"]

    def test_multi_hop(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.path_between("/a.py", "/c.py")
        assert result
        assert list(result.paths) == ["/a.py", "/b.py", "/c.py"]

    def test_no_path(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert not g.path_between("/a.py", "/b.py")

    def test_same_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        result = g.path_between("/a.py", "/a.py")
        assert result
        assert list(result.paths) == ["/a.py"]

    def test_source_missing(self) -> None:
        g = RustworkxGraph()
        g.add_node("/b.py")
        with pytest.raises(KeyError):
            g.path_between("/missing.py", "/b.py")

    def test_target_missing(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        with pytest.raises(KeyError):
            g.path_between("/a.py", "/missing.py")


# ======================================================================
# TestContains
# ======================================================================


class TestContains:
    def test_returns_all_successors(self) -> None:
        # Minimal storage — contains() returns all successors (no type filtering)
        g = RustworkxGraph()
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/file.py", "/file.py::bar", "contains")
        g.add_edge("/file.py", "/other.py", "imports")
        refs = g.contains("/file.py")
        assert _ref_paths(refs) == {"/file.py::Foo", "/file.py::bar", "/other.py"}

    def test_empty(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        assert g.contains("/file.py") == []

    def test_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.contains("/missing.py")


# ======================================================================
# TestByParent
# ======================================================================


class TestByParent:
    def test_returns_empty_with_minimal_storage(self) -> None:
        # Minimal storage — parent_path attrs not stored
        g = RustworkxGraph()
        g.add_node("/dir/a.py", parent_path="/dir")
        g.add_node("/dir/b.py", parent_path="/dir")
        assert g.by_parent("/dir") == []

    def test_no_matches(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py", parent_path="/root")
        assert g.by_parent("/nowhere") == []


# ======================================================================
# TestRemoveFileSubgraph
# ======================================================================


class TestRemoveFileSubgraph:
    def test_file_and_successors_removed(self) -> None:
        # Minimal storage uses edges (not parent_path attr) to find children
        g = RustworkxGraph()
        g.add_node("/file.py")
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/file.py", "/file.py::bar", "contains")
        g.add_node("/other.py")
        removed = g.remove_file_subgraph("/file.py")
        assert set(removed) == {"/file.py", "/file.py::Foo", "/file.py::bar"}
        assert not g.has_node("/file.py")
        assert not g.has_node("/file.py::Foo")
        assert g.has_node("/other.py")

    def test_edges_cleaned(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/other.py", "/file.py", "imports")
        g.remove_file_subgraph("/file.py")
        assert g.edge_count == 0

    def test_returns_paths(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        removed = g.remove_file_subgraph("/file.py")
        assert removed == ["/file.py"]

    def test_no_chunks_case(self) -> None:
        g = RustworkxGraph()
        g.add_node("/file.py")
        removed = g.remove_file_subgraph("/file.py")
        assert removed == ["/file.py"]
        assert g.node_count == 0

    def test_not_found(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.remove_file_subgraph("/missing.py")


# ======================================================================
# TestGraphLevel
# ======================================================================


class TestGraphLevel:
    def test_is_dag_true(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert g.is_dag()

    def test_is_dag_false(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        assert not g.is_dag()

    def test_node_count(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.node_count == 2

    def test_edge_count(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        assert g.edge_count == 2

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
    async def test_loads_nodes_from_files(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert g.node_count == 2

    async def test_loads_edges(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py"
            )
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_edge("/a.py", "/b.py")

    async def test_skips_deleted_files(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(
            File(
                path="/deleted.py",
                parent_path="/",
                deleted_at=datetime.now(UTC),
            )
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)
        assert g.has_node("/a.py")
        assert not g.has_node("/deleted.py")

    async def test_clears_existing_graph(self, async_session: AsyncSession) -> None:
        g = RustworkxGraph()
        g.add_node("/old.py")
        assert g.has_node("/old.py")

        async_session.add(File(path="/new.py", parent_path="/"))
        await async_session.commit()

        await g.from_sql(async_session)
        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")

    async def test_auto_creates_nodes_for_dangling_edges(self, async_session: AsyncSession) -> None:
        # Edge endpoints not in grover_files — from_sql should still load them
        async_session.add(
            FileConnection(
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
        async_session.add(File(path="/a.py", parent_path="/"))
        async_session.add(File(path="/b.py", parent_path="/"))
        async_session.add(File(path="/c.py", parent_path="/"))
        async_session.add(
            FileConnection(
                source_path="/a.py", target_path="/b.py", type="imports", path="/a.py[imports]/b.py"
            )
        )
        async_session.add(
            FileConnection(
                source_path="/b.py", target_path="/c.py", type="calls", path="/b.py[calls]/c.py"
            )
        )
        await async_session.commit()

        g = RustworkxGraph()
        await g.from_sql(async_session)

        assert g.node_count == 3
        assert g.edge_count == 2
        assert g.has_edge("/a.py", "/b.py")
        assert g.has_edge("/b.py", "/c.py")

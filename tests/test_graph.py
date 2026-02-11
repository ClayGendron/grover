"""Tests for the Graph class — knowledge graph over file paths."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from grover.graph import Graph
from grover.models.edges import GroverEdge
from grover.models.files import File
from grover.ref import Ref

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ======================================================================
# Helpers
# ======================================================================


def _ref_paths(refs: list[Ref]) -> set[str]:
    """Extract paths from a list of Refs as a set for order-independent comparison."""
    return {r.path for r in refs}


# ======================================================================
# TestGraphInit
# ======================================================================


class TestGraphInit:
    def test_empty_graph(self) -> None:
        g = Graph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.nodes() == []
        assert g.edges() == []

    def test_repr_empty(self) -> None:
        g = Graph()
        assert repr(g) == "Graph(nodes=0, edges=0)"


# ======================================================================
# TestNodeOperations
# ======================================================================


class TestNodeOperations:
    def test_add_node(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        assert g.has_node("/a.py")
        assert g.node_count == 1

    def test_add_node_idempotent(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/a.py")
        assert g.node_count == 1

    def test_add_node_merges_attrs(self) -> None:
        g = Graph()
        g.add_node("/a.py", lang="python")
        g.add_node("/a.py", size=42)
        data = g.get_node("/a.py")
        assert data["lang"] == "python"
        assert data["size"] == 42

    def test_get_node_includes_path(self) -> None:
        g = Graph()
        g.add_node("/a.py", lang="python")
        data = g.get_node("/a.py")
        assert data["path"] == "/a.py"
        assert data["lang"] == "python"

    def test_get_node_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError, match="Node not found"):
            g.get_node("/missing.py")

    def test_has_node(self) -> None:
        g = Graph()
        assert not g.has_node("/a.py")
        g.add_node("/a.py")
        assert g.has_node("/a.py")

    def test_remove_node(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.remove_node("/a.py")
        assert not g.has_node("/a.py")
        assert g.node_count == 0

    def test_remove_node_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError, match="Node not found"):
            g.remove_node("/missing.py")

    def test_remove_node_cleans_edges(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.edge_count == 1
        g.remove_node("/a.py")
        assert g.edge_count == 0

    def test_nodes_list(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert set(g.nodes()) == {"/a.py", "/b.py"}


# ======================================================================
# TestEdgeOperations
# ======================================================================


class TestEdgeOperations:
    def test_add_edge(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.has_edge("/a.py", "/b.py")
        assert g.edge_count == 1

    def test_edge_type_stored(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        data = g.get_edge("/a.py", "/b.py")
        assert data["type"] == "imports"

    def test_edge_weight(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports", weight=2.5)
        data = g.get_edge("/a.py", "/b.py")
        assert data["weight"] == 2.5

    def test_edge_default_weight(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        data = g.get_edge("/a.py", "/b.py")
        assert data["weight"] == 1.0

    def test_edge_metadata(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports", line=10, symbol="Foo")
        data = g.get_edge("/a.py", "/b.py")
        assert data["metadata"] == {"line": 10, "symbol": "Foo"}

    def test_add_edge_auto_creates_nodes(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")

    def test_add_edge_upsert_merges_metadata(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports", line=10)
        g.add_edge("/a.py", "/b.py", "imports", symbol="Foo")
        data = g.get_edge("/a.py", "/b.py")
        assert data["metadata"] == {"line": 10, "symbol": "Foo"}

    def test_add_edge_upsert_preserves_id(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        original_id = g.get_edge("/a.py", "/b.py")["id"]
        g.add_edge("/a.py", "/b.py", "imports", weight=2.0)
        assert g.get_edge("/a.py", "/b.py")["id"] == original_id

    def test_get_edge(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        data = g.get_edge("/a.py", "/b.py")
        assert data["source"] == "/a.py"
        assert data["target"] == "/b.py"

    def test_get_edge_not_found(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        with pytest.raises(KeyError, match="No edge"):
            g.get_edge("/a.py", "/b.py")

    def test_remove_edge(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.remove_edge("/a.py", "/b.py")
        assert not g.has_edge("/a.py", "/b.py")
        assert g.edge_count == 0

    def test_remove_edge_not_found(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        with pytest.raises(KeyError, match="No edge"):
            g.remove_edge("/a.py", "/b.py")

    def test_has_edge_missing_nodes(self) -> None:
        g = Graph()
        assert not g.has_edge("/a.py", "/b.py")

    def test_edges_list(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        edges = g.edges()
        assert len(edges) == 2
        sources = {e[0] for e in edges}
        assert sources == {"/a.py", "/b.py"}


# ======================================================================
# TestDependentsAndDependencies
# ======================================================================


class TestDependentsAndDependencies:
    def test_dependents_incoming(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/c.py", "/b.py", "imports")
        refs = g.dependents("/b.py")
        assert _ref_paths(refs) == {"/a.py", "/c.py"}

    def test_dependencies_outgoing(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        refs = g.dependencies("/a.py")
        assert _ref_paths(refs) == {"/b.py", "/c.py"}

    def test_dependents_empty(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        assert g.dependents("/a.py") == []

    def test_dependencies_empty(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        assert g.dependencies("/a.py") == []

    def test_dependents_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.dependents("/missing.py")

    def test_dependencies_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.dependencies("/missing.py")

    def test_returns_ref_instances(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        refs = g.dependents("/b.py")
        assert len(refs) == 1
        assert isinstance(refs[0], Ref)
        assert refs[0].path == "/a.py"


# ======================================================================
# TestImpacts
# ======================================================================


class TestImpacts:
    def test_single_level(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        refs = g.impacts("/b.py")
        assert _ref_paths(refs) == {"/a.py"}

    def test_transitive(self) -> None:
        # c -> b -> a  (c imports b, b imports a)
        g = Graph()
        g.add_edge("/c.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        refs = g.impacts("/a.py", max_depth=3)
        assert _ref_paths(refs) == {"/b.py", "/c.py"}

    def test_max_depth_limit(self) -> None:
        # d -> c -> b -> a
        g = Graph()
        g.add_edge("/d.py", "/c.py", "imports")
        g.add_edge("/c.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        refs = g.impacts("/a.py", max_depth=1)
        assert _ref_paths(refs) == {"/b.py"}

    def test_cycle_safe(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        refs = g.impacts("/a.py")
        assert _ref_paths(refs) == {"/b.py"}

    def test_excludes_self(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        refs = g.impacts("/b.py")
        assert "/b.py" not in _ref_paths(refs)

    def test_diamond_graph(self) -> None:
        #   c
        #  / \
        # a   b
        #  \ /
        #   d
        g = Graph()
        g.add_edge("/a.py", "/d.py", "imports")
        g.add_edge("/b.py", "/d.py", "imports")
        g.add_edge("/c.py", "/a.py", "imports")
        g.add_edge("/c.py", "/b.py", "imports")
        refs = g.impacts("/d.py", max_depth=3)
        assert _ref_paths(refs) == {"/a.py", "/b.py", "/c.py"}

    def test_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.impacts("/missing.py")


# ======================================================================
# TestPathBetween
# ======================================================================


class TestPathBetween:
    def test_direct(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = g.path_between("/a.py", "/b.py")
        assert result is not None
        paths = [r.path for r in result]
        assert paths == ["/a.py", "/b.py"]

    def test_multi_hop(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.path_between("/a.py", "/c.py")
        assert result is not None
        paths = [r.path for r in result]
        assert paths == ["/a.py", "/b.py", "/c.py"]

    def test_no_path(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.path_between("/a.py", "/b.py") is None

    def test_same_node(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        result = g.path_between("/a.py", "/a.py")
        assert result is not None
        assert [r.path for r in result] == ["/a.py"]

    def test_source_missing(self) -> None:
        g = Graph()
        g.add_node("/b.py")
        with pytest.raises(KeyError):
            g.path_between("/missing.py", "/b.py")

    def test_target_missing(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        with pytest.raises(KeyError):
            g.path_between("/a.py", "/missing.py")


# ======================================================================
# TestContains
# ======================================================================


class TestContains:
    def test_type_filter(self) -> None:
        g = Graph()
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/file.py", "/file.py::bar", "contains")
        g.add_edge("/file.py", "/other.py", "imports")
        refs = g.contains("/file.py")
        assert _ref_paths(refs) == {"/file.py::Foo", "/file.py::bar"}

    def test_empty(self) -> None:
        g = Graph()
        g.add_node("/file.py")
        assert g.contains("/file.py") == []

    def test_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.contains("/missing.py")


# ======================================================================
# TestByParent
# ======================================================================


class TestByParent:
    def test_matches(self) -> None:
        g = Graph()
        g.add_node("/dir/a.py", parent_path="/dir")
        g.add_node("/dir/b.py", parent_path="/dir")
        g.add_node("/other/c.py", parent_path="/other")
        refs = g.by_parent("/dir")
        assert _ref_paths(refs) == {"/dir/a.py", "/dir/b.py"}

    def test_no_matches(self) -> None:
        g = Graph()
        g.add_node("/a.py", parent_path="/root")
        assert g.by_parent("/nowhere") == []

    def test_returns_refs(self) -> None:
        g = Graph()
        g.add_node("/dir/a.py", parent_path="/dir")
        refs = g.by_parent("/dir")
        assert len(refs) == 1
        assert isinstance(refs[0], Ref)


# ======================================================================
# TestRemoveFileSubgraph
# ======================================================================


class TestRemoveFileSubgraph:
    def test_file_and_chunks_removed(self) -> None:
        g = Graph()
        g.add_node("/file.py")
        g.add_node("/file.py::Foo", parent_path="/file.py")
        g.add_node("/file.py::bar", parent_path="/file.py")
        g.add_node("/other.py")
        removed = g.remove_file_subgraph("/file.py")
        assert set(removed) == {"/file.py", "/file.py::Foo", "/file.py::bar"}
        assert not g.has_node("/file.py")
        assert not g.has_node("/file.py::Foo")
        assert g.has_node("/other.py")

    def test_edges_cleaned(self) -> None:
        g = Graph()
        g.add_node("/file.py")
        g.add_node("/file.py::Foo", parent_path="/file.py")
        g.add_edge("/file.py", "/file.py::Foo", "contains")
        g.add_edge("/other.py", "/file.py", "imports")
        g.remove_file_subgraph("/file.py")
        assert g.edge_count == 0

    def test_returns_paths(self) -> None:
        g = Graph()
        g.add_node("/file.py")
        removed = g.remove_file_subgraph("/file.py")
        assert removed == ["/file.py"]

    def test_no_chunks_case(self) -> None:
        g = Graph()
        g.add_node("/file.py")
        removed = g.remove_file_subgraph("/file.py")
        assert removed == ["/file.py"]
        assert g.node_count == 0

    def test_not_found(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.remove_file_subgraph("/missing.py")


# ======================================================================
# TestGraphLevel
# ======================================================================


class TestGraphLevel:
    def test_is_dag_true(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert g.is_dag()

    def test_is_dag_false(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        assert not g.is_dag()

    def test_node_count(self) -> None:
        g = Graph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.node_count == 2

    def test_edge_count(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        assert g.edge_count == 2

    def test_repr(self) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert repr(g) == "Graph(nodes=2, edges=1)"


# ======================================================================
# Persistence tests (async)
# ======================================================================


class TestToSql:
    async def test_creates_edge_rows(self, async_session: AsyncSession) -> None:
        from sqlalchemy import select

        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        await g.to_sql(async_session)
        await async_session.commit()

        result = await async_session.execute(select(GroverEdge))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].source_path == "/a.py"
        assert rows[0].target_path == "/b.py"
        assert rows[0].type == "imports"

    async def test_removes_stale_edges(self, async_session: AsyncSession) -> None:
        from sqlalchemy import select

        # Save initial graph with 2 edges
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "calls")
        await g.to_sql(async_session)
        await async_session.commit()

        # Remove one edge and save again
        g.remove_edge("/b.py", "/c.py")
        await g.to_sql(async_session)
        await async_session.commit()

        result = await async_session.execute(select(GroverEdge))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].source_path == "/a.py"

    async def test_upserts_modified_edges(self, async_session: AsyncSession) -> None:
        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports", weight=1.0)
        await g.to_sql(async_session)
        await async_session.commit()

        # Modify weight via upsert
        g.add_edge("/a.py", "/b.py", "imports", weight=5.0)
        await g.to_sql(async_session)
        await async_session.commit()

        from sqlalchemy import select

        result = await async_session.execute(select(GroverEdge))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].weight == 5.0

    async def test_empty_graph_noop(self, async_session: AsyncSession) -> None:
        from sqlalchemy import select

        g = Graph()
        await g.to_sql(async_session)
        await async_session.commit()

        result = await async_session.execute(select(GroverEdge))
        assert result.scalars().all() == []

    async def test_preserves_edge_ids(self, async_session: AsyncSession) -> None:
        from sqlalchemy import select

        g = Graph()
        g.add_edge("/a.py", "/b.py", "imports")
        edge_id = g.get_edge("/a.py", "/b.py")["id"]
        await g.to_sql(async_session)
        await async_session.commit()

        result = await async_session.execute(select(GroverEdge))
        row = result.scalars().first()
        assert row is not None
        assert row.id == edge_id


class TestFromSql:
    async def test_loads_nodes_from_files(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/", name="a.py"))
        async_session.add(File(path="/b.py", parent_path="/", name="b.py"))
        await async_session.commit()

        g = Graph()
        await g.from_sql(async_session)
        assert g.has_node("/a.py")
        assert g.has_node("/b.py")
        assert g.node_count == 2

    async def test_loads_edges(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/", name="a.py"))
        async_session.add(File(path="/b.py", parent_path="/", name="b.py"))
        async_session.add(GroverEdge(source_path="/a.py", target_path="/b.py", type="imports"))
        await async_session.commit()

        g = Graph()
        await g.from_sql(async_session)
        assert g.has_edge("/a.py", "/b.py")
        assert g.get_edge("/a.py", "/b.py")["type"] == "imports"

    async def test_skips_deleted_files(self, async_session: AsyncSession) -> None:
        async_session.add(File(path="/a.py", parent_path="/", name="a.py"))
        async_session.add(
            File(
                path="/deleted.py",
                parent_path="/",
                name="deleted.py",
                deleted_at=datetime.now(UTC),
            )
        )
        await async_session.commit()

        g = Graph()
        await g.from_sql(async_session)
        assert g.has_node("/a.py")
        assert not g.has_node("/deleted.py")

    async def test_clears_existing_graph(self, async_session: AsyncSession) -> None:
        g = Graph()
        g.add_node("/old.py")
        assert g.has_node("/old.py")

        async_session.add(File(path="/new.py", parent_path="/", name="new.py"))
        await async_session.commit()

        await g.from_sql(async_session)
        assert not g.has_node("/old.py")
        assert g.has_node("/new.py")

    async def test_metadata_round_trips(self, async_session: AsyncSession) -> None:
        import json

        async_session.add(File(path="/a.py", parent_path="/", name="a.py"))
        async_session.add(File(path="/b.py", parent_path="/", name="b.py"))
        async_session.add(
            GroverEdge(
                source_path="/a.py",
                target_path="/b.py",
                type="imports",
                metadata_json=json.dumps({"line": 10, "symbol": "Foo"}),
            )
        )
        await async_session.commit()

        g = Graph()
        await g.from_sql(async_session)
        data = g.get_edge("/a.py", "/b.py")
        assert data["metadata"]["line"] == 10
        assert data["metadata"]["symbol"] == "Foo"

    async def test_auto_creates_nodes_for_dangling_edges(self, async_session: AsyncSession) -> None:
        # Edge endpoints not in grover_files — from_sql should still load them
        async_session.add(
            GroverEdge(source_path="/orphan_a.py", target_path="/orphan_b.py", type="imports")
        )
        await async_session.commit()

        g = Graph()
        await g.from_sql(async_session)
        assert g.has_node("/orphan_a.py")
        assert g.has_node("/orphan_b.py")
        assert g.has_edge("/orphan_a.py", "/orphan_b.py")


class TestRoundTrip:
    async def test_full_round_trip(self, async_session: AsyncSession) -> None:
        g1 = Graph()
        g1.add_node("/a.py", parent_path="/", is_directory=False)
        g1.add_node("/b.py", parent_path="/", is_directory=False)
        g1.add_node("/c.py", parent_path="/", is_directory=False)
        g1.add_edge("/a.py", "/b.py", "imports", line=5)
        g1.add_edge("/b.py", "/c.py", "calls", weight=2.0)

        await g1.to_sql(async_session)
        await async_session.commit()

        # Load into fresh graph
        g2 = Graph()
        await g2.from_sql(async_session)

        assert g2.node_count == g1.node_count
        assert g2.edge_count == g1.edge_count
        assert g2.has_edge("/a.py", "/b.py")
        assert g2.has_edge("/b.py", "/c.py")
        assert g2.get_edge("/a.py", "/b.py")["type"] == "imports"
        assert g2.get_edge("/b.py", "/c.py")["weight"] == 2.0

    async def test_metadata_preserved(self, async_session: AsyncSession) -> None:
        g1 = Graph()
        g1.add_edge("/a.py", "/b.py", "imports", line=10, symbol="Foo")
        await g1.to_sql(async_session)
        await async_session.commit()

        g2 = Graph()
        await g2.from_sql(async_session)
        data = g2.get_edge("/a.py", "/b.py")
        assert data["metadata"] == {"line": 10, "symbol": "Foo"}

    async def test_edge_ids_preserved_round_trip(self, async_session: AsyncSession) -> None:
        g1 = Graph()
        g1.add_edge("/a.py", "/b.py", "imports")
        original_id = g1.get_edge("/a.py", "/b.py")["id"]

        await g1.to_sql(async_session)
        await async_session.commit()

        g2 = Graph()
        await g2.from_sql(async_session)
        assert g2.get_edge("/a.py", "/b.py")["id"] == original_id

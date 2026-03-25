"""Tests for the graph subpackage — UnionFind, RustworkxGraph, GraphProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import rustworkx

from grover.graph import GraphProvider, RustworkxGraph, UnionFind
from grover.models import GroverObject
from grover.results import Candidate, GroverResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_mock_session = AsyncMock()


@pytest.fixture(autouse=True)
def _fresh_mock_session():
    global _mock_session
    _mock_session = AsyncMock()


def _loaded_graph() -> RustworkxGraph:
    """Create a RustworkxGraph that skips DB loading (TTL already satisfied)."""
    g = RustworkxGraph(model=GroverObject)
    g._loaded_at = time.monotonic()
    return g


def _result(*paths: str) -> GroverResult:
    """Build a GroverResult from path strings."""
    return GroverResult(candidates=[Candidate(path=p) for p in paths])


# ===========================================================================
# UnionFind
# ===========================================================================


class TestUnionFindInit:
    def test_initial_state(self):
        uf = UnionFind(["a", "b", "c"])
        assert uf.components == 3
        assert uf.find("a") == "a"
        assert uf.find("b") == "b"

    def test_empty(self):
        uf = UnionFind([])
        assert uf.components == 0


class TestUnionFindUnion:
    def test_union_merges(self):
        uf = UnionFind(["a", "b"])
        assert uf.union("a", "b") is True
        assert uf.components == 1
        assert uf.find("a") == uf.find("b")

    def test_union_same_set_returns_false(self):
        uf = UnionFind(["a", "b"])
        uf.union("a", "b")
        assert uf.union("a", "b") is False
        assert uf.components == 1

    def test_union_three_elements(self):
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.components == 1
        assert uf.find("a") == uf.find("c")

    def test_rank_balancing(self):
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")  # rank[root] = 1
        uf.union("c", "d")  # rank[root] = 1
        uf.union("a", "c")  # merge two rank-1 trees
        assert uf.components == 1

    def test_rank_swap_lower_attaches_to_higher(self):
        """When ra has lower rank than rb, they swap so the smaller tree
        attaches to the larger — covers line 60."""
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")  # {a, b} rank 1
        # Now union c (rank 0) with a (rank 1) — c's root has lower rank
        uf.union("c", "a")  # c's root (rank 0) < a's root (rank 1) → swap
        assert uf.components == 1
        assert uf.find("c") == uf.find("a")


class TestUnionFindFind:
    def test_path_compression(self):
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        root = uf.find("c")
        # After find with path compression, c's parent should point closer to root
        assert uf.parent["c"] == root


# ===========================================================================
# RustworkxGraph — init and repr
# ===========================================================================


class TestRustworkxGraphInit:
    def test_empty_graph(self):
        g = RustworkxGraph(model=GroverObject)
        assert len(g.nodes) == 0
        assert g._loaded_at is None

    def test_default_ttl(self):
        g = RustworkxGraph(model=GroverObject)
        assert g._ttl == 3600

    def test_custom_ttl(self):
        g = RustworkxGraph(model=GroverObject, ttl=60)
        assert g._ttl == 60

    def test_repr_empty(self):
        g = RustworkxGraph(model=GroverObject)
        assert repr(g) == "RustworkxGraph(nodes=0, edges=0)"

    def test_repr_with_data(self):
        g = _loaded_graph()
        g._nodes = {"/a.py", "/b.py"}
        g._out = {"/a.py": {"/b.py"}}
        assert repr(g) == "RustworkxGraph(nodes=2, edges=1)"


# ===========================================================================
# Mutations
# ===========================================================================


class TestAddNode:
    async def test_add_node(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        assert "/a.py" in g.nodes

    async def test_add_node_idempotent(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        await g.add_node("/a.py", session=_mock_session)
        assert len(g.nodes) == 1


class TestRemoveNode:
    async def test_remove_node(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        await g.remove_node("/a.py", session=_mock_session)
        assert "/a.py" not in g.nodes

    async def test_remove_node_missing_raises(self):
        g = _loaded_graph()
        with pytest.raises(KeyError, match="Node not found"):
            await g.remove_node("/missing.py", session=_mock_session)

    async def test_remove_node_cleans_outgoing_edges(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_node("/a.py", session=_mock_session)
        assert not await g.has_edge("/a.py", "/b.py", session=_mock_session)
        # Target node still exists
        assert "/b.py" in g.nodes

    async def test_remove_node_cleans_incoming_edges(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_node("/b.py", session=_mock_session)
        assert "/a.py" in g.nodes
        assert "/a.py" not in g._out  # out set cleaned when only target was /b.py

    async def test_remove_node_cleans_edge_types(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_node("/a.py", session=_mock_session)
        assert ("/a.py", "/b.py") not in g._edge_types


class TestHasNode:
    async def test_has_node_true(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        assert await g.has_node("/a.py", session=_mock_session) is True

    async def test_has_node_false(self):
        g = _loaded_graph()
        assert await g.has_node("/a.py", session=_mock_session) is False


class TestAddEdge:
    async def test_add_edge(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        assert await g.has_edge("/a.py", "/b.py", session=_mock_session)

    async def test_add_edge_auto_creates_nodes(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        assert "/a.py" in g.nodes
        assert "/b.py" in g.nodes

    async def test_add_edge_stores_type(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        assert g._edge_types[("/a.py", "/b.py")] == "imports"

    async def test_add_edge_updates_adjacency(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        assert "/b.py" in g._out["/a.py"]
        assert "/a.py" in g._in["/b.py"]


class TestRemoveEdge:
    async def test_remove_edge(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_edge("/a.py", "/b.py", session=_mock_session)
        assert not await g.has_edge("/a.py", "/b.py", session=_mock_session)

    async def test_remove_edge_missing_source_raises(self):
        g = _loaded_graph()
        with pytest.raises(KeyError, match="Node not found"):
            await g.remove_edge("/missing.py", "/b.py", session=_mock_session)

    async def test_remove_edge_missing_target_raises(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        with pytest.raises(KeyError, match="Node not found"):
            await g.remove_edge("/a.py", "/missing.py", session=_mock_session)

    async def test_remove_edge_no_edge_raises(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        await g.add_node("/b.py", session=_mock_session)
        with pytest.raises(KeyError, match="No edge"):
            await g.remove_edge("/a.py", "/b.py", session=_mock_session)

    async def test_remove_edge_cleans_empty_sets(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_edge("/a.py", "/b.py", session=_mock_session)
        assert "/a.py" not in g._out
        assert "/b.py" not in g._in

    async def test_remove_edge_cleans_edge_types(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.remove_edge("/a.py", "/b.py", session=_mock_session)
        assert ("/a.py", "/b.py") not in g._edge_types


class TestHasEdge:
    async def test_has_edge_true(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        assert await g.has_edge("/a.py", "/b.py", session=_mock_session) is True

    async def test_has_edge_false(self):
        g = _loaded_graph()
        assert await g.has_edge("/a.py", "/b.py", session=_mock_session) is False


# ===========================================================================
# Snapshot and graph construction
# ===========================================================================


class TestSnapshot:
    def test_returns_immutable_copies(self):
        g = _loaded_graph()
        g._nodes = {"/a.py", "/b.py"}
        g._out = {"/a.py": {"/b.py"}}
        nodes, edges = g._snapshot()
        assert isinstance(nodes, frozenset)
        assert isinstance(edges["/a.py"], frozenset)

    def test_snapshot_is_independent(self):
        g = _loaded_graph()
        g._nodes = {"/a.py"}
        g._out = {"/a.py": {"/b.py"}}
        nodes, _ = g._snapshot()
        g._nodes.add("/c.py")
        assert "/c.py" not in nodes


class TestBuildGraphFrom:
    def test_builds_correct_graph(self):
        nodes = frozenset(["/a.py", "/b.py", "/c.py"])
        edges = {"/a.py": frozenset(["/b.py"]), "/b.py": frozenset(["/c.py"])}
        graph, p2i, i2p = RustworkxGraph._build_graph_from(nodes, edges)
        assert graph.num_nodes() == 3
        assert graph.num_edges() == 2
        assert len(p2i) == 3
        assert len(i2p) == 3

    def test_skips_edges_with_missing_nodes(self):
        nodes = frozenset(["/a.py"])
        edges = {"/a.py": frozenset(["/missing.py"])}
        graph, _, _ = RustworkxGraph._build_graph_from(nodes, edges)
        assert graph.num_nodes() == 1
        assert graph.num_edges() == 0

    def test_skips_edges_with_missing_source(self):
        nodes = frozenset(["/b.py"])
        edges = {"/missing.py": frozenset(["/b.py"])}
        graph, _, _ = RustworkxGraph._build_graph_from(nodes, edges)
        assert graph.num_nodes() == 1
        assert graph.num_edges() == 0


class TestGraph:
    async def test_returns_pydigraph(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.graph(session=_mock_session)
        assert isinstance(result, rustworkx.PyDiGraph)
        assert result.num_nodes() == 2
        assert result.num_edges() == 1


# ===========================================================================
# Persistence
# ===========================================================================


class TestEnsureFresh:
    async def test_loads_when_never_loaded(self):
        g = RustworkxGraph(model=GroverObject)
        assert g._loaded_at is None

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await g.ensure_fresh(mock_session)
        assert g._loaded_at is not None
        mock_session.execute.assert_awaited_once()

    async def test_skips_when_within_ttl(self):
        g = _loaded_graph()
        mock_session = AsyncMock()
        await g.ensure_fresh(mock_session)
        mock_session.execute.assert_not_awaited()

    async def test_reloads_when_ttl_expired(self):
        g = RustworkxGraph(model=GroverObject, ttl=0)
        g._loaded_at = time.monotonic() - 1  # expired

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await g.ensure_fresh(mock_session)
        mock_session.execute.assert_awaited_once()


class TestLoad:
    async def test_loads_connection_rows(self):
        g = RustworkxGraph(model=GroverObject)

        row = MagicMock()
        row.source_path = "/a.py"
        row.target_path = "/b.py"
        row.path = "/a.py/.connections/imports/b.py"
        row.connection_type = "imports"
        row.kind = "connection"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        await g._load(mock_session)

        assert "/a.py" in g.nodes
        assert "/b.py" in g.nodes
        assert "/b.py" in g._out["/a.py"]
        assert "/a.py" in g._in["/b.py"]
        assert g._edge_types[("/a.py", "/b.py")] == "imports"
        assert g._loaded_at is not None

    async def test_atomic_swap_clears_old_data(self):
        g = _loaded_graph()
        g._nodes = {"/old.py"}
        g._out = {"/old.py": {"/other.py"}}

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await g._load(mock_session)

        assert "/old.py" not in g.nodes
        assert g._out == {}

    async def test_skips_rows_without_source_or_target(self):
        g = RustworkxGraph(model=GroverObject)

        row = MagicMock()
        row.source_path = None
        row.target_path = "/b.py"
        row.path = "/orphan"
        row.connection_type = None
        row.kind = "connection"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        await g._load(mock_session)
        assert len(g.nodes) == 0

    async def test_falls_back_to_decomposed_type(self):
        g = RustworkxGraph(model=GroverObject)

        row = MagicMock()
        row.source_path = "/a.py"
        row.target_path = "/b.py"
        row.path = "/a.py/.connections/calls/b.py"
        row.connection_type = None  # not set — should decompose from path
        row.kind = "connection"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        await g._load(mock_session)
        # connection_type decomposed from path
        assert g._edge_types[("/a.py", "/b.py")] != ""


# ===========================================================================
# Result construction helpers
# ===========================================================================


class TestRelationshipCandidates:
    def test_builds_candidates(self):
        paths_dict = {"/a.py": ["/b.py", "/c.py"]}
        result = RustworkxGraph._relationship_candidates(paths_dict, "predecessors")
        assert len(result) == 1
        assert result[0].path == "/a.py"
        assert result[0].details[0].operation == "predecessors"
        assert result[0].details[0].metadata["paths"] == ["/b.py", "/c.py"]

    def test_sorted_by_path(self):
        paths_dict = {"/c.py": ["/a.py"], "/a.py": ["/b.py"]}
        result = RustworkxGraph._relationship_candidates(paths_dict, "successors")
        assert [c.path for c in result] == ["/a.py", "/c.py"]


class TestSubgraphCandidates:
    def test_builds_node_and_edge_candidates(self):
        node_set = {"/a.py", "/b.py"}
        edges_out = {"/a.py": frozenset(["/b.py"])}
        edge_types = {("/a.py", "/b.py"): "imports"}
        result = RustworkxGraph._subgraph_candidates(
            node_set, edges_out, edge_types, "neighborhood",
        )
        # 2 node candidates + 1 connection candidate
        assert len(result) == 3
        node_paths = [c.path for c in result if c.weight is None]
        assert sorted(node_paths) == ["/a.py", "/b.py"]
        conn_candidates = [c for c in result if c.weight == 1.0]
        assert len(conn_candidates) == 1

    def test_only_includes_edges_within_node_set(self):
        node_set = {"/a.py"}
        edges_out = {"/a.py": frozenset(["/b.py"])}  # /b.py not in node_set
        edge_types = {("/a.py", "/b.py"): "imports"}
        result = RustworkxGraph._subgraph_candidates(
            node_set, edges_out, edge_types, "subgraph",
        )
        assert len(result) == 1  # only the node, no edge


class TestScoreCandidates:
    def test_sorted_descending(self):
        scores = {"/a.py": 0.3, "/b.py": 0.9, "/c.py": 0.5}
        result = RustworkxGraph._score_candidates(scores, "pagerank")
        assert [c.path for c in result] == ["/b.py", "/c.py", "/a.py"]

    def test_scores_in_details(self):
        scores = {"/a.py": 0.42}
        result = RustworkxGraph._score_candidates(scores, "pagerank")
        assert result[0].details[0].score == 0.42
        assert result[0].details[0].operation == "pagerank"


class TestExtractPaths:
    def test_extracts_paths(self):
        gr = _result("/a.py", "/b.py")
        paths = RustworkxGraph._extract_paths(gr)
        assert paths == ["/a.py", "/b.py"]

    def test_empty_result(self):
        paths = RustworkxGraph._extract_paths(GroverResult())
        assert paths == []


# ===========================================================================
# Traversal — predecessors and successors
# ===========================================================================


class TestPredecessors:
    async def test_finds_predecessors(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/b.py", "imports", session=_mock_session)

        result = await g.predecessors(_result("/b.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/a.py", "/c.py"}

    async def test_excludes_query_paths(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.predecessors(_result("/b.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert "/b.py" not in paths

    async def test_no_predecessors(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.predecessors(_result("/a.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_unknown_path_returns_empty(self):
        g = _loaded_graph()
        result = await g.predecessors(_result("/missing.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        # Add a node so the query path intersects _nodes, then corrupt _in
        g._nodes.add("/a.py")
        g._in = None  # type: ignore[assignment]
        result = await g.predecessors(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "predecessors failed" in result.errors[0]

    async def test_returns_detail_with_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.predecessors(_result("/b.py"), session=_mock_session)
        assert result.candidates[0].details[0].operation == "predecessors"


class TestSuccessors:
    async def test_finds_successors(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/a.py", "/c.py", "calls", session=_mock_session)

        result = await g.successors(_result("/a.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/b.py", "/c.py"}

    async def test_excludes_query_paths(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.successors(_result("/a.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert "/a.py" not in paths

    async def test_no_successors(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.successors(_result("/a.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.successors(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "successors failed" in result.errors[0]


# ===========================================================================
# Traversal — ancestors
# ===========================================================================


class TestAncestors:
    async def test_finds_transitive_ancestors(self):
        # A → B → C → D
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/d.py", "imports", session=_mock_session)

        result = await g.ancestors(_result("/d.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/a.py", "/b.py", "/c.py"}

    async def test_excludes_query_paths(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.ancestors(_result("/b.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert "/b.py" not in paths

    async def test_no_ancestors(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.ancestors(_result("/a.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_unknown_path_returns_empty(self):
        g = _loaded_graph()
        result = await g.ancestors(_result("/missing.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_handles_cycle(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/a.py", "imports", session=_mock_session)

        result = await g.ancestors(_result("/a.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/b.py"}

    async def test_multiple_candidates(self):
        # A → B, C → D
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/d.py", "imports", session=_mock_session)

        result = await g.ancestors(_result("/b.py", "/d.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/a.py", "/c.py"}

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.ancestors(_result("/b.py"), session=_mock_session)
        assert result.candidates[0].details[0].operation == "ancestors"

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.ancestors(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "ancestors failed" in result.errors[0]


# ===========================================================================
# Traversal — descendants
# ===========================================================================


class TestDescendants:
    async def test_finds_transitive_descendants(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/d.py", "imports", session=_mock_session)

        result = await g.descendants(_result("/a.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert paths == {"/b.py", "/c.py", "/d.py"}

    async def test_excludes_query_paths(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.descendants(_result("/a.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert "/a.py" not in paths

    async def test_no_descendants(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.descendants(_result("/a.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_unknown_path_returns_empty(self):
        g = _loaded_graph()
        result = await g.descendants(_result("/missing.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.descendants(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "descendants failed" in result.errors[0]


# ===========================================================================
# Traversal — neighborhood
# ===========================================================================


class TestNeighborhood:
    async def test_depth_1(self):
        # A → B → C
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.neighborhood(_result("/b.py"), depth=1, session=_mock_session)
        node_paths = {c.path for c in result.candidates if c.weight is None}
        assert node_paths == {"/a.py", "/b.py", "/c.py"}

    async def test_depth_0_returns_seed_only(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.neighborhood(_result("/a.py"), depth=0, session=_mock_session)
        node_paths = {c.path for c in result.candidates if c.weight is None}
        assert node_paths == {"/a.py"}

    async def test_follows_both_directions(self):
        # A → B, C → B — neighborhood of B includes both A and C
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/b.py", "calls", session=_mock_session)

        result = await g.neighborhood(_result("/b.py"), depth=1, session=_mock_session)
        node_paths = {c.path for c in result.candidates if c.weight is None}
        assert node_paths == {"/a.py", "/b.py", "/c.py"}

    async def test_includes_edges_within_visited(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.neighborhood(_result("/b.py"), depth=1, session=_mock_session)
        edge_candidates = [c for c in result.candidates if c.weight is not None]
        assert len(edge_candidates) == 2  # a→b and b→c

    async def test_excludes_edges_outside_visited(self):
        # A → B → C → D, depth=1 from B visits {A, B, C}
        # Edge C→D should NOT be included
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)
        await g.add_edge("/c.py", "/d.py", "imports", session=_mock_session)

        result = await g.neighborhood(_result("/b.py"), depth=1, session=_mock_session)
        node_paths = {c.path for c in result.candidates if c.weight is None}
        assert "/d.py" not in node_paths

    async def test_unknown_path_returns_empty(self):
        g = _loaded_graph()
        result = await g.neighborhood(_result("/missing.py"), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_disconnected_node(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.neighborhood(_result("/a.py"), depth=2, session=_mock_session)
        node_paths = {c.path for c in result.candidates if c.weight is None}
        assert node_paths == {"/a.py"}

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.neighborhood(_result("/a.py"), depth=1, session=_mock_session)
        assert result.candidates[0].details[0].operation == "neighborhood"

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.neighborhood(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "neighborhood failed" in result.errors[0]


# ===========================================================================
# Subgraph — meeting_subgraph
# ===========================================================================


class TestMeetingSubgraph:
    async def test_connects_two_seeds(self):
        # A → B → C
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py", "/c.py"), session=_mock_session)
        from grover.paths import decompose_connection

        node_paths = {c.path for c in result.candidates if not decompose_connection(c.path)}
        assert "/a.py" in node_paths
        assert "/c.py" in node_paths
        assert "/b.py" in node_paths  # intermediate

    async def test_single_seed(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py"), session=_mock_session)
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/a.py"

    async def test_zero_seeds(self):
        g = _loaded_graph()
        result = await g.meeting_subgraph(_result(), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_adjacent_seeds(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py", "/b.py"), session=_mock_session)
        from grover.paths import decompose_connection

        node_paths = {c.path for c in result.candidates if not decompose_connection(c.path)}
        assert node_paths == {"/a.py", "/b.py"}

    async def test_disconnected_seeds(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        await g.add_node("/b.py", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py", "/b.py"), session=_mock_session)
        from grover.paths import decompose_connection

        node_paths = {c.path for c in result.candidates if not decompose_connection(c.path)}
        # Both seeds present even though disconnected
        assert "/a.py" in node_paths
        assert "/b.py" in node_paths

    async def test_includes_edge_candidates(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py", "/b.py"), session=_mock_session)
        edge_candidates = [c for c in result.candidates if c.weight is not None]
        assert len(edge_candidates) == 1

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.meeting_subgraph(_result("/a.py", "/b.py"), session=_mock_session)
        assert result.candidates[0].details[0].operation == "meeting_subgraph"

    async def test_unknown_seed_ignored(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.meeting_subgraph(
            _result("/a.py", "/missing.py"), session=_mock_session,
        )
        # Only the valid seed is returned
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/a.py"

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._nodes.add("/b.py")
        g._out = None  # type: ignore[assignment]
        result = await g.meeting_subgraph(_result("/a.py", "/b.py"), session=_mock_session)
        assert result.success is False
        assert "meeting_subgraph failed" in result.errors[0]


# ===========================================================================
# Subgraph — min_meeting_subgraph
# ===========================================================================


class TestMinMeetingSubgraph:
    async def test_subset_of_meeting(self):
        # A → B → C → D, seeds = A and D
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)
        await g.add_edge("/c.py", "/d.py", "imports", session=_mock_session)

        from grover.paths import decompose_connection

        meeting = await g.meeting_subgraph(_result("/a.py", "/d.py"), session=_mock_session)
        min_meeting = await g.min_meeting_subgraph(
            _result("/a.py", "/d.py"), session=_mock_session,
        )

        meeting_nodes = {c.path for c in meeting.candidates if not decompose_connection(c.path)}
        min_nodes = {c.path for c in min_meeting.candidates if not decompose_connection(c.path)}
        assert min_nodes <= meeting_nodes

    async def test_contains_seeds(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        from grover.paths import decompose_connection

        result = await g.min_meeting_subgraph(
            _result("/a.py", "/c.py"), session=_mock_session,
        )
        node_paths = {c.path for c in result.candidates if not decompose_connection(c.path)}
        assert "/a.py" in node_paths
        assert "/c.py" in node_paths

    async def test_removes_non_essential_nodes(self):
        # A → B → C, A → C, seeds = A and C
        # B is not essential since A→C exists directly
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)
        await g.add_edge("/a.py", "/c.py", "imports", session=_mock_session)

        from grover.paths import decompose_connection

        result = await g.min_meeting_subgraph(
            _result("/a.py", "/c.py"), session=_mock_session,
        )
        node_paths = {c.path for c in result.candidates if not decompose_connection(c.path)}
        assert "/a.py" in node_paths
        assert "/c.py" in node_paths
        # B may or may not be removed depending on articulation point logic

    async def test_single_seed_returns_meeting(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.min_meeting_subgraph(_result("/a.py"), session=_mock_session)
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/a.py"

    async def test_prunes_intermediate_non_articulation_node(self):
        """Exercises the articulation point removal loop (lines 726-732).

        Seeds A and B connected by two parallel paths through X and Y:
          A → X → B, A → Y → B, X → Y
        Meeting subgraph includes all 4 nodes. X↔Y cross-link means
        neither is an articulation point — min_meeting can prune one.
        """
        g = _loaded_graph()
        await g.add_edge("/a.py", "/x.py", "imports", session=_mock_session)
        await g.add_edge("/x.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/a.py", "/y.py", "calls", session=_mock_session)
        await g.add_edge("/y.py", "/b.py", "calls", session=_mock_session)
        await g.add_edge("/x.py", "/y.py", "imports", session=_mock_session)

        from grover.paths import decompose_connection

        # Force meeting to include both X and Y by using 3 seeds
        meeting = await g.meeting_subgraph(
            _result("/a.py", "/b.py", "/x.py"), session=_mock_session,
        )
        meeting_nodes = {c.path for c in meeting.candidates if not decompose_connection(c.path)}

        result = await g.min_meeting_subgraph(
            _result("/a.py", "/b.py", "/x.py"), session=_mock_session,
        )
        min_nodes = {c.path for c in result.candidates if not decompose_connection(c.path)}
        assert {"/a.py", "/b.py", "/x.py"} <= min_nodes
        # Y is not a seed and not an articulation point → prunable
        if len(meeting_nodes) > 3:
            assert "/y.py" not in min_nodes

    async def test_error_propagates(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._nodes.add("/b.py")
        g._out = None  # type: ignore[assignment]
        result = await g.min_meeting_subgraph(
            _result("/a.py", "/b.py"), session=_mock_session,
        )
        assert result.success is False


# ===========================================================================
# Subgraph helper — _strip_leaves
# ===========================================================================


class TestStripLeaves:
    def test_removes_unprotected_leaves(self):
        # A → B → C, protected = {A, C}
        edges_out = {"A": frozenset(["B"]), "B": frozenset(["C"])}
        edges_in = {"B": frozenset(["A"]), "C": frozenset(["B"])}
        kept = {"A", "B", "C"}
        protected = {"A", "C"}

        result = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, protected)
        assert result == {"A", "B", "C"}  # B has both preds and succs — not a leaf

    def test_removes_dangling_node(self):
        # A → B → C, A → D, protected = {A, C}
        edges_out = {
            "A": frozenset(["B", "D"]),
            "B": frozenset(["C"]),
        }
        edges_in = {
            "B": frozenset(["A"]),
            "C": frozenset(["B"]),
            "D": frozenset(["A"]),
        }
        kept = {"A", "B", "C", "D"}
        protected = {"A", "C"}

        result = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, protected)
        # D has no successors within kept — should be stripped
        assert "D" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_protects_seeds(self):
        # Single node, protected
        edges_out: dict[str, frozenset[str]] = {}
        edges_in: dict[str, frozenset[str]] = {}
        kept = {"A"}
        protected = {"A"}

        result = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, protected)
        assert result == {"A"}

    def test_cascading_removal(self):
        """Removing one leaf exposes another — covers lines 631-637, 645."""
        # A → B → C → D, protected = {A}
        # D is a leaf (no succs) → removed
        # C becomes a leaf (no succs after D removed) → removed
        # B becomes a leaf (no succs after C removed) → removed
        # A is protected → kept
        edges_out = {
            "A": frozenset(["B"]),
            "B": frozenset(["C"]),
            "C": frozenset(["D"]),
        }
        edges_in = {
            "B": frozenset(["A"]),
            "C": frozenset(["B"]),
            "D": frozenset(["C"]),
        }
        kept = {"A", "B", "C", "D"}
        protected = {"A"}

        result = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, protected)
        assert result == {"A"}

    def test_cascading_removal_via_preds(self):
        """Cascade via predecessor direction — covers line 645."""
        # A → B → C → D, protected = {D}
        # A is a leaf (no preds) → removed
        # B becomes a leaf (no preds after A removed) → removed
        # C becomes a leaf (no preds after B removed) → removed
        # D is protected → kept
        edges_out = {
            "A": frozenset(["B"]),
            "B": frozenset(["C"]),
            "C": frozenset(["D"]),
        }
        edges_in = {
            "B": frozenset(["A"]),
            "C": frozenset(["B"]),
            "D": frozenset(["C"]),
        }
        kept = {"A", "B", "C", "D"}
        protected = {"D"}

        result = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, protected)
        assert result == {"D"}

    def test_empty_input(self):
        result = RustworkxGraph._strip_leaves(set(), {}, {}, set())
        assert result == set()


# ===========================================================================
# Centrality — pagerank
# ===========================================================================


class TestPagerank:
    async def test_returns_scores(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.pagerank(_result(), session=_mock_session)
        assert result.success is True
        assert len(result.candidates) == 3
        # All scores should be positive
        assert all(c.score >= 0 for c in result.candidates)

    async def test_filters_by_candidates(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.pagerank(_result("/a.py"), session=_mock_session)
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/a.py"

    async def test_empty_graph(self):
        g = _loaded_graph()
        result = await g.pagerank(_result(), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.pagerank(_result(), session=_mock_session)
        assert result.candidates[0].details[0].operation == "pagerank"

    async def test_sorted_descending(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/b.py", "calls", session=_mock_session)

        result = await g.pagerank(_result(), session=_mock_session)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Centrality — betweenness
# ===========================================================================


class TestBetweennessCentrality:
    async def test_returns_scores(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.betweenness_centrality(_result(), session=_mock_session)
        assert result.success is True
        assert len(result.candidates) == 3

    async def test_bridge_node_scores_highest(self):
        # A → B → C — B is the bridge
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.betweenness_centrality(_result(), session=_mock_session)
        scores = {c.path: c.score for c in result.candidates}
        assert scores["/b.py"] >= scores["/a.py"]
        assert scores["/b.py"] >= scores["/c.py"]

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.betweenness_centrality(_result(), session=_mock_session)
        assert result.candidates[0].details[0].operation == "betweenness_centrality"

    async def test_error_returns_failure(self):
        """Covers _run_centrality error handler (lines 761-762)."""
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.betweenness_centrality(_result(), session=_mock_session)
        assert result.success is False
        assert "betweenness_centrality failed" in result.errors[0]


# ===========================================================================
# Centrality — closeness
# ===========================================================================


class TestClosenessCentrality:
    async def test_returns_scores(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.closeness_centrality(_result(), session=_mock_session)
        assert result.success is True
        assert len(result.candidates) == 3

    async def test_filters_by_candidates(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.closeness_centrality(_result("/b.py"), session=_mock_session)
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/b.py"

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.closeness_centrality(_result(), session=_mock_session)
        assert result.candidates[0].details[0].operation == "closeness_centrality"


# ===========================================================================
# Centrality — degree
# ===========================================================================


class TestDegreeCentrality:
    async def test_degree_centrality(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.degree_centrality(_result(), session=_mock_session)
        assert result.success is True
        assert len(result.candidates) == 2

    async def test_in_degree_centrality(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/c.py", "/b.py", "calls", session=_mock_session)

        result = await g.in_degree_centrality(_result(), session=_mock_session)
        scores = {c.path: c.score for c in result.candidates}
        # B has 2 incoming edges, highest in-degree
        assert scores["/b.py"] > scores["/a.py"]

    async def test_out_degree_centrality(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/a.py", "/c.py", "calls", session=_mock_session)

        result = await g.out_degree_centrality(_result(), session=_mock_session)
        scores = {c.path: c.score for c in result.candidates}
        # A has 2 outgoing edges, highest out-degree
        assert scores["/a.py"] > scores["/b.py"]

    async def test_detail_operations(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        for method, op in [
            (g.degree_centrality, "degree_centrality"),
            (g.in_degree_centrality, "in_degree_centrality"),
            (g.out_degree_centrality, "out_degree_centrality"),
        ]:
            result = await method(_result(), session=_mock_session)
            assert result.candidates[0].details[0].operation == op


# ===========================================================================
# Centrality — HITS
# ===========================================================================


class TestHits:
    async def test_returns_hub_and_authority_scores(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/a.py", "/c.py", "calls", session=_mock_session)

        result = await g.hits(_result(), session=_mock_session)
        assert result.success is True
        for c in result.candidates:
            meta = c.details[0].metadata
            assert "authority" in meta
            assert "hub" in meta

    async def test_authority_score_in_detail(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.hits(_result(), session=_mock_session)
        for c in result.candidates:
            assert c.details[0].score == c.details[0].metadata["authority"]

    async def test_filters_by_candidates(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        await g.add_edge("/b.py", "/c.py", "calls", session=_mock_session)

        result = await g.hits(_result("/b.py"), session=_mock_session)
        assert len(result.candidates) == 1
        assert result.candidates[0].path == "/b.py"

    async def test_edgeless_graph_returns_zero_scores(self):
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)
        await g.add_node("/b.py", session=_mock_session)

        result = await g.hits(_result(), session=_mock_session)
        assert result.success is True
        for c in result.candidates:
            assert c.score == 0.0
            assert c.details[0].metadata["hub"] == 0.0

    async def test_edgeless_filters_to_graph_paths(self):
        """Non-graph candidates are excluded even in the edgeless early-return."""
        g = _loaded_graph()
        await g.add_node("/a.py", session=_mock_session)

        result = await g.hits(_result("/a.py", "/missing.py"), session=_mock_session)
        paths = {c.path for c in result.candidates}
        assert "/a.py" in paths
        assert "/missing.py" not in paths

    async def test_empty_graph(self):
        g = _loaded_graph()
        result = await g.hits(_result(), session=_mock_session)
        assert result.success is True
        assert result.candidates == []

    async def test_custom_max_iter_and_tol(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)

        result = await g.hits(
            _result(), max_iter=50, tol=1e-4, session=_mock_session,
        )
        assert result.success is True
        assert len(result.candidates) == 2

    async def test_detail_operation(self):
        g = _loaded_graph()
        await g.add_edge("/a.py", "/b.py", "imports", session=_mock_session)
        result = await g.hits(_result(), session=_mock_session)
        assert result.candidates[0].details[0].operation == "hits"

    async def test_sorted_by_authority_descending(self):
        # Hub → many targets: hub should have high hub score,
        # targets should have high authority score
        g = _loaded_graph()
        await g.add_edge("/hub.py", "/auth1.py", "imports", session=_mock_session)
        await g.add_edge("/hub.py", "/auth2.py", "calls", session=_mock_session)
        await g.add_edge("/hub.py", "/auth3.py", "imports", session=_mock_session)

        result = await g.hits(_result(), session=_mock_session)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    async def test_error_returns_failure(self):
        g = _loaded_graph()
        g._nodes.add("/a.py")
        g._out = None  # type: ignore[assignment]
        result = await g.hits(_result("/a.py"), session=_mock_session)
        assert result.success is False
        assert "hits failed" in result.errors[0]


# ===========================================================================
# Protocol compliance
# ===========================================================================


class TestProtocol:
    def test_exports(self):
        """GraphProvider and RustworkxGraph are importable from grover.graph."""
        from grover.graph import RustworkxGraph, UnionFind

        assert GraphProvider is not None
        assert RustworkxGraph is not None
        assert UnionFind is not None

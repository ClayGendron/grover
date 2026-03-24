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
# Protocol compliance
# ===========================================================================


class TestProtocol:
    def test_exports(self):
        """GraphProvider and RustworkxGraph are importable from grover.graph."""
        from grover.graph import RustworkxGraph, UnionFind

        assert GraphProvider is not None
        assert RustworkxGraph is not None
        assert UnionFind is not None

"""Tests for graph centrality, connectivity, and traversal algorithms."""

from __future__ import annotations

import pytest

from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider

# ======================================================================
# Centrality — PageRank
# ======================================================================


class TestPageRank:
    def test_basic_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.pagerank()
        assert len(result) == 3
        total = sum(c.evidence[0].score for c in result.file_candidates)
        assert abs(total - 1.0) < 0.01
        # Sink node (c) should have highest score in a chain
        assert result.explain("/c.py")[0].score > result.explain("/a.py")[0].score

    def test_personalized(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.pagerank(personalization={"/a.py": 1.0})
        assert len(result) == 3
        # Personalization biases toward /a.py's neighborhood
        assert result.explain("/a.py")[0].score > 0

    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert len(g.pagerank()) == 0

    def test_single_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        result = g.pagerank()
        assert len(result) == 1
        assert abs(result.explain("/a.py")[0].score - 1.0) < 0.01

    def test_nonexistent_personalization_key_ignored(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # /missing.py is not in the graph — should be silently skipped
        result = g.pagerank(personalization={"/missing.py": 1.0, "/a.py": 0.5})
        assert len(result) == 2

    def test_all_personalization_keys_missing(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # All keys missing — falls back to uniform personalization
        result = g.pagerank(personalization={"/missing.py": 1.0})
        assert len(result) == 2


# ======================================================================
# Centrality — Betweenness
# ======================================================================


class TestBetweennessCentrality:
    def test_bridge_node_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.betweenness_centrality()
        assert result.explain("/b.py")[0].score >= result.explain("/a.py")[0].score
        assert result.explain("/b.py")[0].score >= result.explain("/c.py")[0].score

    def test_no_edges(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        result = g.betweenness_centrality()
        assert result.explain("/a.py")[0].score == 0.0
        assert result.explain("/b.py")[0].score == 0.0


# ======================================================================
# Centrality — Closeness
# ======================================================================


class TestClosenessCentrality:
    def test_hub_highest(self) -> None:
        # Star graph with bidirectional edges — center is closest to all
        g = RustworkxGraph()
        g.add_edge("/center.py", "/a.py", "imports")
        g.add_edge("/a.py", "/center.py", "imports")
        g.add_edge("/center.py", "/b.py", "imports")
        g.add_edge("/b.py", "/center.py", "imports")
        g.add_edge("/center.py", "/c.py", "imports")
        g.add_edge("/c.py", "/center.py", "imports")
        result = g.closeness_centrality()
        assert result.explain("/center.py")[0].score >= result.explain("/a.py")[0].score


# ======================================================================
# Centrality — Katz
# ======================================================================


class TestKatzCentrality:
    def test_basic(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.katz_centrality()
        assert len(result) == 3
        assert all(c.evidence[0].score > 0 for c in result.file_candidates)

    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert len(g.katz_centrality()) == 0


# ======================================================================
# Centrality — Degree
# ======================================================================


class TestDegreeCentrality:
    def test_hub_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/hub.py", "/a.py", "imports")
        g.add_edge("/hub.py", "/b.py", "imports")
        g.add_edge("/c.py", "/hub.py", "imports")
        result = g.degree_centrality()
        assert result.explain("/hub.py")[0].score >= result.explain("/a.py")[0].score

    def test_in_vs_out(self) -> None:
        # Asymmetric: /a.py has 2 outgoing, /b.py has 2 incoming
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/d.py", "/b.py", "calls")
        in_result = g.in_degree_centrality()
        out_result = g.out_degree_centrality()
        # /b.py has 2 incoming edges
        assert in_result.explain("/b.py")[0].score > in_result.explain("/a.py")[0].score
        # /a.py has 2 outgoing edges
        assert out_result.explain("/a.py")[0].score > out_result.explain("/b.py")[0].score


# ======================================================================
# Connectivity
# ======================================================================


class TestConnectivity:
    def test_weakly_connected_single(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        components = g.weakly_connected_components()
        assert len(components) == 1
        assert components[0] == {"/a.py", "/b.py", "/c.py"}

    def test_weakly_connected_multiple(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_node("/c.py")
        components = g.weakly_connected_components()
        assert len(components) == 2

    def test_strongly_connected_cycle(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        components = g.strongly_connected_components()
        # The cycle {a, b} forms one SCC
        scc_sizes = sorted(len(c) for c in components)
        assert 2 in scc_sizes

    def test_strongly_connected_dag(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        components = g.strongly_connected_components()
        # In a DAG, each node is its own SCC
        assert all(len(c) == 1 for c in components)
        assert len(components) == 3

    def test_is_weakly_connected_true(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.is_weakly_connected() is True

    def test_is_weakly_connected_false(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.is_weakly_connected() is False

    def test_is_weakly_connected_empty(self) -> None:
        g = RustworkxGraph()
        # Empty graph — NullGraph handled, returns True
        assert g.is_weakly_connected() is True


# ======================================================================
# Traversal
# ======================================================================


class TestTraversal:
    def test_ancestors_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert set(g.ancestors("/c.py").paths) == {"/a.py", "/b.py"}

    def test_ancestors_no_parents(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert set(g.ancestors("/a.py").paths) == set()

    def test_ancestors_missing_node(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.ancestors("/missing.py")

    def test_descendants_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert set(g.descendants("/a.py").paths) == {"/b.py", "/c.py"}

    def test_descendants_leaf(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert set(g.descendants("/b.py").paths) == set()

    def test_all_simple_paths_diamond(self) -> None:
        # A->B->D and A->C->D: 2 distinct paths
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/b.py", "/d.py", "imports")
        g.add_edge("/c.py", "/d.py", "imports")
        paths = g.all_simple_paths("/a.py", "/d.py")
        assert len(paths) == 2
        path_sets = {tuple(p) for p in paths}
        assert ("/a.py", "/b.py", "/d.py") in path_sets
        assert ("/a.py", "/c.py", "/d.py") in path_sets

    def test_all_simple_paths_cutoff(self) -> None:
        # A->B->C->D, A->D (direct)
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        g.add_edge("/c.py", "/d.py", "imports")
        g.add_edge("/a.py", "/d.py", "imports")
        # With small cutoff, only short paths
        paths = g.all_simple_paths("/a.py", "/d.py", cutoff=2)
        # cutoff=2 limits path length — only the direct A->D path (2 nodes)
        assert len(paths) >= 1
        for p in paths:
            assert len(p) <= 2

    def test_all_simple_paths_no_path(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        paths = g.all_simple_paths("/a.py", "/b.py")
        assert paths == []

    def test_topological_sort_dag(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        order = g.topological_sort()
        assert len(order) == 3
        # All edges should go forward in the sort order
        idx = {path: i for i, path in enumerate(order)}
        assert idx["/a.py"] < idx["/b.py"]
        assert idx["/b.py"] < idx["/c.py"]

    def test_topological_sort_cycle(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/a.py", "imports")
        with pytest.raises(ValueError, match="Graph contains cycles"):
            g.topological_sort()

    def test_shortest_path_length_adjacent(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = g.shortest_path_length("/a.py", "/b.py")
        assert result == 1.0

    def test_shortest_path_length_multi_hop(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = g.shortest_path_length("/a.py", "/c.py")
        # Minimal storage — all edges are unit weight
        assert result == 2.0

    def test_shortest_path_length_no_path(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.shortest_path_length("/a.py", "/b.py") is None


# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

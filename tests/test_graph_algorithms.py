"""Tests for graph centrality, connectivity, and traversal algorithms."""

from __future__ import annotations

import pytest

from grover.graph import RustworkxGraph
from grover.graph.protocols import SupportsCentrality, SupportsConnectivity, SupportsTraversal

# ======================================================================
# Centrality — PageRank
# ======================================================================


class TestPageRank:
    def test_basic_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        scores = g.pagerank()
        assert len(scores) == 3
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01
        # Sink node (c) should have highest score in a chain
        assert scores["/c.py"] > scores["/a.py"]

    def test_personalized(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        scores = g.pagerank(personalization={"/a.py": 1.0})
        assert len(scores) == 3
        # Personalization biases toward /a.py's neighborhood
        assert scores["/a.py"] > 0

    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert g.pagerank() == {}

    def test_single_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        scores = g.pagerank()
        assert len(scores) == 1
        assert abs(scores["/a.py"] - 1.0) < 0.01

    def test_nonexistent_personalization_key_ignored(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # /missing.py is not in the graph — should be silently skipped
        scores = g.pagerank(personalization={"/missing.py": 1.0, "/a.py": 0.5})
        assert len(scores) == 2

    def test_all_personalization_keys_missing(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # All keys missing — falls back to uniform personalization
        scores = g.pagerank(personalization={"/missing.py": 1.0})
        assert len(scores) == 2


# ======================================================================
# Centrality — Betweenness
# ======================================================================


class TestBetweennessCentrality:
    def test_bridge_node_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        scores = g.betweenness_centrality()
        assert scores["/b.py"] >= scores["/a.py"]
        assert scores["/b.py"] >= scores["/c.py"]

    def test_no_edges(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        scores = g.betweenness_centrality()
        assert scores["/a.py"] == 0.0
        assert scores["/b.py"] == 0.0


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
        scores = g.closeness_centrality()
        assert scores["/center.py"] >= scores["/a.py"]


# ======================================================================
# Centrality — Katz
# ======================================================================


class TestKatzCentrality:
    def test_basic(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        scores = g.katz_centrality()
        assert len(scores) == 3
        assert all(v > 0 for v in scores.values())

    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        assert g.katz_centrality() == {}


# ======================================================================
# Centrality — Degree
# ======================================================================


class TestDegreeCentrality:
    def test_hub_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/hub.py", "/a.py", "imports")
        g.add_edge("/hub.py", "/b.py", "imports")
        g.add_edge("/c.py", "/hub.py", "imports")
        scores = g.degree_centrality()
        assert scores["/hub.py"] >= scores["/a.py"]

    def test_in_vs_out(self) -> None:
        # Asymmetric: /a.py has 2 outgoing, /b.py has 2 incoming
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/d.py", "/b.py", "calls")
        in_scores = g.in_degree_centrality()
        out_scores = g.out_degree_centrality()
        # /b.py has 2 incoming edges
        assert in_scores["/b.py"] > in_scores["/a.py"]
        # /a.py has 2 outgoing edges
        assert out_scores["/a.py"] > out_scores["/b.py"]


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
        assert g.ancestors("/c.py") == {"/a.py", "/b.py"}

    def test_ancestors_no_parents(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.ancestors("/a.py") == set()

    def test_ancestors_missing_node(self) -> None:
        g = RustworkxGraph()
        with pytest.raises(KeyError):
            g.ancestors("/missing.py")

    def test_descendants_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert g.descendants("/a.py") == {"/b.py", "/c.py"}

    def test_descendants_leaf(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert g.descendants("/b.py") == set()

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
        g.add_edge("/a.py", "/b.py", "imports", weight=2.0)
        g.add_edge("/b.py", "/c.py", "imports", weight=3.0)
        result = g.shortest_path_length("/a.py", "/c.py")
        assert result == 5.0

    def test_shortest_path_length_no_path(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.shortest_path_length("/a.py", "/b.py") is None


# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_centrality(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, SupportsCentrality)

    def test_supports_connectivity(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, SupportsConnectivity)

    def test_supports_traversal(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, SupportsTraversal)

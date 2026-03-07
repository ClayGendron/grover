"""Tests for node similarity."""

from __future__ import annotations

from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider

# ======================================================================
# node_similarity
# ======================================================================


class TestNodeSimilarity:
    def test_identical_neighbors(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/a.py", "/d.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        g.add_edge("/b.py", "/d.py", "imports")
        assert g.node_similarity("/a.py", "/b.py") == 1.0

    def test_disjoint_neighbors(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/x.py", "imports")
        g.add_edge("/b.py", "/y.py", "imports")
        assert g.node_similarity("/a.py", "/b.py") == 0.0

    def test_partial_overlap(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/a.py", "/d.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        g.add_edge("/b.py", "/e.py", "imports")
        sim = g.node_similarity("/a.py", "/b.py")
        assert abs(sim - 1 / 3) < 0.001

    def test_no_neighbors(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        assert g.node_similarity("/a.py", "/b.py") == 0.0


# ======================================================================
# similar_nodes
# ======================================================================


class TestSimilarNodes:
    def test_returns_top_k(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        g.add_node("/d.py")
        result = g.similar_nodes("/a.py", k=2)
        assert len(result) <= 2
        assert result[0][0] == "/b.py"
        assert result[0][1] > 0

    def test_self_excluded(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        result = g.similar_nodes("/a.py")
        paths = [r[0] for r in result]
        assert "/a.py" not in paths

    def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        result = g.similar_nodes("/a.py")
        assert result == []


# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

"""Tests for graph centrality, connectivity, and traversal algorithms."""

from __future__ import annotations

from unittest.mock import AsyncMock

from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider

_session = AsyncMock()

# ======================================================================
# Helper
# ======================================================================


def _score_for(result, path: str) -> float:
    """Return the first evidence score for *path* in a FileSearchResult."""
    for f in result.files:
        if f.path == path:
            return f.evidence[0].score if f.evidence else 0.0
    raise KeyError(f"Path not found in result: {path!r}")


def _paths(*paths: str) -> FileSearchSet:
    return FileSearchSet.from_paths(list(paths))


# ======================================================================
# Centrality — PageRank
# ======================================================================


class TestPageRank:
    async def test_basic_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = await g.pagerank(FileSearchSet(), session=_session)
        assert len(result) == 3
        total = sum(f.evidence[0].score for f in result.files)
        assert abs(total - 1.0) < 0.01
        # Sink node (c) should have highest score in a chain
        assert _score_for(result, "/c.py") > _score_for(result, "/a.py")

    async def test_personalized(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = await g.pagerank(FileSearchSet(), personalization={"/a.py": 1.0}, session=_session)
        assert len(result) == 3
        # Personalization biases toward /a.py's neighborhood
        assert _score_for(result, "/a.py") > 0

    async def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 1.0  # prevent _ensure_fresh from calling from_sql
        assert len(await g.pagerank(FileSearchSet(), session=_session)) == 0

    async def test_single_node(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        result = await g.pagerank(FileSearchSet(), session=_session)
        assert len(result) == 1
        assert abs(_score_for(result, "/a.py") - 1.0) < 0.01

    async def test_nonexistent_personalization_key_ignored(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # /missing.py is not in the graph — should be silently skipped
        result = await g.pagerank(FileSearchSet(), personalization={"/missing.py": 1.0, "/a.py": 0.5}, session=_session)
        assert len(result) == 2

    async def test_all_personalization_keys_missing(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        # All keys missing — falls back to uniform personalization
        result = await g.pagerank(FileSearchSet(), personalization={"/missing.py": 1.0}, session=_session)
        assert len(result) == 2


# ======================================================================
# Centrality — Betweenness
# ======================================================================


class TestBetweennessCentrality:
    async def test_bridge_node_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = await g.betweenness_centrality(FileSearchSet(), session=_session)
        assert _score_for(result, "/b.py") >= _score_for(result, "/a.py")
        assert _score_for(result, "/b.py") >= _score_for(result, "/c.py")

    async def test_no_edges(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        result = await g.betweenness_centrality(FileSearchSet(), session=_session)
        assert _score_for(result, "/a.py") == 0.0
        assert _score_for(result, "/b.py") == 0.0


# ======================================================================
# Centrality — Closeness
# ======================================================================


class TestClosenessCentrality:
    async def test_hub_highest(self) -> None:
        # Star graph with bidirectional edges — center is closest to all
        g = RustworkxGraph()
        g.add_edge("/center.py", "/a.py", "imports")
        g.add_edge("/a.py", "/center.py", "imports")
        g.add_edge("/center.py", "/b.py", "imports")
        g.add_edge("/b.py", "/center.py", "imports")
        g.add_edge("/center.py", "/c.py", "imports")
        g.add_edge("/c.py", "/center.py", "imports")
        result = await g.closeness_centrality(FileSearchSet(), session=_session)
        assert _score_for(result, "/center.py") >= _score_for(result, "/a.py")


# ======================================================================
# Centrality — Katz
# ======================================================================


class TestKatzCentrality:
    async def test_basic(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        result = await g.katz_centrality(FileSearchSet(), session=_session)
        assert len(result) == 3
        assert all(f.evidence[0].score > 0 for f in result.files)

    async def test_empty_graph(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 1.0  # prevent _ensure_fresh from calling from_sql
        assert len(await g.katz_centrality(FileSearchSet(), session=_session)) == 0


# ======================================================================
# Centrality — Degree
# ======================================================================


class TestDegreeCentrality:
    async def test_hub_highest(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/hub.py", "/a.py", "imports")
        g.add_edge("/hub.py", "/b.py", "imports")
        g.add_edge("/c.py", "/hub.py", "imports")
        result = await g.degree_centrality(FileSearchSet(), session=_session)
        assert _score_for(result, "/hub.py") >= _score_for(result, "/a.py")

    async def test_in_vs_out(self) -> None:
        # Asymmetric: /a.py has 2 outgoing, /b.py has 2 incoming
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/d.py", "/b.py", "calls")
        in_result = await g.in_degree_centrality(FileSearchSet(), session=_session)
        out_result = await g.out_degree_centrality(FileSearchSet(), session=_session)
        # /b.py has 2 incoming edges
        assert _score_for(in_result, "/b.py") > _score_for(in_result, "/a.py")
        # /a.py has 2 outgoing edges
        assert _score_for(out_result, "/a.py") > _score_for(out_result, "/b.py")


# ======================================================================
# Traversal
# ======================================================================


class TestTraversal:
    async def test_ancestors_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert set((await g.ancestors(_paths("/c.py"), session=_session)).paths) == {"/a.py", "/b.py"}

    async def test_ancestors_no_parents(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert set((await g.ancestors(_paths("/a.py"), session=_session)).paths) == set()

    async def test_ancestors_unknown_returns_empty(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 1.0  # prevent _ensure_fresh from calling from_sql
        result = await g.ancestors(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0

    async def test_descendants_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        assert set((await g.descendants(_paths("/a.py"), session=_session)).paths) == {"/b.py", "/c.py"}

    async def test_descendants_leaf(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        assert set((await g.descendants(_paths("/b.py"), session=_session)).paths) == set()


# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

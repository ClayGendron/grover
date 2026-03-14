"""Tests for neighborhood, meeting subgraph.

Subgraph, common_reachable, and common_neighbors have been removed from the API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider

_session = AsyncMock()


def _paths(*paths: str) -> FileSearchSet:
    return FileSearchSet.from_paths(list(paths))


# ======================================================================
# Neighborhood
# ======================================================================


class TestNeighborhood:
    async def test_depth_1(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        sub = await g.neighborhood(_paths("/a.py"), max_depth=1, session=_session)
        # Undirected BFS: /a.py plus its immediate neighbors
        assert "/a.py" in sub.paths
        assert "/b.py" in sub.paths

    async def test_depth_2(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        sub = await g.neighborhood(_paths("/a.py"), max_depth=2, session=_session)
        assert set(sub.paths) == {"/a.py", "/b.py", "/c.py"}

    async def test_both_directions(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        sub = await g.neighborhood(_paths("/b.py"), max_depth=1, session=_session)
        # Undirected BFS: both in and out neighbors
        assert set(sub.paths) == {"/a.py", "/b.py", "/c.py"}

    async def test_unknown_node_returns_empty(self) -> None:
        g = RustworkxGraph()
        g.loaded_at = 0  # Prevent _ensure_fresh from calling from_sql on mock
        result = await g.neighborhood(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0

    async def test_depth_0(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        sub = await g.neighborhood(_paths("/a.py"), max_depth=0, session=_session)
        assert set(sub.paths) == {"/a.py"}


# ======================================================================
# Meeting subgraph
# ======================================================================


class TestMeetingSubgraph:
    async def test_linear_chain(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        g.add_edge("/c.py", "/d.py", "imports")
        sub = await g.meeting_subgraph(_paths("/a.py", "/d.py"), session=_session)
        # Should include intermediate nodes B and C
        assert "/b.py" in sub.paths
        assert "/c.py" in sub.paths

    async def test_diamond(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/a.py", "/c.py", "imports")
        g.add_edge("/b.py", "/d.py", "imports")
        g.add_edge("/c.py", "/d.py", "imports")
        sub = await g.meeting_subgraph(_paths("/a.py", "/d.py"), session=_session)
        assert "/a.py" in sub.paths
        assert "/d.py" in sub.paths

    async def test_disconnected(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        g.add_node("/b.py")
        sub = await g.meeting_subgraph(_paths("/a.py", "/b.py"), session=_session)
        # No connection — should at least contain start nodes
        assert "/a.py" in sub.paths
        assert "/b.py" in sub.paths

    async def test_single_start(self) -> None:
        g = RustworkxGraph()
        g.add_node("/a.py")
        sub = await g.meeting_subgraph(_paths("/a.py"), session=_session)
        assert set(sub.paths) == {"/a.py"}
        assert sub.connections == []

    async def test_three_starts(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/hub.py", "imports")
        g.add_edge("/b.py", "/hub.py", "imports")
        g.add_edge("/c.py", "/hub.py", "imports")
        sub = await g.meeting_subgraph(_paths("/a.py", "/b.py", "/c.py"), session=_session)
        # Hub should be found via common descendants
        assert len(sub) >= 3

    async def test_scores_populated(self) -> None:
        g = RustworkxGraph()
        g.add_edge("/a.py", "/b.py", "imports")
        g.add_edge("/b.py", "/c.py", "imports")
        sub = await g.meeting_subgraph(_paths("/a.py", "/c.py"), session=_session)
        # Every file should have a score from the enrichment
        assert len(sub) > 0
        for f in sub.files:
            assert len(f.evidence) > 0
            assert isinstance(f.evidence[0].score, float)


# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

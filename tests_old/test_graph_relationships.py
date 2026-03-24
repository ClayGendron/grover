"""Tests for graph relationship queries — predecessors, successors, ancestors, descendants.

Validates that:
- Results use GraphRelationshipEvidence with correct paths
- Multi-path candidates produce correct evidence (which candidates each result relates to)
- Candidate nodes are excluded from results
- Empty/missing inputs return empty results
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from grover.models.internal.evidence import GraphRelationshipEvidence
from grover.models.internal.results import FileSearchSet
from grover.providers.graph import RustworkxGraph

_session = AsyncMock()


def _paths(*paths: str) -> FileSearchSet:
    return FileSearchSet.from_paths(list(paths))


def _diamond() -> RustworkxGraph:
    """Build a diamond graph: a -> b, a -> c, b -> d, c -> d."""
    g = RustworkxGraph()
    g.add_edge("/a.py", "/b.py", "imports")
    g.add_edge("/a.py", "/c.py", "imports")
    g.add_edge("/b.py", "/d.py", "imports")
    g.add_edge("/c.py", "/d.py", "imports")
    return g


def _chain() -> RustworkxGraph:
    """Build a chain: a -> b -> c -> d -> e."""
    g = RustworkxGraph()
    g.add_edge("/a.py", "/b.py", "imports")
    g.add_edge("/b.py", "/c.py", "imports")
    g.add_edge("/c.py", "/d.py", "imports")
    g.add_edge("/d.py", "/e.py", "imports")
    return g


def _fan_in() -> RustworkxGraph:
    """Build a fan-in: a -> d, b -> d, c -> d."""
    g = RustworkxGraph()
    g.add_edge("/a.py", "/d.py", "imports")
    g.add_edge("/b.py", "/d.py", "imports")
    g.add_edge("/c.py", "/d.py", "imports")
    return g


# ======================================================================
# Predecessors
# ======================================================================


class TestPredecessors:
    async def test_single_candidate(self) -> None:
        g = _diamond()
        result = await g.predecessors(_paths("/d.py"), session=_session)
        assert set(result.paths) == {"/b.py", "/c.py"}

    async def test_multi_candidate_union(self) -> None:
        g = _diamond()
        result = await g.predecessors(_paths("/b.py", "/c.py"), session=_session)
        # /a.py points to both /b.py and /c.py
        assert set(result.paths) == {"/a.py"}

    async def test_evidence_tracks_which_candidates(self) -> None:
        g = _diamond()
        result = await g.predecessors(_paths("/b.py", "/c.py"), session=_session)
        a_file = next(f for f in result.files if f.path == "/a.py")
        assert len(a_file.evidence) == 1
        ev = a_file.evidence[0]
        assert isinstance(ev, GraphRelationshipEvidence)
        assert ev.operation == "predecessors"
        # /a.py is a predecessor of both /b.py and /c.py
        assert sorted(ev.paths) == ["/b.py", "/c.py"]

    async def test_evidence_single_path(self) -> None:
        g = _diamond()
        result = await g.predecessors(_paths("/b.py"), session=_session)
        a_file = next(f for f in result.files if f.path == "/a.py")
        ev = a_file.evidence[0]
        assert isinstance(ev, GraphRelationshipEvidence)
        assert ev.paths == ["/b.py"]

    async def test_excludes_candidates(self) -> None:
        g = _diamond()
        # /a.py -> /b.py, so /a.py is a predecessor of /b.py
        # but if /a.py is also a candidate, it should be excluded
        result = await g.predecessors(_paths("/a.py", "/b.py"), session=_session)
        assert "/a.py" not in result.paths

    async def test_empty_when_no_incoming(self) -> None:
        g = _diamond()
        result = await g.predecessors(_paths("/a.py"), session=_session)
        assert len(result) == 0

    async def test_unknown_path_returns_empty(self) -> None:
        g = _diamond()
        g._loaded_at = 0.0
        result = await g.predecessors(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0

    async def test_fan_in_multi_candidate(self) -> None:
        """Multiple candidates share the same predecessor with different paths."""
        g = RustworkxGraph()
        g.add_edge("/x.py", "/a.py", "imports")
        g.add_edge("/x.py", "/b.py", "imports")
        g.add_edge("/y.py", "/b.py", "imports")
        g.add_edge("/y.py", "/c.py", "imports")

        result = await g.predecessors(_paths("/a.py", "/b.py", "/c.py"), session=_session)
        paths_dict = {f.path: f.evidence[0] for f in result.files}

        # /x.py is predecessor of /a.py and /b.py
        assert sorted(paths_dict["/x.py"].paths) == ["/a.py", "/b.py"]
        # /y.py is predecessor of /b.py and /c.py
        assert sorted(paths_dict["/y.py"].paths) == ["/b.py", "/c.py"]


# ======================================================================
# Successors
# ======================================================================


class TestSuccessors:
    async def test_single_candidate(self) -> None:
        g = _diamond()
        result = await g.successors(_paths("/a.py"), session=_session)
        assert set(result.paths) == {"/b.py", "/c.py"}

    async def test_multi_candidate_union(self) -> None:
        g = _diamond()
        result = await g.successors(_paths("/b.py", "/c.py"), session=_session)
        # Both /b.py and /c.py point to /d.py
        assert set(result.paths) == {"/d.py"}

    async def test_evidence_tracks_which_candidates(self) -> None:
        g = _diamond()
        result = await g.successors(_paths("/b.py", "/c.py"), session=_session)
        d_file = next(f for f in result.files if f.path == "/d.py")
        ev = d_file.evidence[0]
        assert isinstance(ev, GraphRelationshipEvidence)
        assert ev.operation == "successors"
        # /d.py is a successor of both /b.py and /c.py
        assert sorted(ev.paths) == ["/b.py", "/c.py"]

    async def test_excludes_candidates(self) -> None:
        g = _diamond()
        result = await g.successors(_paths("/b.py", "/d.py"), session=_session)
        assert "/d.py" not in result.paths

    async def test_empty_when_no_outgoing(self) -> None:
        g = _diamond()
        result = await g.successors(_paths("/d.py"), session=_session)
        assert len(result) == 0

    async def test_unknown_path_returns_empty(self) -> None:
        g = _diamond()
        g._loaded_at = 0.0
        result = await g.successors(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0


# ======================================================================
# Ancestors
# ======================================================================


class TestAncestors:
    async def test_single_candidate(self) -> None:
        g = _chain()
        result = await g.ancestors(_paths("/d.py"), session=_session)
        assert set(result.paths) == {"/a.py", "/b.py", "/c.py"}

    async def test_multi_candidate_union(self) -> None:
        g = _chain()
        result = await g.ancestors(_paths("/c.py", "/e.py"), session=_session)
        # ancestors of /c.py = {/a.py, /b.py}
        # ancestors of /e.py = {/a.py, /b.py, /c.py, /d.py}
        # /c.py excluded (candidate), union = {/a.py, /b.py, /d.py}
        assert set(result.paths) == {"/a.py", "/b.py", "/d.py"}

    async def test_evidence_tracks_which_candidates(self) -> None:
        g = _chain()
        result = await g.ancestors(_paths("/c.py", "/e.py"), session=_session)

        # /a.py is an ancestor of both /c.py and /e.py
        a_file = next(f for f in result.files if f.path == "/a.py")
        ev = a_file.evidence[0]
        assert isinstance(ev, GraphRelationshipEvidence)
        assert ev.operation == "ancestors"
        assert sorted(ev.paths) == ["/c.py", "/e.py"]

        # /d.py is an ancestor of /e.py only (not /c.py)
        d_file = next(f for f in result.files if f.path == "/d.py")
        ev = d_file.evidence[0]
        assert ev.paths == ["/e.py"]

    async def test_excludes_candidates(self) -> None:
        g = _chain()
        result = await g.ancestors(_paths("/b.py", "/d.py"), session=_session)
        assert "/b.py" not in result.paths
        assert "/d.py" not in result.paths

    async def test_empty_when_root(self) -> None:
        g = _chain()
        result = await g.ancestors(_paths("/a.py"), session=_session)
        assert len(result) == 0

    async def test_unknown_path_returns_empty(self) -> None:
        g = _chain()
        g._loaded_at = 0.0
        result = await g.ancestors(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0

    async def test_diamond_shared_ancestor(self) -> None:
        """In a diamond, /a.py is an ancestor of both /b.py and /c.py."""
        g = _diamond()
        result = await g.ancestors(_paths("/b.py", "/c.py"), session=_session)
        a_file = next(f for f in result.files if f.path == "/a.py")
        ev = a_file.evidence[0]
        assert sorted(ev.paths) == ["/b.py", "/c.py"]


# ======================================================================
# Descendants
# ======================================================================


class TestDescendants:
    async def test_single_candidate(self) -> None:
        g = _chain()
        result = await g.descendants(_paths("/b.py"), session=_session)
        assert set(result.paths) == {"/c.py", "/d.py", "/e.py"}

    async def test_multi_candidate_union(self) -> None:
        g = _chain()
        result = await g.descendants(_paths("/a.py", "/c.py"), session=_session)
        # descendants of /a.py = {/b.py, /c.py, /d.py, /e.py}
        # descendants of /c.py = {/d.py, /e.py}
        # /c.py excluded (candidate), union = {/b.py, /d.py, /e.py}
        assert set(result.paths) == {"/b.py", "/d.py", "/e.py"}

    async def test_evidence_tracks_which_candidates(self) -> None:
        g = _chain()
        result = await g.descendants(_paths("/a.py", "/c.py"), session=_session)

        # /e.py is a descendant of both /a.py and /c.py
        e_file = next(f for f in result.files if f.path == "/e.py")
        ev = e_file.evidence[0]
        assert isinstance(ev, GraphRelationshipEvidence)
        assert ev.operation == "descendants"
        assert sorted(ev.paths) == ["/a.py", "/c.py"]

        # /b.py is a descendant of /a.py only
        b_file = next(f for f in result.files if f.path == "/b.py")
        ev = b_file.evidence[0]
        assert ev.paths == ["/a.py"]

    async def test_excludes_candidates(self) -> None:
        g = _chain()
        result = await g.descendants(_paths("/b.py", "/d.py"), session=_session)
        assert "/b.py" not in result.paths
        assert "/d.py" not in result.paths

    async def test_empty_when_leaf(self) -> None:
        g = _chain()
        result = await g.descendants(_paths("/e.py"), session=_session)
        assert len(result) == 0

    async def test_unknown_path_returns_empty(self) -> None:
        g = _chain()
        g._loaded_at = 0.0
        result = await g.descendants(_paths("/missing.py"), session=_session)
        assert result.success
        assert len(result) == 0

    async def test_diamond_shared_descendant(self) -> None:
        """In a diamond, /d.py is a descendant of both /b.py and /c.py."""
        g = _diamond()
        result = await g.descendants(_paths("/b.py", "/c.py"), session=_session)
        d_file = next(f for f in result.files if f.path == "/d.py")
        ev = d_file.evidence[0]
        assert sorted(ev.paths) == ["/b.py", "/c.py"]

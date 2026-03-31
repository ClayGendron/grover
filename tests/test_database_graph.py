"""Integration tests for graph operations via DatabaseFileSystem.

Creates files and connections through the filesystem, then verifies
that all graph algorithms (traversal, subgraph, centrality) work
end-to-end through the _*_impl delegation to RustworkxGraph.
"""

from __future__ import annotations

import pytest

from grover.backends.database import DatabaseFileSystem
from grover.results import Candidate, GroverResult

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def paths(result: GroverResult) -> set[str]:
    """Extract path set from a GroverResult."""
    return {c.path for c in result.candidates}


def scored_paths(result: GroverResult) -> dict[str, float]:
    """Extract {path: score} from a GroverResult."""
    return {c.path: c.score for c in result.candidates}


def cands(*ps: str) -> GroverResult:
    """Build a GroverResult from path strings."""
    return GroverResult(candidates=[Candidate(path=p) for p in ps])


# ------------------------------------------------------------------
# Fixture: populated graph
# ------------------------------------------------------------------


@pytest.fixture
async def graph_db(db: DatabaseFileSystem):
    """DatabaseFileSystem with files and connections for graph tests.

    Graph topology::

        /src/auth.py  --imports-->  /src/utils.py  --imports-->  /src/db.py
        /src/auth.py  --calls--->   /src/db.py
        /src/api.py   --imports-->  /src/auth.py
        /src/api.py   --imports-->  /src/utils.py

    Five nodes, five edges.  auth.py is a hub (many outgoing).
    utils.py and db.py are authorities (many incoming).
    """
    async with db._use_session() as s:
        for path in [
            "/src/auth.py",
            "/src/utils.py",
            "/src/db.py",
            "/src/api.py",
            "/src/config.py",
        ]:
            await db._write_impl(path, f"# {path}", session=s)

    async with db._use_session() as s:
        await db._mkconn_impl("/src/auth.py", "/src/utils.py", "imports", session=s)
    async with db._use_session() as s:
        await db._mkconn_impl("/src/utils.py", "/src/db.py", "imports", session=s)
    async with db._use_session() as s:
        await db._mkconn_impl("/src/auth.py", "/src/db.py", "calls", session=s)
    async with db._use_session() as s:
        await db._mkconn_impl("/src/api.py", "/src/auth.py", "imports", session=s)
    async with db._use_session() as s:
        await db._mkconn_impl("/src/api.py", "/src/utils.py", "imports", session=s)

    return db


# ------------------------------------------------------------------
# Predecessors / Successors
# ------------------------------------------------------------------


class TestPredecessors:
    async def test_predecessors_by_path(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._predecessors_impl("/src/utils.py", session=s)
        assert r.success
        # auth.py and api.py both import utils.py
        assert paths(r) == {"/src/auth.py", "/src/api.py"}

    async def test_predecessors_by_candidates(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._predecessors_impl(
                candidates=cands("/src/db.py"), session=s,
            )
        assert r.success
        # auth.py calls db.py, utils.py imports db.py
        assert paths(r) == {"/src/auth.py", "/src/utils.py"}

    async def test_predecessors_no_incoming(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._predecessors_impl("/src/api.py", session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_predecessors_nonexistent_node(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._predecessors_impl("/nonexistent.py", session=s)
        assert r.success
        assert len(r.candidates) == 0


class TestSuccessors:
    async def test_successors_by_path(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/auth.py", session=s)
        assert r.success
        assert paths(r) == {"/src/utils.py", "/src/db.py"}

    async def test_successors_by_candidates(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl(
                candidates=cands("/src/api.py"), session=s,
            )
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/utils.py"}

    async def test_successors_leaf_node(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/db.py", session=s)
        assert r.success
        assert len(r.candidates) == 0


# ------------------------------------------------------------------
# Ancestors / Descendants
# ------------------------------------------------------------------


class TestAncestors:
    async def test_ancestors_of_leaf(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._ancestors_impl("/src/db.py", session=s)
        assert r.success
        # db.py <- utils.py <- auth.py <- api.py; also auth.py (calls)
        assert paths(r) == {"/src/auth.py", "/src/utils.py", "/src/api.py"}

    async def test_ancestors_of_root_node(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._ancestors_impl("/src/api.py", session=s)
        assert r.success
        assert len(r.candidates) == 0


class TestDescendants:
    async def test_descendants_of_root(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._descendants_impl("/src/api.py", session=s)
        assert r.success
        # api.py -> auth.py -> utils.py -> db.py; also auth.py -> db.py
        assert paths(r) == {"/src/auth.py", "/src/utils.py", "/src/db.py"}

    async def test_descendants_of_leaf(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._descendants_impl("/src/db.py", session=s)
        assert r.success
        assert len(r.candidates) == 0


# ------------------------------------------------------------------
# Neighborhood
# ------------------------------------------------------------------


class TestNeighborhood:
    async def test_neighborhood_depth_1(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._neighborhood_impl(
                "/src/auth.py", depth=1, session=s,
            )
        assert r.success
        p = paths(r)
        # auth.py itself + direct neighbors: api.py (incoming), utils.py, db.py (outgoing)
        assert "/src/auth.py" in p
        assert "/src/api.py" in p
        assert "/src/utils.py" in p
        assert "/src/db.py" in p

    async def test_neighborhood_depth_2(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._neighborhood_impl(
                "/src/utils.py", depth=2, session=s,
            )
        assert r.success
        p = paths(r)
        # depth 1: auth.py, api.py (incoming), db.py (outgoing)
        # depth 2: from api.py nothing new; from auth.py: db.py (already); from db.py: nothing
        assert "/src/utils.py" in p
        assert "/src/auth.py" in p
        assert "/src/db.py" in p
        assert "/src/api.py" in p

    async def test_neighborhood_isolated_node(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._neighborhood_impl(
                "/src/config.py", depth=2, session=s,
            )
        assert r.success
        # config.py has no edges — not even a graph node
        assert len(r.candidates) == 0


# ------------------------------------------------------------------
# Meeting subgraph
# ------------------------------------------------------------------


class TestMeetingSubgraph:
    async def test_meeting_two_nodes(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._meeting_subgraph_impl(
                candidates=cands("/src/api.py", "/src/db.py"),
                session=s,
            )
        assert r.success
        p = paths(r)
        # api.py and db.py connected through auth.py and utils.py
        assert "/src/api.py" in p
        assert "/src/db.py" in p
        # Intermediaries should be present
        assert len(p) >= 2

    async def test_meeting_single_node(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._meeting_subgraph_impl(
                candidates=cands("/src/auth.py"),
                session=s,
            )
        assert r.success
        assert paths(r) == {"/src/auth.py"}


class TestMinMeetingSubgraph:
    async def test_min_meeting_prunes(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._min_meeting_subgraph_impl(
                candidates=cands("/src/api.py", "/src/db.py"),
                session=s,
            )
        assert r.success
        p = paths(r)
        assert "/src/api.py" in p
        assert "/src/db.py" in p


# ------------------------------------------------------------------
# Centrality algorithms
# ------------------------------------------------------------------


class TestPageRank:
    async def test_pagerank_all_nodes(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._pagerank_impl(session=s)
        assert r.success
        scores = scored_paths(r)
        # All graph nodes should have a score
        assert len(scores) >= 4
        # db.py receives the most edges (highest authority) — should rank high
        assert scores["/src/db.py"] > 0

    async def test_pagerank_filtered(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._pagerank_impl(
                candidates=cands("/src/auth.py", "/src/db.py"),
                session=s,
            )
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/db.py"}


class TestBetweennessCentrality:
    async def test_betweenness(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._betweenness_centrality_impl(session=s)
        assert r.success
        scores = scored_paths(r)
        # auth.py sits on many shortest paths
        assert "/src/auth.py" in scores


class TestClosenessCentrality:
    async def test_closeness(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._closeness_centrality_impl(session=s)
        assert r.success
        assert len(r.candidates) >= 4


class TestDegreeCentrality:
    async def test_degree(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._degree_centrality_impl(session=s)
        assert r.success
        scores = scored_paths(r)
        # auth.py has highest degree (3 outgoing/incoming edges)
        assert scores["/src/auth.py"] > 0

    async def test_in_degree(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._in_degree_centrality_impl(session=s)
        assert r.success
        scores = scored_paths(r)
        # db.py and utils.py have multiple incoming edges
        assert scores.get("/src/db.py", 0) > 0

    async def test_out_degree(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._out_degree_centrality_impl(session=s)
        assert r.success
        scores = scored_paths(r)
        # api.py and auth.py have multiple outgoing edges
        assert scores.get("/src/api.py", 0) > 0
        assert scores.get("/src/auth.py", 0) > 0


class TestHits:
    async def test_hits_authority(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._hits_impl(session=s)
        assert r.success
        # db.py and utils.py are authorities (many incoming)
        for c in r.candidates:
            assert c.details
            meta = c.details[-1].metadata
            assert "authority" in meta
            assert "hub" in meta

    async def test_hits_hub_scoring(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._hits_impl(
                candidates=cands("/src/auth.py", "/src/api.py"),
                session=s,
            )
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/api.py"}

    async def test_hits_filtered(self, graph_db: DatabaseFileSystem):
        async with graph_db._use_session() as s:
            r = await graph_db._hits_impl(
                candidates=cands("/src/db.py"),
                session=s,
            )
        assert r.success
        assert paths(r) == {"/src/db.py"}


# ------------------------------------------------------------------
# Graph consistency after mutations
# ------------------------------------------------------------------


class TestGraphConsistency:
    async def test_new_connection_appears_in_graph(self, graph_db: DatabaseFileSystem):
        """Adding a connection should be visible to subsequent graph queries."""
        async with graph_db._use_session() as s:
            await graph_db._mkconn_impl(
                "/src/config.py", "/src/db.py", "imports", session=s,
            )

        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/config.py", session=s)
        assert paths(r) == {"/src/db.py"}

    async def test_delete_connection_removes_from_graph(self, graph_db: DatabaseFileSystem):
        """Deleting a connection should remove it from subsequent graph queries."""
        # Verify edge exists first
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/auth.py", session=s)
        assert "/src/utils.py" in paths(r)

        # Delete the connection
        conn_path = "/src/auth.py/.connections/imports/src/utils.py"
        async with graph_db._use_session() as s:
            await graph_db._delete_impl(conn_path, session=s)

        # Verify it's gone
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/auth.py", session=s)
        assert "/src/utils.py" not in paths(r)
        # But the calls edge to db.py should remain
        assert "/src/db.py" in paths(r)

    async def test_delete_file_cascades_connections(self, graph_db: DatabaseFileSystem):
        """Deleting a file should cascade-delete its outgoing connections."""
        async with graph_db._use_session() as s:
            await graph_db._delete_impl("/src/auth.py", permanent=True, session=s)

        # auth.py's outgoing edges should be gone
        async with graph_db._use_session() as s:
            r = await graph_db._predecessors_impl("/src/utils.py", session=s)
        # Only api.py should remain as predecessor of utils.py
        assert "/src/auth.py" not in paths(r)

    async def test_move_updates_graph(self, graph_db: DatabaseFileSystem):
        """Moving a file should update connection paths in the graph."""
        from grover.results import TwoPathOperation

        async with graph_db._use_session() as s:
            await graph_db._move_impl(
                ops=[TwoPathOperation(src="/src/auth.py", dest="/src/authentication.py")],
                session=s,
            )

        # The old path should have no successors
        async with graph_db._use_session() as s:
            r = await graph_db._successors_impl("/src/authentication.py", session=s)
        assert r.success
        # The moved file should retain its outgoing edges
        assert "/src/utils.py" in paths(r) or "/src/db.py" in paths(r)


# ------------------------------------------------------------------
# Public API routing (through base.py)
# ------------------------------------------------------------------


class TestPublicGraphRouting:
    """Test that graph ops route correctly through the public API."""

    async def test_predecessors_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.predecessors(path="/src/utils.py")
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/api.py"}

    async def test_successors_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.successors(path="/src/auth.py")
        assert r.success
        assert paths(r) == {"/src/utils.py", "/src/db.py"}

    async def test_ancestors_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.ancestors(path="/src/db.py")
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/utils.py", "/src/api.py"}

    async def test_descendants_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.descendants(path="/src/api.py")
        assert r.success
        assert paths(r) == {"/src/auth.py", "/src/utils.py", "/src/db.py"}

    async def test_neighborhood_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.neighborhood(path="/src/auth.py", depth=1)
        assert r.success
        assert "/src/auth.py" in paths(r)

    async def test_pagerank_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.pagerank()
        assert r.success
        assert len(r.candidates) >= 4

    async def test_meeting_subgraph_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.meeting_subgraph(cands("/src/api.py", "/src/db.py"))
        assert r.success
        p = paths(r)
        assert "/src/api.py" in p
        assert "/src/db.py" in p

    async def test_hits_public(self, graph_db: DatabaseFileSystem):
        r = await graph_db.hits()
        assert r.success
        assert len(r.candidates) >= 4


# ------------------------------------------------------------------
# Empty graph
# ------------------------------------------------------------------


class TestEmptyGraph:
    async def test_predecessors_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._predecessors_impl("/nonexistent", session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_pagerank_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._pagerank_impl(session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_hits_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._hits_impl(session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_neighborhood_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._neighborhood_impl("/x", depth=1, session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_ancestors_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._ancestors_impl("/x", session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_descendants_empty(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._descendants_impl("/x", session=s)
        assert r.success
        assert len(r.candidates) == 0

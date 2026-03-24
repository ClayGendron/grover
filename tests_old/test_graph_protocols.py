"""Tests for graph protocols, SubgraphResult, and import migration."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider
from grover.providers.graph.types import SubgraphResult, subgraph_result

# ======================================================================
# GraphProvider protocol
# ======================================================================


class TestGraphProviderProtocol:
    def test_rustworkx_satisfies_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

    def test_graph_provider_is_runtime_checkable(self) -> None:
        assert isinstance(RustworkxGraph(), GraphProvider)
        assert not isinstance(object(), GraphProvider)


# ======================================================================
# SupportsPersistence protocol
# ======================================================================


class TestGraphProviderIncludesPersistence:
    def test_rustworkx_has_from_sql(self) -> None:
        g = RustworkxGraph()
        assert hasattr(g, "from_sql")

    def test_to_sql_removed(self) -> None:
        g = RustworkxGraph()
        assert not hasattr(g, "to_sql")


# ======================================================================
# SubgraphResult immutability
# ======================================================================


class TestSubgraphResult:
    def test_frozen(self) -> None:
        sr = SubgraphResult(nodes=("/a.py",), edges=())
        with pytest.raises(AttributeError):
            sr.nodes = ("/b.py",)  # type: ignore[misc]

    def test_nodes_is_tuple(self) -> None:
        sr = SubgraphResult(nodes=("/a.py", "/b.py"), edges=())
        assert isinstance(sr.nodes, tuple)

    def test_edges_is_tuple(self) -> None:
        sr = SubgraphResult(
            nodes=("/a.py",),
            edges=(("/a.py", "/b.py", {"type": "imports"}),),
        )
        assert isinstance(sr.edges, tuple)

    def test_scores_is_mapping_proxy(self) -> None:
        sr = SubgraphResult(
            nodes=("/a.py",),
            edges=(),
            scores=MappingProxyType({"/a.py": 0.5}),
        )
        assert isinstance(sr.scores, MappingProxyType)
        with pytest.raises(TypeError):
            sr.scores["/b.py"] = 0.3  # type: ignore[index]

    def test_defaults(self) -> None:
        sr = SubgraphResult(nodes=(), edges=())
        assert sr.scores == MappingProxyType({})
        assert len(sr.scores) == 0

    def test_creation_via_factory(self) -> None:
        sr = subgraph_result(
            nodes=["/a.py", "/b.py"],
            edges=[("/a.py", "/b.py", {"type": "imports"})],
            scores={"/a.py": 0.7},
        )
        assert isinstance(sr.nodes, tuple)
        assert isinstance(sr.edges, tuple)
        assert isinstance(sr.scores, MappingProxyType)
        assert sr.nodes == ("/a.py", "/b.py")
        assert sr.scores["/a.py"] == 0.7


# ======================================================================
# No backward alias
# ======================================================================


class TestNoBackwardAlias:
    def test_no_graph_in_exports(self) -> None:
        import grover.providers.graph

        assert "Graph" not in grover.providers.graph.__all__

    def test_import_graph_raises(self) -> None:
        with pytest.raises(ImportError):
            from grover.providers.graph import Graph  # type: ignore[attr-defined]  # noqa: F401

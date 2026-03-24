"""Tests for node similarity (removed methods).

common_neighbors, node_similarity, and similar_nodes have been removed
from RustworkxGraph. This file is kept as a placeholder.
"""

from __future__ import annotations

from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider

# ======================================================================
# Protocol satisfaction
# ======================================================================


class TestProtocolSatisfaction:
    def test_supports_graph_provider(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)

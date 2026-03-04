"""GraphOpsMixin — graph query operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.exceptions import CapabilityNotSupportedError
from grover.results import (
    FileSearchCandidate,
    GraphEvidence,
    GraphResult,
)

if TYPE_CHECKING:
    from grover.api.context import GroverContext
    from grover.providers.graph.protocol import GraphStore


class GraphOpsMixin:
    """Graph query operations extracted from GroverAsync."""

    _ctx: GroverContext

    def get_graph(self, path: str | None = None) -> GraphStore:
        """Return the graph for the mount owning *path*, or the first available.

        This replaces the old ``self.graph`` attribute which was removed
        in favour of per-mount graphs.
        """
        return self._ctx.resolve_graph_any(path)

    # ------------------------------------------------------------------
    # Graph query wrappers (resolve mount → delegate to backend's graph)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> GraphResult:
        """Return files that depend on *path*."""
        refs = self._ctx.resolve_graph(path).dependents(path)
        return GraphResult.from_refs(refs, strategy="dependents")

    def dependencies(self, path: str) -> GraphResult:
        """Return files that *path* depends on."""
        refs = self._ctx.resolve_graph(path).dependencies(path)
        return GraphResult.from_refs(refs, strategy="dependencies")

    def impacts(self, path: str, max_depth: int = 3) -> GraphResult:
        """Return files transitively impacted by changes to *path*."""
        refs = self._ctx.resolve_graph(path).impacts(path, max_depth)
        return GraphResult.from_refs(refs, strategy="impacts")

    def path_between(self, source: str, target: str) -> GraphResult:
        """Return the shortest path from *source* to *target*."""
        refs = self._ctx.resolve_graph(source).path_between(source, target)
        if refs is None:
            return GraphResult(
                success=True,
                message="No path found",
            )
        return GraphResult.from_refs(refs, strategy="path_between")

    def contains(self, path: str) -> GraphResult:
        """Return files contained by *path*."""
        refs = self._ctx.resolve_graph(path).contains(path)
        return GraphResult.from_refs(refs, strategy="contains")

    # ------------------------------------------------------------------
    # Graph algorithm wrappers (capability-checked)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        personalization: dict[str, float] | None = None,
        path: str | None = None,
    ) -> GraphResult:
        """Run PageRank on the knowledge graph.

        *path* selects which mount's graph to use (defaults to first visible).
        Raises :class:`~grover.exceptions.CapabilityNotSupportedError` if
        the graph backend does not support centrality algorithms.
        """
        graph = self._ctx.resolve_graph_any(path)
        try:
            scores = graph.pagerank(personalization=personalization)
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support centrality algorithms"
            raise CapabilityNotSupportedError(msg) from None
        candidates = [
            FileSearchCandidate(
                path=node_path,
                evidence=[
                    GraphEvidence(
                        strategy="pagerank",
                        path=node_path,
                        algorithm="pagerank",
                    )
                ],
            )
            for node_path in scores
        ]
        return GraphResult(
            success=True,
            message=f"PageRank computed for {len(candidates)} node(s)",
            candidates=candidates,
        )

    def ancestors(self, path: str) -> GraphResult:
        """All transitive predecessors of *path* in the knowledge graph."""
        graph = self._ctx.resolve_graph(path)
        try:
            node_set = graph.ancestors(path)
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg) from None
        return GraphResult.from_paths(sorted(node_set), strategy="ancestors")

    def descendants(self, path: str) -> GraphResult:
        """All transitive successors of *path* in the knowledge graph."""
        graph = self._ctx.resolve_graph(path)
        try:
            node_set = graph.descendants(path)
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg) from None
        return GraphResult.from_paths(sorted(node_set), strategy="descendants")

    def meeting_subgraph(
        self,
        paths: list[str],
        *,
        max_size: int = 50,
    ) -> GraphResult:
        """Extract the subgraph connecting *paths* via shortest paths."""
        graph = self._ctx.resolve_graph_any(paths[0] if paths else None)
        try:
            sub = graph.meeting_subgraph(paths, max_size=max_size)
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg) from None
        return GraphResult.from_paths(sorted(sub.nodes), strategy="meeting_subgraph")

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> GraphResult:
        """Extract the neighborhood subgraph around *path*."""
        graph = self._ctx.resolve_graph(path)
        try:
            sub = graph.neighborhood(
                path,
                max_depth=max_depth,
                direction=direction,
                edge_types=edge_types,
            )
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg) from None
        return GraphResult.from_paths(sorted(sub.nodes), strategy="neighborhood")

    def find_nodes(self, *, path: str | None = None, **attrs: object) -> GraphResult:
        """Find graph nodes matching all attribute predicates."""
        graph = self._ctx.resolve_graph_any(path)
        try:
            node_list = graph.find_nodes(**attrs)
        except (AttributeError, NotImplementedError):
            msg = "Graph backend does not support filtering"
            raise CapabilityNotSupportedError(msg) from None
        return GraphResult.from_paths(node_list, strategy="find_nodes")

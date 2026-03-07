"""GraphOpsMixin — graph query and connection operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.exceptions import MountNotFoundError
from grover.results import ConnectionResult
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext
    from grover.mount import Mount
    from grover.providers.graph.protocol import GraphProvider
    from grover.results import (
        AncestorsResult,
        BetweennessResult,
        ClosenessResult,
        CommonNeighborsResult,
        DegreeResult,
        DescendantsResult,
        EgoGraphResult,
        FileSearchResult,
        HarmonicResult,
        HasPathResult,
        HitsResult,
        KatzResult,
        MeetingSubgraphResult,
        PageRankResult,
        PredecessorsResult,
        ShortestPathResult,
        SubgraphSearchResult,
        SuccessorsResult,
    )


class GraphOpsMixin:
    """Graph query, algorithm, and connection operations extracted from GroverAsync."""

    _ctx: GroverContext

    def get_graph(self, path: str | None = None) -> GraphProvider:
        """Return the graph for the mount owning *path*, or the first available."""
        return self._ctx.resolve_graph_any(path)

    # ------------------------------------------------------------------
    # Connection operations (persist through FS, graph updated via worker)
    # ------------------------------------------------------------------

    def _validate_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
    ) -> tuple[Mount, None] | tuple[None, ConnectionResult]:
        """Validate writable access and resolve mount for a connection operation."""
        if err := self._ctx.check_writable(source_path):
            return None, ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return None, ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        assert mount.filesystem is not None
        return mount, None

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
    ) -> ConnectionResult:
        """Add a connection between two files, persisted through the filesystem."""
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        mount, err = self._validate_connection(source_path, target_path, connection_type)
        if err:
            return err
        assert mount is not None
        assert mount.filesystem is not None

        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.add_connection(
                source_path,
                target_path,
                connection_type,
                weight=weight,
                session=sess,
            )

        if result.success:
            self._ctx.worker.schedule_immediate(
                self._process_connection_added(source_path, target_path, connection_type, weight)  # type: ignore[attr-defined]
            )

        return result

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
    ) -> ConnectionResult:
        """Delete a connection between two files."""
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        mount, err = self._validate_connection(source_path, target_path, connection_type or "")
        if err:
            return err
        assert mount is not None
        assert mount.filesystem is not None

        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.delete_connection(
                source_path,
                target_path,
                connection_type=connection_type,
                session=sess,
            )

        if result.success:
            self._ctx.worker.schedule_immediate(
                self._process_connection_deleted(source_path, target_path)  # type: ignore[attr-defined]
            )

        return result

    # ------------------------------------------------------------------
    # Traversal queries — pure delegates
    # ------------------------------------------------------------------

    def predecessors(self, path: str) -> PredecessorsResult:
        """Return graph predecessors of *path* (nodes with edges pointing to it)."""
        return self._ctx.resolve_graph(path).predecessors(path)

    def successors(self, path: str) -> SuccessorsResult:
        """Return graph successors of *path* (nodes it points to)."""
        return self._ctx.resolve_graph(path).successors(path)

    def ancestors(self, path: str) -> AncestorsResult:
        """Return all nodes reachable by following edges backward from *path*."""
        return self._ctx.resolve_graph(path).ancestors(path)

    def descendants(self, path: str) -> DescendantsResult:
        """Return all nodes reachable by following edges forward from *path*."""
        return self._ctx.resolve_graph(path).descendants(path)

    def shortest_path(self, source: str, target: str) -> ShortestPathResult:
        """Return the shortest path from *source* to *target*."""
        return self._ctx.resolve_graph(source).path_between(source, target)

    def has_path(self, source: str, target: str) -> HasPathResult:
        """Check if a directed path exists from *source* to *target*."""
        return self._ctx.resolve_graph(source).has_path(source, target)

    # ------------------------------------------------------------------
    # Subgraph extraction — pure delegates
    # ------------------------------------------------------------------

    def subgraph(
        self,
        candidates: FileSearchResult,
        *,
        path: str | None = None,
    ) -> SubgraphSearchResult:
        """Extract the induced subgraph for nodes in *candidates*."""
        graph = self._ctx.resolve_graph_any(path)
        return graph.subgraph(list(candidates.paths))

    def min_meeting_subgraph(
        self,
        candidates: FileSearchResult,
        *,
        max_size: int = 50,
    ) -> MeetingSubgraphResult:
        """Extract the subgraph connecting candidate nodes via shortest paths."""
        paths = list(candidates.paths)
        graph = self._ctx.resolve_graph_any(paths[0] if paths else None)
        return graph.meeting_subgraph(paths, max_size=max_size)

    def ego_graph(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> EgoGraphResult:
        """Extract the neighborhood subgraph around *path*."""
        return self._ctx.resolve_graph(path).neighborhood(
            path,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
        )

    # ------------------------------------------------------------------
    # Centrality algorithms — pure delegates
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
        personalization: dict[str, float] | None = None,
    ) -> PageRankResult:
        """Run PageRank on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).pagerank(
            candidates, personalization=personalization
        )

    def betweenness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> BetweennessResult:
        """Betweenness centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).betweenness_centrality(candidates)

    def closeness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> ClosenessResult:
        """Closeness centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).closeness_centrality(candidates)

    def harmonic_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> HarmonicResult:
        """Harmonic centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).harmonic_centrality(candidates)

    def katz_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> KatzResult:
        """Katz centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).katz_centrality(candidates)

    def degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """Degree centrality (in + out) on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).degree_centrality(candidates)

    def in_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """In-degree centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).in_degree_centrality(candidates)

    def out_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """Out-degree centrality on the knowledge graph."""
        return self._ctx.resolve_graph_any(path).out_degree_centrality(candidates)

    def hits(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> HitsResult:
        """HITS hub and authority scores."""
        return self._ctx.resolve_graph_any(path).hits(candidates)

    # ------------------------------------------------------------------
    # Other graph operations — pure delegates
    # ------------------------------------------------------------------

    def common_neighbors(
        self,
        path1: str,
        path2: str,
        *,
        path: str | None = None,
    ) -> CommonNeighborsResult:
        """Find common neighbors of two nodes."""
        return self._ctx.resolve_graph_any(path).common_neighbors(path1, path2)

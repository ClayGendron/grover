"""GraphOpsMixin — graph query and connection operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.exceptions import MountNotFoundError
from grover.models.internal.results import FileOperationResult, FileSearchResult
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext
    from grover.mount import Mount
    from grover.providers.graph.protocol import GraphProvider


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
    ) -> tuple[Mount, None] | tuple[None, FileOperationResult]:
        """Validate writable access and resolve mount for a connection operation."""
        if err := self._ctx.check_writable(source_path):
            return None, FileOperationResult(
                success=False,
                message=err,
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return None, FileOperationResult(
                success=False,
                message=f"No mount found for path: {source_path}",
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
    ) -> FileOperationResult:
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
    ) -> FileOperationResult:
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
    # Traversal queries — async delegates with session pass-through
    # ------------------------------------------------------------------

    async def predecessors(self, path: str) -> FileSearchResult:
        """Return graph predecessors of *path* (nodes with edges pointing to it)."""
        gp, mount = self._ctx.resolve_graph_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.predecessors(path, session=sess)  # type: ignore[arg-type]

    async def successors(self, path: str) -> FileSearchResult:
        """Return graph successors of *path* (nodes it points to)."""
        gp, mount = self._ctx.resolve_graph_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.successors(path, session=sess)  # type: ignore[arg-type]

    async def ancestors(self, path: str) -> FileSearchResult:
        """Return all nodes reachable by following edges backward from *path*."""
        gp, mount = self._ctx.resolve_graph_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.ancestors(path, session=sess)  # type: ignore[arg-type]

    async def descendants(self, path: str) -> FileSearchResult:
        """Return all nodes reachable by following edges forward from *path*."""
        gp, mount = self._ctx.resolve_graph_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.descendants(path, session=sess)  # type: ignore[arg-type]

    async def shortest_path(self, source: str, target: str) -> FileSearchResult:
        """Return the shortest path from *source* to *target*."""
        gp, mount = self._ctx.resolve_graph_with_mount(source)
        async with self._ctx.session_for(mount) as sess:
            return await gp.path_between(source, target, session=sess)  # type: ignore[arg-type]

    async def has_path(self, source: str, target: str) -> FileSearchResult:
        """Check if a directed path exists from *source* to *target*."""
        gp, mount = self._ctx.resolve_graph_with_mount(source)
        async with self._ctx.session_for(mount) as sess:
            return await gp.has_path(source, target, session=sess)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Subgraph extraction — async delegates with session pass-through
    # ------------------------------------------------------------------

    async def subgraph(
        self,
        candidates: FileSearchResult,
        *,
        path: str | None = None,
    ) -> FileSearchResult:
        """Extract the induced subgraph for nodes in *candidates*."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.subgraph(list(candidates.paths), session=sess)  # type: ignore[arg-type]

    async def min_meeting_subgraph(
        self,
        candidates: FileSearchResult,
        *,
        max_size: int = 50,
    ) -> FileSearchResult:
        """Extract the subgraph connecting candidate nodes via shortest paths."""
        paths = list(candidates.paths)
        gp, mount = self._ctx.resolve_graph_any_with_mount(paths[0] if paths else None)
        async with self._ctx.session_for(mount) as sess:
            return await gp.meeting_subgraph(paths, max_size=max_size, session=sess)  # type: ignore[arg-type]

    async def ego_graph(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> FileSearchResult:
        """Extract the neighborhood subgraph around *path*."""
        gp, mount = self._ctx.resolve_graph_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.neighborhood(
                path,
                max_depth=max_depth,
                direction=direction,
                edge_types=edge_types,
                session=sess,  # type: ignore[arg-type]
            )

    # ------------------------------------------------------------------
    # Centrality algorithms — async delegates with session pass-through
    # ------------------------------------------------------------------

    async def pagerank(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
        personalization: dict[str, float] | None = None,
    ) -> FileSearchResult:
        """Run PageRank on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.pagerank(candidates, personalization=personalization, session=sess)  # type: ignore[arg-type]

    async def betweenness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Betweenness centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.betweenness_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def closeness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Closeness centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.closeness_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def harmonic_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Harmonic centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.harmonic_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def katz_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Katz centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.katz_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Degree centrality (in + out) on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.degree_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def in_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """In-degree centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.in_degree_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def out_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """Out-degree centrality on the knowledge graph."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.out_degree_centrality(candidates, session=sess)  # type: ignore[arg-type]

    async def hits(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> FileSearchResult:
        """HITS hub and authority scores."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.hits(candidates, session=sess)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Other graph operations — async delegates with session pass-through
    # ------------------------------------------------------------------

    async def common_neighbors(
        self,
        path1: str,
        path2: str,
        *,
        path: str | None = None,
    ) -> FileSearchResult:
        """Find common neighbors of two nodes."""
        gp, mount = self._ctx.resolve_graph_any_with_mount(path)
        async with self._ctx.session_for(mount) as sess:
            return await gp.common_neighbors(path1, path2, session=sess)  # type: ignore[arg-type]

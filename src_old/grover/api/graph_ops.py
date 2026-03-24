"""GraphOpsMixin — graph query and connection operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.exceptions import MountNotFoundError
from grover.models.internal.results import FileOperationResult, FileSearchResult, FileSearchSet, GroverResult
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.api.context import GroverContext
    from grover.models.database.connection import FileConnectionModelBase
    from grover.mount import Mount
    from grover.providers.graph.protocol import GraphProvider


class GraphOpsMixin:
    """Graph query, algorithm, and connection operations extracted from GroverAsync."""

    _ctx: GroverContext

    def get_graph(self, path: str | None = None) -> GraphProvider:
        """Return the graph for the mount owning *path*, or the first available."""
        return self._ctx.resolve_graph_any(path)

    # ------------------------------------------------------------------
    # Multi-mount dispatch
    # ------------------------------------------------------------------

    def _group_by_mount(self, candidates: FileSearchSet) -> list[tuple[Mount, FileSearchSet]]:
        """Group candidate paths by their owning mount."""
        groups: dict[str, tuple[Mount, list[str]]] = {}
        for path in candidates.paths:
            try:
                mount, _rel = self._ctx.registry.resolve(path)
            except MountNotFoundError:
                continue
            key = mount.path
            if key not in groups:
                groups[key] = (mount, [])
            groups[key][1].append(path)
        return [(mount, FileSearchSet.from_paths(paths)) for mount, paths in groups.values()]

    def _all_graph_mounts(self) -> list[Mount]:
        """Return all mounts that have a graph provider."""
        return [
            m
            for m in self._ctx.registry.list_mounts()
            if m.filesystem is not None and getattr(m.filesystem, "graph_provider", None) is not None
        ]

    async def _dispatch_graph(
        self,
        candidates: FileSearchSet,
        method: str,
        **kwargs: object,
    ) -> FileSearchResult:
        """Split candidates by mount, call filesystem graph method, union results."""
        if candidates.paths:
            mount_groups = self._group_by_mount(candidates)
        else:
            # Empty candidates = run on all mounts with graphs
            mount_groups = [(m, FileSearchSet()) for m in self._all_graph_mounts()]

        if not mount_groups:
            return FileSearchResult(success=False, message="No mounts found")

        results: list[FileSearchResult] = []
        for mount, subset in mount_groups:
            assert mount.filesystem is not None
            fn = getattr(mount.filesystem, method)
            async with self._ctx.session_for(mount) as sess:
                result = await fn(subset, session=sess, **kwargs)
                results.append(result)

        if len(results) == 1:
            return results[0]
        combined = results[0]
        for r in results[1:]:
            combined = combined | r
        return combined

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
                message=err.message,
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

    async def add_connections(
        self,
        connections: list[FileConnectionModelBase],
    ) -> GroverResult:
        """Batch add connections, grouped by mount."""
        if not connections:
            return GroverResult(success=True, message="No connections to add")

        groups, err = self._ctx.group_by_mount_writable(connections, lambda c: c.source_path)
        if err:
            return err

        async def _handler(mount: Mount, group: list[FileConnectionModelBase], session: AsyncSession) -> GroverResult:
            assert mount.filesystem is not None
            return await mount.filesystem.add_connections(group, session=session)

        result = await self._ctx.dispatch_to_mounts(groups, _handler)
        result.message = f"Added {len(result.connections)} connection(s)"

        # Schedule graph projection for successful connections
        for conn in result.connections:
            self._ctx.worker.schedule_immediate(
                self._process_connection_added(  # type: ignore[attr-defined]
                    conn.source_path, conn.target_path, conn.type, conn.weight
                )
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
    # Traversal queries
    # ------------------------------------------------------------------

    async def predecessors(self, candidates: FileSearchSet) -> FileSearchResult:
        """Return graph predecessors (nodes with edges pointing to candidates)."""
        return await self._dispatch_graph(candidates, "predecessors")

    async def successors(self, candidates: FileSearchSet) -> FileSearchResult:
        """Return graph successors (nodes candidates point to)."""
        return await self._dispatch_graph(candidates, "successors")

    async def ancestors(self, candidates: FileSearchSet) -> FileSearchResult:
        """Return all nodes reachable by following edges backward from candidates."""
        return await self._dispatch_graph(candidates, "ancestors")

    async def descendants(self, candidates: FileSearchSet) -> FileSearchResult:
        """Return all nodes reachable by following edges forward from candidates."""
        return await self._dispatch_graph(candidates, "descendants")

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    async def min_meeting_subgraph(self, candidates: FileSearchSet) -> FileSearchResult:
        """Extract the minimum subgraph connecting candidate nodes."""
        return await self._dispatch_graph(candidates, "min_meeting_subgraph")

    async def ego_graph(self, candidates: FileSearchSet, *, max_depth: int = 2) -> FileSearchResult:
        """Extract the neighborhood subgraph around candidates."""
        return await self._dispatch_graph(candidates, "neighborhood", max_depth=max_depth)

    # ------------------------------------------------------------------
    # Centrality algorithms
    # ------------------------------------------------------------------

    async def pagerank(
        self,
        candidates: FileSearchSet,
        *,
        personalization: dict[str, float] | None = None,
    ) -> FileSearchResult:
        """Run PageRank on the knowledge graph."""
        return await self._dispatch_graph(candidates, "pagerank", personalization=personalization)

    async def betweenness_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """Betweenness centrality on the knowledge graph."""
        return await self._dispatch_graph(candidates, "betweenness_centrality")

    async def closeness_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """Closeness centrality on the knowledge graph."""
        return await self._dispatch_graph(candidates, "closeness_centrality")

    async def katz_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """Katz centrality on the knowledge graph."""
        return await self._dispatch_graph(candidates, "katz_centrality")

    async def degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """Degree centrality (in + out) on the knowledge graph."""
        return await self._dispatch_graph(candidates, "degree_centrality")

    async def in_degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """In-degree centrality on the knowledge graph."""
        return await self._dispatch_graph(candidates, "in_degree_centrality")

    async def out_degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        """Out-degree centrality on the knowledge graph."""
        return await self._dispatch_graph(candidates, "out_degree_centrality")

    async def hits(self, candidates: FileSearchSet) -> FileSearchResult:
        """HITS hub and authority scores."""
        return await self._dispatch_graph(candidates, "hits")

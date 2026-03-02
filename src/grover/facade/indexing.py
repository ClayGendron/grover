"""IndexMixin — processing methods, analysis pipeline, and persistence for GroverAsync."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from grover.fs.exceptions import MountNotFoundError
from grover.fs.permissions import Permission
from grover.fs.protocol import SupportsConnections, SupportsFileChunks
from grover.fs.utils import normalize_path
from grover.search.extractors import extract_from_chunks, extract_from_file
from grover.types import ListDirEvidence

if TYPE_CHECKING:
    from grover.facade.context import GroverContext

logger = logging.getLogger(__name__)


class IndexMixin:
    """Processing methods, analysis pipeline, and persistence extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Processing methods (called by worker via schedule/schedule_immediate)
    # ------------------------------------------------------------------

    async def _process_write(
        self, path: str, content: str | None = None, user_id: str | None = None
    ) -> None:
        """Analyze and integrate a written/edited/restored file."""
        if self._ctx.meta_fs is None:
            return
        if "/.grover/" in path:
            return
        if content is None:
            result = await self.read(path)  # type: ignore[attr-defined]
            if not result.success:
                return
            content = result.content
        if content is not None:
            await self._analyze_and_integrate(path, content, user_id=user_id)

    async def _process_delete(self, path: str, user_id: str | None = None) -> None:
        """Clean up graph, search, chunks, and connections for a deleted file."""
        if self._ctx.meta_fs is None:
            return
        if "/.grover/" in path:
            return
        # In-memory graph cleanup
        try:
            graph = self._ctx.resolve_graph(path)
            if graph.has_node(path):
                graph.remove_file_subgraph(path)
        except RuntimeError:
            pass  # Mount may not have a graph
        # DB cleanup: search, chunks, connections in a single session
        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return
        async with self._ctx.session_for(mount) as sess:
            search_engine = mount.search
            if search_engine is not None:
                await search_engine.remove_file(path, session=sess)
            if isinstance(mount.filesystem, SupportsFileChunks):
                await mount.filesystem.delete_file_chunks(path, session=sess)
            conn_svc = getattr(mount.filesystem, "connections", None)
            if conn_svc is not None:
                await conn_svc.delete_connections_for_path(sess, path)

    async def _process_move(self, old_path: str, new_path: str, user_id: str | None = None) -> None:
        """Clean up old path and re-analyze new path after a move."""
        if self._ctx.meta_fs is None:
            return
        if old_path and "/.grover/" not in old_path:
            await self._process_delete(old_path)
        if "/.grover/" not in new_path:
            await self._process_write(new_path, None, user_id)

    async def _process_connection_added(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        weight: float = 1.0,
    ) -> None:
        """Update the in-memory graph when a connection is persisted through FS."""
        try:
            graph = self._ctx.resolve_graph(source_path)
        except RuntimeError:
            return
        graph.add_edge(
            source_path,
            target_path,
            edge_type=connection_type,
            weight=weight,
        )

    async def _process_connection_deleted(self, source_path: str, target_path: str) -> None:
        """Update the in-memory graph when a connection is removed from FS."""
        try:
            graph = self._ctx.resolve_graph(source_path)
        except RuntimeError:
            return
        if graph.has_edge(source_path, target_path):
            graph.remove_edge(source_path, target_path)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _analyze_and_integrate(
        self, path: str, content: str, *, user_id: str | None = None
    ) -> dict[str, int]:
        import hashlib

        stats = {"chunks_created": 0, "edges_added": 0}

        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return stats

        graph = mount.graph
        if graph is None:
            return stats

        search_engine = mount.search

        # In-memory graph cleanup + re-add
        if graph.has_node(path):
            graph.remove_file_subgraph(path)
        graph.add_node(path)

        analysis = self._ctx.analyzer_registry.analyze_file(path, content)

        # Edges to project into graph after DB commit
        edges_to_project: list[tuple[str, str, str, float]] = []

        # Single session for all DB operations (search, chunks, connections)
        async with self._ctx.session_for(mount) as sess:
            # Remove old search entries
            if search_engine is not None:
                await search_engine.remove_file(path, session=sess)

            if analysis is not None:
                chunks, edges = analysis

                # Write chunk DB rows
                if isinstance(mount.filesystem, SupportsFileChunks) and chunks:
                    chunk_dicts = [
                        {
                            "path": chunk.path,
                            "line_start": chunk.line_start,
                            "line_end": chunk.line_end,
                            "content": chunk.content,
                            "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                        }
                        for chunk in chunks
                    ]
                    await mount.filesystem.replace_file_chunks(
                        path, chunk_dicts, session=sess
                    )

                # In-memory graph: chunk nodes + "contains" edges
                for chunk in chunks:
                    graph.add_node(
                        chunk.path,
                        parent_path=path,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        name=chunk.name,
                    )
                    graph.add_edge(path, chunk.path, edge_type="contains")
                    stats["chunks_created"] += 1

                # Persist dependency edges through FS (graph projection
                # updated post-commit).  "contains" edges are structural
                # and remain in-memory only.
                # Skip connection writes for read-only mounts (defensive).
                dep_edges = [e for e in edges if e.edge_type != "contains"]
                is_writable = mount.permission != Permission.READ_ONLY
                if isinstance(mount.filesystem, SupportsConnections) and dep_edges and is_writable:
                    # Delete stale outgoing connections before re-adding.
                    # Only outgoing (source_path == path) so we preserve
                    # edges from OTHER files that point to this one.
                    conn_svc = getattr(mount.filesystem, "connections", None)
                    if conn_svc is not None:
                        await conn_svc.delete_outgoing_connections(sess, path)
                    for edge in dep_edges:
                        _w: float = (
                            float(edge.metadata.get("weight", 1.0))  # type: ignore[arg-type]
                            if edge.metadata
                            else 1.0
                        )
                        await mount.filesystem.add_connection(
                            edge.source,
                            edge.target,
                            edge.edge_type,
                            weight=_w,
                            session=sess,
                        )
                        edges_to_project.append((edge.source, edge.target, edge.edge_type, _w))
                        stats["edges_added"] += 1
                elif dep_edges:
                    # Fallback: no SupportsConnections, add directly to graph
                    for edge in dep_edges:
                        meta: dict[str, Any] = dict(edge.metadata)
                        graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **meta)
                        stats["edges_added"] += 1

                # Index chunks for search
                if search_engine is not None:
                    embeddable = extract_from_chunks(chunks)
                    if embeddable:
                        await search_engine.add_batch(embeddable, session=sess)
            else:
                # No analysis — index whole file for search
                if search_engine is not None:
                    embeddable = extract_from_file(path, content)
                    if embeddable:
                        await search_engine.add_batch(embeddable, session=sess)

        # Project edges into graph after commit (post-commit ordering)
        for source, target, edge_type, weight in edges_to_project:
            graph.add_edge(source, target, edge_type=edge_type, weight=weight)

        return stats

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    async def index(self, mount_path: str | None = None) -> dict[str, int]:
        stats = {
            "files_scanned": 0,
            "chunks_created": 0,
            "edges_added": 0,
            "files_skipped": 0,
        }

        if mount_path is not None:
            await self._walk_and_index(mount_path, stats)
        else:
            for mount in self._ctx.registry.list_visible_mounts():
                await self._walk_and_index(mount.path, stats)

        await self._async_save()
        return stats

    async def _walk_and_index(self, path: str, stats: dict[str, int]) -> None:
        # Skip read-only mounts — indexing writes chunks, edges, and
        # search entries which are all mutations.
        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return
        if mount.permission == Permission.READ_ONLY:
            logger.debug("Skipping read-only mount for indexing: %s", path)
            return

        result = await self.list_dir(path)  # type: ignore[attr-defined]
        if not result.success:
            return

        for entry_path in result.paths:
            if "/.grover/" in entry_path:
                continue
            evs = result.explain(entry_path)
            is_dir = any(isinstance(e, ListDirEvidence) and e.is_directory for e in evs)
            if is_dir:
                await self._walk_and_index(entry_path, stats)
            else:
                content = await self._read_file_content(entry_path)
                if content is None:
                    stats["files_skipped"] += 1
                    continue
                file_stats = await self._analyze_and_integrate(entry_path, content)
                stats["files_scanned"] += 1
                stats["chunks_created"] += file_stats["chunks_created"]
                stats["edges_added"] += file_stats["edges_added"]

    async def _read_file_content(self, path: str) -> str | None:
        read_result = await self.read(path)  # type: ignore[attr-defined]
        if read_result.success:
            return read_result.content

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return None

        backend = mount.filesystem
        if hasattr(backend, "_read_content"):
            if mount.session_factory is not None:
                async with self._ctx.session_for(mount) as sess:
                    content: str | None = await backend._read_content(rel_path, sess)  # type: ignore[union-attr]
            else:
                content = await backend._read_content(rel_path, None)  # type: ignore[union-attr]
            return content

        return None

    async def flush(self) -> None:
        """Wait for all pending background indexing to complete.

        In ``background`` mode this drains the debounce queue and waits
        for all active analysis tasks to finish.  In ``manual`` mode this
        is a no-op.  Call before querying if you need guaranteed consistency
        after recent writes.
        """
        await self._ctx.drain()

    async def save(self) -> None:
        await self._ctx.drain()
        await self._async_save()

    async def sync(self, *, path: str | None = None) -> None:
        """Reload graph and search index from DB for a mount or all mounts.

        This is useful after external changes to the database — it
        re-reads the persisted graph edges and search index from storage.
        """
        if path is not None:
            mount, _rel = self._ctx.registry.resolve(normalize_path(path))
            await self._load_mount_state(mount)  # type: ignore[attr-defined]
        else:
            for mount in self._ctx.registry.list_mounts():
                await self._load_mount_state(mount)  # type: ignore[attr-defined]

    async def _async_save(self) -> None:
        """Save per-mount search state.

        Note: graph edges are no longer saved via ``to_sql()`` here.
        Edge persistence is now handled by the filesystem layer —
        ``add_connection()`` writes edges to DB, and the graph is a
        pure in-memory projection loaded via ``from_sql()`` on mount.
        """
        for mount in self._ctx.registry.list_visible_mounts():
            # Save search index to disk
            search_engine = mount.search
            if search_engine is not None and self._ctx.meta_data_dir is not None:
                slug = mount.path.strip("/").replace("/", "_") or "_default"
                search_dir = self._ctx.meta_data_dir / "search" / slug
                try:
                    search_engine.save(str(search_dir))
                except Exception:
                    logger.debug(
                        "Failed to save search index for %s",
                        mount.path,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._ctx.closed:
            return
        self._ctx.closed = True

        await self._ctx.drain()
        await self._async_save()
        # Close all backends directly
        for mount in self._ctx.registry.list_mounts():
            if hasattr(mount.filesystem, "close"):
                try:
                    await mount.filesystem.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.path, exc_info=True)

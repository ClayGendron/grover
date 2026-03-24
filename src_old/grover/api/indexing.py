"""IndexMixin — processing methods, analysis pipeline, and persistence for GroverAsync."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from grover.exceptions import MountNotFoundError
from grover.models.internal.evidence import ListDirEvidence
from grover.permissions import Permission
from grover.providers.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)

if TYPE_CHECKING:
    from grover.api.context import GroverContext
    from grover.models.database.chunk import FileChunkModelBase

logger = logging.getLogger(__name__)


class IndexMixin:
    """Processing methods, analysis pipeline, and persistence extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Processing methods (called by worker via schedule/schedule_immediate)
    # ------------------------------------------------------------------

    async def _process_write(self, path: str, content: str | None = None, user_id: str | None = None) -> None:
        """Analyze and integrate a written/edited/restored file."""
        if not self._ctx.initialized:
            return
        if "/.grover/" in path:
            return
        if content is None:
            result = await self.read(path)  # type: ignore[attr-defined]
            if not result.success:
                return
            content = result.file.content if result.file else None
        if content is not None:
            await self._analyze_and_integrate(path, content, user_id=user_id)

    async def _process_delete(self, path: str, user_id: str | None = None) -> None:
        """Clean up graph, search, chunks, and connections for a deleted file."""
        if not self._ctx.initialized:
            return
        if "/.grover/" in path:
            return
        # In-memory graph cleanup
        try:
            graph = self._ctx.resolve_graph(path)
            if graph.has_node(path):
                graph.remove_node(path)
        except RuntimeError:
            pass  # Mount may not have a graph
        # DB cleanup: search, chunks, connections in a single session
        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return
        async with self._ctx.session_for(mount) as sess:
            assert mount.filesystem is not None
            await mount.filesystem.search_remove_file(path, session=sess)
            await mount.filesystem.delete_file_chunks(path, session=sess)
            await mount.filesystem.delete_connections_for_path(sess, path)  # type: ignore[union-attr]

    async def _process_move(self, old_path: str, new_path: str, user_id: str | None = None) -> None:
        """Clean up old path and re-analyze new path after a move."""
        if not self._ctx.initialized:
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

    async def _process_chunk_write(self, chunk: FileChunkModelBase) -> None:
        """Update graph and search index after a chunk write."""
        if not self._ctx.initialized:
            return

        file_path = chunk.file_path
        chunk_path = chunk.path

        # Graph: ensure parent node exists, add chunk node + "contains" edge
        try:
            graph = self._ctx.resolve_graph(file_path)
            if not graph.has_node(file_path):
                graph.add_node(file_path)
            graph.add_node(chunk_path)
            if not graph.has_edge(file_path, chunk_path):
                graph.add_edge(file_path, chunk_path, edge_type="contains")
        except RuntimeError:
            pass  # No graph on this mount

        # Search: remove old entry, add new
        try:
            mount, _rel = self._ctx.registry.resolve(file_path)
        except MountNotFoundError:
            return

        if chunk.content and chunk.content.strip():
            embeddable = EmbeddableChunk(
                path=chunk_path,
                content=chunk.content,
                parent_path=file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
            async with self._ctx.session_for(mount) as sess:
                assert mount.filesystem is not None
                await mount.filesystem.search_remove_file(chunk_path, session=sess)
                await mount.filesystem.search_add_batch([embeddable], session=sess)

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

    async def _analyze_and_integrate(self, path: str, content: str, *, user_id: str | None = None) -> dict[str, int]:
        import hashlib

        stats = {"chunks_created": 0, "edges_added": 0}

        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return stats

        graph = getattr(mount.filesystem, "graph_provider", None)
        if graph is None:
            return stats

        # In-memory graph cleanup + re-add
        if graph.has_node(path):
            graph.remove_node(path)
        graph.add_node(path)

        analysis = self._ctx.analyzer_registry.analyze_file(path, content)

        # Edges to project into graph after DB commit
        edges_to_project: list[tuple[str, str, str, float]] = []

        # Single session for all DB operations (search, chunks, connections)
        async with self._ctx.session_for(mount) as sess:
            assert mount.filesystem is not None
            # Remove old search entries
            await mount.filesystem.search_remove_file(path, session=sess)

            if analysis is not None:
                chunks, edges = analysis

                # Write chunk DB rows
                if chunks:
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
                    await mount.filesystem.replace_file_chunks(path, chunk_dicts, session=sess)

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
                if dep_edges and is_writable:
                    # Delete stale outgoing connections before re-adding.
                    # Only outgoing (source_path == path) so we preserve
                    # edges from OTHER files that point to this one.
                    await mount.filesystem.delete_outgoing_connections(sess, path)  # type: ignore[union-attr]
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
                    # Read-only mount: add directly to in-memory graph
                    for edge in dep_edges:
                        meta: dict[str, Any] = dict(edge.metadata)
                        graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **meta)
                        stats["edges_added"] += 1

                # Index chunks for search
                embeddable = extract_from_chunks(chunks)
                if embeddable:
                    await mount.filesystem.search_add_batch(embeddable, session=sess)
            else:
                # No analysis — index whole file for search
                embeddable = extract_from_file(path, content)
                if embeddable:
                    await mount.filesystem.search_add_batch(embeddable, session=sess)

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
            for mount in self._ctx.registry.list_mounts():
                await self._walk_and_index(mount.path, stats)

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

        for entry_file in result.files:
            entry_path = entry_file.path
            if "/.grover/" in entry_path:
                continue
            evs = entry_file.evidence
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
        if read_result.success and read_result.file:
            return read_result.file.content

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._ctx.closed:
            return
        self._ctx.closed = True

        await self._ctx.drain()
        # Close all backends directly
        for mount in self._ctx.registry.list_mounts():
            if hasattr(mount.filesystem, "close"):
                try:
                    await mount.filesystem.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.path, exc_info=True)
            # Dispose engine if Grover owns it (EngineConfig path)
            if mount.engine is not None:
                try:
                    await mount.engine.dispose()
                except Exception:
                    logger.warning("Engine dispose failed for %s", mount.path, exc_info=True)

"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grover.events import EventBus, EventType, FileEvent
from grover.facade.context import GroverContext
from grover.facade.file_ops import FileOpsMixin
from grover.facade.graph_ops import GraphOpsMixin
from grover.facade.mounting import MountMixin
from grover.facade.search_ops import SearchOpsMixin
from grover.fs.exceptions import (
    CapabilityNotSupportedError,
    MountNotFoundError,
)
from grover.fs.mounts import MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import (
    SupportsConnections,
    SupportsFileChunks,
    SupportsReBAC,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.fs.utils import normalize_path
from grover.graph.analyzers import AnalyzerRegistry
from grover.search.extractors import extract_from_chunks, extract_from_file
from grover.types import (
    ConnectionResult,
    DeleteResult,
    FileSearchCandidate,
    GetVersionContentResult,
    ListDirEvidence,
    RestoreResult,
    ShareResult,
    ShareSearchResult,
    TrashResult,
    VersionResult,
)

if TYPE_CHECKING:
    from datetime import datetime

    from grover.search.protocols import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)


class GroverAsync(MountMixin, FileOpsMixin, SearchOpsMixin, GraphOpsMixin):
    """Async facade wiring filesystem, graph, analyzers, event bus, and search.

    Mount-first API: create an instance, then add mounts.

    Engine-based DB mount (primary API)::

        engine = create_async_engine("postgresql+asyncpg://...")
        g = GroverAsync(data_dir="/myapp/.grover")
        await g.add_mount("/data", engine=engine)

    Direct access — auto-commits per operation::

        g = GroverAsync()
        await g.add_mount("/app", backend)
        await g.write("/app/test.py", "print('hi')")
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._ctx = GroverContext(
            event_bus=EventBus(),
            registry=MountRegistry(),
            analyzer_registry=AnalyzerRegistry(),
            embedding_provider=embedding_provider,
            explicit_vector_store=vector_store,
            explicit_data_dir=Path(data_dir) if data_dir else None,
        )

        # Register event handlers
        self._ctx.event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._ctx.event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._ctx.event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._ctx.event_bus.register(EventType.FILE_RESTORED, self._on_file_restored)
        self._ctx.event_bus.register(EventType.CONNECTION_ADDED, self._on_connection_added)
        self._ctx.event_bus.register(EventType.CONNECTION_DELETED, self._on_connection_deleted)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_file_written(self, event: FileEvent) -> None:
        if self._ctx.meta_fs is None:
            return
        if "/.grover/" in event.path:
            return
        content = event.content
        if content is None:
            result = await self.read(event.path)
            if not result.success:
                return
            content = result.content
        if content is not None:
            await self._analyze_and_integrate(event.path, content, user_id=event.user_id)

    async def _on_file_deleted(self, event: FileEvent) -> None:
        if self._ctx.meta_fs is None:
            return
        if "/.grover/" in event.path:
            return
        try:
            graph = self._ctx.resolve_graph(event.path)
            if graph.has_node(event.path):
                graph.remove_file_subgraph(event.path)
        except RuntimeError:
            pass  # Mount may not have a graph
        try:
            search_engine = self._ctx.resolve_search_engine(event.path)
            if search_engine is not None:
                mount, _rel = self._ctx.registry.resolve(event.path)
                async with self._ctx.session_for(mount) as sess:
                    await search_engine.remove_file(event.path, session=sess)
        except RuntimeError:
            pass
        # Clean up chunk DB rows and connection DB rows
        await self._delete_chunks_for_path(event.path)
        await self._delete_connections_for_path(event.path)

    async def _on_file_moved(self, event: FileEvent) -> None:
        if self._ctx.meta_fs is None:
            return
        if event.old_path and "/.grover/" not in event.old_path:
            try:
                graph = self._ctx.resolve_graph(event.old_path)
                if graph.has_node(event.old_path):
                    graph.remove_file_subgraph(event.old_path)
            except RuntimeError:
                pass
            try:
                search_engine = self._ctx.resolve_search_engine(event.old_path)
                if search_engine is not None:
                    mount, _rel = self._ctx.registry.resolve(event.old_path)
                    async with self._ctx.session_for(mount) as sess:
                        await search_engine.remove_file(event.old_path, session=sess)
            except RuntimeError:
                pass
            # Clean up chunk and connection DB rows for old path
            await self._delete_chunks_for_path(event.old_path)
            await self._delete_connections_for_path(event.old_path)

        if "/.grover/" in event.path:
            return
        result = await self.read(event.path)
        if result.success:
            content = result.content
            if content is not None:
                await self._analyze_and_integrate(event.path, content, user_id=event.user_id)

    async def _on_file_restored(self, event: FileEvent) -> None:
        await self._on_file_written(event)

    async def _on_connection_added(self, event: FileEvent) -> None:
        """Update the in-memory graph when a connection is persisted through FS."""
        if event.source_path is None or event.target_path is None or event.connection_type is None:
            return
        try:
            graph = self._ctx.resolve_graph(event.source_path)
        except RuntimeError:
            return
        graph.add_edge(
            event.source_path,
            event.target_path,
            edge_type=event.connection_type,
            weight=event.weight,
        )

    async def _on_connection_deleted(self, event: FileEvent) -> None:
        """Update the in-memory graph when a connection is removed from FS."""
        if event.source_path is None or event.target_path is None:
            return
        try:
            graph = self._ctx.resolve_graph(event.source_path)
        except RuntimeError:
            return
        if graph.has_edge(event.source_path, event.target_path):
            graph.remove_edge(event.source_path, event.target_path)

    async def _delete_chunks_for_path(self, path: str) -> None:
        """Delete chunk DB rows for *path* if the backend supports it."""
        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return
        if isinstance(mount.filesystem, SupportsFileChunks):
            async with self._ctx.session_for(mount) as sess:
                await mount.filesystem.delete_file_chunks(path, session=sess)

    async def _delete_connections_for_path(self, path: str) -> None:
        """Delete connection DB rows for *path* if the backend supports it."""
        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return
        conn_svc = getattr(mount.filesystem, "connections", None)
        if conn_svc is not None:
            async with self._ctx.session_for(mount) as sess:
                await conn_svc.delete_connections_for_path(sess, path)

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

        if graph.has_node(path):
            graph.remove_file_subgraph(path)
        if search_engine is not None:
            async with self._ctx.session_for(mount) as sess:
                await search_engine.remove_file(path, session=sess)

        graph.add_node(path)

        analysis = self._ctx.analyzer_registry.analyze_file(path, content)

        if analysis is not None:
            chunks, edges = analysis

            # Write chunk DB rows instead of VFS files
            if isinstance(mount.filesystem, SupportsFileChunks) and chunks:
                chunk_dicts = [
                    {
                        "path": chunk.path,
                        "name": chunk.name,
                        "description": "",
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "content": chunk.content,
                        "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                    }
                    for chunk in chunks
                ]
                async with self._ctx.session_for(mount) as sess:
                    await mount.filesystem.replace_file_chunks(
                        path, chunk_dicts, session=sess, user_id=user_id
                    )

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

            # Persist dependency edges through FS (graph updated via event).
            # "contains" edges are structural (chunk membership) and remain
            # in-memory only — they are already added to the graph above.
            # Skip connection writes for read-only mounts (defensive).
            dep_edges = [e for e in edges if e.edge_type != "contains"]
            is_writable = mount.permission != Permission.READ_ONLY
            if isinstance(mount.filesystem, SupportsConnections) and dep_edges and is_writable:
                # Delete stale outgoing connections for this source before
                # re-adding.  Only outgoing (source_path == path) so we
                # preserve edges from OTHER files that point to this one.
                conn_svc = getattr(mount.filesystem, "connections", None)
                if conn_svc is not None:
                    async with self._ctx.session_for(mount) as sess:
                        await conn_svc.delete_outgoing_connections(sess, path)
                for edge in dep_edges:
                    _w: float = (
                        float(edge.metadata.get("weight", 1.0))  # type: ignore[arg-type]
                        if edge.metadata
                        else 1.0
                    )
                    async with self._ctx.session_for(mount) as sess:
                        await mount.filesystem.add_connection(
                            edge.source,
                            edge.target,
                            edge.edge_type,
                            weight=_w,
                            metadata=dict(edge.metadata) if edge.metadata else None,
                            session=sess,
                        )
                    # Emit event AFTER session commits (post-commit ordering)
                    await self._ctx.emit(
                        FileEvent(
                            event_type=EventType.CONNECTION_ADDED,
                            path=f"{edge.source}[{edge.edge_type}]{edge.target}",
                            source_path=edge.source,
                            target_path=edge.target,
                            connection_type=edge.edge_type,
                            weight=_w,
                        )
                    )
                    stats["edges_added"] += 1
            elif dep_edges:
                # Fallback: no SupportsConnections, add directly to graph
                for edge in dep_edges:
                    meta: dict[str, Any] = dict(edge.metadata)
                    graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **meta)
                    stats["edges_added"] += 1

            if search_engine is not None:
                embeddable = extract_from_chunks(chunks)
                if embeddable:
                    async with self._ctx.session_for(mount) as sess:
                        await search_engine.add_batch(embeddable, session=sess)
        else:
            if search_engine is not None:
                embeddable = extract_from_file(path, content)
                if embeddable:
                    async with self._ctx.session_for(mount) as sess:
                        await search_engine.add_batch(embeddable, session=sess)

        return stats

    # ------------------------------------------------------------------
    # Version operations (absorbed from VFS, capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> VersionResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            cap = self._ctx.get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._ctx.session_for(mount) as sess:
                return await cap.list_versions(rel_path, session=sess, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return VersionResult(success=False, message=str(e))

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            cap = self._ctx.get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._ctx.session_for(mount) as sess:
                return await cap.get_version_content(
                    rel_path, version, session=sess, user_id=user_id
                )
        except CapabilityNotSupportedError as e:
            return GetVersionContentResult(success=False, message=str(e))

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return RestoreResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            cap = self._ctx.get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._ctx.session_for(mount) as sess:
                result = await cap.restore_version(rel_path, version, session=sess, user_id=user_id)
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
                )
            return result
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    # ------------------------------------------------------------------
    # Trash operations (absorbed from VFS, capability-gated)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> TrashResult:
        """List all items in trash across all mounts."""
        combined = TrashResult(success=True, message="")
        for mount in self._ctx.registry.list_mounts():
            cap = self._ctx.get_capability(mount.filesystem, SupportsTrash)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                result = await cap.list_trash(session=sess, user_id=user_id)
            if result.success:
                rebased = result.rebase(mount.path)
                combined = combined | rebased
        combined.message = f"Found {len(combined)} item(s) in trash"
        return combined

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return RestoreResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            cap = self._ctx.get_capability(mount.filesystem, SupportsTrash)
            if cap is None:
                raise CapabilityNotSupportedError(f"Mount at {mount.path} does not support trash")
            async with self._ctx.session_for(mount) as sess:
                result = await cap.restore_from_trash(rel_path, session=sess, user_id=user_id)
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
                )
            return result
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts.  Skips read-only mounts."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._ctx.registry.list_mounts():
            # Skip read-only mounts — empty_trash is a mutation
            if mount.permission == Permission.READ_ONLY:
                continue
            cap = self._ctx.get_capability(mount.filesystem, SupportsTrash)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                result = await cap.empty_trash(session=sess, user_id=user_id)
            if not result.success:
                return result
            total_deleted += result.total_deleted or 0
            mounts_processed += 1
        return DeleteResult(
            success=True,
            message=f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)",
            total_deleted=total_deleted,
            permanent=True,
        )

    # ------------------------------------------------------------------
    # Share operations
    # ------------------------------------------------------------------

    async def share(
        self,
        path: str,
        grantee_id: str,
        permission: str = "read",
        *,
        user_id: str,
        expires_at: datetime | None = None,
    ) -> ShareResult:
        """Share a file or directory with another user.

        Requires a backend that supports sharing (e.g. ``UserScopedFileSystem``).
        """

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            try:
                share_info = await cap.share(
                    rel_path,
                    grantee_id,
                    permission,
                    user_id=user_id,
                    session=sess,
                    expires_at=expires_at,
                )
            except ValueError as e:
                return ShareResult(success=False, message=str(e))

        return ShareResult(
            success=True,
            message=f"Shared {path} with {grantee_id} ({permission})",
            path=path,
            grantee_id=share_info.grantee_id,
            permission=share_info.permission,
            granted_by=share_info.granted_by,
        )

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
    ) -> ShareResult:
        """Remove a share for a file or directory."""

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            removed = await cap.unshare(rel_path, grantee_id, user_id=user_id, session=sess)

        if removed:
            return ShareResult(
                success=True,
                message=f"Removed share on {path} for {grantee_id}",
            )
        return ShareResult(
            success=False,
            message=f"No share found on {path} for {grantee_id}",
        )

    async def list_shares(
        self,
        path: str,
        *,
        user_id: str,
    ) -> ShareSearchResult:
        """List all shares on a given path."""

        path = normalize_path(path)
        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareSearchResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareSearchResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            result = await cap.list_shares_on_path(rel_path, user_id=user_id, session=sess)

        # Rebase paths from backend-relative to absolute mount paths
        return result.rebase(mount.path)

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
    ) -> ShareSearchResult:
        """List all files shared with the current user across all mounts."""
        all_candidates: list[FileSearchCandidate] = []
        for mount in self._ctx.registry.list_mounts():
            cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                assert sess is not None
                result = await cap.list_shared_with_me(user_id=user_id, session=sess)
            # Backend returns paths like /@shared/alice/a.md — rebase to mount
            rebased = result.rebase(mount.path)
            all_candidates.extend(rebased.candidates)

        return ShareSearchResult(
            success=True,
            message=f"Found {len(all_candidates)} share(s)",
            candidates=all_candidates,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        """Reconcile disk ↔ DB for capable mounts."""
        total = {"created": 0, "updated": 0, "deleted": 0}
        mounts = self._ctx.registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.path == mount_path]

        for mount in mounts:
            # Skip read-only mounts — reconcile is a mutation
            if mount.permission == Permission.READ_ONLY:
                continue
            cap = self._ctx.get_capability(mount.filesystem, SupportsReconcile)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            for k in total:
                total[k] += stats.get(k, 0)

        return total

    # ------------------------------------------------------------------
    # Connection operations (persist through FS, graph updated via events)
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> ConnectionResult:
        """Add a connection between two files, persisted through the filesystem.

        The graph is updated via the CONNECTION_ADDED event handler after
        the DB transaction commits.
        """
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        if err := self._ctx.check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        async with self._ctx.session_for(mount) as sess:
            result = await backend.add_connection(
                source_path,
                target_path,
                connection_type,
                weight=weight,
                metadata=metadata,
                session=sess,
            )

        # Emit AFTER session commits (post-commit event ordering)
        if result.success:
            await self._ctx.emit(
                FileEvent(
                    event_type=EventType.CONNECTION_ADDED,
                    path=result.path,
                    source_path=source_path,
                    target_path=target_path,
                    connection_type=connection_type,
                    weight=weight,
                )
            )

        return result

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
    ) -> ConnectionResult:
        """Delete a connection between two files.

        The graph is updated via the CONNECTION_DELETED event handler after
        the DB transaction commits.
        """
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        if err := self._ctx.check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        async with self._ctx.session_for(mount) as sess:
            result = await backend.delete_connection(
                source_path,
                target_path,
                connection_type=connection_type,
                session=sess,
            )

        # Emit AFTER session commits (post-commit event ordering)
        if result.success:
            await self._ctx.emit(
                FileEvent(
                    event_type=EventType.CONNECTION_DELETED,
                    path=result.path,
                    source_path=source_path,
                    target_path=target_path,
                    connection_type=connection_type,
                )
            )

        return result

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
    ) -> list[object]:
        """List connections for a path."""
        path = normalize_path(path)

        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return []

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return []

        async with self._ctx.session_for(mount) as sess:
            return await backend.list_connections(
                path,
                direction=direction,
                connection_type=connection_type,
                session=sess,
            )

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

        result = await self.list_dir(path)
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
        read_result = await self.read(path)
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

    async def save(self) -> None:
        await self._async_save()

    async def sync(self, *, path: str | None = None) -> None:
        """Reload graph and search index from DB for a mount or all mounts.

        This is useful after external changes to the database — it
        re-reads the persisted graph edges and search index from storage.
        """
        if path is not None:
            mount, _rel = self._ctx.registry.resolve(normalize_path(path))
            await self._load_mount_state(mount)
        else:
            for mount in self._ctx.registry.list_mounts():
                await self._load_mount_state(mount)

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

        await self._async_save()
        # Close all backends directly
        for mount in self._ctx.registry.list_mounts():
            if hasattr(mount.filesystem, "close"):
                try:
                    await mount.filesystem.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.path, exc_info=True)

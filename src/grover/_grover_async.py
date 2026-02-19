"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grover.events import EventBus, EventType, FileEvent
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import CapabilityNotSupportedError, MountNotFoundError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import SupportsReBAC
from grover.fs.types import (
    DeleteResult,
    EditResult,
    GetVersionContentResult,
    GlobResult,
    GrepResult,
    ListSharesResult,
    ListVersionsResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareInfo,
    ShareResult,
    TreeResult,
    WriteResult,
)
from grover.fs.utils import normalize_path
from grover.fs.vfs import VFS
from grover.graph._rustworkx import RustworkxGraph
from grover.graph.analyzers import AnalyzerRegistry
from grover.models.chunks import FileChunk
from grover.models.edges import GroverEdge
from grover.models.embeddings import Embedding
from grover.models.files import File, FileVersion
from grover.models.shares import FileShare
from grover.search._engine import SearchEngine
from grover.search.extractors import extract_from_chunks, extract_from_file

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from grover.fs.protocol import StorageBackend
    from grover.graph.protocols import GraphStore
    from grover.graph.types import SubgraphResult
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase
    from grover.ref import Ref
    from grover.search.types import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".grover" / "_default"


class GroverAsync:
    """Async facade wiring filesystem, graph, analyzers, event bus, and search.

    Mount-first API: create an instance, then mount backends.

    Engine-based DB mount (primary API)::

        engine = create_async_engine("postgresql+asyncpg://...")
        g = GroverAsync(data_dir="/myapp/.grover")
        await g.mount("/data", engine=engine)

    Direct access — auto-commits per operation::

        g = GroverAsync()
        await g.mount("/app", backend)
        await g.write("/app/test.py", "print('hi')")
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        embedding_provider: Any = None,
        vector_store: Any = None,
    ) -> None:
        self._explicit_data_dir = Path(data_dir) if data_dir else None
        self._closed = False

        # Core subsystems (sync init)
        self._event_bus = EventBus()
        self._registry = MountRegistry()
        self._vfs = VFS(self._registry, self._event_bus)
        self._analyzer_registry = AnalyzerRegistry()

        # Internal metadata mount — lazily created on first mount()
        self._meta_fs: LocalFileSystem | None = None
        self._meta_data_dir: Path | None = None

        # Search configuration (per-mount engines created at mount time)
        self._embedding_provider = self._resolve_embedding_provider(embedding_provider)
        self._explicit_vector_store = vector_store

        # Register event handlers
        self._event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._event_bus.register(EventType.FILE_RESTORED, self._on_file_restored)

    # ------------------------------------------------------------------
    # Search / Graph factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_embedding_provider(provider: Any) -> Any:
        """Return the provider if given, else auto-discover, else None."""
        if provider is not None:
            return provider
        try:
            from grover.search.providers.sentence_transformers import (
                SentenceTransformerEmbedding,
            )

            return SentenceTransformerEmbedding()
        except Exception:
            logger.debug("No embedding provider available; search disabled")
            return None

    def _create_search_engine(self) -> SearchEngine | None:
        """Create a new SearchEngine for a mount."""
        if self._explicit_vector_store is not None:
            return SearchEngine(self._explicit_vector_store, self._embedding_provider)
        if self._embedding_provider is None:
            return None
        from grover.search.stores.local import LocalVectorStore

        store = LocalVectorStore(dimension=self._embedding_provider.dimensions)
        return SearchEngine(store, self._embedding_provider)

    # ------------------------------------------------------------------
    # Per-mount graph / search resolution
    # ------------------------------------------------------------------

    def _resolve_graph(self, path: str) -> GraphStore:
        """Return the graph for the mount owning *path*."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            msg = f"No mount found for path: {path!r}"
            raise RuntimeError(msg) from None
        graph = getattr(mount.backend, "_graph", None)
        if graph is None:
            msg = f"No graph on mount at {mount.mount_path}"
            raise RuntimeError(msg)
        return graph

    def _resolve_search_engine(self, path: str) -> SearchEngine | None:
        """Return the search engine for the mount owning *path*, or None."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return None
        return getattr(mount.backend, "_search_engine", None)

    def _resolve_graph_any(self, path: str | None = None) -> GraphStore:
        """Get graph for a specific path, or first available mount's graph."""
        if path is not None:
            return self._resolve_graph(path)
        for mount in self._registry.list_visible_mounts():
            graph = getattr(mount.backend, "_graph", None)
            if graph is not None:
                return graph
        msg = "No graph available on any mount"
        raise RuntimeError(msg)

    def get_graph(self, path: str | None = None) -> GraphStore:
        """Return the graph for the mount owning *path*, or the first available.

        This replaces the old ``self.graph`` attribute which was removed
        in favour of per-mount graphs.
        """
        return self._resolve_graph_any(path)

    # ------------------------------------------------------------------
    # Mount / Unmount
    # ------------------------------------------------------------------

    async def mount(
        self,
        path: str,
        backend: StorageBackend | None = None,
        *,
        engine: AsyncEngine | None = None,
        session_factory: Callable[..., AsyncSession] | None = None,
        dialect: str = "sqlite",
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
        db_schema: str | None = None,
        mount_type: str | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        hidden: bool = False,
    ) -> None:
        """Mount a backend at *path*.

        When *engine* is provided, a session factory is created from it.
        If *backend* is also provided, that backend is used; otherwise a
        plain ``DatabaseFileSystem`` is created.  For user-scoped mounts,
        pass a ``UserScopedFileSystem`` as *backend* together with *engine*.
        """
        if engine is not None:
            if session_factory is not None:
                raise ValueError("Provide engine or session_factory, not both")
            config = await self._create_engine_mount(
                path,
                engine,
                backend,
                file_model,
                file_version_model,
                file_chunk_model,
                db_schema,
                mount_type,
                permission,
                label,
                hidden,
            )
        elif session_factory is not None:
            config = self._create_session_factory_mount(
                path,
                session_factory,
                backend,
                dialect,
                file_model,
                file_version_model,
                file_chunk_model,
                db_schema,
                mount_type,
                permission,
                label,
                hidden,
            )
        else:
            if backend is None:
                raise ValueError("Provide backend, engine, or session_factory")
            config = await self._create_backend_mount(
                path,
                backend,
                mount_type,
                permission,
                label,
                hidden,
            )

        # Inject per-mount graph and search engine (skip for hidden meta mount)
        if not hidden:
            graph = RustworkxGraph()
            search_engine = self._create_search_engine()
            config.backend._graph = graph  # type: ignore[union-attr]
            config.backend._search_engine = search_engine  # type: ignore[union-attr]

        # Call open() on the backend BEFORE registering (skip if already opened for LFS)
        if not isinstance(config.backend, LocalFileSystem) and hasattr(config.backend, "open"):
            await config.backend.open()

        self._registry.add_mount(config)

        # Lazily initialise meta_fs on first non-hidden mount
        if not hidden and self._meta_fs is None:
            await self._init_meta_fs(config.backend)

        # Load existing graph + search state for this mount
        if not hidden:
            await self._load_mount_state(config)

    async def _create_engine_mount(
        self,
        path: str,
        engine: AsyncEngine,
        backend: StorageBackend | None,
        file_model: type[FileBase] | None,
        file_version_model: type[FileVersionBase] | None,
        file_chunk_model: type[FileChunkBase] | None,
        db_schema: str | None,
        mount_type: str | None,
        permission: Permission,
        label: str,
        hidden: bool,
    ) -> MountConfig:
        """Build a MountConfig from an async engine."""
        sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dialect = engine.dialect.name

        # Ensure base tables exist
        fm = file_model or File
        fvm = file_version_model or FileVersion
        fcm = file_chunk_model or FileChunk
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: fm.__table__.create(c, checkfirst=True)  # type: ignore[attr-defined]
            )
            await conn.run_sync(
                lambda c: fvm.__table__.create(c, checkfirst=True)  # type: ignore[attr-defined]
            )
            await conn.run_sync(
                lambda c: fcm.__table__.create(c, checkfirst=True)  # type: ignore[attr-defined]
            )
            # Edges table for per-mount graph persistence
            await conn.run_sync(
                lambda c: GroverEdge.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )

        if backend is None:
            backend = DatabaseFileSystem(
                dialect=dialect,
                file_model=file_model,
                file_version_model=file_version_model,
                file_chunk_model=file_chunk_model,
                schema=db_schema,
            )

        # Create share table if backend supports sharing
        if isinstance(backend, SupportsReBAC):
            async with engine.begin() as conn:
                await conn.run_sync(
                    lambda c: FileShare.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
                )

        return MountConfig(
            mount_path=path,
            backend=backend,
            session_factory=sf,
            mount_type=mount_type or "vfs",
            permission=permission,
            label=label,
            hidden=hidden,
        )

    def _create_session_factory_mount(
        self,
        path: str,
        session_factory: Callable[..., AsyncSession],
        backend: StorageBackend | None,
        dialect: str,
        file_model: type[FileBase] | None,
        file_version_model: type[FileVersionBase] | None,
        file_chunk_model: type[FileChunkBase] | None,
        db_schema: str | None,
        mount_type: str | None,
        permission: Permission,
        label: str,
        hidden: bool,
    ) -> MountConfig:
        """Build a MountConfig from a caller-provided session factory."""
        if backend is None:
            backend = DatabaseFileSystem(
                dialect=dialect,
                file_model=file_model,
                file_version_model=file_version_model,
                file_chunk_model=file_chunk_model,
                schema=db_schema,
            )

        return MountConfig(
            mount_path=path,
            backend=backend,
            session_factory=session_factory,
            mount_type=mount_type or "vfs",
            permission=permission,
            label=label,
            hidden=hidden,
        )

    async def _create_backend_mount(
        self,
        path: str,
        backend: StorageBackend,
        mount_type: str | None,
        permission: Permission,
        label: str,
        hidden: bool,
    ) -> MountConfig:
        """Build a MountConfig from a pre-constructed backend."""
        if mount_type is None:
            mount_type = "local" if isinstance(backend, LocalFileSystem) else "vfs"

        # For local backends, eagerly init DB and expose session_factory
        sf: Callable[..., AsyncSession] | None = None
        if isinstance(backend, LocalFileSystem):
            await backend.open()
            sf = backend.session_factory

        return MountConfig(
            mount_path=path,
            backend=backend,
            session_factory=sf,
            mount_type=mount_type,
            permission=permission,
            label=label,
            hidden=hidden,
        )

    async def unmount(self, path: str) -> None:
        """Unmount the backend at *path*."""

        path = normalize_path(path).rstrip("/")
        if path == "/.grover":
            raise ValueError("Cannot unmount /.grover")

        try:
            mount, _ = self._registry.resolve(path)
        except MountNotFoundError:
            return

        # Only unmount if the path is an exact mount point, not a subpath
        if mount.mount_path != path:
            return

        backend = mount.backend
        if hasattr(backend, "close"):
            await backend.close()
        self._registry.remove_mount(path)

    # ------------------------------------------------------------------
    # Internal metadata mount
    # ------------------------------------------------------------------

    async def _init_meta_fs(self, first_backend: Any) -> None:
        """Create the internal /.grover metadata mount."""
        if self._explicit_data_dir is not None:
            data_dir = self._explicit_data_dir
        elif isinstance(first_backend, LocalFileSystem):
            data_dir = first_backend.data_dir
        else:
            data_dir = _DEFAULT_DATA_DIR

        self._meta_data_dir = data_dir

        self._meta_fs = LocalFileSystem(
            workspace_dir=data_dir,
            data_dir=data_dir / "_meta",
        )

        # Eagerly init DB
        await self._meta_fs.open()

        self._registry.add_mount(
            MountConfig(
                mount_path="/.grover",
                backend=self._meta_fs,
                session_factory=self._meta_fs.session_factory,
                mount_type="local",
                hidden=True,
            )
        )

        # Create extra tables on the meta engine
        await self._ensure_extra_tables()

    async def _ensure_extra_tables(self) -> None:
        if self._meta_fs is None:
            return
        await self._meta_fs.open()
        engine = self._meta_fs.engine
        if engine is None:
            return

        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: GroverEdge.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )
            await conn.run_sync(
                lambda c: Embedding.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )

    # ------------------------------------------------------------------
    # Per-mount state loading
    # ------------------------------------------------------------------

    async def _load_mount_state(self, mount: MountConfig) -> None:
        """Load graph and search state for a single mount."""
        graph = getattr(mount.backend, "_graph", None)
        if graph is None:
            return

        # Load graph: file nodes from mount's DB, edges from mount's DB
        if mount.has_session_factory:
            try:
                from grover.graph.protocols import SupportsPersistence

                if isinstance(graph, SupportsPersistence):
                    file_model = getattr(mount.backend, "file_model", None) or File
                    async with self._vfs.session_for(mount) as session:
                        if session is not None:
                            await graph.from_sql(session, file_model=file_model)
            except Exception:
                logger.debug(
                    "No existing graph state to load for %s",
                    mount.mount_path,
                    exc_info=True,
                )

        # Load search index from disk
        search_engine = getattr(mount.backend, "_search_engine", None)
        if search_engine is not None and self._meta_data_dir is not None:
            slug = mount.mount_path.strip("/").replace("/", "_") or "_default"
            search_dir = self._meta_data_dir / "search" / slug
            meta_file = search_dir / "search_meta.json"
            if meta_file.exists():
                try:
                    search_engine.load(str(search_dir))
                except Exception:
                    logger.debug(
                        "Failed to load search index for %s",
                        mount.mount_path,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_file_written(self, event: FileEvent) -> None:
        if self._meta_fs is None:
            return
        if "/.grover/" in event.path:
            return
        content = event.content
        if content is None:
            result = await self._vfs.read(event.path)
            if not result.success:
                return
            content = result.content
        if content is not None:
            await self._analyze_and_integrate(event.path, content)

    async def _on_file_deleted(self, event: FileEvent) -> None:
        if self._meta_fs is None:
            return
        if "/.grover/" in event.path:
            return
        try:
            graph = self._resolve_graph(event.path)
            if graph.has_node(event.path):
                graph.remove_file_subgraph(event.path)
        except RuntimeError:
            pass  # Mount may not have a graph
        try:
            search_engine = self._resolve_search_engine(event.path)
            if search_engine is not None:
                await search_engine.remove_file(event.path)
        except RuntimeError:
            pass

    async def _on_file_moved(self, event: FileEvent) -> None:
        if self._meta_fs is None:
            return
        if event.old_path and "/.grover/" not in event.old_path:
            try:
                graph = self._resolve_graph(event.old_path)
                if graph.has_node(event.old_path):
                    graph.remove_file_subgraph(event.old_path)
            except RuntimeError:
                pass
            try:
                search_engine = self._resolve_search_engine(event.old_path)
                if search_engine is not None:
                    await search_engine.remove_file(event.old_path)
            except RuntimeError:
                pass

        if "/.grover/" in event.path:
            return
        result = await self._vfs.read(event.path)
        if result.success:
            content = result.content
            if content is not None:
                await self._analyze_and_integrate(event.path, content)

    async def _on_file_restored(self, event: FileEvent) -> None:
        await self._on_file_written(event)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _analyze_and_integrate(self, path: str, content: str) -> dict[str, int]:
        stats = {"chunks_created": 0, "edges_added": 0}

        try:
            graph = self._resolve_graph(path)
        except RuntimeError:
            return stats

        search_engine = self._resolve_search_engine(path)

        if graph.has_node(path):
            graph.remove_file_subgraph(path)
        if search_engine is not None:
            await search_engine.remove_file(path)

        graph.add_node(path)

        analysis = self._analyzer_registry.analyze_file(path, content)

        if analysis is not None:
            chunks, edges = analysis

            for chunk in chunks:
                await self._vfs.write(chunk.chunk_path, chunk.content)
                graph.add_node(
                    chunk.chunk_path,
                    parent_path=path,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    name=chunk.name,
                )
                graph.add_edge(path, chunk.chunk_path, edge_type="contains")
                stats["chunks_created"] += 1

            for edge in edges:
                meta: dict[str, Any] = dict(edge.metadata)
                graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **meta)
                stats["edges_added"] += 1

            if search_engine is not None:
                embeddable = extract_from_chunks(chunks)
                if embeddable:
                    await search_engine.add_batch(embeddable)
        else:
            if search_engine is not None:
                embeddable = extract_from_file(path, content)
                if embeddable:
                    await search_engine.add_batch(embeddable)

        return stats

    # ------------------------------------------------------------------
    # FS Operations
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        user_id: str | None = None,
    ) -> ReadResult:
        return await self._vfs.read(path, offset, limit, user_id=user_id)

    async def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        try:
            return await self._vfs.write(path, content, overwrite=overwrite, user_id=user_id)
        except Exception as e:
            return WriteResult(success=False, message=f"Write failed: {e}")

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> EditResult:
        try:
            return await self._vfs.edit(path, old, new, replace_all, user_id=user_id)
        except Exception as e:
            return EditResult(success=False, message=f"Edit failed: {e}")

    async def delete(
        self, path: str, permanent: bool = False, *, user_id: str | None = None
    ) -> DeleteResult:
        try:
            return await self._vfs.delete(path, permanent, user_id=user_id)
        except Exception as e:
            return DeleteResult(success=False, message=f"Delete failed: {e}")

    async def list_dir(
        self, path: str = "/", *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        result = await self._vfs.list_dir(path, user_id=user_id)
        return [
            {"path": e.path, "name": e.name, "is_directory": e.is_directory} for e in result.entries
        ]

    async def exists(self, path: str, *, user_id: str | None = None) -> bool:
        return await self._vfs.exists(path, user_id=user_id)

    async def move(
        self, src: str, dest: str, *, user_id: str | None = None, follow: bool = False
    ) -> MoveResult:
        try:
            return await self._vfs.move(src, dest, user_id=user_id, follow=follow)
        except Exception as e:
            return MoveResult(success=False, message=f"Move failed: {e}")

    async def copy(self, src: str, dest: str, *, user_id: str | None = None) -> WriteResult:
        try:
            return await self._vfs.copy(src, dest, user_id=user_id)
        except Exception as e:
            return WriteResult(success=False, message=f"Copy failed: {e}")

    # ------------------------------------------------------------------
    # Search / Query operations
    # ------------------------------------------------------------------

    async def glob(
        self, pattern: str, path: str = "/", *, user_id: str | None = None
    ) -> GlobResult:
        try:
            return await self._vfs.glob(pattern, path, user_id=user_id)
        except Exception as e:
            return GlobResult(
                success=False, message=f"Glob failed: {e}", pattern=pattern, path=path
            )

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        user_id: str | None = None,
    ) -> GrepResult:
        try:
            return await self._vfs.grep(
                pattern,
                path,
                glob_filter=glob_filter,
                case_sensitive=case_sensitive,
                fixed_string=fixed_string,
                invert=invert,
                word_match=word_match,
                context_lines=context_lines,
                max_results=max_results,
                max_results_per_file=max_results_per_file,
                count_only=count_only,
                files_only=files_only,
                user_id=user_id,
            )
        except Exception as e:
            return GrepResult(
                success=False, message=f"Grep failed: {e}", pattern=pattern, path=path
            )

    async def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        try:
            return await self._vfs.tree(path, max_depth=max_depth, user_id=user_id)
        except Exception as e:
            return TreeResult(success=False, message=f"Tree failed: {e}", path=path)

    # ------------------------------------------------------------------
    # Version operations (normalize exceptions to Results)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> ListVersionsResult:
        try:
            return await self._vfs.list_versions(path, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return ListVersionsResult(success=False, versions=[], message=str(e))

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        try:
            return await self._vfs.get_version_content(path, version, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return GetVersionContentResult(success=False, content=None, message=str(e))

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        try:
            return await self._vfs.restore_version(path, version, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    # ------------------------------------------------------------------
    # Trash operations (normalize exceptions to Results)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> Any:
        return await self._vfs.list_trash(user_id=user_id)

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        try:
            return await self._vfs.restore_from_trash(path, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        return await self._vfs.empty_trash(user_id=user_id)

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
        expires_at: Any = None,
    ) -> ShareResult:
        """Share a file or directory with another user.

        Requires a backend that supports sharing (e.g. ``UserScopedFileSystem``).
        """

        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._vfs.get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._vfs.session_for(mount) as sess:
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
            share=ShareInfo(
                path=path,
                grantee_id=share_info.grantee_id,
                permission=share_info.permission,
                granted_by=share_info.granted_by,
                created_at=share_info.created_at,
                expires_at=share_info.expires_at,
            ),
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
        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._vfs.get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._vfs.session_for(mount) as sess:
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
    ) -> ListSharesResult:
        """List all shares on a given path."""

        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ListSharesResult(success=False, message=str(e))

        cap = self._vfs.get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ListSharesResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._vfs.session_for(mount) as sess:
            assert sess is not None
            shares = await cap.list_shares_on_path(rel_path, user_id=user_id, session=sess)

        return ListSharesResult(
            success=True,
            message=f"Found {len(shares)} share(s)",
            shares=[
                ShareInfo(
                    path=path,
                    grantee_id=s.grantee_id,
                    permission=s.permission,
                    granted_by=s.granted_by,
                    created_at=s.created_at,
                    expires_at=s.expires_at,
                )
                for s in shares
            ],
        )

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
    ) -> ListSharesResult:
        """List all files shared with the current user across all mounts."""
        all_shares: list[ShareInfo] = []
        for mount in self._registry.list_mounts():
            cap = self._vfs.get_capability(mount.backend, SupportsReBAC)
            if cap is None:
                continue
            async with self._vfs.session_for(mount) as sess:
                assert sess is not None
                shares = await cap.list_shared_with_me(user_id=user_id, session=sess)
            for s in shares:
                # Backend returns paths like /@shared/alice/a.md — prepend mount
                full_path = mount.mount_path + s.path if s.path != "/" else mount.mount_path
                all_shares.append(
                    ShareInfo(
                        path=full_path,
                        grantee_id=s.grantee_id,
                        permission=s.permission,
                        granted_by=s.granted_by,
                        created_at=s.created_at,
                        expires_at=s.expires_at,
                    )
                )

        return ListSharesResult(
            success=True,
            message=f"Found {len(all_shares)} share(s)",
            shares=all_shares,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        return await self._vfs.reconcile(mount_path)

    # ------------------------------------------------------------------
    # Graph query wrappers (resolve mount → delegate to backend's graph)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Ref]:
        return self._resolve_graph(path).dependents(path)

    def dependencies(self, path: str) -> list[Ref]:
        return self._resolve_graph(path).dependencies(path)

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        return self._resolve_graph(path).impacts(path, max_depth)

    def path_between(self, source: str, target: str) -> list[Ref] | None:
        return self._resolve_graph(source).path_between(source, target)

    def contains(self, path: str) -> list[Ref]:
        return self._resolve_graph(path).contains(path)

    # ------------------------------------------------------------------
    # Graph algorithm wrappers (capability-checked)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        personalization: dict[str, float] | None = None,
        path: str | None = None,
    ) -> dict[str, float]:
        """Run PageRank on the knowledge graph.

        *path* selects which mount's graph to use (defaults to first visible).
        Raises :class:`~grover.fs.exceptions.CapabilityNotSupportedError` if
        the graph backend does not support centrality algorithms.
        """
        from grover.graph.protocols import SupportsCentrality

        graph = self._resolve_graph_any(path)
        if not isinstance(graph, SupportsCentrality):
            msg = "Graph backend does not support centrality algorithms"
            raise CapabilityNotSupportedError(msg)
        return graph.pagerank(personalization=personalization)

    def ancestors(self, path: str) -> set[str]:
        """All transitive predecessors of *path* in the knowledge graph."""
        from grover.graph.protocols import SupportsTraversal

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsTraversal):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg)
        return graph.ancestors(path)

    def descendants(self, path: str) -> set[str]:
        """All transitive successors of *path* in the knowledge graph."""
        from grover.graph.protocols import SupportsTraversal

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsTraversal):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg)
        return graph.descendants(path)

    def meeting_subgraph(
        self,
        paths: list[str],
        *,
        max_size: int = 50,
    ) -> SubgraphResult:
        """Extract the subgraph connecting *paths* via shortest paths."""
        from grover.graph.protocols import SupportsSubgraph

        graph = self._resolve_graph_any(paths[0] if paths else None)
        if not isinstance(graph, SupportsSubgraph):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg)
        return graph.meeting_subgraph(paths, max_size=max_size)

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> SubgraphResult:
        """Extract the neighborhood subgraph around *path*."""
        from grover.graph.protocols import SupportsSubgraph

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsSubgraph):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg)
        return graph.neighborhood(
            path,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
        )

    def find_nodes(self, *, path: str | None = None, **attrs: Any) -> list[str]:
        """Find graph nodes matching all attribute predicates."""
        from grover.graph.protocols import SupportsFiltering

        graph = self._resolve_graph_any(path)
        if not isinstance(graph, SupportsFiltering):
            msg = "Graph backend does not support filtering"
            raise CapabilityNotSupportedError(msg)
        return graph.find_nodes(**attrs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self, query: str, k: int = 10, *, path: str = "/", user_id: str | None = None
    ) -> list[SearchResult]:
        """Semantic search, routed through VFS to per-mount search engines."""
        # Check if any mount has a search engine
        has_search = False
        for mount in self._registry.list_visible_mounts():
            if getattr(mount.backend, "_search_engine", None) is not None:
                has_search = True
                break
        if not has_search:
            msg = (
                "Search is not available: no embedding provider configured. "
                "Install sentence-transformers or pass embedding_provider= to GroverAsync()."
            )
            raise RuntimeError(msg)
        return await self._vfs.search(query, k, path=path, user_id=user_id)

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    async def index(self, mount_path: str | None = None) -> dict[str, int]:
        stats = {"files_scanned": 0, "chunks_created": 0, "edges_added": 0}

        if mount_path is not None:
            await self._walk_and_index(mount_path, stats)
        else:
            for mount in self._registry.list_visible_mounts():
                await self._walk_and_index(mount.mount_path, stats)

        await self._async_save()
        return stats

    async def _walk_and_index(self, path: str, stats: dict[str, int]) -> None:
        result = await self._vfs.list_dir(path)
        if not result.success:
            return

        for entry in result.entries:
            if "/.grover/" in entry.path:
                continue
            if entry.is_directory:
                await self._walk_and_index(entry.path, stats)
            else:
                content = await self._read_file_content(entry.path)
                if content is None:
                    continue
                file_stats = await self._analyze_and_integrate(entry.path, content)
                stats["files_scanned"] += 1
                stats["chunks_created"] += file_stats["chunks_created"]
                stats["edges_added"] += file_stats["edges_added"]

    async def _read_file_content(self, path: str) -> str | None:
        read_result = await self._vfs.read(path)
        if read_result.success:
            return read_result.content

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        backend = mount.backend
        if hasattr(backend, "_read_content"):
            if mount.has_session_factory:
                async with self._vfs.session_for(mount) as sess:
                    content: str | None = await backend._read_content(rel_path, sess)  # type: ignore[union-attr]
            else:
                content = await backend._read_content(rel_path, None)  # type: ignore[union-attr]
            return content

        return None

    async def save(self) -> None:
        await self._async_save()

    async def _async_save(self) -> None:
        """Save per-mount graph and search state."""
        # Save each mount's graph to its own DB
        for mount in self._registry.list_visible_mounts():
            graph = getattr(mount.backend, "_graph", None)
            if graph is not None and mount.has_session_factory:
                from grover.graph.protocols import SupportsPersistence

                if isinstance(graph, SupportsPersistence):
                    try:
                        async with self._vfs.session_for(mount) as session:
                            if session is not None:
                                await graph.to_sql(session)
                    except Exception:
                        logger.debug(
                            "Failed to save graph for %s",
                            mount.mount_path,
                            exc_info=True,
                        )

            # Save search index to disk
            search_engine = getattr(mount.backend, "_search_engine", None)
            if search_engine is not None and self._meta_data_dir is not None:
                slug = mount.mount_path.strip("/").replace("/", "_") or "_default"
                search_dir = self._meta_data_dir / "search" / slug
                try:
                    search_engine.save(str(search_dir))
                except Exception:
                    logger.debug(
                        "Failed to save search index for %s",
                        mount.mount_path,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        await self._async_save()
        await self._vfs.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fs(self) -> VFS:
        return self._vfs

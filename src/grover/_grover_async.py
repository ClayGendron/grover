"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grover.events import EventBus, EventType, FileEvent
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import CapabilityNotSupportedError, MountNotFoundError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import (
    SupportsFileChunks,
    SupportsReBAC,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.fs.query_types import (
    ChunkMatch,
    SearchHit,
    SearchQueryResult,
)
from grover.fs.types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    ListResult,
    ListSharesResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareInfo,
    ShareResult,
    WriteResult,
)
from grover.fs.utils import normalize_path
from grover.graph._rustworkx import RustworkxGraph
from grover.graph.analyzers import AnalyzerRegistry
from grover.models.chunks import FileChunk
from grover.models.connections import FileConnection
from grover.models.embeddings import Embedding
from grover.models.files import File, FileVersion
from grover.models.shares import FileShare
from grover.search._engine import SearchEngine
from grover.search.extractors import extract_from_chunks, extract_from_file
from grover.search.results import GlobResult, GrepResult, ListDirResult, TreeResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from grover.fs.protocol import StorageBackend
    from grover.graph.protocols import GraphStore
    from grover.graph.types import SubgraphResult
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase
    from grover.ref import Ref

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
    # Session management & helpers (absorbed from VFS)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _session_for(self, mount: MountConfig) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or ``None`` for non-SQL backends."""
        if not mount.has_session_factory:
            yield None
            return

        assert mount.session_factory is not None
        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def _emit(self, event: FileEvent) -> None:
        """Emit a file event via the event bus."""
        await self._event_bus.emit(event)

    def _check_writable(self, virtual_path: str) -> None:
        """Raise ``PermissionError`` if *virtual_path* is on a read-only mount."""
        perm = self._registry.get_permission(virtual_path)
        if perm == Permission.READ_ONLY:
            raise PermissionError(f"Cannot write to read-only path: {virtual_path}")

    @staticmethod
    def _get_capability(backend: Any, protocol: type[T]) -> T | None:
        """Return *backend* if it satisfies *protocol*, else ``None``."""
        if isinstance(backend, protocol):
            return backend
        return None

    def _prefix_path(self, path: str | None, mount_path: str) -> str | None:
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def _prefix_file_info(self, info: FileInfo, mount: MountConfig) -> FileInfo:
        prefixed_path = self._prefix_path(info.path, mount.mount_path) or info.path
        info.path = prefixed_path
        info.mount_type = mount.mount_type
        info.permission = self._registry.get_permission(prefixed_path).value
        return info

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

    def _create_search_engine(self, *, lexical: Any | None = None) -> SearchEngine | None:
        """Create a new SearchEngine for a mount."""
        vector: Any = None
        embedding = self._embedding_provider

        if self._explicit_vector_store is not None:
            vector = self._explicit_vector_store
        elif embedding is not None:
            from grover.search.stores.local import LocalVectorStore

            vector = LocalVectorStore(dimension=embedding.dimensions)

        # If we have nothing at all, no search engine
        if vector is None and embedding is None and lexical is None:
            return None

        return SearchEngine(vector=vector, embedding=embedding, lexical=lexical)

    async def _create_fulltext_store(self, config: MountConfig) -> Any | None:
        """Create a FullTextStore for the mount based on its dialect."""
        from grover.search.fulltext.sqlite import SQLiteFullTextStore

        if isinstance(config.backend, LocalFileSystem):
            engine = getattr(config.backend, "_engine", None)
            if engine is not None:
                fts = SQLiteFullTextStore(engine=engine)
                await fts.ensure_table()
                return fts
            return None

        if config.has_session_factory:
            # Detect dialect from session factory's engine if available
            sf = config.session_factory
            bind = getattr(sf, "kw", {}).get("bind", None)
            if bind is None:
                bind = getattr(sf, "class_", None)
                if bind is None:
                    return None
                bind = getattr(bind, "bind", None)
            if bind is None:
                return None

            from grover.fs.dialect import get_dialect

            dialect = get_dialect(bind)
            if dialect == "sqlite":
                fts = SQLiteFullTextStore(engine=bind)
                await fts.ensure_table()
                return fts
            if dialect == "postgresql":
                from grover.search.fulltext.postgres import PostgresFullTextStore

                fts_pg = PostgresFullTextStore(engine=bind)
                await fts_pg.ensure_table()
                return fts_pg
            if dialect == "mssql":
                from grover.search.fulltext.mssql import MSSQLFullTextStore

                fts_mssql = MSSQLFullTextStore(engine=bind)
                await fts_mssql.ensure_table()
                return fts_mssql

        return None

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
        if mount.graph is None:
            msg = f"No graph on mount at {mount.mount_path}"
            raise RuntimeError(msg)
        return mount.graph

    def _resolve_search_engine(self, path: str) -> SearchEngine | None:
        """Return the search engine for the mount owning *path*, or None."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return None
        return mount.search

    def _resolve_graph_any(self, path: str | None = None) -> GraphStore:
        """Get graph for a specific path, or first available mount's graph."""
        if path is not None:
            return self._resolve_graph(path)
        for mount in self._registry.list_visible_mounts():
            if mount.graph is not None:
                return mount.graph
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

    async def add_mount(
        self,
        mount_or_path: Any = None,
        *,
        filesystem: StorageBackend | None = None,
        graph: GraphStore | None = None,
        search: SearchEngine | None = None,
        session_factory: Callable[..., AsyncSession] | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        hidden: bool = False,
        path: str | None = None,
    ) -> None:
        """Register a pre-built :class:`~grover.mount.Mount` or build one from kwargs.

        Usage::

            # From a Mount object
            mount = Mount(path="/project", filesystem=LocalFileSystem(...))
            await g.add_mount(mount)

            # From keyword arguments
            await g.add_mount(
                path="/data",
                filesystem=DatabaseFileSystem(dialect="postgresql"),
                graph=RustworkxGraph(),
            )
        """
        from grover.mount import Mount

        if isinstance(mount_or_path, Mount):
            new_mount = mount_or_path
        else:
            # Resolve path: either from positional arg or keyword
            actual_path = mount_or_path if mount_or_path is not None else path
            if actual_path is None or filesystem is None:
                raise ValueError("Provide a Mount object or (path + filesystem)")

            new_mount = Mount(
                path=actual_path,
                filesystem=filesystem,
                graph=graph,
                search=search,
                session_factory=session_factory,
                permission=permission,
                label=label,
                hidden=hidden,
            )

        # Auto-create graph if not provided and not hidden
        if not new_mount.hidden and new_mount.graph is None:
            new_mount.graph = RustworkxGraph()

        # Auto-create search engine if not provided and not hidden
        if not new_mount.hidden and new_mount.search is None:
            se = self._create_search_engine()
            if se is not None:
                new_mount.search = se

        # Call open() on the filesystem if needed
        if not isinstance(new_mount.filesystem, LocalFileSystem) and hasattr(
            new_mount.filesystem, "open"
        ):
            await new_mount.filesystem.open()

        self._registry.add_mount(new_mount)

        # Lazily initialise meta_fs on first non-hidden mount
        if not new_mount.hidden and self._meta_fs is None:
            await self._init_meta_fs(new_mount.filesystem)

        # Load existing graph + search state for this mount
        if not new_mount.hidden:
            await self._load_mount_state(new_mount)

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
            config.graph = RustworkxGraph()
            lexical = await self._create_fulltext_store(config)
            config.search = self._create_search_engine(lexical=lexical)

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
                lambda c: FileConnection.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
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
                lambda c: FileConnection.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )
            await conn.run_sync(
                lambda c: Embedding.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )

    # ------------------------------------------------------------------
    # Per-mount state loading
    # ------------------------------------------------------------------

    async def _load_mount_state(self, mount: MountConfig) -> None:
        """Load graph and search state for a single mount."""
        graph = mount.graph
        if graph is None:
            return

        # Load graph: file nodes from mount's DB, edges from mount's DB
        if mount.has_session_factory:
            try:
                from grover.graph.protocols import SupportsPersistence

                if isinstance(graph, SupportsPersistence):
                    file_model = getattr(mount.backend, "file_model", None) or File
                    async with self._session_for(mount) as session:
                        if session is not None:
                            await graph.from_sql(session, file_model=file_model)
            except Exception:
                logger.debug(
                    "No existing graph state to load for %s",
                    mount.mount_path,
                    exc_info=True,
                )

        # Load search index from disk
        search_engine = mount.search
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
            result = await self.read(event.path)
            if not result.success:
                return
            content = result.content
        if content is not None:
            await self._analyze_and_integrate(event.path, content, user_id=event.user_id)

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
                mount, _rel = self._registry.resolve(event.path)
                async with self._session_for(mount) as sess:
                    await search_engine.remove_file(event.path, session=sess)
        except RuntimeError:
            pass
        # Clean up chunk DB rows
        await self._delete_chunks_for_path(event.path)

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
                    mount, _rel = self._registry.resolve(event.old_path)
                    async with self._session_for(mount) as sess:
                        await search_engine.remove_file(event.old_path, session=sess)
            except RuntimeError:
                pass
            # Clean up chunk DB rows for old path
            await self._delete_chunks_for_path(event.old_path)

        if "/.grover/" in event.path:
            return
        result = await self.read(event.path)
        if result.success:
            content = result.content
            if content is not None:
                await self._analyze_and_integrate(event.path, content, user_id=event.user_id)

    async def _on_file_restored(self, event: FileEvent) -> None:
        await self._on_file_written(event)

    async def _delete_chunks_for_path(self, path: str) -> None:
        """Delete chunk DB rows for *path* if the backend supports it."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return
        if isinstance(mount.backend, SupportsFileChunks):
            async with self._session_for(mount) as sess:
                await mount.backend.delete_file_chunks(path, session=sess)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _analyze_and_integrate(
        self, path: str, content: str, *, user_id: str | None = None
    ) -> dict[str, int]:
        import hashlib

        stats = {"chunks_created": 0, "edges_added": 0}

        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return stats

        graph = mount.graph
        if graph is None:
            return stats

        search_engine = mount.search

        if graph.has_node(path):
            graph.remove_file_subgraph(path)
        if search_engine is not None:
            async with self._session_for(mount) as sess:
                await search_engine.remove_file(path, session=sess)

        graph.add_node(path)

        analysis = self._analyzer_registry.analyze_file(path, content)

        if analysis is not None:
            chunks, edges = analysis

            # Write chunk DB rows instead of VFS files
            if isinstance(mount.backend, SupportsFileChunks) and chunks:
                chunk_dicts = [
                    {
                        "chunk_path": chunk.chunk_path,
                        "name": chunk.name,
                        "description": "",
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "content": chunk.content,
                        "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                    }
                    for chunk in chunks
                ]
                async with self._session_for(mount) as sess:
                    await mount.backend.replace_file_chunks(
                        path, chunk_dicts, session=sess, user_id=user_id
                    )

            for chunk in chunks:
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
                    async with self._session_for(mount) as sess:
                        await search_engine.add_batch(embeddable, session=sess)
        else:
            if search_engine is not None:
                embeddable = extract_from_file(path, content)
                if embeddable:
                    async with self._session_for(mount) as sess:
                        await search_engine.add_batch(embeddable, session=sess)

        return stats

    # ------------------------------------------------------------------
    # FS Operations (absorbed from VFS)
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        user_id: str | None = None,
    ) -> ReadResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.read(
                rel_path, offset, limit, session=sess, user_id=user_id
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        try:
            mount, rel_path = self._registry.resolve(path)
            async with self._session_for(mount) as sess:
                result = await mount.backend.write(
                    rel_path,
                    content,
                    "agent",
                    overwrite=overwrite,
                    session=sess,
                    user_id=user_id,
                )
            result.file_path = self._prefix_path(result.file_path, mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(
                        event_type=EventType.FILE_WRITTEN,
                        path=path,
                        content=content,
                        user_id=user_id,
                    )
                )
            return result
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
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return EditResult(success=False, message=str(e))

        try:
            mount, rel_path = self._registry.resolve(path)
            async with self._session_for(mount) as sess:
                result = await mount.backend.edit(
                    rel_path,
                    old,
                    new,
                    replace_all,
                    "agent",
                    session=sess,
                    user_id=user_id,
                )
            result.file_path = self._prefix_path(result.file_path, mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=path, user_id=user_id)
                )
            return result
        except Exception as e:
            return EditResult(success=False, message=f"Edit failed: {e}")

    async def delete(
        self, path: str, permanent: bool = False, *, user_id: str | None = None
    ) -> DeleteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        try:
            mount, rel_path = self._registry.resolve(path)

            if not permanent and not self._get_capability(mount.backend, SupportsTrash):
                return DeleteResult(
                    success=False,
                    message="Trash not supported on this mount. "
                    "Use permanent=True to delete permanently.",
                )

            async with self._session_for(mount) as sess:
                result = await mount.backend.delete(
                    rel_path, permanent, session=sess, user_id=user_id
                )
            result.file_path = self._prefix_path(result.file_path, mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_DELETED, path=path, user_id=user_id)
                )
            return result
        except Exception as e:
            return DeleteResult(success=False, message=f"Delete failed: {e}")

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        user_id: str | None = None,
    ) -> MkdirResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.mkdir(rel_path, parents, session=sess, user_id=user_id)
        result.path = self._prefix_path(result.path, mount.mount_path)
        result.created_dirs = [
            self._prefix_path(d, mount.mount_path) or d for d in result.created_dirs
        ]
        return result

    async def list_dir(self, path: str = "/", *, user_id: str | None = None) -> ListDirResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.list_dir(rel_path, session=sess, user_id=user_id)
        return result.rebase(mount.mount_path)

    def _list_root(self) -> ListDirResult:
        from grover.search.results import ListDirEvidence

        entries: dict[str, list] = {}
        for mount in self._registry.list_visible_mounts():
            entries[mount.mount_path] = [
                ListDirEvidence(
                    strategy="list_dir",
                    path=mount.mount_path,
                    is_directory=True,
                )
            ]
        return ListDirResult(
            success=True,
            message=f"Found {len(entries)} mount(s)",
            _entries=entries,
        )

    async def exists(self, path: str, *, user_id: str | None = None) -> bool:
        path = normalize_path(path)

        if path == "/":
            return True

        if self._registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return False

        async with self._session_for(mount) as sess:
            return await mount.backend.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileInfo | None:
        path = normalize_path(path)

        if self._registry.has_mount(path):
            for mount in self._registry.list_mounts():
                if mount.mount_path == path:
                    name = mount.mount_path.lstrip("/")
                    return FileInfo(
                        path=mount.mount_path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        async with self._session_for(mount) as sess:
            info = await mount.backend.get_info(rel_path, session=sess, user_id=user_id)
        if info is not None:
            info = self._prefix_file_info(info, mount)
        return info

    def get_permission_info(self, path: str) -> tuple[str, bool]:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        permission = self._registry.get_permission(path)
        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths
        return permission.value, is_override

    async def move(
        self, src: str, dest: str, *, user_id: str | None = None, follow: bool = False
    ) -> MoveResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(src)
            self._check_writable(dest)
        except PermissionError as e:
            return MoveResult(success=False, message=str(e))

        try:
            src_mount, src_rel = self._registry.resolve(src)
            dest_mount, dest_rel = self._registry.resolve(dest)

            if src_mount is dest_mount:
                async with self._session_for(src_mount) as sess:
                    result = await src_mount.backend.move(
                        src_rel, dest_rel, session=sess, follow=follow, user_id=user_id
                    )
                result.old_path = self._prefix_path(result.old_path, src_mount.mount_path)
                result.new_path = self._prefix_path(result.new_path, dest_mount.mount_path)
                if result.success:
                    await self._emit(
                        FileEvent(
                            event_type=EventType.FILE_MOVED,
                            path=dest,
                            old_path=src,
                            user_id=user_id,
                        )
                    )
                return result

            # Cross-mount move: read → write → delete (non-atomic)
            async with self._session_for(src_mount) as src_sess:
                read_result = await src_mount.backend.read(
                    src_rel, session=src_sess, user_id=user_id
                )
            if not read_result.success:
                return MoveResult(
                    success=False,
                    message=f"Cannot read source for cross-mount move: {read_result.message}",
                )
            if read_result.content is None:
                return MoveResult(success=False, message=f"Source file has no content: {src}")

            async with self._session_for(dest_mount) as dest_sess:
                write_result = await dest_mount.backend.write(
                    dest_rel, read_result.content, session=dest_sess, user_id=user_id
                )
            if not write_result.success:
                return MoveResult(
                    success=False,
                    message=(
                        f"Cannot write to destination for cross-mount move: {write_result.message}"
                    ),
                )

            async with self._session_for(src_mount) as src_sess:
                delete_result = await src_mount.backend.delete(
                    src_rel, permanent=False, session=src_sess, user_id=user_id
                )
            if not delete_result.success:
                return MoveResult(
                    success=False,
                    message=f"Copied but failed to delete source: {delete_result.message}",
                )

            await self._emit(
                FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src, user_id=user_id)
            )
            return MoveResult(
                success=True,
                message=f"Moved {src} -> {dest} (cross-mount)",
                old_path=src,
                new_path=dest,
            )
        except Exception as e:
            return MoveResult(success=False, message=f"Move failed: {e}")

    async def copy(self, src: str, dest: str, *, user_id: str | None = None) -> WriteResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        try:
            src_mount, src_rel = self._registry.resolve(src)
            dest_mount, dest_rel = self._registry.resolve(dest)

            if src_mount is dest_mount:
                async with self._session_for(src_mount) as sess:
                    result = await src_mount.backend.copy(
                        src_rel, dest_rel, session=sess, user_id=user_id
                    )
                result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
                if result.success:
                    await self._emit(
                        FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                    )
                return result

            # Cross-mount copy: read → write
            async with self._session_for(src_mount) as src_sess:
                read_result = await src_mount.backend.read(
                    src_rel, session=src_sess, user_id=user_id
                )
            if not read_result.success:
                return WriteResult(
                    success=False,
                    message=f"Cannot read source for cross-mount copy: {read_result.message}",
                )
            if read_result.content is None:
                return WriteResult(success=False, message=f"Source file has no content: {src}")

            async with self._session_for(dest_mount) as dest_sess:
                result = await dest_mount.backend.write(
                    dest_rel, read_result.content, session=dest_sess, user_id=user_id
                )
            result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                )
            return result
        except Exception as e:
            return WriteResult(success=False, message=f"Copy failed: {e}")

    # ------------------------------------------------------------------
    # Search / Query operations (absorbed from VFS)
    # ------------------------------------------------------------------

    async def glob(
        self, pattern: str, path: str = "/", *, user_id: str | None = None
    ) -> GlobResult:
        path = normalize_path(path)
        try:
            if path == "/":
                combined = GlobResult(success=True, message="", _entries={}, pattern=pattern)
                for mount in self._registry.list_visible_mounts():
                    async with self._session_for(mount) as sess:
                        result = await mount.backend.glob(
                            pattern, "/", session=sess, user_id=user_id
                        )
                    if result.success:
                        combined = combined | result.rebase(mount.mount_path)
                combined.message = f"Found {len(combined)} match(es)"
                combined.pattern = pattern
                return combined

            mount, rel_path = self._registry.resolve(path)
            async with self._session_for(mount) as sess:
                result = await mount.backend.glob(pattern, rel_path, session=sess, user_id=user_id)
            return result.rebase(mount.mount_path)
        except Exception as e:
            return GlobResult(success=False, message=f"Glob failed: {e}", pattern=pattern)

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
        path = normalize_path(path)
        try:
            if path == "/":
                combined_entries: dict[str, list] = {}
                total_matches = 0
                total_searched = 0
                total_matched = 0
                truncated = False

                for mount in self._registry.list_visible_mounts():
                    remaining = max_results - total_matches if max_results > 0 else max_results
                    if max_results > 0 and remaining <= 0:
                        truncated = True
                        break
                    async with self._session_for(mount) as sess:
                        result = await mount.backend.grep(
                            pattern,
                            "/",
                            session=sess,
                            glob_filter=glob_filter,
                            case_sensitive=case_sensitive,
                            fixed_string=fixed_string,
                            invert=invert,
                            word_match=word_match,
                            context_lines=context_lines,
                            max_results=remaining,
                            max_results_per_file=max_results_per_file,
                            count_only=False,
                            files_only=files_only,
                            user_id=user_id,
                        )
                    if result.success:
                        rebased = result.rebase(mount.mount_path)
                        for p, evs in rebased._entries.items():
                            combined_entries.setdefault(p, []).extend(evs)
                            total_matches += sum(
                                len(e.line_matches) for e in evs if hasattr(e, "line_matches")
                            )
                        total_searched += result.files_searched
                        total_matched += result.files_matched
                        if result.truncated:
                            truncated = True

                if count_only:
                    total = total_matched if files_only else total_matches
                    return GrepResult(
                        success=True,
                        message=f"Count: {total}",
                        pattern=pattern,
                        files_searched=total_searched,
                        files_matched=total_matched,
                        truncated=truncated,
                    )

                return GrepResult(
                    success=True,
                    message=f"Found {total_matches} match(es) in {total_matched} file(s)",
                    _entries=combined_entries,
                    pattern=pattern,
                    files_searched=total_searched,
                    files_matched=total_matched,
                    truncated=truncated,
                )

            mount, rel_path = self._registry.resolve(path)
            async with self._session_for(mount) as sess:
                result = await mount.backend.grep(
                    pattern,
                    rel_path,
                    session=sess,
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
            return result.rebase(mount.mount_path)
        except Exception as e:
            return GrepResult(success=False, message=f"Grep failed: {e}", pattern=pattern)

    async def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        path = normalize_path(path)
        try:
            if path == "/":
                from grover.search.results import TreeEvidence

                root_entries: dict[str, list] = {}
                for mount in self._registry.list_visible_mounts():
                    root_entries[mount.mount_path] = [
                        TreeEvidence(
                            strategy="tree",
                            path=mount.mount_path,
                            depth=0,
                            is_directory=True,
                        )
                    ]
                combined = TreeResult(success=True, message="", _entries=root_entries)

                if max_depth is None or max_depth > 0:
                    for mount in self._registry.list_visible_mounts():
                        async with self._session_for(mount) as sess:
                            result = await mount.backend.tree(
                                "/", max_depth=max_depth, session=sess, user_id=user_id
                            )
                        if result.success:
                            combined = combined | result.rebase(mount.mount_path)

                combined.message = (
                    f"{combined.total_dirs} directories, {combined.total_files} files"
                )
                return combined

            mount, rel_path = self._registry.resolve(path)
            async with self._session_for(mount) as sess:
                result = await mount.backend.tree(
                    rel_path, max_depth=max_depth, session=sess, user_id=user_id
                )
            return result.rebase(mount.mount_path)
        except Exception as e:
            return TreeResult(success=False, message=f"Tree failed: {e}")

    # ------------------------------------------------------------------
    # Version operations (absorbed from VFS, capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> ListVersionsResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.backend, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.mount_path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                return await cap.list_versions(rel_path, session=sess, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return ListVersionsResult(success=False, versions=[], message=str(e))

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.backend, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.mount_path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                return await cap.get_version_content(
                    rel_path, version, session=sess, user_id=user_id
                )
        except CapabilityNotSupportedError as e:
            return GetVersionContentResult(success=False, content=None, message=str(e))

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.backend, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.mount_path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                result = await cap.restore_version(rel_path, version, session=sess, user_id=user_id)
            result.file_path = self._prefix_path(result.file_path, mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
                )
            return result
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    # ------------------------------------------------------------------
    # Trash operations (absorbed from VFS, capability-gated)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> ListResult:
        """List all items in trash across all mounts."""
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
                result = await cap.list_trash(session=sess, user_id=user_id)
            if result.success:
                all_entries.extend(self._prefix_file_info(entry, mount) for entry in result.entries)
        return ListResult(
            success=True,
            message=f"Found {len(all_entries)} item(s) in trash",
            entries=all_entries,
            path="/__trash__",
        )

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.mount_path} does not support trash"
                )
            async with self._session_for(mount) as sess:
                result = await cap.restore_from_trash(rel_path, session=sess, user_id=user_id)
            result.file_path = self._prefix_path(result.file_path, mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
                )
            return result
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
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

        cap = self._get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._session_for(mount) as sess:
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

        cap = self._get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._session_for(mount) as sess:
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

        cap = self._get_capability(mount.backend, SupportsReBAC)
        if cap is None:
            return ListSharesResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._session_for(mount) as sess:
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
            cap = self._get_capability(mount.backend, SupportsReBAC)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
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
        """Reconcile disk ↔ DB for capable mounts."""
        total = {"created": 0, "updated": 0, "deleted": 0}
        mounts = self._registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.mount_path == mount_path]

        for mount in mounts:
            cap = self._get_capability(mount.backend, SupportsReconcile)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            for k in total:
                total[k] += stats.get(k, 0)

        return total

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
    ) -> SearchQueryResult:
        """Semantic search, routed to per-mount search engines."""
        path = normalize_path(path)

        # Check if any mount has a search engine
        has_search = False
        for mount in self._registry.list_visible_mounts():
            if mount.search is not None:
                has_search = True
                break
        if not has_search:
            return SearchQueryResult(
                success=False,
                message=(
                    "Search is not available: no embedding provider configured. "
                    "Install sentence-transformers or pass embedding_provider= "
                    "to GroverAsync()."
                ),
                query=query,
                path=path,
            )
        try:
            # Inline VFS search routing
            if path == "/":
                all_search_results: list = []
                for mount in self._registry.list_visible_mounts():
                    if mount.search is None:
                        continue
                    results = await mount.search.search(query, k)
                    all_search_results.extend(results)
                all_search_results.sort(key=lambda r: r.score, reverse=True)
                raw_results = all_search_results[:k]
            else:
                mount, rel_path = self._registry.resolve(path)
                if mount.search is None:
                    raw_results = []
                else:
                    results = await mount.search.search(query, k)
                    if rel_path != "/":
                        prefix = rel_path.rstrip("/") + "/"
                        results = [
                            r
                            for r in results
                            if (r.parent_path or r.ref.path).startswith(prefix)
                            or (r.parent_path or r.ref.path) == rel_path.rstrip("/")
                        ]
                    raw_results = results
        except Exception as e:
            return SearchQueryResult(
                success=False,
                message=f"Search failed: {e}",
                query=query,
                path=path,
            )

        # Group results by parent file (document-first)
        file_groups: dict[str, list[Any]] = {}
        for r in raw_results:
            file_path = r.parent_path or r.ref.path
            file_groups.setdefault(file_path, []).append(r)

        search_hits: list[SearchHit] = []
        for file_path, results in file_groups.items():
            chunk_matches_list: list[ChunkMatch] = []
            max_score = 0.0
            for r in results:
                if r.score > max_score:
                    max_score = r.score
                # Build snippet: first 200 chars + "..." if truncated
                snippet = r.content[:200]
                if len(r.content) > 200:
                    snippet += "..."
                # Extract chunk metadata from ref or fallback
                chunk_name = getattr(r.ref, "chunk_name", None) or r.ref.path.rsplit("/", 1)[-1]
                line_start = getattr(r.ref, "line_start", None) or 0
                line_end = getattr(r.ref, "line_end", None) or 0
                chunk_matches_list.append(
                    ChunkMatch(
                        name=chunk_name,
                        line_start=line_start,
                        line_end=line_end,
                        score=r.score,
                        snippet=snippet,
                    )
                )
            search_hits.append(
                SearchHit(
                    path=file_path,
                    score=max_score,
                    chunk_matches=tuple(chunk_matches_list),
                )
            )

        # Sort by score desc, truncate to k
        search_hits.sort(key=lambda h: h.score, reverse=True)
        search_hits = search_hits[:k]

        return SearchQueryResult(
            success=True,
            message=f"{len(search_hits)} file(s) matched",
            hits=tuple(search_hits),
            query=query,
            path=path,
            files_matched=len(search_hits),
            truncated=len(raw_results) >= k,
        )

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
        result = await self.list_dir(path)
        if not result.success:
            return

        from grover.search.results import ListDirEvidence

        for entry_path in result.paths:
            if "/.grover/" in entry_path:
                continue
            evs = result._entries.get(entry_path, [])
            is_dir = any(isinstance(e, ListDirEvidence) and e.is_directory for e in evs)
            if is_dir:
                await self._walk_and_index(entry_path, stats)
            else:
                content = await self._read_file_content(entry_path)
                if content is None:
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
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        backend = mount.backend
        if hasattr(backend, "_read_content"):
            if mount.has_session_factory:
                async with self._session_for(mount) as sess:
                    content: str | None = await backend._read_content(rel_path, sess)
            else:
                content = await backend._read_content(rel_path, None)
            return content

        return None

    async def save(self) -> None:
        await self._async_save()

    async def _async_save(self) -> None:
        """Save per-mount graph and search state."""
        # Save each mount's graph to its own DB
        for mount in self._registry.list_visible_mounts():
            graph = mount.graph
            if graph is not None and mount.has_session_factory:
                from grover.graph.protocols import SupportsPersistence

                if isinstance(graph, SupportsPersistence):
                    try:
                        async with self._session_for(mount) as session:
                            if session is not None:
                                await graph.to_sql(session)
                    except Exception:
                        logger.debug(
                            "Failed to save graph for %s",
                            mount.mount_path,
                            exc_info=True,
                        )

            # Save search index to disk
            search_engine = mount.search
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
        # Close all backends directly
        for mount in self._registry.list_mounts():
            if hasattr(mount.backend, "close"):
                try:
                    await mount.backend.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.mount_path, exc_info=True)

"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grover.events import EventBus, EventType, FileEvent
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import (
    CapabilityNotSupportedError,
    MountNotFoundError,
    SchemaIncompatibleError,
)
from grover.fs.local_fs import LocalFileSystem
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
from grover.graph._rustworkx import RustworkxGraph
from grover.graph.analyzers import AnalyzerRegistry
from grover.models.chunks import FileChunk
from grover.models.connections import FileConnection
from grover.models.files import File, FileVersion
from grover.models.shares import FileShare
from grover.mount import Mount
from grover.search._engine import SearchEngine
from grover.search.extractors import extract_from_chunks, extract_from_file
from grover.types import (
    ConnectionResult,
    DeleteResult,
    EditResult,
    FileInfoResult,
    FileSearchCandidate,
    FileSearchResult,
    GetVersionContentResult,
    GlobResult,
    GraphEvidence,
    GraphResult,
    GrepResult,
    LexicalEvidence,
    LexicalSearchResult,
    ListDirEvidence,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareResult,
    ShareSearchResult,
    TrashResult,
    TreeEvidence,
    TreeResult,
    VectorEvidence,
    VectorSearchResult,
    VersionResult,
    WriteResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from grover.fs.protocol import StorageBackend
    from grover.graph.protocols import GraphStore
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase
    from grover.search.fulltext.protocol import FullTextStore
    from grover.search.protocols import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_DATA_DIR = Path.home() / ".grover" / "_default"


class GroverAsync:
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
        self._embedding_provider = embedding_provider
        self._explicit_vector_store = vector_store

        # Register event handlers
        self._event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._event_bus.register(EventType.FILE_RESTORED, self._on_file_restored)
        self._event_bus.register(EventType.CONNECTION_ADDED, self._on_connection_added)
        self._event_bus.register(EventType.CONNECTION_DELETED, self._on_connection_deleted)

    # ------------------------------------------------------------------
    # Session management & helpers (absorbed from VFS)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _session_for(self, mount: Mount) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or ``None`` for non-SQL backends."""
        if not mount.session_factory is not None:
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

    def _check_writable(self, virtual_path: str) -> str | None:
        """Return an error message if *virtual_path* is read-only, else ``None``.

        Replaces the previous raise-based pattern to avoid unnecessary
        exception overhead in the common (writable) case.
        """
        try:
            perm = self._registry.get_permission(virtual_path)
        except MountNotFoundError as e:
            return str(e)
        if perm == Permission.READ_ONLY:
            return f"Cannot write to read-only path: {virtual_path}"
        return None

    @staticmethod
    def _get_capability(backend: object, protocol: type[T]) -> T | None:
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

    def _prefix_file_info(self, info: FileInfoResult, mount: Mount) -> FileInfoResult:
        prefixed_path = self._prefix_path(info.path, mount.path) or info.path
        info.path = prefixed_path
        info.mount_type = mount.mount_type
        info.permission = self._registry.get_permission(prefixed_path).value
        return info

    # ------------------------------------------------------------------
    # Search / Graph factory helpers
    # ------------------------------------------------------------------

    def _create_search_engine(self, *, lexical: FullTextStore | None = None) -> SearchEngine | None:
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

    async def _create_fulltext_store(self, config: Mount) -> FullTextStore | None:
        """Create a FullTextStore for the mount based on its dialect."""
        from grover.search.fulltext.sqlite import SQLiteFullTextStore

        if isinstance(config.filesystem, LocalFileSystem):
            engine = getattr(config.filesystem, "_engine", None)
            if engine is not None:
                fts = SQLiteFullTextStore(engine=engine)
                await fts.ensure_table()
                return fts
            return None

        if config.session_factory is not None:
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
            msg = f"No graph on mount at {mount.path}"
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
        mount_or_path: str | Mount | None = None,
        filesystem: StorageBackend | None = None,
        *,
        graph: GraphStore | None = None,
        search: SearchEngine | None = None,
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
        path: str | None = None,
    ) -> None:
        """Register a :class:`~grover.mount.Mount` or build one from kwargs.

        Usage::

            # From a Mount object
            mount = Mount(path="/project", filesystem=LocalFileSystem(...))
            await g.add_mount(mount)

            # From keyword arguments (filesystem-based)
            await g.add_mount("/data", LocalFileSystem(workspace_dir="."))

            # Engine-based (auto-creates session factory + DatabaseFileSystem)
            await g.add_mount("/data", engine=engine)

            # Session-factory-based
            await g.add_mount("/data", filesystem, session_factory=sf)
        """
        if isinstance(mount_or_path, Mount):
            new_mount = mount_or_path
            # Auto-detect LocalFileSystem: open() and extract session_factory if not set
            if (
                isinstance(new_mount.filesystem, LocalFileSystem)
                and new_mount.session_factory is None
            ):
                await new_mount.filesystem.open()
                new_mount.session_factory = new_mount.filesystem.session_factory
        elif engine is not None:
            if session_factory is not None:
                raise ValueError("Provide engine or session_factory, not both")
            new_mount = await self._create_engine_mount(
                mount_or_path or path or "",
                engine,
                filesystem,
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
            new_mount = self._create_session_factory_mount(
                mount_or_path or path or "",
                session_factory,
                filesystem,
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
            # Resolve path: either from positional arg or keyword
            actual_path = mount_or_path if mount_or_path is not None else path
            if actual_path is None or filesystem is None:
                raise ValueError(
                    "Provide a Mount object, (path + filesystem), or engine/session_factory"
                )

            # For local backends, eagerly init DB and extract session_factory
            sf = session_factory
            mt = mount_type
            if isinstance(filesystem, LocalFileSystem):
                await filesystem.open()
                sf = filesystem.session_factory
                if mt is None:
                    mt = "local"

            new_mount = Mount(
                path=actual_path,
                filesystem=filesystem,
                graph=graph,
                search=search,
                session_factory=sf,
                permission=permission,
                label=label,
                mount_type=mt or "vfs",
                hidden=hidden,
            )

        # Auto-create graph if not provided and not hidden
        if not new_mount.hidden and new_mount.graph is None:
            new_mount.graph = RustworkxGraph()  # type: ignore[assignment]  # satisfies GraphStore protocol

        # Auto-create search engine if not provided and not hidden
        if not new_mount.hidden and new_mount.search is None:
            lexical = await self._create_fulltext_store(new_mount)
            se = self._create_search_engine(lexical=lexical)
            if se is not None:
                new_mount.search = se

        # Call open() on the filesystem if needed (skip LocalFileSystem — already opened above)
        if not isinstance(new_mount.filesystem, LocalFileSystem) and hasattr(
            new_mount.filesystem, "open"
        ):
            await new_mount.filesystem.open()

        self._registry.add_mount(new_mount)

        # Lazily initialise meta_fs on first non-hidden mount
        if not new_mount.hidden and self._meta_fs is None:
            assert new_mount.filesystem is not None
            await self._init_meta_fs(new_mount.filesystem)

        # Load existing graph + search state for this mount
        if not new_mount.hidden:
            await self._load_mount_state(new_mount)

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
    ) -> Mount:
        """Build a Mount from an async engine."""
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

        # Fail-fast schema validation — detect stale schemas from older Grover versions.
        # Only fires when tables already existed (checkfirst=True didn't create them)
        # and are missing columns required by the current code.
        from grover.migrations.backfill_alpha_refactor import check_schema_compatibility

        schema_errors = await check_schema_compatibility(
            engine,
            file_chunks_table=getattr(fcm, "__tablename__", "grover_file_chunks"),
            file_connections_table=getattr(
                FileConnection, "__tablename__", "grover_file_connections"
            ),
            file_versions_table=getattr(fvm, "__tablename__", "grover_file_versions"),
        )
        if schema_errors:
            msg = (
                "Database schema is incompatible with this version of Grover. "
                "Run the migration script to update:\n\n"
                "    from grover.migrations import backfill_alpha_refactor\n"
                "    await backfill_alpha_refactor(engine)\n\n"
                "Issues found:\n" + "\n".join(f"  - {e}" for e in schema_errors)
            )
            raise SchemaIncompatibleError(msg)

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

        return Mount(
            path=path,
            filesystem=backend,
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
    ) -> Mount:
        """Build a Mount from a caller-provided session factory."""
        if backend is None:
            backend = DatabaseFileSystem(
                dialect=dialect,
                file_model=file_model,
                file_version_model=file_version_model,
                file_chunk_model=file_chunk_model,
                schema=db_schema,
            )

        return Mount(
            path=path,
            filesystem=backend,
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
    ) -> Mount:
        """Build a Mount from a pre-constructed backend."""
        if mount_type is None:
            mount_type = "local" if isinstance(backend, LocalFileSystem) else "vfs"

        # For local backends, eagerly init DB and expose session_factory
        sf: Callable[..., AsyncSession] | None = None
        if isinstance(backend, LocalFileSystem):
            await backend.open()
            sf = backend.session_factory

        return Mount(
            path=path,
            filesystem=backend,
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
        if mount.path != path:
            return

        backend = mount.filesystem
        if hasattr(backend, "close"):
            await backend.close()
        self._registry.remove_mount(path)

    # ------------------------------------------------------------------
    # Internal metadata mount
    # ------------------------------------------------------------------

    async def _init_meta_fs(self, first_backend: StorageBackend) -> None:
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
            Mount(
                path="/.grover",
                filesystem=self._meta_fs,
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

    # ------------------------------------------------------------------
    # Per-mount state loading
    # ------------------------------------------------------------------

    async def _load_mount_state(self, mount: Mount) -> None:
        """Load graph and search state for a single mount."""
        graph = mount.graph
        if graph is None:
            return

        # Load graph: file nodes from mount's DB, edges from mount's DB
        if mount.session_factory is not None:
            try:
                from grover.graph.protocols import SupportsPersistence

                if isinstance(graph, SupportsPersistence):
                    file_model = getattr(mount.filesystem, "file_model", None) or File
                    async with self._session_for(mount) as session:
                        if session is not None:
                            await graph.from_sql(
                                session,
                                file_model=file_model,
                                path_prefix=mount.path,
                            )
            except Exception:
                logger.debug(
                    "No existing graph state to load for %s",
                    mount.path,
                    exc_info=True,
                )

        # Load search index from disk
        search_engine = mount.search
        if search_engine is not None and self._meta_data_dir is not None:
            slug = mount.path.strip("/").replace("/", "_") or "_default"
            search_dir = self._meta_data_dir / "search" / slug
            meta_file = search_dir / "search_meta.json"
            if meta_file.exists():
                try:
                    search_engine.load(str(search_dir))
                except Exception:
                    logger.debug(
                        "Failed to load search index for %s",
                        mount.path,
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
        # Clean up chunk DB rows and connection DB rows
        await self._delete_chunks_for_path(event.path)
        await self._delete_connections_for_path(event.path)

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
            graph = self._resolve_graph(event.source_path)
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
            graph = self._resolve_graph(event.source_path)
        except RuntimeError:
            return
        if graph.has_edge(event.source_path, event.target_path):
            graph.remove_edge(event.source_path, event.target_path)

    async def _delete_chunks_for_path(self, path: str) -> None:
        """Delete chunk DB rows for *path* if the backend supports it."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return
        if isinstance(mount.filesystem, SupportsFileChunks):
            async with self._session_for(mount) as sess:
                await mount.filesystem.delete_file_chunks(path, session=sess)

    async def _delete_connections_for_path(self, path: str) -> None:
        """Delete connection DB rows for *path* if the backend supports it."""
        try:
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return
        conn_svc = getattr(mount.filesystem, "connections", None)
        if conn_svc is not None:
            async with self._session_for(mount) as sess:
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
                async with self._session_for(mount) as sess:
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
                    async with self._session_for(mount) as sess:
                        await conn_svc.delete_outgoing_connections(sess, path)
                for edge in dep_edges:
                    _w: float = (
                        float(edge.metadata.get("weight", 1.0))  # type: ignore[arg-type]
                        if edge.metadata
                        else 1.0
                    )
                    async with self._session_for(mount) as sess:
                        await mount.filesystem.add_connection(
                            edge.source,
                            edge.target,
                            edge.edge_type,
                            weight=_w,
                            metadata=dict(edge.metadata) if edge.metadata else None,
                            session=sess,
                        )
                    # Emit event AFTER session commits (post-commit ordering)
                    await self._emit(
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
        assert mount.filesystem is not None
        async with self._session_for(mount) as sess:
            result = await mount.filesystem.read(
                rel_path, offset, limit, session=sess, user_id=user_id
            )
        result.path = self._prefix_path(result.path, mount.path) or result.path
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
        if err := self._check_writable(path):
            return WriteResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None
            async with self._session_for(mount) as sess:
                result = await mount.filesystem.write(
                    rel_path,
                    content,
                    "agent",
                    overwrite=overwrite,
                    session=sess,
                    user_id=user_id,
                )
            result.path = self._prefix_path(result.path, mount.path) or result.path
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
        if err := self._check_writable(path):
            return EditResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None
            async with self._session_for(mount) as sess:
                result = await mount.filesystem.edit(
                    rel_path,
                    old,
                    new,
                    replace_all,
                    "agent",
                    session=sess,
                    user_id=user_id,
                )
            result.path = self._prefix_path(result.path, mount.path) or result.path
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
        if err := self._check_writable(path):
            return DeleteResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None

            if not permanent and not self._get_capability(mount.filesystem, SupportsTrash):
                return DeleteResult(
                    success=False,
                    message="Trash not supported on this mount. "
                    "Use permanent=True to delete permanently.",
                )

            async with self._session_for(mount) as sess:
                result = await mount.filesystem.delete(
                    rel_path, permanent, session=sess, user_id=user_id
                )
            result.path = self._prefix_path(result.path, mount.path) or result.path
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
        if err := self._check_writable(path):
            return MkdirResult(success=False, message=err)

        mount, rel_path = self._registry.resolve(path)
        assert mount.filesystem is not None
        async with self._session_for(mount) as sess:
            result = await mount.filesystem.mkdir(rel_path, parents, session=sess, user_id=user_id)
        result.path = self._prefix_path(result.path, mount.path) or result.path
        result.created_dirs = [self._prefix_path(d, mount.path) or d for d in result.created_dirs]
        return result

    async def list_dir(self, path: str = "/", *, user_id: str | None = None) -> ListDirResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        assert mount.filesystem is not None
        async with self._session_for(mount) as sess:
            result = await mount.filesystem.list_dir(rel_path, session=sess, user_id=user_id)
        return result.rebase(mount.path)

    def _list_root(self) -> ListDirResult:
        candidates = [
            FileSearchCandidate(
                path=mount.path,
                evidence=[
                    ListDirEvidence(
                        strategy="list_dir",
                        path=mount.path,
                        is_directory=True,
                    )
                ],
            )
            for mount in self._registry.list_visible_mounts()
        ]
        return ListDirResult(
            success=True,
            message=f"Found {len(candidates)} mount(s)",
            candidates=candidates,
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

        assert mount.filesystem is not None
        async with self._session_for(mount) as sess:
            return await mount.filesystem.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileInfoResult | None:
        path = normalize_path(path)

        if self._registry.has_mount(path):
            for mount in self._registry.list_mounts():
                if mount.path == path:
                    name = mount.path.lstrip("/")
                    return FileInfoResult(
                        path=mount.path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        assert mount.filesystem is not None
        async with self._session_for(mount) as sess:
            info = await mount.filesystem.get_info(rel_path, session=sess, user_id=user_id)
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

        if err := self._check_writable(src):
            return MoveResult(success=False, message=err)
        if err := self._check_writable(dest):
            return MoveResult(success=False, message=err)

        try:
            src_mount, src_rel = self._registry.resolve(src)
            dest_mount, dest_rel = self._registry.resolve(dest)

            assert src_mount.filesystem is not None
            assert dest_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._session_for(src_mount) as sess:
                    result = await src_mount.filesystem.move(
                        src_rel, dest_rel, session=sess, follow=follow, user_id=user_id
                    )
                result.old_path = (
                    self._prefix_path(result.old_path, src_mount.path) or result.old_path
                )
                result.new_path = (
                    self._prefix_path(result.new_path, dest_mount.path) or result.new_path
                )
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
                read_result = await src_mount.filesystem.read(
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
                write_result = await dest_mount.filesystem.write(
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
                delete_result = await src_mount.filesystem.delete(
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

        if err := self._check_writable(dest):
            return WriteResult(success=False, message=err)

        try:
            src_mount, src_rel = self._registry.resolve(src)
            dest_mount, dest_rel = self._registry.resolve(dest)

            assert src_mount.filesystem is not None
            assert dest_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._session_for(src_mount) as sess:
                    result = await src_mount.filesystem.copy(
                        src_rel, dest_rel, session=sess, user_id=user_id
                    )
                result.path = self._prefix_path(result.path, dest_mount.path) or result.path
                if result.success:
                    await self._emit(
                        FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                    )
                return result

            # Cross-mount copy: read → write
            async with self._session_for(src_mount) as src_sess:
                read_result = await src_mount.filesystem.read(
                    src_rel, session=src_sess, user_id=user_id
                )
            if not read_result.success:
                return WriteResult(
                    success=False,
                    message=f"Cannot read source for cross-mount copy: {read_result.message}",
                )
            if not read_result.content:
                return WriteResult(success=False, message=f"Source file has no content: {src}")

            async with self._session_for(dest_mount) as dest_sess:
                result = await dest_mount.filesystem.write(
                    dest_rel, read_result.content, session=dest_sess, user_id=user_id
                )
            result.path = self._prefix_path(result.path, dest_mount.path) or result.path
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
                combined = GlobResult(success=True, message="", pattern=pattern)
                for mount in self._registry.list_visible_mounts():
                    assert mount.filesystem is not None
                    async with self._session_for(mount) as sess:
                        result = await mount.filesystem.glob(
                            pattern, "/", session=sess, user_id=user_id
                        )
                    if result.success:
                        combined = combined | result.rebase(mount.path)
                combined.message = f"Found {len(combined)} match(es)"
                combined.pattern = pattern
                return combined

            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None
            async with self._session_for(mount) as sess:
                result = await mount.filesystem.glob(
                    pattern, rel_path, session=sess, user_id=user_id
                )
            return result.rebase(mount.path)
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
                    assert mount.filesystem is not None
                    async with self._session_for(mount) as sess:
                        result = await mount.filesystem.grep(
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
                        rebased = result.rebase(mount.path)
                        for c in rebased.candidates:
                            combined_entries.setdefault(c.path, []).extend(c.evidence)
                            total_matches += sum(
                                len(e.line_matches)  # type: ignore[union-attr]
                                for e in c.evidence
                                if hasattr(e, "line_matches")
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
                    candidates=FileSearchResult._dict_to_candidates(combined_entries),
                    pattern=pattern,
                    files_searched=total_searched,
                    files_matched=total_matched,
                    truncated=truncated,
                )

            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None
            async with self._session_for(mount) as sess:
                result = await mount.filesystem.grep(
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
            return result.rebase(mount.path)
        except Exception as e:
            return GrepResult(success=False, message=f"Grep failed: {e}", pattern=pattern)

    async def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        path = normalize_path(path)
        try:
            if path == "/":
                root_candidates = [
                    FileSearchCandidate(
                        path=mount.path,
                        evidence=[
                            TreeEvidence(
                                strategy="tree",
                                path=mount.path,
                                depth=0,
                                is_directory=True,
                            )
                        ],
                    )
                    for mount in self._registry.list_visible_mounts()
                ]
                combined = TreeResult(success=True, message="", candidates=root_candidates)

                if max_depth is None or max_depth > 0:
                    for mount in self._registry.list_visible_mounts():
                        assert mount.filesystem is not None
                        async with self._session_for(mount) as sess:
                            result = await mount.filesystem.tree(
                                "/", max_depth=max_depth, session=sess, user_id=user_id
                            )
                        if result.success:
                            combined = combined | result.rebase(mount.path)

                combined.message = (
                    f"{combined.total_dirs} directories, {combined.total_files} files"
                )
                return combined

            mount, rel_path = self._registry.resolve(path)
            assert mount.filesystem is not None
            async with self._session_for(mount) as sess:
                result = await mount.filesystem.tree(
                    rel_path, max_depth=max_depth, session=sess, user_id=user_id
                )
            return result.rebase(mount.path)
        except Exception as e:
            return TreeResult(success=False, message=f"Tree failed: {e}")

    # ------------------------------------------------------------------
    # Version operations (absorbed from VFS, capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> VersionResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                return await cap.list_versions(rel_path, session=sess, user_id=user_id)
        except CapabilityNotSupportedError as e:
            return VersionResult(success=False, message=str(e))

    async def get_version_content(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                return await cap.get_version_content(
                    rel_path, version, session=sess, user_id=user_id
                )
        except CapabilityNotSupportedError as e:
            return GetVersionContentResult(success=False, message=str(e))

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        if err := self._check_writable(path):
            return RestoreResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.filesystem, SupportsVersions)
            if cap is None:
                raise CapabilityNotSupportedError(
                    f"Mount at {mount.path} does not support versioning"
                )
            async with self._session_for(mount) as sess:
                result = await cap.restore_version(rel_path, version, session=sess, user_id=user_id)
            result.path = self._prefix_path(result.path, mount.path) or result.path
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

    async def list_trash(self, *, user_id: str | None = None) -> TrashResult:
        """List all items in trash across all mounts."""
        combined = TrashResult(success=True, message="")
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.filesystem, SupportsTrash)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
                result = await cap.list_trash(session=sess, user_id=user_id)
            if result.success:
                rebased = result.rebase(mount.path)
                combined = combined | rebased
        combined.message = f"Found {len(combined)} item(s) in trash"
        return combined

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        path = normalize_path(path)
        if err := self._check_writable(path):
            return RestoreResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
            cap = self._get_capability(mount.filesystem, SupportsTrash)
            if cap is None:
                raise CapabilityNotSupportedError(f"Mount at {mount.path} does not support trash")
            async with self._session_for(mount) as sess:
                result = await cap.restore_from_trash(rel_path, session=sess, user_id=user_id)
            result.path = self._prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_RESTORED, path=path, user_id=user_id)
                )
            return result
        except CapabilityNotSupportedError as e:
            return RestoreResult(success=False, message=str(e))

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts.  Skips read-only mounts."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            # Skip read-only mounts — empty_trash is a mutation
            if mount.permission == Permission.READ_ONLY:
                continue
            cap = self._get_capability(mount.filesystem, SupportsTrash)
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
        expires_at: datetime | None = None,
    ) -> ShareResult:
        """Share a file or directory with another user.

        Requires a backend that supports sharing (e.g. ``UserScopedFileSystem``).
        """

        path = normalize_path(path)
        if err := self._check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._get_capability(mount.filesystem, SupportsReBAC)
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
        if err := self._check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._get_capability(mount.filesystem, SupportsReBAC)
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
    ) -> ShareSearchResult:
        """List all shares on a given path."""

        path = normalize_path(path)
        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError as e:
            return ShareSearchResult(success=False, message=str(e))

        cap = self._get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareSearchResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._session_for(mount) as sess:
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
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.filesystem, SupportsReBAC)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
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
        mounts = self._registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.path == mount_path]

        for mount in mounts:
            # Skip read-only mounts — reconcile is a mutation
            if mount.permission == Permission.READ_ONLY:
                continue
            cap = self._get_capability(mount.filesystem, SupportsReconcile)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
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

        if err := self._check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        try:
            mount, _rel = self._registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        backend = self._get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        async with self._session_for(mount) as sess:
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
            await self._emit(
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

        if err := self._check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        try:
            mount, _rel = self._registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        backend = self._get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        async with self._session_for(mount) as sess:
            result = await backend.delete_connection(
                source_path,
                target_path,
                connection_type=connection_type,
                session=sess,
            )

        # Emit AFTER session commits (post-commit event ordering)
        if result.success:
            await self._emit(
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
            mount, _rel = self._registry.resolve(path)
        except MountNotFoundError:
            return []

        backend = self._get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return []

        async with self._session_for(mount) as sess:
            return await backend.list_connections(
                path,
                direction=direction,
                connection_type=connection_type,
                session=sess,
            )

    # ------------------------------------------------------------------
    # Graph query wrappers (resolve mount → delegate to backend's graph)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> GraphResult:
        """Return files that depend on *path*."""
        refs = self._resolve_graph(path).dependents(path)
        return GraphResult.from_refs(refs, strategy="dependents")

    def dependencies(self, path: str) -> GraphResult:
        """Return files that *path* depends on."""
        refs = self._resolve_graph(path).dependencies(path)
        return GraphResult.from_refs(refs, strategy="dependencies")

    def impacts(self, path: str, max_depth: int = 3) -> GraphResult:
        """Return files transitively impacted by changes to *path*."""
        refs = self._resolve_graph(path).impacts(path, max_depth)
        return GraphResult.from_refs(refs, strategy="impacts")

    def path_between(self, source: str, target: str) -> GraphResult:
        """Return the shortest path from *source* to *target*."""
        refs = self._resolve_graph(source).path_between(source, target)
        if refs is None:
            return GraphResult(
                success=True,
                message="No path found",
            )
        return GraphResult.from_refs(refs, strategy="path_between")

    def contains(self, path: str) -> GraphResult:
        """Return files contained by *path*."""
        refs = self._resolve_graph(path).contains(path)
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
        Raises :class:`~grover.fs.exceptions.CapabilityNotSupportedError` if
        the graph backend does not support centrality algorithms.
        """
        from grover.graph.protocols import SupportsCentrality

        graph = self._resolve_graph_any(path)
        if not isinstance(graph, SupportsCentrality):
            msg = "Graph backend does not support centrality algorithms"
            raise CapabilityNotSupportedError(msg)
        scores = graph.pagerank(personalization=personalization)
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
        from grover.graph.protocols import SupportsTraversal

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsTraversal):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg)
        node_set = graph.ancestors(path)
        return GraphResult.from_paths(sorted(node_set), strategy="ancestors")

    def descendants(self, path: str) -> GraphResult:
        """All transitive successors of *path* in the knowledge graph."""
        from grover.graph.protocols import SupportsTraversal

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsTraversal):
            msg = "Graph backend does not support traversal algorithms"
            raise CapabilityNotSupportedError(msg)
        node_set = graph.descendants(path)
        return GraphResult.from_paths(sorted(node_set), strategy="descendants")

    def meeting_subgraph(
        self,
        paths: list[str],
        *,
        max_size: int = 50,
    ) -> GraphResult:
        """Extract the subgraph connecting *paths* via shortest paths."""
        from grover.graph.protocols import SupportsSubgraph

        graph = self._resolve_graph_any(paths[0] if paths else None)
        if not isinstance(graph, SupportsSubgraph):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg)
        sub = graph.meeting_subgraph(paths, max_size=max_size)
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
        from grover.graph.protocols import SupportsSubgraph

        graph = self._resolve_graph(path)
        if not isinstance(graph, SupportsSubgraph):
            msg = "Graph backend does not support subgraph extraction"
            raise CapabilityNotSupportedError(msg)
        sub = graph.neighborhood(
            path,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
        )
        return GraphResult.from_paths(sorted(sub.nodes), strategy="neighborhood")

    def find_nodes(self, *, path: str | None = None, **attrs: object) -> GraphResult:
        """Find graph nodes matching all attribute predicates."""
        from grover.graph.protocols import SupportsFiltering

        graph = self._resolve_graph_any(path)
        if not isinstance(graph, SupportsFiltering):
            msg = "Graph backend does not support filtering"
            raise CapabilityNotSupportedError(msg)
        node_list = graph.find_nodes(**attrs)
        return GraphResult.from_paths(node_list, strategy="find_nodes")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        user_id: str | None = None,
    ) -> VectorSearchResult:
        """Semantic (vector) search, routed to per-mount search engines."""
        path = normalize_path(path)

        # Check if any mount has a search engine with vector capability
        has_search = any(mount.search is not None for mount in self._registry.list_visible_mounts())
        if not has_search:
            return VectorSearchResult(
                success=False,
                message=(
                    "Vector search is not available: no embedding provider "
                    "configured. Install sentence-transformers or pass "
                    "embedding_provider= to GroverAsync()."
                ),
            )
        # Collect (result, mount_path) pairs — SearchResult is frozen so we
        # cannot attach attributes to it.  Use a parallel list instead.
        try:
            if path == "/":
                tagged: list[tuple[Any, str]] = []
                for mount in self._registry.list_visible_mounts():
                    if mount.search is None:
                        continue
                    results = await mount.search.search(query, k)
                    tagged.extend((r, mount.path) for r in results)
                tagged.sort(key=lambda t: t[0].score, reverse=True)
                tagged = tagged[:k]
            else:
                mount, rel_path = self._registry.resolve(path)
                if mount.search is None:
                    tagged = []
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
                    tagged = [(r, mount.path) for r in results]
        except Exception as e:
            return VectorSearchResult(
                success=False,
                message=f"Vector search failed: {e}",
            )

        # Build VectorSearchResult with VectorEvidence
        entries: dict[str, list[Any]] = {}
        for r, mount_path in tagged:
            fp = r.parent_path or r.ref.path
            if mount_path and not fp.startswith(mount_path):
                fp = mount_path + fp
            snippet = r.content[:200]
            if len(r.content) > 200:
                snippet += "..."
            ev = VectorEvidence(
                strategy="vector_search",
                path=fp,
                snippet=snippet,
            )
            entries.setdefault(fp, []).append(ev)

        return VectorSearchResult(
            success=True,
            message=f"Found matches in {len(entries)} file(s)",
            candidates=FileSearchResult._dict_to_candidates(entries),
        )

    async def lexical_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        user_id: str | None = None,
    ) -> LexicalSearchResult:
        """BM25/full-text search, routed to per-mount search engines."""
        path = normalize_path(path)

        try:
            if path == "/":
                combined: LexicalSearchResult = LexicalSearchResult(success=True, message="")
                for mount in self._registry.list_visible_mounts():
                    if mount.search is None or mount.search.lexical is None:
                        continue
                    async with self._session_for(mount) as sess:
                        fts_results = await mount.search.lexical_search(query, k=k, session=sess)
                    mount_entries: dict[str, list[Any]] = {}
                    for ftr in fts_results:
                        fp = mount.path + ftr.path
                        ev = LexicalEvidence(
                            strategy="lexical_search",
                            path=fp,
                            snippet=ftr.snippet,
                        )
                        mount_entries.setdefault(fp, []).append(ev)
                    mount_result = LexicalSearchResult(
                        success=True,
                        message="",
                        candidates=FileSearchResult._dict_to_candidates(mount_entries),
                    )
                    combined = combined | mount_result
                combined.message = f"Found matches in {len(combined)} file(s)"
                return combined
            else:
                mount, _rel_path = self._registry.resolve(path)
                if mount.search is None or mount.search.lexical is None:
                    return LexicalSearchResult(
                        success=False,
                        message="Lexical search not available on this mount",
                    )
                async with self._session_for(mount) as sess:
                    fts_results = await mount.search.lexical_search(query, k=k, session=sess)
                entries: dict[str, list[Any]] = {}
                for ftr in fts_results:
                    fp = mount.path + ftr.path
                    ev = LexicalEvidence(
                        strategy="lexical_search",
                        path=fp,
                        snippet=ftr.snippet,
                    )
                    entries.setdefault(fp, []).append(ev)
                return LexicalSearchResult(
                    success=True,
                    message=f"Found matches in {len(entries)} file(s)",
                    candidates=FileSearchResult._dict_to_candidates(entries),
                )
        except Exception as e:
            return LexicalSearchResult(
                success=False,
                message=f"Lexical search failed: {e}",
            )

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        *,
        alpha: float = 0.5,
        path: str = "/",
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Hybrid search combining vector and lexical results.

        *alpha* controls the blend: 1.0 = pure vector, 0.0 = pure lexical.
        Falls back to whichever is available if only one is configured.
        """
        path = normalize_path(path)

        vec_result: FileSearchResult | None = None
        lex_result: FileSearchResult | None = None

        has_vector = any(
            mount.search is not None
            and mount.search.vector is not None
            and mount.search.embedding is not None
            for mount in self._registry.list_visible_mounts()
        )
        has_lexical = any(
            mount.search is not None and mount.search.lexical is not None
            for mount in self._registry.list_visible_mounts()
        )

        if has_vector:
            vec_result = await self.vector_search(query, k=k, path=path, user_id=user_id)
        if has_lexical:
            lex_result = await self.lexical_search(query, k=k, path=path, user_id=user_id)

        if vec_result is not None and lex_result is not None:
            return vec_result | lex_result
        if vec_result is not None:
            return vec_result
        if lex_result is not None:
            return lex_result

        return FileSearchResult(
            success=False,
            message="Hybrid search not available: no vector or lexical search configured",
        )

    async def search(
        self,
        query: str,
        *,
        path: str = "/",
        glob: str | None = None,
        grep: str | None = None,
        k: int = 10,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Composable search pipeline: optional glob/grep filters → vector search.

        If *glob* is provided, files are first filtered by glob pattern.
        If *grep* is provided, files are further filtered by content pattern.
        Then vector search is applied as the final stage.
        Results are chained using ``>>`` (intersection/pipeline).
        """
        result: FileSearchResult | None = None

        if glob is not None:
            glob_r = await self.glob(glob, path=path, user_id=user_id)
            result = glob_r

        if grep is not None:
            grep_r = await self.grep(grep, path=path, user_id=user_id)
            result = grep_r if result is None else (result >> grep_r)

        vec_r = await self.vector_search(query, k=k, path=path, user_id=user_id)
        result = vec_r if result is None else (result >> vec_r)

        return result

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
            for mount in self._registry.list_visible_mounts():
                await self._walk_and_index(mount.path, stats)

        await self._async_save()
        return stats

    async def _walk_and_index(self, path: str, stats: dict[str, int]) -> None:
        # Skip read-only mounts — indexing writes chunks, edges, and
        # search entries which are all mutations.
        try:
            mount, _rel = self._registry.resolve(path)
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
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        backend = mount.filesystem
        if hasattr(backend, "_read_content"):
            if mount.session_factory is not None:
                async with self._session_for(mount) as sess:
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
            mount, _rel = self._registry.resolve(normalize_path(path))
            await self._load_mount_state(mount)
        else:
            for mount in self._registry.list_mounts():
                await self._load_mount_state(mount)

    async def _async_save(self) -> None:
        """Save per-mount search state.

        Note: graph edges are no longer saved via ``to_sql()`` here.
        Edge persistence is now handled by the filesystem layer —
        ``add_connection()`` writes edges to DB, and the graph is a
        pure in-memory projection loaded via ``from_sql()`` on mount.
        """
        for mount in self._registry.list_visible_mounts():
            # Save search index to disk
            search_engine = mount.search
            if search_engine is not None and self._meta_data_dir is not None:
                slug = mount.path.strip("/").replace("/", "_") or "_default"
                search_dir = self._meta_data_dir / "search" / slug
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
        if self._closed:
            return
        self._closed = True

        await self._async_save()
        # Close all backends directly
        for mount in self._registry.list_mounts():
            if hasattr(mount.filesystem, "close"):
                try:
                    await mount.filesystem.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.path, exc_info=True)

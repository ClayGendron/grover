"""MountMixin — mount lifecycle methods for GroverAsync."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grover.backends.database import DatabaseFileSystem
from grover.backends.local import LocalFileSystem
from grover.backends.protocol import SupportsReBAC
from grover.exceptions import MountNotFoundError
from grover.models.database.share import FileShareModel
from grover.mount import Mount
from grover.permissions import Permission
from grover.providers.graph.rustworkx import RustworkxGraph
from grover.util.dialect import check_tables_exist, ensure_schema
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from sqlalchemy import Connection

    from grover.api.context import GroverContext
    from grover.backends.protocol import GroverFileSystem
    from grover.models.config import EngineConfig, SessionConfig
    from grover.providers.embedding.protocol import EmbeddingProvider
    from grover.providers.search.protocol import SearchProvider

logger = logging.getLogger(__name__)


class MountMixin:
    """Mount lifecycle methods extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Mount / Unmount
    # ------------------------------------------------------------------

    async def add_mount(
        self,
        path: str | None = None,
        *,
        mount: Mount | None = None,
        filesystem: GroverFileSystem | None = None,
        engine_config: EngineConfig | None = None,
        session_config: SessionConfig | None = None,
        mount_type: str | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        hidden: bool = False,
        embedding_provider: EmbeddingProvider | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        """Register a :class:`~grover.mount.Mount` or build one from kwargs.

        Usage::

            # From a pre-built Mount object
            mount = Mount(path="/project", filesystem=LocalFileSystem(...))
            await g.add_mount(mount=mount)

            # Filesystem-based (LocalFileSystem or custom backend)
            await g.add_mount("/data", filesystem=LocalFileSystem(workspace_dir="."))

            # Engine-based — Grover creates and owns the engine
            await g.add_mount(
                "/data", engine_config=EngineConfig(url="sqlite+aiosqlite:///db")
            )

            # Session-factory-based — app owns engine lifecycle
            await g.add_mount("/data", session_config=SessionConfig(session_factory=sf))
        """
        if mount is not None:
            new_mount = mount
            # Auto-detect LocalFileSystem: open() and extract session_factory if not set
            if isinstance(new_mount.filesystem, LocalFileSystem) and new_mount.session_factory is None:
                await new_mount.filesystem.open()
                new_mount.session_factory = new_mount.filesystem.session_factory
        elif engine_config is not None:
            if session_config is not None:
                raise ValueError("Provide engine_config or session_config, not both")
            new_mount = await self._create_engine_mount(
                path or "",
                engine_config,
                filesystem,
                mount_type,
                permission,
                label,
                hidden,
            )
        elif session_config is not None:
            new_mount = self._create_session_factory_mount(
                path or "",
                session_config,
                filesystem,
                mount_type,
                permission,
                label,
                hidden,
            )
        else:
            if path is None or filesystem is None:
                raise ValueError("Provide mount=Mount(...), (path + filesystem=), engine_config=, or session_config=")

            # For local backends, eagerly init DB and extract session_factory
            sf = None
            mt = mount_type
            if isinstance(filesystem, LocalFileSystem):
                await filesystem.open()
                sf = filesystem.session_factory
                if mt is None:
                    mt = "local"

            new_mount = Mount(
                path=path,
                filesystem=filesystem,
                session_factory=sf,
                permission=permission,
                label=label,
                mount_type=mt or "vfs",
                hidden=hidden,
            )

        # ------------------------------------------------------------------
        # Auto-inject providers on the filesystem
        # ------------------------------------------------------------------
        fs = new_mount.filesystem
        if fs is not None and not new_mount.hidden:
            # Graph provider: auto-create RustworkxGraph if not already set
            if getattr(fs, "graph_provider", None) is None:
                fs.graph_provider = RustworkxGraph()  # type: ignore[union-attr]

            # Search providers: inject from kwargs if not already set on filesystem
            if embedding_provider is not None and getattr(fs, "embedding_provider", None) is None:
                fs.embedding_provider = embedding_provider  # type: ignore[union-attr]
            if search_provider is not None and getattr(fs, "search_provider", None) is None:
                fs.search_provider = search_provider  # type: ignore[union-attr]
            # Validate dimensions if both providers are set
            if hasattr(fs, "_validate_search_dimensions"):
                fs._validate_search_dimensions()  # type: ignore[union-attr]

        # Call open() on the filesystem if needed (skip LocalFileSystem — already opened above)
        if not isinstance(new_mount.filesystem, LocalFileSystem) and hasattr(new_mount.filesystem, "open"):
            await new_mount.filesystem.open()

        self._ctx.registry.add_mount(new_mount)

        if not new_mount.hidden:
            self._ctx.initialized = True

    async def _create_engine_mount(
        self,
        path: str,
        config: EngineConfig,
        backend: GroverFileSystem | None,
        mount_type: str | None,
        permission: Permission,
        label: str,
        hidden: bool,
    ) -> Mount:
        """Build a Mount from an EngineConfig — Grover owns the engine."""
        engine = config.create_engine()
        sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dialect = engine.dialect.name

        if backend is None:
            backend = DatabaseFileSystem()  # type: ignore[assignment]

        # Configure the filesystem with settings from EngineConfig
        if isinstance(backend, DatabaseFileSystem):
            backend._configure(config, dialect)

        # Ensure schema and base tables exist
        if config.create_tables:
            schema = config.schema

            # Create schema if needed (non-SQLite only)
            if schema:
                async with engine.begin() as conn:
                    schema_created = await conn.run_sync(lambda c: ensure_schema(c, dialect, schema))
                if schema_created:
                    logger.info('Schema created: "%s"', schema)

            # Gather all tables to create
            tables = [
                config.file_model.__table__,  # type: ignore[attr-defined]
                config.file_version_model.__table__,  # type: ignore[attr-defined]
                config.file_chunk_model.__table__,  # type: ignore[attr-defined]
                config.file_connection_model.__table__,  # type: ignore[attr-defined]
            ]
            if isinstance(backend, SupportsReBAC):
                tables.append(FileShareModel.__table__)  # type: ignore[unresolved-attribute]

            table_names = [t.name for t in tables]

            # Check which tables already exist
            async with engine.begin() as conn:
                existing_before = await conn.run_sync(lambda c: check_tables_exist(c, table_names, schema))

            # Create tables (with schema_translate_map if schema is set)
            def _create_tables_sync(c: Connection) -> None:
                if schema:
                    c = c.execution_options(schema_translate_map={None: schema})
                for t in tables:
                    t.create(c, checkfirst=True)

            async with engine.begin() as conn:
                await conn.run_sync(_create_tables_sync)

            # Print newly created tables
            new_tables = [name for name in table_names if name not in existing_before]
            if new_tables:
                table_list = ", ".join(new_tables)
                if schema:
                    logger.info('Tables created in schema "%s": %s', schema, table_list)
                else:
                    logger.info("Tables created: %s", table_list)

        return Mount(
            path=path,
            filesystem=backend,
            session_factory=sf,
            engine=engine,
            mount_type=mount_type or "vfs",
            permission=permission,
            label=label,
            hidden=hidden,
        )

    def _create_session_factory_mount(
        self,
        path: str,
        config: SessionConfig,
        backend: GroverFileSystem | None,
        mount_type: str | None,
        permission: Permission,
        label: str,
        hidden: bool,
    ) -> Mount:
        """Build a Mount from a SessionConfig — app owns the engine."""
        # Infer dialect from session factory's bind
        dialect = config.dialect
        if dialect is None:
            bind = getattr(config.session_factory, "kw", {}).get("bind")
            if bind is not None:
                dialect = bind.dialect.name
        if dialect is None:
            raise ValueError("Cannot infer dialect from session factory. Pass dialect= explicitly in SessionConfig.")

        if backend is None:
            backend = DatabaseFileSystem()  # type: ignore[assignment]

        # Configure the filesystem with settings from SessionConfig
        if isinstance(backend, DatabaseFileSystem):
            backend._configure(config, dialect)

        return Mount(
            path=path,
            filesystem=backend,
            session_factory=config.session_factory,
            mount_type=mount_type or "vfs",
            permission=permission,
            label=label,
            hidden=hidden,
        )

    async def unmount(self, path: str) -> None:
        """Unmount the backend at *path*."""

        path = normalize_path(path).rstrip("/")
        try:
            mount, _ = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return

        # Only unmount if the path is an exact mount point, not a subpath
        if mount.path != path:
            return

        backend = mount.filesystem
        if hasattr(backend, "close"):
            await backend.close()
        # Dispose engine if Grover owns it (EngineConfig path)
        if mount.engine is not None:
            await mount.engine.dispose()
        self._ctx.registry.remove_mount(path)

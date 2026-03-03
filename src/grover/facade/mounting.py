"""MountMixin — mount lifecycle methods for GroverAsync."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import MountNotFoundError, SchemaIncompatibleError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.permissions import Permission
from grover.fs.protocol import SupportsReBAC
from grover.fs.providers.graph.rustworkx import RustworkxGraph
from grover.fs.utils import normalize_path
from grover.models.chunks import FileChunk
from grover.models.connections import FileConnection
from grover.models.files import File, FileVersion
from grover.models.shares import FileShare
from grover.mount import Mount

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from grover.facade.context import GroverContext
    from grover.fs.protocol import StorageBackend
    from grover.fs.providers.protocols import EmbeddingProvider, SearchProvider
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase

logger = logging.getLogger(__name__)


class MountMixin:
    """Mount lifecycle methods extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Mount / Unmount
    # ------------------------------------------------------------------

    async def add_mount(
        self,
        mount_or_path: str | Mount | None = None,
        filesystem: StorageBackend | None = None,
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
        path: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        search_provider: SearchProvider | None = None,
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

            # With search — pass embedding_provider to enable vector search
            await g.add_mount("/data", engine=engine, embedding_provider=embed)

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
        if not isinstance(new_mount.filesystem, LocalFileSystem) and hasattr(
            new_mount.filesystem, "open"
        ):
            await new_mount.filesystem.open()

        self._ctx.registry.add_mount(new_mount)

        if not new_mount.hidden:
            self._ctx.initialized = True

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
        self._ctx.registry.remove_mount(path)

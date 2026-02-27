"""GroverContext — shared state and helpers for GroverAsync mixins."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from grover.events import IndexingMode
from grover.fs.exceptions import MountNotFoundError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.permissions import Permission
from grover.search._engine import SearchEngine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.events import EventBus, FileEvent
    from grover.graph.analyzers import AnalyzerRegistry
    from grover.graph.protocols import GraphStore
    from grover.mount import Mount
    from grover.mount.mounts import MountRegistry
    from grover.search.fulltext.protocol import FullTextStore
    from grover.search.protocols import EmbeddingProvider, VectorStore
    from grover.types import FileInfoResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GroverContext:
    """Shared state for GroverAsync operations."""

    event_bus: EventBus
    registry: MountRegistry
    analyzer_registry: AnalyzerRegistry
    embedding_provider: EmbeddingProvider | None = None
    explicit_vector_store: VectorStore | None = None
    explicit_data_dir: Path | None = None
    meta_fs: LocalFileSystem | None = None
    meta_data_dir: Path | None = None
    indexing_mode: IndexingMode = IndexingMode.BACKGROUND
    closed: bool = False

    # ------------------------------------------------------------------
    # Session management & helpers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session_for(self, mount: Mount) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or ``None`` for non-SQL backends."""
        if mount.session_factory is None:
            yield None
            return

        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def emit(self, event: FileEvent) -> None:
        """Emit a file event via the event bus."""
        await self.event_bus.emit(event)

    async def drain(self) -> None:
        """Drain all pending background events."""
        await self.event_bus.drain()

    def check_writable(self, virtual_path: str) -> str | None:
        """Return an error message if *virtual_path* is read-only, else ``None``.

        Replaces the previous raise-based pattern to avoid unnecessary
        exception overhead in the common (writable) case.
        """
        try:
            perm = self.registry.get_permission(virtual_path)
        except MountNotFoundError as e:
            return str(e)
        if perm == Permission.READ_ONLY:
            return f"Cannot write to read-only path: {virtual_path}"
        return None

    @staticmethod
    def get_capability(backend: object, protocol: type[T]) -> T | None:
        """Return *backend* if it satisfies *protocol*, else ``None``."""
        if isinstance(backend, protocol):
            return backend
        return None

    def prefix_path(self, path: str | None, mount_path: str) -> str | None:
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def prefix_file_info(self, info: FileInfoResult, mount: Mount) -> FileInfoResult:
        prefixed_path = self.prefix_path(info.path, mount.path) or info.path
        info.path = prefixed_path
        info.mount_type = mount.mount_type
        info.permission = self.registry.get_permission(prefixed_path).value
        return info

    # ------------------------------------------------------------------
    # Per-mount graph / search resolution
    # ------------------------------------------------------------------

    def resolve_graph(self, path: str) -> GraphStore:
        """Return the graph for the mount owning *path*."""
        try:
            mount, _rel = self.registry.resolve(path)
        except MountNotFoundError:
            msg = f"No mount found for path: {path!r}"
            raise RuntimeError(msg) from None
        if mount.graph is None:
            msg = f"No graph on mount at {mount.path}"
            raise RuntimeError(msg)
        return mount.graph

    def resolve_search_engine(self, path: str) -> SearchEngine | None:
        """Return the search engine for the mount owning *path*, or None."""
        try:
            mount, _rel = self.registry.resolve(path)
        except MountNotFoundError:
            return None
        return mount.search

    def resolve_graph_any(self, path: str | None = None) -> GraphStore:
        """Get graph for a specific path, or first available mount's graph."""
        if path is not None:
            return self.resolve_graph(path)
        for mount in self.registry.list_visible_mounts():
            if mount.graph is not None:
                return mount.graph
        msg = "No graph available on any mount"
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Search / Graph factory helpers
    # ------------------------------------------------------------------

    def create_search_engine(self, *, lexical: FullTextStore | None = None) -> SearchEngine | None:
        """Create a new SearchEngine for a mount."""
        vector: Any = None
        embedding = self.embedding_provider

        if self.explicit_vector_store is not None:
            vector = self.explicit_vector_store
        elif embedding is not None:
            from grover.search.stores.local import LocalVectorStore

            vector = LocalVectorStore(dimension=embedding.dimensions)

        # If we have nothing at all, no search engine
        if vector is None and embedding is None and lexical is None:
            return None

        return SearchEngine(vector=vector, embedding=embedding, lexical=lexical)

    async def create_fulltext_store(self, config: Mount) -> FullTextStore | None:
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

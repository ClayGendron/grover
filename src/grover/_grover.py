"""Main Grover class — lifecycle, sync wrappers, integration."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from grover.events import EventBus, EventType, FileEvent
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.unified import UnifiedFileSystem
from grover.graph._graph import Graph
from grover.graph.analyzers import AnalyzerRegistry
from grover.models.edges import GroverEdge
from grover.models.embeddings import Embedding
from grover.search._index import SearchIndex, SearchResult
from grover.search.extractors import extract_from_chunks, extract_from_file

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class Grover:
    """Facade wiring filesystem, graph, analyzers, event bus, and search.

    Presents a synchronous API backed by a private event loop in a
    background thread.  All subsystems are async internally; the loop
    bridges the gap so callers can use Grover from plain sync code,
    Jupyter notebooks, or inside an existing async context (FastAPI).

    Usage::

        with Grover("/path/to/project") as g:
            g.write("/project/hello.py", "print('hi')")
            results = g.search("hello")
            g.save()
    """

    def __init__(
        self,
        source: str,
        *,
        data_dir: str | None = None,
        mount_path: str = "/project",
        embedding_provider: Any = None,
    ) -> None:
        self._closed = False
        self._mount_path = mount_path

        # Private event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        self._run(
            self._async_init(source, data_dir, mount_path, embedding_provider)
        )

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------

    async def _async_init(
        self,
        source: str,
        data_dir: str | None,
        mount_path: str,
        embedding_provider: Any,
    ) -> None:
        # 1. Detect backend
        if "://" in source:
            await self._init_database_backend(source, data_dir)
            mount_type = "vfs"
        else:
            self._init_local_backend(source, data_dir)
            mount_type = "local"

        # 2. Event bus
        self._event_bus = EventBus()

        # 3. Mount registry
        self._registry = MountRegistry()
        self._registry.add_mount(
            MountConfig(
                mount_path=mount_path,
                backend=self._backend,
                mount_type=mount_type,
            )
        )

        # Mount for .grover internal storage (chunks, etc.)
        self._grover_fs: LocalFileSystem | None = None
        if self._local_fs is not None:
            self._grover_fs = LocalFileSystem(
                workspace_dir=self._local_fs.data_dir,
                data_dir=self._local_fs.data_dir / "_meta",
            )
            self._registry.add_mount(
                MountConfig(
                    mount_path="/.grover",
                    backend=self._grover_fs,
                    mount_type="local",
                )
            )

        # 4. Unified filesystem
        self._ufs = UnifiedFileSystem(self._registry, self._event_bus)
        await self._ufs.__aenter__()

        # 5. Extra tables (edges + embeddings) on the shared engine
        await self._ensure_extra_tables()

        # 6. Analyzer registry
        self._analyzer_registry = AnalyzerRegistry()

        # 7. Graph
        self._graph = Graph()

        # 8. Search index (optional)
        self._search_index: SearchIndex | None = None
        if embedding_provider is not None:
            self._search_index = SearchIndex(embedding_provider)
        else:
            try:
                from grover.search.providers.sentence_transformers import (
                    SentenceTransformerProvider,
                )

                self._search_index = SearchIndex(SentenceTransformerProvider())
            except Exception:
                logger.debug(
                    "No embedding provider available; search disabled"
                )

        # 9. Register event handlers
        self._event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._event_bus.register(
            EventType.FILE_RESTORED, self._on_file_restored
        )

        # 10. Load existing state
        await self._load_existing_state()

    def _init_local_backend(
        self, source: str, data_dir: str | None
    ) -> None:
        kw: dict[str, Any] = {"workspace_dir": source}
        if data_dir is not None:
            kw["data_dir"] = data_dir
        self._local_fs = LocalFileSystem(**kw)
        self._backend: Any = self._local_fs

    async def _init_database_backend(
        self, source: str, data_dir: str | None
    ) -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from grover.fs.database_fs import DatabaseFileSystem

        engine = create_async_engine(source, echo=False)
        session_factory = async_sessionmaker(engine)
        dialect = engine.dialect.name
        self._local_fs = None
        self._db_engine = engine
        self._backend = DatabaseFileSystem(session_factory, dialect)

    async def _ensure_extra_tables(self) -> None:
        """Create GroverEdge and Embedding tables on the shared engine."""
        if self._local_fs is not None:
            await self._local_fs._ensure_db()
            engine = self._local_fs._engine
        else:
            engine = self._db_engine

        async with engine.begin() as conn:  # type: ignore[union-attr]
            await conn.run_sync(
                lambda c: GroverEdge.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )
            await conn.run_sync(
                lambda c: Embedding.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )

    async def _load_existing_state(self) -> None:
        """Hydrate graph from SQL and search from disk if available."""
        if self._local_fs is not None:
            session = await self._local_fs._get_session()
            try:
                await self._graph.from_sql(session)
                await self._local_fs._commit(session)
            except Exception:
                logger.debug("No existing graph state to load", exc_info=True)

            # Load search index from disk
            if self._search_index is not None:
                search_dir = self._local_fs.data_dir / "search"
                meta_file = search_dir / "search_meta.json"
                if meta_file.exists():
                    try:
                        self._search_index.load(str(search_dir))
                    except Exception:
                        logger.debug(
                            "Failed to load search index", exc_info=True
                        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Submit *coro* to the private loop and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def _data_dir(self) -> Path | None:
        if self._local_fs is not None:
            return self._local_fs.data_dir
        return None

    # ------------------------------------------------------------------
    # Event handlers (async, registered on EventBus)
    # ------------------------------------------------------------------

    async def _on_file_written(self, event: FileEvent) -> None:
        if "/.grover/" in event.path:
            return
        content = event.content
        if content is None:
            result = await self._ufs.read(event.path)
            if not result.success:
                return
            content = self._ufs._extract_raw_content(result.content)
        if content is not None:
            await self._analyze_and_integrate(event.path, content)

    async def _on_file_deleted(self, event: FileEvent) -> None:
        if "/.grover/" in event.path:
            return
        if self._graph.has_node(event.path):
            self._graph.remove_file_subgraph(event.path)
        if self._search_index is not None:
            self._search_index.remove_file(event.path)

    async def _on_file_moved(self, event: FileEvent) -> None:
        if event.old_path and "/.grover/" not in event.old_path:
            if self._graph.has_node(event.old_path):
                self._graph.remove_file_subgraph(event.old_path)
            if self._search_index is not None:
                self._search_index.remove_file(event.old_path)

        if "/.grover/" in event.path:
            return
        result = await self._ufs.read(event.path)
        if result.success:
            content = self._ufs._extract_raw_content(result.content)
            if content is not None:
                await self._analyze_and_integrate(event.path, content)

    async def _on_file_restored(self, event: FileEvent) -> None:
        await self._on_file_written(event)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _analyze_and_integrate(self, path: str, content: str) -> dict[str, int]:
        """Analyze a file and integrate results into graph and search.

        Returns counts of chunks created and edges added.
        """
        stats = {"chunks_created": 0, "edges_added": 0}

        # Remove old subgraph and search entries
        if self._graph.has_node(path):
            self._graph.remove_file_subgraph(path)
        if self._search_index is not None:
            self._search_index.remove_file(path)

        # Add file node
        self._graph.add_node(path)

        # Run analyzer
        analysis = self._analyzer_registry.analyze_file(path, content)

        if analysis is not None:
            chunks, edges = analysis

            # Write chunks, add nodes + edges
            for chunk in chunks:
                await self._ufs.write(chunk.chunk_path, chunk.content)
                self._graph.add_node(
                    chunk.chunk_path,
                    parent_path=path,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    name=chunk.name,
                )
                self._graph.add_edge(path, chunk.chunk_path, "contains")
                stats["chunks_created"] += 1

            for edge in edges:
                meta: dict[str, Any] = dict(edge.metadata)
                self._graph.add_edge(
                    edge.source, edge.target, edge.edge_type, **meta
                )
                stats["edges_added"] += 1

            # Embed chunks for search
            if self._search_index is not None:
                embeddable = extract_from_chunks(chunks)
                if embeddable:
                    self._search_index.add_batch(embeddable)
        else:
            # Unsupported file type — embed whole file
            if self._search_index is not None:
                embeddable = extract_from_file(path, content)
                if embeddable:
                    self._search_index.add_batch(embeddable)

        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down subsystems, stop the event loop and join the thread."""
        if self._closed:
            return
        self._closed = True

        try:
            self._run(self._async_close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    async def _async_close(self) -> None:
        await self._async_save()
        await self._ufs.__aexit__(None, None, None)

    def __enter__(self) -> Grover:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Filesystem wrappers (sync)
    # ------------------------------------------------------------------

    def read(self, path: str) -> str | None:
        """Read file content at *path*, returning raw text or ``None``."""
        result = self._run(self._ufs.read(path))
        if not result.success:
            return None
        return self._ufs._extract_raw_content(result.content)

    def write(self, path: str, content: str) -> bool:
        """Write *content* to *path*. Returns ``True`` on success."""
        result = self._run(self._ufs.write(path, content))
        return result.success

    def edit(self, path: str, old: str, new: str) -> bool:
        """Replace *old* with *new* in the file at *path*."""
        result = self._run(self._ufs.edit(path, old, new))
        return result.success

    def delete(self, path: str) -> bool:
        """Delete the file at *path*."""
        result = self._run(self._ufs.delete(path))
        return result.success

    def list_dir(self, path: str = "/") -> list[dict[str, Any]]:
        """List entries under *path*."""
        result = self._run(self._ufs.list_dir(path))
        return [
            {"path": e.path, "name": e.name, "is_directory": e.is_directory}
            for e in result.entries
        ]

    def exists(self, path: str) -> bool:
        """Check whether *path* exists."""
        return self._run(self._ufs.exists(path))

    # ------------------------------------------------------------------
    # Graph query wrappers (sync — Graph methods are already sync)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Any]:
        """Nodes with edges pointing to *path* (predecessors)."""
        return self._graph.dependents(path)

    def dependencies(self, path: str) -> list[Any]:
        """Nodes that *path* points to (successors)."""
        return self._graph.dependencies(path)

    def impacts(self, path: str, max_depth: int = 3) -> list[Any]:
        """Reverse transitive reachability from *path*."""
        return self._graph.impacts(path, max_depth)

    def path_between(self, source: str, target: str) -> list[Any] | None:
        """Shortest path from *source* to *target*, or ``None``."""
        return self._graph.path_between(source, target)

    def contains(self, path: str) -> list[Any]:
        """Successors connected by "contains" edges."""
        return self._graph.contains(path)

    # ------------------------------------------------------------------
    # Search wrapper (sync — SearchIndex is already sync)
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Semantic search over indexed content."""
        if self._search_index is None:
            msg = (
                "Search is not available: no embedding provider configured. "
                "Install sentence-transformers or pass embedding_provider= to Grover()."
            )
            raise RuntimeError(msg)
        return self._search_index.search(query, k)

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    def index(self) -> dict[str, int]:
        """Walk the filesystem, analyze all files, build graph + search.

        Returns stats: ``{"files_scanned", "chunks_created", "edges_added"}``.
        """
        return self._run(self._async_index())

    async def _async_index(self) -> dict[str, int]:
        stats = {"files_scanned": 0, "chunks_created": 0, "edges_added": 0}
        await self._walk_and_index(self._mount_path, stats)
        await self._async_save()
        return stats

    async def _walk_and_index(
        self, path: str, stats: dict[str, int]
    ) -> None:
        result = await self._ufs.list_dir(path)
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
                file_stats = await self._analyze_and_integrate(
                    entry.path, content
                )
                stats["files_scanned"] += 1
                stats["chunks_created"] += file_stats["chunks_created"]
                stats["edges_added"] += file_stats["edges_added"]

    async def _read_file_content(self, path: str) -> str | None:
        """Read raw file content, trying UFS first then direct backend."""
        read_result = await self._ufs.read(path)
        if read_result.success:
            return self._ufs._extract_raw_content(read_result.content)

        # File may exist on disk but not in the DB — read directly
        if self._local_fs is not None:
            mount, rel_path = self._registry.resolve(path)
            backend = mount.backend
            if hasattr(backend, "_read_content"):
                content: str | None = await backend._read_content(rel_path)  # type: ignore[union-attr]
                return content

        return None

    def save(self) -> None:
        """Persist graph and search index to disk."""
        self._run(self._async_save())

    async def _async_save(self) -> None:
        # Save graph to SQL
        if self._local_fs is not None:
            session = await self._local_fs._get_session()
            await self._graph.to_sql(session)
            await self._local_fs._commit(session)

        # Save search index to disk
        if self._search_index is not None and self._data_dir is not None:
            search_dir = self._data_dir / "search"
            self._search_index.save(str(search_dir))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fs(self) -> UnifiedFileSystem:
        """The underlying ``UnifiedFileSystem`` (for advanced async use)."""
        return self._ufs

    @property
    def graph(self) -> Graph:
        """The knowledge graph."""
        return self._graph

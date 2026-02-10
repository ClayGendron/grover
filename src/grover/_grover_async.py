"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grover.events import EventBus, EventType, FileEvent
from grover.fs.exceptions import MountNotFoundError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.unified import UnifiedFileSystem
from grover.graph._graph import Graph
from grover.graph.analyzers import AnalyzerRegistry
from grover.models.edges import GroverEdge
from grover.models.embeddings import Embedding
from grover.search._index import SearchIndex, SearchResult
from grover.fs.types import ReadResult
from grover.search.extractors import extract_from_chunks, extract_from_file

if TYPE_CHECKING:
    from grover.fs.protocol import StorageBackend
    from grover.ref import Ref

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".grover" / "_default"


class GroverAsync:
    """Async facade wiring filesystem, graph, analyzers, event bus, and search.

    Mount-first API: create an instance, then mount backends.

    Direct access (outside context manager) — auto-commits per operation::

        g = GroverAsync()
        await g.mount("/app", db)
        await g.write("/app/test.py", "print('hi')")

    Batch/transaction mode (inside context manager)::

        async with g:
            await g.write("/app/a.py", "a")
            await g.write("/app/b.py", "b")
            # committed together on clean exit
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        embedding_provider: Any = None,
    ) -> None:
        self._explicit_data_dir = Path(data_dir) if data_dir else None
        self._in_transaction = False
        self._closed = False

        # Core subsystems (sync init)
        self._event_bus = EventBus()
        self._registry = MountRegistry()
        self._ufs = UnifiedFileSystem(self._registry, self._event_bus)
        self._analyzer_registry = AnalyzerRegistry()
        self._graph = Graph()

        # Internal metadata mount — lazily created on first mount()
        self._meta_fs: LocalFileSystem | None = None
        self._meta_data_dir: Path | None = None

        # Search index (optional)
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

        # Register event handlers
        self._event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._event_bus.register(
            EventType.FILE_RESTORED, self._on_file_restored
        )

    # ------------------------------------------------------------------
    # Context manager (transaction mode)
    # ------------------------------------------------------------------

    def _set_backends_transaction(self, in_transaction: bool) -> None:
        """Toggle ``in_transaction`` on every mounted backend."""
        for mount in self._registry.list_mounts():
            backend = mount.backend
            if hasattr(backend, "in_transaction"):
                backend.in_transaction = in_transaction

    async def __aenter__(self) -> GroverAsync:
        self._in_transaction = True
        self._set_backends_transaction(True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._in_transaction = False
        self._set_backends_transaction(False)

        # Commit or rollback all entered backends
        for backend in list(self._ufs._entered_backends):
            if hasattr(backend, "__aexit__"):
                try:
                    await backend.__aexit__(exc_type, exc_val, exc_tb)
                except Exception:
                    logger.warning("Backend exit failed", exc_info=True)
        self._ufs._entered_backends.clear()

        # Re-enter backends for future operations
        for mount in self._registry.list_mounts():
            backend = mount.backend
            if hasattr(backend, "__aenter__"):
                await backend.__aenter__()
                self._ufs._entered_backends.append(backend)

        if exc_type is None:
            await self._async_save()

    # ------------------------------------------------------------------
    # Mount / Unmount
    # ------------------------------------------------------------------

    async def mount(
        self,
        path: str,
        backend: StorageBackend | Any,
        *,
        mount_type: str | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        hidden: bool = False,
    ) -> None:
        """Mount a backend at *path*."""
        # Auto-detect mount type
        if mount_type is None:
            mount_type = "local" if isinstance(backend, LocalFileSystem) else "vfs"

        config = MountConfig(
            mount_path=path,
            backend=backend,
            mount_type=mount_type,
            permission=permission,
            label=label,
            hidden=hidden,
        )
        self._registry.add_mount(config)

        # Enter the backend context
        await self._ufs.enter_backend(backend)

        # Lazily initialise meta_fs on first non-hidden mount
        if not hidden and self._meta_fs is None:
            await self._init_meta_fs(backend)

    async def unmount(self, path: str) -> None:
        """Unmount the backend at *path*."""
        from grover.fs.utils import normalize_path

        path = normalize_path(path).rstrip("/")
        if path == "/.grover":
            raise ValueError("Cannot unmount /.grover")

        try:
            mount, _ = self._registry.resolve(path)
        except MountNotFoundError:
            return

        backend = mount.backend
        await self._ufs.exit_backend(backend)
        self._registry.remove_mount(path)

        # Clean graph nodes and search entries with this mount prefix
        prefix = path + "/"
        nodes_to_remove = [
            n for n in self._graph.nodes()
            if n == path or n.startswith(prefix)
        ]
        for node in nodes_to_remove:
            if self._graph.has_node(node):
                self._graph.remove_file_subgraph(node)
            if self._search_index is not None:
                self._search_index.remove_file(node)

    # ------------------------------------------------------------------
    # Internal metadata mount
    # ------------------------------------------------------------------

    async def _init_meta_fs(self, first_backend: Any) -> None:
        """Create the internal /.grover metadata mount."""
        # Determine data_dir
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

        self._registry.add_mount(
            MountConfig(
                mount_path="/.grover",
                backend=self._meta_fs,
                mount_type="local",
                hidden=True,
            )
        )
        await self._ufs.enter_backend(self._meta_fs)

        # Create extra tables on the meta engine
        await self._ensure_extra_tables()

        # Load existing state
        await self._load_existing_state()

    async def _ensure_extra_tables(self) -> None:
        """Create GroverEdge and Embedding tables on the meta engine."""
        if self._meta_fs is None:
            return
        await self._meta_fs._ensure_db()
        engine = self._meta_fs._engine
        if engine is None:
            return

        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: GroverEdge.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )
            await conn.run_sync(
                lambda c: Embedding.__table__.create(c, checkfirst=True)  # type: ignore[unresolved-attribute]
            )

    async def _load_existing_state(self) -> None:
        """Hydrate graph from SQL and search from disk if available."""
        if self._meta_fs is None:
            return

        session = await self._meta_fs._get_session()
        try:
            # Find the first non-hidden mount's backend for file_model
            file_model = None
            for mount in self._registry.list_visible_mounts():
                if hasattr(mount.backend, "file_model"):
                    file_model = mount.backend.file_model
                    break
            await self._graph.from_sql(session, file_model=file_model)  # type: ignore[arg-type]
            await self._meta_fs._commit(session)
        except Exception:
            logger.debug("No existing graph state to load", exc_info=True)

        # Load search index from disk
        if self._search_index is not None and self._meta_data_dir is not None:
            search_dir = self._meta_data_dir / "search"
            meta_file = search_dir / "search_meta.json"
            if meta_file.exists():
                try:
                    self._search_index.load(str(search_dir))
                except Exception:
                    logger.debug(
                        "Failed to load search index", exc_info=True
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
            result = await self._ufs.read(event.path)
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
        if self._graph.has_node(event.path):
            self._graph.remove_file_subgraph(event.path)
        if self._search_index is not None:
            self._search_index.remove_file(event.path)

    async def _on_file_moved(self, event: FileEvent) -> None:
        if self._meta_fs is None:
            return
        if event.old_path and "/.grover/" not in event.old_path:
            if self._graph.has_node(event.old_path):
                self._graph.remove_file_subgraph(event.old_path)
            if self._search_index is not None:
                self._search_index.remove_file(event.old_path)

        if "/.grover/" in event.path:
            return
        result = await self._ufs.read(event.path)
        if result.success:
            content = result.content
            if content is not None:
                await self._analyze_and_integrate(event.path, content)

    async def _on_file_restored(self, event: FileEvent) -> None:
        await self._on_file_written(event)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _analyze_and_integrate(
        self, path: str, content: str
    ) -> dict[str, int]:
        """Analyze a file and integrate results into graph and search."""
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
    # FS Operations
    # ------------------------------------------------------------------

    async def read(self, path: str) -> ReadResult:
        """Read file content at *path*."""
        return await self._ufs.read(path)

    async def write(self, path: str, content: str) -> bool:
        """Write *content* to *path*. Returns ``True`` on success."""
        result = await self._ufs.write(path, content)
        return result.success

    async def edit(self, path: str, old: str, new: str) -> bool:
        """Replace *old* with *new* in the file at *path*."""
        result = await self._ufs.edit(path, old, new)
        return result.success

    async def delete(self, path: str) -> bool:
        """Delete the file at *path*."""
        result = await self._ufs.delete(path)
        return result.success

    async def list_dir(self, path: str = "/") -> list[dict[str, Any]]:
        """List entries under *path*."""
        result = await self._ufs.list_dir(path)
        return [
            {"path": e.path, "name": e.name, "is_directory": e.is_directory}
            for e in result.entries
        ]

    async def exists(self, path: str) -> bool:
        """Check whether *path* exists."""
        return await self._ufs.exists(path)

    # ------------------------------------------------------------------
    # Graph query wrappers (sync — Graph methods are already sync)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Ref]:
        """Nodes with edges pointing to *path* (predecessors)."""
        return self._graph.dependents(path)

    def dependencies(self, path: str) -> list[Ref]:
        """Nodes that *path* points to (successors)."""
        return self._graph.dependencies(path)

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        """Reverse transitive reachability from *path*."""
        return self._graph.impacts(path, max_depth)

    def path_between(self, source: str, target: str) -> list[Ref] | None:
        """Shortest path from *source* to *target*, or ``None``."""
        return self._graph.path_between(source, target)

    def contains(self, path: str) -> list[Ref]:
        """Successors connected by "contains" edges."""
        return self._graph.contains(path)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Semantic search over indexed content."""
        if self._search_index is None:
            msg = (
                "Search is not available: no embedding provider configured. "
                "Install sentence-transformers or pass embedding_provider= to GroverAsync()."
            )
            raise RuntimeError(msg)
        return self._search_index.search(query, k)

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    async def index(self, mount_path: str | None = None) -> dict[str, int]:
        """Walk the filesystem, analyze all files, build graph + search.

        If *mount_path* is given, only that mount is walked.
        """
        stats = {"files_scanned": 0, "chunks_created": 0, "edges_added": 0}

        if mount_path is not None:
            await self._walk_and_index(mount_path, stats)
        else:
            for mount in self._registry.list_visible_mounts():
                await self._walk_and_index(mount.mount_path, stats)

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
            return read_result.content

        # File may exist on disk but not in the DB — read directly
        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None
        backend = mount.backend
        if hasattr(backend, "_read_content"):
            content: str | None = await backend._read_content(rel_path)  # type: ignore[union-attr]
            return content

        return None

    async def save(self) -> None:
        """Persist graph and search index."""
        await self._async_save()

    async def _async_save(self) -> None:
        # Save graph to SQL
        if self._meta_fs is not None:
            session = await self._meta_fs._get_session()
            await self._graph.to_sql(session)
            await self._meta_fs._commit(session)

        # Save search index to disk
        if self._search_index is not None and self._meta_data_dir is not None:
            search_dir = self._meta_data_dir / "search"
            self._search_index.save(str(search_dir))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down subsystems."""
        if self._closed:
            return
        self._closed = True

        await self._async_save()
        # Exit all backends
        for backend in list(self._ufs._entered_backends):
            if hasattr(backend, "__aexit__"):
                try:
                    await backend.__aexit__(None, None, None)
                except Exception:
                    logger.warning("Backend exit failed", exc_info=True)
        self._ufs._entered_backends.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Graph:
        """The knowledge graph."""
        return self._graph

    @property
    def fs(self) -> UnifiedFileSystem:
        """The underlying ``UnifiedFileSystem``."""
        return self._ufs

"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from grover.events import EventBus, EventType, IndexingMode
from grover.facade.connections import ConnectionMixin
from grover.facade.context import GroverContext
from grover.facade.file_ops import FileOpsMixin
from grover.facade.graph_ops import GraphOpsMixin
from grover.facade.indexing import IndexMixin
from grover.facade.mounting import MountMixin
from grover.facade.search_ops import SearchOpsMixin
from grover.facade.sharing import ShareMixin
from grover.facade.version_trash import VersionTrashMixin
from grover.graph.analyzers import AnalyzerRegistry
from grover.mount.mounts import MountRegistry

if TYPE_CHECKING:
    from grover.search.protocols import EmbeddingProvider, VectorStore


class GroverAsync(
    MountMixin,
    FileOpsMixin,
    SearchOpsMixin,
    GraphOpsMixin,
    VersionTrashMixin,
    ShareMixin,
    ConnectionMixin,
    IndexMixin,
):
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
        indexing_mode: IndexingMode = IndexingMode.BACKGROUND,
        debounce_delay: float = 0.1,
    ) -> None:
        self._ctx = GroverContext(
            event_bus=EventBus(indexing_mode=indexing_mode, debounce_delay=debounce_delay),
            registry=MountRegistry(),
            analyzer_registry=AnalyzerRegistry(),
            embedding_provider=embedding_provider,
            explicit_vector_store=vector_store,
            explicit_data_dir=Path(data_dir) if data_dir else None,
            indexing_mode=indexing_mode,
        )

        # Register event handlers
        self._ctx.event_bus.register(EventType.FILE_WRITTEN, self._on_file_written)
        self._ctx.event_bus.register(EventType.FILE_DELETED, self._on_file_deleted)
        self._ctx.event_bus.register(EventType.FILE_MOVED, self._on_file_moved)
        self._ctx.event_bus.register(EventType.FILE_RESTORED, self._on_file_restored)
        self._ctx.event_bus.register(EventType.CONNECTION_ADDED, self._on_connection_added)
        self._ctx.event_bus.register(EventType.CONNECTION_DELETED, self._on_connection_deleted)

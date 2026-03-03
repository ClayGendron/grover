"""GroverAsync — primary async class with mount-first API."""

from __future__ import annotations

from grover.analyzers import AnalyzerRegistry
from grover.facade.connections import ConnectionMixin
from grover.facade.context import GroverContext
from grover.facade.file_ops import FileOpsMixin
from grover.facade.graph_ops import GraphOpsMixin
from grover.facade.indexing import IndexMixin
from grover.facade.mounting import MountMixin
from grover.facade.search_ops import SearchOpsMixin
from grover.facade.sharing import ShareMixin
from grover.facade.version_trash import VersionTrashMixin
from grover.mount.mounts import MountRegistry
from grover.worker import BackgroundWorker, IndexingMode


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
    """Async facade wiring filesystem, graph, analyzers, worker, and search.

    Mount-first API: create an instance, then add mounts.

    Engine-based DB mount (primary API)::

        engine = create_async_engine("postgresql+asyncpg://...")
        g = GroverAsync()
        await g.add_mount("/data", engine=engine)

    With search (pass embedding_provider to add_mount)::

        g = GroverAsync()
        await g.add_mount("/data", engine=engine, embedding_provider=embed)

    Direct access — auto-commits per operation::

        g = GroverAsync()
        await g.add_mount("/app", backend)
        await g.write("/app/test.py", "print('hi')")
    """

    def __init__(
        self,
        *,
        indexing_mode: IndexingMode = IndexingMode.BACKGROUND,
        debounce_delay: float = 0.1,
    ) -> None:
        self._ctx = GroverContext(
            worker=BackgroundWorker(indexing_mode=indexing_mode, debounce_delay=debounce_delay),
            registry=MountRegistry(),
            analyzer_registry=AnalyzerRegistry(),
            indexing_mode=indexing_mode,
        )

"""GroverAsync and Grover — async and sync clients.

``GroverAsync`` is the async facade for long-running application servers.
Errors return as ``GroverResult(success=False)``.

``Grover`` is the sync facade for data pipelines and backend processes.
It sets ``raise_on_error=True`` so that failed operations raise
``GroverError`` (or a subclass) immediately.

Usage (async)::

    g = GroverAsync()
    await g.add_mount("data", engine_url="sqlite+aiosqlite:///my.db")
    result = await g.read("/data/hello.txt")

Usage (sync)::

    g = Grover()
    g.add_mount("data", engine_url="sqlite+aiosqlite:///my.db")
    candidate = g.read("/data/hello.txt")  # raises NotFoundError if missing
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, TypeVar

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.base import GroverFileSystem
from grover.models import GroverObject, GroverObjectBase
from grover.paths import normalize_path

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from grover.embedding import EmbeddingProvider
    from grover.query import QueryPlan
    from grover.results import Candidate, GroverResult
    from grover.vector_store import VectorStore

_T = TypeVar("_T")


class GroverAsync(GroverFileSystem):
    """Async facade — storageless router with mount-first API.

    All filesystem operations are inherited from ``GroverFileSystem``.
    This subclass adds a rich ``add_mount()`` that constructs backends
    from engine URLs, engines, or session factories, plus lifecycle
    management via ``close()``.
    """

    def __init__(self) -> None:
        super().__init__(storage=False)

    # ------------------------------------------------------------------
    # Mount lifecycle
    # ------------------------------------------------------------------

    async def add_mount(  # type: ignore[override]
        self,
        name: str,
        *,
        filesystem: GroverFileSystem | None = None,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        engine_url: str | None = None,
        model: type[GroverObjectBase] = GroverObject,
        create_tables: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        user_scoped: bool = False,
    ) -> None:
        """Mount a filesystem at ``/<name>``.

        Exactly one of ``filesystem``, ``engine``, ``session_factory``, or
        ``engine_url`` must be provided.

        When ``engine_url`` is provided, an ``AsyncEngine`` is created
        automatically.  If ``create_tables`` is ``True`` (the default), the
        ``grover_objects`` table is created before mounting.

        ``embedding_provider`` and ``vector_store`` are injected onto the
        filesystem if it is a ``DatabaseFileSystem`` and does not already
        have them set.
        """
        fs: GroverFileSystem

        if filesystem is not None:
            fs = filesystem

        elif engine is not None:
            fs = DatabaseFileSystem(
                engine=engine,
                model=model,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                user_scoped=user_scoped,
            )

        elif session_factory is not None:
            fs = DatabaseFileSystem(
                session_factory=session_factory,
                model=model,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                user_scoped=user_scoped,
            )

        elif engine_url is not None:
            engine = create_async_engine(engine_url)
            if create_tables:
                async with engine.begin() as conn:
                    await conn.run_sync(SQLModel.metadata.create_all)
            fs = DatabaseFileSystem(
                engine=engine,
                model=model,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                user_scoped=user_scoped,
            )

        else:
            msg = (
                "add_mount() requires one of: filesystem, engine, "
                "session_factory, or engine_url"
            )
            raise ValueError(msg)

        # Inject providers onto pre-built filesystems if not already set
        if isinstance(fs, DatabaseFileSystem):
            if embedding_provider is not None and fs._embedding_provider is None:
                fs._embedding_provider = embedding_provider
            if vector_store is not None and fs._vector_store is None:
                fs._vector_store = vector_store

        path = normalize_path(f"/{name}")
        await super().add_mount(path, fs)

    async def remove_mount(self, name: str) -> None:  # type: ignore[override]
        """Unmount ``/<name>`` and dispose its engine if present."""
        path = normalize_path(f"/{name}")
        fs = self._mounts.get(path)
        await super().remove_mount(path)
        if fs is not None:
            engine = fs._engine
            if engine is not None:
                await engine.dispose()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Dispose all engines and clear mounts."""
        for fs in self._mounts.values():
            engine = fs._engine
            if engine is not None:
                await engine.dispose()
        self._mounts.clear()
        self._sorted_mount_paths.clear()


# ======================================================================
# Grover — synchronous facade with raise-on-error
# ======================================================================


class Grover:
    """Synchronous facade for data pipelines and backend processes.

    Sets ``raise_on_error=True`` on the internal ``GroverAsync`` so that
    all mounted filesystems raise ``GroverError`` (or subclasses) on
    failure instead of returning ``GroverResult(success=False)``.

    Single-path operations (``read``, ``write``, ``edit``, ``delete``,
    ``stat``, ``mkdir``, ``mkconn``) return ``Candidate`` directly.
    Multi-result operations (search, graph, listing) return ``GroverResult``.

    Usage::

        g = Grover()
        g.add_mount("data", engine_url="sqlite+aiosqlite:///my.db")
        g.write("/data/hello.txt", "content")       # raises on failure
        c = g.read("/data/hello.txt")                # returns Candidate
        print(c.content)
        g.close()
    """

    def __init__(self) -> None:
        self._closed = False
        self._lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._async = GroverAsync()
        self._async._raise_on_error = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Coroutine[object, object, _T]) -> _T:
        """Submit *coro* to the private loop and block for the result."""
        with self._lock:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_mount(
        self,
        name: str,
        *,
        filesystem: GroverFileSystem | None = None,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        engine_url: str | None = None,
        model: type[GroverObjectBase] = GroverObject,
        create_tables: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        user_scoped: bool = False,
    ) -> None:
        """Mount a filesystem at ``/<name>``."""
        self._run(
            self._async.add_mount(
                name,
                filesystem=filesystem,
                engine=engine,
                session_factory=session_factory,
                engine_url=engine_url,
                model=model,
                create_tables=create_tables,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                user_scoped=user_scoped,
            )
        )

    def remove_mount(self, name: str) -> None:
        """Unmount ``/<name>`` and dispose its engine."""
        self._run(self._async.remove_mount(name))

    def close(self) -> None:
        """Dispose all engines, stop the event loop, join the thread."""
        if self._closed:
            return
        self._closed = True
        try:
            self._run(self._async.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # CRUD — single-path operations returning Candidate
    # ------------------------------------------------------------------

    def read(self, path: str, *, user_id: str | None = None) -> Candidate:
        """Read file content. Raises ``NotFoundError`` if missing."""
        result = self._run(self._async.read(path, user_id=user_id))
        assert result.file is not None
        return result.file

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> Candidate:
        """Write content to *path*. Raises on conflict."""
        result = self._run(
            self._async.write(path, content, overwrite=overwrite, user_id=user_id)
        )
        assert result.file is not None
        return result.file

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> Candidate:
        """Replace *old* with *new* in the file at *path*."""
        result = self._run(
            self._async.edit(path, old, new, replace_all=replace_all, user_id=user_id)
        )
        assert result.file is not None
        return result.file

    def delete(
        self,
        path: str,
        *,
        permanent: bool = False,
        cascade: bool = True,
        user_id: str | None = None,
    ) -> Candidate:
        """Delete the object at *path*."""
        result = self._run(
            self._async.delete(path, permanent=permanent, cascade=cascade, user_id=user_id)
        )
        assert result.file is not None
        return result.file

    def stat(self, path: str, *, user_id: str | None = None) -> Candidate:
        """Return metadata for *path*."""
        result = self._run(self._async.stat(path, user_id=user_id))
        assert result.file is not None
        return result.file

    def mkdir(self, path: str, *, user_id: str | None = None) -> Candidate:
        """Create a directory at *path*."""
        result = self._run(self._async.mkdir(path, user_id=user_id))
        assert result.file is not None
        return result.file

    def mkconn(
        self,
        source: str,
        target: str,
        connection_type: str,
        *,
        user_id: str | None = None,
    ) -> Candidate:
        """Create a connection from *source* to *target*."""
        result = self._run(
            self._async.mkconn(source, target, connection_type, user_id=user_id)
        )
        assert result.file is not None
        return result.file

    # ------------------------------------------------------------------
    # CRUD — multi-result operations returning GroverResult
    # ------------------------------------------------------------------

    def move(
        self,
        src: str,
        dest: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> GroverResult:
        """Move *src* to *dest*."""
        return self._run(
            self._async.move(src, dest, overwrite=overwrite, user_id=user_id)
        )

    def copy(
        self,
        src: str,
        dest: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> GroverResult:
        """Copy *src* to *dest*."""
        return self._run(
            self._async.copy(src, dest, overwrite=overwrite, user_id=user_id)
        )

    def ls(self, path: str = "/", *, user_id: str | None = None) -> GroverResult:
        """List entries under *path*."""
        return self._run(self._async.ls(path, user_id=user_id))

    def tree(
        self,
        path: str,
        *,
        max_depth: int | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Recursive listing under *path*."""
        return self._run(self._async.tree(path, max_depth=max_depth, user_id=user_id))

    # ------------------------------------------------------------------
    # Search — returning GroverResult (set algebra preserved)
    # ------------------------------------------------------------------

    def glob(
        self,
        pattern: str,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Find files matching *pattern*."""
        return self._run(
            self._async.glob(pattern, candidates=candidates, user_id=user_id)
        )

    def grep(
        self,
        pattern: str,
        *,
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Search file contents for *pattern*."""
        return self._run(
            self._async.grep(
                pattern,
                case_sensitive=case_sensitive,
                max_results=max_results,
                candidates=candidates,
                user_id=user_id,
            )
        )

    def semantic_search(
        self,
        query: str,
        k: int = 15,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Semantic (vector) search."""
        return self._run(
            self._async.semantic_search(query, k, candidates=candidates, user_id=user_id)
        )

    def vector_search(
        self,
        vector: list[float],
        k: int = 15,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Raw vector search."""
        return self._run(
            self._async.vector_search(vector, k, candidates=candidates, user_id=user_id)
        )

    def lexical_search(
        self,
        query: str,
        k: int = 15,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """BM25 lexical search."""
        return self._run(
            self._async.lexical_search(query, k, candidates=candidates, user_id=user_id)
        )

    # ------------------------------------------------------------------
    # Graph — returning GroverResult (set algebra preserved)
    # ------------------------------------------------------------------

    def predecessors(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.predecessors(path, candidates=candidates, user_id=user_id)
        )

    def successors(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.successors(path, candidates=candidates, user_id=user_id)
        )

    def ancestors(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.ancestors(path, candidates=candidates, user_id=user_id)
        )

    def descendants(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.descendants(path, candidates=candidates, user_id=user_id)
        )

    def neighborhood(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        depth: int = 2,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.neighborhood(path, candidates=candidates, depth=depth, user_id=user_id)
        )

    def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.meeting_subgraph(candidates, user_id=user_id)
        )

    def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.min_meeting_subgraph(candidates, user_id=user_id)
        )

    def pagerank(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.pagerank(candidates=candidates, user_id=user_id)
        )

    def betweenness_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.betweenness_centrality(candidates=candidates, user_id=user_id)
        )

    def closeness_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.closeness_centrality(candidates=candidates, user_id=user_id)
        )

    def degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.degree_centrality(candidates=candidates, user_id=user_id)
        )

    def in_degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.in_degree_centrality(candidates=candidates, user_id=user_id)
        )

    def out_degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.out_degree_centrality(candidates=candidates, user_id=user_id)
        )

    def hits(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(
            self._async.hits(candidates=candidates, user_id=user_id)
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        *,
        user_id: str | None = None,
        initial: GroverResult | None = None,
    ) -> GroverResult:
        """Execute a CLI-style query."""
        return self._run(
            self._async.run_query(query, user_id=user_id, initial=initial)
        )

    def cli(
        self,
        query: str,
        *,
        user_id: str | None = None,
        initial: GroverResult | None = None,
    ) -> str:
        """Execute a query and return rendered text."""
        return self._run(
            self._async.cli(query, user_id=user_id, initial=initial)
        )

    def parse_query(self, query: str) -> QueryPlan:
        """Parse a CLI-style query string into a plan (sync)."""
        return self._async.parse_query(query)

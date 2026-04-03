"""GroverAsync and Grover — async and sync clients.

``GroverAsync`` is the async facade for long-running application servers.
Errors return as ``GroverResult(success=False)``.

``Grover`` is the sync facade for data pipelines and backend processes.
It sets ``raise_on_error=True`` so that failed operations raise
``GroverError`` (or a subclass) immediately.

Usage (async)::

    g = GroverAsync()
    await g.add_mount("data", DatabaseFileSystem(engine=engine))
    result = await g.read("/data/hello.txt")

Usage (sync)::

    g = Grover()
    g.add_mount("data", DatabaseFileSystem(engine=engine))
    result = g.read("/data/hello.txt")  # raises NotFoundError if missing
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, TypeVar

from grover.base import GroverFileSystem

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence

    from grover.models import GroverObjectBase
    from grover.query import QueryPlan
    from grover.results import EditOperation, GroverResult, TwoPathOperation

_T = TypeVar("_T")


class GroverAsync(GroverFileSystem):
    """Async facade — storageless router with mount-first API.

    All filesystem operations are inherited from ``GroverFileSystem``.
    This subclass sets ``storage=False`` so it acts as a pure router.
    """

    def __init__(self) -> None:
        super().__init__(storage=False)


# ======================================================================
# Grover — synchronous facade with raise-on-error
# ======================================================================


class Grover:
    """Synchronous facade for data pipelines and backend processes.

    Sets ``raise_on_error=True`` on the internal ``GroverAsync`` so that
    all mounted filesystems raise ``GroverError`` (or subclasses) on
    failure instead of returning ``GroverResult(success=False)``.

    All operations return ``GroverResult``, matching the ``GroverFileSystem``
    async API exactly.

    Usage::

        g = Grover()
        fs = DatabaseFileSystem(engine=engine)
        g.add_mount("data", fs)
        g.write("/data/hello.txt", "content")  # raises on failure
        result = g.read("/data/hello.txt")  # returns GroverResult
        print(result.content)
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

    def add_mount(self, path: str, filesystem: GroverFileSystem) -> None:
        """Mount a filesystem at *path*."""
        self._run(self._async.add_mount(path, filesystem))

    def remove_mount(self, path: str) -> None:
        """Unmount the filesystem at *path*."""
        self._run(self._async.remove_mount(path))

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
    # CRUD — all operations return GroverResult
    # ------------------------------------------------------------------

    def read(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Read file content. Raises ``NotFoundError`` if missing."""
        return self._run(self._async.read(path, candidates=candidates, user_id=user_id))

    def write(
        self,
        path: str | None = None,
        content: str | None = None,
        objects: Sequence[GroverObjectBase] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Write content to *path*. Raises on conflict."""
        return self._run(self._async.write(path, content, objects=objects, overwrite=overwrite, user_id=user_id))

    def edit(
        self,
        path: str | None = None,
        old: str | None = None,
        new: str | None = None,
        edits: list[EditOperation] | None = None,
        candidates: GroverResult | None = None,
        replace_all: bool = False,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Replace *old* with *new* in the file at *path*."""
        return self._run(
            self._async.edit(
                path,
                old,
                new,
                edits=edits,
                candidates=candidates,
                replace_all=replace_all,
                user_id=user_id,
            )
        )

    def delete(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
        cascade: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Delete the object at *path*."""
        return self._run(
            self._async.delete(path, candidates=candidates, permanent=permanent, cascade=cascade, user_id=user_id)
        )

    def stat(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Return metadata for *path*."""
        return self._run(self._async.stat(path, candidates=candidates, user_id=user_id))

    def mkdir(self, path: str, *, user_id: str | None = None) -> GroverResult:
        """Create a directory at *path*."""
        return self._run(self._async.mkdir(path, user_id=user_id))

    def mkconn(
        self,
        source: str,
        target: str,
        connection_type: str,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Create a connection from *source* to *target*."""
        return self._run(self._async.mkconn(source, target, connection_type, user_id=user_id))

    def move(
        self,
        src: str | None = None,
        dest: str | None = None,
        moves: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Move *src* to *dest*."""
        return self._run(self._async.move(src, dest, moves=moves, overwrite=overwrite, user_id=user_id))

    def copy(
        self,
        src: str | None = None,
        dest: str | None = None,
        copies: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Copy *src* to *dest*."""
        return self._run(self._async.copy(src, dest, copies=copies, overwrite=overwrite, user_id=user_id))

    def ls(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """List entries under *path*."""
        return self._run(self._async.ls(path, candidates=candidates, user_id=user_id))

    def tree(
        self,
        path: str,
        max_depth: int | None = None,
        *,
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
        return self._run(self._async.glob(pattern, candidates=candidates, user_id=user_id))

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
        return self._run(self._async.semantic_search(query, k, candidates=candidates, user_id=user_id))

    def vector_search(
        self,
        vector: list[float],
        k: int = 15,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """Raw vector search."""
        return self._run(self._async.vector_search(vector, k, candidates=candidates, user_id=user_id))

    def lexical_search(
        self,
        query: str,
        k: int = 15,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """BM25 lexical search."""
        return self._run(self._async.lexical_search(query, k, candidates=candidates, user_id=user_id))

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
        return self._run(self._async.predecessors(path, candidates=candidates, user_id=user_id))

    def successors(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.successors(path, candidates=candidates, user_id=user_id))

    def ancestors(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.ancestors(path, candidates=candidates, user_id=user_id))

    def descendants(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.descendants(path, candidates=candidates, user_id=user_id))

    def neighborhood(
        self,
        path: str | None = None,
        *,
        candidates: GroverResult | None = None,
        depth: int = 2,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.neighborhood(path, candidates=candidates, depth=depth, user_id=user_id))

    def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.meeting_subgraph(candidates, user_id=user_id))

    def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.min_meeting_subgraph(candidates, user_id=user_id))

    def pagerank(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.pagerank(candidates=candidates, user_id=user_id))

    def betweenness_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.betweenness_centrality(candidates=candidates, user_id=user_id))

    def closeness_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.closeness_centrality(candidates=candidates, user_id=user_id))

    def degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.degree_centrality(candidates=candidates, user_id=user_id))

    def in_degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.in_degree_centrality(candidates=candidates, user_id=user_id))

    def out_degree_centrality(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.out_degree_centrality(candidates=candidates, user_id=user_id))

    def hits(
        self,
        *,
        candidates: GroverResult | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        return self._run(self._async.hits(candidates=candidates, user_id=user_id))

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
        return self._run(self._async.run_query(query, user_id=user_id, initial=initial))

    def cli(
        self,
        query: str,
        *,
        user_id: str | None = None,
        initial: GroverResult | None = None,
    ) -> str:
        """Execute a query and return rendered text."""
        return self._run(self._async.cli(query, user_id=user_id, initial=initial))

    def parse_query(self, query: str) -> QueryPlan:
        """Parse a CLI-style query string into a plan (sync)."""
        return self._async.parse_query(query)

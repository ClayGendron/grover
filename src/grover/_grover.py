"""Grover — thin sync wrapper around GroverAsync with RLock."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from grover._grover_async import GroverAsync
from grover.fs.permissions import Permission

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from grover.fs.types import ReadResult
    from grover.fs.vfs import VFS
    from grover.graph._graph import Graph
    from grover.ref import Ref
    from grover.search._index import SearchResult


class Grover:
    """Synchronous facade backed by :class:`GroverAsync`.

    Presents a mount-first API with a private event loop in a daemon
    thread.  All public methods delegate to ``GroverAsync`` via
    :meth:`_run`.  Thread-safe via ``RLock``.

    Usage::

        g = Grover(embedding_provider=FakeProvider())
        g.mount("/project", LocalFileSystem(workspace_dir="."))
        g.write("/project/hello.py", "print('hi')")
        g.close()
    """

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        embedding_provider: Any = None,
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()

        # Private event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        self._async = GroverAsync(
            data_dir=data_dir,
            embedding_provider=embedding_provider,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Submit *coro* to the private loop and block for the result."""
        with self._lock:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down subsystems, stop the event loop and join the thread."""
        if self._closed:
            return
        self._closed = True

        try:
            self._run(self._async.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Mount / Unmount
    # ------------------------------------------------------------------

    def mount(
        self,
        path: str,
        backend: Any = None,
        *,
        engine: AsyncEngine | None = None,
        session_factory: Callable[..., AsyncSession] | None = None,
        dialect: str = "sqlite",
        file_model: type | None = None,
        file_version_model: type | None = None,
        db_schema: str | None = None,
        mount_type: str | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        hidden: bool = False,
    ) -> None:
        """Mount a backend at *path*."""
        self._run(
            self._async.mount(
                path,
                backend,
                engine=engine,
                session_factory=session_factory,
                dialect=dialect,
                file_model=file_model,
                file_version_model=file_version_model,
                db_schema=db_schema,
                mount_type=mount_type,
                permission=permission,
                label=label,
                hidden=hidden,
            )
        )

    def unmount(self, path: str) -> None:
        """Unmount the backend at *path*."""
        self._run(self._async.unmount(path))

    # ------------------------------------------------------------------
    # Filesystem wrappers (sync)
    # ------------------------------------------------------------------

    def read(self, path: str) -> ReadResult:
        """Read file content at *path*."""
        return self._run(self._async.read(path))

    def write(self, path: str, content: str) -> bool:
        """Write *content* to *path*. Returns ``True`` on success."""
        return self._run(self._async.write(path, content))

    def edit(self, path: str, old: str, new: str) -> bool:
        """Replace *old* with *new* in the file at *path*."""
        return self._run(self._async.edit(path, old, new))

    def delete(self, path: str, permanent: bool = False) -> bool:
        """Delete the file at *path*."""
        return self._run(self._async.delete(path, permanent))

    def list_dir(self, path: str = "/") -> list[dict[str, Any]]:
        """List entries under *path*."""
        return self._run(self._async.list_dir(path))

    def exists(self, path: str) -> bool:
        """Check whether *path* exists."""
        return self._run(self._async.exists(path))

    # ------------------------------------------------------------------
    # Version / Trash / Reconciliation wrappers (sync)
    # ------------------------------------------------------------------

    def list_versions(self, path: str) -> Any:
        return self._run(self._async.list_versions(path))

    def get_version_content(self, path: str, version: int) -> Any:
        return self._run(self._async.get_version_content(path, version))

    def restore_version(self, path: str, version: int) -> Any:
        return self._run(self._async.restore_version(path, version))

    def list_trash(self) -> Any:
        return self._run(self._async.list_trash())

    def restore_from_trash(self, path: str) -> Any:
        return self._run(self._async.restore_from_trash(path))

    def empty_trash(self) -> Any:
        return self._run(self._async.empty_trash())

    def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        return self._run(self._async.reconcile(mount_path))

    # ------------------------------------------------------------------
    # Graph query wrappers (sync — Graph methods are already sync)
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Ref]:
        return self._async.dependents(path)

    def dependencies(self, path: str) -> list[Ref]:
        return self._async.dependencies(path)

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        return self._async.impacts(path, max_depth)

    def path_between(self, source: str, target: str) -> list[Ref] | None:
        return self._async.path_between(source, target)

    def contains(self, path: str) -> list[Ref]:
        return self._async.contains(path)

    # ------------------------------------------------------------------
    # Search wrapper (sync)
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Semantic search over indexed content."""
        return self._run(self._async.search(query, k))

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    def index(self, mount_path: str | None = None) -> dict[str, int]:
        """Walk the filesystem, analyze all files, build graph + search."""
        return self._run(self._async.index(mount_path))

    def save(self) -> None:
        """Persist graph and search index to disk."""
        self._run(self._async.save())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fs(self) -> VFS:
        """The underlying ``VFS`` (for advanced async use)."""
        return self._async.fs

    @property
    def graph(self) -> Graph:
        """The knowledge graph."""
        return self._async.graph

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

    from grover.fs.query_types import GlobQueryResult, GrepQueryResult, SearchQueryResult
    from grover.fs.types import (
        DeleteResult,
        EditResult,
        ListSharesResult,
        MoveResult,
        ReadResult,
        ShareResult,
        TreeResult,
        WriteResult,
    )
    from grover.fs.vfs import VFS
    from grover.graph.protocols import GraphStore
    from grover.graph.types import SubgraphResult
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase
    from grover.ref import Ref


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
        vector_store: Any = None,
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()

        # Private event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._async = GroverAsync(
            data_dir=data_dir,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
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
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
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
                file_chunk_model=file_chunk_model,
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

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        user_id: str | None = None,
    ) -> ReadResult:
        """Read file content at *path*."""
        return self._run(self._async.read(path, offset=offset, limit=limit, user_id=user_id))

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        """Write *content* to *path*."""
        return self._run(self._async.write(path, content, overwrite=overwrite, user_id=user_id))

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> EditResult:
        """Replace *old* with *new* in the file at *path*."""
        return self._run(self._async.edit(path, old, new, replace_all=replace_all, user_id=user_id))

    def delete(
        self, path: str, permanent: bool = False, *, user_id: str | None = None
    ) -> DeleteResult:
        """Delete the file at *path*."""
        return self._run(self._async.delete(path, permanent, user_id=user_id))

    def list_dir(self, path: str = "/", *, user_id: str | None = None) -> list[dict[str, Any]]:
        """List entries under *path*."""
        return self._run(self._async.list_dir(path, user_id=user_id))

    def exists(self, path: str, *, user_id: str | None = None) -> bool:
        """Check whether *path* exists."""
        return self._run(self._async.exists(path, user_id=user_id))

    def move(
        self, src: str, dest: str, *, user_id: str | None = None, follow: bool = False
    ) -> MoveResult:
        """Move a file from *src* to *dest*."""
        return self._run(self._async.move(src, dest, user_id=user_id, follow=follow))

    def copy(self, src: str, dest: str, *, user_id: str | None = None) -> WriteResult:
        """Copy a file from *src* to *dest*."""
        return self._run(self._async.copy(src, dest, user_id=user_id))

    # ------------------------------------------------------------------
    # Search / Query wrappers (sync)
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: str = "/", *, user_id: str | None = None) -> GlobQueryResult:
        """Find files matching a glob *pattern* under *path*."""
        return self._run(self._async.glob(pattern, path, user_id=user_id))

    def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        user_id: str | None = None,
    ) -> GrepQueryResult:
        """Search file contents for *pattern* under *path*."""
        return self._run(
            self._async.grep(
                pattern,
                path,
                glob_filter=glob_filter,
                case_sensitive=case_sensitive,
                fixed_string=fixed_string,
                invert=invert,
                word_match=word_match,
                context_lines=context_lines,
                max_results=max_results,
                max_results_per_file=max_results_per_file,
                count_only=count_only,
                files_only=files_only,
                user_id=user_id,
            )
        )

    def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        """List all entries under *path* recursively."""
        return self._run(self._async.tree(path, max_depth=max_depth, user_id=user_id))

    # ------------------------------------------------------------------
    # Version / Trash / Reconciliation wrappers (sync)
    # ------------------------------------------------------------------

    def list_versions(self, path: str, *, user_id: str | None = None) -> Any:
        return self._run(self._async.list_versions(path, user_id=user_id))

    def get_version_content(self, path: str, version: int, *, user_id: str | None = None) -> Any:
        return self._run(self._async.get_version_content(path, version, user_id=user_id))

    def restore_version(self, path: str, version: int, *, user_id: str | None = None) -> Any:
        return self._run(self._async.restore_version(path, version, user_id=user_id))

    def list_trash(self, *, user_id: str | None = None) -> Any:
        return self._run(self._async.list_trash(user_id=user_id))

    def restore_from_trash(self, path: str, *, user_id: str | None = None) -> Any:
        return self._run(self._async.restore_from_trash(path, user_id=user_id))

    def empty_trash(self, *, user_id: str | None = None) -> Any:
        return self._run(self._async.empty_trash(user_id=user_id))

    def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        return self._run(self._async.reconcile(mount_path))

    # ------------------------------------------------------------------
    # Share wrappers (sync)
    # ------------------------------------------------------------------

    def share(
        self,
        path: str,
        grantee_id: str,
        permission: str = "read",
        *,
        user_id: str,
        expires_at: Any = None,
    ) -> ShareResult:
        """Share a file or directory with another user."""
        return self._run(
            self._async.share(
                path,
                grantee_id,
                permission,
                user_id=user_id,
                expires_at=expires_at,
            )
        )

    def unshare(self, path: str, grantee_id: str, *, user_id: str) -> ShareResult:
        """Remove a share for a file or directory."""
        return self._run(self._async.unshare(path, grantee_id, user_id=user_id))

    def list_shares(self, path: str, *, user_id: str) -> ListSharesResult:
        """List all shares on a given path."""
        return self._run(self._async.list_shares(path, user_id=user_id))

    def list_shared_with_me(self, *, user_id: str) -> ListSharesResult:
        """List all files shared with the current user."""
        return self._run(self._async.list_shared_with_me(user_id=user_id))

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
    # Graph algorithm wrappers (capability-checked)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        personalization: dict[str, float] | None = None,
        path: str | None = None,
    ) -> dict[str, float]:
        """Run PageRank on the knowledge graph."""
        return self._async.pagerank(personalization=personalization, path=path)

    def ancestors(self, path: str) -> set[str]:
        """All transitive predecessors of *path* in the knowledge graph."""
        return self._async.ancestors(path)

    def descendants(self, path: str) -> set[str]:
        """All transitive successors of *path* in the knowledge graph."""
        return self._async.descendants(path)

    def meeting_subgraph(
        self,
        paths: list[str],
        *,
        max_size: int = 50,
    ) -> SubgraphResult:
        """Extract the subgraph connecting *paths* via shortest paths."""
        return self._async.meeting_subgraph(paths, max_size=max_size)

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> SubgraphResult:
        """Extract the neighborhood subgraph around *path*."""
        return self._async.neighborhood(
            path,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
        )

    def find_nodes(self, *, path: str | None = None, **attrs: Any) -> list[str]:
        """Find graph nodes matching all attribute predicates."""
        return self._async.find_nodes(path=path, **attrs)

    # ------------------------------------------------------------------
    # Search wrapper (sync)
    # ------------------------------------------------------------------

    def search(
        self, query: str, k: int = 10, *, path: str = "/", user_id: str | None = None
    ) -> SearchQueryResult:
        """Semantic search over indexed content."""
        return self._run(self._async.search(query, k, path=path, user_id=user_id))

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

    def get_graph(self, path: str | None = None) -> GraphStore:
        """Return the graph for the mount owning *path*, or the first available."""
        return self._async.get_graph(path)

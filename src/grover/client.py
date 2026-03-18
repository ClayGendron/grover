"""GroverAsync and Grover — async and sync clients with mount-first API."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, TypeVar

from grover.analyzers import AnalyzerRegistry
from grover.api.context import GroverContext
from grover.api.file_ops import FileOpsMixin
from grover.api.graph_ops import GraphOpsMixin
from grover.api.indexing import IndexMixin
from grover.api.mounting import MountMixin
from grover.api.search_ops import SearchOpsMixin
from grover.api.sharing import ShareMixin
from grover.mount import MountRegistry
from grover.permissions import Permission
from grover.worker import BackgroundWorker, IndexingMode

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from datetime import datetime

    from grover.backends.protocol import GroverFileSystem
    from grover.models.config import EngineConfig, SessionConfig
    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.file import FileModelBase
    from grover.models.internal.results import FileOperationResult, FileSearchResult, FileSearchSet, GroverResult
    from grover.mount import Mount
    from grover.providers.embedding.protocol import EmbeddingProvider
    from grover.providers.graph.protocol import GraphProvider
    from grover.providers.search.protocol import SearchProvider

_T = TypeVar("_T")


class GroverAsync(
    MountMixin,
    FileOpsMixin,
    SearchOpsMixin,
    GraphOpsMixin,
    ShareMixin,
    IndexMixin,
):
    """Async facade wiring filesystem, graph, analyzers, worker, and search.

    Mount-first API: create an instance, then add mounts.

    EngineConfig (Grover manages the engine)::

        g = GroverAsync()
        await g.add_mount(
            "data", engine_config=EngineConfig(url="postgresql+asyncpg://...")
        )

    SessionConfig (app manages the engine)::

        g = GroverAsync()
        await g.add_mount("data", session_config=SessionConfig(session_factory=sf))

    Direct filesystem::

        g = GroverAsync()
        await g.add_mount("app", filesystem=LocalFileSystem(workspace_dir="."))
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


class Grover:
    """Synchronous facade backed by :class:`GroverAsync`.

    Presents a mount-first API with a private event loop in a daemon
    thread.  All public methods delegate to ``GroverAsync`` via
    :meth:`_run`.  Thread-safe via ``RLock``.

    Usage::

        g = Grover()
        g.add_mount("project", filesystem=LocalFileSystem(workspace_dir="."))
        g.write("/project/hello.py", "print('hi')")
        g.close()
    """

    def __init__(
        self,
        *,
        indexing_mode: IndexingMode = IndexingMode.BACKGROUND,
        debounce_delay: float = 0.1,
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()

        # Private event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._async = GroverAsync(
            indexing_mode=indexing_mode,
            debounce_delay=debounce_delay,
        )

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

    def add_mount(
        self,
        name: str | None = None,
        *,
        mount: Mount | None = None,
        filesystem: GroverFileSystem | None = None,
        engine_config: EngineConfig | None = None,
        session_config: SessionConfig | None = None,
        permission: Permission = Permission.READ_WRITE,
        embedding_provider: EmbeddingProvider | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        """Add a mount with *name* and *filesystem*."""
        self._run(
            self._async.add_mount(
                name,
                mount=mount,
                filesystem=filesystem,
                engine_config=engine_config,
                session_config=session_config,
                permission=permission,
                embedding_provider=embedding_provider,
                search_provider=search_provider,
            )
        )

    def unmount(self, name: str) -> None:
        """Unmount the backend with the given *name*."""
        self._run(self._async.unmount(name))

    # ------------------------------------------------------------------
    # Filesystem wrappers (sync)
    # ------------------------------------------------------------------

    def read(
        self,
        path: str,
        *,
        user_id: str | None = None,
    ) -> FileOperationResult:
        """Read file content at *path*."""
        return self._run(self._async.read(path, user_id=user_id))

    def read_files(self, paths: list[str], *, user_id: str | None = None) -> GroverResult:
        """Batch read files."""
        return self._run(self._async.read_files(paths, user_id=user_id))

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> GroverResult:
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
    ) -> FileOperationResult:
        """Replace *old* with *new* in the file at *path*."""
        return self._run(self._async.edit(path, old, new, replace_all=replace_all, user_id=user_id))

    def delete(self, path: str, permanent: bool = False, *, user_id: str | None = None) -> GroverResult:
        """Delete the file at *path*."""
        return self._run(self._async.delete(path, permanent, user_id=user_id))

    def list_dir(self, path: str = "/", *, user_id: str | None = None) -> GroverResult:
        """List entries under *path*."""
        return self._run(self._async.list_dir(path, user_id=user_id))

    def exists(self, path: str, *, user_id: str | None = None) -> GroverResult:
        """Check whether *path* exists."""
        return self._run(self._async.exists(path, user_id=user_id))

    def move(self, src: str, dest: str, *, user_id: str | None = None) -> GroverResult:
        """Move a file from *src* to *dest*."""
        return self._run(self._async.move(src, dest, user_id=user_id))

    def move_files(self, pairs: list[tuple[str, str]], *, user_id: str | None = None) -> GroverResult:
        """Batch move files."""
        return self._run(self._async.move_files(pairs, user_id=user_id))

    def copy(self, src: str, dest: str, *, user_id: str | None = None) -> GroverResult:
        """Copy a file from *src* to *dest*."""
        return self._run(self._async.copy(src, dest, user_id=user_id))

    def copy_files(self, pairs: list[tuple[str, str]], *, user_id: str | None = None) -> GroverResult:
        """Batch copy files."""
        return self._run(self._async.copy_files(pairs, user_id=user_id))

    # ------------------------------------------------------------------
    # Search / Query wrappers (sync)
    # ------------------------------------------------------------------

    def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Find files matching a glob *pattern* under *path*."""
        return self._run(self._async.glob(pattern, path, candidates=candidates, user_id=user_id))

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
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
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
                candidates=candidates,
                user_id=user_id,
            )
        )

    def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        """List all entries under *path* recursively."""
        return self._run(self._async.tree(path, max_depth=max_depth, user_id=user_id))

    # ------------------------------------------------------------------
    # Version / Trash / Reconciliation wrappers (sync)
    # ------------------------------------------------------------------

    def list_versions(self, path: str, *, user_id: str | None = None) -> FileSearchResult:
        return self._run(self._async.list_versions(path, user_id=user_id))

    def read_version(self, path: str, version: int, *, user_id: str | None = None) -> FileOperationResult:
        return self._run(self._async.read_version(path, version, user_id=user_id))

    def diff_versions(
        self, path: str, version_a: int, version_b: int, *, user_id: str | None = None
    ) -> FileOperationResult:
        return self._run(self._async.diff_versions(path, version_a, version_b, user_id=user_id))

    def restore_version(self, path: str, version: int, *, user_id: str | None = None) -> FileOperationResult:
        return self._run(self._async.restore_version(path, version, user_id=user_id))

    def list_trash(self, *, user_id: str | None = None) -> FileSearchResult:
        return self._run(self._async.list_trash(user_id=user_id))

    def restore_from_trash(self, path: str, *, user_id: str | None = None) -> FileOperationResult:
        return self._run(self._async.restore_from_trash(path, user_id=user_id))

    def empty_trash(self, *, user_id: str | None = None) -> FileOperationResult:
        return self._run(self._async.empty_trash(user_id=user_id))

    def reconcile(self, mount_path: str | None = None) -> FileOperationResult:
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
        expires_at: datetime | None = None,
    ) -> FileOperationResult:
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

    def unshare(self, path: str, grantee_id: str, *, user_id: str) -> FileOperationResult:
        """Remove a share for a file or directory."""
        return self._run(self._async.unshare(path, grantee_id, user_id=user_id))

    def list_shares(self, path: str, *, user_id: str) -> FileSearchResult:
        """List all shares on a given path."""
        return self._run(self._async.list_shares(path, user_id=user_id))

    def list_shared_with_me(self, *, user_id: str) -> FileSearchResult:
        """List all files shared with the current user."""
        return self._run(self._async.list_shared_with_me(user_id=user_id))

    # ------------------------------------------------------------------
    # Connection operations (sync)
    # ------------------------------------------------------------------

    def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
    ) -> FileOperationResult:
        return self._run(
            self._async.add_connection(
                source_path,
                target_path,
                connection_type,
                weight=weight,
            )
        )

    def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
    ) -> FileOperationResult:
        return self._run(
            self._async.delete_connection(
                source_path,
                target_path,
                connection_type=connection_type,
            )
        )

    # ------------------------------------------------------------------
    # Chunk write wrappers (sync)
    # ------------------------------------------------------------------

    def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch write (upsert) chunks."""
        return self._run(self._async.write_chunks(chunks, user_id=user_id))

    # ------------------------------------------------------------------
    # File write from model wrappers (sync)
    # ------------------------------------------------------------------

    def write_files(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch write files from model instances."""
        return self._run(self._async.write_files(files, overwrite=overwrite, user_id=user_id))

    # ------------------------------------------------------------------
    # Graph traversal wrappers (sync)
    # ------------------------------------------------------------------

    def predecessors(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.predecessors(candidates))

    def successors(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.successors(candidates))

    def ancestors(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.ancestors(candidates))

    def descendants(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.descendants(candidates))

    # ------------------------------------------------------------------
    # Graph subgraph wrappers (sync)
    # ------------------------------------------------------------------

    def min_meeting_subgraph(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.min_meeting_subgraph(candidates))

    def ego_graph(self, candidates: FileSearchSet, *, max_depth: int = 2) -> FileSearchResult:
        return self._run(self._async.ego_graph(candidates, max_depth=max_depth))

    # ------------------------------------------------------------------
    # Graph centrality wrappers (sync)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        candidates: FileSearchSet,
        *,
        personalization: dict[str, float] | None = None,
    ) -> FileSearchResult:
        return self._run(self._async.pagerank(candidates, personalization=personalization))

    def betweenness_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.betweenness_centrality(candidates))

    def closeness_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.closeness_centrality(candidates))

    def katz_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.katz_centrality(candidates))

    def degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.degree_centrality(candidates))

    def in_degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.in_degree_centrality(candidates))

    def out_degree_centrality(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.out_degree_centrality(candidates))

    def hits(self, candidates: FileSearchSet) -> FileSearchResult:
        return self._run(self._async.hits(candidates))

    # ------------------------------------------------------------------
    # Search wrappers (sync)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Semantic (vector) search over indexed content."""
        return self._run(self._async.vector_search(query, k, path=path, candidates=candidates, user_id=user_id))

    def lexical_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """BM25/full-text search over indexed content."""
        return self._run(self._async.lexical_search(query, k, path=path, candidates=candidates, user_id=user_id))

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        *,
        alpha: float = 0.5,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Hybrid search combining vector and lexical results."""
        return self._run(
            self._async.hybrid_search(query, k, alpha=alpha, path=path, candidates=candidates, user_id=user_id)
        )

    def search(
        self,
        query: str,
        *,
        path: str = "/",
        glob: str | None = None,
        grep: str | None = None,
        k: int = 10,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Composable search pipeline: optional glob/grep → vector search."""
        return self._run(
            self._async.search(query, path=path, glob=glob, grep=grep, k=k, candidates=candidates, user_id=user_id)
        )

    # ------------------------------------------------------------------
    # Index and persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for all pending background indexing to complete."""
        self._run(self._async.flush())

    def index(self, mount_path: str | None = None) -> dict[str, int]:
        """Walk the filesystem, analyze all files, build graph + search."""
        return self._run(self._async.index(mount_path))

    def save(self) -> None:
        """Drain pending background work."""
        self._run(self._async.save())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def get_graph(self, path: str | None = None) -> GraphProvider:
        """Return the graph for the mount owning *path*, or the first available."""
        return self._async.get_graph(path)

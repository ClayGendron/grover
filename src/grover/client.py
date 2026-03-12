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
    from collections.abc import Callable, Coroutine
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from grover.backends.protocol import GroverFileSystem
    from grover.models.chunk import FileChunkBase
    from grover.models.file import FileBase
    from grover.models.version import FileVersionBase
    from grover.mount import Mount
    from grover.providers.embedding.protocol import EmbeddingProvider
    from grover.providers.graph.protocol import GraphProvider
    from grover.providers.search.protocol import SearchProvider
    from grover.results import (
        AncestorsResult,
        BatchChunkResult,
        BatchWriteResult,
        BetweennessResult,
        ChunkResult,
        ClosenessResult,
        CommonNeighborsResult,
        ConnectionResult,
        DegreeResult,
        DeleteResult,
        DescendantsResult,
        DiffVersionsResult,
        EditResult,
        EgoGraphResult,
        ExistsResult,
        FileInfoResult,
        FileSearchResult,
        GetVersionContentResult,
        GlobResult,
        GrepResult,
        HarmonicResult,
        HasPathResult,
        HitsResult,
        KatzResult,
        LexicalSearchResult,
        ListDirResult,
        MeetingSubgraphResult,
        MoveResult,
        PageRankResult,
        PredecessorsResult,
        ReadResult,
        ReconcileResult,
        RestoreResult,
        ShareResult,
        ShareSearchResult,
        ShortestPathResult,
        SubgraphSearchResult,
        SuccessorsResult,
        TrashResult,
        TreeResult,
        VectorSearchResult,
        VersionResult,
        WriteResult,
    )

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


class Grover:
    """Synchronous facade backed by :class:`GroverAsync`.

    Presents a mount-first API with a private event loop in a daemon
    thread.  All public methods delegate to ``GroverAsync`` via
    :meth:`_run`.  Thread-safe via ``RLock``.

    Usage::

        g = Grover()
        g.add_mount(
            "/project", LocalFileSystem(workspace_dir="."), embedding_provider=embed
        )
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
        path_or_mount: str | Mount | None = None,
        filesystem: GroverFileSystem | None = None,
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
        embedding_provider: EmbeddingProvider | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        """Add a mount at *path* with *filesystem*."""
        self._run(
            self._async.add_mount(
                path_or_mount,
                filesystem,
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
                embedding_provider=embedding_provider,
                search_provider=search_provider,
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

    def list_dir(self, path: str = "/", *, user_id: str | None = None) -> ListDirResult:
        """List entries under *path*."""
        return self._run(self._async.list_dir(path, user_id=user_id))

    def exists(self, path: str, *, user_id: str | None = None) -> ExistsResult:
        """Check whether *path* exists."""
        return self._run(self._async.exists(path, user_id=user_id))

    def get_info(self, path: str, *, user_id: str | None = None) -> FileInfoResult:
        """Return metadata for *path*."""
        return self._run(self._async.get_info(path, user_id=user_id))

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

    def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> GlobResult:
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
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> GrepResult:
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
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        """List all entries under *path* recursively."""
        return self._run(self._async.tree(path, max_depth=max_depth, user_id=user_id))

    # ------------------------------------------------------------------
    # Version / Trash / Reconciliation wrappers (sync)
    # ------------------------------------------------------------------

    def list_versions(self, path: str, *, user_id: str | None = None) -> VersionResult:
        return self._run(self._async.list_versions(path, user_id=user_id))

    def read_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        return self._run(self._async.read_version(path, version, user_id=user_id))

    def diff_versions(
        self, path: str, version_a: int, version_b: int, *, user_id: str | None = None
    ) -> DiffVersionsResult:
        return self._run(self._async.diff_versions(path, version_a, version_b, user_id=user_id))

    def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        return self._run(self._async.restore_version(path, version, user_id=user_id))

    def list_trash(self, *, user_id: str | None = None) -> TrashResult:
        return self._run(self._async.list_trash(user_id=user_id))

    def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        return self._run(self._async.restore_from_trash(path, user_id=user_id))

    def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        return self._run(self._async.empty_trash(user_id=user_id))

    def reconcile(self, mount_path: str | None = None) -> ReconcileResult:
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

    def list_shares(self, path: str, *, user_id: str) -> ShareSearchResult:
        """List all shares on a given path."""
        return self._run(self._async.list_shares(path, user_id=user_id))

    def list_shared_with_me(self, *, user_id: str) -> ShareSearchResult:
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
    ) -> ConnectionResult:
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
    ) -> ConnectionResult:
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

    def write_chunk(
        self,
        chunk: FileChunkBase,
        *,
        user_id: str | None = None,
    ) -> ChunkResult:
        """Write (upsert) a single chunk."""
        return self._run(self._async.write_chunk(chunk, user_id=user_id))

    def write_chunks(
        self,
        chunks: list[FileChunkBase],
        *,
        user_id: str | None = None,
    ) -> BatchChunkResult:
        """Batch write (upsert) chunks."""
        return self._run(self._async.write_chunks(chunks, user_id=user_id))

    # ------------------------------------------------------------------
    # File write from model wrappers (sync)
    # ------------------------------------------------------------------

    def write_file(
        self,
        file: FileBase,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        """Write a single file from a model instance."""
        return self._run(self._async.write_file(file, overwrite=overwrite, user_id=user_id))

    def write_files(
        self,
        files: list[FileBase],
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> BatchWriteResult:
        """Batch write files from model instances."""
        return self._run(self._async.write_files(files, overwrite=overwrite, user_id=user_id))

    # ------------------------------------------------------------------
    # Graph traversal wrappers (sync)
    # ------------------------------------------------------------------

    def predecessors(self, path: str) -> PredecessorsResult:
        return self._run(self._async.predecessors(path))

    def successors(self, path: str) -> SuccessorsResult:
        return self._run(self._async.successors(path))

    def ancestors(self, path: str) -> AncestorsResult:
        return self._run(self._async.ancestors(path))

    def descendants(self, path: str) -> DescendantsResult:
        return self._run(self._async.descendants(path))

    def shortest_path(self, source: str, target: str) -> ShortestPathResult:
        return self._run(self._async.shortest_path(source, target))

    def has_path(self, source: str, target: str) -> HasPathResult:
        return self._run(self._async.has_path(source, target))

    # ------------------------------------------------------------------
    # Graph subgraph wrappers (sync)
    # ------------------------------------------------------------------

    def subgraph(
        self,
        candidates: FileSearchResult,
        *,
        path: str | None = None,
    ) -> SubgraphSearchResult:
        return self._run(self._async.subgraph(candidates, path=path))

    def min_meeting_subgraph(
        self,
        candidates: FileSearchResult,
        *,
        max_size: int = 50,
    ) -> MeetingSubgraphResult:
        return self._run(self._async.min_meeting_subgraph(candidates, max_size=max_size))

    def ego_graph(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> EgoGraphResult:
        return self._run(
            self._async.ego_graph(
                path,
                max_depth=max_depth,
                direction=direction,
                edge_types=edge_types,
            )
        )

    # ------------------------------------------------------------------
    # Graph centrality wrappers (sync)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
        personalization: dict[str, float] | None = None,
    ) -> PageRankResult:
        return self._run(
            self._async.pagerank(path=path, candidates=candidates, personalization=personalization)
        )

    def betweenness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> BetweennessResult:
        return self._run(self._async.betweenness_centrality(path=path, candidates=candidates))

    def closeness_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> ClosenessResult:
        return self._run(self._async.closeness_centrality(path=path, candidates=candidates))

    def harmonic_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> HarmonicResult:
        return self._run(self._async.harmonic_centrality(path=path, candidates=candidates))

    def katz_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> KatzResult:
        return self._run(self._async.katz_centrality(path=path, candidates=candidates))

    def degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        return self._run(self._async.degree_centrality(path=path, candidates=candidates))

    def in_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        return self._run(self._async.in_degree_centrality(path=path, candidates=candidates))

    def out_degree_centrality(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        return self._run(self._async.out_degree_centrality(path=path, candidates=candidates))

    def hits(
        self,
        *,
        path: str | None = None,
        candidates: FileSearchResult | None = None,
    ) -> HitsResult:
        return self._run(self._async.hits(path=path, candidates=candidates))

    # ------------------------------------------------------------------
    # Other graph operations (sync)
    # ------------------------------------------------------------------

    def common_neighbors(
        self,
        path1: str,
        path2: str,
        *,
        path: str | None = None,
    ) -> CommonNeighborsResult:
        return self._run(self._async.common_neighbors(path1, path2, path=path))

    # ------------------------------------------------------------------
    # Search wrappers (sync)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> VectorSearchResult:
        """Semantic (vector) search over indexed content."""
        return self._run(
            self._async.vector_search(query, k, path=path, candidates=candidates, user_id=user_id)
        )

    def lexical_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> LexicalSearchResult:
        """BM25/full-text search over indexed content."""
        return self._run(
            self._async.lexical_search(query, k, path=path, candidates=candidates, user_id=user_id)
        )

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        *,
        alpha: float = 0.5,
        path: str = "/",
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Hybrid search combining vector and lexical results."""
        return self._run(
            self._async.hybrid_search(
                query, k, alpha=alpha, path=path, candidates=candidates, user_id=user_id
            )
        )

    def search(
        self,
        query: str,
        *,
        path: str = "/",
        glob: str | None = None,
        grep: str | None = None,
        k: int = 10,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Composable search pipeline: optional glob/grep → vector search."""
        return self._run(
            self._async.search(query, path=path, glob=glob, grep=grep, k=k, user_id=user_id)
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

"""GroverFileSystem — concrete async base class with mount routing.

The base class owns mount routing, session management, path rebasing, and
``_grover`` binding.  The filesystem object itself owns ``/`` — mounting at
``"/"`` is illegal.

Public methods are routers.  They resolve the terminal filesystem via
longest-prefix mount matching, delegate to ``_*_impl`` methods for actual
storage work, then rebase paths and stamp ``_grover`` before returning.

Subclasses override ``_*_impl`` for their storage backend:
- ``DatabaseFileSystem`` — SQL via ``GroverObject``
- ``LocalFileSystem`` — disk bytes + SQL metadata
- ``AsyncGrover`` — no storage, mount-only router
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from grover.paths import normalize_path
from grover.results import Candidate, EditOperation, GroverResult, TwoPathOperation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GroverFileSystem:
    """Async base class for all Grover filesystems."""

    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        if engine is None and session_factory is None:
            msg = "GroverFileSystem requires either engine or session_factory"
            raise ValueError(msg)
        self._engine = engine
        if session_factory is not None:
            self._session_factory = session_factory
        else:
            self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
        self._mounts: dict[str, GroverFileSystem] = {}
        self._sorted_mount_paths: list[str] = []
        # TODO: load existing top-level directory paths from DB at init time
        # and store as a set. add_mount() checks against this set to prevent
        # mounting over paths that already have data (shadow prevention).
        # AsyncGrover (no storage) skips this.

    # -------------------------------------------------------------------
    # Mount management
    # -------------------------------------------------------------------

    def _rebuild_sorted_mounts(self) -> None:
        """Rebuild the pre-sorted mount path list (longest first)."""
        self._sorted_mount_paths: list[str] = sorted(self._mounts.keys(), key=len, reverse=True)

    async def add_mount(self, path: str, filesystem: GroverFileSystem) -> None:
        """Mount a child filesystem at *path*.

        Validation:
        - Path must be absolute (starts with ``/``)
        - Path must not be ``"/"`` (the root is owned by the filesystem itself)
        - Path must already be normalized
        - Exact path collision is forbidden
        - Nested mounts are allowed (``/data`` and ``/data/archive``)
        """
        normalized = normalize_path(path)
        if path != normalized:
            msg = f"Mount path must be normalized: {path!r} (did you mean {normalized!r}?)"
            raise ValueError(msg)
        if path == "/":
            msg = "Cannot mount at '/': the filesystem owns its own root"
            raise ValueError(msg)
        if not path.startswith("/"):
            msg = f"Mount path must be absolute: {path}"
            raise ValueError(msg)
        if path in self._mounts:
            msg = f"Mount already exists at: {path}"
            raise ValueError(msg)
        self._mounts[path] = filesystem
        self._rebuild_sorted_mounts()

    async def remove_mount(self, path: str) -> None:
        """Unmount the filesystem at *path*."""
        if path not in self._mounts:
            normalized = normalize_path(path)
            hint = f" (did you mean {normalized!r}?)" if normalized in self._mounts else ""
            msg = f"No mount at: {path!r}{hint}"
            raise ValueError(msg)
        del self._mounts[path]
        self._rebuild_sorted_mounts()

    # -------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------

    def _match_mount(self, path: str) -> tuple[str, GroverFileSystem] | None:
        """Longest-prefix mount match for *path*."""
        for mount_path in self._sorted_mount_paths:
            if path == mount_path or path.startswith(mount_path + "/"):
                return mount_path, self._mounts[mount_path]
        return None

    def _resolve_terminal(self, path: str) -> tuple[GroverFileSystem, str, str]:
        """Walk mount chain to find the terminal filesystem.

        Returns ``(terminal_fs, relative_path, prefix)`` where:
        - *terminal_fs* is the filesystem that owns the path
        - *relative_path* is the path within that filesystem
        - *prefix* is the accumulated mount path for rebasing results
        """
        fs = self
        prefix = ""
        rel = normalize_path(path)
        while True:
            matched = fs._match_mount(rel)
            if matched is None:
                break
            mount_path, mount_fs = matched

            fs = mount_fs
            prefix = prefix + mount_path
            rel = rel[len(mount_path) :] or "/"
        return fs, rel, prefix

    # -------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------

    @asynccontextmanager
    async def _use_session(self) -> AsyncIterator[AsyncSession]:
        """Create a session from this filesystem's factory.

        Commits on success, rolls back on error.
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # -------------------------------------------------------------------
    # Result helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _rebase_result(result: GroverResult, prefix: str) -> GroverResult:
        """Restore absolute paths on candidates after a mount delegation."""
        if not prefix:
            return result
        rebased = []
        for c in result.candidates:
            new_path = prefix + c.path if c.path != "/" else prefix
            rebased.append(c.model_copy(update={"path": new_path}))
        return result._with_candidates(rebased)

    def _exclude_mounted_paths(self, result: GroverResult) -> GroverResult:
        """Remove self-storage candidates that fall under a mount prefix.

        Prevents shadow results when self has storage AND child mounts —
        the mount owns those paths, not self.
        """
        prefixes = list(self._mounts.keys())
        if not prefixes:
            return result
        filtered = [
            c for c in result.candidates if not any(c.path == p or c.path.startswith(p + "/") for p in prefixes)
        ]
        return result._with_candidates(filtered)

    def _error(self, message: str) -> GroverResult:
        """Create a failed result with ``_grover`` already stamped."""
        result = GroverResult(success=False, errors=[message])
        result._grover = self
        return result

    def _stamp(self, result: GroverResult) -> GroverResult:
        """Stamp ``_grover`` on a result and return it."""
        result._grover = self
        return result

    @staticmethod
    def _require_same_mount(
        resolved: list[tuple[GroverFileSystem, str, str]],
        label: str,
    ) -> tuple[GroverFileSystem, str] | str:
        """Validate all resolved paths share the same filesystem and prefix.

        Returns ``(filesystem, prefix)`` on success, or an error message string.
        """
        fs, _, prefix = resolved[0]
        for r_fs, _, r_prefix in resolved[1:]:
            if r_fs is not fs or r_prefix != prefix:
                return f"All {label} must resolve to the same mount"
        return fs, prefix

    @staticmethod
    def _merge_results(results: list[GroverResult]) -> GroverResult:
        """Merge multiple results — any failure = overall failure.

        ``|`` already propagates ``success=False`` and concatenates
        ``errors``, so the merged result naturally reflects  all failures
        while preserving all successful candidates.
        """
        if not results:
            return GroverResult(success=True, candidates=[])
        merged = results[0]
        for r in results[1:]:
            merged = merged | r
        return merged

    # -------------------------------------------------------------------
    # Candidate dispatch
    # -------------------------------------------------------------------

    def _group_by_terminal(
        self,
        candidates: GroverResult,
    ) -> list[tuple[GroverFileSystem, str, GroverResult]]:
        """Group candidates by terminal filesystem, rebasing paths.

        Returns ``[(filesystem, prefix, rebased_candidates)]`` where each
        ``GroverResult`` contains candidates with paths relative to that
        terminal filesystem.
        """
        groups: dict[tuple[int, str], tuple[GroverFileSystem, list[Candidate]]] = {}
        for c in candidates:
            fs, rel, prefix = self._resolve_terminal(c.path)
            key = (id(fs), prefix)
            if key not in groups:
                groups[key] = (fs, [])
            groups[key][1].append(c.model_copy(update={"path": rel}))
        return [(fs, pfx, GroverResult(candidates=cands)) for ((_id, pfx), (fs, cands)) in groups.items()]

    async def _dispatch_candidates(
        self,
        op: str,
        candidates: GroverResult,
        **kwargs: object,
    ) -> GroverResult:
        """Route a candidate-based operation to terminal filesystems in parallel.

        Groups candidates by terminal filesystem, calls ``_{op}_impl``
        with rebased candidates on each concurrently, then rebases and
        merges results.
        """
        groups = self._group_by_terminal(candidates)

        async def _run_group(
            fs: GroverFileSystem,
            prefix: str,
            group_cands: GroverResult,
        ) -> GroverResult:
            async with fs._use_session() as s:
                impl = getattr(fs, f"_{op}_impl")
                r = await impl(candidates=group_cands, session=s, **kwargs)
            return self._rebase_result(r, prefix)

        results = await asyncio.gather(
            *(_run_group(fs, pfx, gc) for fs, pfx, gc in groups),
        )
        merged = self._merge_results(list(results))
        merged._grover = self
        return merged

    async def _route_single(
        self,
        op: str,
        path: str | None,
        candidates: GroverResult | None,
        **kwargs: object,
    ) -> GroverResult:
        """Route a single-path or candidate-based operation.

        With candidates: group by filesystem, dispatch in parallel.
        With path: resolve one terminal, call impl once.
        """
        if candidates:
            return await self._dispatch_candidates(op, candidates, **kwargs)

        assert path is not None
        fs, rel, prefix = self._resolve_terminal(path)

        async with fs._use_session() as s:
            result = await getattr(fs, f"_{op}_impl")(rel, session=s, **kwargs)

        result = self._rebase_result(result, prefix)
        result._grover = self
        return result

    async def _route_two_path(
        self,
        op: str,
        ops: list[TwoPathOperation],
        *,
        overwrite: bool = True,
    ) -> GroverResult:
        """Route a batch of two-path operations (move/copy).

        All sources must resolve to the same mount, and all destinations
        must resolve to the same mount.

        Same-mount: call impl with the full batch.
        Cross-mount: batch-read from source, batch-write to dest.
        For move, also soft-deletes the sources on success.

        Note: cross-mount operations are not atomic — writes commit before
        deletes. A crash between phases may leave data on both filesystems.
        """
        if not ops:
            return GroverResult(success=True, candidates=[])

        src_resolved = [self._resolve_terminal(o.src) for o in ops]
        dst_resolved = [self._resolve_terminal(o.dest) for o in ops]

        src_check = self._require_same_mount(src_resolved, f"{op} sources")
        if isinstance(src_check, str):
            return self._error(src_check)
        src_fs, _ = src_check

        dst_check = self._require_same_mount(dst_resolved, f"{op} destinations")
        if isinstance(dst_check, str):
            return self._error(dst_check)
        dst_fs, dst_prefix = dst_check

        src_rels = [r[1] for r in src_resolved]
        dst_rels = [r[1] for r in dst_resolved]

        if src_fs is dst_fs:
            batch = [TwoPathOperation(src=s, dest=d) for s, d in zip(src_rels, dst_rels, strict=True)]
            async with src_fs._use_session() as s:
                result = await getattr(src_fs, f"_{op}_impl")(
                    ops=batch,
                    overwrite=overwrite,
                    session=s,
                )
            return self._stamp(self._rebase_result(result, dst_prefix))

        return self._stamp(
            await self._cross_mount_transfer(
                op,
                src_fs,
                dst_fs,
                src_rels,
                dst_rels,
                dst_prefix,
                overwrite=overwrite,
            ),
        )

    async def _cross_mount_transfer(
        self,
        op: str,
        src_fs: GroverFileSystem,
        dst_fs: GroverFileSystem,
        src_rels: list[str],
        dst_rels: list[str],
        dst_prefix: str,
        *,
        overwrite: bool,
    ) -> GroverResult:
        """Execute a cross-mount move/copy via read → write → delete."""
        # Read all sources
        async with src_fs._use_session() as s:
            read_results = self._merge_results(
                [await src_fs._read_impl(p, session=s) for p in src_rels],
            )
        if not read_results.success:
            return self._rebase_result(read_results, dst_prefix)

        # Write all to destination
        async with dst_fs._use_session() as s:
            write_results = self._merge_results(
                [
                    await dst_fs._write_impl(
                        dst_rel,
                        content=candidate.content or "",
                        overwrite=overwrite,
                        session=s,
                    )
                    for dst_rel, candidate in zip(dst_rels, read_results, strict=True)
                ]
            )
        if not write_results.success:
            return self._rebase_result(write_results, dst_prefix)

        # Soft-delete sources for move
        if op == "move":
            async with src_fs._use_session() as s:
                delete_results = self._merge_results(
                    [await src_fs._delete_impl(p, permanent=False, session=s) for p in src_rels],
                )
            if not delete_results.success:
                # Writes succeeded but deletes failed — caller needs to know
                return self._rebase_result(delete_results, dst_prefix)

        return self._rebase_result(write_results, dst_prefix)

    async def _route_fanout(
        self,
        op: str,
        candidates: GroverResult | None,
        **kwargs: object,
    ) -> GroverResult:
        """Route a namespace operation — dispatch candidates or fan-out.

        With candidates: group by filesystem, dispatch in parallel.
        Without: query self + every mount in parallel, merge results.
        """
        if candidates:
            return await self._dispatch_candidates(op, candidates, **kwargs)

        async def _query_self() -> GroverResult:
            async with self._use_session() as s:
                return await getattr(self, f"_{op}_impl")(session=s, **kwargs)

        all_results = await asyncio.gather(
            _query_self(),
            *(getattr(fs, op)(**kwargs) for fs in self._mounts.values()),
        )

        self_result = self._exclude_mounted_paths(all_results[0])
        results = [self_result]
        for mount_path, r in zip(self._mounts, all_results[1:], strict=True):
            results.append(self._rebase_result(r, prefix=mount_path))

        merged = self._merge_results(results)
        merged._grover = self
        return merged

    # ===================================================================
    # Public methods — routers
    # ===================================================================

    # --- CRUD chainable (path | candidates) ---

    async def read(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("read", path, candidates)

    async def stat(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("stat", path, candidates)

    async def edit(
        self,
        path: str | None = None,
        old: str = "",
        new: str = "",
        candidates: GroverResult | None = None,
        replace_all: bool = False,
        edits: list[EditOperation] | None = None,
    ) -> GroverResult:
        if edits is None:
            edits = [EditOperation(old=old, new=new, replace_all=replace_all)]
        return await self._route_single(
            "edit",
            path,
            candidates,
            edits=edits,
        )

    async def ls(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("ls", path, candidates)

    async def delete(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
    ) -> GroverResult:
        return await self._route_single(
            "delete",
            path,
            candidates,
            permanent=permanent,
        )

    # --- CRUD path-only ---

    async def write(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
    ) -> GroverResult:
        return await self._route_single(
            "write",
            path,
            None,
            content=content,
            overwrite=overwrite,
        )

    async def mkdir(self, path: str) -> GroverResult:
        return await self._route_single("mkdir", path, None)

    async def tree(
        self,
        path: str,
        max_depth: int | None = None,
    ) -> GroverResult:
        return await self._route_single("tree", path, None, max_depth=max_depth)

    # --- Two-path ops (custom routing) ---

    async def move(
        self,
        src: str = "",
        dest: str = "",
        moves: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
    ) -> GroverResult:
        if moves is None:
            moves = [TwoPathOperation(src=src, dest=dest)]
        return await self._route_two_path("move", moves, overwrite=overwrite)

    async def copy(
        self,
        src: str = "",
        dest: str = "",
        copies: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
    ) -> GroverResult:
        if copies is None:
            copies = [TwoPathOperation(src=src, dest=dest)]
        return await self._route_two_path("copy", copies, overwrite=overwrite)

    async def mkconn(
        self,
        source: str,
        target: str,
        connection_type: str,
    ) -> GroverResult:
        src_fs, src_rel, src_pfx = self._resolve_terminal(source)
        tgt_fs, tgt_rel, _ = self._resolve_terminal(target)
        if src_fs is not tgt_fs:
            return self._error(
                f"Cross-mount connections not supported: {source} and {target} resolve to different filesystems",
            )
        async with src_fs._use_session() as s:
            result = await src_fs._mkconn_impl(
                src_rel,
                tgt_rel,
                connection_type,
                session=s,
            )
        return self._stamp(self._rebase_result(result, src_pfx))

    # --- Search (fan-out) ---

    async def glob(
        self,
        pattern: str,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_fanout("glob", candidates, pattern=pattern)

    async def grep(
        self,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "grep",
            candidates,
            pattern=pattern,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )

    async def semantic_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "semantic_search",
            candidates,
            query=query,
            k=k,
        )

    async def vector_search(
        self,
        vector: list[float],
        k: int = 15,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "vector_search",
            candidates,
            vector=vector,
            k=k,
        )

    async def lexical_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "lexical_search",
            candidates,
            query=query,
            k=k,
        )

    # --- Graph traversal (path | candidates) ---

    async def predecessors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("predecessors", path, candidates)

    async def successors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("successors", path, candidates)

    async def ancestors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("ancestors", path, candidates)

    async def descendants(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
    ) -> GroverResult:
        return await self._route_single("descendants", path, candidates)

    async def neighborhood(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        depth: int = 2,
    ) -> GroverResult:
        return await self._route_single("neighborhood", path, candidates, depth=depth)

    # --- Graph candidate-only (dispatch per filesystem) ---

    async def meeting_subgraph(self, candidates: GroverResult) -> GroverResult:
        return await self._dispatch_candidates("meeting_subgraph", candidates)

    async def min_meeting_subgraph(self, candidates: GroverResult) -> GroverResult:
        return await self._dispatch_candidates("min_meeting_subgraph", candidates)

    # --- Graph algorithms (fan-out) ---

    async def pagerank(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("pagerank", candidates)

    async def betweenness_centrality(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("betweenness_centrality", candidates)

    async def closeness_centrality(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("closeness_centrality", candidates)

    async def degree_centrality(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("degree_centrality", candidates)

    async def in_degree_centrality(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("in_degree_centrality", candidates)

    async def out_degree_centrality(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("out_degree_centrality", candidates)

    async def hits(self, candidates: GroverResult | None = None) -> GroverResult:
        return await self._route_fanout("hits", candidates)

    # ===================================================================
    # _*_impl stubs — subclasses override these
    # ===================================================================

    # --- CRUD ---

    async def _read_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _stat_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _edit_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        edits: list[EditOperation] | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _ls_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _delete_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _write_impl(
        self,
        path: str = "",
        content: str = "",
        overwrite: bool = True,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _mkdir_impl(
        self,
        path: str = "",
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _move_impl(
        self,
        ops: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _copy_impl(
        self,
        ops: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _mkconn_impl(
        self,
        source: str = "",
        target: str = "",
        connection_type: str = "",
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _tree_impl(
        self,
        path: str = "",
        max_depth: int | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    # --- Search ---

    async def _glob_impl(
        self,
        pattern: str = "",
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _grep_impl(
        self,
        pattern: str = "",
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _semantic_search_impl(
        self,
        query: str = "",
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _vector_search_impl(
        self,
        vector: list[float] | None = None,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _lexical_search_impl(
        self,
        query: str = "",
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    # --- Graph traversal ---

    async def _predecessors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _successors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _ancestors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _descendants_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _neighborhood_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        depth: int = 2,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    # --- Graph candidate-only ---

    async def _meeting_subgraph_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _min_meeting_subgraph_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    # --- Graph algorithms ---

    async def _pagerank_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _betweenness_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _closeness_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _in_degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _out_degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _hits_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

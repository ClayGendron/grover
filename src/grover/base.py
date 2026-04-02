"""GroverFileSystem — concrete async base class with mount routing.

The base class owns mount routing, session management, and path rebasing.
The filesystem object itself owns ``/`` — mounting at ``"/"`` is illegal.

Public methods are routers.  They resolve the terminal filesystem via
longest-prefix mount matching, delegate to ``_*_impl`` methods for actual
storage work, then rebase paths before returning.

Subclasses override ``_*_impl`` for their storage backend:
- ``DatabaseFileSystem`` — SQL via ``GroverObject``
- ``LocalFileSystem`` — disk bytes + SQL metadata
- ``GroverAsync`` — no storage, mount-only router (``storage=False``)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from grover.exceptions import _classify_error
from grover.paths import normalize_path
from grover.results import Candidate, EditOperation, GroverResult, TwoPathOperation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from grover.models import GroverObjectBase
    from grover.query import QueryPlan


class GroverFileSystem:
    """Async base class for all Grover filesystems."""

    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        storage: bool = True,
        raise_on_error: bool = False,
    ) -> None:
        self._storage = storage
        self._raise_on_error = raise_on_error
        self._engine = engine
        if session_factory is not None:
            self._session_factory: async_sessionmaker[AsyncSession] | None = session_factory
        elif engine is not None:
            self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
        elif storage:
            msg = "GroverFileSystem requires either engine or session_factory when storage=True"
            raise ValueError(msg)
        else:
            self._session_factory = None
        self._mounts: dict[str, GroverFileSystem] = {}
        self._sorted_mount_paths: list[str] = []
        self._name = self.__class__.__name__

    # -------------------------------------------------------------------
    # mounts and routing
    # -------------------------------------------------------------------

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
        if path in self._mounts:
            msg = f"Mount already exists at: {path}"
            raise ValueError(msg)
        filesystem._raise_on_error = self._raise_on_error
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

    def _rebuild_sorted_mounts(self) -> None:
        """Rebuild the pre-sorted mount path list (longest first)."""
        self._sorted_mount_paths = cast(
            "list[str]",
            sorted(self._mounts.keys(), key=len, reverse=True),
        )

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

    def _group_candidates_by_terminal(
        self,
        candidates: GroverResult,
    ) -> list[tuple[GroverFileSystem, str, GroverResult]]:
        """Group candidates by terminal filesystem, rebasing paths.

        Returns ``[(filesystem, prefix, rebased_candidates)]`` where each
        ``GroverResult`` contains candidates with paths relative to that
        terminal filesystem.
        """
        groups: dict[tuple[int, str], tuple[GroverFileSystem, list[Candidate]]] = {}
        for c in candidates.candidates:
            fs, rel, prefix = self._resolve_terminal(c.path)
            key = (id(fs), prefix)
            if key not in groups:
                groups[key] = (fs, [])
            groups[key][1].append(c.model_copy(update={"path": rel}))
        return [(fs, pfx, GroverResult(candidates=cands)) for ((_id, pfx), (fs, cands)) in groups.items()]

    def _group_objects_by_terminal(
        self,
        objects: Sequence[GroverObjectBase],
    ) -> list[tuple[GroverFileSystem, str, list[GroverObjectBase]]]:
        """Group objects by terminal filesystem, rebasing paths."""
        groups: dict[tuple[int, str], tuple[GroverFileSystem, str, list[GroverObjectBase]]] = {}
        for obj in objects:
            fs, _rel, prefix = self._resolve_terminal(obj.path)
            key = (id(fs), prefix)
            if key not in groups:
                groups[key] = (fs, prefix, [])
            obj.strip_prefix(prefix)
            groups[key][2].append(obj)
        return list(groups.values())

    @staticmethod
    def _require_same_mount(
        resolved: Sequence[tuple[GroverFileSystem, str, str]],
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

    async def _dispatch_candidates(
        self,
        op: str,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
        **kwargs: object,
    ) -> GroverResult:
        """Route a candidate-based operation to terminal filesystems in parallel.

        Groups candidates by terminal filesystem, calls ``_{op}_impl``
        with rebased candidates on each concurrently, then rebases and
        merges results.
        """
        groups = self._group_candidates_by_terminal(candidates)
        if not groups:
            return GroverResult(
                success=candidates.success,
                errors=list(candidates.errors),
                candidates=[],
            )

        async def _run_group(
            fs: GroverFileSystem,
            prefix: str,
            group_cands: GroverResult,
        ) -> GroverResult:
            async with fs._use_session() as s:
                impl = getattr(fs, f"_{op}_impl")
                r = await impl(candidates=group_cands, user_id=user_id, session=s, **kwargs)
            return r.add_prefix(prefix)

        results = await asyncio.gather(
            *(_run_group(fs, pfx, gc) for fs, pfx, gc in groups),
        )
        return self._merge_results(list(results)).inject_details(candidates)

    async def _route_single(
        self,
        op: str,
        path: str | None,
        candidates: GroverResult | None,
        *,
        user_id: str | None = None,
        **kwargs: object,
    ) -> GroverResult:
        """Route a single-path or candidate-based operation.

        With candidates: group by filesystem, dispatch in parallel.
        With path: resolve one terminal, call impl once.
        """
        if path is not None and candidates is not None:
            msg = "Exactly one of path or candidates must be provided"
            raise ValueError(msg)
        if path is None and candidates is None:
            msg = "Exactly one of path or candidates must be provided"
            raise ValueError(msg)

        if candidates is not None:
            return await self._dispatch_candidates(op, candidates, user_id=user_id, **kwargs)

        assert path is not None
        fs, rel, prefix = self._resolve_terminal(path)

        if fs is self and not self._storage:
            return self._error(f"No mount found for path: {path}")

        async with fs._use_session() as s:
            result = await getattr(fs, f"_{op}_impl")(rel, user_id=user_id, session=s, **kwargs)

        return result.add_prefix(prefix)

    async def _route_two_path(
        self,
        op: str,
        ops: Sequence[TwoPathOperation],
        *,
        overwrite: bool = True,
        user_id: str | None = None,
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
        _, src_prefix = src_check

        src_rels = [r[1] for r in src_resolved]
        dst_rels = [r[1] for r in dst_resolved]

        if src_fs is dst_fs:
            batch = [TwoPathOperation(src=s, dest=d) for s, d in zip(src_rels, dst_rels, strict=True)]
            async with src_fs._use_session() as s:
                result = await getattr(src_fs, f"_{op}_impl")(
                    ops=batch,
                    overwrite=overwrite,
                    user_id=user_id,
                    session=s,
                )
            return result.add_prefix(dst_prefix)

        return await self._cross_mount_transfer(
            op,
            src_fs,
            dst_fs,
            src_rels,
            dst_rels,
            src_prefix,
            dst_prefix,
            overwrite=overwrite,
            user_id=user_id,
        )

    async def _cross_mount_transfer(
        self,
        op: str,
        src_fs: GroverFileSystem,
        dst_fs: GroverFileSystem,
        src_rels: list[str],
        dst_rels: list[str],
        src_prefix: str,
        dst_prefix: str,
        *,
        overwrite: bool,
        user_id: str | None = None,
    ) -> GroverResult:
        """Execute a cross-mount move/copy via read → write → delete."""
        # Read all sources
        async with src_fs._use_session() as s:
            read_results = self._merge_results(
                [await src_fs._read_impl(p, user_id=user_id, session=s) for p in src_rels],
            )
        if not read_results.success:
            return read_results.add_prefix(src_prefix)

        # Write all to destination
        async with dst_fs._use_session() as s:
            write_results = self._merge_results(
                [
                    await dst_fs._write_impl(
                        dst_rel,
                        content=candidate.content or "",
                        overwrite=overwrite,
                        user_id=user_id,
                        session=s,
                    )
                    for dst_rel, candidate in zip(dst_rels, read_results.candidates, strict=True)
                ]
            )
        if not write_results.success:
            return write_results.add_prefix(dst_prefix)

        # Soft-delete sources for move
        if op == "move":
            async with src_fs._use_session() as s:
                delete_results = self._merge_results(
                    [await src_fs._delete_impl(p, permanent=False, user_id=user_id, session=s) for p in src_rels],
                )
            if not delete_results.success:
                # Writes succeeded but deletes failed — caller needs to know
                return delete_results.add_prefix(src_prefix)

        return write_results.add_prefix(dst_prefix)

    async def _route_fanout(
        self,
        op: str,
        candidates: GroverResult | None,
        *,
        user_id: str | None = None,
        **kwargs: object,
    ) -> GroverResult:
        """Route a namespace operation — dispatch candidates or fan-out.

        With candidates: group by filesystem, dispatch in parallel.
        Without: query self + every mount in parallel, merge results.
        When ``storage=False``, skips the self-query and fans out to mounts only.
        """
        if candidates is not None:
            return await self._dispatch_candidates(op, candidates, user_id=user_id, **kwargs)

        if not self._storage:
            if not self._mounts:
                return GroverResult(success=True, candidates=[])
            mount_results = await asyncio.gather(
                *(getattr(fs, op)(user_id=user_id, **kwargs) for fs in self._mounts.values()),
            )
            rebased = [r.add_prefix(mp) for mp, r in zip(self._mounts, mount_results, strict=True)]
            return self._merge_results(rebased)

        async def _query_self() -> GroverResult:
            async with self._use_session() as s:
                return await getattr(self, f"_{op}_impl")(user_id=user_id, session=s, **kwargs)

        all_results = await asyncio.gather(
            _query_self(),
            *(getattr(fs, op)(user_id=user_id, **kwargs) for fs in self._mounts.values()),
        )

        self_result = self._exclude_mounted_paths(all_results[0])
        results = [self_result]
        for mount_path, r in zip(self._mounts, all_results[1:], strict=True):
            results.append(r.add_prefix(mount_path))

        return self._merge_results(results)

    async def _route_write_batch(
        self,
        objects: Sequence[GroverObjectBase],
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Route a batch of object writes to terminal filesystems in parallel."""
        if not objects:
            return GroverResult(success=True, candidates=[])

        groups = self._group_objects_by_terminal(objects)

        async def _write_group(fs: GroverFileSystem, prefix: str, group_objs: list[GroverObjectBase]) -> GroverResult:
            async with fs._use_session() as s:
                result = await fs._write_impl(objects=group_objs, overwrite=overwrite, user_id=user_id, session=s)
            return result.add_prefix(prefix)

        results = await asyncio.gather(
            *(_write_group(fs, pfx, objs) for fs, pfx, objs in groups),
        )
        return self._merge_results(list(results))

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
    # sessions
    # -------------------------------------------------------------------

    @asynccontextmanager
    async def _use_session(self) -> AsyncIterator[AsyncSession]:
        """Create a session from this filesystem's factory.

        Commits on success, rolls back on error.
        """
        if self._session_factory is None:
            msg = f"{self._name} has no session factory (storage=False)"
            raise RuntimeError(msg)
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # -------------------------------------------------------------------
    # errors
    # -------------------------------------------------------------------

    def _error(self, errors: str | list[str] | GroverResult) -> GroverResult:
        """Create or check a failed result, raising if ``raise_on_error`` is set.

        Accepts either:
        - A string or list of strings → creates ``GroverResult(success=False)``
        - An existing ``GroverResult`` → returns it as-is if successful,
          raises if ``raise_on_error`` is set and ``success`` is ``False``
        """
        if isinstance(errors, GroverResult):
            if errors.success or not self._raise_on_error:
                return errors
            result = errors
        else:
            error_list = [errors] if isinstance(errors, str) else errors
            result = GroverResult(success=False, errors=error_list)
            if not self._raise_on_error:
                return result

        raise _classify_error(result.error_message, result.errors, result)

    def parse_query(self, query: str) -> QueryPlan:
        """Parse a CLI-style query string into an execution plan."""
        from grover.query import parse_query

        return parse_query(query)

    async def run_query(
        self,
        query: str,
        *,
        user_id: str | None = None,
        initial: GroverResult | None = None,
    ) -> GroverResult:
        """Execute a parsed CLI-style query against this filesystem."""
        from grover.query import execute_query

        plan = self.parse_query(query)
        return await execute_query(self, plan, initial=initial, user_id=user_id)

    async def cli(
        self,
        query: str,
        *,
        user_id: str | None = None,
        initial: GroverResult | None = None,
    ) -> str:
        """Execute *query* and render a human-readable text response."""
        from grover.query import execute_query, render_query_result

        plan = self.parse_query(query)
        result = await execute_query(self, plan, initial=initial, user_id=user_id)
        return render_query_result(result, mode=plan.render_mode)

    # -------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------

    # crud

    async def read(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("read", path, candidates, user_id=user_id)

    async def stat(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("stat", path, candidates, user_id=user_id)

    async def edit(
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
        if edits is None:
            if old is None or new is None:
                return self._error("edit requires old and new strings, or edits list")
            edits = [EditOperation(old=old, new=new, replace_all=replace_all)]
        return await self._route_single("edit", path, candidates, edits=edits, user_id=user_id)

    async def ls(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("ls", path, candidates, user_id=user_id)

    async def delete(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
        cascade: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single(
            "delete",
            path,
            candidates,
            permanent=permanent,
            cascade=cascade,
            user_id=user_id,
        )

    async def write(
        self,
        path: str | None = None,
        content: str | None = None,
        objects: Sequence[GroverObjectBase] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        if objects is not None:
            return await self._route_write_batch(objects, overwrite=overwrite, user_id=user_id)
        return await self._route_single(
            "write",
            path,
            None,
            content=content,
            overwrite=overwrite,
            user_id=user_id,
        )

    async def mkdir(self, path: str, *, user_id: str | None = None) -> GroverResult:
        return await self._route_single("mkdir", path, None, user_id=user_id)

    async def tree(
        self,
        path: str,
        max_depth: int | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("tree", path, None, max_depth=max_depth, user_id=user_id)

    async def move(
        self,
        src: str | None = None,
        dest: str | None = None,
        moves: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        if moves is None:
            if not src or not dest:
                return self._error("move requires src and dest, or moves")
            moves = [TwoPathOperation(src=src, dest=dest)]
        return await self._route_two_path("move", moves, overwrite=overwrite, user_id=user_id)

    async def copy(
        self,
        src: str | None = None,
        dest: str | None = None,
        copies: list[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        if copies is None:
            if not src or not dest:
                return self._error("copy requires src and dest, or copies")
            copies = [TwoPathOperation(src=src, dest=dest)]
        return await self._route_two_path("copy", copies, overwrite=overwrite, user_id=user_id)

    async def mkconn(
        self,
        source: str,
        target: str,
        connection_type: str,
        *,
        user_id: str | None = None,
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
                user_id=user_id,
                session=s,
            )
        return result.add_prefix(src_pfx)

    # search

    async def glob(
        self,
        pattern: str,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("glob", candidates, pattern=pattern, user_id=user_id)

    async def grep(
        self,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "grep",
            candidates,
            pattern=pattern,
            case_sensitive=case_sensitive,
            max_results=max_results,
            user_id=user_id,
        )

    async def semantic_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "semantic_search",
            candidates,
            query=query,
            k=k,
            user_id=user_id,
        )

    async def vector_search(
        self,
        vector: list[float],
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "vector_search",
            candidates,
            vector=vector,
            k=k,
            user_id=user_id,
        )

    async def lexical_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout(
            "lexical_search",
            candidates,
            query=query,
            k=k,
            user_id=user_id,
        )

    # graph

    async def predecessors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("predecessors", path, candidates, user_id=user_id)

    async def successors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("successors", path, candidates, user_id=user_id)

    async def ancestors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("ancestors", path, candidates, user_id=user_id)

    async def descendants(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("descendants", path, candidates, user_id=user_id)

    async def neighborhood(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        depth: int = 2,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_single("neighborhood", path, candidates, depth=depth, user_id=user_id)

    async def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._dispatch_candidates("meeting_subgraph", candidates, user_id=user_id)

    async def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._dispatch_candidates("min_meeting_subgraph", candidates, user_id=user_id)

    async def pagerank(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("pagerank", candidates, user_id=user_id)

    async def betweenness_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("betweenness_centrality", candidates, user_id=user_id)

    async def closeness_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("closeness_centrality", candidates, user_id=user_id)

    async def degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("degree_centrality", candidates, user_id=user_id)

    async def in_degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("in_degree_centrality", candidates, user_id=user_id)

    async def out_degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("out_degree_centrality", candidates, user_id=user_id)

    async def hits(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        return await self._route_fanout("hits", candidates, user_id=user_id)

    # -------------------------------------------------------------------
    # impl stubs
    # -------------------------------------------------------------------

    async def _read_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _stat_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _edit_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        edits: list[EditOperation] | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _ls_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _delete_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
        cascade: bool = True,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _write_impl(
        self,
        path: str | None = None,
        content: str | None = None,
        objects: Sequence[GroverObjectBase] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _mkdir_impl(
        self,
        path: str,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _move_impl(
        self,
        ops: Sequence[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _copy_impl(
        self,
        ops: Sequence[TwoPathOperation] | None = None,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _mkconn_impl(
        self,
        source: str | None = None,
        target: str | None = None,
        connection_type: str | None = None,
        objects: Sequence[GroverObjectBase] | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _tree_impl(
        self,
        path: str,
        max_depth: int | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _glob_impl(
        self,
        pattern: str,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _grep_impl(
        self,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _semantic_search_impl(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _vector_search_impl(
        self,
        vector: list[float] | None = None,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _lexical_search_impl(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _predecessors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _successors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _ancestors_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _descendants_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _neighborhood_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        depth: int = 2,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _meeting_subgraph_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _min_meeting_subgraph_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _pagerank_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _betweenness_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _closeness_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _in_degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _out_degree_centrality_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

    async def _hits_impl(
        self,
        candidates: GroverResult | None = None,
        *,
        user_id: str | None = None,
        session: AsyncSession,
    ) -> GroverResult:
        raise NotImplementedError

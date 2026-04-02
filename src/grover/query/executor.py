"""Execution engine for the CLI query AST."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

from grover.paths import normalize_path, parse_kind
from grover.query.ast import (
    CopyCommand,
    DeleteCommand,
    EditCommand,
    ExceptStage,
    GlobCommand,
    GraphTraversalCommand,
    GrepCommand,
    IntersectStage,
    KindsCommand,
    LexicalSearchCommand,
    LsCommand,
    MeetingGraphCommand,
    MkconnCommand,
    MkdirCommand,
    MoveCommand,
    PipelineNode,
    QueryNode,
    QueryPlan,
    RankCommand,
    ReadCommand,
    SemanticSearchCommand,
    SortCommand,
    StageNode,
    StatCommand,
    TopCommand,
    TreeCommand,
    UnionNode,
    VectorSearchCommand,
    Visibility,
    WriteCommand,
)
from grover.results import Candidate, GroverResult, TwoPathOperation

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from grover.base import GroverFileSystem


async def execute_query(
    filesystem: GroverFileSystem,
    plan: QueryPlan,
    *,
    initial: GroverResult | None = None,
    user_id: str | None = None,
) -> GroverResult:
    """Execute a parsed query plan against *filesystem*."""
    return await _execute_node(filesystem, plan.ast, initial, user_id=user_id)


async def _execute_node(
    filesystem: GroverFileSystem,
    node: QueryNode,
    current: GroverResult | None,
    *,
    user_id: str | None = None,
) -> GroverResult:
    match node:
        case PipelineNode(source=source, stages=stages):
            result = await _execute_node(filesystem, source, current, user_id=user_id)
            for stage in stages:
                result = await _execute_stage(filesystem, stage, result, user_id=user_id)
            return result
        case UnionNode(operands=operands):
            results = await asyncio.gather(
                *(_execute_node(filesystem, operand, current, user_id=user_id) for operand in operands),
            )
            return filesystem._merge_results(list(results))
        case _:
            assert isinstance(node, StageNode)
            return await _execute_stage(filesystem, node, current, user_id=user_id)


async def _execute_stage(
    filesystem: GroverFileSystem,
    stage: StageNode,
    current: GroverResult | None,
    *,
    user_id: str | None = None,
) -> GroverResult:
    match stage:
        case ReadCommand(paths=paths):
            return await _read_like(filesystem.read, current, paths, command_name="read", user_id=user_id)
        case StatCommand(paths=paths):
            return await _read_like(filesystem.stat, current, paths, command_name="stat", user_id=user_id)
        case DeleteCommand(paths=paths):
            return await _read_like(filesystem.delete, current, paths, command_name="rm", user_id=user_id)
        case EditCommand(old=old, new=new, paths=paths, replace_all=replace_all):
            if current is not None and paths:
                raise ValueError("edit cannot combine piped input with explicit paths")
            if current is not None:
                return await filesystem.edit(
                    candidates=current, old=old, new=new, replace_all=replace_all, user_id=user_id,
                )
            if not paths:
                raise ValueError("edit requires a path when it is not used in a pipeline")
            explicit = _paths_result(paths)
            return await filesystem.edit(
                candidates=explicit, old=old, new=new, replace_all=replace_all, user_id=user_id,
            )
        case WriteCommand(path=path, content=content, overwrite=overwrite):
            if current is not None:
                raise ValueError("write cannot be used in a pipeline")
            return await filesystem.write(
                path=normalize_path(path), content=content, overwrite=overwrite, user_id=user_id,
            )
        case MkdirCommand(paths=paths):
            if current is not None:
                raise ValueError("mkdir cannot be used in a pipeline")
            results = await asyncio.gather(
                *(filesystem.mkdir(normalize_path(path), user_id=user_id) for path in paths),
            )
            return filesystem._merge_results(list(results))
        case MoveCommand(src=src, dest=dest, overwrite=overwrite):
            return await _execute_transfer(filesystem, "move", current, src, dest, overwrite, user_id=user_id)
        case CopyCommand(src=src, dest=dest, overwrite=overwrite):
            return await _execute_transfer(filesystem, "copy", current, src, dest, overwrite, user_id=user_id)
        case MkconnCommand(source=source, connection_type=connection_type, target=target):
            return await _execute_mkconn(filesystem, current, source, connection_type, target, user_id=user_id)
        case LsCommand(paths=paths):
            if current is not None and paths:
                raise ValueError("ls cannot combine piped input with explicit paths")
            if current is not None:
                return await filesystem.ls(candidates=current, user_id=user_id)
            if not paths:
                return await filesystem.ls(path="/", user_id=user_id)
            explicit = _paths_result(paths)
            return await filesystem.ls(candidates=explicit, user_id=user_id)
        case TreeCommand(paths=paths, max_depth=max_depth, visibility=visibility):
            roots = tuple(normalize_path(path) for path in paths) if paths else ()
            return await _execute_tree(filesystem, current, roots, max_depth, visibility, user_id=user_id)
        case GlobCommand(pattern=pattern, visibility=visibility):
            result = await _execute_glob(filesystem, current, pattern, visibility, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory"})
        case GrepCommand(
            pattern=pattern,
            case_sensitive=case_sensitive,
            max_results=max_results,
            visibility=visibility,
        ):
            result = await _execute_grep(
                filesystem, current, pattern, case_sensitive, max_results, visibility, user_id=user_id,
            )
            return _apply_visibility(result, visibility, {"file", "directory"})
        case SemanticSearchCommand(query=query, k=k, visibility=visibility):
            result = await filesystem.semantic_search(query=query, k=k, candidates=current, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory"})
        case LexicalSearchCommand(query=query, k=k, visibility=visibility):
            result = await _execute_lexical(filesystem, current, query, k, visibility, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory"})
        case VectorSearchCommand(vector=vector, k=k, visibility=visibility):
            result = await filesystem.vector_search(vector=list(vector), k=k, candidates=current, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory"})
        case GraphTraversalCommand(method_name=method_name, paths=paths, depth=depth, visibility=visibility):
            result = await _execute_graph_traversal(filesystem, current, method_name, paths, depth, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory", "connection"})
        case MeetingGraphCommand(paths=paths, minimal=minimal, visibility=visibility):
            seeds = _seed_candidates(current, paths, "meetinggraph")
            result = (
                await filesystem.min_meeting_subgraph(candidates=seeds, user_id=user_id)
                if minimal
                else await filesystem.meeting_subgraph(candidates=seeds, user_id=user_id)
            )
            return _apply_visibility(result, visibility, {"file", "directory", "connection"})
        case RankCommand(method_name=method_name, paths=paths, visibility=visibility):
            result = await _execute_rank(filesystem, current, method_name, paths, user_id=user_id)
            return _apply_visibility(result, visibility, {"file", "directory", "connection"})
        case SortCommand(operation=operation, reverse=reverse):
            if current is None:
                raise ValueError("sort requires piped input")
            return current.sort(operation=operation, reverse=reverse)
        case TopCommand(k=k):
            if current is None:
                raise ValueError("top requires piped input")
            return current.top(k)
        case KindsCommand(kinds=kinds):
            if current is None:
                raise ValueError("kinds requires piped input")
            return current.kinds(*kinds)
        case IntersectStage(query=query):
            if current is None:
                raise ValueError("intersect requires piped input")
            other = await _execute_node(filesystem, query, None, user_id=user_id)
            return current & other
        case ExceptStage(query=query):
            if current is None:
                raise ValueError("except requires piped input")
            other = await _execute_node(filesystem, query, None, user_id=user_id)
            return current - other
        case _:  # pragma: no cover
            raise ValueError(f"Unknown stage: {stage}")


async def _read_like(
    method: Callable[..., Awaitable[GroverResult]],
    current: GroverResult | None,
    paths: tuple[str, ...],
    *,
    command_name: str,
    user_id: str | None = None,
) -> GroverResult:
    if current is not None and paths:
        raise ValueError(f"{command_name} cannot combine piped input with explicit paths")
    if current is not None:
        return await method(candidates=current, user_id=user_id)
    if not paths:
        raise ValueError(f"{command_name} requires explicit paths when it is not used in a pipeline")
    explicit = _paths_result(paths)
    return await method(candidates=explicit, user_id=user_id)


async def _execute_transfer(
    filesystem: GroverFileSystem,
    op: str,
    current: GroverResult | None,
    src: str | None,
    dest: str,
    overwrite: bool,
    *,
    user_id: str | None = None,
) -> GroverResult:
    normalized_dest = normalize_path(dest)
    match (current, src):
        case (None, None):
            raise ValueError(f"{op} requires a source path when it is not used in a pipeline")
        case (None, str() as source):
            method = filesystem.move if op == "move" else filesystem.copy
            return await method(src=normalize_path(source), dest=normalized_dest, overwrite=overwrite, user_id=user_id)
        case (GroverResult(), str()):
            raise ValueError(f"{op} cannot combine piped input with an explicit source path")
        case (GroverResult(candidates=candidates), None):
            if not candidates:
                return GroverResult(candidates=[])
            ops = [
                TwoPathOperation(src=candidate.path, dest=_preserve_under_root(normalized_dest, candidate.path))
                for candidate in candidates
            ]
            if op == "move":
                return await filesystem.move(moves=ops, overwrite=overwrite, user_id=user_id)
            return await filesystem.copy(copies=ops, overwrite=overwrite, user_id=user_id)
        case _:  # pragma: no cover
            raise ValueError(f"Invalid arguments for {op}")


async def _execute_mkconn(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    source: str | None,
    connection_type: str,
    target: str,
    *,
    user_id: str | None = None,
) -> GroverResult:
    normalized_target = normalize_path(target)
    if current is None:
        if source is None:
            raise ValueError("mkconn requires a source path when it is not used in a pipeline")
        return await filesystem.mkconn(
            source=normalize_path(source),
            target=normalized_target,
            connection_type=connection_type,
            user_id=user_id,
        )
    if source is not None:
        raise ValueError("mkconn cannot combine piped input with an explicit source path")
    results = await asyncio.gather(
        *(
            filesystem.mkconn(
                source=candidate.path,
                target=normalized_target,
                connection_type=connection_type,
                user_id=user_id,
            )
            for candidate in current.candidates
        )
    )
    return filesystem._merge_results(list(results))


async def _execute_tree(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    roots: tuple[str, ...],
    max_depth: int | None,
    visibility: Visibility,
    *,
    user_id: str | None = None,
) -> GroverResult:
    if current is not None and roots:
        raise ValueError("tree cannot combine piped input with explicit paths")

    if current is not None:
        roots = tuple(candidate.path for candidate in current.candidates)
    elif not roots:
        roots = ("/",)

    if visibility.include_all or visibility.include_kinds:
        return await _collect_tree(filesystem, roots, max_depth=max_depth, visibility=visibility, user_id=user_id)

    results = await asyncio.gather(
        *(filesystem.tree(path=root, max_depth=max_depth, user_id=user_id) for root in roots),
    )
    return filesystem._merge_results(list(results))


async def _execute_glob(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    pattern: str,
    visibility: Visibility,
    *,
    user_id: str | None = None,
) -> GroverResult:
    if current is not None:
        return await filesystem.glob(pattern=pattern, candidates=current, user_id=user_id)
    if visibility.include_all or visibility.include_kinds:
        universe = await _collect_tree(filesystem, ("/",), max_depth=None, visibility=visibility, user_id=user_id)
        return await filesystem.glob(pattern=pattern, candidates=universe, user_id=user_id)
    return await filesystem.glob(pattern=pattern, user_id=user_id)


async def _execute_grep(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    pattern: str,
    case_sensitive: bool,
    max_results: int | None,
    visibility: Visibility,
    *,
    user_id: str | None = None,
) -> GroverResult:
    candidates = current
    if candidates is None and (visibility.include_all or visibility.include_kinds):
        candidates = await _collect_tree(filesystem, ("/",), max_depth=None, visibility=visibility, user_id=user_id)
    if candidates is not None and any(
        _candidate_kind(candidate) != "file" and candidate.content is None
        for candidate in candidates.candidates
    ):
        candidates = await filesystem.read(candidates=candidates, user_id=user_id)
    return await filesystem.grep(
        pattern=pattern,
        case_sensitive=case_sensitive,
        max_results=max_results,
        candidates=candidates,
        user_id=user_id,
    )


async def _execute_lexical(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    query: str,
    k: int,
    visibility: Visibility,
    *,
    user_id: str | None = None,
) -> GroverResult:
    candidates = current
    if candidates is None and (visibility.include_all or visibility.include_kinds):
        candidates = await _collect_tree(filesystem, ("/",), max_depth=None, visibility=visibility, user_id=user_id)
    return await filesystem.lexical_search(query=query, k=k, candidates=candidates, user_id=user_id)


async def _execute_graph_traversal(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    method_name: str,
    paths: tuple[str, ...],
    depth: int,
    *,
    user_id: str | None = None,
) -> GroverResult:
    method = getattr(filesystem, method_name)
    if current is not None and paths:
        raise ValueError(f"{method_name} cannot combine piped input with explicit paths")
    if current is not None:
        if method_name == "neighborhood":
            return await method(candidates=current, depth=depth, user_id=user_id)
        return await method(candidates=current, user_id=user_id)
    if paths:
        explicit = _paths_result(paths)
        if len(paths) > 1:
            if method_name == "neighborhood":
                return await method(candidates=explicit, depth=depth, user_id=user_id)
            return await method(candidates=explicit, user_id=user_id)
        if method_name == "neighborhood":
            return await method(path=explicit.candidates[0].path, depth=depth, user_id=user_id)
        return await method(path=explicit.candidates[0].path, user_id=user_id)
    raise ValueError(f"{method_name} requires explicit paths when it is not used in a pipeline")


async def _execute_rank(
    filesystem: GroverFileSystem,
    current: GroverResult | None,
    method_name: str,
    paths: tuple[str, ...],
    *,
    user_id: str | None = None,
) -> GroverResult:
    method = getattr(filesystem, method_name)
    if current is not None and paths:
        raise ValueError(f"{method_name} cannot combine piped input with explicit paths")
    if current is not None:
        return await method(candidates=current, user_id=user_id)
    if paths:
        return await method(candidates=_paths_result(paths), user_id=user_id)
    return await method(user_id=user_id)


def _seed_candidates(current: GroverResult | None, paths: tuple[str, ...], label: str) -> GroverResult:
    if current is not None and paths:
        raise ValueError(f"{label} cannot combine piped input with explicit paths")
    if current is not None:
        return current
    if not paths:
        raise ValueError(f"{label} requires explicit paths when it is not used in a pipeline")
    return _paths_result(paths)


def _paths_result(paths: tuple[str, ...]) -> GroverResult:
    candidates = []
    for path in paths:
        normalized = normalize_path(path)
        candidates.append(Candidate(path=normalized, kind=parse_kind(normalized)))
    return GroverResult(candidates=candidates)


def _candidate_kind(candidate: Candidate) -> str:
    return candidate.kind or parse_kind(candidate.path)


def _preserve_under_root(root: str, source_path: str) -> str:
    relative = normalize_path(source_path).lstrip("/")
    if not relative:
        raise ValueError("Cannot preserve the root path under a destination root")
    return normalize_path(f"{root}/{relative}")


def _apply_visibility(
    result: GroverResult,
    visibility: Visibility,
    defaults: set[str],
) -> GroverResult:
    if visibility.include_all:
        return result
    allowed = set(defaults) | set(visibility.include_kinds)
    filtered = [candidate for candidate in result.candidates if _candidate_kind(candidate) in allowed]
    return result._with_candidates(filtered)


async def _collect_tree(
    filesystem: GroverFileSystem,
    roots: tuple[str, ...],
    *,
    max_depth: int | None,
    visibility: Visibility,
    user_id: str | None = None,
) -> GroverResult:
    queue: deque[tuple[str, int]] = deque((normalize_path(root), 0) for root in roots)
    seen: set[str] = set()
    collected: list[Candidate] = []
    while queue:
        path, depth = queue.popleft()
        if path in seen:
            continue
        seen.add(path)

        if max_depth is not None and depth >= max_depth:
            continue

        listing = await filesystem.ls(path=path, user_id=user_id)
        if not listing.success:
            return listing

        visible_listing = _apply_visibility(listing, visibility, {"file", "directory"})
        collected.extend(visible_listing.candidates)

        for candidate in listing.candidates:
            kind = _candidate_kind(candidate)
            if kind in {"file", "directory"}:
                queue.append((candidate.path, depth + 1))

    deduped: dict[str, Candidate] = {candidate.path: candidate for candidate in collected}
    ordered = [deduped[path] for path in sorted(deduped)]
    return GroverResult(candidates=ordered)

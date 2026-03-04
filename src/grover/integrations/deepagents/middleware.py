"""GroverMiddleware — deepagents AgentMiddleware exposing Grover-specific tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from grover.client import Grover, GroverAsync


# ------------------------------------------------------------------
# String formatters for tool return values
# ------------------------------------------------------------------


def _format_graph_result(result: object, label: str) -> str:
    """Format a GraphResult into a readable string."""
    from grover.results import GraphResult

    if isinstance(result, GraphResult):
        if len(result) == 0:
            return f"No {label} found."
        lines = [f"Found {len(result)} {label}:"]
        lines.extend(f"  - {path}" for path in result.paths)
        return "\n".join(lines)
    # Fallback for any unexpected type
    return f"No {label} found."


def _format_version_list(result: object, path: str) -> str:
    """Format a VersionResult into readable text."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    if len(result) == 0:  # type: ignore[arg-type]
        return f"No versions found for {path}."

    from grover.results import VersionEvidence

    lines = [f"Version history for {path} ({len(result)} versions):"]  # type: ignore[arg-type]
    for candidate in result.candidates:  # type: ignore[union-attr]
        for ev in candidate.evidence:
            if isinstance(ev, VersionEvidence):
                line = f"  v{ev.version}: {ev.created_at:%Y-%m-%d %H:%M:%S}"
                line += f" | {ev.size_bytes} bytes | hash={ev.content_hash[:12]}"
                if ev.created_by:
                    line += f" | by {ev.created_by}"
                lines.append(line)
                break
    return "\n".join(lines)


def _format_version_content(result: object, path: str, version: int) -> str:
    """Format a get_version_content result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    if not result.content:  # type: ignore[union-attr]
        return f"Error: No content found for {path} v{version}."

    return f"Content of {path} v{version}:\n{result.content}"  # type: ignore[union-attr]


def _format_restore_result(result: object) -> str:
    """Format a restore_version result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    return (
        f"Restored {result.path} to v{result.restored_version}. "  # type: ignore[union-attr]
        f"Current version is now v{result.version}."  # type: ignore[union-attr]
    )


def _format_delete_result(result: object, path: str) -> str:
    """Format a delete result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    return f"Deleted {path} (moved to trash). Use restore_from_trash to recover it."


def _format_trash_list(result: object) -> str:
    """Format a list_trash result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    if len(result) == 0:  # type: ignore[arg-type]
        return "Trash is empty."

    lines = [f"Trash ({len(result)} items):"]  # type: ignore[arg-type]
    lines.extend(f"  - {path}" for path in result.paths)  # type: ignore[union-attr]
    return "\n".join(lines)


def _format_trash_restore(result: object) -> str:
    """Format a restore_from_trash result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    return f"Restored {result.path} from trash."  # type: ignore[union-attr]


def _format_search_result(result: object, query: str) -> str:
    """Format a vector_search result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    if len(result) == 0:  # type: ignore[arg-type]
        return f"No results found for: {query}"

    lines = [f"Search results for '{query}' ({len(result)} files):"]  # type: ignore[arg-type]
    for i, path in enumerate(result.paths, 1):  # type: ignore[union-attr]
        line = f"  {i}. {path}"
        lines.append(line)
        for snippet in result.snippets(path)[:3]:  # type: ignore[union-attr]
            snippet_text = snippet.replace("\n", " ")
            lines.append(f"     {snippet_text}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# GroverMiddleware
# ------------------------------------------------------------------


class GroverMiddleware(AgentMiddleware):
    """deepagents middleware exposing Grover version, search, graph, and trash tools.

    Adds tools beyond standard file operations: version history, semantic
    search, dependency graph queries, and soft-delete trash management.

    Accepts either a sync :class:`~grover.Grover` or async
    :class:`~grover.GroverAsync` instance. When ``GroverAsync`` is passed,
    tools include both sync and async (coroutine) implementations for native
    async execution.

    Usage::

        from grover import Grover
        from grover.integrations.deepagents import GroverMiddleware

        g = Grover()
        middleware = GroverMiddleware(g)
        # Pass middleware.tools to the agent
    """

    def __init__(
        self,
        grover: Grover | GroverAsync,
        *,
        enable_search: bool = True,
        enable_graph: bool = True,
    ) -> None:
        from grover.client import GroverAsync

        self.grover = grover
        self._is_async = isinstance(grover, GroverAsync)
        tool_list: list[BaseTool] = [
            self._create_list_versions_tool(),
            self._create_get_version_content_tool(),
            self._create_restore_version_tool(),
            self._create_delete_file_tool(),
            self._create_list_trash_tool(),
            self._create_restore_from_trash_tool(),
        ]
        if enable_search:
            tool_list.append(self._create_search_semantic_tool())
        if enable_graph:
            tool_list.extend(
                [
                    self._create_dependencies_tool(),
                    self._create_dependents_tool(),
                    self._create_impacts_tool(),
                ]
            )
        self.tools: list[BaseTool] = tool_list

    # ------------------------------------------------------------------
    # Version tools
    # ------------------------------------------------------------------

    def _create_list_versions_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def list_versions(
            path: Annotated[str, "Absolute virtual path to the file, e.g. /project/main.py"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(list_versions_async(path))
            assert grover_s is not None
            try:
                result = grover_s.list_versions(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_version_list(result, path)

        async def list_versions_async(path: str) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.list_versions(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_version_list(result, path)

        kwargs: dict[str, object] = {
            "name": "list_versions",
            "description": (
                "Show the version history of a file. Returns a list of versions "
                "with timestamps, sizes, hashes, and who made each change. "
                "Use this to understand how a file has evolved over time."
            ),
            "func": list_versions,
        }
        if is_async:
            kwargs["coroutine"] = list_versions_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    def _create_get_version_content_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def get_version_content(
            path: Annotated[str, "Absolute virtual path to the file"],
            version: Annotated[int, "Version number to retrieve (from list_versions output)"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(get_version_content_async(path, version))
            assert grover_s is not None
            try:
                result = grover_s.get_version_content(path, version)
            except Exception as e:
                return f"Error: {e}"
            return _format_version_content(result, path, version)

        async def get_version_content_async(path: str, version: int) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.get_version_content(path, version)
            except Exception as e:
                return f"Error: {e}"
            return _format_version_content(result, path, version)

        kwargs: dict[str, object] = {
            "name": "get_version_content",
            "description": (
                "Read the content of a specific past version of a file. "
                "Use list_versions first to see available versions, then "
                "pass the version number here to read that version's content."
            ),
            "func": get_version_content,
        }
        if is_async:
            kwargs["coroutine"] = get_version_content_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    def _create_restore_version_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def restore_version(
            path: Annotated[str, "Absolute virtual path to the file"],
            version: Annotated[int, "Version number to restore to"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(restore_version_async(path, version))
            assert grover_s is not None
            try:
                result = grover_s.restore_version(path, version)
            except Exception as e:
                return f"Error: {e}"
            return _format_restore_result(result)

        async def restore_version_async(path: str, version: int) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.restore_version(path, version)
            except Exception as e:
                return f"Error: {e}"
            return _format_restore_result(result)

        kwargs: dict[str, object] = {
            "name": "restore_version",
            "description": (
                "Restore a file to a previous version. This creates a new "
                "version with the content from the specified old version — "
                "it does not discard history. Use list_versions to find "
                "the version number, then restore to it."
            ),
            "func": restore_version,
        }
        if is_async:
            kwargs["coroutine"] = restore_version_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Trash tools
    # ------------------------------------------------------------------

    def _create_delete_file_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def delete_file(
            path: Annotated[str, "Absolute virtual path to the file to delete"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(delete_file_async(path))
            assert grover_s is not None
            try:
                result = grover_s.delete(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_delete_result(result, path)

        async def delete_file_async(path: str) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.delete(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_delete_result(result, path)

        kwargs: dict[str, object] = {
            "name": "delete_file",
            "description": (
                "Soft-delete a file by moving it to trash. The file can be "
                "recovered later using restore_from_trash. This is safer "
                "than permanent deletion."
            ),
            "func": delete_file,
        }
        if is_async:
            kwargs["coroutine"] = delete_file_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    def _create_list_trash_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def list_trash() -> str:
            if grover_a is not None:
                return asyncio.run(list_trash_async())
            assert grover_s is not None
            try:
                result = grover_s.list_trash()
            except Exception as e:
                return f"Error: {e}"
            return _format_trash_list(result)

        async def list_trash_async() -> str:
            assert grover_a is not None
            try:
                result = await grover_a.list_trash()
            except Exception as e:
                return f"Error: {e}"
            return _format_trash_list(result)

        kwargs: dict[str, object] = {
            "name": "list_trash",
            "description": (
                "List all soft-deleted files in the trash. Shows file paths "
                "and sizes. Use restore_from_trash to recover a specific file."
            ),
            "func": list_trash,
        }
        if is_async:
            kwargs["coroutine"] = list_trash_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    def _create_restore_from_trash_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def restore_from_trash(
            path: Annotated[str, "Path of the trashed file to restore (from list_trash output)"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(restore_from_trash_async(path))
            assert grover_s is not None
            try:
                result = grover_s.restore_from_trash(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_trash_restore(result)

        async def restore_from_trash_async(path: str) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.restore_from_trash(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_trash_restore(result)

        kwargs: dict[str, object] = {
            "name": "restore_from_trash",
            "description": (
                "Restore a previously deleted file from trash. Use list_trash "
                "first to see available files, then pass the path here."
            ),
            "func": restore_from_trash,
        }
        if is_async:
            kwargs["coroutine"] = restore_from_trash_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    def _create_search_semantic_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def search_semantic(
            query: Annotated[
                str, "Natural language search query describing what you're looking for"
            ],
            k: Annotated[int, "Maximum number of results to return"] = 10,
        ) -> str:
            if grover_a is not None:
                return asyncio.run(search_semantic_async(query, k))
            assert grover_s is not None
            try:
                result = grover_s.vector_search(query, k=k)
            except Exception as e:
                return f"Error: {e}"
            return _format_search_result(result, query)

        async def search_semantic_async(query: str, k: int = 10) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.vector_search(query, k=k)
            except Exception as e:
                return f"Error: {e}"
            return _format_search_result(result, query)

        kwargs: dict[str, object] = {
            "name": "search_semantic",
            "description": (
                "Search the codebase using semantic similarity. Finds files "
                "by meaning, not just text pattern. For example, searching "
                "'authentication logic' will find files about login, tokens, "
                "and sessions even if they don't contain the exact phrase. "
                "Results are ranked by relevance score (0-1, higher is better)."
            ),
            "func": search_semantic,
        }
        if is_async:
            kwargs["coroutine"] = search_semantic_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Graph tools (sync on both Grover and GroverAsync — in-memory rustworkx)
    # ------------------------------------------------------------------

    def _create_dependencies_tool(self) -> BaseTool:
        grover = self.grover

        def dependencies(
            path: Annotated[str, "Absolute virtual path to the file"],
        ) -> str:
            try:
                result = grover.dependencies(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "dependencies")

        return StructuredTool.from_function(
            name="dependencies",
            description=(
                "Show what files this file imports or depends on. Returns "
                "the direct dependency list from the knowledge graph."
            ),
            func=dependencies,
        )

    def _create_dependents_tool(self) -> BaseTool:
        grover = self.grover

        def dependents(
            path: Annotated[str, "Absolute virtual path to the file"],
        ) -> str:
            try:
                result = grover.dependents(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "dependents")

        return StructuredTool.from_function(
            name="dependents",
            description=(
                "Show what files depend on or import this file. Useful for "
                "understanding the impact of changes — if many files depend "
                "on this one, changes need extra care."
            ),
            func=dependents,
        )

    def _create_impacts_tool(self) -> BaseTool:
        grover = self.grover

        def impacts(
            path: Annotated[str, "Absolute virtual path to the file"],
            max_depth: Annotated[int, "Maximum depth for transitive impact analysis"] = 3,
        ) -> str:
            try:
                result = grover.impacts(path, max_depth=max_depth)
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "impacted files")

        return StructuredTool.from_function(
            name="impacts",
            description=(
                "Show all files transitively affected if this file changes. "
                "Follows the dependency graph up to max_depth levels deep. "
                "Use this before making changes to understand the blast radius."
            ),
            func=impacts,
        )

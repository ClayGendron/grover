"""GroverMiddleware — deepagents AgentMiddleware exposing Grover-specific tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool, StructuredTool

from grover.models.internal.results import FileSearchSet

if TYPE_CHECKING:
    from grover.client import Grover, GroverAsync


# ------------------------------------------------------------------
# String formatters for tool return values
# ------------------------------------------------------------------


def _format_graph_result(result: object, label: str) -> str:
    """Format a graph FileSearchResult into a readable string."""
    from grover.models.internal.results import FileSearchResult

    if isinstance(result, FileSearchResult):
        if len(result) == 0:
            return f"No {label} found."
        lines = [f"Found {len(result)} {label}:"]
        lines.extend(f"  - {path}" for path in result.paths)
        return "\n".join(lines)
    # Fallback for any unexpected type
    return f"No {label} found."


def _format_delete_result(result: object, path: str) -> str:
    """Format a delete result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    return f"Deleted {path} (moved to trash)."


def _format_search_result(result: object, query: str) -> str:
    """Format a vector_search result."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    if len(result) == 0:  # type: ignore[arg-type]
        return f"No results found for: {query}"

    from grover.models.internal.evidence import VectorEvidence

    lines = [f"Search results for '{query}' ({len(result)} files):"]  # type: ignore[arg-type]
    for i, f in enumerate(result.files, 1):  # type: ignore[union-attr]
        lines.append(f"  {i}. {f.path}")
        snippets = [ev.snippet for ev in f.evidence if isinstance(ev, VectorEvidence) and ev.snippet]
        lines.extend(f"     {s.replace(chr(10), ' ')}" for s in snippets[:3])
    return "\n".join(lines)


# ------------------------------------------------------------------
# GroverMiddleware
# ------------------------------------------------------------------


class GroverMiddleware(AgentMiddleware):
    """deepagents middleware exposing Grover search, graph, and delete tools.

    Adds tools beyond standard file operations: semantic search,
    dependency graph queries, and soft-delete.

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
            self._create_delete_file_tool(),
        ]
        if enable_search:
            tool_list.append(self._create_search_semantic_tool())
        if enable_graph:
            tool_list.extend(
                [
                    self._create_successors_tool(),
                    self._create_predecessors_tool(),
                ]
            )
        self.tools: list[BaseTool] = tool_list

    # ------------------------------------------------------------------
    # Delete tool
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

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    def _create_search_semantic_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def search_semantic(
            query: Annotated[str, "Natural language search query describing what you're looking for"],
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
    # Graph tools
    # ------------------------------------------------------------------

    def _create_successors_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def successors(
            path: Annotated[str, "Absolute virtual path to the file"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(successors_async(path))
            assert grover_s is not None
            try:
                result = grover_s.successors(FileSearchSet.from_paths([path]))
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "successors")

        async def successors_async(path: str) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.successors(FileSearchSet.from_paths([path]))
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "successors")

        kwargs: dict[str, object] = {
            "name": "successors",
            "description": (
                "Show graph successors of this file — nodes it points to "
                "(outgoing edges). Returns the direct successor list from "
                "the knowledge graph."
            ),
            "func": successors,
        }
        if is_async:
            kwargs["coroutine"] = successors_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

    def _create_predecessors_tool(self) -> BaseTool:
        is_async = self._is_async
        grover_s = cast("Grover", self.grover) if not is_async else None
        grover_a = cast("GroverAsync", self.grover) if is_async else None

        def predecessors(
            path: Annotated[str, "Absolute virtual path to the file"],
        ) -> str:
            if grover_a is not None:
                return asyncio.run(predecessors_async(path))
            assert grover_s is not None
            try:
                result = grover_s.predecessors(FileSearchSet.from_paths([path]))
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "predecessors")

        async def predecessors_async(path: str) -> str:
            assert grover_a is not None
            try:
                result = await grover_a.predecessors(FileSearchSet.from_paths([path]))
            except Exception as e:
                return f"Error: {e}"
            return _format_graph_result(result, "predecessors")

        kwargs: dict[str, object] = {
            "name": "predecessors",
            "description": (
                "Show graph predecessors of this file — nodes with edges "
                "pointing to it (incoming edges). Useful for understanding "
                "the impact of changes."
            ),
            "func": predecessors,
        }
        if is_async:
            kwargs["coroutine"] = predecessors_async
        return StructuredTool.from_function(**kwargs)  # type: ignore[arg-type]

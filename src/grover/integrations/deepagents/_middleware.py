"""GroverMiddleware — deepagents AgentMiddleware exposing Grover-specific tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from grover._grover import Grover


# ------------------------------------------------------------------
# String formatters for tool return values
# ------------------------------------------------------------------


def _format_ref_list(refs: list, label: str) -> str:
    """Format a list of Ref objects into a readable string."""
    if not refs:
        return f"No {label} found."
    lines = [f"Found {len(refs)} {label}:"]
    for ref in refs:
        line = f"  - {ref.path}"
        if ref.version is not None:
            line += f" (v{ref.version})"
        if ref.line_start is not None:
            line += f" L{ref.line_start}"
            if ref.line_end is not None:
                line += f"-{ref.line_end}"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------
# GroverMiddleware
# ------------------------------------------------------------------


class GroverMiddleware(AgentMiddleware):
    """deepagents middleware exposing Grover version, search, graph, and trash tools.

    Adds tools beyond standard file operations: version history, semantic
    search, dependency graph queries, and soft-delete trash management.

    Usage::

        from grover import Grover
        from grover.integrations.deepagents import GroverMiddleware

        g = Grover()
        middleware = GroverMiddleware(g)
        # Pass middleware.tools to the agent
    """

    def __init__(
        self,
        grover: Grover,
        *,
        enable_search: bool = True,
        enable_graph: bool = True,
    ) -> None:
        self.grover = grover
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
        grover = self.grover

        def list_versions(
            path: Annotated[str, "Absolute virtual path to the file, e.g. /project/main.py"],
        ) -> str:
            try:
                result = grover.list_versions(path)
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            if not result.versions:
                return f"No versions found for {path}."

            lines = [f"Version history for {path} ({len(result.versions)} versions):"]
            for v in result.versions:
                line = f"  v{v.version}: {v.created_at:%Y-%m-%d %H:%M:%S}"
                line += f" | {v.size_bytes} bytes | hash={v.content_hash[:12]}"
                if v.created_by:
                    line += f" | by {v.created_by}"
                lines.append(line)
            return "\n".join(lines)

        return StructuredTool.from_function(
            name="list_versions",
            description=(
                "Show the version history of a file. Returns a list of versions "
                "with timestamps, sizes, hashes, and who made each change. "
                "Use this to understand how a file has evolved over time."
            ),
            func=list_versions,
        )

    def _create_get_version_content_tool(self) -> BaseTool:
        grover = self.grover

        def get_version_content(
            path: Annotated[str, "Absolute virtual path to the file"],
            version: Annotated[int, "Version number to retrieve (from list_versions output)"],
        ) -> str:
            try:
                result = grover.get_version_content(path, version)
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            if result.content is None:
                return f"Error: No content found for {path} v{version}."

            return f"Content of {path} v{version}:\n{result.content}"

        return StructuredTool.from_function(
            name="get_version_content",
            description=(
                "Read the content of a specific past version of a file. "
                "Use list_versions first to see available versions, then "
                "pass the version number here to read that version's content."
            ),
            func=get_version_content,
        )

    def _create_restore_version_tool(self) -> BaseTool:
        grover = self.grover

        def restore_version(
            path: Annotated[str, "Absolute virtual path to the file"],
            version: Annotated[int, "Version number to restore to"],
        ) -> str:
            try:
                result = grover.restore_version(path, version)
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            return (
                f"Restored {path} to v{result.restored_version}. "
                f"Current version is now v{result.current_version}."
            )

        return StructuredTool.from_function(
            name="restore_version",
            description=(
                "Restore a file to a previous version. This creates a new "
                "version with the content from the specified old version — "
                "it does not discard history. Use list_versions to find "
                "the version number, then restore to it."
            ),
            func=restore_version,
        )

    # ------------------------------------------------------------------
    # Trash tools
    # ------------------------------------------------------------------

    def _create_delete_file_tool(self) -> BaseTool:
        grover = self.grover

        def delete_file(
            path: Annotated[str, "Absolute virtual path to the file to delete"],
        ) -> str:
            try:
                result = grover.delete(path)
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            return f"Deleted {path} (moved to trash). Use restore_from_trash to recover it."

        return StructuredTool.from_function(
            name="delete_file",
            description=(
                "Soft-delete a file by moving it to trash. The file can be "
                "recovered later using restore_from_trash. This is safer "
                "than permanent deletion."
            ),
            func=delete_file,
        )

    def _create_list_trash_tool(self) -> BaseTool:
        grover = self.grover

        def list_trash() -> str:
            try:
                result = grover.list_trash()
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            if not result.entries:
                return "Trash is empty."

            lines = [f"Trash ({len(result.entries)} items):"]
            for entry in result.entries:
                line = f"  - {entry.path}"
                if entry.size_bytes is not None:
                    line += f" ({entry.size_bytes} bytes)"
                lines.append(line)
            return "\n".join(lines)

        return StructuredTool.from_function(
            name="list_trash",
            description=(
                "List all soft-deleted files in the trash. Shows file paths "
                "and sizes. Use restore_from_trash to recover a specific file."
            ),
            func=list_trash,
        )

    def _create_restore_from_trash_tool(self) -> BaseTool:
        grover = self.grover

        def restore_from_trash(
            path: Annotated[str, "Path of the trashed file to restore (from list_trash output)"],
        ) -> str:
            try:
                result = grover.restore_from_trash(path)
            except Exception as e:
                return f"Error: {e}"

            if not result.success:
                return f"Error: {result.message}"

            return f"Restored {result.file_path} from trash."

        return StructuredTool.from_function(
            name="restore_from_trash",
            description=(
                "Restore a previously deleted file from trash. Use list_trash "
                "first to see available files, then pass the path here."
            ),
            func=restore_from_trash,
        )

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    def _create_search_semantic_tool(self) -> BaseTool:
        grover = self.grover

        def search_semantic(
            query: Annotated[
                str, "Natural language search query describing what you're looking for"
            ],
            k: Annotated[int, "Maximum number of results to return"] = 10,
        ) -> str:
            try:
                results = grover.search(query, k=k)
            except Exception as e:
                return f"Error: {e}"

            if not results:
                return f"No results found for: {query}"

            lines = [f"Search results for '{query}' ({len(results)} matches):"]
            for i, sr in enumerate(results, 1):
                line = f"  {i}. {sr.ref.path} (score: {sr.score:.3f})"
                if sr.parent_path:
                    line += f" [chunk of {sr.parent_path}]"
                lines.append(line)
                # Show a snippet of the matched content
                snippet = sr.content[:200].replace("\n", " ")
                if len(sr.content) > 200:
                    snippet += "..."
                lines.append(f"     {snippet}")
            return "\n".join(lines)

        return StructuredTool.from_function(
            name="search_semantic",
            description=(
                "Search the codebase using semantic similarity. Finds files "
                "by meaning, not just text pattern. For example, searching "
                "'authentication logic' will find files about login, tokens, "
                "and sessions even if they don't contain the exact phrase. "
                "Results are ranked by relevance score (0-1, higher is better)."
            ),
            func=search_semantic,
        )

    # ------------------------------------------------------------------
    # Graph tools
    # ------------------------------------------------------------------

    def _create_dependencies_tool(self) -> BaseTool:
        grover = self.grover

        def dependencies(
            path: Annotated[str, "Absolute virtual path to the file"],
        ) -> str:
            try:
                refs = grover.dependencies(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_ref_list(refs, "dependencies")

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
                refs = grover.dependents(path)
            except Exception as e:
                return f"Error: {e}"
            return _format_ref_list(refs, "dependents")

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
                refs = grover.impacts(path, max_depth=max_depth)
            except Exception as e:
                return f"Error: {e}"
            return _format_ref_list(refs, "impacted files")

        return StructuredTool.from_function(
            name="impacts",
            description=(
                "Show all files transitively affected if this file changes. "
                "Follows the dependency graph up to max_depth levels deep. "
                "Use this before making changes to understand the blast radius."
            ),
            func=impacts,
        )

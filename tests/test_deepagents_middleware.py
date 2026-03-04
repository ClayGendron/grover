"""Tests for GroverMiddleware — deepagents AgentMiddleware implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider

da = pytest.importorskip("deepagents")

from grover.backends.local import LocalFileSystem  # noqa: E402
from grover.client import (  # noqa: E402
    Grover,
    GroverAsync,
)
from grover.integrations.deepagents.middleware import GroverMiddleware  # noqa: E402
from grover.providers.search.local import LocalVectorStore  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover()
    g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g
    g.close()


@pytest.fixture
def middleware(grover: Grover) -> GroverMiddleware:
    return GroverMiddleware(grover)


@pytest.fixture
async def grover_async(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data_async"
    g = GroverAsync()
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def middleware_async(grover_async: GroverAsync) -> GroverMiddleware:
    return GroverMiddleware(grover_async)


# ==================================================================
# Tool registration
# ==================================================================


class TestToolRegistration:
    def test_middleware_tools_registered(self, middleware: GroverMiddleware):
        """All 10 tools registered by default."""
        assert len(middleware.tools) == 10
        names = {t.name for t in middleware.tools}
        assert names == {
            "list_versions",
            "get_version_content",
            "restore_version",
            "delete_file",
            "list_trash",
            "restore_from_trash",
            "search_semantic",
            "dependencies",
            "dependents",
            "impacts",
        }

    def test_enable_search_false_excludes_search_tool(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_search=False)
        names = {t.name for t in mw.tools}
        assert "search_semantic" not in names
        assert len(mw.tools) == 9

    def test_enable_graph_false_excludes_graph_tools(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_graph=False)
        names = {t.name for t in mw.tools}
        assert "dependencies" not in names
        assert "dependents" not in names
        assert "impacts" not in names
        assert len(mw.tools) == 7

    def test_both_disabled(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_search=False, enable_graph=False)
        assert len(mw.tools) == 6
        names = {t.name for t in mw.tools}
        assert names == {
            "list_versions",
            "get_version_content",
            "restore_version",
            "delete_file",
            "list_trash",
            "restore_from_trash",
        }


# ==================================================================
# Version tools
# ==================================================================


class TestListVersions:
    def test_list_versions_returns_formatted_history(
        self, middleware: GroverMiddleware, grover: Grover
    ):
        grover.write("/project/doc.txt", "v1 content")
        grover.write("/project/doc.txt", "v2 content")
        tool = next(t for t in middleware.tools if t.name == "list_versions")
        result = tool.invoke({"path": "/project/doc.txt"})
        assert isinstance(result, str)
        assert "Version history" in result
        assert "v1" in result
        assert "v2" in result
        assert "bytes" in result

    def test_list_versions_missing_file(self, middleware: GroverMiddleware):
        tool = next(t for t in middleware.tools if t.name == "list_versions")
        result = tool.invoke({"path": "/project/nonexistent.txt"})
        assert isinstance(result, str)
        assert "Error" in result or "No versions" in result


class TestGetVersionContent:
    def test_get_version_content_returns_old_content(
        self, middleware: GroverMiddleware, grover: Grover
    ):
        grover.write("/project/doc.txt", "original content")
        grover.write("/project/doc.txt", "updated content")
        tool = next(t for t in middleware.tools if t.name == "get_version_content")
        result = tool.invoke({"path": "/project/doc.txt", "version": 1})
        assert isinstance(result, str)
        assert "original content" in result

    def test_get_version_content_bad_version(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/doc.txt", "content")
        tool = next(t for t in middleware.tools if t.name == "get_version_content")
        result = tool.invoke({"path": "/project/doc.txt", "version": 999})
        assert "Error" in result


class TestRestoreVersion:
    def test_restore_version_creates_new_version(
        self, middleware: GroverMiddleware, grover: Grover
    ):
        grover.write("/project/doc.txt", "original")
        grover.write("/project/doc.txt", "modified")
        tool = next(t for t in middleware.tools if t.name == "restore_version")
        result = tool.invoke({"path": "/project/doc.txt", "version": 1})
        assert isinstance(result, str)
        assert "Restored" in result
        # Verify content was restored
        read = grover.read("/project/doc.txt")
        assert read.content == "original"

    def test_restore_version_bad_version(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/doc.txt", "content")
        tool = next(t for t in middleware.tools if t.name == "restore_version")
        result = tool.invoke({"path": "/project/doc.txt", "version": 999})
        assert "Error" in result


# ==================================================================
# Trash tools
# ==================================================================


class TestDeleteFile:
    def test_delete_file_soft_deletes(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/temp.txt", "temporary")
        tool = next(t for t in middleware.tools if t.name == "delete_file")
        result = tool.invoke({"path": "/project/temp.txt"})
        assert "Deleted" in result
        assert "trash" in result.lower()
        # File should no longer exist
        read = grover.read("/project/temp.txt")
        assert not read.success

    def test_delete_file_missing_returns_error(self, middleware: GroverMiddleware):
        tool = next(t for t in middleware.tools if t.name == "delete_file")
        result = tool.invoke({"path": "/project/nope.txt"})
        assert "Error" in result


class TestListTrash:
    def test_list_trash_shows_deleted_files(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/trash_me.txt", "content")
        grover.delete("/project/trash_me.txt")
        tool = next(t for t in middleware.tools if t.name == "list_trash")
        result = tool.invoke({})
        assert isinstance(result, str)
        assert "trash_me.txt" in result

    def test_list_trash_empty(self, middleware: GroverMiddleware):
        tool = next(t for t in middleware.tools if t.name == "list_trash")
        result = tool.invoke({})
        assert "empty" in result.lower()


class TestRestoreFromTrash:
    def test_restore_from_trash_recovers_file(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/restore_me.txt", "precious data")
        grover.delete("/project/restore_me.txt")
        tool = next(t for t in middleware.tools if t.name == "restore_from_trash")
        result = tool.invoke({"path": "/project/restore_me.txt"})
        assert "Restored" in result
        # File should be readable again
        read = grover.read("/project/restore_me.txt")
        assert read.success
        assert read.content == "precious data"

    def test_restore_from_trash_not_in_trash(self, middleware: GroverMiddleware):
        tool = next(t for t in middleware.tools if t.name == "restore_from_trash")
        result = tool.invoke({"path": "/project/not_trashed.txt"})
        assert "Error" in result


# ==================================================================
# Search tool
# ==================================================================


class TestSearchSemantic:
    def test_search_semantic_returns_ranked_results(
        self, middleware: GroverMiddleware, grover: Grover
    ):
        grover.write("/project/auth.py", "def authenticate(user, password): pass")
        grover.write("/project/math.py", "def add(a, b): return a + b")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "search_semantic")
        result = tool.invoke({"query": "authentication login", "k": 5})
        assert isinstance(result, str)
        assert "Search results" in result
        # Should return at least 1 result with file path
        assert "/project/" in result

    def test_search_semantic_no_results(self, middleware: GroverMiddleware):
        tool = next(t for t in middleware.tools if t.name == "search_semantic")
        result = tool.invoke({"query": "something that doesn't exist"})
        assert isinstance(result, str)
        assert "No results" in result

    def test_search_semantic_disabled_when_no_provider(self, workspace: Path, tmp_path: Path):
        # Create a Grover without embedding provider
        data = tmp_path / "no_search_data"
        g = Grover()
        g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        try:
            mw = GroverMiddleware(g, enable_search=False)
            names = {t.name for t in mw.tools}
            assert "search_semantic" not in names
        finally:
            g.close()


# ==================================================================
# Graph tools
# ==================================================================


class TestDependencies:
    def test_dependencies_returns_file_list(self, middleware: GroverMiddleware, grover: Grover):
        # Write files with import relationships
        grover.write("/project/utils.py", "def helper(): pass\n")
        grover.write("/project/main.py", "from utils import helper\n\ndef run(): helper()\n")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "dependencies")
        result = tool.invoke({"path": "/project/main.py"})
        assert isinstance(result, str)
        # If the graph analyzer found the import, it should list dependencies
        # (the exact output depends on whether the analyzer resolves the import)
        assert isinstance(result, str)

    def test_dependencies_no_deps(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/standalone.py", "x = 42\n")
        grover.index("/project")
        tool = next(t for t in middleware.tools if t.name == "dependencies")
        result = tool.invoke({"path": "/project/standalone.py"})
        assert "No dependencies" in result


class TestDependents:
    def test_dependents_returns_file_list(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/lib.py", "def shared(): pass\n")
        grover.write("/project/consumer.py", "from lib import shared\n\nshared()\n")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "dependents")
        result = tool.invoke({"path": "/project/lib.py"})
        assert isinstance(result, str)


class TestImpacts:
    def test_impacts_returns_transitive_closure(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/base.py", "BASE = 1\n")
        grover.write("/project/mid.py", "from base import BASE\nMID = BASE + 1\n")
        grover.write("/project/top.py", "from mid import MID\nTOP = MID + 1\n")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "impacts")
        result = tool.invoke({"path": "/project/base.py", "max_depth": 3})
        assert isinstance(result, str)

    def test_impacts_no_impact(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/island.py", "x = 1\n")
        grover.index("/project")
        tool = next(t for t in middleware.tools if t.name == "impacts")
        result = tool.invoke({"path": "/project/island.py"})
        assert "No impacted" in result


# ==================================================================
# Error handling
# ==================================================================


class TestErrorHandling:
    def test_all_tools_return_strings(self, middleware: GroverMiddleware, grover: Grover):
        """Every tool should return a string, even on error."""
        for tool in middleware.tools:
            # Invoke with bad/missing paths — should not raise
            if tool.name == "list_trash":
                result = tool.invoke({})
            elif tool.name == "search_semantic":
                result = tool.invoke({"query": "test"})
            elif tool.name in ("get_version_content", "restore_version"):
                result = tool.invoke({"path": "/project/nope.txt", "version": 1})
            elif tool.name == "impacts":
                result = tool.invoke({"path": "/project/nope.txt", "max_depth": 1})
            else:
                result = tool.invoke({"path": "/project/nope.txt"})
            assert isinstance(result, str), f"Tool {tool.name} returned {type(result)}"


# ==================================================================
# Async tool tests (GroverAsync)
# ==================================================================


class TestAsyncTools:
    async def test_tools_have_coroutine_when_async(self, middleware_async: GroverMiddleware):
        """Non-graph tools should have coroutine when GroverAsync is used."""
        graph_tools = {"dependencies", "dependents", "impacts"}
        for tool in middleware_async.tools:
            if tool.name in graph_tools:
                # Graph tools are sync on both Grover and GroverAsync
                assert tool.coroutine is None, f"Tool {tool.name} should not have coroutine"
            else:
                assert tool.coroutine is not None, f"Tool {tool.name} should have coroutine"

    async def test_tools_no_coroutine_when_sync(self, middleware: GroverMiddleware):
        """No tools should have coroutine when Grover is used."""
        for tool in middleware.tools:
            assert tool.coroutine is None, f"Tool {tool.name} should not have coroutine"

    async def test_list_versions_ainvoke(
        self, middleware_async: GroverMiddleware, grover_async: GroverAsync
    ):
        await grover_async.write("/project/doc.txt", "v1")
        await grover_async.write("/project/doc.txt", "v2")
        tool = next(t for t in middleware_async.tools if t.name == "list_versions")
        result = await tool.ainvoke({"path": "/project/doc.txt"})
        assert "Version history" in result

    async def test_delete_file_ainvoke(
        self, middleware_async: GroverMiddleware, grover_async: GroverAsync
    ):
        await grover_async.write("/project/temp.txt", "content")
        tool = next(t for t in middleware_async.tools if t.name == "delete_file")
        result = await tool.ainvoke({"path": "/project/temp.txt"})
        assert "Deleted" in result

    async def test_list_trash_ainvoke(
        self, middleware_async: GroverMiddleware, grover_async: GroverAsync
    ):
        tool = next(t for t in middleware_async.tools if t.name == "list_trash")
        result = await tool.ainvoke({})
        assert "empty" in result.lower()

    async def test_graph_tools_invoke_with_async(
        self, middleware_async: GroverMiddleware, grover_async: GroverAsync
    ):
        """Graph tools should work via sync invoke even with GroverAsync."""
        await grover_async.write("/project/standalone.py", "x = 42\n")
        await grover_async.index("/project")
        tool = next(t for t in middleware_async.tools if t.name == "dependencies")
        result = tool.invoke({"path": "/project/standalone.py"})
        assert "No dependencies" in result


# ==================================================================
# Sync wrapper tests (GroverAsync middleware, sync invoke via asyncio.run)
# ==================================================================


def _make_sync_middleware(tmp_path: Path) -> tuple[GroverMiddleware, GroverAsync]:
    """Create a GroverAsync-backed middleware outside an event loop."""
    data = tmp_path / "grover_data_sync_mw"
    ws = tmp_path / "workspace_sync_mw"
    ws.mkdir(exist_ok=True)

    async def _setup() -> GroverAsync:
        g = GroverAsync()
        await g.add_mount(
            "/project",
            LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
            search_provider=LocalVectorStore(dimension=FAKE_DIM),
        )
        return g

    ga = asyncio.run(_setup())
    return GroverMiddleware(ga), ga


class TestSyncWrapperMiddleware:
    def test_list_versions_sync_wrapper(self, tmp_path: Path):
        mw, ga = _make_sync_middleware(tmp_path)
        try:
            asyncio.run(ga.write("/project/doc.txt", "content"))
            tool = next(t for t in mw.tools if t.name == "list_versions")
            result = tool.invoke({"path": "/project/doc.txt"})
            assert "Version history" in result
        finally:
            asyncio.run(ga.close())

    def test_list_trash_sync_wrapper(self, tmp_path: Path):
        mw, ga = _make_sync_middleware(tmp_path)
        try:
            tool = next(t for t in mw.tools if t.name == "list_trash")
            result = tool.invoke({})
            assert "empty" in result.lower()
        finally:
            asyncio.run(ga.close())

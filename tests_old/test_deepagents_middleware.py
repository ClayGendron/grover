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
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
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
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
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
        """All 4 tools registered by default."""
        assert len(middleware.tools) == 4
        names = {t.name for t in middleware.tools}
        assert names == {
            "delete_file",
            "search_semantic",
            "successors",
            "predecessors",
        }

    def test_enable_search_false_excludes_search_tool(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_search=False)
        names = {t.name for t in mw.tools}
        assert "search_semantic" not in names
        assert len(mw.tools) == 3

    def test_enable_graph_false_excludes_graph_tools(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_graph=False)
        names = {t.name for t in mw.tools}
        assert "successors" not in names
        assert "predecessors" not in names
        assert len(mw.tools) == 2

    def test_both_disabled(self, grover: Grover):
        mw = GroverMiddleware(grover, enable_search=False, enable_graph=False)
        assert len(mw.tools) == 1
        names = {t.name for t in mw.tools}
        assert names == {
            "delete_file",
        }


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


# ==================================================================
# Search tool
# ==================================================================


class TestSearchSemantic:
    def test_search_semantic_returns_ranked_results(self, middleware: GroverMiddleware, grover: Grover):
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
        g.add_mount("project", filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        try:
            mw = GroverMiddleware(g, enable_search=False)
            names = {t.name for t in mw.tools}
            assert "search_semantic" not in names
        finally:
            g.close()


# ==================================================================
# Graph tools
# ==================================================================


class TestSuccessors:
    def test_successors_returns_file_list(self, middleware: GroverMiddleware, grover: Grover):
        # Write files with import relationships
        grover.write("/project/utils.py", "def helper(): pass\n")
        grover.write("/project/main.py", "from utils import helper\n\ndef run(): helper()\n")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "successors")
        result = tool.invoke({"path": "/project/main.py"})
        assert isinstance(result, str)

    def test_successors_no_results(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/standalone.py", "x = 42\n")
        grover.index("/project")
        tool = next(t for t in middleware.tools if t.name == "successors")
        result = tool.invoke({"path": "/project/standalone.py"})
        assert "No successors" in result


class TestPredecessors:
    def test_predecessors_returns_file_list(self, middleware: GroverMiddleware, grover: Grover):
        grover.write("/project/lib.py", "def shared(): pass\n")
        grover.write("/project/consumer.py", "from lib import shared\n\nshared()\n")
        grover.index("/project")

        tool = next(t for t in middleware.tools if t.name == "predecessors")
        result = tool.invoke({"path": "/project/lib.py"})
        assert isinstance(result, str)


# ==================================================================
# Error handling
# ==================================================================


class TestErrorHandling:
    def test_all_tools_return_strings(self, middleware: GroverMiddleware, grover: Grover):
        """Every tool should return a string, even on error."""
        for tool in middleware.tools:
            # Invoke with bad/missing paths — should not raise
            if tool.name == "search_semantic":
                result = tool.invoke({"query": "test"})
            else:
                result = tool.invoke({"path": "/project/nope.txt"})
            assert isinstance(result, str), f"Tool {tool.name} returned {type(result)}"


# ==================================================================
# Async tool tests (GroverAsync)
# ==================================================================


class TestAsyncTools:
    async def test_tools_have_coroutine_when_async(self, middleware_async: GroverMiddleware):
        """All tools should have coroutine when GroverAsync is used."""
        for tool in middleware_async.tools:
            assert tool.coroutine is not None, f"Tool {tool.name} should have coroutine"

    async def test_tools_no_coroutine_when_sync(self, middleware: GroverMiddleware):
        """No tools should have coroutine when Grover is used."""
        for tool in middleware.tools:
            assert tool.coroutine is None, f"Tool {tool.name} should not have coroutine"

    async def test_delete_file_ainvoke(self, middleware_async: GroverMiddleware, grover_async: GroverAsync):
        await grover_async.write("/project/temp.txt", "content")
        tool = next(t for t in middleware_async.tools if t.name == "delete_file")
        result = await tool.ainvoke({"path": "/project/temp.txt"})
        assert "Deleted" in result

    async def test_graph_tools_invoke_with_async(self, middleware_async: GroverMiddleware, grover_async: GroverAsync):
        """Graph tools should work via async invoke with GroverAsync."""
        await grover_async.write("/project/standalone.py", "x = 42\n")
        await grover_async.index("/project")
        tool = next(t for t in middleware_async.tools if t.name == "successors")
        result = await tool.ainvoke({"path": "/project/standalone.py"})
        assert "No successors" in result


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
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
            search_provider=LocalVectorStore(dimension=FAKE_DIM),
        )
        return g

    ga = asyncio.run(_setup())
    return GroverMiddleware(ga), ga


class TestSyncWrapperMiddleware:
    pass

"""Tests for GroverLoader — LangChain BaseLoader implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

lc = pytest.importorskip("langchain_core")

from collections.abc import Iterator  # noqa: E402

from langchain_core.documents import Document  # noqa: E402

from grover.backends.local import LocalFileSystem  # noqa: E402
from grover.client import (  # noqa: E402
    Grover,
    GroverAsync,
)
from grover.integrations.langchain.loader import GroverLoader  # noqa: E402

if TYPE_CHECKING:
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
    g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def loader(grover: Grover) -> GroverLoader:
    return GroverLoader(grover=grover, path="/project")


@pytest.fixture
async def grover_async(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data_async"
    g = GroverAsync()
    await g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def loader_async(grover_async: GroverAsync) -> GroverLoader:
    return GroverLoader(grover=grover_async, path="/project")


# ==================================================================
# Sync tests (Grover)
# ==================================================================


class TestLoaderLoadsAllFiles:
    def test_loader_loads_all_files(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/a.txt", "content a")
        grover.write("/project/b.py", "print('b')")
        docs = loader.load()
        assert len(docs) == 2
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/a.txt" in paths
        assert "/project/b.py" in paths


class TestLoaderLazyLoadIsGenerator:
    def test_loader_lazy_load_is_generator(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/file.txt", "content")
        result = loader.lazy_load()
        assert isinstance(result, Iterator)


class TestLoaderGlobFilter:
    def test_loader_glob_filter(self, grover: Grover):
        grover.write("/project/code.py", "python code")
        grover.write("/project/readme.txt", "text")
        grover.write("/project/main.py", "more python")

        loader = GroverLoader(grover=grover, path="/project", glob_pattern="*.py")
        docs = loader.load()
        assert len(docs) == 2
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/code.py" in paths
        assert "/project/main.py" in paths
        assert "/project/readme.txt" not in paths


class TestLoaderDocumentMetadata:
    def test_loader_document_metadata(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/doc.txt", "hello world")
        docs = loader.load()
        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["path"] == "/project/doc.txt"
        assert doc.metadata["source"] == "/project/doc.txt"
        assert "size_bytes" in doc.metadata
        assert doc.id == "/project/doc.txt"
        assert doc.page_content == "hello world"


class TestLoaderEmptyDirectory:
    def test_loader_empty_directory(self, loader: GroverLoader):
        docs = loader.load()
        assert docs == []


class TestLoaderSkipsDirectories:
    def test_loader_skips_directories(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/sub/file.txt", "nested content")
        docs = loader.load()
        # Should only contain files, not directories
        for doc in docs:
            assert not doc.metadata["path"].endswith("/sub")
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/sub/file.txt" in paths


class TestLoaderNonRecursive:
    def test_loader_non_recursive(self, grover: Grover):
        grover.write("/project/top.txt", "top level")
        grover.write("/project/sub/nested.txt", "nested")

        loader = GroverLoader(grover=grover, path="/project", recursive=False)
        docs = loader.load()
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/top.txt" in paths
        # Nested file should NOT be included in non-recursive mode
        assert "/project/sub/nested.txt" not in paths


class TestLoaderSkipsBinaryFiles:
    def test_loader_skips_binary_files(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/code.py", "print('hi')")
        grover.write("/project/image.png", "fake binary")
        grover.write("/project/archive.zip", "fake zip")
        docs = loader.load()
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/code.py" in paths
        assert "/project/image.png" not in paths
        assert "/project/archive.zip" not in paths


class TestLoaderLoadMethod:
    def test_loader_load_method(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/file.txt", "content")
        result = loader.load()
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], Document)


# ==================================================================
# is_async flag
# ==================================================================


class TestIsAsyncFlag:
    def test_is_async_false_with_grover(self, loader: GroverLoader):
        assert loader._is_async is False

    async def test_is_async_true_with_grover_async(self, loader_async: GroverLoader):
        assert loader_async._is_async is True


# ==================================================================
# Async native tests (GroverAsync)
# ==================================================================


class TestLoaderAlazLoad:
    async def test_alazy_load_yields_documents(
        self, loader_async: GroverLoader, grover_async: GroverAsync
    ):
        await grover_async.write("/project/a.txt", "content a")
        await grover_async.write("/project/b.py", "print('b')")

        docs = [doc async for doc in loader_async.alazy_load()]
        assert len(docs) == 2
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/a.txt" in paths
        assert "/project/b.py" in paths

    async def test_alazy_load_glob_filter(self, grover_async: GroverAsync):
        await grover_async.write("/project/code.py", "python code")
        await grover_async.write("/project/readme.txt", "text")

        loader = GroverLoader(grover=grover_async, path="/project", glob_pattern="*.py")
        docs = [doc async for doc in loader.alazy_load()]
        assert len(docs) == 1
        assert docs[0].metadata["path"] == "/project/code.py"

    async def test_alazy_load_skips_binary(
        self, loader_async: GroverLoader, grover_async: GroverAsync
    ):
        await grover_async.write("/project/code.py", "print('hi')")
        await grover_async.write("/project/image.png", "fake binary")

        docs = [doc async for doc in loader_async.alazy_load()]
        paths = {doc.metadata["path"] for doc in docs}
        assert "/project/code.py" in paths
        assert "/project/image.png" not in paths


# ==================================================================
# TypeError when calling async with sync Grover
# ==================================================================


class TestLoaderAsyncTypeError:
    async def test_alazy_load_raises_type_error(self, loader: GroverLoader):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            async for _doc in loader.alazy_load():
                pass


# ==================================================================
# Sync wrapper tests (GroverAsync loader, sync methods via asyncio.run)
# ==================================================================


def _make_sync_loader(tmp_path: Path) -> tuple[GroverLoader, GroverAsync]:
    """Create a GroverAsync-backed loader outside an event loop."""
    data = tmp_path / "grover_data_sync_loader"
    ws = tmp_path / "workspace_sync_loader"
    ws.mkdir(exist_ok=True)

    async def _setup() -> GroverAsync:
        g = GroverAsync()
        await g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
        return g

    ga = asyncio.run(_setup())
    return GroverLoader(grover=ga, path="/project"), ga


class TestLoaderSyncWrapper:
    def test_lazy_load_sync_wrapper(self, tmp_path: Path):
        loader, ga = _make_sync_loader(tmp_path)
        try:
            asyncio.run(ga.write("/project/file.txt", "content"))
            docs = loader.load()
            assert len(docs) == 1
            assert docs[0].page_content == "content"
        finally:
            asyncio.run(ga.close())

    def test_lazy_load_glob_sync_wrapper(self, tmp_path: Path):
        _loader_base, ga = _make_sync_loader(tmp_path)
        try:
            asyncio.run(ga.write("/project/code.py", "python"))
            asyncio.run(ga.write("/project/readme.txt", "text"))
            loader = GroverLoader(grover=ga, path="/project", glob_pattern="*.py")
            docs = loader.load()
            assert len(docs) == 1
            assert docs[0].metadata["path"] == "/project/code.py"
        finally:
            asyncio.run(ga.close())

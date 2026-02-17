"""Tests for GroverLoader — LangChain BaseLoader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

lc = pytest.importorskip("langchain_core")

from collections.abc import Iterator  # noqa: E402

from langchain_core.documents import Document  # noqa: E402

from grover._grover import Grover  # noqa: E402
from grover.fs.local_fs import LocalFileSystem  # noqa: E402
from grover.integrations.langchain._loader import GroverLoader  # noqa: E402

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
    g = Grover(data_dir=str(data))
    g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def loader(grover: Grover) -> GroverLoader:
    return GroverLoader(grover=grover, path="/project")


# ==================================================================
# Tests
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


class TestLoaderLoadMethod:
    def test_loader_load_method(self, loader: GroverLoader, grover: Grover):
        grover.write("/project/file.txt", "content")
        result = loader.load()
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], Document)

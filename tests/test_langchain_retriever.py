"""Tests for GroverRetriever — LangChain BaseRetriever implementation."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

lc = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402

from grover._grover import Grover  # noqa: E402
from grover.fs.local_fs import LocalFileSystem  # noqa: E402
from grover.integrations.langchain._retriever import GroverRetriever  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider (deterministic, fast)
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def grover_with_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
    g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def grover_no_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data_nosearch"
    g = Grover(data_dir=str(data))
    g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def retriever(grover_with_search: Grover) -> GroverRetriever:
    return GroverRetriever(grover=grover_with_search, k=10)


# ==================================================================
# Tests
# ==================================================================


class TestRetrieverReturnsDocuments:
    def test_retriever_returns_documents(
        self, retriever: GroverRetriever, grover_with_search: Grover
    ):
        grover_with_search.write("/project/auth.py", "def authenticate(user, password): pass")
        grover_with_search.index()

        docs = retriever._get_relevant_documents(
            "authenticate",
            run_manager=_fake_run_manager(),
        )
        assert isinstance(docs, list)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0


class TestRetrieverDocumentMetadata:
    def test_retriever_document_metadata(
        self, retriever: GroverRetriever, grover_with_search: Grover
    ):
        grover_with_search.write("/project/utils.py", "def helper(): return 42")
        grover_with_search.index()

        docs = retriever._get_relevant_documents(
            "helper",
            run_manager=_fake_run_manager(),
        )
        assert len(docs) > 0
        doc = docs[0]
        assert "path" in doc.metadata
        assert "score" in doc.metadata
        assert isinstance(doc.metadata["score"], float)
        assert doc.id is not None


class TestRetrieverEmptyResults:
    def test_retriever_empty_results(self, retriever: GroverRetriever):
        # No files indexed — search returns empty
        docs = retriever._get_relevant_documents(
            "nonexistent_query_xyz",
            run_manager=_fake_run_manager(),
        )
        assert isinstance(docs, list)
        assert len(docs) == 0


class TestRetrieverInvokeInterface:
    def test_retriever_invoke_interface(
        self, retriever: GroverRetriever, grover_with_search: Grover
    ):
        grover_with_search.write("/project/main.py", "print('hello world')")
        grover_with_search.index()

        # .invoke() is the Runnable interface
        docs = retriever.invoke("hello")
        assert isinstance(docs, list)
        for doc in docs:
            assert isinstance(doc, Document)


class TestRetrieverKParameter:
    def test_retriever_k_parameter(self, grover_with_search: Grover):
        # Write multiple files
        for i in range(5):
            grover_with_search.write(f"/project/file{i}.py", f"def func{i}(): return {i}")
        grover_with_search.index()

        retriever_k2 = GroverRetriever(grover=grover_with_search, k=2)
        docs = retriever_k2.invoke("func")
        assert len(docs) <= 2


class TestRetrieverNoSearchIndex:
    def test_retriever_no_search_index_returns_empty(self, grover_no_search: Grover):
        grover_no_search.write("/project/file.py", "content")
        retriever = GroverRetriever(grover=grover_no_search, k=5)
        docs = retriever.invoke("content")
        assert isinstance(docs, list)
        assert len(docs) == 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fake_run_manager():
    """Create a minimal fake CallbackManagerForRetrieverRun."""
    from unittest.mock import MagicMock

    return MagicMock()

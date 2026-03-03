"""Tests for GroverRetriever — LangChain BaseRetriever implementation."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import TYPE_CHECKING

import pytest

lc = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402

from grover._grover import Grover  # noqa: E402
from grover._grover_async import GroverAsync  # noqa: E402
from grover.fs.local_fs import LocalFileSystem  # noqa: E402
from grover.fs.providers.search.local import LocalVectorStore  # noqa: E402
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
    g = Grover()
    g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=_FAKE_DIM),
    )
    yield g
    g.close()


@pytest.fixture
def grover_no_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data_nosearch"
    g = Grover()
    g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def retriever(grover_with_search: Grover) -> GroverRetriever:
    return GroverRetriever(grover=grover_with_search, k=10)


@pytest.fixture
async def grover_async_with_search(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data_async"
    g = GroverAsync()
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=_FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def retriever_async(grover_async_with_search: GroverAsync) -> GroverRetriever:
    return GroverRetriever(grover=grover_async_with_search, k=10)


# ==================================================================
# Sync tests (Grover)
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
        assert isinstance(doc.metadata["path"], str)
        assert doc.id is not None
        assert isinstance(doc.id, str)


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


# ==================================================================
# is_async flag
# ==================================================================


class TestIsAsyncFlag:
    def test_is_async_false_with_grover(self, retriever: GroverRetriever):
        assert retriever._is_async is False

    async def test_is_async_true_with_grover_async(self, retriever_async: GroverRetriever):
        assert retriever_async._is_async is True


# ==================================================================
# Async native tests (GroverAsync)
# ==================================================================


class TestRetrieverAsync:
    async def test_aget_relevant_documents(
        self, retriever_async: GroverRetriever, grover_async_with_search: GroverAsync
    ):
        await grover_async_with_search.write(
            "/project/auth.py", "def authenticate(user, password): pass"
        )
        await grover_async_with_search.index()

        docs = await retriever_async._aget_relevant_documents(
            "authenticate",
            run_manager=None,
        )
        assert isinstance(docs, list)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)

    async def test_ainvoke(
        self, retriever_async: GroverRetriever, grover_async_with_search: GroverAsync
    ):
        await grover_async_with_search.write("/project/main.py", "print('hello world')")
        await grover_async_with_search.index()

        docs = await retriever_async.ainvoke("hello")
        assert isinstance(docs, list)


# ==================================================================
# TypeError when calling async with sync Grover
# ==================================================================


class TestRetrieverAsyncTypeError:
    async def test_aget_raises_type_error(self, retriever: GroverRetriever):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await retriever._aget_relevant_documents("query", run_manager=None)


# ==================================================================
# Sync wrapper tests (GroverAsync retriever, sync methods via asyncio.run)
# ==================================================================


def _make_sync_retriever(tmp_path: Path) -> tuple[GroverRetriever, GroverAsync]:
    """Create a GroverAsync-backed retriever outside an event loop."""
    data = tmp_path / "grover_data_sync_ret"
    ws = tmp_path / "workspace_sync_ret"
    ws.mkdir(exist_ok=True)

    async def _setup() -> GroverAsync:
        g = GroverAsync()
        await g.add_mount(
            "/project",
            LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
            search_provider=LocalVectorStore(dimension=_FAKE_DIM),
        )
        return g

    ga = asyncio.run(_setup())
    return GroverRetriever(grover=ga, k=10), ga


class TestRetrieverSyncWrapper:
    def test_get_relevant_documents_sync_wrapper(self, tmp_path: Path):
        retriever, ga = _make_sync_retriever(tmp_path)
        try:
            asyncio.run(ga.write("/project/auth.py", "def authenticate(): pass"))
            asyncio.run(ga.index())
            docs = retriever._get_relevant_documents("authenticate", run_manager=None)
            assert isinstance(docs, list)
            assert len(docs) > 0
        finally:
            asyncio.run(ga.close())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fake_run_manager():
    """Create a minimal fake CallbackManagerForRetrieverRun."""
    from unittest.mock import MagicMock

    return MagicMock()

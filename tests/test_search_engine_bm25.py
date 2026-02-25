"""Integration tests — SearchEngine with lexical (BM25) store."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from grover.search._engine import SearchEngine
from grover.search.extractors import EmbeddableChunk
from grover.search.fulltext.sqlite import SQLiteFullTextStore
from grover.search.fulltext.types import FullTextResult

# ==================================================================
# Fixtures
# ==================================================================

_FAKE_DIM = 4


class FakeEmbeddingProvider:
    """Minimal embedding provider for testing."""

    @property
    def dimensions(self):
        return _FAKE_DIM

    @property
    def model_name(self):
        return "fake"

    def embed(self, text: str) -> list[float]:
        return [0.1] * _FAKE_DIM

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _FAKE_DIM for _ in texts]


@pytest.fixture
async def bm25_engine():
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    yield eng
    await eng.dispose()


@pytest.fixture
async def bm25_fts(bm25_engine) -> SQLiteFullTextStore:
    store = SQLiteFullTextStore(engine=bm25_engine)
    await store.ensure_table()
    return store


@pytest.fixture
async def bm25_session(bm25_engine):
    factory = async_sessionmaker(
        bm25_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session


@pytest.fixture
def search_engine_lexical_only(bm25_fts) -> SearchEngine:
    """SearchEngine with only a lexical store (no vector/embedding)."""
    return SearchEngine(lexical=bm25_fts)


@pytest.fixture
def search_engine_full(bm25_fts) -> SearchEngine:
    """SearchEngine with vector, embedding, and lexical."""
    from grover.search.stores.local import LocalVectorStore

    return SearchEngine(
        vector=LocalVectorStore(dimension=_FAKE_DIM),
        embedding=FakeEmbeddingProvider(),
        lexical=bm25_fts,
    )


# ==================================================================
# Lexical-only engine
# ==================================================================


class TestLexicalOnlyEngine:
    @pytest.mark.asyncio
    async def test_add_indexes_in_fts(self, search_engine_lexical_only, bm25_session):
        await search_engine_lexical_only.add(
            "/auth.py", "login authentication", session=bm25_session
        )
        await bm25_session.commit()

        results = await search_engine_lexical_only.lexical_search("login", session=bm25_session)
        assert len(results) == 1
        assert results[0].path == "/auth.py"

    @pytest.mark.asyncio
    async def test_add_batch_indexes_in_fts(self, search_engine_lexical_only, bm25_session):
        entries = [
            EmbeddableChunk(path="/a.py", content="login function"),
            EmbeddableChunk(path="/b.py", content="database pool"),
        ]
        await search_engine_lexical_only.add_batch(entries, session=bm25_session)
        await bm25_session.commit()

        results = await search_engine_lexical_only.lexical_search("login", session=bm25_session)
        assert len(results) == 1
        assert results[0].path == "/a.py"

        results = await search_engine_lexical_only.lexical_search("database", session=bm25_session)
        assert len(results) == 1
        assert results[0].path == "/b.py"

    @pytest.mark.asyncio
    async def test_remove_clears_fts(self, search_engine_lexical_only, bm25_session):
        await search_engine_lexical_only.add("/a.py", "hello world", session=bm25_session)
        await bm25_session.commit()

        await search_engine_lexical_only.remove("/a.py", session=bm25_session)
        await bm25_session.commit()

        results = await search_engine_lexical_only.lexical_search("hello", session=bm25_session)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_remove_file_clears_fts_with_chunks(
        self, search_engine_lexical_only, bm25_session
    ):
        await search_engine_lexical_only.add("/auth.py", "main content", session=bm25_session)
        await search_engine_lexical_only.add(
            "/auth.py#login", "login function", session=bm25_session
        )
        await search_engine_lexical_only.add(
            "/auth.py#logout", "logout function", session=bm25_session
        )
        await bm25_session.commit()

        await search_engine_lexical_only.remove_file("/auth.py", session=bm25_session)
        await bm25_session.commit()

        results = await search_engine_lexical_only.lexical_search("login", session=bm25_session)
        assert len(results) == 0
        results = await search_engine_lexical_only.lexical_search("logout", session=bm25_session)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_lexical_search_raises_without_lexical(self):
        engine = SearchEngine()
        with pytest.raises(RuntimeError, match="no lexical store configured"):
            await engine.lexical_search("query")


# ==================================================================
# Full engine (vector + embedding + lexical)
# ==================================================================


class TestFullEngine:
    @pytest.mark.asyncio
    async def test_add_indexes_both_vector_and_fts(self, search_engine_full, bm25_session):
        await search_engine_full.add("/auth.py", "login authentication", session=bm25_session)
        await bm25_session.commit()

        # Lexical search works
        lex_results = await search_engine_full.lexical_search("login", session=bm25_session)
        assert len(lex_results) == 1

        # Vector search also works
        vec_results = await search_engine_full.search("login", k=5)
        assert len(vec_results) == 1

    @pytest.mark.asyncio
    async def test_add_batch_indexes_both(self, search_engine_full, bm25_session):
        entries = [
            EmbeddableChunk(path="/a.py", content="login function"),
            EmbeddableChunk(path="/b.py", content="database pool"),
        ]
        await search_engine_full.add_batch(entries, session=bm25_session)
        await bm25_session.commit()

        # Lexical
        lex = await search_engine_full.lexical_search("login", session=bm25_session)
        assert len(lex) == 1

        # Vector
        vec = await search_engine_full.search("anything", k=10)
        assert len(vec) == 2

    @pytest.mark.asyncio
    async def test_remove_clears_both(self, search_engine_full, bm25_session):
        await search_engine_full.add("/a.py", "hello world", session=bm25_session)
        await bm25_session.commit()

        await search_engine_full.remove("/a.py", session=bm25_session)
        await bm25_session.commit()

        lex = await search_engine_full.lexical_search("hello", session=bm25_session)
        assert len(lex) == 0

        vec = await search_engine_full.search("hello", k=5)
        assert len(vec) == 0

    @pytest.mark.asyncio
    async def test_remove_file_clears_both(self, search_engine_full, bm25_session):
        await search_engine_full.add("/auth.py", "main content", session=bm25_session)
        await search_engine_full.add(
            "/auth.py#login",
            "login function",
            parent_path="/auth.py",
            session=bm25_session,
        )
        await bm25_session.commit()

        await search_engine_full.remove_file("/auth.py", session=bm25_session)
        await bm25_session.commit()

        lex = await search_engine_full.lexical_search("login", session=bm25_session)
        assert len(lex) == 0


# ==================================================================
# Lexical property and supported_protocols
# ==================================================================


class TestLexicalProperty:
    def test_lexical_property_returns_store(self, search_engine_lexical_only, bm25_fts):
        assert search_engine_lexical_only.lexical is bm25_fts

    def test_lexical_property_none_when_not_configured(self):
        engine = SearchEngine()
        assert engine.lexical is None

    def test_supported_protocols_includes_lexical(self, search_engine_lexical_only):
        from grover.mount.protocols import SupportsLexicalSearch

        protos = search_engine_lexical_only.supported_protocols()
        assert SupportsLexicalSearch in protos

    def test_supported_protocols_full_engine(self, search_engine_full):
        from grover.mount.protocols import (
            SupportsEmbedding,
            SupportsLexicalSearch,
            SupportsVectorSearch,
        )

        protos = search_engine_full.supported_protocols()
        assert SupportsVectorSearch in protos
        assert SupportsLexicalSearch in protos
        assert SupportsEmbedding in protos


# ==================================================================
# Edge cases
# ==================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_add_without_session_skips_fts(self, search_engine_lexical_only):
        # Should not raise — just skips FTS indexing
        await search_engine_lexical_only.add("/a.py", "content")

    @pytest.mark.asyncio
    async def test_add_batch_empty_list(self, search_engine_lexical_only, bm25_session):
        # Empty list — early return
        await search_engine_lexical_only.add_batch([], session=bm25_session)

    @pytest.mark.asyncio
    async def test_lexical_search_returns_fulltext_results(
        self, search_engine_lexical_only, bm25_session
    ):
        await search_engine_lexical_only.add("/a.py", "test content here", session=bm25_session)
        await bm25_session.commit()

        results = await search_engine_lexical_only.lexical_search("test", session=bm25_session)
        assert all(isinstance(r, FullTextResult) for r in results)

    @pytest.mark.asyncio
    async def test_connect_close_with_no_vector_store(self, search_engine_lexical_only):
        # Lexical-only engine has no vector store — connect/close must not crash
        await search_engine_lexical_only.connect()
        await search_engine_lexical_only.close()

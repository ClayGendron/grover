"""Tests for SQLiteFullTextStore — FTS5 keyword search."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from grover.search.fulltext.sqlite import SQLiteFullTextStore
from grover.search.fulltext.types import FullTextResult


@pytest.fixture
async def fts_engine():
    """Async in-memory SQLite engine for FTS tests."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    yield eng
    await eng.dispose()


@pytest.fixture
async def fts_store(fts_engine) -> SQLiteFullTextStore:
    """SQLiteFullTextStore with table created."""
    store = SQLiteFullTextStore(engine=fts_engine)
    await store.ensure_table()
    return store


@pytest.fixture
async def fts_session(fts_engine):
    """Async session for FTS tests."""
    factory = async_sessionmaker(
        fts_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session


# ==================================================================
# Table creation
# ==================================================================


class TestTableCreation:
    @pytest.mark.asyncio
    async def test_ensure_table_creates_fts5_table(self, fts_engine):
        store = SQLiteFullTextStore(engine=fts_engine)
        await store.ensure_table()
        # Table should exist — verify by inserting
        factory = async_sessionmaker(fts_engine, class_=AsyncSession)
        async with factory() as sess:
            await store.index("/test.py", "hello", session=sess)
            await sess.commit()

    @pytest.mark.asyncio
    async def test_ensure_table_idempotent(self, fts_engine):
        store = SQLiteFullTextStore(engine=fts_engine)
        await store.ensure_table()
        await store.ensure_table()  # Should not raise

    @pytest.mark.asyncio
    async def test_ensure_table_skips_second_call(self, fts_engine):
        store = SQLiteFullTextStore(engine=fts_engine)
        await store.ensure_table()
        assert store._table_created is True
        await store.ensure_table()  # Short-circuits

    @pytest.mark.asyncio
    async def test_ensure_table_no_engine_noop(self):
        store = SQLiteFullTextStore()
        await store.ensure_table()  # No engine — noop
        assert not store._table_created

    @pytest.mark.asyncio
    async def test_custom_table_name(self, fts_engine):
        store = SQLiteFullTextStore(engine=fts_engine, table_name="custom_fts")
        await store.ensure_table()
        factory = async_sessionmaker(fts_engine, class_=AsyncSession)
        async with factory() as sess:
            await store.index("/test.py", "content", session=sess)
            await sess.commit()
            results = await store.search("content", session=sess)
            assert len(results) == 1


# ==================================================================
# Index
# ==================================================================


class TestIndex:
    @pytest.mark.asyncio
    async def test_index_single_document(self, fts_store, fts_session):
        await fts_store.index("/hello.py", "hello world", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("hello", session=fts_session)
        assert len(results) == 1
        assert results[0].path == "/hello.py"

    @pytest.mark.asyncio
    async def test_index_replaces_existing(self, fts_store, fts_session):
        await fts_store.index("/a.py", "old content", session=fts_session)
        await fts_session.commit()
        await fts_store.index("/a.py", "new content", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("old", session=fts_session)
        assert len(results) == 0
        results = await fts_store.search("new", session=fts_session)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_index_multiple_documents(self, fts_store, fts_session):
        await fts_store.index("/a.py", "authentication login", session=fts_session)
        await fts_store.index("/b.py", "database connection pool", session=fts_session)
        await fts_store.index("/c.py", "authentication tokens", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("authentication", session=fts_session)
        assert len(results) == 2
        paths = {r.path for r in results}
        assert paths == {"/a.py", "/c.py"}

    @pytest.mark.asyncio
    async def test_index_no_session_noop(self, fts_store):
        # Should not raise when session is None
        await fts_store.index("/a.py", "content")

    @pytest.mark.asyncio
    async def test_index_empty_content(self, fts_store, fts_session):
        await fts_store.index("/empty.py", "", session=fts_session)
        await fts_session.commit()
        # Empty content is indexed — searching for unrelated term won't match
        results = await fts_store.search("nonexistent", session=fts_session)
        assert len(results) == 0


# ==================================================================
# Search
# ==================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_fulltext_results(self, fts_store, fts_session):
        await fts_store.index("/a.py", "hello world function", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("hello", session=fts_session)
        assert len(results) == 1
        assert isinstance(results[0], FullTextResult)
        assert results[0].path == "/a.py"
        assert isinstance(results[0].rank, float)

    @pytest.mark.asyncio
    async def test_search_no_session_returns_empty(self, fts_store):
        results = await fts_store.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self, fts_store, fts_session):
        await fts_store.index("/a.py", "hello world", session=fts_session)
        await fts_session.commit()
        results = await fts_store.search("", session=fts_session)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_whitespace_query_returns_empty(self, fts_store, fts_session):
        await fts_store.index("/a.py", "hello world", session=fts_session)
        await fts_session.commit()
        results = await fts_store.search("   ", session=fts_session)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_match(self, fts_store, fts_session):
        await fts_store.index("/a.py", "hello world", session=fts_session)
        await fts_session.commit()
        results = await fts_store.search("zzzzunmatched", session=fts_session)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_k(self, fts_store, fts_session):
        for i in range(5):
            await fts_store.index(f"/file{i}.py", f"keyword content {i}", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("keyword", k=2, session=fts_session)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_snippets(self, fts_store, fts_session):
        await fts_store.index(
            "/a.py",
            "the quick brown fox jumps over the lazy dog",
            session=fts_session,
        )
        await fts_session.commit()

        results = await fts_store.search("fox", session=fts_session)
        assert len(results) == 1
        assert results[0].snippet  # Should have a snippet

    @pytest.mark.asyncio
    async def test_search_ranking(self, fts_store, fts_session):
        # Document with more keyword occurrences should rank differently
        await fts_store.index("/few.py", "login once", session=fts_session)
        await fts_store.index(
            "/many.py",
            "login login login authentication login",
            session=fts_session,
        )
        await fts_session.commit()

        results = await fts_store.search("login", session=fts_session)
        assert len(results) == 2
        # Both results should have rank values
        for r in results:
            assert isinstance(r.rank, float)

    @pytest.mark.asyncio
    async def test_search_special_chars_in_query(self, fts_store, fts_session):
        await fts_store.index("/a.py", 'hello "world" test', session=fts_session)
        await fts_session.commit()

        # Should not raise — double quotes are escaped
        results = await fts_store.search('"hello"', session=fts_session)
        assert isinstance(results, list)


# ==================================================================
# Remove
# ==================================================================


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_single(self, fts_store, fts_session):
        await fts_store.index("/a.py", "hello world", session=fts_session)
        await fts_session.commit()

        await fts_store.remove("/a.py", session=fts_session)
        await fts_session.commit()

        results = await fts_store.search("hello", session=fts_session)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_noop(self, fts_store, fts_session):
        # Should not raise
        await fts_store.remove("/nonexistent.py", session=fts_session)
        await fts_session.commit()

    @pytest.mark.asyncio
    async def test_remove_no_session_noop(self, fts_store):
        await fts_store.remove("/a.py")


# ==================================================================
# Remove file (with chunks)
# ==================================================================


class TestRemoveFile:
    @pytest.mark.asyncio
    async def test_remove_file_removes_base_and_chunks(self, fts_store, fts_session):
        await fts_store.index("/auth.py", "main file content", session=fts_session)
        await fts_store.index("/auth.py#login", "login function", session=fts_session)
        await fts_store.index("/auth.py#logout", "logout function", session=fts_session)
        await fts_store.index("/other.py", "other file", session=fts_session)
        await fts_session.commit()

        await fts_store.remove_file("/auth.py", session=fts_session)
        await fts_session.commit()

        # All auth.py entries gone
        results = await fts_store.search("login", session=fts_session)
        assert len(results) == 0
        results = await fts_store.search("logout", session=fts_session)
        assert len(results) == 0
        results = await fts_store.search("main", session=fts_session)
        assert len(results) == 0

        # other.py still present
        results = await fts_store.search("other", session=fts_session)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_remove_file_no_session_noop(self, fts_store):
        await fts_store.remove_file("/a.py")

    @pytest.mark.asyncio
    async def test_remove_file_nonexistent(self, fts_store, fts_session):
        await fts_store.remove_file("/nonexistent.py", session=fts_session)
        await fts_session.commit()

    @pytest.mark.asyncio
    async def test_remove_file_escapes_like_wildcards(self, fts_store, fts_session):
        """Paths with _ (common in Python) must not act as LIKE wildcards."""
        await fts_store.index("/my_module.py", "main content", session=fts_session)
        await fts_store.index("/my_module.py#login", "login chunk", session=fts_session)
        await fts_store.index("/myxmodule.py#login", "should survive", session=fts_session)
        await fts_session.commit()

        await fts_store.remove_file("/my_module.py", session=fts_session)
        await fts_session.commit()

        # my_module.py and its chunks are gone
        results = await fts_store.search("main", session=fts_session)
        assert len(results) == 0
        results = await fts_store.search("login chunk", session=fts_session)
        assert len(results) == 0

        # myxmodule.py#login must NOT be deleted (underscore is not a wildcard)
        results = await fts_store.search("should survive", session=fts_session)
        assert len(results) == 1
        assert results[0].path == "/myxmodule.py#login"

"""Tests for FullTextStore protocol satisfaction."""

from __future__ import annotations

import pytest

from grover.search.fulltext.mssql import MSSQLFullTextStore
from grover.search.fulltext.postgres import PostgresFullTextStore
from grover.search.fulltext.protocol import FullTextStore
from grover.search.fulltext.sqlite import SQLiteFullTextStore
from grover.search.fulltext.types import FullTextResult

# ==================================================================
# Protocol satisfaction
# ==================================================================


class TestProtocolSatisfaction:
    def test_sqlite_satisfies_protocol(self):
        store = SQLiteFullTextStore()
        assert isinstance(store, FullTextStore)

    def test_postgres_satisfies_protocol(self):
        store = PostgresFullTextStore()
        assert isinstance(store, FullTextStore)

    def test_mssql_satisfies_protocol(self):
        store = MSSQLFullTextStore()
        assert isinstance(store, FullTextStore)


# ==================================================================
# FullTextResult
# ==================================================================


class TestFullTextResult:
    def test_frozen(self):
        r = FullTextResult(path="/a.py", snippet="hello", rank=0.5)
        with pytest.raises(AttributeError):
            r.path = "/b.py"  # type: ignore[misc]

    def test_defaults(self):
        r = FullTextResult(path="/a.py")
        assert r.snippet == ""
        assert r.rank == 0.0

    def test_fields(self):
        r = FullTextResult(path="/a.py", snippet="some text", rank=1.5)
        assert r.path == "/a.py"
        assert r.snippet == "some text"
        assert r.rank == 1.5

    def test_equality(self):
        a = FullTextResult(path="/a.py", snippet="hi", rank=1.0)
        b = FullTextResult(path="/a.py", snippet="hi", rank=1.0)
        assert a == b

    def test_inequality(self):
        a = FullTextResult(path="/a.py", rank=1.0)
        b = FullTextResult(path="/b.py", rank=1.0)
        assert a != b


# ==================================================================
# Constructor defaults
# ==================================================================


class TestConstructorDefaults:
    def test_sqlite_defaults(self):
        store = SQLiteFullTextStore()
        assert store._engine is None
        assert store._table == "grover_fts"
        assert store._table_created is False

    def test_postgres_defaults(self):
        store = PostgresFullTextStore()
        assert store._engine is None
        assert store._table == "grover_fts"
        assert store._config == "english"
        assert store._table_created is False

    def test_mssql_defaults(self):
        store = MSSQLFullTextStore()
        assert store._engine is None
        assert store._table == "grover_fts"
        assert store._table_created is False

    def test_sqlite_custom_table(self):
        store = SQLiteFullTextStore(table_name="my_fts")
        assert store._table == "my_fts"

    def test_postgres_custom_config(self):
        store = PostgresFullTextStore(config="spanish")
        assert store._config == "spanish"


# ==================================================================
# No-session behavior (all implementations)
# ==================================================================


class TestNoSessionBehavior:
    @pytest.mark.asyncio
    async def test_sqlite_index_no_session(self):
        store = SQLiteFullTextStore()
        await store.index("/a.py", "content")  # Should not raise

    @pytest.mark.asyncio
    async def test_sqlite_remove_no_session(self):
        store = SQLiteFullTextStore()
        await store.remove("/a.py")

    @pytest.mark.asyncio
    async def test_sqlite_remove_file_no_session(self):
        store = SQLiteFullTextStore()
        await store.remove_file("/a.py")

    @pytest.mark.asyncio
    async def test_sqlite_search_no_session(self):
        store = SQLiteFullTextStore()
        results = await store.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_postgres_index_no_session(self):
        store = PostgresFullTextStore()
        await store.index("/a.py", "content")

    @pytest.mark.asyncio
    async def test_postgres_search_no_session(self):
        store = PostgresFullTextStore()
        results = await store.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_mssql_index_no_session(self):
        store = MSSQLFullTextStore()
        await store.index("/a.py", "content")

    @pytest.mark.asyncio
    async def test_mssql_search_no_session(self):
        store = MSSQLFullTextStore()
        results = await store.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_postgres_remove_no_session(self):
        store = PostgresFullTextStore()
        await store.remove("/a.py")

    @pytest.mark.asyncio
    async def test_mssql_remove_no_session(self):
        store = MSSQLFullTextStore()
        await store.remove("/a.py")

    @pytest.mark.asyncio
    async def test_postgres_remove_file_no_session(self):
        store = PostgresFullTextStore()
        await store.remove_file("/a.py")

    @pytest.mark.asyncio
    async def test_mssql_remove_file_no_session(self):
        store = MSSQLFullTextStore()
        await store.remove_file("/a.py")


# ==================================================================
# ensure_table with no engine
# ==================================================================


class TestEnsureTableNoEngine:
    @pytest.mark.asyncio
    async def test_sqlite_no_engine(self):
        store = SQLiteFullTextStore()
        await store.ensure_table()
        assert not store._table_created

    @pytest.mark.asyncio
    async def test_postgres_no_engine(self):
        store = PostgresFullTextStore()
        await store.ensure_table()
        assert not store._table_created

    @pytest.mark.asyncio
    async def test_mssql_no_engine(self):
        store = MSSQLFullTextStore()
        await store.ensure_table()
        assert not store._table_created

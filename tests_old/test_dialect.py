"""Tests for fs/dialect.py — dialect detection and upsert."""

from __future__ import annotations

import unittest.mock
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.models.database.file import FileModel
from grover.util.dialect import (
    _get_schema_from_session,
    _upsert_mssql,
    check_tables_exist,
    get_dialect,
    now_expression,
    upsert_file,
)

# =========================================================================
# check_tables_exist()
# =========================================================================


class TestCheckTablesExist:
    def test_returns_intersection(self):
        conn = MagicMock()
        inspector = MagicMock()
        inspector.get_table_names.return_value = ["grover_files", "other_table"]
        with unittest.mock.patch("grover.util.dialect.inspect", return_value=inspector):
            result = check_tables_exist(
                conn,
                ["grover_files", "grover_file_versions"],
            )
        assert result == {"grover_files"}
        inspector.get_table_names.assert_called_once_with()

    def test_returns_empty_when_none_exist(self):
        conn = MagicMock()
        inspector = MagicMock()
        inspector.get_table_names.return_value = []
        with unittest.mock.patch("grover.util.dialect.inspect", return_value=inspector):
            result = check_tables_exist(conn, ["grover_files"])
        assert result == set()


class TestGetDialect:
    async def test_sqlite_async(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        assert get_dialect(engine) == "sqlite"
        await engine.dispose()

    def test_sqlite_sync(self):
        from sqlmodel import create_engine

        engine = create_engine("sqlite://", echo=False)
        assert get_dialect(engine) == "sqlite"


class TestNowExpression:
    def test_sqlite(self):
        expr = now_expression("sqlite")
        assert expr is not None

    def test_postgresql(self):
        expr = now_expression("postgresql")
        assert expr is not None

    def test_mssql(self):
        expr = now_expression("mssql")
        assert expr is not None


class TestUpsertFile:
    async def test_insert(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            rowcount = await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "test-id-1",
                    "path": "/hello.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()
            assert rowcount >= 0

            result = await session.execute(select(FileModel).where(FileModel.path == "/hello.txt"))
            file = result.scalar_one_or_none()
            assert file is not None
            assert file.path == "/hello.txt"

        await engine.dispose()

    async def test_update_on_conflict(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Insert first
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "test-id-1",
                    "path": "/hello.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()

        async with factory() as session:
            # Upsert with same path but different data
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "test-id-2",
                    "path": "/hello.txt",
                    "is_directory": False,
                    "current_version": 2,
                },
                conflict_keys=["path"],
            )
            await session.commit()

            result = await session.execute(select(FileModel).where(FileModel.path == "/hello.txt"))
            file = result.scalar_one_or_none()
            assert file is not None
            assert file.current_version == 2

        await engine.dispose()


# =========================================================================
# get_dialect() edge cases
# =========================================================================


class TestGetDialectEdgeCases:
    def _mock_engine(self, dialect_name: str) -> MagicMock:
        """Create a mock engine (sync-style) with the given dialect name."""
        engine = MagicMock(spec=["dialect"])
        engine.dialect = MagicMock()
        engine.dialect.name = dialect_name
        return engine

    def test_unknown_dialect_returns_name(self):
        # sync engine has no sync_engine attr → getattr returns itself
        engine = self._mock_engine("oracle")
        result = get_dialect(engine)
        assert result == "oracle"

    def test_postgres_variant(self):
        engine = self._mock_engine("postgres")
        result = get_dialect(engine)
        assert result == "postgresql"

    def test_pyodbc_variant(self):
        engine = self._mock_engine("pyodbc")
        result = get_dialect(engine)
        assert result == "mssql"


# =========================================================================
# now_expression() return validation
# =========================================================================


class TestNowExpressionValidation:
    def test_sqlite_now_compiles(self):
        from sqlalchemy.dialects import sqlite

        expr = now_expression("sqlite")
        compiled = expr.compile(dialect=sqlite.dialect())
        assert "datetime" in str(compiled).lower()

    def test_postgresql_now_compiles(self):
        from sqlalchemy.dialects import postgresql

        expr = now_expression("postgresql")
        compiled = expr.compile(dialect=postgresql.dialect())
        assert "now" in str(compiled).lower()

    def test_mssql_now_compiles(self):
        from sqlalchemy.dialects import mssql

        expr = now_expression("mssql")
        compiled = expr.compile(dialect=mssql.dialect())
        assert "sysdatetimeoffset" in str(compiled).lower()


# =========================================================================
# _upsert_sqlite_pg() — branch coverage
# =========================================================================


class TestUpsertSqlitePgBranches:
    async def test_upsert_with_explicit_update_keys(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Insert initial row
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "u1",
                    "path": "/up.txt",
                    "is_directory": False,
                    "mime_type": "text/plain",
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()

        async with factory() as session:
            # Upsert with update_keys=["mime_type"] — only mime_type should update
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "u2",
                    "path": "/up.txt",
                    "is_directory": False,
                    "mime_type": "text/markdown",
                    "current_version": 99,
                },
                conflict_keys=["path"],
                update_keys=["mime_type"],
            )
            await session.commit()

            result = await session.execute(select(FileModel).where(FileModel.path == "/up.txt"))
            f = result.scalar_one()
            assert f.mime_type == "text/markdown"
            # current_version should NOT be updated (not in update_keys)
            assert f.current_version == 1

        await engine.dispose()

    async def test_upsert_on_conflict_do_nothing(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "dn1",
                    "path": "/dn.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()

        async with factory() as session:
            # All value keys are in conflict_keys → on_conflict_do_nothing
            await upsert_file(
                session,
                "sqlite",
                values={"path": "/dn.txt"},
                conflict_keys=["path"],
            )
            await session.commit()

            result = await session.execute(select(FileModel).where(FileModel.path == "/dn.txt"))
            f = result.scalar_one()
            assert f.current_version == 1

        await engine.dispose()

    async def test_upsert_with_custom_model(self):
        from tests.test_configurable_model import WikiFile

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            rowcount = await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "w1",
                    "path": "/wiki.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
                model=WikiFile,
            )
            await session.commit()
            assert rowcount >= 0

            result = await session.execute(select(WikiFile).where(WikiFile.path == "/wiki.txt"))
            w = result.scalar_one()
            assert w.path == "/wiki.txt"

        await engine.dispose()


# =========================================================================
# _upsert_mssql() — SQL generation via mock
# =========================================================================


class TestUpsertMssql:
    @staticmethod
    def _mssql_mock_session():
        """Mock session with no schema_translate_map."""
        session = AsyncMock(spec=AsyncSession)
        mock_bind = MagicMock(spec=["_execution_options"])
        mock_bind._execution_options = {}
        session.get_bind.return_value = mock_bind
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute.return_value = mock_result
        return session

    async def test_mssql_upsert_generates_merge(self):
        mock_session = self._mssql_mock_session()

        await _upsert_mssql(
            mock_session,
            values={"id": "m1", "path": "/m.txt", "current_version": 1},
            conflict_keys=["path"],
            model=FileModel,
        )
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "MERGE INTO" in sql_text
        assert "WITH (HOLDLOCK)" in sql_text
        assert "WHEN NOT MATCHED" in sql_text
        assert "WHEN MATCHED THEN" in sql_text
        assert "UPDATE SET" in sql_text

    async def test_mssql_upsert_with_update_keys(self):
        mock_session = self._mssql_mock_session()

        await _upsert_mssql(
            mock_session,
            values={
                "id": "m3",
                "path": "/m.txt",
                "mime_type": "text/markdown",
                "current_version": 5,
            },
            conflict_keys=["path"],
            model=FileModel,
            update_keys=["mime_type"],
        )
        sql_text = str(mock_session.execute.call_args[0][0])
        assert "target.mime_type" in sql_text
        # current_version should NOT appear in UPDATE SET
        assert "target.current_version" not in sql_text

    async def test_mssql_upsert_no_update_cols(self):
        mock_session = self._mssql_mock_session()

        # All keys are conflict_keys → no WHEN MATCHED
        await _upsert_mssql(
            mock_session,
            values={"path": "/m.txt"},
            conflict_keys=["path"],
            model=FileModel,
        )
        sql_text = str(mock_session.execute.call_args[0][0])
        assert "WHEN NOT MATCHED" in sql_text
        assert "WHEN MATCHED" not in sql_text

    async def test_mssql_upsert_custom_model(self):
        from tests.test_configurable_model import WikiFile

        mock_session = self._mssql_mock_session()

        await _upsert_mssql(
            mock_session,
            values={"id": "w1", "path": "/w.txt", "mime_type": "text/plain"},
            conflict_keys=["path"],
            model=WikiFile,
        )
        sql_text = str(mock_session.execute.call_args[0][0])
        assert "wiki_files" in sql_text


# =========================================================================
# _get_schema_from_session() + schema-qualified MERGE
# =========================================================================


class TestGetSchemaFromSession:
    @staticmethod
    def _mock_session_with_bind(execution_options: dict, *, has_sync_engine: bool = False):
        """Build a mock AsyncSession whose get_bind() returns a bind with given options.

        When *has_sync_engine* is False the bind acts like a sync engine
        (no ``sync_engine`` attribute).  When True an outer async-engine
        wrapper is simulated.
        """
        mock_session = AsyncMock(spec=AsyncSession)
        if has_sync_engine:
            mock_async_bind = MagicMock()
            mock_sync = MagicMock(spec=["_execution_options"])
            mock_sync._execution_options = execution_options
            mock_async_bind.sync_engine = mock_sync
            mock_session.get_bind.return_value = mock_async_bind
        else:
            mock_bind = MagicMock(spec=["_execution_options"])
            mock_bind._execution_options = execution_options
            mock_session.get_bind.return_value = mock_bind
        return mock_session

    def test_no_schema_translate_map(self):
        session = self._mock_session_with_bind({})
        assert _get_schema_from_session(session) is None

    def test_schema_translate_map_with_default(self):
        session = self._mock_session_with_bind({"schema_translate_map": {None: "grover"}})
        assert _get_schema_from_session(session) == "grover"

    def test_schema_translate_map_without_default(self):
        session = self._mock_session_with_bind({"schema_translate_map": {"other": "schema"}})
        assert _get_schema_from_session(session) is None

    def test_async_engine_unwraps_sync_engine(self):
        session = self._mock_session_with_bind({"schema_translate_map": {None: "dbo"}}, has_sync_engine=True)
        assert _get_schema_from_session(session) == "dbo"


class TestMssqlSchemaQualifiedMerge:
    async def test_merge_uses_schema_when_present(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_bind = MagicMock(spec=["_execution_options"])
        mock_bind._execution_options = {"schema_translate_map": {None: "grover"}}
        mock_session.get_bind.return_value = mock_bind
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        await _upsert_mssql(
            mock_session,
            values={"id": "s1", "path": "/s.txt", "current_version": 1},
            conflict_keys=["path"],
            model=FileModel,
        )
        sql_text = str(mock_session.execute.call_args[0][0])
        assert "[grover].grover_files" in sql_text
        assert "MERGE INTO [grover].grover_files" in sql_text

    async def test_merge_no_schema_uses_bare_table(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_bind = MagicMock(spec=["_execution_options"])
        mock_bind._execution_options = {}
        mock_session.get_bind.return_value = mock_bind
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        await _upsert_mssql(
            mock_session,
            values={"id": "s2", "path": "/s.txt", "current_version": 1},
            conflict_keys=["path"],
            model=FileModel,
        )
        sql_text = str(mock_session.execute.call_args[0][0])
        assert "MERGE INTO grover_files" in sql_text
        assert "[" not in sql_text.split("WITH")[0]  # no brackets before WITH

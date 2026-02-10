"""Tests for fs/dialect.py — dialect detection and upsert."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.fs.dialect import get_dialect, now_expression, upsert_file
from grover.models.files import File


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
                    "name": "hello.txt",
                    "is_directory": False,

                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()
            assert rowcount >= 0

            result = await session.execute(
                select(File).where(File.path == "/hello.txt")
            )
            file = result.scalar_one_or_none()
            assert file is not None
            assert file.name == "hello.txt"

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
                    "name": "hello.txt",
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
                    "name": "hello_updated.txt",
                    "is_directory": False,

                    "current_version": 2,
                },
                conflict_keys=["path"],
            )
            await session.commit()

            result = await session.execute(
                select(File).where(File.path == "/hello.txt")
            )
            file = result.scalar_one_or_none()
            assert file is not None
            assert file.name == "hello_updated.txt"
            assert file.current_version == 2

        await engine.dispose()

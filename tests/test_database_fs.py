"""Tests for DatabaseFileSystem — session lifecycle, transactions, context manager."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem

# =========================================================================
# Helpers
# =========================================================================


async def _make_db_fs():
    """Create a DatabaseFileSystem with in-memory SQLite."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = DatabaseFileSystem(session_factory=factory, dialect="sqlite")
    return db, engine


# =========================================================================
# Session Lifecycle
# =========================================================================


class TestSessionLifecycle:
    async def test_session_created_on_first_op(self):
        db, engine = await _make_db_fs()
        assert db._session is None
        async with db:
            await db.write("/hello.txt", "hello")
            assert db._session is not None
        await engine.dispose()

    async def test_session_reused_across_ops(self):
        db, engine = await _make_db_fs()
        async with db:
            await db.write("/a.txt", "a")
            session_a = db._session
            await db.write("/b.txt", "b")
            session_b = db._session
            assert session_a is session_b
        await engine.dispose()

    async def test_close_clears_session(self):
        db, engine = await _make_db_fs()
        await db.write("/x.txt", "x")
        assert db._session is not None
        await db.close()
        assert db._session is None
        await engine.dispose()


# =========================================================================
# Transaction vs Commit
# =========================================================================


class TestTransactionBehavior:
    async def test_commit_flushes_in_transaction(self):
        db, engine = await _make_db_fs()
        db.in_transaction = True
        async with db:
            await db.write("/flushed.txt", "data")
            session = db._session
            assert session is not None
            # In transaction mode, _commit flushes but does not commit.
            # The session should still have the data available for reading.
            result = await db.read("/flushed.txt")
            assert result.success
            assert result.content == "data"
        await engine.dispose()

    async def test_commit_commits_outside_transaction(self):
        db, engine = await _make_db_fs()
        assert db.in_transaction is False
        await db.write("/committed.txt", "data")
        # Data should persist: read it back through a fresh session
        result = await db.read("/committed.txt")
        assert result.success
        assert result.content == "data"
        await db.close()
        await engine.dispose()


# =========================================================================
# Context Manager
# =========================================================================


class TestContextManagerBehavior:
    async def test_aexit_commits_on_success(self):
        db, engine = await _make_db_fs()
        async with db:
            await db.write("/persisted.txt", "persisted")
        # After context exit, session should be None
        assert db._session is None
        # Re-open and verify data persisted
        db2 = DatabaseFileSystem(
            session_factory=async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False),
            dialect="sqlite",
        )
        result = await db2.read("/persisted.txt")
        assert result.success
        assert result.content == "persisted"
        await db2.close()
        await engine.dispose()

    async def test_aexit_rollback_on_exception(self):
        db, engine = await _make_db_fs()
        # Use in_transaction so writes are flushed (not committed)
        db.in_transaction = True
        try:
            async with db:
                await db.write("/doomed.txt", "doomed")
                # Data should be readable within the transaction
                result = await db.read("/doomed.txt")
                assert result.success
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # After rollback, session should be cleared
        assert db._session is None
        # Verify data did NOT persist after rollback
        db2 = DatabaseFileSystem(
            session_factory=async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False),
            dialect="sqlite",
        )
        result = await db2.read("/doomed.txt")
        assert not result.success
        await db2.close()
        await engine.dispose()

    async def test_aexit_clears_session(self):
        db, engine = await _make_db_fs()
        async with db:
            await db.write("/temp.txt", "temp")
            assert db._session is not None
        assert db._session is None
        await engine.dispose()


# =========================================================================
# Content Filtering (soft-delete)
# =========================================================================


class TestContentFiltering:
    async def test_read_deleted_file_returns_failure(self):
        db, engine = await _make_db_fs()
        async with db:
            await db.write("/alive.txt", "alive")
            # Verify it can be read
            result = await db.read("/alive.txt")
            assert result.success
            # Soft-delete it
            await db.delete("/alive.txt", permanent=False)
            # Now read should fail
            result = await db.read("/alive.txt")
            assert not result.success
        await engine.dispose()

    async def test_content_exists_false_for_deleted(self):
        db, engine = await _make_db_fs()
        async with db:
            await db.write("/soon_gone.txt", "content")
            assert await db._content_exists("/soon_gone.txt") is True
            await db.delete("/soon_gone.txt", permanent=False)
            assert await db._content_exists("/soon_gone.txt") is False
        await engine.dispose()

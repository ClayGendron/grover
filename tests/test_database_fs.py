"""Tests for DatabaseFileSystem — stateless, session-injected."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem

# =========================================================================
# Helpers
# =========================================================================


async def _make_db_fs():
    """Create a stateless DatabaseFileSystem + session factory."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = DatabaseFileSystem(dialect="sqlite")
    return db, factory, engine


# =========================================================================
# Stateless Construction
# =========================================================================


class TestStatelessConstruction:
    async def test_no_session_state(self):
        db, _factory, engine = await _make_db_fs()
        # DFS should have no session-related attributes
        assert not hasattr(db, "_session")
        assert not hasattr(db, "session_factory")
        assert not hasattr(db, "_txn_session")
        await engine.dispose()

    async def test_context_manager_is_noop(self):
        db, _factory, engine = await _make_db_fs()
        async with db:
            pass  # Should not raise
        await engine.dispose()

    async def test_close_is_noop(self):
        db, _factory, engine = await _make_db_fs()
        await db.close()  # Should not raise
        await engine.dispose()


# =========================================================================
# Session Injection
# =========================================================================


class TestSessionInjection:
    async def test_write_read_with_session(self):
        db, factory, engine = await _make_db_fs()
        async with factory() as session:
            result = await db.write("/hello.txt", "hello", session=session)
            assert result.success
            read = await db.read("/hello.txt", session=session)
            assert read.success
            assert read.content == "hello"
        await engine.dispose()

    async def test_write_without_session_raises(self):
        import pytest

        db, _factory, engine = await _make_db_fs()
        with pytest.raises(TypeError):
            await db.write("/hello.txt", "hello")
        await engine.dispose()

    async def test_read_without_session_raises(self):
        import pytest

        db, _factory, engine = await _make_db_fs()
        with pytest.raises(TypeError):
            await db.read("/hello.txt")
        await engine.dispose()


# =========================================================================
# Flush Behavior (DFS flushes, never commits)
# =========================================================================


class TestFlushBehavior:
    async def test_flush_without_commit(self):
        """DFS flushes within session but never commits."""
        db, factory, engine = await _make_db_fs()
        async with factory() as session:
            await db.write("/flushed.txt", "data", session=session)
            # Data visible in same session (flushed)
            read = await db.read("/flushed.txt", session=session)
            assert read.success
            assert read.content == "data"
            # Don't commit — session will rollback on close
        # Data should NOT persist without explicit commit
        async with factory() as session:
            read = await db.read("/flushed.txt", session=session)
            assert not read.success
        await engine.dispose()

    async def test_data_persists_with_explicit_commit(self):
        db, factory, engine = await _make_db_fs()
        async with factory() as session:
            await db.write("/committed.txt", "data", session=session)
            await session.commit()
        async with factory() as session:
            read = await db.read("/committed.txt", session=session)
            assert read.success
            assert read.content == "data"
        await engine.dispose()


# =========================================================================
# Content Filtering (soft-delete)
# =========================================================================


class TestContentFiltering:
    async def test_read_deleted_file_returns_failure(self):
        db, factory, engine = await _make_db_fs()
        async with factory() as session:
            await db.write("/alive.txt", "alive", session=session)
            result = await db.read("/alive.txt", session=session)
            assert result.success
            await db.delete("/alive.txt", permanent=False, session=session)
            result = await db.read("/alive.txt", session=session)
            assert not result.success
        await engine.dispose()

    async def test_content_exists_false_for_deleted(self):
        db, factory, engine = await _make_db_fs()
        async with factory() as session:
            await db.write("/soon_gone.txt", "content", session=session)
            assert await db._content_exists("/soon_gone.txt", session) is True
            await db.delete("/soon_gone.txt", permanent=False, session=session)
            assert await db._content_exists("/soon_gone.txt", session) is False
        await engine.dispose()


# =========================================================================
# Concurrency Safety
# =========================================================================


class TestConcurrencySafety:
    async def test_dfs_has_no_instance_state(self):
        """DFS should have no mutable session state — safe for concurrent use."""
        db, factory, engine = await _make_db_fs()
        s1 = factory()
        s2 = factory()
        try:
            # Writing via s1 should not pollute DFS instance state
            await db.write("/s1.txt", "from s1", session=s1)
            # DFS should still have no session-related attributes
            assert not hasattr(db, "_session")
            assert not hasattr(db, "session_factory")
            assert not hasattr(db, "_txn_session")
            # s2 can independently write without conflict
            await db.write("/s2.txt", "from s2", session=s2)
            read = await db.read("/s2.txt", session=s2)
            assert read.success
            assert read.content == "from s2"
        finally:
            await s1.close()
            await s2.close()
        await engine.dispose()

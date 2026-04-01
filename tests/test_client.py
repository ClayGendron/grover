"""Tests for GroverAsync facade and GroverFileSystem storage flag."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.base import GroverFileSystem
from grover.client import GroverAsync

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _sqlite_engine():
    """Create an in-memory SQLite engine with tables."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


# ==================================================================
# GroverFileSystem storage flag
# ==================================================================


class TestStorageFlag:
    """Tests for the storage parameter on GroverFileSystem."""

    def test_storage_true_requires_engine_or_session(self):
        with pytest.raises(ValueError, match="storage=True"):
            GroverFileSystem(storage=True)

    def test_storage_true_default_requires_engine_or_session(self):
        with pytest.raises(ValueError, match="storage=True"):
            GroverFileSystem()

    async def test_storage_true_with_engine_works(self):
        engine = await _sqlite_engine()
        try:
            fs = GroverFileSystem(engine=engine)
            assert fs._storage is True
            assert fs._session_factory is not None
        finally:
            await engine.dispose()

    def test_storage_false_without_engine_works(self):
        fs = GroverFileSystem(storage=False)
        assert fs._storage is False
        assert fs._session_factory is None
        assert fs._engine is None

    async def test_use_session_raises_when_storageless(self):
        fs = GroverFileSystem(storage=False)
        with pytest.raises(RuntimeError, match="storage=False"):
            async with fs._use_session():
                pass

    async def test_route_single_unmounted_path_returns_error(self):
        fs = GroverFileSystem(storage=False)
        result = await fs.read("/nothing/here.txt")
        assert not result.success
        assert "No mount found" in result.error_message

    async def test_route_fanout_no_mounts_returns_empty(self):
        fs = GroverFileSystem(storage=False)
        result = await fs.glob("*.py")
        assert result.success
        assert len(result.candidates) == 0

    async def test_route_fanout_fans_out_to_mounts(self):
        router = GroverFileSystem(storage=False)
        engine = await _sqlite_engine()
        try:
            child = DatabaseFileSystem(engine=engine)
            await router.add_mount("/data", child)

            async with child._use_session() as s:
                await child._write_impl("/hello.py", content="print('hi')", session=s)

            result = await router.glob("**/*.py")
            assert result.success
            assert any("hello.py" in c.path for c in result.candidates)
        finally:
            await engine.dispose()


# ==================================================================
# GroverAsync
# ==================================================================


class TestGroverAsyncConstruction:
    """Tests for GroverAsync construction."""

    def test_creates_without_engine(self):
        g = GroverAsync()
        assert g._storage is False
        assert g._session_factory is None

    def test_inherits_grover_filesystem(self):
        g = GroverAsync()
        assert isinstance(g, GroverFileSystem)


class TestGroverAsyncAddMount:
    """Tests for GroverAsync.add_mount with different config paths."""

    async def test_add_mount_with_engine_url(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        try:
            w = await g.write("/data/hello.txt", "hello")
            assert w.success

            r = await g.read("/data/hello.txt")
            assert r.success
            assert r.content == "hello"
        finally:
            await g.close()

    async def test_add_mount_with_engine(self):
        engine = await _sqlite_engine()
        g = GroverAsync()
        await g.add_mount("data", engine=engine)
        try:
            w = await g.write("/data/test.txt", "content")
            assert w.success

            r = await g.read("/data/test.txt")
            assert r.content == "content"
        finally:
            await g.close()

    async def test_add_mount_with_session_factory(self):
        engine = await _sqlite_engine()
        from sqlalchemy.ext.asyncio import async_sessionmaker

        sf = async_sessionmaker(engine, expire_on_commit=False)
        g = GroverAsync()
        await g.add_mount("data", session_factory=sf)
        try:
            w = await g.write("/data/test.txt", "via sf")
            assert w.success

            r = await g.read("/data/test.txt")
            assert r.content == "via sf"
        finally:
            await g.close()

    async def test_add_mount_with_filesystem(self):
        engine = await _sqlite_engine()
        fs = DatabaseFileSystem(engine=engine)
        g = GroverAsync()
        await g.add_mount("data", filesystem=fs)
        try:
            w = await g.write("/data/test.txt", "direct fs")
            assert w.success

            r = await g.read("/data/test.txt")
            assert r.content == "direct fs"
        finally:
            await g.close()

    async def test_add_mount_no_args_raises(self):
        g = GroverAsync()
        with pytest.raises(ValueError, match="requires one of"):
            await g.add_mount("data")

    async def test_add_mount_creates_tables(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        try:
            # If tables weren't created, this would fail
            w = await g.write("/data/test.txt", "tables exist")
            assert w.success
        finally:
            await g.close()

    async def test_add_mount_user_scoped(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://", user_scoped=True)
        try:
            w = await g.write("/data/hello.txt", "user content", user_id="alice")
            assert w.success

            r = await g.read("/data/hello.txt", user_id="alice")
            assert r.content == "user content"

            # Different user can't see it
            r2 = await g.read("/data/hello.txt", user_id="bob")
            assert not r2.success
        finally:
            await g.close()


class TestGroverAsyncProviderInjection:
    """Tests for embedding/vector store provider injection."""

    async def test_providers_injected_on_new_filesystem(self):
        from unittest.mock import MagicMock

        engine = await _sqlite_engine()
        ep = MagicMock()
        vs = MagicMock()
        g = GroverAsync()
        await g.add_mount(
            "data", engine=engine, embedding_provider=ep, vector_store=vs
        )
        try:
            fs = g._mounts["/data"]
            assert isinstance(fs, DatabaseFileSystem)
            assert fs._embedding_provider is ep
            assert fs._vector_store is vs
        finally:
            await g.close()

    async def test_providers_injected_on_existing_filesystem(self):
        from unittest.mock import MagicMock

        engine = await _sqlite_engine()
        fs = DatabaseFileSystem(engine=engine)
        assert fs._embedding_provider is None

        ep = MagicMock()
        g = GroverAsync()
        await g.add_mount("data", filesystem=fs, embedding_provider=ep)
        try:
            assert fs._embedding_provider is ep
        finally:
            await g.close()

    async def test_existing_providers_not_overwritten(self):
        from unittest.mock import MagicMock

        engine = await _sqlite_engine()
        original_ep = MagicMock()
        fs = DatabaseFileSystem(engine=engine, embedding_provider=original_ep)

        new_ep = MagicMock()
        g = GroverAsync()
        await g.add_mount("data", filesystem=fs, embedding_provider=new_ep)
        try:
            assert fs._embedding_provider is original_ep
        finally:
            await g.close()


class TestGroverAsyncRouting:
    """Tests for operation routing through GroverAsync."""

    async def test_read_unmounted_path_returns_error(self):
        g = GroverAsync()
        result = await g.read("/nothing/file.txt")
        assert not result.success
        assert "No mount found" in result.error_message

    async def test_multi_mount_routing(self):
        g = GroverAsync()
        await g.add_mount("alpha", engine_url="sqlite+aiosqlite://")
        await g.add_mount("beta", engine_url="sqlite+aiosqlite://")
        try:
            await g.write("/alpha/a.txt", "alpha content")
            await g.write("/beta/b.txt", "beta content")

            ra = await g.read("/alpha/a.txt")
            assert ra.content == "alpha content"

            rb = await g.read("/beta/b.txt")
            assert rb.content == "beta content"

            # Cross-check: alpha doesn't have beta's file
            rx = await g.read("/alpha/b.txt")
            assert not rx.success
        finally:
            await g.close()

    async def test_glob_fans_out_across_mounts(self):
        g = GroverAsync()
        await g.add_mount("one", engine_url="sqlite+aiosqlite://")
        await g.add_mount("two", engine_url="sqlite+aiosqlite://")
        try:
            await g.write("/one/file.py", "one")
            await g.write("/two/file.py", "two")

            result = await g.glob("**/*.py")
            assert result.success
            paths = {c.path for c in result.candidates}
            assert "/one/file.py" in paths
            assert "/two/file.py" in paths
        finally:
            await g.close()

    async def test_grep_fans_out_across_mounts(self):
        g = GroverAsync()
        await g.add_mount("a", engine_url="sqlite+aiosqlite://")
        await g.add_mount("b", engine_url="sqlite+aiosqlite://")
        try:
            await g.write("/a/file.txt", "needle in a haystack")
            await g.write("/b/file.txt", "another needle here")

            result = await g.grep("needle")
            assert result.success
            paths = {c.path for c in result.candidates}
            assert "/a/file.txt" in paths
            assert "/b/file.txt" in paths
        finally:
            await g.close()

    async def test_fanout_no_mounts_returns_empty(self):
        g = GroverAsync()
        result = await g.glob("**")
        assert result.success
        assert len(result.candidates) == 0


class TestGroverAsyncQueryEngine:
    """Tests for run_query and cli through GroverAsync."""

    async def test_run_query(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        try:
            await g.write("/data/hello.py", "print('hi')")
            result = await g.run_query('glob "**/*.py"')
            assert result.success
            assert any("hello.py" in c.path for c in result.candidates)
        finally:
            await g.close()

    async def test_cli(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        try:
            await g.write("/data/hello.py", "print('hi')")
            output = await g.cli('glob "**/*.py"')
            assert isinstance(output, str)
            assert "hello.py" in output
        finally:
            await g.close()


class TestGroverAsyncLifecycle:
    """Tests for remove_mount and close."""

    async def test_remove_mount(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        assert "/data" in g._mounts
        await g.remove_mount("data")
        assert "/data" not in g._mounts

    async def test_remove_mount_removes_and_disposes(self):
        g = GroverAsync()
        await g.add_mount("data", engine_url="sqlite+aiosqlite://")
        assert "/data" in g._mounts
        # Should not raise — engine.dispose() is called internally
        await g.remove_mount("data")
        assert "/data" not in g._mounts

    async def test_close_clears_mounts(self):
        g = GroverAsync()
        await g.add_mount("a", engine_url="sqlite+aiosqlite://")
        await g.add_mount("b", engine_url="sqlite+aiosqlite://")
        assert len(g._mounts) == 2
        await g.close()
        assert len(g._mounts) == 0

    async def test_close_without_mounts(self):
        g = GroverAsync()
        await g.close()
        assert len(g._mounts) == 0

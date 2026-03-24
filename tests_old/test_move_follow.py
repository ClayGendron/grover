"""Tests for move with follow parameter (Phase 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grover.backends.database import DatabaseFileSystem
from grover.backends.user_scoped import UserScopedFileSystem
from grover.client import GroverAsync
from grover.models.config import SessionConfig
from grover.models.database.share import FileShareModel
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path) -> AsyncIterator[AsyncEngine]:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel

    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def dfs() -> DatabaseFileSystem:
    return DatabaseFileSystem()


@pytest.fixture
async def grover_with_sharing(session_factory, engine: AsyncEngine, tmp_path: Path) -> AsyncIterator[GroverAsync]:
    """GroverAsync with a UserScopedFileSystem backend that has sharing configured."""
    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    backend = UserScopedFileSystem(share_model=FileShareModel)
    sc = SessionConfig(session_factory=session_factory, dialect="sqlite")
    await g.add_mount("ws", filesystem=backend, session_config=sc)
    yield g
    await g.close()


# ==================================================================
# move_file level tests (operations.py)
# ==================================================================


class TestMoveFileFollow:
    """Test follow parameter on the operations.py move_file function."""

    @pytest.mark.asyncio
    async def test_move_default_clean_break(self, dfs: DatabaseFileSystem, session_factory):
        """follow=False creates a new file record at dest, source soft-deleted."""
        async with session_factory() as sess:
            await dfs.write("/old.py", "content", session=sess)
            old_file = await dfs._get_file_record(sess, "/old.py")
            assert old_file is not None
            old_id = old_file.id

            result = await dfs.move("/old.py", "/new.py", session=sess)
            assert result.success is True

            # New file should exist at dest
            new_file = await dfs._get_file_record(sess, "/new.py")
            assert new_file is not None
            # Different file record (clean break)
            assert new_file.id != old_id

            # Source should be soft-deleted
            old_after = await dfs._get_file_record(sess, "/old.py")
            assert old_after is None  # Not found (in trash)

    @pytest.mark.asyncio
    async def test_move_follow_same_file_record(self, dfs: DatabaseFileSystem, session_factory):
        """follow=True keeps the same file record (in-place rename)."""
        async with session_factory() as sess:
            await dfs.write("/old.py", "content", session=sess)
            old_file = await dfs._get_file_record(sess, "/old.py")
            assert old_file is not None
            old_id = old_file.id

            result = await dfs.move("/old.py", "/new.py", session=sess, follow=True)
            assert result.success is True

            # Same file record at new path
            new_file = await dfs._get_file_record(sess, "/new.py")
            assert new_file is not None
            assert new_file.id == old_id

    @pytest.mark.asyncio
    async def test_move_follow_shares_updated(self, session_factory):
        """follow=True with sharing updates share paths (via USFS)."""
        usfs = UserScopedFileSystem(share_model=FileShareModel)
        async with session_factory() as sess:
            await usfs.write("/doc.md", "data", session=sess, user_id="alice")
            await usfs._create_share(sess, "/alice/doc.md", "bob", "read", "alice")

            result = await usfs.move("/doc.md", "/renamed.md", session=sess, follow=True, user_id="alice")
            assert result.success is True

            # Share should be updated to new path
            shares = await usfs._list_shares_on_path(sess, "/alice/renamed.md")
            assert len(shares) == 1
            assert shares[0].grantee_id == "bob"

            # Old path should have no shares
            old_shares = await usfs._list_shares_on_path(sess, "/alice/doc.md")
            assert len(old_shares) == 0

    @pytest.mark.asyncio
    async def test_move_no_follow_shares_stale(self, session_factory):
        """follow=False does NOT update share paths -- they become stale."""
        usfs = UserScopedFileSystem(share_model=FileShareModel)
        async with session_factory() as sess:
            await usfs.write("/doc.md", "data", session=sess, user_id="alice")
            await usfs._create_share(sess, "/alice/doc.md", "bob", "read", "alice")

            result = await usfs.move("/doc.md", "/renamed.md", session=sess, follow=False, user_id="alice")
            assert result.success is True

            # Share still points to old path (stale)
            old_shares = await usfs._list_shares_on_path(sess, "/alice/doc.md")
            assert len(old_shares) == 1

            # No share at new path
            new_shares = await usfs._list_shares_on_path(sess, "/alice/renamed.md")
            assert len(new_shares) == 0

    @pytest.mark.asyncio
    async def test_move_follow_directory(self, dfs: DatabaseFileSystem, session_factory):
        """follow=True on a directory moves all children in-place."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.write("/src/a.py", "file a", session=sess)
            await dfs.write("/src/b.py", "file b", session=sess)

            a_file = await dfs._get_file_record(sess, "/src/a.py")
            assert a_file is not None
            a_id = a_file.id

            result = await dfs.move("/src", "/dst", session=sess, follow=True)
            assert result.success is True

            # Children moved in-place (same IDs)
            new_a = await dfs._get_file_record(sess, "/dst/a.py")
            assert new_a is not None
            assert new_a.id == a_id

            # Content preserved
            r = await dfs.read("/dst/b.py", session=sess)
            assert r.success
            assert r.file.content == "file b"

    @pytest.mark.asyncio
    async def test_move_follow_nested_dest_creates_parent_dirs(self, dfs: DatabaseFileSystem, session_factory):
        """follow=True to a nested path creates parent directories."""
        async with session_factory() as sess:
            await dfs.write("/src.py", "content", session=sess)

            result = await dfs.move("/src.py", "/deep/nested/dest.py", session=sess, follow=True)
            assert result.success is True

            # Parent directories should exist
            deep = await dfs._get_file_record(sess, "/deep")
            assert deep is not None
            assert deep.is_directory is True

            nested = await dfs._get_file_record(sess, "/deep/nested")
            assert nested is not None
            assert nested.is_directory is True

            # FileModel should be readable
            r = await dfs.read("/deep/nested/dest.py", session=sess)
            assert r.success
            assert r.file.content == "content"

    @pytest.mark.asyncio
    async def test_move_no_follow_directory(self, dfs: DatabaseFileSystem, session_factory):
        """follow=False on a directory creates new records for children."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.write("/src/a.py", "file a", session=sess)

            a_file = await dfs._get_file_record(sess, "/src/a.py")
            assert a_file is not None
            a_id = a_file.id

            result = await dfs.move("/src", "/dst", session=sess, follow=False)
            assert result.success is True

            # New record at dest (different ID)
            new_a = await dfs._get_file_record(sess, "/dst/a.py")
            assert new_a is not None
            assert new_a.id != a_id

            # Content preserved
            r = await dfs.read("/dst/a.py", session=sess)
            assert r.success
            assert r.file.content == "file a"

    @pytest.mark.asyncio
    async def test_move_follow_overwrite_shares_updated(self, session_factory):
        """follow=True overwrite: shares on src path updated to dest (via USFS)."""
        usfs = UserScopedFileSystem(share_model=FileShareModel)
        async with session_factory() as sess:
            await usfs.write("/src.md", "source", session=sess, user_id="alice")
            await usfs.write("/dst.md", "dest", session=sess, user_id="alice")
            await usfs._create_share(sess, "/alice/src.md", "bob", "read", "alice")

            result = await usfs.move("/src.md", "/dst.md", session=sess, follow=True, user_id="alice")
            assert result.success is True

            # Share moved to dest
            dst_shares = await usfs._list_shares_on_path(sess, "/alice/dst.md")
            assert len(dst_shares) == 1
            assert dst_shares[0].grantee_id == "bob"

    @pytest.mark.asyncio
    async def test_move_no_follow_directory_preserves_is_directory(self, dfs: DatabaseFileSystem, session_factory):
        """follow=False on dir with nested subdirs preserves is_directory flag."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.mkdir("/src/sub", session=sess)
            await dfs.write("/src/sub/file.py", "code", session=sess)

            result = await dfs.move("/src", "/dst", session=sess, follow=False)
            assert result.success is True

            # Subdirectory should still be a directory
            sub = await dfs._get_file_record(sess, "/dst/sub")
            assert sub is not None
            assert sub.is_directory is True

            # FileModel should not be a directory
            f = await dfs._get_file_record(sess, "/dst/sub/file.py")
            assert f is not None
            assert f.is_directory is False

            # Content preserved
            r = await dfs.read("/dst/sub/file.py", session=sess)
            assert r.success
            assert r.file.content == "code"


# ==================================================================
# GroverAsync-level tests (authenticated mounts)
# ==================================================================


class TestGroverMoveFollow:
    """Facade-level follow tests — deferred (USFS batch write integration incomplete)."""

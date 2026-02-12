"""Tests for move with follow parameter (Phase 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.sharing import SharingService
from grover.fs.vfs import VFS
from grover.models.shares import FileShare

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
    return DatabaseFileSystem(dialect="sqlite")


@pytest.fixture
def sharing() -> SharingService:
    return SharingService(FileShare)


@pytest.fixture
def vfs_with_sharing(
    dfs: DatabaseFileSystem, session_factory, sharing: SharingService, engine: AsyncEngine
) -> VFS:
    """VFS with an authenticated mount that has a SharingService."""
    from grover.events import EventBus

    registry = MountRegistry()
    bus = EventBus()
    v = VFS(registry, bus)
    config = MountConfig(
        mount_path="/ws",
        backend=dfs,
        session_factory=session_factory,
        mount_type="vfs",
        authenticated=True,
        sharing=sharing,
    )
    registry.add_mount(config)
    return v


# ==================================================================
# move_file level tests (operations.py)
# ==================================================================


class TestMoveFileFollow:
    """Test follow parameter on the operations.py move_file function."""

    @pytest.mark.asyncio
    async def test_move_default_clean_break(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=False creates a new file record at dest, source soft-deleted."""
        async with session_factory() as sess:
            await dfs.write("/old.py", "content", session=sess)
            old_file = await dfs.metadata.get_file(sess, "/old.py")
            assert old_file is not None
            old_id = old_file.id

            result = await dfs.move("/old.py", "/new.py", session=sess)
            assert result.success is True

            # New file should exist at dest
            new_file = await dfs.metadata.get_file(sess, "/new.py")
            assert new_file is not None
            # Different file record (clean break)
            assert new_file.id != old_id

            # Source should be soft-deleted
            old_after = await dfs.metadata.get_file(sess, "/old.py")
            assert old_after is None  # Not found (in trash)

    @pytest.mark.asyncio
    async def test_move_follow_same_file_record(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=True keeps the same file record (in-place rename)."""
        async with session_factory() as sess:
            await dfs.write("/old.py", "content", session=sess)
            old_file = await dfs.metadata.get_file(sess, "/old.py")
            assert old_file is not None
            old_id = old_file.id

            result = await dfs.move(
                "/old.py", "/new.py", session=sess, follow=True
            )
            assert result.success is True

            # Same file record at new path
            new_file = await dfs.metadata.get_file(sess, "/new.py")
            assert new_file is not None
            assert new_file.id == old_id

    @pytest.mark.asyncio
    async def test_move_follow_versions_preserved(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=True preserves version history at the new path."""
        async with session_factory() as sess:
            await dfs.write("/versioned.py", "v1", session=sess)
            await dfs.write("/versioned.py", "v2", session=sess)

            versions_before = await dfs.list_versions("/versioned.py", session=sess)
            assert versions_before.success
            assert len(versions_before.versions) == 2

            result = await dfs.move(
                "/versioned.py", "/renamed.py", session=sess, follow=True
            )
            assert result.success is True

            versions_after = await dfs.list_versions("/renamed.py", session=sess)
            assert versions_after.success
            assert len(versions_after.versions) == 2

    @pytest.mark.asyncio
    async def test_move_no_follow_no_version_history(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=False creates fresh file — version history starts at v1."""
        async with session_factory() as sess:
            await dfs.write("/old.py", "v1", session=sess)
            await dfs.write("/old.py", "v2", session=sess)

            result = await dfs.move(
                "/old.py", "/new.py", session=sess, follow=False
            )
            assert result.success is True

            versions = await dfs.list_versions("/new.py", session=sess)
            assert versions.success
            # New file starts at version 1
            assert len(versions.versions) == 1

    @pytest.mark.asyncio
    async def test_move_follow_shares_updated(
        self, dfs: DatabaseFileSystem, session_factory, sharing: SharingService
    ):
        """follow=True with sharing updates share paths."""
        async with session_factory() as sess:
            await dfs.write("/alice/doc.md", "data", session=sess)
            await sharing.create_share(
                sess, "/alice/doc.md", "bob", "read", "alice"
            )

            result = await dfs.move(
                "/alice/doc.md",
                "/alice/renamed.md",
                session=sess,
                follow=True,
                sharing=sharing,
            )
            assert result.success is True

            # Share should be updated to new path
            shares = await sharing.list_shares_on_path(sess, "/alice/renamed.md")
            assert len(shares) == 1
            assert shares[0].grantee_id == "bob"

            # Old path should have no shares
            old_shares = await sharing.list_shares_on_path(sess, "/alice/doc.md")
            assert len(old_shares) == 0

    @pytest.mark.asyncio
    async def test_move_no_follow_shares_stale(
        self, dfs: DatabaseFileSystem, session_factory, sharing: SharingService
    ):
        """follow=False does NOT update share paths — they become stale."""
        async with session_factory() as sess:
            await dfs.write("/alice/doc.md", "data", session=sess)
            await sharing.create_share(
                sess, "/alice/doc.md", "bob", "read", "alice"
            )

            result = await dfs.move(
                "/alice/doc.md",
                "/alice/renamed.md",
                session=sess,
                follow=False,
            )
            assert result.success is True

            # Share still points to old path (stale)
            old_shares = await sharing.list_shares_on_path(sess, "/alice/doc.md")
            assert len(old_shares) == 1

            # No share at new path
            new_shares = await sharing.list_shares_on_path(sess, "/alice/renamed.md")
            assert len(new_shares) == 0

    @pytest.mark.asyncio
    async def test_move_follow_directory(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=True on a directory moves all children in-place."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.write("/src/a.py", "file a", session=sess)
            await dfs.write("/src/b.py", "file b", session=sess)

            a_file = await dfs.metadata.get_file(sess, "/src/a.py")
            assert a_file is not None
            a_id = a_file.id

            result = await dfs.move(
                "/src", "/dst", session=sess, follow=True
            )
            assert result.success is True

            # Children moved in-place (same IDs)
            new_a = await dfs.metadata.get_file(sess, "/dst/a.py")
            assert new_a is not None
            assert new_a.id == a_id

            # Content preserved
            r = await dfs.read("/dst/b.py", session=sess)
            assert r.success
            assert r.content == "file b"

    @pytest.mark.asyncio
    async def test_move_follow_nested_dest_creates_parent_dirs(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=True to a nested path creates parent directories."""
        async with session_factory() as sess:
            await dfs.write("/src.py", "content", session=sess)

            result = await dfs.move(
                "/src.py", "/deep/nested/dest.py", session=sess, follow=True
            )
            assert result.success is True

            # Parent directories should exist
            deep = await dfs.metadata.get_file(sess, "/deep")
            assert deep is not None
            assert deep.is_directory is True

            nested = await dfs.metadata.get_file(sess, "/deep/nested")
            assert nested is not None
            assert nested.is_directory is True

            # File should be readable
            r = await dfs.read("/deep/nested/dest.py", session=sess)
            assert r.success
            assert r.content == "content"

    @pytest.mark.asyncio
    async def test_move_no_follow_directory(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=False on a directory creates new records for children."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.write("/src/a.py", "file a", session=sess)

            a_file = await dfs.metadata.get_file(sess, "/src/a.py")
            assert a_file is not None
            a_id = a_file.id

            result = await dfs.move(
                "/src", "/dst", session=sess, follow=False
            )
            assert result.success is True

            # New record at dest (different ID)
            new_a = await dfs.metadata.get_file(sess, "/dst/a.py")
            assert new_a is not None
            assert new_a.id != a_id

            # Content preserved
            r = await dfs.read("/dst/a.py", session=sess)
            assert r.success
            assert r.content == "file a"

    @pytest.mark.asyncio
    async def test_move_follow_overwrite_shares_updated(
        self, dfs: DatabaseFileSystem, session_factory, sharing: SharingService
    ):
        """follow=True overwrite: shares on src path updated to dest."""
        async with session_factory() as sess:
            await dfs.write("/alice/src.md", "source", session=sess)
            await dfs.write("/alice/dst.md", "dest", session=sess)
            await sharing.create_share(
                sess, "/alice/src.md", "bob", "read", "alice"
            )

            result = await dfs.move(
                "/alice/src.md",
                "/alice/dst.md",
                session=sess,
                follow=True,
                sharing=sharing,
            )
            assert result.success is True

            # Share moved to dest
            dst_shares = await sharing.list_shares_on_path(sess, "/alice/dst.md")
            assert len(dst_shares) == 1
            assert dst_shares[0].grantee_id == "bob"


    @pytest.mark.asyncio
    async def test_move_no_follow_directory_preserves_is_directory(
        self, dfs: DatabaseFileSystem, session_factory
    ):
        """follow=False on dir with nested subdirs preserves is_directory flag."""
        async with session_factory() as sess:
            await dfs.mkdir("/src", session=sess)
            await dfs.mkdir("/src/sub", session=sess)
            await dfs.write("/src/sub/file.py", "code", session=sess)

            result = await dfs.move(
                "/src", "/dst", session=sess, follow=False
            )
            assert result.success is True

            # Subdirectory should still be a directory
            sub = await dfs.metadata.get_file(sess, "/dst/sub")
            assert sub is not None
            assert sub.is_directory is True

            # File should not be a directory
            f = await dfs.metadata.get_file(sess, "/dst/sub/file.py")
            assert f is not None
            assert f.is_directory is False

            # Content preserved
            r = await dfs.read("/dst/sub/file.py", session=sess)
            assert r.success
            assert r.content == "code"


# ==================================================================
# VFS-level tests (authenticated mounts)
# ==================================================================


class TestVFSMoveFollow:
    """Test follow parameter through VFS with authenticated mounts."""

    @pytest.mark.asyncio
    async def test_vfs_move_follow_authenticated(self, vfs_with_sharing: VFS):
        vfs = vfs_with_sharing
        await vfs.write("/ws/notes.md", "data", user_id="alice")
        result = await vfs.move(
            "/ws/notes.md", "/ws/moved.md", user_id="alice", follow=True
        )
        assert result.success is True
        assert result.new_path == "/ws/moved.md"

        r = await vfs.read("/ws/moved.md", user_id="alice")
        assert r.success
        assert r.content == "data"

    @pytest.mark.asyncio
    async def test_vfs_move_follow_shares_updated(self, vfs_with_sharing: VFS):
        """VFS passes sharing through to backend when follow=True."""
        vfs = vfs_with_sharing
        mount = vfs._registry._mounts["/ws"]

        await vfs.write("/ws/doc.md", "content", user_id="alice")

        # Create share at stored path level
        async with vfs._session_for(mount) as sess:
            assert sess is not None
            assert mount.sharing is not None
            await mount.sharing.create_share(
                sess, "/alice/doc.md", "bob", "read", "alice"
            )

        result = await vfs.move(
            "/ws/doc.md", "/ws/renamed.md", user_id="alice", follow=True
        )
        assert result.success is True

        # Verify share updated
        async with vfs._session_for(mount) as sess:
            assert sess is not None
            assert mount.sharing is not None
            new_shares = await mount.sharing.list_shares_on_path(
                sess, "/alice/renamed.md"
            )
            assert len(new_shares) == 1
            old_shares = await mount.sharing.list_shares_on_path(
                sess, "/alice/doc.md"
            )
            assert len(old_shares) == 0

    @pytest.mark.asyncio
    async def test_vfs_move_default_no_follow(self, vfs_with_sharing: VFS):
        """Default move (follow=False) creates clean break."""
        vfs = vfs_with_sharing
        await vfs.write("/ws/old.md", "old", user_id="alice")
        result = await vfs.move(
            "/ws/old.md", "/ws/new.md", user_id="alice"
        )
        assert result.success is True

        r = await vfs.read("/ws/new.md", user_id="alice")
        assert r.success
        assert r.content == "old"

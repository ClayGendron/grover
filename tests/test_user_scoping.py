"""Tests for user-scoped file system operations via GroverAsync.

Covers GroverAsync routing with UserScopedFileSystem backend, user isolation,
and shared access via @shared/ virtual namespace.

Unit tests for path resolution helpers live in test_user_scoped_fs.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from grover.backends.database import DatabaseFileSystem
from grover.backends.user_scoped import UserScopedFileSystem
from grover.client import GroverAsync
from grover.exceptions import AuthenticationRequiredError
from grover.models.file import File
from grover.models.share import FileShare
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def auth_grover(async_engine: AsyncEngine, tmp_path: Path) -> GroverAsync:
    """GroverAsync with a single UserScopedFileSystem mount (no sharing service)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = UserScopedFileSystem()
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    await g.add_mount("/ws", backend, session_factory=session_factory)
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def shared_grover(async_engine: AsyncEngine, tmp_path: Path) -> GroverAsync:
    """GroverAsync with UserScopedFileSystem mount and sharing configured."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = UserScopedFileSystem(share_model=FileShare)
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    await g.add_mount("/ws", backend, session_factory=session_factory)
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def regular_grover(async_engine: AsyncEngine, tmp_path: Path) -> GroverAsync:
    """GroverAsync with a single plain DatabaseFileSystem mount (no user scoping)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = DatabaseFileSystem()
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    await g.add_mount("/ws", backend, session_factory=session_factory)
    yield g  # type: ignore[misc]
    await g.close()


# ---------------------------------------------------------------------------
# Integration: GroverAsync read/write with UserScopedFileSystem backend
# ---------------------------------------------------------------------------


class TestAuthenticatedReadWrite:
    async def test_write_authenticated_mount(self, auth_grover: GroverAsync):
        result = await auth_grover.write("/ws/notes.md", "hello alice", user_id="alice")
        assert result.success is True

    async def test_read_authenticated_mount(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "hello alice", user_id="alice")
        result = await auth_grover.read("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert result.content == "hello alice"

    async def test_read_authenticated_no_user_id_error(self, auth_grover: GroverAsync):
        with pytest.raises(AuthenticationRequiredError):
            await auth_grover.read("/ws/notes.md", user_id=None)

    async def test_write_sets_owner_id(self, auth_grover: GroverAsync, async_session: AsyncSession):
        await auth_grover.write("/ws/notes.md", "owned content", user_id="alice")

        # Query the file record directly to verify owner_id
        result = await async_session.execute(select(File).where(File.path == "/alice/notes.md"))
        file = result.scalar_one_or_none()
        assert file is not None
        assert file.owner_id == "alice"

    async def test_two_users_isolated(self, auth_grover: GroverAsync):
        """Two users write to the same virtual path, get different content."""
        await auth_grover.write("/ws/notes.md", "alice's notes", user_id="alice")
        await auth_grover.write("/ws/notes.md", "bob's notes", user_id="bob")

        r1 = await auth_grover.read("/ws/notes.md", user_id="alice")
        r2 = await auth_grover.read("/ws/notes.md", user_id="bob")

        assert r1.content == "alice's notes"
        assert r2.content == "bob's notes"

    async def test_user_cannot_see_other_users_files(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/secret.md", "alice only", user_id="alice")
        result = await auth_grover.read("/ws/secret.md", user_id="bob")
        assert result.success is False  # File not found for bob

    async def test_regular_mount_ignores_user_id(self, regular_grover: GroverAsync):
        """Regular (non-user-scoped) mounts ignore user_id."""
        await regular_grover.write("/ws/notes.md", "shared content", user_id="alice")
        result = await regular_grover.read("/ws/notes.md", user_id="bob")
        assert result.success is True
        assert result.content == "shared content"

    async def test_write_read_roundtrip_authenticated(self, auth_grover: GroverAsync):
        """Full write-read roundtrip with path stripping on read."""
        await auth_grover.write("/ws/project/src/main.py", "print('hello')", user_id="alice")
        result = await auth_grover.read("/ws/project/src/main.py", user_id="alice")
        assert result.success is True
        assert result.content == "print('hello')"
        # path should be user-facing (no user prefix)
        assert result.path == "/ws/project/src/main.py"

    async def test_write_no_user_id_error(self, auth_grover: GroverAsync):
        result = await auth_grover.write("/ws/notes.md", "content", user_id=None)
        assert result.success is False
        assert "user_id is required" in result.message.lower()


# ---------------------------------------------------------------------------
# Integration: edit, delete, mkdir on user-scoped mount
# ---------------------------------------------------------------------------


class TestAuthenticatedOtherOps:
    async def test_edit_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "hello world", user_id="alice")
        result = await auth_grover.edit("/ws/notes.md", "hello", "goodbye", user_id="alice")
        assert result.success is True
        read_result = await auth_grover.read("/ws/notes.md", user_id="alice")
        assert read_result.content == "goodbye world"

    async def test_delete_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        result = await auth_grover.delete("/ws/notes.md", user_id="alice")
        assert result.success is True

    async def test_mkdir_authenticated(self, auth_grover: GroverAsync):
        result = await auth_grover.mkdir("/ws/mydir", user_id="alice")
        assert result.success is True

    async def test_exists_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        assert (await auth_grover.exists("/ws/notes.md", user_id="alice")).exists is True
        assert (await auth_grover.exists("/ws/notes.md", user_id="bob")).exists is False

    async def test_get_info_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        info = await auth_grover.get_info("/ws/notes.md", user_id="alice")
        assert info.success
        assert info.path == "/ws/notes.md"

    async def test_get_info_other_user_none(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        info = await auth_grover.get_info("/ws/notes.md", user_id="bob")
        assert not info.success

    async def test_copy_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/a.md", "content", user_id="alice")
        result = await auth_grover.copy("/ws/a.md", "/ws/b.md", user_id="alice")
        assert result.success is True
        read_result = await auth_grover.read("/ws/b.md", user_id="alice")
        assert read_result.content == "content"

    async def test_move_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/old.md", "content", user_id="alice")
        result = await auth_grover.move("/ws/old.md", "/ws/new.md", user_id="alice")
        assert result.success is True
        read_result = await auth_grover.read("/ws/new.md", user_id="alice")
        assert read_result.content == "content"

    async def test_list_dir_shows_shared_entry(self, auth_grover: GroverAsync):
        """User-scoped mount root listing includes virtual @shared/ entry."""
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        result = await auth_grover.list_dir("/ws", user_id="alice")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "@shared" in names

    async def test_glob_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "content", user_id="alice")
        await auth_grover.write("/ws/notes.md", "other", user_id="bob")
        result = await auth_grover.glob("*.md", "/ws", user_id="alice")
        assert result.success is True
        paths = set(result.paths)
        assert "/ws/notes.md" in paths
        # Bob's file should not appear
        assert len(result) == 1

    async def test_tree_authenticated(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/a.md", "a", user_id="alice")
        await auth_grover.write("/ws/b.md", "b", user_id="alice")
        result = await auth_grover.tree("/ws", user_id="alice")
        assert result.success is True
        paths = set(result.paths)
        assert "/ws/a.md" in paths
        assert "/ws/b.md" in paths

    async def test_regular_mount_all_ops_unchanged(self, regular_grover: GroverAsync):
        """Regular mount operations work with user_id (ignored)."""
        await regular_grover.write("/ws/notes.md", "content", user_id="alice")
        assert (await regular_grover.exists("/ws/notes.md", user_id="bob")).exists is True
        info = await regular_grover.get_info("/ws/notes.md", user_id="bob")
        assert info.success


# ---------------------------------------------------------------------------
# Integration: @shared access with SharingService
# ---------------------------------------------------------------------------


class TestSharedAccess:
    async def _create_share(
        self,
        grover: GroverAsync,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        """Helper to create a share record directly via USFS private method."""
        mount = grover._ctx.registry.get_mount("/ws")
        assert mount is not None
        backend = mount.filesystem
        assert isinstance(backend, UserScopedFileSystem)
        assert backend._share_model is not None
        await backend._create_share(async_session, path, grantee_id, permission, granted_by)
        await async_session.commit()

    async def test_read_shared_file(self, shared_grover: GroverAsync, async_session: AsyncSession):
        """Bob reads alice's file via @shared/ path with read share."""
        await shared_grover.write("/ws/notes.md", "alice's notes", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "read")

        result = await shared_grover.read("/ws/@shared/alice/notes.md", user_id="bob")
        assert result.success is True
        assert result.content == "alice's notes"

    async def test_read_shared_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob cannot read alice's file without a share."""
        await shared_grover.write("/ws/notes.md", "alice's notes", user_id="alice")

        with pytest.raises(PermissionError, match="Access denied"):
            await shared_grover.read("/ws/@shared/alice/notes.md", user_id="bob")

    async def test_write_shared_file_with_write_perm(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob writes to alice's file via @shared/ with write share."""
        await shared_grover.write("/ws/notes.md", "original", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "write")

        result = await shared_grover.write(
            "/ws/@shared/alice/notes.md", "updated by bob", user_id="bob"
        )
        assert result.success is True

        # Alice sees bob's changes
        read_result = await shared_grover.read("/ws/notes.md", user_id="alice")
        assert read_result.content == "updated by bob"

    async def test_write_shared_file_read_only(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob cannot write to alice's file with only read share."""
        await shared_grover.write("/ws/notes.md", "original", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "read")

        result = await shared_grover.write("/ws/@shared/alice/notes.md", "hacked", user_id="bob")
        assert result.success is False
        assert "access denied" in result.message.lower()

    async def test_edit_shared_file_with_write_perm(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob edits alice's file via @shared/ with write share."""
        await shared_grover.write("/ws/notes.md", "hello world", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "write")

        result = await shared_grover.edit(
            "/ws/@shared/alice/notes.md", "hello", "goodbye", user_id="bob"
        )
        assert result.success is True

    async def test_exists_shared(self, shared_grover: GroverAsync, async_session: AsyncSession):
        """exists returns True for shared path with permission."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "read")

        result = await shared_grover.exists("/ws/@shared/alice/notes.md", user_id="bob")
        assert result.exists is True

    async def test_exists_shared_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """exists returns False for shared path without permission."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        result = await shared_grover.exists("/ws/@shared/alice/notes.md", user_id="bob")
        assert result.exists is False

    async def test_get_info_shared(self, shared_grover: GroverAsync, async_session: AsyncSession):
        """get_info works for shared paths with permission."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "read")

        info = await shared_grover.get_info("/ws/@shared/alice/notes.md", user_id="bob")
        assert info.success

    async def test_get_info_shared_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """get_info returns success=False for shared path without permission."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        info = await shared_grover.get_info("/ws/@shared/alice/notes.md", user_id="bob")
        assert not info.success

    async def test_directory_share_grants_children(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Share on /alice/projects grants read to /alice/projects/docs/file.md."""
        await shared_grover.write("/ws/projects/docs/file.md", "content", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/projects", "bob", "read")

        result = await shared_grover.read("/ws/@shared/alice/projects/docs/file.md", user_id="bob")
        assert result.success is True
        assert result.content == "content"


# ---------------------------------------------------------------------------
# Integration: @shared list_dir virtual directories
# ---------------------------------------------------------------------------


class TestSharedListDir:
    async def _create_share(
        self,
        grover: GroverAsync,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        mount = grover._ctx.registry.get_mount("/ws")
        assert mount is not None
        backend = mount.filesystem
        assert isinstance(backend, UserScopedFileSystem)
        assert backend._share_model is not None
        await backend._create_share(async_session, path, grantee_id, permission, granted_by)
        await async_session.commit()

    async def test_list_dir_shared_root(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """/@shared lists distinct owners who shared with user."""
        await shared_grover.write("/ws/a.md", "a", user_id="alice")
        await shared_grover.write("/ws/b.md", "b", user_id="charlie")
        await self._create_share(shared_grover, async_session, "/alice/a.md", "bob", "read")
        await self._create_share(
            shared_grover,
            async_session,
            "/charlie/b.md",
            "bob",
            "read",
            granted_by="charlie",
        )

        result = await shared_grover.list_dir("/ws/@shared", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "alice" in names
        assert "charlie" in names
        assert set(result.paths) == set(result.directories())

    async def test_list_dir_shared_owner(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """/@shared/{owner} lists that owner's shared content."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        await shared_grover.write("/ws/readme.md", "readme", user_id="alice")
        # Share the alice root dir so bob can list everything
        await self._create_share(shared_grover, async_session, "/alice", "bob", "read")

        result = await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "notes.md" in names
        assert "readme.md" in names

    async def test_list_dir_shared_no_sharing_configured(self, auth_grover: GroverAsync):
        """@shared list_dir with no SharingService returns empty."""
        result = await auth_grover.list_dir("/ws/@shared", user_id="alice")
        assert result.success is True
        assert len(result) == 0

    async def test_list_dir_shared_owner_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """/@shared/{owner} without share raises PermissionError."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")

    async def test_list_dir_shared_owner_file_shares(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """File-level shares show just those files when listing owner dir."""
        await shared_grover.write("/ws/doc1.md", "doc1", user_id="alice")
        await shared_grover.write("/ws/doc2.md", "doc2", user_id="alice")
        await shared_grover.write("/ws/secret.md", "secret", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/doc1.md", "bob", "read")
        await self._create_share(shared_grover, async_session, "/alice/doc2.md", "bob", "read")

        result = await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert names == {"doc1.md", "doc2.md"}
        # secret.md should NOT appear
        assert "secret.md" not in names

    async def test_list_dir_shared_owner_mixed_shares(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Directory share lists everything; file shares outside that dir also appear."""
        await shared_grover.write("/ws/projects/a.py", "a", user_id="alice")
        await shared_grover.write("/ws/projects/b.py", "b", user_id="alice")
        await shared_grover.write("/ws/readme.md", "readme", user_id="alice")
        # Directory share on /alice/projects gives full listing at that level
        await self._create_share(shared_grover, async_session, "/alice/projects", "bob", "read")
        # File share on readme
        await self._create_share(shared_grover, async_session, "/alice/readme.md", "bob", "read")

        # At the /alice level, bob should see both projects/ dir and readme.md
        result = await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "projects" in names
        assert "readme.md" in names

    async def test_list_dir_shared_deep_navigation(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Deep file share shows intermediate dirs at each level."""
        await shared_grover.write("/ws/deep/nested/file.md", "deep content", user_id="alice")
        await self._create_share(
            shared_grover, async_session, "/alice/deep/nested/file.md", "bob", "read"
        )

        # Level 1: /@shared/alice -> shows "deep/"
        result = await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert names == {"deep"}
        assert len(result.directories()) == 1

        # Level 2: /@shared/alice/deep -> shows "nested/"
        result = await shared_grover.list_dir("/ws/@shared/alice/deep", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert names == {"nested"}
        assert len(result.directories()) == 1

        # Level 3: /@shared/alice/deep/nested -> shows "file.md"
        result = await shared_grover.list_dir("/ws/@shared/alice/deep/nested", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert names == {"file.md"}
        assert len(result.files()) == 1

    async def test_list_dir_shared_directory_share_unchanged(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Existing directory share behavior is preserved (fast path)."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        await shared_grover.write("/ws/readme.md", "readme", user_id="alice")
        # Share the entire alice root
        await self._create_share(shared_grover, async_session, "/alice", "bob", "read")

        result = await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")
        assert result.success is True
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "notes.md" in names
        assert "readme.md" in names

    async def test_list_dir_shared_no_shares_still_raises(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """No shares at all still raises PermissionError."""
        await shared_grover.write("/ws/notes.md", "content", user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_grover.list_dir("/ws/@shared/alice", user_id="bob")


# ---------------------------------------------------------------------------
# Integration: move/copy via @shared paths
# ---------------------------------------------------------------------------


class TestSharedMoveAndCopy:
    async def _create_share(
        self,
        grover: GroverAsync,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        mount = grover._ctx.registry.get_mount("/ws")
        assert mount is not None
        backend = mount.filesystem
        assert isinstance(backend, UserScopedFileSystem)
        assert backend._share_model is not None
        await backend._create_share(async_session, path, grantee_id, permission, granted_by)
        await async_session.commit()

    async def test_copy_shared_file_with_read_perm(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob can copy alice's file to his own space with read share."""
        await shared_grover.write("/ws/notes.md", "alice's content", user_id="alice")
        await self._create_share(shared_grover, async_session, "/alice/notes.md", "bob", "read")

        result = await shared_grover.copy(
            "/ws/@shared/alice/notes.md", "/ws/my_copy.md", user_id="bob"
        )
        assert result.success is True
        read_result = await shared_grover.read("/ws/my_copy.md", user_id="bob")
        assert read_result.content == "alice's content"

    async def test_copy_shared_file_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob cannot copy alice's file without a share."""
        await shared_grover.write("/ws/notes.md", "alice's content", user_id="alice")
        result = await shared_grover.copy(
            "/ws/@shared/alice/notes.md", "/ws/stolen.md", user_id="bob"
        )
        assert result.success is False
        assert "access denied" in result.message.lower()

    async def test_move_shared_file_no_permission(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob cannot move alice's file without a share."""
        await shared_grover.write("/ws/notes.md", "alice's content", user_id="alice")
        result = await shared_grover.move(
            "/ws/@shared/alice/notes.md", "/ws/stolen.md", user_id="bob"
        )
        assert result.success is False
        assert "access denied" in result.message.lower()

    async def test_move_shared_file_with_write_perm(
        self, shared_grover: GroverAsync, async_session: AsyncSession
    ):
        """Bob can move alice's shared file with write permission on directory."""
        await shared_grover.write("/ws/old.md", "content", user_id="alice")
        # Directory-level share covers both source and destination
        await self._create_share(shared_grover, async_session, "/alice", "bob", "write")

        result = await shared_grover.move(
            "/ws/@shared/alice/old.md", "/ws/@shared/alice/new.md", user_id="bob"
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# Trash scoping tests
# ---------------------------------------------------------------------------


class TestTrashScoping:
    """Trash operations scoped by owner_id on user-scoped mounts."""

    async def test_list_trash_scoped_by_owner(self, auth_grover: GroverAsync):
        """Each user only sees their own trashed files."""
        await auth_grover.write("/ws/a.md", "alice's file", user_id="alice")
        await auth_grover.write("/ws/b.md", "bob's file", user_id="bob")

        await auth_grover.delete("/ws/a.md", user_id="alice")
        await auth_grover.delete("/ws/b.md", user_id="bob")

        alice_trash = await auth_grover.list_trash(user_id="alice")
        assert alice_trash.success
        assert len(alice_trash) == 1
        assert any(p.endswith("a.md") for p in alice_trash.paths)

        bob_trash = await auth_grover.list_trash(user_id="bob")
        assert bob_trash.success
        assert len(bob_trash) == 1
        assert any(p.endswith("b.md") for p in bob_trash.paths)

    async def test_list_trash_regular_mount_shows_all(self, regular_grover: GroverAsync):
        """Non-user-scoped mount shows all trashed files regardless."""
        await regular_grover.write("/ws/a.md", "file a")
        await regular_grover.write("/ws/b.md", "file b")
        await regular_grover.delete("/ws/a.md")
        await regular_grover.delete("/ws/b.md")

        trash = await regular_grover.list_trash()
        assert trash.success
        assert len(trash) == 2

    async def test_restore_own_file(self, auth_grover: GroverAsync):
        """User can restore their own trashed file."""
        await auth_grover.write("/ws/mine.md", "my data", user_id="alice")
        await auth_grover.delete("/ws/mine.md", user_id="alice")

        result = await auth_grover.restore_from_trash("/ws/mine.md", user_id="alice")
        assert result.success is True

        r = await auth_grover.read("/ws/mine.md", user_id="alice")
        assert r.success
        assert r.content == "my data"

    async def test_restore_other_user_denied(self, auth_grover: GroverAsync):
        """User cannot restore another user's trashed file."""
        await auth_grover.write("/ws/secret.md", "alice's secret", user_id="alice")
        await auth_grover.delete("/ws/secret.md", user_id="alice")

        result = await auth_grover.restore_from_trash("/ws/secret.md", user_id="bob")
        assert result.success is False
        assert "not in trash" in result.message.lower()

    async def test_empty_trash_scoped(self, auth_grover: GroverAsync):
        """Emptying trash only deletes the requesting user's files."""
        await auth_grover.write("/ws/a.md", "alice", user_id="alice")
        await auth_grover.write("/ws/b.md", "bob", user_id="bob")
        await auth_grover.delete("/ws/a.md", user_id="alice")
        await auth_grover.delete("/ws/b.md", user_id="bob")

        # Alice empties her trash
        result = await auth_grover.empty_trash(user_id="alice")
        assert result.success
        assert result.total_deleted == 1

        # Bob's trash still has his file
        bob_trash = await auth_grover.list_trash(user_id="bob")
        assert bob_trash.success
        assert len(bob_trash) == 1
        assert any(p.endswith("b.md") for p in bob_trash.paths)

        # Alice's trash is now empty
        alice_trash = await auth_grover.list_trash(user_id="alice")
        assert alice_trash.success
        assert len(alice_trash) == 0

    async def test_empty_trash_regular_mount_deletes_all(self, regular_grover: GroverAsync):
        """Non-user-scoped mount empties all trash."""
        await regular_grover.write("/ws/a.md", "file a")
        await regular_grover.write("/ws/b.md", "file b")
        await regular_grover.delete("/ws/a.md")
        await regular_grover.delete("/ws/b.md")

        result = await regular_grover.empty_trash()
        assert result.success
        assert result.total_deleted == 2

        trash = await regular_grover.list_trash()
        assert len(trash) == 0

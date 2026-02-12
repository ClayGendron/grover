"""Tests for user-scoped file system operations.

Covers VFS path resolution, authenticated mounts, user isolation,
and shared access via @shared/ virtual namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from grover.events import EventBus
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import AuthenticationRequiredError
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.sharing import SharingService
from grover.fs.vfs import VFS
from grover.models.files import File
from grover.models.shares import FileShare

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def auth_vfs(async_engine: AsyncEngine) -> VFS:
    """VFS with a single authenticated DatabaseFileSystem mount."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = DatabaseFileSystem()
    registry = MountRegistry()

    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    config = MountConfig(
        mount_path="/ws",
        backend=backend,
        session_factory=session_factory,
        authenticated=True,
    )
    registry.add_mount(config)
    return VFS(registry, EventBus())


@pytest.fixture
async def shared_vfs(async_engine: AsyncEngine) -> VFS:
    """VFS with authenticated mount and SharingService configured."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = DatabaseFileSystem()
    registry = MountRegistry()
    sharing = SharingService(FileShare)

    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    config = MountConfig(
        mount_path="/ws",
        backend=backend,
        session_factory=session_factory,
        authenticated=True,
        sharing=sharing,
    )
    registry.add_mount(config)
    return VFS(registry, EventBus())


@pytest.fixture
async def regular_vfs(async_engine: AsyncEngine) -> VFS:
    """VFS with a single non-authenticated DatabaseFileSystem mount."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    backend = DatabaseFileSystem()
    registry = MountRegistry()

    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    config = MountConfig(
        mount_path="/ws",
        backend=backend,
        session_factory=session_factory,
        authenticated=False,
    )
    registry.add_mount(config)
    return VFS(registry, EventBus())


# ---------------------------------------------------------------------------
# _resolve_user_path unit tests
# ---------------------------------------------------------------------------


class TestResolveUserPath:
    def test_regular_mount_passthrough(self):
        """Non-authenticated mounts pass paths through unchanged."""
        registry = MountRegistry()
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=False,
        )
        registry.add_mount(config)
        vfs = VFS(registry)

        result = vfs._resolve_user_path(config, "/notes.md", user_id="alice")
        assert result == "/notes.md"

    def test_prepends_user_id(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        result = vfs._resolve_user_path(config, "/notes.md", user_id="alice")
        assert result == "/alice/notes.md"

    def test_root_path(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        result = vfs._resolve_user_path(config, "/", user_id="alice")
        assert result == "/alice"

    def test_no_user_id_raises(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        with pytest.raises(AuthenticationRequiredError):
            vfs._resolve_user_path(config, "/notes.md", user_id=None)

    def test_empty_user_id_raises(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        with pytest.raises(AuthenticationRequiredError):
            vfs._resolve_user_path(config, "/notes.md", user_id="")

    def test_shared_access(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        result = vfs._resolve_user_path(
            config, "/@shared/alice/notes.md", user_id="bob"
        )
        assert result == "/alice/notes.md"

    def test_shared_access_root(self):
        config = MountConfig(
            mount_path="/ws",
            backend=DatabaseFileSystem(),
            authenticated=True,
        )
        vfs = VFS(MountRegistry())

        result = vfs._resolve_user_path(
            config, "/@shared/alice", user_id="bob"
        )
        assert result == "/alice"


# ---------------------------------------------------------------------------
# _strip_user_prefix unit tests
# ---------------------------------------------------------------------------


class TestStripUserPrefix:
    def test_strip_prefix(self):
        assert VFS._strip_user_prefix("/alice/notes.md", "alice") == "/notes.md"

    def test_strip_prefix_root(self):
        assert VFS._strip_user_prefix("/alice", "alice") == "/"

    def test_no_match(self):
        assert VFS._strip_user_prefix("/bob/notes.md", "alice") == "/bob/notes.md"

    def test_nested_path(self):
        result = VFS._strip_user_prefix("/alice/projects/src/main.py", "alice")
        assert result == "/projects/src/main.py"


# ---------------------------------------------------------------------------
# _is_shared_access unit tests
# ---------------------------------------------------------------------------


class TestIsSharedAccess:
    def test_shared_path(self):
        is_shared, owner, rest = VFS._is_shared_access("/@shared/alice/notes.md")
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/notes.md"

    def test_shared_root(self):
        is_shared, owner, rest = VFS._is_shared_access("/@shared/alice")
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/"

    def test_not_shared(self):
        is_shared, owner, rest = VFS._is_shared_access("/notes.md")
        assert is_shared is False
        assert owner is None
        assert rest is None

    def test_shared_nested(self):
        is_shared, owner, rest = VFS._is_shared_access("/@shared/alice/projects/docs/file.md")
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/projects/docs/file.md"


# ---------------------------------------------------------------------------
# Integration: VFS read/write with authenticated mount
# ---------------------------------------------------------------------------


class TestVFSAuthenticatedReadWrite:
    async def test_vfs_write_authenticated_mount(self, auth_vfs: VFS):
        result = await auth_vfs.write("/ws/notes.md", "hello alice", user_id="alice")
        assert result.success is True

    async def test_vfs_read_authenticated_mount(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "hello alice", user_id="alice")
        result = await auth_vfs.read("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert result.content == "hello alice"

    async def test_vfs_read_authenticated_no_user_id_error(self, auth_vfs: VFS):
        with pytest.raises(AuthenticationRequiredError):
            await auth_vfs.read("/ws/notes.md", user_id=None)

    async def test_vfs_write_sets_owner_id(
        self, auth_vfs: VFS, async_session: AsyncSession
    ):
        await auth_vfs.write("/ws/notes.md", "owned content", user_id="alice")

        # Query the file record directly to verify owner_id
        result = await async_session.execute(
            select(File).where(File.path == "/alice/notes.md")
        )
        file = result.scalar_one_or_none()
        assert file is not None
        assert file.owner_id == "alice"

    async def test_two_users_isolated(self, auth_vfs: VFS):
        """Two users write to the same virtual path, get different content."""
        await auth_vfs.write("/ws/notes.md", "alice's notes", user_id="alice")
        await auth_vfs.write("/ws/notes.md", "bob's notes", user_id="bob")

        r1 = await auth_vfs.read("/ws/notes.md", user_id="alice")
        r2 = await auth_vfs.read("/ws/notes.md", user_id="bob")

        assert r1.content == "alice's notes"
        assert r2.content == "bob's notes"

    async def test_user_cannot_see_other_users_files(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/secret.md", "alice only", user_id="alice")
        result = await auth_vfs.read("/ws/secret.md", user_id="bob")
        assert result.success is False  # File not found for bob

    async def test_regular_mount_ignores_user_id(self, regular_vfs: VFS):
        """Regular (non-authenticated) mounts ignore user_id."""
        await regular_vfs.write("/ws/notes.md", "shared content", user_id="alice")
        result = await regular_vfs.read("/ws/notes.md", user_id="bob")
        assert result.success is True
        assert result.content == "shared content"

    async def test_write_read_roundtrip_authenticated(self, auth_vfs: VFS):
        """Full write-read roundtrip with path stripping on read."""
        await auth_vfs.write("/ws/project/src/main.py", "print('hello')", user_id="alice")
        result = await auth_vfs.read("/ws/project/src/main.py", user_id="alice")
        assert result.success is True
        assert result.content == "print('hello')"
        # file_path should be user-facing (no user prefix)
        assert result.file_path == "/ws/project/src/main.py"

    async def test_write_no_user_id_error(self, auth_vfs: VFS):
        with pytest.raises(AuthenticationRequiredError):
            await auth_vfs.write("/ws/notes.md", "content", user_id=None)


# ---------------------------------------------------------------------------
# Integration: edit, delete, mkdir on authenticated mount
# ---------------------------------------------------------------------------


class TestVFSAuthenticatedOtherOps:
    async def test_edit_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "hello world", user_id="alice")
        result = await auth_vfs.edit(
            "/ws/notes.md", "hello", "goodbye", user_id="alice"
        )
        assert result.success is True
        read_result = await auth_vfs.read("/ws/notes.md", user_id="alice")
        assert read_result.content == "goodbye world"

    async def test_delete_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        result = await auth_vfs.delete("/ws/notes.md", user_id="alice")
        assert result.success is True

    async def test_mkdir_authenticated(self, auth_vfs: VFS):
        result = await auth_vfs.mkdir("/ws/mydir", user_id="alice")
        assert result.success is True

    async def test_exists_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        assert await auth_vfs.exists("/ws/notes.md", user_id="alice") is True
        assert await auth_vfs.exists("/ws/notes.md", user_id="bob") is False

    async def test_get_info_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        info = await auth_vfs.get_info("/ws/notes.md", user_id="alice")
        assert info is not None
        assert info.path == "/ws/notes.md"

    async def test_get_info_other_user_none(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        info = await auth_vfs.get_info("/ws/notes.md", user_id="bob")
        assert info is None

    async def test_copy_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/a.md", "content", user_id="alice")
        result = await auth_vfs.copy("/ws/a.md", "/ws/b.md", user_id="alice")
        assert result.success is True
        read_result = await auth_vfs.read("/ws/b.md", user_id="alice")
        assert read_result.content == "content"

    async def test_move_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/old.md", "content", user_id="alice")
        result = await auth_vfs.move("/ws/old.md", "/ws/new.md", user_id="alice")
        assert result.success is True
        read_result = await auth_vfs.read("/ws/new.md", user_id="alice")
        assert read_result.content == "content"

    async def test_list_dir_shows_shared_entry(self, auth_vfs: VFS):
        """Authenticated mount root listing includes virtual @shared/ entry."""
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        result = await auth_vfs.list_dir("/ws", user_id="alice")
        names = {e.name for e in result.entries}
        assert "@shared" in names

    async def test_glob_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/notes.md", "content", user_id="alice")
        await auth_vfs.write("/ws/notes.md", "other", user_id="bob")
        result = await auth_vfs.glob("*.md", "/ws", user_id="alice")
        assert result.success is True
        paths = {e.path for e in result.entries}
        assert "/ws/notes.md" in paths
        # Bob's file should not appear
        assert len(result.entries) == 1

    async def test_tree_authenticated(self, auth_vfs: VFS):
        await auth_vfs.write("/ws/a.md", "a", user_id="alice")
        await auth_vfs.write("/ws/b.md", "b", user_id="alice")
        result = await auth_vfs.tree("/ws", user_id="alice")
        assert result.success is True
        paths = {e.path for e in result.entries}
        assert "/ws/a.md" in paths
        assert "/ws/b.md" in paths

    async def test_regular_mount_all_ops_unchanged(self, regular_vfs: VFS):
        """Regular mount operations work with user_id (ignored)."""
        await regular_vfs.write("/ws/notes.md", "content", user_id="alice")
        assert await regular_vfs.exists("/ws/notes.md", user_id="bob") is True
        info = await regular_vfs.get_info("/ws/notes.md", user_id="bob")
        assert info is not None


# ---------------------------------------------------------------------------
# Integration: @shared access with SharingService
# ---------------------------------------------------------------------------


class TestVFSSharedAccess:
    async def _create_share(
        self,
        vfs: VFS,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        """Helper to create a share record directly via SharingService."""
        mount = vfs._registry.list_mounts()[0]
        assert mount.sharing is not None
        await mount.sharing.create_share(
            async_session, path, grantee_id, permission, granted_by
        )
        await async_session.commit()

    async def test_read_shared_file(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob reads alice's file via @shared/ path with read share."""
        await shared_vfs.write("/ws/notes.md", "alice's notes", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "read"
        )

        result = await shared_vfs.read(
            "/ws/@shared/alice/notes.md", user_id="bob"
        )
        assert result.success is True
        assert result.content == "alice's notes"

    async def test_read_shared_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob cannot read alice's file without a share."""
        await shared_vfs.write("/ws/notes.md", "alice's notes", user_id="alice")

        with pytest.raises(PermissionError, match="Access denied"):
            await shared_vfs.read("/ws/@shared/alice/notes.md", user_id="bob")

    async def test_write_shared_file_with_write_perm(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob writes to alice's file via @shared/ with write share."""
        await shared_vfs.write("/ws/notes.md", "original", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "write"
        )

        result = await shared_vfs.write(
            "/ws/@shared/alice/notes.md", "updated by bob", user_id="bob"
        )
        assert result.success is True

        # Alice sees bob's changes
        read_result = await shared_vfs.read("/ws/notes.md", user_id="alice")
        assert read_result.content == "updated by bob"

    async def test_write_shared_file_read_only(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob cannot write to alice's file with only read share."""
        await shared_vfs.write("/ws/notes.md", "original", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "read"
        )

        with pytest.raises(PermissionError, match="Access denied"):
            await shared_vfs.write(
                "/ws/@shared/alice/notes.md", "hacked", user_id="bob"
            )

    async def test_edit_shared_file_with_write_perm(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob edits alice's file via @shared/ with write share."""
        await shared_vfs.write("/ws/notes.md", "hello world", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "write"
        )

        result = await shared_vfs.edit(
            "/ws/@shared/alice/notes.md", "hello", "goodbye", user_id="bob"
        )
        assert result.success is True

    async def test_exists_shared(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """exists returns True for shared path with permission."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "read"
        )

        assert await shared_vfs.exists(
            "/ws/@shared/alice/notes.md", user_id="bob"
        ) is True

    async def test_exists_shared_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """exists returns False for shared path without permission."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        assert await shared_vfs.exists(
            "/ws/@shared/alice/notes.md", user_id="bob"
        ) is False

    async def test_get_info_shared(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """get_info works for shared paths with permission."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "read"
        )

        info = await shared_vfs.get_info(
            "/ws/@shared/alice/notes.md", user_id="bob"
        )
        assert info is not None

    async def test_get_info_shared_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """get_info returns None for shared path without permission."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        info = await shared_vfs.get_info(
            "/ws/@shared/alice/notes.md", user_id="bob"
        )
        assert info is None

    async def test_directory_share_grants_children(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Share on /alice/projects grants read to /alice/projects/docs/file.md."""
        await shared_vfs.write(
            "/ws/projects/docs/file.md", "content", user_id="alice"
        )
        await self._create_share(
            shared_vfs, async_session, "/alice/projects", "bob", "read"
        )

        result = await shared_vfs.read(
            "/ws/@shared/alice/projects/docs/file.md", user_id="bob"
        )
        assert result.success is True
        assert result.content == "content"


# ---------------------------------------------------------------------------
# Integration: @shared list_dir virtual directories
# ---------------------------------------------------------------------------


class TestVFSSharedListDir:
    async def _create_share(
        self,
        vfs: VFS,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        mount = vfs._registry.list_mounts()[0]
        assert mount.sharing is not None
        await mount.sharing.create_share(
            async_session, path, grantee_id, permission, granted_by
        )
        await async_session.commit()

    async def test_list_dir_shared_root(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """/@shared lists distinct owners who shared with user."""
        await shared_vfs.write("/ws/a.md", "a", user_id="alice")
        await shared_vfs.write("/ws/b.md", "b", user_id="charlie")
        await self._create_share(
            shared_vfs, async_session, "/alice/a.md", "bob", "read"
        )
        await self._create_share(
            shared_vfs, async_session, "/charlie/b.md", "bob", "read",
            granted_by="charlie",
        )

        result = await shared_vfs.list_dir("/ws/@shared", user_id="bob")
        assert result.success is True
        names = {e.name for e in result.entries}
        assert "alice" in names
        assert "charlie" in names
        assert all(e.is_directory for e in result.entries)

    async def test_list_dir_shared_owner(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """/@shared/{owner} lists that owner's shared content."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        await shared_vfs.write("/ws/readme.md", "readme", user_id="alice")
        # Share the alice root dir so bob can list everything
        await self._create_share(
            shared_vfs, async_session, "/alice", "bob", "read"
        )

        result = await shared_vfs.list_dir(
            "/ws/@shared/alice", user_id="bob"
        )
        assert result.success is True
        names = {e.name for e in result.entries}
        assert "notes.md" in names
        assert "readme.md" in names

    async def test_list_dir_shared_no_sharing_configured(
        self, auth_vfs: VFS
    ):
        """@shared list_dir with no SharingService returns empty."""
        result = await auth_vfs.list_dir("/ws/@shared", user_id="alice")
        assert result.success is True
        assert result.entries == []

    async def test_list_dir_shared_owner_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """/@shared/{owner} without share raises PermissionError."""
        await shared_vfs.write("/ws/notes.md", "content", user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_vfs.list_dir(
                "/ws/@shared/alice", user_id="bob"
            )


# ---------------------------------------------------------------------------
# Integration: move/copy via @shared paths
# ---------------------------------------------------------------------------


class TestVFSSharedMoveAndCopy:
    async def _create_share(
        self,
        vfs: VFS,
        async_session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        mount = vfs._registry.list_mounts()[0]
        assert mount.sharing is not None
        await mount.sharing.create_share(
            async_session, path, grantee_id, permission, granted_by
        )
        await async_session.commit()

    async def test_copy_shared_file_with_read_perm(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob can copy alice's file to his own space with read share."""
        await shared_vfs.write("/ws/notes.md", "alice's content", user_id="alice")
        await self._create_share(
            shared_vfs, async_session, "/alice/notes.md", "bob", "read"
        )

        result = await shared_vfs.copy(
            "/ws/@shared/alice/notes.md", "/ws/my_copy.md", user_id="bob"
        )
        assert result.success is True
        read_result = await shared_vfs.read("/ws/my_copy.md", user_id="bob")
        assert read_result.content == "alice's content"

    async def test_copy_shared_file_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob cannot copy alice's file without a share."""
        await shared_vfs.write("/ws/notes.md", "alice's content", user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_vfs.copy(
                "/ws/@shared/alice/notes.md", "/ws/stolen.md", user_id="bob"
            )

    async def test_move_shared_file_no_permission(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob cannot move alice's file without a share."""
        await shared_vfs.write("/ws/notes.md", "alice's content", user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_vfs.move(
                "/ws/@shared/alice/notes.md", "/ws/stolen.md", user_id="bob"
            )

    async def test_move_shared_file_with_write_perm(
        self, shared_vfs: VFS, async_session: AsyncSession
    ):
        """Bob can move alice's shared file with write permission on directory."""
        await shared_vfs.write("/ws/old.md", "content", user_id="alice")
        # Directory-level share covers both source and destination
        await self._create_share(
            shared_vfs, async_session, "/alice", "bob", "write"
        )

        result = await shared_vfs.move(
            "/ws/@shared/alice/old.md", "/ws/@shared/alice/new.md", user_id="bob"
        )
        assert result.success is True

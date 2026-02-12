"""Tests for user-scoped file system operations.

Covers VFS path resolution, authenticated mounts, and user isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from grover.events import EventBus
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import AuthenticationRequiredError
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.vfs import VFS
from grover.models.files import File

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

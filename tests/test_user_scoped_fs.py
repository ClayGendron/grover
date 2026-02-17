"""Tests for UserScopedFileSystem in isolation (no VFS).

Exercises user-scoping, path resolution, sharing, trash scoping,
versioning, and search operations directly on the backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from grover.fs.exceptions import AuthenticationRequiredError
from grover.fs.sharing import SharingService
from grover.fs.types import ShareInfo
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.models.files import File
from grover.models.shares import FileShare

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def usfs(async_engine: AsyncEngine) -> UserScopedFileSystem:
    """UserScopedFileSystem with no sharing service."""
    return UserScopedFileSystem()


@pytest.fixture
async def shared_usfs(async_engine: AsyncEngine) -> UserScopedFileSystem:
    """UserScopedFileSystem with SharingService configured."""
    sharing = SharingService(FileShare)
    return UserScopedFileSystem(sharing=sharing)


@pytest.fixture
async def session_factory(async_engine: AsyncEngine):
    """Return an async session factory."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )


# ---------------------------------------------------------------------------
# Path resolution unit tests
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_resolve_regular_path(self):
        result = UserScopedFileSystem._resolve_path("/notes.md", "alice")
        assert result == "/alice/notes.md"

    def test_resolve_root_path(self):
        result = UserScopedFileSystem._resolve_path("/", "alice")
        assert result == "/alice"

    def test_resolve_shared_path(self):
        result = UserScopedFileSystem._resolve_path(
            "/@shared/alice/notes.md", "bob"
        )
        assert result == "/alice/notes.md"

    def test_resolve_shared_root(self):
        result = UserScopedFileSystem._resolve_path("/@shared/alice", "bob")
        assert result == "/alice"

    def test_resolve_nested_path(self):
        result = UserScopedFileSystem._resolve_path(
            "/projects/src/main.py", "alice"
        )
        assert result == "/alice/projects/src/main.py"


class TestStripUserPrefix:
    def test_strip_prefix(self):
        assert UserScopedFileSystem._strip_user_prefix(
            "/alice/notes.md", "alice"
        ) == "/notes.md"

    def test_strip_prefix_root(self):
        assert UserScopedFileSystem._strip_user_prefix(
            "/alice", "alice"
        ) == "/"

    def test_no_match(self):
        assert UserScopedFileSystem._strip_user_prefix(
            "/bob/notes.md", "alice"
        ) == "/bob/notes.md"

    def test_nested_path(self):
        result = UserScopedFileSystem._strip_user_prefix(
            "/alice/projects/src/main.py", "alice"
        )
        assert result == "/projects/src/main.py"


class TestIsSharedAccess:
    def test_shared_path(self):
        is_shared, owner, rest = UserScopedFileSystem._is_shared_access(
            "/@shared/alice/notes.md"
        )
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/notes.md"

    def test_shared_root(self):
        is_shared, owner, rest = UserScopedFileSystem._is_shared_access(
            "/@shared/alice"
        )
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/"

    def test_not_shared(self):
        is_shared, owner, rest = UserScopedFileSystem._is_shared_access(
            "/notes.md"
        )
        assert is_shared is False
        assert owner is None
        assert rest is None

    def test_shared_nested(self):
        is_shared, owner, rest = UserScopedFileSystem._is_shared_access(
            "/@shared/alice/projects/docs/file.md"
        )
        assert is_shared is True
        assert owner == "alice"
        assert rest == "/projects/docs/file.md"


class TestRequireUserId:
    def test_valid_user_id(self):
        assert UserScopedFileSystem._require_user_id("alice") == "alice"

    def test_none_raises(self):
        with pytest.raises(AuthenticationRequiredError):
            UserScopedFileSystem._require_user_id(None)

    def test_empty_raises(self):
        with pytest.raises(AuthenticationRequiredError):
            UserScopedFileSystem._require_user_id("")


# ---------------------------------------------------------------------------
# Read/Write operations
# ---------------------------------------------------------------------------


class TestReadWrite:
    async def test_write_read_roundtrip(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        result = await usfs.write(
            "/notes.md", "hello", session=async_session, user_id="alice"
        )
        assert result.success is True
        assert result.file_path == "/notes.md"

        read_result = await usfs.read(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert read_result.success is True
        assert read_result.content == "hello"
        assert read_result.file_path == "/notes.md"

    async def test_two_users_isolated(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "alice's notes",
            session=async_session, user_id="alice",
        )
        await usfs.write(
            "/notes.md", "bob's notes",
            session=async_session, user_id="bob",
        )

        r1 = await usfs.read("/notes.md", session=async_session, user_id="alice")
        r2 = await usfs.read("/notes.md", session=async_session, user_id="bob")

        assert r1.content == "alice's notes"
        assert r2.content == "bob's notes"

    async def test_user_cannot_see_other_user(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/secret.md", "alice only",
            session=async_session, user_id="alice",
        )
        result = await usfs.read(
            "/secret.md", session=async_session, user_id="bob"
        )
        assert result.success is False

    async def test_write_sets_owner_id(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "owned content",
            session=async_session, user_id="alice",
        )
        result = await async_session.execute(
            select(File).where(File.path == "/alice/notes.md")
        )
        file = result.scalar_one_or_none()
        assert file is not None
        assert file.owner_id == "alice"

    async def test_no_user_id_raises_on_read(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        with pytest.raises(AuthenticationRequiredError):
            await usfs.read("/notes.md", session=async_session, user_id=None)

    async def test_no_user_id_raises_on_write(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        with pytest.raises(AuthenticationRequiredError):
            await usfs.write(
                "/notes.md", "content",
                session=async_session, user_id=None,
            )


# ---------------------------------------------------------------------------
# Other CRUD operations
# ---------------------------------------------------------------------------


class TestOperations:
    async def test_edit(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        result = await usfs.edit(
            "/notes.md", "hello", "goodbye",
            session=async_session, user_id="alice",
        )
        assert result.success is True
        read_result = await usfs.read(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert read_result.content == "goodbye world"

    async def test_delete(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        result = await usfs.delete(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert result.success is True

    async def test_mkdir(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        result = await usfs.mkdir(
            "/mydir", session=async_session, user_id="alice"
        )
        assert result.success is True
        assert result.path == "/mydir"

    async def test_mkdir_sets_owner_id(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.mkdir(
            "/mydir", session=async_session, user_id="alice"
        )
        result = await async_session.execute(
            select(File).where(File.path == "/alice/mydir")
        )
        file = result.scalar_one_or_none()
        assert file is not None
        assert file.owner_id == "alice"

    async def test_exists(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        assert await usfs.exists(
            "/notes.md", session=async_session, user_id="alice"
        ) is True
        assert await usfs.exists(
            "/notes.md", session=async_session, user_id="bob"
        ) is False

    async def test_get_info(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        info = await usfs.get_info(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert info is not None
        assert info.path == "/notes.md"

    async def test_get_info_other_user_none(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        info = await usfs.get_info(
            "/notes.md", session=async_session, user_id="bob"
        )
        assert info is None

    async def test_copy(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/a.md", "content",
            session=async_session, user_id="alice",
        )
        result = await usfs.copy(
            "/a.md", "/b.md",
            session=async_session, user_id="alice",
        )
        assert result.success is True
        read_result = await usfs.read(
            "/b.md", session=async_session, user_id="alice"
        )
        assert read_result.content == "content"

    async def test_move(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/old.md", "content",
            session=async_session, user_id="alice",
        )
        result = await usfs.move(
            "/old.md", "/new.md",
            session=async_session, user_id="alice",
        )
        assert result.success is True
        read_result = await usfs.read(
            "/new.md", session=async_session, user_id="alice"
        )
        assert read_result.content == "content"

    async def test_glob(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await usfs.write(
            "/notes.md", "other",
            session=async_session, user_id="bob",
        )
        result = await usfs.glob(
            "*.md", "/", session=async_session, user_id="alice"
        )
        assert result.success is True
        paths = {e.path for e in result.entries}
        assert "/notes.md" in paths
        assert len(result.entries) == 1

    async def test_grep(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "hello world\nfoo bar",
            session=async_session, user_id="alice",
        )
        result = await usfs.grep(
            "hello", "/", session=async_session, user_id="alice"
        )
        assert result.success is True
        assert len(result.matches) == 1
        assert result.matches[0].file_path == "/notes.md"

    async def test_grep_strips_user_prefix_from_matches(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/src/main.py", "import os",
            session=async_session, user_id="alice",
        )
        result = await usfs.grep(
            "import", "/", session=async_session, user_id="alice"
        )
        assert result.success is True
        for m in result.matches:
            assert not m.file_path.startswith("/alice/")

    async def test_tree(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/a.md", "a", session=async_session, user_id="alice"
        )
        await usfs.write(
            "/b.md", "b", session=async_session, user_id="alice"
        )
        result = await usfs.tree(
            "/", session=async_session, user_id="alice"
        )
        assert result.success is True
        paths = {e.path for e in result.entries}
        assert "/a.md" in paths
        assert "/b.md" in paths

    async def test_grep_with_glob_filter(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        await usfs.write(
            "/data.py", "hello python",
            session=async_session, user_id="alice",
        )
        result = await usfs.grep(
            "hello", "/",
            glob_filter="*.md",
            session=async_session, user_id="alice",
        )
        assert result.success is True
        assert len(result.matches) == 1
        assert result.matches[0].file_path == "/notes.md"

    async def test_list_dir_shows_shared_entry(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        result = await usfs.list_dir(
            "/", session=async_session, user_id="alice"
        )
        names = {e.name for e in result.entries}
        assert "@shared" in names


# ---------------------------------------------------------------------------
# Shared access tests
# ---------------------------------------------------------------------------


class TestSharedAccess:
    async def _create_share(
        self,
        usfs: UserScopedFileSystem,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        assert usfs._sharing is not None
        await usfs._sharing.create_share(
            session, path, grantee_id, permission, granted_by
        )

    async def test_read_shared_file(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice's notes",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        result = await shared_usfs.read(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert result.content == "alice's notes"

    async def test_read_shared_no_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice's notes",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.read(
                "/@shared/alice/notes.md",
                session=async_session, user_id="bob",
            )

    async def test_write_shared_with_write_perm(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "original",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "write",
        )

        result = await shared_usfs.write(
            "/@shared/alice/notes.md", "updated by bob",
            session=async_session, user_id="bob",
        )
        assert result.success is True

        read_result = await shared_usfs.read(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert read_result.content == "updated by bob"

    async def test_write_shared_read_only_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "original",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.write(
                "/@shared/alice/notes.md", "hacked",
                session=async_session, user_id="bob",
            )

    async def test_edit_shared_with_write_perm(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "write",
        )

        result = await shared_usfs.edit(
            "/@shared/alice/notes.md", "hello", "goodbye",
            session=async_session, user_id="bob",
        )
        assert result.success is True

    async def test_exists_shared(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        assert await shared_usfs.exists(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        ) is True

    async def test_exists_shared_no_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        assert await shared_usfs.exists(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        ) is False

    async def test_get_info_shared(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        info = await shared_usfs.get_info(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert info is not None

    async def test_get_info_shared_no_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        info = await shared_usfs.get_info(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert info is None

    async def test_directory_share_grants_children(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/projects/docs/file.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/projects", "bob", "read",
        )

        result = await shared_usfs.read(
            "/@shared/alice/projects/docs/file.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert result.content == "content"

    async def test_shared_read_returns_shared_path(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        """Result file_path should be the @shared path, not internal stored path."""
        await shared_usfs.write(
            "/notes.md", "alice's notes",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        result = await shared_usfs.read(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert result.file_path == "/@shared/alice/notes.md"

    async def test_shared_write_returns_shared_path(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "original",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "write",
        )

        result = await shared_usfs.write(
            "/@shared/alice/notes.md", "updated",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert result.file_path == "/@shared/alice/notes.md"

    async def test_shared_get_info_returns_shared_path(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/notes.md", "bob", "read",
        )

        info = await shared_usfs.get_info(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert info is not None
        assert info.path == "/@shared/alice/notes.md"

    async def test_shared_access_denied_without_sharing_service(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        """@shared paths are denied when no SharingService is configured."""
        await usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="sharing is not configured"):
            await usfs.read(
                "/@shared/alice/notes.md",
                session=async_session, user_id="bob",
            )


# ---------------------------------------------------------------------------
# Shared list_dir tests
# ---------------------------------------------------------------------------


class TestSharedListDir:
    async def _create_share(
        self,
        usfs: UserScopedFileSystem,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        assert usfs._sharing is not None
        await usfs._sharing.create_share(
            session, path, grantee_id, permission, granted_by
        )

    async def test_shared_root_lists_owners(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/a.md", "a", session=async_session, user_id="alice"
        )
        await shared_usfs.write(
            "/b.md", "b", session=async_session, user_id="charlie"
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/a.md", "bob", "read"
        )
        await self._create_share(
            shared_usfs, async_session, "/charlie/b.md", "bob", "read",
            granted_by="charlie",
        )

        result = await shared_usfs.list_dir(
            "/@shared", session=async_session, user_id="bob"
        )
        assert result.success is True
        names = {e.name for e in result.entries}
        assert "alice" in names
        assert "charlie" in names
        assert all(e.is_directory for e in result.entries)

    async def test_shared_owner_level(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await shared_usfs.write(
            "/readme.md", "readme",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )

        result = await shared_usfs.list_dir(
            "/@shared/alice", session=async_session, user_id="bob"
        )
        assert result.success is True
        names = {e.name for e in result.entries}
        assert "notes.md" in names
        assert "readme.md" in names

    async def test_shared_no_sharing_configured(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        result = await usfs.list_dir(
            "/@shared", session=async_session, user_id="alice"
        )
        assert result.success is True
        assert result.entries == []

    async def test_shared_owner_no_permission_raises(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.list_dir(
                "/@shared/alice", session=async_session, user_id="bob"
            )

    async def test_file_level_shares_filtered(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/doc1.md", "doc1", session=async_session, user_id="alice"
        )
        await shared_usfs.write(
            "/doc2.md", "doc2", session=async_session, user_id="alice"
        )
        await shared_usfs.write(
            "/secret.md", "secret", session=async_session, user_id="alice"
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/doc1.md", "bob", "read"
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/doc2.md", "bob", "read"
        )

        result = await shared_usfs.list_dir(
            "/@shared/alice", session=async_session, user_id="bob"
        )
        assert result.success is True
        names = {e.name for e in result.entries}
        assert names == {"doc1.md", "doc2.md"}

    async def test_deep_navigation(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/deep/nested/file.md", "deep content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/deep/nested/file.md", "bob", "read",
        )

        # Level 1
        result = await shared_usfs.list_dir(
            "/@shared/alice", session=async_session, user_id="bob"
        )
        assert result.success is True
        names = {e.name for e in result.entries}
        assert names == {"deep"}
        assert result.entries[0].is_directory is True

        # Level 2
        result = await shared_usfs.list_dir(
            "/@shared/alice/deep", session=async_session, user_id="bob"
        )
        names = {e.name for e in result.entries}
        assert names == {"nested"}

        # Level 3
        result = await shared_usfs.list_dir(
            "/@shared/alice/deep/nested",
            session=async_session, user_id="bob",
        )
        names = {e.name for e in result.entries}
        assert names == {"file.md"}
        assert result.entries[0].is_directory is False


# ---------------------------------------------------------------------------
# Trash scoping tests
# ---------------------------------------------------------------------------


class TestTrashScoping:
    async def test_list_trash_scoped_by_owner(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/a.md", "alice's file",
            session=async_session, user_id="alice",
        )
        await usfs.write(
            "/b.md", "bob's file",
            session=async_session, user_id="bob",
        )
        await usfs.delete(
            "/a.md", session=async_session, user_id="alice"
        )
        await usfs.delete(
            "/b.md", session=async_session, user_id="bob"
        )

        alice_trash = await usfs.list_trash(
            session=async_session, user_id="alice"
        )
        assert alice_trash.success
        assert len(alice_trash.entries) == 1
        assert alice_trash.entries[0].name == "a.md"

        bob_trash = await usfs.list_trash(
            session=async_session, user_id="bob"
        )
        assert bob_trash.success
        assert len(bob_trash.entries) == 1
        assert bob_trash.entries[0].name == "b.md"

    async def test_restore_own_file(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/mine.md", "my data",
            session=async_session, user_id="alice",
        )
        await usfs.delete(
            "/mine.md", session=async_session, user_id="alice"
        )

        result = await usfs.restore_from_trash(
            "/mine.md", session=async_session, user_id="alice"
        )
        assert result.success is True

        r = await usfs.read(
            "/mine.md", session=async_session, user_id="alice"
        )
        assert r.success
        assert r.content == "my data"

    async def test_restore_other_user_denied(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/secret.md", "alice's secret",
            session=async_session, user_id="alice",
        )
        await usfs.delete(
            "/secret.md", session=async_session, user_id="alice"
        )

        result = await usfs.restore_from_trash(
            "/secret.md", session=async_session, user_id="bob"
        )
        assert result.success is False

    async def test_empty_trash_scoped(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/a.md", "alice", session=async_session, user_id="alice"
        )
        await usfs.write(
            "/b.md", "bob", session=async_session, user_id="bob"
        )
        await usfs.delete(
            "/a.md", session=async_session, user_id="alice"
        )
        await usfs.delete(
            "/b.md", session=async_session, user_id="bob"
        )

        result = await usfs.empty_trash(
            session=async_session, user_id="alice"
        )
        assert result.success
        assert result.total_deleted == 1

        bob_trash = await usfs.list_trash(
            session=async_session, user_id="bob"
        )
        assert len(bob_trash.entries) == 1


# ---------------------------------------------------------------------------
# Version tests
# ---------------------------------------------------------------------------


class TestVersions:
    async def test_list_versions(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "v1", session=async_session, user_id="alice"
        )
        await usfs.write(
            "/notes.md", "v2", session=async_session, user_id="alice"
        )

        result = await usfs.list_versions(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert result.success is True
        assert len(result.versions) == 2

    async def test_get_version_content(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "v1", session=async_session, user_id="alice"
        )
        await usfs.write(
            "/notes.md", "v2", session=async_session, user_id="alice"
        )

        result = await usfs.get_version_content(
            "/notes.md", 1, session=async_session, user_id="alice"
        )
        assert result.success is True
        assert result.content == "v1"

    async def test_restore_version(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await usfs.write(
            "/notes.md", "v1", session=async_session, user_id="alice"
        )
        await usfs.write(
            "/notes.md", "v2", session=async_session, user_id="alice"
        )

        result = await usfs.restore_version(
            "/notes.md", 1, session=async_session, user_id="alice"
        )
        assert result.success is True

        read_result = await usfs.read(
            "/notes.md", session=async_session, user_id="alice"
        )
        assert read_result.content == "v1"


# ---------------------------------------------------------------------------
# SupportsReBAC share CRUD tests
# ---------------------------------------------------------------------------


class TestShareCRUD:
    async def test_share_creates_share_record(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        share = await shared_usfs.share(
            "/notes.md", "bob", "read",
            user_id="alice", session=async_session,
        )
        assert isinstance(share, ShareInfo)
        assert share.path == "/notes.md"
        assert share.grantee_id == "bob"
        assert share.permission == "read"
        assert share.granted_by == "alice"

    async def test_unshare_removes_record(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await shared_usfs.share(
            "/notes.md", "bob", "read",
            user_id="alice", session=async_session,
        )

        removed = await shared_usfs.unshare(
            "/notes.md", "bob",
            user_id="alice", session=async_session,
        )
        assert removed is True

        # Verify bob can no longer read
        with pytest.raises(PermissionError):
            await shared_usfs.read(
                "/@shared/alice/notes.md",
                session=async_session, user_id="bob",
            )

    async def test_list_shares_on_path(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await shared_usfs.share(
            "/notes.md", "bob", "read",
            user_id="alice", session=async_session,
        )
        await shared_usfs.share(
            "/notes.md", "charlie", "write",
            user_id="alice", session=async_session,
        )

        shares = await shared_usfs.list_shares_on_path(
            "/notes.md", user_id="alice", session=async_session
        )
        assert len(shares) == 2
        grantees = {s.grantee_id for s in shares}
        assert grantees == {"bob", "charlie"}

    async def test_list_shared_with_me(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice notes",
            session=async_session, user_id="alice",
        )
        await shared_usfs.share(
            "/notes.md", "bob", "read",
            user_id="alice", session=async_session,
        )

        shares = await shared_usfs.list_shared_with_me(
            user_id="bob", session=async_session
        )
        assert len(shares) == 1
        assert shares[0].path == "/@shared/alice/notes.md"
        assert shares[0].permission == "read"

    async def test_share_no_sharing_service_raises(
        self, usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        with pytest.raises(ValueError, match="No sharing service"):
            await usfs.share(
                "/notes.md", "bob", "read",
                user_id="alice", session=async_session,
            )

    async def test_share_and_access_roundtrip(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        """Share via ReBAC API, then access via @shared path."""
        await shared_usfs.write(
            "/notes.md", "shared content",
            session=async_session, user_id="alice",
        )
        await shared_usfs.share(
            "/notes.md", "bob", "read",
            user_id="alice", session=async_session,
        )

        result = await shared_usfs.read(
            "/@shared/alice/notes.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert result.content == "shared content"


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_supports_rebac_protocol(self):
        from grover.fs.protocol import SupportsReBAC

        sharing = SharingService(FileShare)
        usfs = UserScopedFileSystem(sharing=sharing)
        assert isinstance(usfs, SupportsReBAC)

    def test_supports_versions_protocol(self):
        from grover.fs.protocol import SupportsVersions

        usfs = UserScopedFileSystem()
        assert isinstance(usfs, SupportsVersions)

    def test_supports_trash_protocol(self):
        from grover.fs.protocol import SupportsTrash

        usfs = UserScopedFileSystem()
        assert isinstance(usfs, SupportsTrash)

    def test_supports_storage_backend_protocol(self):
        from grover.fs.protocol import StorageBackend

        usfs = UserScopedFileSystem()
        assert isinstance(usfs, StorageBackend)


# ---------------------------------------------------------------------------
# Security vulnerability regression tests
# ---------------------------------------------------------------------------


class TestSecurityVulnerabilities:
    """Regression tests for 6 authorization bypass vulnerabilities."""

    async def _create_share(
        self,
        usfs: UserScopedFileSystem,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str = "read",
        granted_by: str = "alice",
    ) -> None:
        assert usfs._sharing is not None
        await usfs._sharing.create_share(
            session, path, grantee_id, permission, granted_by
        )

    # -- Fix 1: glob/grep/tree share permission checks + path restoration --

    async def test_glob_shared_no_permission_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.glob(
                "*.md", "/@shared/alice",
                session=async_session, user_id="bob",
            )

    async def test_glob_shared_with_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.glob(
            "*.md", "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert len(result.entries) >= 1

    async def test_glob_shared_returns_shared_paths(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.glob(
            "*.md", "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        for e in result.entries:
            assert e.path.startswith("/@shared/alice"), f"Expected @shared path, got {e.path}"

    async def test_grep_shared_no_permission_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.grep(
                "hello", "/@shared/alice",
                session=async_session, user_id="bob",
            )

    async def test_grep_shared_with_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.grep(
            "hello", "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert len(result.matches) >= 1

    async def test_grep_shared_returns_shared_paths(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "hello world",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.grep(
            "hello", "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        for m in result.matches:
            assert m.file_path.startswith("/@shared/alice"), (
                f"Expected @shared path, got {m.file_path}"
            )

    async def test_tree_shared_no_permission_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.tree(
                "/@shared/alice",
                session=async_session, user_id="bob",
            )

    async def test_tree_shared_with_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.tree(
            "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        assert len(result.entries) >= 1

    async def test_tree_shared_returns_shared_paths(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        result = await shared_usfs.tree(
            "/@shared/alice",
            session=async_session, user_id="bob",
        )
        assert result.success is True
        for e in result.entries:
            assert e.path.startswith("/@shared/alice"), f"Expected @shared path, got {e.path}"

    # -- Fix 2: copy destination share permission check --

    async def test_copy_to_shared_dest_no_permission_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/src.md", "content",
            session=async_session, user_id="bob",
        )
        await shared_usfs.write(
            "/target.md", "existing",
            session=async_session, user_id="alice",
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.copy(
                "/src.md", "/@shared/alice/target.md",
                session=async_session, user_id="bob",
            )

    async def test_copy_to_shared_dest_with_write_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/src.md", "content",
            session=async_session, user_id="bob",
        )
        await shared_usfs.write(
            "/target.md", "existing",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session,
            "/alice/target.md", "bob", "write",
        )
        result = await shared_usfs.copy(
            "/src.md", "/@shared/alice/target.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True

    # -- Fix 3: share/unshare/list_shares block @shared paths --

    async def test_share_on_shared_path_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/notes.md", "bob", "write"
        )
        with pytest.raises(PermissionError, match="Cannot manage shares"):
            await shared_usfs.share(
                "/@shared/alice/notes.md", "charlie", "read",
                user_id="bob", session=async_session,
            )

    async def test_unshare_on_shared_path_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/notes.md", "bob", "write"
        )
        with pytest.raises(PermissionError, match="Cannot manage shares"):
            await shared_usfs.unshare(
                "/@shared/alice/notes.md", "charlie",
                user_id="bob", session=async_session,
            )

    async def test_list_shares_on_shared_path_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "content",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/notes.md", "bob", "write"
        )
        with pytest.raises(PermissionError, match="Cannot manage shares"):
            await shared_usfs.list_shares_on_path(
                "/@shared/alice/notes.md",
                user_id="bob", session=async_session,
            )

    # -- Fix 4: move cross-namespace block --

    async def test_move_shared_to_own_namespace_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice's file",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/notes.md", "bob", "write"
        )
        with pytest.raises(PermissionError, match="Cannot move shared files"):
            await shared_usfs.move(
                "/@shared/alice/notes.md", "/stolen.md",
                session=async_session, user_id="bob",
            )

    async def test_move_shared_to_own_namespace_denied_follow_true(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice's file",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice/notes.md", "bob", "write"
        )
        with pytest.raises(PermissionError, match="Cannot move shared files"):
            await shared_usfs.move(
                "/@shared/alice/notes.md", "/stolen.md",
                session=async_session, user_id="bob", follow=True,
            )

    async def test_move_within_shared_namespace_allowed(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await shared_usfs.write(
            "/notes.md", "alice's file",
            session=async_session, user_id="alice",
        )
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "write"
        )
        result = await shared_usfs.move(
            "/@shared/alice/notes.md", "/@shared/alice/renamed.md",
            session=async_session, user_id="bob",
        )
        assert result.success is True

    # -- Fix 5: mkdir share permission check --

    async def test_mkdir_shared_no_permission_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.mkdir(
                "/@shared/alice/newdir",
                session=async_session, user_id="bob",
            )

    async def test_mkdir_shared_with_write_permission(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "write"
        )
        result = await shared_usfs.mkdir(
            "/@shared/alice/newdir",
            session=async_session, user_id="bob",
        )
        assert result.success is True

    async def test_mkdir_shared_read_only_denied(
        self, shared_usfs: UserScopedFileSystem, async_session: AsyncSession
    ):
        await self._create_share(
            shared_usfs, async_session, "/alice", "bob", "read"
        )
        with pytest.raises(PermissionError, match="Access denied"):
            await shared_usfs.mkdir(
                "/@shared/alice/newdir",
                session=async_session, user_id="bob",
            )

    # -- Fix 6: _require_user_id format validation --

    def test_user_id_with_slash_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("user/name")

    def test_user_id_with_backslash_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("user\\name")

    def test_user_id_with_null_byte_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("user\0name")

    def test_user_id_with_at_sign_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("user@name")

    def test_user_id_with_dotdot_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("user..name")

    def test_user_id_traversal_attack_rejected(self):
        with pytest.raises(AuthenticationRequiredError, match="invalid characters"):
            UserScopedFileSystem._require_user_id("../alice")

    def test_valid_user_ids_still_work(self):
        assert UserScopedFileSystem._require_user_id("alice") == "alice"
        assert UserScopedFileSystem._require_user_id("bob-smith") == "bob-smith"
        assert UserScopedFileSystem._require_user_id("user_name") == "user_name"
        assert UserScopedFileSystem._require_user_id("user.name") == "user.name"
        assert UserScopedFileSystem._require_user_id("user123") == "user123"

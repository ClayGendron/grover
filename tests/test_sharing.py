"""Tests for SharingService — share CRUD and permission resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from grover.fs.sharing import SharingService
from grover.models.shares import FileShare

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def sharing() -> SharingService:
    return SharingService(FileShare)


# ---------------------------------------------------------------------------
# create_share
# ---------------------------------------------------------------------------


class TestCreateShare:
    async def test_create_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        share = await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        assert share.path == "/alice/notes.md"
        assert share.grantee_id == "bob"
        assert share.permission == "read"
        assert share.granted_by == "alice"
        assert share.id  # UUID set

    async def test_create_share_write(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        share = await sharing.create_share(
            async_session,
            "/alice/project/",
            grantee_id="bob",
            permission="write",
            granted_by="alice",
        )
        assert share.permission == "write"

    async def test_create_share_invalid_permission(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        with pytest.raises(ValueError, match="Invalid permission"):
            await sharing.create_share(
                async_session,
                "/alice/notes.md",
                grantee_id="bob",
                permission="admin",
                granted_by="alice",
            )

    async def test_create_share_with_expiry(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        expires = datetime.now(UTC) + timedelta(hours=1)
        share = await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
            expires_at=expires,
        )
        assert share.expires_at is not None


# ---------------------------------------------------------------------------
# remove_share
# ---------------------------------------------------------------------------


class TestRemoveShare:
    async def test_remove_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        removed = await sharing.remove_share(
            async_session, "/alice/notes.md", "bob"
        )
        assert removed is True

    async def test_remove_share_nonexistent(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        removed = await sharing.remove_share(
            async_session, "/nonexistent.md", "bob"
        )
        assert removed is False


# ---------------------------------------------------------------------------
# list_shares_on_path
# ---------------------------------------------------------------------------


class TestListSharesOnPath:
    async def test_list_shares_on_path(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="charlie",
            permission="write",
            granted_by="alice",
        )
        shares = await sharing.list_shares_on_path(
            async_session, "/alice/notes.md"
        )
        assert len(shares) == 2
        grantees = {s.grantee_id for s in shares}
        assert grantees == {"bob", "charlie"}

    async def test_list_shares_on_path_empty(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        shares = await sharing.list_shares_on_path(
            async_session, "/nobody/file.md"
        )
        assert shares == []


# ---------------------------------------------------------------------------
# list_shared_with
# ---------------------------------------------------------------------------


class TestListSharedWith:
    async def test_list_shared_with(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/a.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        await sharing.create_share(
            async_session,
            "/charlie/b.md",
            grantee_id="bob",
            permission="write",
            granted_by="charlie",
        )
        shares = await sharing.list_shared_with(async_session, "bob")
        assert len(shares) == 2
        paths = {s.path for s in shares}
        assert paths == {"/alice/a.md", "/charlie/b.md"}


# ---------------------------------------------------------------------------
# check_permission
# ---------------------------------------------------------------------------


class TestCheckPermission:
    async def test_exact_match(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob"
        )
        assert result is True

    async def test_directory_inherit(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Share on /alice/projects/ grants /alice/projects/docs/file.md."""
        await sharing.create_share(
            async_session,
            "/alice/projects",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/projects/docs/file.md", "bob"
        )
        assert result is True

    async def test_no_match(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        result = await sharing.check_permission(
            async_session, "/alice/secret.md", "bob"
        )
        assert result is False

    async def test_write_required_read_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Write required but only read share exists → False."""
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob", required="write"
        )
        assert result is False

    async def test_write_required_write_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="write",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob", required="write"
        )
        assert result is True

    async def test_read_required_write_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Write share implies read access."""
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="write",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob", required="read"
        )
        assert result is True

    async def test_expired_share(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Expired shares are ignored."""
        expired = datetime.now(UTC) - timedelta(hours=1)
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
            expires_at=expired,
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob"
        )
        assert result is False

    async def test_not_yet_expired(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Non-expired shares are valid."""
        future = datetime.now(UTC) + timedelta(hours=1)
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
            expires_at=future,
        )
        result = await sharing.check_permission(
            async_session, "/alice/notes.md", "bob"
        )
        assert result is True

    async def test_root_share_grants_all(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Share on / grants access to everything under it."""
        await sharing.create_share(
            async_session,
            "/",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        result = await sharing.check_permission(
            async_session, "/alice/deep/nested/file.md", "bob"
        )
        assert result is True


# ---------------------------------------------------------------------------
# update_share_paths
# ---------------------------------------------------------------------------


class TestUpdateSharePaths:
    async def test_update_share_paths(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        await sharing.create_share(
            async_session,
            "/alice/old/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        count = await sharing.update_share_paths(
            async_session, "/alice/old", "/alice/new"
        )
        assert count == 1
        shares = await sharing.list_shared_with(async_session, "bob")
        assert shares[0].path == "/alice/new/notes.md"

    async def test_update_share_paths_no_matches(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        count = await sharing.update_share_paths(
            async_session, "/nonexistent", "/other"
        )
        assert count == 0

    async def test_update_share_paths_exact_match(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Exact path match also gets updated."""
        await sharing.create_share(
            async_session,
            "/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        count = await sharing.update_share_paths(
            async_session, "/alice/notes.md", "/alice/final.md"
        )
        assert count == 1
        shares = await sharing.list_shared_with(async_session, "bob")
        assert shares[0].path == "/alice/final.md"

    async def test_update_share_paths_directory(
        self, sharing: SharingService, async_session: AsyncSession
    ):
        """Directory share path and children get updated."""
        await sharing.create_share(
            async_session,
            "/alice/project",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        await sharing.create_share(
            async_session,
            "/alice/project/src/main.py",
            grantee_id="charlie",
            permission="write",
            granted_by="alice",
        )
        count = await sharing.update_share_paths(
            async_session, "/alice/project", "/alice/renamed"
        )
        assert count == 2
        bob_shares = await sharing.list_shared_with(async_session, "bob")
        assert bob_shares[0].path == "/alice/renamed"
        charlie_shares = await sharing.list_shared_with(
            async_session, "charlie"
        )
        assert charlie_shares[0].path == "/alice/renamed/src/main.py"

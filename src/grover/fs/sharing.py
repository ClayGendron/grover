"""SharingService — share CRUD and permission resolution.

Stateless service that receives the share model at construction
and a session at call time, following the MetadataService pattern.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from .utils import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.shares import FileShareBase


class SharingService:
    """Manages file shares between users.

    Constructor receives the concrete share model so callers can use
    custom SQLModel subclasses with different table names.
    """

    def __init__(self, share_model: type[FileShareBase]) -> None:
        self._share_model = share_model

    async def create_share(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        permission: str,
        granted_by: str,
        *,
        expires_at: datetime | None = None,
    ) -> FileShareBase:
        """Create a share record. Flushes but does not commit."""
        if permission not in ("read", "write"):
            raise ValueError(
                f"Invalid permission: {permission!r}. Must be 'read' or 'write'."
            )

        path = normalize_path(path)
        share = self._share_model(
            id=str(uuid.uuid4()),
            path=path,
            grantee_id=grantee_id,
            permission=permission,
            granted_by=granted_by,
            expires_at=expires_at,
        )
        session.add(share)
        await session.flush()
        return share

    async def remove_share(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
    ) -> bool:
        """Remove an exact share match. Returns True if found."""
        path = normalize_path(path)
        model = self._share_model
        result = await session.execute(
            select(model).where(
                model.path == path,
                model.grantee_id == grantee_id,
            )
        )
        share = result.scalar_one_or_none()
        if share is None:
            return False
        await session.delete(share)
        await session.flush()
        return True

    async def list_shares_on_path(
        self,
        session: AsyncSession,
        path: str,
    ) -> list[FileShareBase]:
        """List all shares for a given path."""
        path = normalize_path(path)
        model = self._share_model
        result = await session.execute(
            select(model).where(model.path == path)
        )
        return list(result.scalars().all())

    async def list_shared_with(
        self,
        session: AsyncSession,
        grantee_id: str,
    ) -> list[FileShareBase]:
        """List all shares granted to a grantee."""
        model = self._share_model
        result = await session.execute(
            select(model).where(model.grantee_id == grantee_id)
        )
        return list(result.scalars().all())

    async def list_shares_under_prefix(
        self,
        session: AsyncSession,
        grantee_id: str,
        prefix: str,
    ) -> list[FileShareBase]:
        """List non-expired shares for *grantee_id* strictly under *prefix*.

        Returns shares whose path starts with ``prefix + "/"``.  Shares
        exactly at *prefix* are excluded (those are handled by the fast-path
        ``check_permission`` call).
        """
        prefix = normalize_path(prefix)
        model = self._share_model
        now = datetime.now(UTC)

        # Escape SQL LIKE wildcards in the prefix itself
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = escaped + "/%"
        prefix_slash = prefix + "/"

        result = await session.execute(
            select(model).where(
                model.grantee_id == grantee_id,
                model.path.like(like_pattern, escape="\\"),  # type: ignore[union-attr]
            )
        )
        shares = result.scalars().all()

        # Filter out expired shares and verify prefix in Python for defense-in-depth
        active: list[FileShareBase] = []
        for share in shares:
            if not share.path.startswith(prefix_slash):
                continue
            if share.expires_at is not None:
                exp = share.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                if exp <= now:
                    continue
            active.append(share)

        return active

    async def check_permission(
        self,
        session: AsyncSession,
        path: str,
        grantee_id: str,
        required: str = "read",
    ) -> bool:
        """Check if *grantee_id* has *required* permission on *path*.

        Permission resolution walks ancestor paths: a share on
        ``/alice/projects/`` grants access to ``/alice/projects/docs/file.md``.
        Write shares imply read access.
        Expired shares are ignored.
        """
        path = normalize_path(path)

        # Build ancestor path list: path, parent, grandparent, ..., root
        ancestors = []
        current = path
        while True:
            ancestors.append(current)
            if current == "/":
                break
            parent = current.rsplit("/", 1)[0] or "/"
            current = parent

        model = self._share_model
        now = datetime.now(UTC)

        result = await session.execute(
            select(model).where(
                model.grantee_id == grantee_id,
                model.path.in_(ancestors),  # type: ignore[union-attr]
            )
        )
        shares = result.scalars().all()

        for share in shares:
            # Skip expired shares (handle naive datetimes from SQLite)
            if share.expires_at is not None:
                exp = share.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                if exp <= now:
                    continue

            # Write share implies read
            if required == "read":
                return True
            if required == "write" and share.permission == "write":
                return True

        return False

    async def update_share_paths(
        self,
        session: AsyncSession,
        old_prefix: str,
        new_prefix: str,
    ) -> int:
        """Bulk-update share paths when a file/directory is renamed.

        Uses prefix matching: shares on ``/alice/projects/`` become
        ``/bob/projects/`` if old_prefix=/alice, new_prefix=/bob.

        Returns the number of shares updated.
        """
        old_prefix = normalize_path(old_prefix)
        new_prefix = normalize_path(new_prefix)
        model = self._share_model

        # Match exact path or path + "/" prefix
        result = await session.execute(
            select(model).where(
                model.path.like(old_prefix + "%", escape="\\"),  # type: ignore[union-attr]
            )
        )
        shares = list(result.scalars().all())

        count = 0
        for share in shares:
            # Exact match or prefix match (must be at directory boundary)
            if share.path == old_prefix or share.path.startswith(old_prefix + "/"):
                share.path = new_prefix + share.path[len(old_prefix):]
                count += 1

        if count > 0:
            await session.flush()
        return count

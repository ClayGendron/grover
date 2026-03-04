"""ShareMixin — sharing operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.backends.protocol import SupportsReBAC
from grover.exceptions import MountNotFoundError
from grover.results import (
    FileSearchCandidate,
    ShareResult,
    ShareSearchResult,
)
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from datetime import datetime

    from grover.api.context import GroverContext


class ShareMixin:
    """Sharing operations extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Share operations
    # ------------------------------------------------------------------

    async def share(
        self,
        path: str,
        grantee_id: str,
        permission: str = "read",
        *,
        user_id: str,
        expires_at: datetime | None = None,
    ) -> ShareResult:
        """Share a file or directory with another user.

        Requires a backend that supports sharing (e.g. ``UserScopedFileSystem``).
        """

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            try:
                share_info = await cap.share(
                    rel_path,
                    grantee_id,
                    permission,
                    user_id=user_id,
                    session=sess,
                    expires_at=expires_at,
                )
            except ValueError as e:
                return ShareResult(success=False, message=str(e))

        return ShareResult(
            success=True,
            message=f"Shared {path} with {grantee_id} ({permission})",
            path=path,
            grantee_id=share_info.grantee_id,
            permission=share_info.permission,
            granted_by=share_info.granted_by,
        )

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
    ) -> ShareResult:
        """Remove a share for a file or directory."""

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return ShareResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            result = await cap.unshare(rel_path, grantee_id, user_id=user_id, session=sess)

        result.path = path
        return result

    async def list_shares(
        self,
        path: str,
        *,
        user_id: str,
    ) -> ShareSearchResult:
        """List all shares on a given path."""

        path = normalize_path(path)
        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return ShareSearchResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return ShareSearchResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            result = await cap.list_shares_on_path(rel_path, user_id=user_id, session=sess)

        # Rebase paths from backend-relative to absolute mount paths
        return result.rebase(mount.path)

    async def list_shared_with_me(
        self,
        *,
        user_id: str,
    ) -> ShareSearchResult:
        """List all files shared with the current user across all mounts."""
        all_candidates: list[FileSearchCandidate] = []
        for mount in self._ctx.registry.list_mounts():
            cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                assert sess is not None
                result = await cap.list_shared_with_me(user_id=user_id, session=sess)
            # Backend returns paths like /@shared/alice/a.md — rebase to mount
            rebased = result.rebase(mount.path)
            all_candidates.extend(rebased.candidates)

        return ShareSearchResult(
            success=True,
            message=f"Found {len(all_candidates)} share(s)",
            candidates=all_candidates,
        )

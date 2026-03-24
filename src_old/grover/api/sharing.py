"""ShareMixin — sharing operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.backends.protocol import SupportsReBAC
from grover.exceptions import MountNotFoundError
from grover.models.internal.ref import File
from grover.models.internal.results import FileOperationResult, FileSearchResult
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
    ) -> FileOperationResult:
        """Share a file or directory with another user.

        Requires a backend that supports sharing (e.g. ``UserScopedFileSystem``).
        """

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err.message)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return FileOperationResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return FileOperationResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            try:
                await cap.share(
                    rel_path,
                    grantee_id,
                    permission,
                    user_id=user_id,
                    session=sess,
                    expires_at=expires_at,
                )
            except ValueError as e:
                return FileOperationResult(success=False, message=str(e))

        return FileOperationResult(
            success=True,
            message=f"Shared {path} with {grantee_id} ({permission})",
            file=File(path=path),
        )

    async def unshare(
        self,
        path: str,
        grantee_id: str,
        *,
        user_id: str,
    ) -> FileOperationResult:
        """Remove a share for a file or directory."""

        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err.message)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return FileOperationResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return FileOperationResult(
                success=False,
                message="Backend does not support sharing",
            )

        async with self._ctx.session_for(mount) as sess:
            assert sess is not None
            result = await cap.unshare(rel_path, grantee_id, user_id=user_id, session=sess)

        result.file.path = path
        return result

    async def list_shares(
        self,
        path: str,
        *,
        user_id: str,
    ) -> FileSearchResult:
        """List all shares on a given path."""

        path = normalize_path(path)
        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError as e:
            return FileSearchResult(success=False, message=str(e))

        cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
        if cap is None:
            return FileSearchResult(
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
    ) -> FileSearchResult:
        """List all files shared with the current user across all mounts."""
        all_files: list[File] = []
        for mount in self._ctx.registry.list_mounts():
            cap = self._ctx.get_capability(mount.filesystem, SupportsReBAC)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                assert sess is not None
                result = await cap.list_shared_with_me(user_id=user_id, session=sess)
            # Backend returns paths like /@shared/alice/a.md — rebase to mount
            rebased = result.rebase(mount.path)
            all_files.extend(rebased.files)

        return FileSearchResult(
            success=True,
            message=f"Found {len(all_files)} share(s)",
            files=all_files,
        )

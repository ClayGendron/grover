"""VFS — mount router with routing, permissions, events, capabilities."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any, TypeVar

from grover.events import EventBus, EventType, FileEvent

from .exceptions import CapabilityNotSupportedError, MountNotFoundError
from .permissions import Permission
from .protocol import SupportsReconcile, SupportsTrash, SupportsVersions
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    ListResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    WriteResult,
)
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from .mounts import MountConfig, MountRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VFS:
    """Routes operations to backends via mount registry.

    Presents a single namespace to callers while delegating to the
    appropriate backend based on the path prefix. Enforces permissions,
    handles cross-mount copy/move, and provides capability gating.

    Session lifecycle is per-operation only.  No transaction mode.
    """

    def __init__(
        self, registry: MountRegistry, event_bus: EventBus | None = None
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    def _get_capability(self, backend: Any, protocol: type[T]) -> T | None:
        if isinstance(backend, protocol):
            return backend
        return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def _emit(self, event: FileEvent) -> None:
        if self._event_bus is not None:
            await self._event_bus.emit(event)

    # ------------------------------------------------------------------
    # Session Management (per-operation only)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _session_for(self, mount: MountConfig) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or None for non-SQL."""
        if not mount.has_session_factory:
            yield None
            return

        assert mount.session_factory is not None
        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all backends."""
        for mount in self._registry.list_mounts():
            if hasattr(mount.backend, "close"):
                try:
                    await mount.backend.close()
                except Exception:
                    logger.warning("Backend close failed for %s", mount.mount_path, exc_info=True)

    # ------------------------------------------------------------------
    # Path Helpers
    # ------------------------------------------------------------------

    def _prefix_path(self, path: str | None, mount_path: str) -> str | None:
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def _prefix_file_info(self, info: FileInfo, mount: MountConfig) -> FileInfo:
        prefixed_path = self._prefix_path(info.path, mount.mount_path) or info.path
        return dc_replace(
            info,
            path=prefixed_path,
            mount_type=mount.mount_type,
            permission=self._registry.get_permission(prefixed_path).value,
        )

    def _check_writable(self, virtual_path: str) -> None:
        perm = self._registry.get_permission(virtual_path)
        if perm == Permission.READ_ONLY:
            raise PermissionError(f"Cannot write to read-only path: {virtual_path}")

    # ------------------------------------------------------------------
    # Read Operations
    # ------------------------------------------------------------------

    async def read(
        self, path: str, offset: int = 0, limit: int = 2000,
    ) -> ReadResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.read(rel_path, offset, limit, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def list_dir(self, path: str = "/") -> ListResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.list_dir(rel_path, session=sess)

        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [
            self._prefix_file_info(entry, mount) for entry in result.entries
        ]

        return result

    def _list_root(self) -> ListResult:
        entries: list[FileInfo] = []
        for mount in self._registry.list_visible_mounts():
            name = mount.mount_path.lstrip("/")
            entries.append(
                FileInfo(
                    path=mount.mount_path,
                    name=name,
                    is_directory=True,
                    permission=mount.permission.value,
                    mount_type=mount.mount_type,
                )
            )
        return ListResult(
            success=True,
            message=f"Found {len(entries)} mount(s)",
            entries=entries,
            path="/",
        )

    async def exists(self, path: str) -> bool:
        path = normalize_path(path)

        if path == "/":
            return True

        if self._registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return False

        async with self._session_for(mount) as sess:
            return await mount.backend.exists(rel_path, session=sess)

    async def get_info(self, path: str) -> FileInfo | None:
        path = normalize_path(path)

        if self._registry.has_mount(path):
            for mount in self._registry.list_mounts():
                if mount.mount_path == path:
                    name = mount.mount_path.lstrip("/")
                    return FileInfo(
                        path=mount.mount_path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._registry.resolve(path)
        except MountNotFoundError:
            return None

        async with self._session_for(mount) as sess:
            info = await mount.backend.get_info(rel_path, session=sess)
        if info is not None:
            info = self._prefix_file_info(info, mount)
        return info

    def get_permission_info(self, path: str) -> tuple[str, bool]:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)

        permission = self._registry.get_permission(path)

        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths

        return permission.value, is_override

    # ------------------------------------------------------------------
    # Write Operations (permission-checked)
    # ------------------------------------------------------------------

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
    ) -> WriteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.write(
                rel_path, content, created_by, overwrite=overwrite, session=sess
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=path, content=content)
            )
        return result

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
    ) -> EditResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return EditResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.edit(
                rel_path, old_string, new_string, replace_all, created_by, session=sess
            )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=path)
            )
        return result

    async def delete(
        self, path: str, permanent: bool = False,
    ) -> DeleteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)

        # If backend doesn't support trash and permanent=False, explicit failure
        if not permanent and not self._get_capability(mount.backend, SupportsTrash):
            return DeleteResult(
                success=False,
                message="Trash not supported on this mount. "
                "Use permanent=True to delete permanently.",
            )

        async with self._session_for(mount) as sess:
            result = await mount.backend.delete(rel_path, permanent, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_DELETED, path=path)
            )
        return result

    async def mkdir(
        self, path: str, parents: bool = True,
    ) -> MkdirResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.mkdir(rel_path, parents, session=sess)
        result.path = self._prefix_path(result.path, mount.mount_path)
        result.created_dirs = [
            self._prefix_path(d, mount.mount_path) or d for d in result.created_dirs
        ]
        return result

    async def move(
        self, src: str, dest: str,
    ) -> MoveResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(src)
            self._check_writable(dest)
        except PermissionError as e:
            return MoveResult(success=False, message=str(e))

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            async with self._session_for(src_mount) as sess:
                result = await src_mount.backend.move(src_rel, dest_rel, session=sess)
            result.old_path = self._prefix_path(result.old_path, src_mount.mount_path)
            result.new_path = self._prefix_path(result.new_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src)
                )
            return result

        # Cross-mount move: read → write → delete (non-atomic).
        # If write commits but delete fails, data exists in both mounts.
        # The error message is returned to the caller; no automatic compensation.
        async with self._session_for(src_mount) as src_sess:
            read_result = await src_mount.backend.read(src_rel, session=src_sess)
        if not read_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot read source for cross-mount move: {read_result.message}",
            )

        if read_result.content is None:
            return MoveResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        async with self._session_for(dest_mount) as dest_sess:
            write_result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess
            )
        if not write_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot write to destination for cross-mount move: {write_result.message}",
            )

        async with self._session_for(src_mount) as src_sess:
            delete_result = await src_mount.backend.delete(
                src_rel, permanent=False, session=src_sess
            )
        if not delete_result.success:
            return MoveResult(
                success=False,
                message=f"Copied but failed to delete source: {delete_result.message}",
            )

        await self._emit(
            FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src)
        )
        return MoveResult(
            success=True,
            message=f"Moved {src} -> {dest} (cross-mount)",
            old_path=src,
            new_path=dest,
        )

    async def copy(
        self, src: str, dest: str,
    ) -> WriteResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            async with self._session_for(src_mount) as sess:
                result = await src_mount.backend.copy(src_rel, dest_rel, session=sess)
            result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=dest)
                )
            return result

        # Cross-mount copy: read → write
        async with self._session_for(src_mount) as src_sess:
            read_result = await src_mount.backend.read(src_rel, session=src_sess)
        if not read_result.success:
            return WriteResult(
                success=False,
                message=f"Cannot read source for cross-mount copy: {read_result.message}",
            )

        if read_result.content is None:
            return WriteResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        async with self._session_for(dest_mount) as dest_sess:
            result = await dest_mount.backend.write(
                dest_rel, read_result.content, session=dest_sess
            )
        result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=dest)
            )
        return result

    # ------------------------------------------------------------------
    # Version Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str) -> ListVersionsResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            return await cap.list_versions(rel_path, session=sess)

    async def restore_version(self, path: str, version: int) -> RestoreResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            result = await cap.restore_version(rel_path, version, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def get_version_content(self, path: str, version: int) -> GetVersionContentResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        cap = self._get_capability(mount.backend, SupportsVersions)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support versioning"
            )
        async with self._session_for(mount) as sess:
            return await cap.get_version_content(rel_path, version, session=sess)

    # ------------------------------------------------------------------
    # Trash Operations (capability-gated)
    # ------------------------------------------------------------------

    async def list_trash(self) -> ListResult:
        """List all items in trash across all mounts (skips unsupported)."""
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self._session_for(mount) as sess:
                result = await cap.list_trash(session=sess)
            if result.success:
                prefixed_entries = [
                    self._prefix_file_info(entry, mount) for entry in result.entries
                ]
                all_entries.extend(prefixed_entries)

        return ListResult(
            success=True,
            message=f"Found {len(all_entries)} item(s) in trash",
            entries=all_entries,
            path="/__trash__",
        )

    async def restore_from_trash(self, path: str) -> RestoreResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        cap = self._get_capability(mount.backend, SupportsTrash)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Mount at {mount.mount_path} does not support trash"
            )
        async with self._session_for(mount) as sess:
            result = await cap.restore_from_trash(rel_path, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def empty_trash(self) -> DeleteResult:
        """Empty trash across all mounts (skips unsupported)."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            cap = self._get_capability(mount.backend, SupportsTrash)
            if cap is None:
                continue  # Skip unsupported mounts silently
            async with self._session_for(mount) as sess:
                result = await cap.empty_trash(session=sess)
            if not result.success:
                return result
            total_deleted += result.total_deleted or 0
            mounts_processed += 1

        return DeleteResult(
            success=True,
            message=f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)",
            total_deleted=total_deleted,
            permanent=True,
        )

    # ------------------------------------------------------------------
    # Reconciliation (capability-gated)
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> dict[str, int]:
        """Reconcile disk ↔ DB for capable mounts."""
        total = {"created": 0, "updated": 0, "deleted": 0}
        mounts = self._registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.mount_path == mount_path]

        for mount in mounts:
            cap = self._get_capability(mount.backend, SupportsReconcile)
            if cap is None:
                continue
            async with self._session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            for k in total:
                total[k] += stats.get(k, 0)

        return total

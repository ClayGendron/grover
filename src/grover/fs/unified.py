"""UnifiedFileSystem — routing, permissions, events."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any

from grover.events import EventBus, EventType, FileEvent

from .exceptions import MountNotFoundError
from .permissions import Permission
from .protocol import StorageBackend
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    ListResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    VersionInfo,
    WriteResult,
)
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from .mounts import MountConfig, MountRegistry

logger = logging.getLogger(__name__)


class UnifiedFileSystem(StorageBackend):
    """Routes operations to backends via mount registry.

    Presents a single namespace to callers while delegating to the
    appropriate backend based on the path prefix. Enforces permissions
    and handles cross-mount copy/move.

    For database mounts the UFS manages session lifecycle via
    ``_session_for()``: per-operation sessions in normal mode,
    reused sessions in transaction mode.
    """

    def __init__(
        self, registry: MountRegistry, event_bus: EventBus | None = None
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._entered_backends: list[Any] = []

        # Transaction state for DB mounts
        self._in_transaction: bool = False
        self._txn_sessions: dict[str, AsyncSession] = {}  # mount_path → session

    async def enter_backend(self, backend: Any) -> None:
        """Enter a backend's context manager and track it."""
        if hasattr(backend, "__aenter__"):
            await backend.__aenter__()
            self._entered_backends.append(backend)

    async def exit_backend(self, backend: Any) -> None:
        """Exit a backend's context manager and stop tracking it."""
        if backend in self._entered_backends:
            if hasattr(backend, "__aexit__"):
                await backend.__aexit__(None, None, None)
            self._entered_backends.remove(backend)

    async def _emit(self, event: FileEvent) -> None:
        if self._event_bus is not None:
            await self._event_bus.emit(event)

    # =========================================================================
    # Session Management
    # =========================================================================

    @asynccontextmanager
    async def _session_for(self, mount: MountConfig) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for DB mounts, or ``None`` for local mounts."""
        if not mount.has_session_factory:
            yield None
            return

        assert mount.session_factory is not None  # for type narrowing

        # Transaction mode: reuse session per mount
        if self._in_transaction:
            if mount.mount_path not in self._txn_sessions:
                self._txn_sessions[mount.mount_path] = mount.session_factory()
            yield self._txn_sessions[mount.mount_path]
            return

        # Per-operation: create, commit, close
        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # =========================================================================
    # Transaction Lifecycle
    # =========================================================================

    async def begin_transaction(self) -> None:
        """Enter transaction mode — DB sessions are reused per mount."""
        self._in_transaction = True

    async def commit_transaction(self) -> None:
        """Commit all open transaction sessions and exit transaction mode."""
        for session in self._txn_sessions.values():
            await session.commit()
        await self._close_txn_sessions()
        self._in_transaction = False

    async def rollback_transaction(self) -> None:
        """Roll back all open transaction sessions and exit transaction mode."""
        for session in self._txn_sessions.values():
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback failed for mount session", exc_info=True)
        await self._close_txn_sessions()
        self._in_transaction = False

    async def _close_txn_sessions(self) -> None:
        """Close and discard all transaction sessions."""
        for session in self._txn_sessions.values():
            try:
                await session.close()
            except Exception:
                logger.warning("Session close failed", exc_info=True)
        self._txn_sessions.clear()

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> UnifiedFileSystem:
        for mount in self._registry.list_mounts():
            backend = mount.backend
            if hasattr(backend, "__aenter__"):
                await backend.__aenter__()
                self._entered_backends.append(backend)
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        errors: list[Exception] = []
        for backend in reversed(self._entered_backends):
            if hasattr(backend, "__aexit__"):
                try:
                    await backend.__aexit__(exc_type, exc_val, exc_tb)
                except Exception as e:
                    errors.append(e)
        self._entered_backends.clear()
        if errors and exc_type is None:
            raise errors[0]

    async def close(self) -> None:
        """Close all entered backends."""
        for backend in reversed(self._entered_backends):
            if hasattr(backend, "close"):
                await backend.close()
        self._entered_backends.clear()

    # =========================================================================
    # Path Helpers
    # =========================================================================

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

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read(
        self, path: str, offset: int = 0, limit: int = 2000, *, session: AsyncSession | None = None
    ) -> ReadResult:
        """Read file content with pagination."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.read(rel_path, offset, limit, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def list_dir(self, path: str = "/", *, session: AsyncSession | None = None) -> ListResult:
        """List directory entries."""
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

    async def exists(self, path: str, *, session: AsyncSession | None = None) -> bool:
        """Check whether a path exists in any mount."""
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

    async def get_info(self, path: str, *, session: AsyncSession | None = None) -> FileInfo | None:
        """Get file metadata, or ``None`` if not found."""
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
        """Get permission and whether it's an explicit override."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)

        permission = self._registry.get_permission(path)

        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths

        return permission.value, is_override

    # =========================================================================
    # Write Operations (permission-checked)
    # =========================================================================

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
    ) -> WriteResult:
        """Write content to a file."""
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
        *,
        session: AsyncSession | None = None,
    ) -> EditResult:
        """Apply a string replacement to a file."""
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
        self, path: str, permanent: bool = False, *, session: AsyncSession | None = None
    ) -> DeleteResult:
        """Delete a file or directory."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.delete(rel_path, permanent, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_DELETED, path=path)
            )
        return result

    async def mkdir(
        self, path: str, parents: bool = True, *, session: AsyncSession | None = None
    ) -> MkdirResult:
        """Create a directory."""
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
        self, src: str, dest: str, *, session: AsyncSession | None = None
    ) -> MoveResult:
        """Move a file within or across mounts."""
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

        # Cross-mount move: read → write → delete
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
        self, src: str, dest: str, *, session: AsyncSession | None = None
    ) -> WriteResult:
        """Copy a file within or across mounts."""
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

    # =========================================================================
    # Version & Trash Operations
    # =========================================================================

    async def list_versions(
        self, path: str, *, session: AsyncSession | None = None
    ) -> list[VersionInfo]:
        """List version history for a file."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            return await mount.backend.list_versions(rel_path, session=sess)

    async def restore_version(
        self, path: str, version: int, *, session: AsyncSession | None = None
    ) -> RestoreResult:
        """Restore a file to a specific version."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        async with self._session_for(mount) as sess:
            result = await mount.backend.restore_version(rel_path, version, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def get_version_content(
        self, path: str, version: int, *, session: AsyncSession | None = None
    ) -> str | None:
        """Get content of a specific file version."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            return await mount.backend.get_version_content(rel_path, version, session=sess)

    async def list_trash(self, *, session: AsyncSession | None = None) -> ListResult:
        """List all items in trash across all mounts."""
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            async with self._session_for(mount) as sess:
                result = await mount.backend.list_trash(session=sess)
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

    async def restore_from_trash(
        self, path: str, *, session: AsyncSession | None = None
    ) -> RestoreResult:
        """Restore a file from trash."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        async with self._session_for(mount) as sess:
            result = await mount.backend.restore_from_trash(rel_path, session=sess)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def empty_trash(self, *, session: AsyncSession | None = None) -> DeleteResult:
        """Permanently delete all trashed files."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            async with self._session_for(mount) as sess:
                result = await mount.backend.empty_trash(session=sess)
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

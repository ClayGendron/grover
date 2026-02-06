"""UnifiedFileSystem — routing, permissions, events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grover.events import EventBus, EventType, FileEvent

from .permissions import Permission
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
    from .mounts import MountConfig, MountRegistry


class UnifiedFileSystem:
    """Routes operations to backends via mount registry.

    Presents a single namespace to callers while delegating to the
    appropriate backend based on the path prefix. Enforces permissions
    and handles cross-mount copy/move.
    """

    def __init__(
        self, registry: MountRegistry, event_bus: EventBus | None = None
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._entered_backends: list[Any] = []

    async def _emit(self, event: FileEvent) -> None:
        if self._event_bus is not None:
            await self._event_bus.emit(event)

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
        from dataclasses import replace as dc_replace

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

    def _extract_raw_content(self, formatted_content: str | None) -> str | None:
        """Extract raw file content from formatted ReadResult output."""
        if formatted_content is None:
            return None

        if not formatted_content.startswith("<file>"):
            return formatted_content

        lines = formatted_content.split("\n")
        raw_lines: list[str] = []
        for line in lines[1:]:
            if line.startswith("(") or line == "</file>" or line == "":
                continue
            if "| " in line:
                raw_lines.append(line.split("| ", 1)[1])
            elif "|" in line:
                raw_lines.append(line.split("|", 1)[1])
            else:
                raw_lines.append(line)
        return "\n".join(raw_lines)

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read(
        self, path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.read(rel_path, offset, limit)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def list_dir(self, path: str = "/") -> ListResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.list_dir(rel_path)

        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [
            self._prefix_file_info(entry, mount) for entry in result.entries
        ]

        return result

    def _list_root(self) -> ListResult:
        entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
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
        except FileNotFoundError:
            return False

        return await mount.backend.exists(rel_path)

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
        except FileNotFoundError:
            return None

        info = await mount.backend.get_info(rel_path)
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
        self, path: str, content: str, created_by: str = "agent"
    ) -> WriteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.write(rel_path, content, created_by)
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
        result = await mount.backend.edit(
            rel_path, old_string, new_string, replace_all, created_by
        )
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=path)
            )
        return result

    async def delete(self, path: str, permanent: bool = False) -> DeleteResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.delete(rel_path, permanent)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_DELETED, path=path)
            )
        return result

    async def mkdir(self, path: str, parents: bool = True) -> MkdirResult:
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.mkdir(rel_path, parents)
        result.path = self._prefix_path(result.path, mount.mount_path)
        result.created_dirs = [
            self._prefix_path(d, mount.mount_path) or d for d in result.created_dirs
        ]
        return result

    async def move(self, src: str, dest: str) -> MoveResult:
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
            result = await src_mount.backend.move(src_rel, dest_rel)
            result.old_path = self._prefix_path(result.old_path, src_mount.mount_path)
            result.new_path = self._prefix_path(result.new_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src)
                )
            return result

        read_result = await src_mount.backend.read(src_rel)
        if not read_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot read source for cross-mount move: {read_result.message}",
            )

        raw_content = self._extract_raw_content(read_result.content)
        if raw_content is None:
            return MoveResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        write_result = await dest_mount.backend.write(dest_rel, raw_content)
        if not write_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot write to destination for cross-mount move: {write_result.message}",
            )

        delete_result = await src_mount.backend.delete(src_rel, permanent=False)
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

    async def copy(self, src: str, dest: str) -> WriteResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            result = await src_mount.backend.copy(src_rel, dest_rel)
            result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
            if result.success:
                await self._emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=dest)
                )
            return result

        read_result = await src_mount.backend.read(src_rel)
        if not read_result.success:
            return WriteResult(
                success=False,
                message=f"Cannot read source for cross-mount copy: {read_result.message}",
            )

        raw_content = self._extract_raw_content(read_result.content)
        if raw_content is None:
            return WriteResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        result = await dest_mount.backend.write(dest_rel, raw_content)
        result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path=dest)
            )
        return result

    # =========================================================================
    # VFS-Only Operations
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return []
        return await mount.backend.list_versions(rel_path)

    async def restore_version(self, path: str, version: int) -> RestoreResult:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return RestoreResult(
                success=False,
                message="Version restore is only supported for VFS mounts.",
            )
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        result = await mount.backend.restore_version(rel_path, version)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def get_version_content(self, path: str, version: int) -> str | None:
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return None
        return await mount.backend.get_version_content(rel_path, version)

    async def list_trash(self) -> ListResult:
        all_entries: list[FileInfo] = []
        for mount in self._registry.list_mounts():
            if mount.mount_type != "vfs":
                continue
            result = await mount.backend.list_trash()
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
        if mount.mount_type != "vfs":
            return RestoreResult(
                success=False,
                message="Trash restore is only supported for VFS mounts.",
            )
        result = await mount.backend.restore_from_trash(rel_path)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        if result.success:
            await self._emit(
                FileEvent(event_type=EventType.FILE_RESTORED, path=path)
            )
        return result

    async def empty_trash(self) -> DeleteResult:
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            if mount.mount_type != "vfs":
                continue
            result = await mount.backend.empty_trash()
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

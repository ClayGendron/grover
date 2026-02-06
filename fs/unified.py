"""
Unified File System — single namespace routing to multiple backends.

Routes operations to storage backends via MountRegistry.
Enforces permissions before delegating write operations.
Handles path rewriting (mount prefix add/strip) transparently.

Usage:
    registry = MountRegistry()
    registry.add_mount(MountConfig(
        mount_path="/web",
        backend=database_fs,
        mount_type="vfs",
    ))
    ufs = UnifiedFileSystem(registry)
    async with ufs:
        result = await ufs.read("/web/main.py")
"""

from __future__ import annotations

from .mounts import MountConfig, MountRegistry
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


class UnifiedFileSystem:
    """
    Routes operations to backends via mount registry.

    Presents a single namespace to callers while delegating to the
    appropriate backend based on the path prefix. Enforces permissions
    and handles cross-mount copy/move.

    Supports async context manager for session lifecycle management.
    """

    def __init__(self, registry: MountRegistry) -> None:
        self._registry = registry
        self._entered_backends: list[object] = []

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> UnifiedFileSystem:
        """Enter context — enter all backend context managers."""
        for mount in self._registry.list_mounts():
            backend = mount.backend
            if hasattr(backend, "__aenter__"):
                await backend.__aenter__()
                self._entered_backends.append(backend)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context — exit all backend context managers.

        Ensures every backend gets its __aexit__ called even if an
        earlier one raises, preventing session leaks.
        """
        errors: list[Exception] = []
        for backend in reversed(self._entered_backends):
            if hasattr(backend, "__aexit__"):
                try:
                    await backend.__aexit__(exc_type, exc_val, exc_tb)
                except Exception as e:
                    errors.append(e)
        self._entered_backends.clear()
        if errors and exc_type is None:
            # Only raise if there was no original exception
            raise errors[0]

    # =========================================================================
    # Path Helpers
    # =========================================================================

    def _prefix_path(self, path: str | None, mount_path: str) -> str | None:
        """Add mount prefix to a backend-relative path."""
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def _prefix_file_info(self, info: FileInfo, mount: MountConfig) -> FileInfo:
        """Add mount prefix and metadata to a FileInfo (returns new instance).

        Creates a copy to avoid mutating the input, which could cause issues
        if the caller retains a reference to the original.
        """
        from dataclasses import replace

        prefixed_path = self._prefix_path(info.path, mount.mount_path) or info.path
        return replace(
            info,
            path=prefixed_path,
            mount_type=mount.mount_type,
            permission=self._registry.get_permission(prefixed_path).value,
        )

    def _check_writable(self, virtual_path: str) -> None:
        """Raise PermissionError if the path is read-only."""
        perm = self._registry.get_permission(virtual_path)
        if perm == Permission.READ_ONLY:
            raise PermissionError(
                f"Cannot write to read-only path: {virtual_path}"
            )

    def _extract_raw_content(self, formatted_content: str | None) -> str | None:
        """Extract raw file content from formatted ReadResult output.

        The read() method returns content wrapped in <file>...</file> tags
        with line numbers. This extracts the original content.
        """
        if formatted_content is None:
            return None

        if not formatted_content.startswith("<file>"):
            # Already raw content
            return formatted_content

        lines = formatted_content.split("\n")
        raw_lines = []
        for line in lines[1:]:  # Skip <file>
            if line.startswith("(") or line == "</file>" or line == "":
                continue
            # Remove line number prefix (e.g., "00001| ")
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
        """Read a file. Resolves mount and delegates."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.read(rel_path, offset, limit)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def list_dir(self, path: str = "/") -> ListResult:
        """
        List directory contents.

        Root "/" returns synthetic entries for each mount point.
        Other paths delegate to the appropriate backend.
        """
        path = normalize_path(path)

        # Root listing: show mount points
        if path == "/":
            return self._list_root()

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.list_dir(rel_path)

        # Prefix all entry paths (creates new instances)
        result.path = self._prefix_path(result.path, mount.mount_path) or path
        result.entries = [
            self._prefix_file_info(entry, mount)
            for entry in result.entries
        ]

        return result

    def _list_root(self) -> ListResult:
        """Build synthetic root listing from mount points."""
        entries = []
        for mount in self._registry.list_mounts():
            name = mount.mount_path.lstrip("/")
            entries.append(FileInfo(
                path=mount.mount_path,
                name=name,
                is_directory=True,
                permission=mount.permission.value,
                mount_type=mount.mount_type,
            ))
        return ListResult(
            success=True,
            message=f"Found {len(entries)} mount(s)",
            entries=entries,
            path="/",
        )

    async def exists(self, path: str) -> bool:
        """Check if a path exists."""
        path = normalize_path(path)

        # Root always exists
        if path == "/":
            return True

        # Check if path is a mount point itself
        if self._registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._registry.resolve(path)
        except FileNotFoundError:
            return False

        return await mount.backend.exists(rel_path)

    async def get_info(self, path: str) -> FileInfo | None:
        """Get metadata for a file or directory."""
        path = normalize_path(path)

        # Mount point info
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
        """
        Get permission and whether it's an explicit override.

        This allows callers to determine permissions without opening
        a separate database session.

        Args:
            path: Virtual path to check.

        Returns:
            Tuple of (permission_value, is_override).
            - permission_value: "read_write" or "read_only"
            - is_override: True if this specific path has an explicit
              read-only override, False if inheriting from mount default.
        """
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)

        permission = self._registry.get_permission(path)

        # Check if this specific path has an override (in read_only_paths)
        # The path is an override if it's explicitly in read_only_paths
        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths

        return permission.value, is_override

    # =========================================================================
    # Write Operations (permission-checked)
    # =========================================================================

    async def write(
        self, path: str, content: str, created_by: str = "agent"
    ) -> WriteResult:
        """Write a file. Checks permissions first."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.write(rel_path, content, created_by)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
    ) -> EditResult:
        """Edit a file using smart text replacement. Checks permissions."""
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
        return result

    async def delete(
        self, path: str, permanent: bool = False
    ) -> DeleteResult:
        """Delete a file or directory. Checks permissions."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.delete(rel_path, permanent)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def mkdir(
        self, path: str, parents: bool = True
    ) -> MkdirResult:
        """Create a directory. Checks permissions."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        result = await mount.backend.mkdir(rel_path, parents)
        result.path = self._prefix_path(result.path, mount.mount_path)
        result.created_dirs = [
            self._prefix_path(d, mount.mount_path) or d
            for d in result.created_dirs
        ]
        return result

    async def move(self, src: str, dest: str) -> MoveResult:
        """
        Move a file or directory.

        Same-mount: delegates directly.
        Cross-mount: copy to dest + delete from src.
        """
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
            # Same mount — delegate directly
            result = await src_mount.backend.move(src_rel, dest_rel)
            result.old_path = self._prefix_path(result.old_path, src_mount.mount_path)
            result.new_path = self._prefix_path(result.new_path, dest_mount.mount_path)
            return result

        # Cross-mount move: use public read() to respect permission checks,
        # write to dest, delete source
        read_result = await src_mount.backend.read(src_rel)
        if not read_result.success:
            return MoveResult(
                success=False,
                message=f"Cannot read source for cross-mount move: {read_result.message}",
            )

        # Extract raw content from formatted read result
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

        return MoveResult(
            success=True,
            message=f"Moved {src} -> {dest} (cross-mount)",
            old_path=src,
            new_path=dest,
        )

    async def copy(self, src: str, dest: str) -> WriteResult:
        """
        Copy a file.

        Same-mount: delegates directly.
        Cross-mount: read from source + write to dest.
        """
        src = normalize_path(src)
        dest = normalize_path(dest)

        try:
            self._check_writable(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        src_mount, src_rel = self._registry.resolve(src)
        dest_mount, dest_rel = self._registry.resolve(dest)

        if src_mount is dest_mount:
            # Same mount — delegate directly
            result = await src_mount.backend.copy(src_rel, dest_rel)
            result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
            return result

        # Cross-mount copy: use public read() to respect permission checks,
        # then write to dest
        read_result = await src_mount.backend.read(src_rel)
        if not read_result.success:
            return WriteResult(
                success=False,
                message=f"Cannot read source for cross-mount copy: {read_result.message}",
            )

        # Extract raw content from formatted read result
        raw_content = self._extract_raw_content(read_result.content)
        if raw_content is None:
            return WriteResult(
                success=False,
                message=f"Source file has no content: {src}",
            )

        result = await dest_mount.backend.write(dest_rel, raw_content)
        result.file_path = self._prefix_path(result.file_path, dest_mount.mount_path)
        return result

    # =========================================================================
    # VFS-Only Operations
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]:
        """List versions. Only VFS backends support versioning."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return []
        return await mount.backend.list_versions(rel_path)

    async def restore_version(
        self, path: str, version: int
    ) -> RestoreResult:
        """Restore a file to a previous version. VFS-only."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return RestoreResult(
                success=False,
                message="Version restore is only supported for cloud-stored files (/web).",
            )
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        result = await mount.backend.restore_version(rel_path, version)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def get_version_content(
        self, path: str, version: int
    ) -> str | None:
        """Get content of a specific version. VFS-only."""
        path = normalize_path(path)
        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return None
        return await mount.backend.get_version_content(rel_path, version)

    async def list_trash(self) -> ListResult:
        """List trashed files across all VFS mounts."""
        all_entries = []
        for mount in self._registry.list_mounts():
            if mount.mount_type != "vfs":
                continue
            result = await mount.backend.list_trash()
            if result.success:
                prefixed_entries = [
                    self._prefix_file_info(entry, mount)
                    for entry in result.entries
                ]
                all_entries.extend(prefixed_entries)

        return ListResult(
            success=True,
            message=f"Found {len(all_entries)} item(s) in trash",
            entries=all_entries,
            path="/__trash__",
        )

    async def restore_from_trash(self, path: str) -> RestoreResult:
        """Restore a file from trash. Paths should include mount prefix."""
        path = normalize_path(path)
        try:
            self._check_writable(path)
        except PermissionError as e:
            return RestoreResult(success=False, message=str(e))

        mount, rel_path = self._registry.resolve(path)
        if mount.mount_type != "vfs":
            return RestoreResult(
                success=False,
                message="Trash restore is only supported for cloud-stored files (/web).",
            )
        result = await mount.backend.restore_from_trash(rel_path)
        result.file_path = self._prefix_path(result.file_path, mount.mount_path)
        return result

    async def empty_trash(self) -> DeleteResult:
        """Empty trash across all VFS mounts."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._registry.list_mounts():
            if mount.mount_type != "vfs":
                continue
            result = await mount.backend.empty_trash()
            if not result.success:
                return result
            # Accumulate actual deleted count, not just mount count
            total_deleted += result.total_deleted or 0
            mounts_processed += 1

        return DeleteResult(
            success=True,
            message=f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)",
            total_deleted=total_deleted,
            permanent=True,
        )

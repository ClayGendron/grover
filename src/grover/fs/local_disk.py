"""LocalDiskBackend — no versioning, direct disk access."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

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
from .utils import (
    guess_mime_type,
    is_binary_file,
    is_text_file,
    normalize_path,
    replace,
    split_path,
    validate_path,
)

# Constants (matching base.py for consistency)
DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024  # 50KB
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


class LocalDiskBackend:
    """Pure local disk access backend. No database, no versioning.

    Implements the StorageBackend protocol used by UnifiedFileSystem.
    All operations are performed directly on the host filesystem.

    Security: _resolve_path() ensures all paths stay within host_dir,
    preventing path traversal attacks.
    """

    def __init__(
        self,
        host_dir: Path | str,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self.host_dir = Path(host_dir).resolve()
        self.max_file_size = max_file_size

        if not self.host_dir.exists():
            raise FileNotFoundError(f"Host directory does not exist: {self.host_dir}")
        if not self.host_dir.is_dir():
            raise NotADirectoryError(f"Host path is not a directory: {self.host_dir}")

    # =========================================================================
    # Path Resolution & Security
    # =========================================================================

    def _resolve_path(self, virtual_path: str, follow_symlinks: bool = False) -> Path:
        """Resolve a virtual path to a physical path on disk.

        Validates that the resolved path stays within host_dir.
        By default, rejects symlinks to prevent TOCTOU attacks.
        """
        virtual_path = normalize_path(virtual_path)
        rel = virtual_path.lstrip("/")
        if not rel:
            return self.host_dir

        candidate = self.host_dir / rel

        if not follow_symlinks:
            current = self.host_dir
            for part in Path(rel).parts:
                current = current / part
                if current.is_symlink():
                    raise PermissionError(
                        f"Symlinks not allowed: {virtual_path} contains symlink at "
                        f"{current.relative_to(self.host_dir)}"
                    )

        resolved = candidate.resolve()

        try:
            resolved.relative_to(self.host_dir.resolve())
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: {virtual_path} resolves outside mount directory"
            ) from None

        return resolved

    def _to_virtual_path(self, physical_path: Path) -> str:
        """Convert a physical path back to a virtual path."""
        rel = physical_path.resolve().relative_to(self.host_dir)
        vpath = "/" + str(rel).replace("\\", "/")
        return vpath if vpath != "/." else "/"

    # =========================================================================
    # Context Manager (no-op for local disk)
    # =========================================================================

    async def __aenter__(self) -> LocalDiskBackend:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        pass

    # =========================================================================
    # Raw Content Access
    # =========================================================================

    async def _read_content(self, path: str) -> str | None:
        """Read raw file content from disk."""
        try:
            resolved = self._resolve_path(path)
            if not resolved.is_file():
                return None
            return await asyncio.to_thread(resolved.read_text, "utf-8")
        except (PermissionError, OSError, UnicodeDecodeError):
            return None

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> ReadResult:
        """Read a file from disk with pagination and line numbers."""
        offset = max(0, offset)
        limit = max(1, limit)

        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return ReadResult(success=False, message=str(e))

        if not resolved.exists():
            return ReadResult(success=False, message=f"File not found: {path}")

        if resolved.is_dir():
            return ReadResult(
                success=False, message=f"Path is a directory, not a file: {path}"
            )

        if await asyncio.to_thread(is_binary_file, resolved):
            return ReadResult(success=False, message=f"Cannot read binary file: {path}")

        try:
            file_size = resolved.stat().st_size
        except OSError as e:
            return ReadResult(success=False, message=f"Cannot access file: {e}")

        if file_size > self.max_file_size:
            return ReadResult(
                success=False,
                message=(
                    f"File too large ({file_size:,} bytes, "
                    f"limit {self.max_file_size:,}): {path}"
                ),
            )

        try:
            content = await asyncio.to_thread(resolved.read_text, "utf-8")
        except UnicodeDecodeError:
            return ReadResult(
                success=False, message=f"Cannot read file (not UTF-8): {path}"
            )
        except OSError as e:
            return ReadResult(success=False, message=f"Cannot read file: {e}")

        return self._format_read_output(content, path, offset, limit)

    def _format_read_output(
        self, content: str, path: str, offset: int, limit: int
    ) -> ReadResult:
        """Format file content with line numbers and pagination."""
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines == 0 or (total_lines == 1 and lines[0] == ""):
            return ReadResult(
                success=True,
                message="File is empty.",
                content="",
                file_path=path,
                total_lines=0,
                lines_read=0,
                truncated=False,
            )

        output_lines: list[str] = []
        bytes_read = 0
        end_line = min(len(lines), offset + limit)

        for i in range(offset, end_line):
            line = lines[i]
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."
            line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)
            if bytes_read + line_bytes > MAX_BYTES:
                break
            output_lines.append(line)
            bytes_read += line_bytes

        formatted_lines = [
            f"{str(i + offset + 1).zfill(5)}| {line}"
            for i, line in enumerate(output_lines)
        ]

        formatted_content = "<file>\n" + "\n".join(formatted_lines)
        last_read_line = offset + len(output_lines)
        has_more = total_lines > last_read_line

        if has_more:
            formatted_content += (
                f"\n\n(File has more lines. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        else:
            formatted_content += f"\n\n(End of file - total {total_lines} lines)"

        formatted_content += "\n</file>"

        return ReadResult(
            success=True,
            message=f"Read {len(output_lines)} lines from {path}",
            content=formatted_content,
            file_path=path,
            total_lines=total_lines,
            lines_read=len(output_lines),
            truncated=has_more,
        )

    async def list_dir(self, path: str = "/") -> ListResult:
        """List directory contents from disk."""
        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return ListResult(success=False, message=str(e))

        if not resolved.exists():
            return ListResult(success=False, message=f"Directory not found: {path}")

        if not resolved.is_dir():
            return ListResult(success=False, message=f"Not a directory: {path}")

        def _scan() -> list[FileInfo]:
            entries: list[FileInfo] = []
            for entry in os.scandir(resolved):
                try:
                    entry_path = Path(entry.path)
                    virtual = self._to_virtual_path(entry_path)
                    st = entry.stat()
                    is_dir = entry.is_dir()
                    entries.append(
                        FileInfo(
                            path=virtual,
                            name=entry.name,
                            is_directory=is_dir,
                            size_bytes=st.st_size if not is_dir else None,
                            mime_type=(
                                guess_mime_type(entry.name) if not is_dir else None
                            ),
                            created_at=datetime.fromtimestamp(st.st_ctime, tz=UTC),
                            updated_at=datetime.fromtimestamp(st.st_mtime, tz=UTC),
                        )
                    )
                except (OSError, ValueError):
                    continue
            entries.sort(key=lambda x: (not x.is_directory, x.name.lower()))
            return entries

        try:
            entries = await asyncio.to_thread(_scan)
        except OSError as e:
            return ListResult(success=False, message=f"Cannot list directory: {e}")

        return ListResult(
            success=True,
            message=f"Listed {len(entries)} items in {path}",
            entries=entries,
            path=path,
        )

    async def exists(self, path: str) -> bool:
        """Check if a path exists on disk."""
        try:
            resolved = self._resolve_path(path)
            return await asyncio.to_thread(resolved.exists)
        except PermissionError:
            return False

    async def get_info(self, path: str) -> FileInfo | None:
        """Get file/directory metadata from disk."""
        try:
            resolved = self._resolve_path(path)
        except PermissionError:
            return None

        if not await asyncio.to_thread(resolved.exists):
            return None

        def _stat() -> FileInfo | None:
            try:
                st = resolved.stat()
                is_dir = resolved.is_dir()
                return FileInfo(
                    path=path,
                    name=resolved.name,
                    is_directory=is_dir,
                    size_bytes=st.st_size if not is_dir else None,
                    mime_type=guess_mime_type(resolved.name) if not is_dir else None,
                    created_at=datetime.fromtimestamp(st.st_ctime, tz=UTC),
                    updated_at=datetime.fromtimestamp(st.st_mtime, tz=UTC),
                )
            except OSError:
                return None

        return await asyncio.to_thread(_stat)

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
    ) -> WriteResult:
        """Write content to a file on disk. Atomic via tempfile + replace."""
        valid, error = validate_path(path)
        if not valid:
            return WriteResult(success=False, message=error)

        path = normalize_path(path)
        _, name = split_path(path)

        if not is_text_file(name):
            return WriteResult(
                success=False,
                message=(
                    f"Cannot write non-text file: {name}. "
                    "Use allowed extensions (.py, .js, .json, .md, etc.)"
                ),
            )

        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        def _write() -> bool:
            was_created = not resolved.exists()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(resolved.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(resolved)
            except Exception:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise
            return was_created

        try:
            created = await asyncio.to_thread(_write)
        except OSError as e:
            return WriteResult(success=False, message=f"Failed to write file: {e}")

        return WriteResult(
            success=True,
            message=f"{'Created' if created else 'Updated'}: {path}",
            file_path=path,
            created=created,
            version=1,
        )

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
    ) -> EditResult:
        """Edit a file using smart text replacement."""
        valid, error = validate_path(path)
        if not valid:
            return EditResult(success=False, message=error)

        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return EditResult(success=False, message=str(e))

        if not resolved.exists():
            return EditResult(success=False, message=f"File not found: {path}")

        if resolved.is_dir():
            return EditResult(success=False, message=f"Cannot edit directory: {path}")

        try:
            content = await asyncio.to_thread(resolved.read_text, "utf-8")
        except (UnicodeDecodeError, OSError) as e:
            return EditResult(
                success=False, message=f"Cannot read file for editing: {e}"
            )

        result = replace(content, old_string, new_string, replace_all)
        if not result.success:
            return EditResult(
                success=False,
                message=result.error or "Edit failed",
                file_path=path,
            )

        new_content = result.content
        assert new_content is not None

        write_result = await self.write(path, new_content, created_by)
        if not write_result.success:
            return EditResult(
                success=False, message=write_result.message, file_path=path
            )

        return EditResult(
            success=True,
            message=f"Edit applied to {path}",
            file_path=path,
            version=1,
        )

    async def delete(self, path: str, permanent: bool = False) -> DeleteResult:
        """Delete a file or directory from disk (always permanent)."""
        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return DeleteResult(success=False, message=str(e))

        if not resolved.exists():
            return DeleteResult(success=False, message=f"File not found: {path}")

        is_dir = resolved.is_dir()

        def _delete() -> None:
            if is_dir:
                shutil.rmtree(resolved)
            else:
                resolved.unlink()

        try:
            await asyncio.to_thread(_delete)
        except OSError as e:
            return DeleteResult(success=False, message=f"Failed to delete: {e}")

        return DeleteResult(
            success=True,
            message=(
                f"Permanently deleted: {path} "
                "(local mount — no trash, cannot be undone)"
            ),
            file_path=path,
            permanent=True,
        )

    async def mkdir(self, path: str, parents: bool = True) -> MkdirResult:
        """Create a directory on disk."""
        valid, error = validate_path(path)
        if not valid:
            return MkdirResult(success=False, message=error)

        try:
            resolved = self._resolve_path(path)
        except PermissionError as e:
            return MkdirResult(success=False, message=str(e))

        if resolved.exists():
            if resolved.is_dir():
                return MkdirResult(
                    success=True,
                    message=f"Directory already exists: {path}",
                    path=path,
                    created_dirs=[],
                )
            return MkdirResult(success=False, message=f"Path exists as file: {path}")

        def _mkdir() -> list[str]:
            created: list[str] = []
            to_create = resolved
            parents_to_check: list[Path] = []
            while not to_create.exists():
                parents_to_check.append(to_create)
                to_create = to_create.parent

            resolved.mkdir(parents=parents, exist_ok=True)

            created = [
                self._to_virtual_path(p)
                for p in reversed(parents_to_check)
                if p.exists()
            ]
            return created

        try:
            created_dirs = await asyncio.to_thread(_mkdir)
        except OSError as e:
            return MkdirResult(
                success=False, message=f"Failed to create directory: {e}"
            )

        return MkdirResult(
            success=True,
            message=f"Created directory: {path}",
            path=path,
            created_dirs=created_dirs,
        )

    async def move(self, src: str, dest: str) -> MoveResult:
        """Move a file or directory on disk."""
        valid, error = validate_path(dest)
        if not valid:
            return MoveResult(success=False, message=error)

        try:
            src_resolved = self._resolve_path(src)
            dest_resolved = self._resolve_path(dest)
        except PermissionError as e:
            return MoveResult(success=False, message=str(e))

        if not src_resolved.exists():
            return MoveResult(success=False, message=f"Source not found: {src}")

        def _move() -> None:
            dest_resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_resolved), str(dest_resolved))

        try:
            await asyncio.to_thread(_move)
        except OSError as e:
            return MoveResult(success=False, message=f"Failed to move: {e}")

        return MoveResult(
            success=True,
            message=f"Moved {src} to {dest}",
            old_path=src,
            new_path=dest,
        )

    async def copy(self, src: str, dest: str) -> WriteResult:
        """Copy a file on disk."""
        valid, error = validate_path(dest)
        if not valid:
            return WriteResult(success=False, message=error)

        try:
            src_resolved = self._resolve_path(src)
            dest_resolved = self._resolve_path(dest)
        except PermissionError as e:
            return WriteResult(success=False, message=str(e))

        if not src_resolved.exists():
            return WriteResult(success=False, message=f"Source not found: {src}")

        if src_resolved.is_dir():
            return WriteResult(
                success=False, message="Directory copy not yet implemented"
            )

        def _copy() -> None:
            dest_resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_resolved), str(dest_resolved))

        try:
            await asyncio.to_thread(_copy)
        except OSError as e:
            return WriteResult(success=False, message=f"Failed to copy: {e}")

        return WriteResult(
            success=True,
            message=f"Copied {src} to {dest}",
            file_path=dest,
            created=True,
            version=1,
        )

    # =========================================================================
    # VFS-Only Operations (stubs — not supported for local disk)
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]:
        """Versioning not supported for local disk."""
        return []

    async def restore_version(self, path: str, version: int) -> RestoreResult:
        """Versioning not supported for local disk."""
        return RestoreResult(
            success=False,
            message="Version restore is not supported for local disk mounts.",
        )

    async def get_version_content(self, path: str, version: int) -> str | None:
        """Versioning not supported for local disk."""
        return None

    async def list_trash(self) -> ListResult:
        """Trash not supported for local disk."""
        return ListResult(
            success=True,
            entries=[],
            message="Trash not supported for local mounts",
            path="/__trash__",
        )

    async def restore_from_trash(self, path: str) -> RestoreResult:
        """Trash not supported for local disk."""
        return RestoreResult(
            success=False,
            message="Trash is not supported for local disk mounts.",
        )

    async def empty_trash(self) -> DeleteResult:
        """Trash not supported for local disk."""
        return DeleteResult(
            success=True,
            message="No trash to empty for local disk mount.",
            permanent=True,
        )

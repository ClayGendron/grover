"""
Storage backend protocol for the Unified File System.

Defines the interface that storage backends must implement.
Both LocalDiskBackend and DatabaseFileSystem implement this protocol.
"""

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the interface for filesystem backends.

    Both LocalDiskBackend and DatabaseFileSystem implement this protocol.
    Using @runtime_checkable allows isinstance() checks at runtime.

    Note: Context manager methods are optional - backends that don't need
    session management can omit them (like LocalDiskBackend's no-op impl).
    """

    # =========================================================================
    # Context Manager (optional for backends without session management)
    # =========================================================================

    async def __aenter__(self) -> "StorageBackend":
        """Enter async context (e.g., acquire database session)."""
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context (e.g., release database session)."""
        ...

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read(
        self, path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read a file with optional pagination.

        Args:
            path: Path to the file (relative to mount root).
            offset: Line number to start reading from (0-based).
            limit: Maximum number of lines to read.

        Returns:
            ReadResult with formatted content including line numbers.
        """
        ...

    async def list_dir(self, path: str = "/") -> ListResult:
        """List directory contents.

        Args:
            path: Directory path to list.

        Returns:
            ListResult with list of FileInfo entries.
        """
        ...

    async def exists(self, path: str) -> bool:
        """Check if a path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        ...

    async def get_info(self, path: str) -> FileInfo | None:
        """Get metadata for a file or directory.

        Args:
            path: Path to get info for.

        Returns:
            FileInfo if path exists, None otherwise.
        """
        ...

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def write(
        self, path: str, content: str, created_by: str = "agent"
    ) -> WriteResult:
        """Write content to a file.

        Creates the file if it doesn't exist, overwrites if it does.
        Backends may create a version snapshot before overwriting.

        Args:
            path: Path to write to.
            content: Content to write.
            created_by: Who initiated the write ("agent" or "user").

        Returns:
            WriteResult indicating success/failure.
        """
        ...

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
    ) -> EditResult:
        """Edit a file using string replacement.

        Args:
            path: Path to the file to edit.
            old_string: Text to find and replace.
            new_string: Replacement text.
            replace_all: If True, replace all occurrences.
            created_by: Who initiated the edit.

        Returns:
            EditResult indicating success/failure.
        """
        ...

    async def delete(self, path: str, permanent: bool = False) -> DeleteResult:
        """Delete a file or directory.

        Args:
            path: Path to delete.
            permanent: If True, skip trash (if supported).

        Returns:
            DeleteResult indicating success/failure.
        """
        ...

    async def mkdir(self, path: str, parents: bool = True) -> MkdirResult:
        """Create a directory.

        Args:
            path: Directory path to create.
            parents: If True, create parent directories as needed.

        Returns:
            MkdirResult indicating success/failure.
        """
        ...

    async def move(self, src: str, dest: str) -> MoveResult:
        """Move a file or directory.

        Args:
            src: Source path.
            dest: Destination path.

        Returns:
            MoveResult indicating success/failure.
        """
        ...

    async def copy(self, src: str, dest: str) -> WriteResult:
        """Copy a file.

        Args:
            src: Source file path.
            dest: Destination file path.

        Returns:
            WriteResult indicating success/failure.
        """
        ...

    # =========================================================================
    # VFS-Only Operations
    #
    # These operations are only meaningful for VFS backends (with database).
    # Local disk backends should return empty/failure results.
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]:
        """List version history for a file.

        Returns empty list for backends without versioning support.
        """
        ...

    async def restore_version(self, path: str, version: int) -> RestoreResult:
        """Restore a file to a previous version.

        Returns failure for backends without versioning support.
        """
        ...

    async def get_version_content(self, path: str, version: int) -> str | None:
        """Get content of a specific file version.

        Returns None for backends without versioning support.
        """
        ...

    async def list_trash(self) -> ListResult:
        """List files in trash.

        Returns empty list for backends without trash support.
        """
        ...

    async def restore_from_trash(self, path: str) -> RestoreResult:
        """Restore a file from trash.

        Returns failure for backends without trash support.
        """
        ...

    async def empty_trash(self) -> DeleteResult:
        """Permanently delete all files in trash.

        Returns success with total_deleted=0 for backends without trash.
        """
        ...
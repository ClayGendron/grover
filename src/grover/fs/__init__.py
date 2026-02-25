"""Filesystem layer — storage backends, mounts, permissions, capabilities."""

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import (
    AuthenticationRequiredError,
    CapabilityNotSupportedError,
    ConsistencyError,
    GroverError,
    MountNotFoundError,
    PathNotFoundError,
    StorageError,
)
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import (
    StorageBackend,
    SupportsReBAC,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.fs.types import (
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
    VersionInfo,
    WriteResult,
)
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.fs.utils import format_read_output
from grover.search.results import GlobResult, GrepResult, ListDirResult, TreeResult

__all__ = [
    "AuthenticationRequiredError",
    "CapabilityNotSupportedError",
    "ConsistencyError",
    "DatabaseFileSystem",
    "DeleteResult",
    "EditResult",
    "FileInfo",
    "GetVersionContentResult",
    "GlobResult",
    "GrepResult",
    "GroverError",
    "ListDirResult",
    "ListResult",
    "ListVersionsResult",
    "LocalFileSystem",
    "MkdirResult",
    "MountConfig",
    "MountNotFoundError",
    "MountRegistry",
    "MoveResult",
    "PathNotFoundError",
    "Permission",
    "ReadResult",
    "RestoreResult",
    "StorageBackend",
    "StorageError",
    "SupportsReBAC",
    "SupportsReconcile",
    "SupportsTrash",
    "SupportsVersions",
    "TreeResult",
    "UserScopedFileSystem",
    "VersionInfo",
    "WriteResult",
    "format_read_output",
]

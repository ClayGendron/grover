"""Filesystem layer — storage backends, mounts, permissions, capabilities."""

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import (
    AuthenticationRequiredError,
    CapabilityNotSupportedError,
    ConsistencyError,
    GroverError,
    MountNotFoundError,
    PathNotFoundError,
    SchemaIncompatibleError,
    StorageError,
)
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import (
    StorageBackend,
    SupportsConnections,
    SupportsReBAC,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.fs.utils import format_read_output
from grover.types.operations import (
    DeleteResult,
    EditResult,
    FileInfoResult,
    GetVersionContentResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    WriteResult,
)
from grover.types.search import (
    GlobResult,
    GrepResult,
    ListDirResult,
    ShareSearchResult,
    TrashResult,
    TreeResult,
    VersionResult,
)

__all__ = [
    "AuthenticationRequiredError",
    "CapabilityNotSupportedError",
    "ConsistencyError",
    "DatabaseFileSystem",
    "DeleteResult",
    "EditResult",
    "FileInfoResult",
    "GetVersionContentResult",
    "GlobResult",
    "GrepResult",
    "GroverError",
    "ListDirResult",
    "LocalFileSystem",
    "MkdirResult",
    "MountNotFoundError",
    "MountRegistry",
    "MoveResult",
    "PathNotFoundError",
    "Permission",
    "ReadResult",
    "RestoreResult",
    "SchemaIncompatibleError",
    "ShareSearchResult",
    "StorageBackend",
    "StorageError",
    "SupportsConnections",
    "SupportsReBAC",
    "SupportsReconcile",
    "SupportsTrash",
    "SupportsVersions",
    "TrashResult",
    "TreeResult",
    "UserScopedFileSystem",
    "VersionResult",
    "WriteResult",
    "format_read_output",
]

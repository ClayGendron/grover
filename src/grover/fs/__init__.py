"""Filesystem layer — storage backends, mounts, permissions."""

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import (
    ConsistencyError,
    GroverError,
    MountNotFoundError,
    PathNotFoundError,
    StorageError,
)
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.protocol import StorageBackend
from grover.fs.types import (
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
from grover.fs.utils import format_read_output
from grover.fs.vfs import VFS

__all__ = [
    "VFS",
    "ConsistencyError",
    "DatabaseFileSystem",
    "DeleteResult",
    "EditResult",
    "FileInfo",
    "GroverError",
    "ListResult",
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
    "VersionInfo",
    "WriteResult",
    "format_read_output",
]

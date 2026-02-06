"""Filesystem layer — storage backends, mounts, permissions."""

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_disk import LocalDiskBackend
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
from grover.fs.unified import UnifiedFileSystem

__all__ = [
    "DatabaseFileSystem",
    "DeleteResult",
    "EditResult",
    "FileInfo",
    "ListResult",
    "LocalDiskBackend",
    "LocalFileSystem",
    "MkdirResult",
    "MountConfig",
    "MountRegistry",
    "MoveResult",
    "Permission",
    "ReadResult",
    "RestoreResult",
    "StorageBackend",
    "UnifiedFileSystem",
    "VersionInfo",
    "WriteResult",
]

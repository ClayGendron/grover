"""Storage providers — disk, database, and future backends (fsspec, etc.)."""

from grover.fs.providers.storage.disk import DiskStorageProvider

__all__ = [
    "DiskStorageProvider",
]

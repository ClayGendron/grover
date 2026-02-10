"""Custom exception hierarchy for the Grover filesystem layer."""


class GroverError(Exception):
    """Base exception for all Grover filesystem errors."""


class PathNotFoundError(GroverError):
    """Raised when a file or directory path does not exist."""


class MountNotFoundError(GroverError):
    """Raised when no mount matches the given virtual path."""


class StorageError(GroverError):
    """Raised on storage backend failures (DB connection, disk I/O, etc.)."""


class ConsistencyError(GroverError):
    """Raised when data integrity is compromised (e.g. phantom metadata)."""

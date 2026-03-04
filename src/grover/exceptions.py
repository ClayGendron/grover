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


class CapabilityNotSupportedError(GroverError):
    """Raised when a backend doesn't support a requested capability."""


class AuthenticationRequiredError(GroverError):
    """Raised when an authenticated mount is accessed without a user_id."""


class SchemaIncompatibleError(GroverError):
    """Raised when a database schema is incompatible with the current code.

    This occurs when an existing database was created by an older version
    of Grover and is missing columns required by the current code. Run the
    migration script to update the schema::

        from grover.migrations import backfill_alpha_refactor

        await backfill_alpha_refactor(engine)
    """

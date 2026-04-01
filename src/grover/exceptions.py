"""Grover exception hierarchy.

When ``GroverFileSystem`` is constructed with ``raises=True``, the
``_error()`` method raises one of these exceptions instead of returning
``GroverResult(success=False)``.  The original ``GroverResult`` is
attached to the exception as ``.result`` so callers can inspect partial
successes in batch operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grover.results import GroverResult


class GroverError(Exception):
    """Base exception for all Grover errors."""

    def __init__(self, message: str, result: GroverResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class NotFoundError(GroverError):
    """A path does not exist or is not the expected kind."""


class MountError(GroverError):
    """No mount found for the given path."""


class WriteConflictError(GroverError):
    """Write rejected — file exists with overwrite=False, or target is invalid."""


class ValidationError(GroverError):
    """Invalid arguments, patterns, or missing configuration."""


class GraphError(GroverError):
    """A graph algorithm failed."""


def _classify_error(
    message: str,
    errors: list[str],
    result: GroverResult,
) -> GroverError:
    """Map error messages to the appropriate exception type."""
    first = errors[0] if errors else message
    if "Not found:" in first or "Not a directory:" in first:
        return NotFoundError(message, result)
    if "No mount found" in first:
        return MountError(message, result)
    if "Already exists" in first or "Cannot write" in first or "Cannot delete" in first:
        return WriteConflictError(message, result)
    if "failed:" in first:
        return GraphError(message, result)
    if any(kw in first for kw in ("requires", "Invalid", "Duplicate", "Source not found")):
        return ValidationError(message, result)
    return GroverError(message, result)

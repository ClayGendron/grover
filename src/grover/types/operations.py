"""Operation result types — FileOperationResult and all subclasses.

Every content-operation method in Grover returns a typed subclass of
``FileOperationResult``.  These are non-chainable (no set algebra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class FileOperationResult:
    """Base for content operations. Non-chainable.

    Subclasses: ``ReadResult``, ``WriteResult``, ``EditResult``,
    ``DeleteResult``, ``MoveResult``, ``MkdirResult``, ``RestoreResult``,
    ``GetVersionContentResult``, ``ShareResult``, ``ConnectionResult``,
    ``FileInfoResult``.
    """

    path: str = ""
    content: str = ""
    message: str = ""
    success: bool = True
    line_start: int = 0
    line_offset: int = 0
    version: int = 0


@dataclass
class ReadResult(FileOperationResult):
    """Result of a read operation."""

    total_lines: int = 0
    lines_read: int = 0
    truncated: bool = False


@dataclass
class WriteResult(FileOperationResult):
    """Result of a write operation."""

    created: bool = False


@dataclass
class EditResult(FileOperationResult):
    """Result of an edit operation."""


@dataclass
class DeleteResult(FileOperationResult):
    """Result of a delete operation."""

    permanent: bool = False
    total_deleted: int | None = None


@dataclass
class MkdirResult(FileOperationResult):
    """Result of a mkdir operation."""

    created_dirs: list[str] = field(default_factory=list)


@dataclass
class MoveResult(FileOperationResult):
    """Result of a move operation."""

    old_path: str = ""
    new_path: str = ""


@dataclass
class RestoreResult(FileOperationResult):
    """Result of a restore operation."""

    restored_version: int = 0


@dataclass
class GetVersionContentResult(FileOperationResult):
    """Result of a get_version_content operation."""


@dataclass
class ShareResult(FileOperationResult):
    """Result of a share/unshare operation."""

    grantee_id: str = ""
    permission: str = ""
    granted_by: str = ""


@dataclass
class ConnectionResult(FileOperationResult):
    """Result of a connection add/remove operation."""

    source_path: str = ""
    target_path: str = ""
    connection_type: str = ""


@dataclass
class FileInfoResult(FileOperationResult):
    """File/directory metadata as a result type.

    Replaces the old ``FileInfo`` dataclass.  Inherits ``path`` and
    ``version`` from the base.
    """

    is_directory: bool = False
    mime_type: str = "text/plain"
    size_bytes: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    permission: str | None = None
    mount_type: str | None = None


@dataclass(frozen=True, slots=True)
class VersionChainError:
    """Detail about a single failed version in chain verification."""

    version: int
    expected_hash: str
    actual_hash: str
    error: str


@dataclass
class VerifyVersionResult(FileOperationResult):
    """Result of verifying a file's version chain integrity."""

    versions_checked: int = 0
    versions_passed: int = 0
    versions_failed: int = 0
    errors: list[VersionChainError] = field(default_factory=list)

"""Result types: ReadResult, WriteResult, EditResult, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class FileInfo:
    """File/directory metadata."""

    path: str
    name: str
    is_directory: bool
    size_bytes: int | None = None
    mime_type: str | None = None
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    permission: str | None = None
    mount_type: str | None = None


@dataclass
class VersionInfo:
    """Version history entry."""

    version: int
    content_hash: str
    size_bytes: int
    created_at: datetime
    created_by: str | None = None


@dataclass
class ReadResult:
    """Result of a read operation."""

    success: bool
    message: str
    content: str | None = None
    file_path: str | None = None
    total_lines: int | None = None
    lines_read: int | None = None
    truncated: bool = False
    offset: int = 0


@dataclass
class WriteResult:
    """Result of a write operation."""

    success: bool
    message: str
    file_path: str | None = None
    created: bool = False
    version: int = 1


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    message: str
    file_path: str | None = None
    version: int = 1


@dataclass
class DeleteResult:
    """Result of a delete operation."""

    success: bool
    message: str
    file_path: str | None = None
    permanent: bool = False
    total_deleted: int | None = None


@dataclass
class MkdirResult:
    """Result of a mkdir operation."""

    success: bool
    message: str
    path: str | None = None
    created_dirs: list[str] = field(default_factory=list)


@dataclass
class ListResult:
    """Result of a list directory operation."""

    success: bool
    message: str
    entries: list[FileInfo] = field(default_factory=list)
    path: str = "/"


@dataclass
class MoveResult:
    """Result of a move operation."""

    success: bool
    message: str
    old_path: str | None = None
    new_path: str | None = None


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    message: str
    file_path: str | None = None
    restored_version: int = 0
    current_version: int = 0


@dataclass
class ListVersionsResult:
    """Result of a list_versions operation."""

    success: bool
    message: str
    versions: list[VersionInfo] = field(default_factory=list)


@dataclass
class GlobResult:
    """Result of a glob operation."""

    success: bool
    message: str
    entries: list[FileInfo] = field(default_factory=list)
    pattern: str = ""
    path: str = "/"


@dataclass
class GrepMatch:
    """A single grep match within a file."""

    file_path: str
    line_number: int  # 1-indexed
    line_content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class GrepResult:
    """Result of a grep operation."""

    success: bool
    message: str
    matches: list[GrepMatch] = field(default_factory=list)
    pattern: str = ""
    path: str = "/"
    files_searched: int = 0
    files_matched: int = 0
    truncated: bool = False


@dataclass
class TreeResult:
    """Result of a tree operation."""

    success: bool
    message: str
    entries: list[FileInfo] = field(default_factory=list)
    path: str = "/"
    total_files: int = 0
    total_dirs: int = 0


@dataclass
class GetVersionContentResult:
    """Result of a get_version_content operation."""

    success: bool
    message: str
    content: str | None = None


@dataclass
class ShareInfo:
    """Share metadata."""

    path: str
    grantee_id: str
    permission: str
    granted_by: str
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass
class ShareResult:
    """Result of a share/unshare operation."""

    success: bool
    message: str
    share: ShareInfo | None = None


@dataclass
class ListSharesResult:
    """Result of a list_shares operation."""

    success: bool
    message: str
    shares: list[ShareInfo] = field(default_factory=list)

"""File and FileVersion models with diff-based versioning.

Provides ``FileBase`` and ``FileVersionBase`` non-table base classes.
Subclass with ``table=True`` and a custom ``__tablename__`` to use a
different table name per backend.
"""

from __future__ import annotations

import difflib
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel
from unidiff import PatchSet
from unidiff.constants import LINE_TYPE_ADDED, LINE_TYPE_CONTEXT, LINE_TYPE_NO_NEWLINE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT_INTERVAL: int = 20
"""Take a full snapshot every N versions (forward diffs between snapshots)."""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FileBase(SQLModel):
    """Base fields for a tracked file. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(index=True, unique=True)
    parent_path: str = Field(default="")
    name: str = Field(default="")
    is_directory: bool = Field(default=False)
    mime_type: str = Field(default="text/plain")
    content: str | None = Field(default=None)
    content_hash: str | None = Field(default=None)
    size_bytes: int = Field(default=0)
    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)
    current_version: int = Field(default=1)
    original_path: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )


class File(FileBase, table=True):
    """Default file table — ``grover_files``."""

    __tablename__ = "grover_files"


class FileVersionBase(SQLModel):
    """Base fields for a file version record. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    file_id: str = Field(index=True)
    version: int = Field(default=1)
    is_snapshot: bool = Field(default=False)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    size_bytes: int = Field(default=0)
    created_by: str | None = Field(default=None)
    change_summary: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )


class FileVersion(FileVersionBase, table=True):
    """Default file version table — ``grover_file_versions``."""

    __tablename__ = "grover_file_versions"


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"


def compute_diff(old: str, new: str) -> str:
    """Compute a unified diff from *old* to *new*.

    Returns a standard unified diff string (empty string if no changes).
    The output is parseable by ``unidiff.PatchSet`` and compatible with
    standard tools like ``patch``.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    raw = list(difflib.unified_diff(old_lines, new_lines, fromfile="a", tofile="b"))
    if not raw:
        return ""
    # difflib doesn't emit "\ No newline at end of file" markers, but
    # unidiff (and GNU patch) require them.  Insert after any content
    # line that lacks a trailing newline.
    out: list[str] = []
    for line in raw:
        out.append(line)
        if line and line[0] in ("+", "-", " ") and not line.endswith("\n"):
            out[-1] = line + "\n"
            out.append(_NO_NEWLINE_MARKER)
    return "".join(out)


def apply_diff(base: str, diff: str) -> str:
    """Apply a unified diff to *base* and return the resulting text.

    Uses ``unidiff.PatchSet`` for robust parsing of the diff, including
    correct handling of ``\\ No newline at end of file`` markers.
    """
    if not diff:
        return base

    patch = PatchSet(diff)
    if not patch:
        return base

    patched_file = patch[0]
    source_lines = base.splitlines(keepends=True)
    result_lines = list(source_lines)

    for hunk in reversed(patched_file):
        new_lines: list[str] = []
        prev_line_type: str | None = None

        for line in hunk:
            if line.line_type == LINE_TYPE_NO_NEWLINE:
                # The marker means the preceding line had no trailing newline.
                # Strip the \n we appended only if that line is part of the
                # target (context or added).
                if (
                    prev_line_type in (LINE_TYPE_CONTEXT, LINE_TYPE_ADDED)
                    and new_lines
                    and new_lines[-1].endswith("\n")
                ):
                    new_lines[-1] = new_lines[-1][:-1]
                prev_line_type = line.line_type
                continue

            if line.line_type in (LINE_TYPE_CONTEXT, LINE_TYPE_ADDED):
                new_lines.append(line.value)

            prev_line_type = line.line_type

        start_idx = hunk.source_start - 1
        end_idx = start_idx + hunk.source_length

        # New file: source_start=0, source_length=0
        if hunk.source_start == 0 and hunk.source_length == 0:
            start_idx = 0
            end_idx = 0

        result_lines[start_idx:end_idx] = new_lines

    return "".join(result_lines)


def reconstruct_version(snapshots_and_diffs: list[tuple[bool, str]]) -> str:
    """Replay a snapshot followed by forward diffs to get the target content.

    *snapshots_and_diffs* is an **ordered** list of ``(is_snapshot, content)``
    tuples.  The first entry **must** be a snapshot (``is_snapshot=True``);
    subsequent entries are forward diffs.
    """
    if not snapshots_and_diffs:
        return ""

    first_is_snap, content = snapshots_and_diffs[0]
    if not first_is_snap:
        msg = "First entry must be a snapshot"
        raise ValueError(msg)

    result = content
    for is_snap, diff_text in snapshots_and_diffs[1:]:
        result = diff_text if is_snap else apply_diff(result, diff_text)

    return result

"""GroverFile and FileVersion models with diff-based versioning."""

from __future__ import annotations

import difflib
import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT_INTERVAL: int = 20
"""Take a full snapshot every N versions (forward diffs between snapshots)."""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GroverFile(SQLModel, table=True):
    """A tracked file (or chunk) in the virtual filesystem."""

    __tablename__ = "grover_files"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(index=True)
    parent_path: str = Field(default="")
    name: str = Field(default="")
    mime_type: str = Field(default="text/plain")
    size_bytes: int = Field(default=0)
    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)
    current_version: int = Field(default=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted: bool = Field(default=False)


class FileVersion(SQLModel, table=True):
    """A version snapshot or forward diff for a GroverFile."""

    __tablename__ = "grover_file_versions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    file_id: str = Field(index=True)
    version: int = Field(default=1)
    is_snapshot: bool = Field(default=False)
    content: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


_DIFF_SEP = "\0"
"""Separator between diff lines in serialised form.

Content lines may lack trailing newlines (for files that don't end in ``\\n``),
so we cannot use ``\\n`` as a record separator.  Null bytes cannot appear in
valid text content, making ``\\0`` a safe sentinel.
"""


def compute_diff(old: str, new: str) -> str:
    """Compute a unified diff from *old* to *new*.

    Returns the diff as a null-separated string (empty string if no
    changes).  Content lines preserve their original endings exactly so
    that files without trailing newlines round-trip correctly.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    raw = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    if not raw:
        return ""
    return _DIFF_SEP.join(raw)


def apply_diff(base: str, diff: str) -> str:
    """Apply a unified diff to *base* and return the resulting text.

    This is a minimal unified-diff applier that handles the ``@@`` hunk
    headers produced by :func:`compute_diff`.
    """
    if not diff:
        return base

    base_lines = base.splitlines(keepends=True)
    diff_lines = diff.split(_DIFF_SEP)
    result_lines: list[str] = []
    base_idx = 0
    headers_seen = 0

    for line in diff_lines:
        # Skip the two file headers (--- and +++) which always come first.
        # We count them rather than pattern-matching to avoid collisions
        # with removal lines like "---some-yaml-delimiter".
        if headers_seen < 2:
            headers_seen += 1
            continue

        # Hunk header — jump to the right position in base
        if line.startswith("@@"):
            # Parse "@@ -start,count +start,count @@"
            parts = line.split()
            old_range = parts[1]  # e.g. "-1,5"
            old_start = int(old_range.split(",")[0].lstrip("-"))
            # Copy unchanged lines before this hunk
            target = old_start - 1  # 0-indexed
            while base_idx < target:
                result_lines.append(base_lines[base_idx])
                base_idx += 1
            continue

        if line.startswith("-"):
            # Line removed from base — skip it
            base_idx += 1
        elif line.startswith("+"):
            # Line added — preserves original ending (no \n = no newline)
            result_lines.append(line[1:])
        elif line.startswith(" "):
            # Context line — copy from base
            result_lines.append(base_lines[base_idx])
            base_idx += 1
        # else: skip (e.g. "\ No newline at end of file")

    # Append any remaining base lines after the last hunk
    while base_idx < len(base_lines):
        result_lines.append(base_lines[base_idx])
        base_idx += 1

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

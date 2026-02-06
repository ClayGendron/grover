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


def compute_diff(old: str, new: str) -> str:
    """Compute a unified diff from *old* to *new*.

    Returns the diff as a string (empty string if no changes).
    Each output line is newline-terminated so the result can be
    reliably re-split with :pymethod:`str.splitlines`.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    # n=3 is the default context; lineterm="" avoids an extra \n on lines
    # that already carry one from keepends=True — but header lines (---,
    # +++, @@) will *not* end in \n.  We normalise below.
    raw = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    if not raw:
        return ""
    # Ensure every line ends with exactly one \n so that
    # "".join(result).splitlines(keepends=True) round-trips cleanly.
    out: list[str] = []
    for line in raw:
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n")
    return "".join(out)


def apply_diff(base: str, diff: str) -> str:
    """Apply a unified diff to *base* and return the resulting text.

    This is a minimal unified-diff applier that handles the ``@@`` hunk
    headers produced by :func:`compute_diff`.
    """
    if not diff:
        return base

    base_lines = base.splitlines(keepends=True)
    diff_lines = diff.splitlines(keepends=True)
    result_lines: list[str] = []
    base_idx = 0

    for line in diff_lines:
        # Strip the trailing newline that *we* added for storage so we
        # can inspect the diff-control character cleanly.
        stripped = line.rstrip("\n")

        # Skip file headers
        if stripped.startswith("---") or stripped.startswith("+++"):
            continue

        # Hunk header — jump to the right position in base
        if stripped.startswith("@@"):
            # Parse "@@ -start,count +start,count @@"
            parts = stripped.split()
            old_range = parts[1]  # e.g. "-1,5"
            old_start = int(old_range.split(",")[0].lstrip("-"))
            # Copy unchanged lines before this hunk
            target = old_start - 1  # 0-indexed
            while base_idx < target:
                result_lines.append(base_lines[base_idx])
                base_idx += 1
            continue

        if stripped.startswith("-"):
            # Line removed from base — skip it
            base_idx += 1
        elif stripped.startswith("+"):
            # Line added — content is everything after the "+"
            result_lines.append(stripped[1:] + "\n")
        elif stripped.startswith(" "):
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

    is_snap, content = snapshots_and_diffs[0]
    if not is_snap:
        msg = "First entry must be a snapshot"
        raise ValueError(msg)

    result = content
    for is_snap, diff_text in snapshots_and_diffs[1:]:
        result = diff_text if is_snap else apply_diff(result, diff_text)

    return result

"""Version provider — pluggable versioning for GroverFileSystem.

The default implementation uses forward unified diffs with periodic full
snapshots (every ``SNAPSHOT_INTERVAL`` versions).  Subclass
``VersionProvider`` to plug in a different strategy.

The provider is purely text-based — it has no awareness of database models
or sessions.  It receives content strings and returns what to store.

Forward diffs transform version N-1's content into version N's content.
Reconstruction starts from the nearest snapshot at or below the target
and applies forward diffs to reach the target version.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from unidiff import PatchSet
from unidiff.constants import LINE_TYPE_ADDED, LINE_TYPE_CONTEXT, LINE_TYPE_NO_NEWLINE

SNAPSHOT_INTERVAL: int = 10
"""Take a full snapshot every N versions (forward diffs between snapshots)."""

_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"


# ---------------------------------------------------------------------------
# Version record — what the provider returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRecord:
    """What the version provider produces — content to store plus metadata."""

    content: str | None
    version_diff: str | None
    is_snapshot: bool


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


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
    out: list[str] = []
    for line in raw:
        out.append(line)
        if line and line[0] in ("+", "-", " ") and not line.endswith("\n"):
            out[-1] = line + "\n"
            out.append(_NO_NEWLINE_MARKER)
    return "".join(out)


def apply_diff(base: str, diff: str) -> str:
    """Apply a unified diff to *base* and return the resulting text."""
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

        if hunk.source_start == 0 and hunk.source_length == 0:
            start_idx = 0
            end_idx = 0
        elif start_idx < 0 or end_idx > len(result_lines):
            raise ValueError(
                f"Hunk out of bounds: source_start={hunk.source_start}, "
                f"source_length={hunk.source_length}, file has {len(result_lines)} lines"
            )

        result_lines[start_idx:end_idx] = new_lines

    return "".join(result_lines)


def reconstruct_version(versions_asc: list[tuple[bool, str]]) -> str:
    """Reconstruct a version by replaying forward diffs from a snapshot.

    *versions_asc* is ordered from **lowest** version number (must be a
    snapshot) to **highest** (the target).  Each entry is
    ``(is_snapshot, stored_content)``.

    Snapshots replace the running content; diffs are applied forward.
    """
    if not versions_asc:
        return ""

    first_is_snap, content = versions_asc[0]
    if not first_is_snap:
        msg = "First entry must be a snapshot"
        raise ValueError(msg)

    result = content
    for is_snap, stored in versions_asc[1:]:
        result = stored if is_snap else apply_diff(result, stored)

    return result


# ---------------------------------------------------------------------------
# Version creation — forward diffs with periodic snapshots
# ---------------------------------------------------------------------------


def create_version(
    prev_content: str | None,
    version_content: str,
    version_number: int,
    *,
    force_snapshot: bool = False,
) -> VersionRecord:
    """Return the content to store for this version.

    Forward diffs transform the previous version's content into this
    version's content.  Reconstruction starts from the nearest snapshot
    and applies diffs forward.

    *prev_content* is the previous version's content (None if this is
    the first version).  *version_content* is the full file content for
    this version.  Set *force_snapshot* to store a snapshot regardless of
    the normal interval rules.
    """
    is_snapshot = (
        force_snapshot
        or version_number == 1
        or version_number % SNAPSHOT_INTERVAL == 0
        or prev_content is None
    )

    stored_snapshot = (
        version_content
        if is_snapshot
        else None
    )
    stored_diff = None if is_snapshot else compute_diff(prev_content or "", version_content)

    return VersionRecord(
        content=stored_snapshot,
        version_diff=stored_diff,
        is_snapshot=is_snapshot,
    )

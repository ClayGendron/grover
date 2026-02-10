"""Tests for the diff engine — compute_diff, apply_diff, reconstruct_version."""

from __future__ import annotations

import pytest

from grover.fs.diff import apply_diff, compute_diff, reconstruct_version


class TestComputeDiff:
    def test_identical_content(self):
        """Empty diff for same content."""
        content = "line1\nline2\nline3\n"
        diff = compute_diff(content, content)
        assert diff == ""

    def test_simple_add_remove(self):
        """Diff contains added and removed lines."""
        old = "aaa\nbbb\nccc\n"
        new = "aaa\nXXX\nccc\n"
        diff = compute_diff(old, new)
        assert "-bbb" in diff
        assert "+XXX" in diff


class TestApplyDiff:
    def test_round_trip(self):
        """compute then apply returns the new content."""
        old = "hello\nworld\n"
        new = "hello\nearth\n"
        diff = compute_diff(old, new)
        result = apply_diff(old, diff)
        assert result == new

    def test_empty_diff(self):
        """Empty diff returns base unchanged."""
        base = "unchanged\n"
        assert apply_diff(base, "") == base

    def test_no_trailing_newline(self):
        """Files without a trailing newline survive round-trip."""
        old = "line1\nline2"
        new = "line1\nline2\nline3"
        diff = compute_diff(old, new)
        result = apply_diff(old, diff)
        assert result == new

    def test_add_content_to_empty(self):
        """Adding content to an empty string."""
        old = ""
        new = "first line\nsecond line\n"
        diff = compute_diff(old, new)
        result = apply_diff(old, diff)
        assert result == new

    def test_multiline_changes(self):
        """Multiple lines changed across the file."""
        old = "a\nb\nc\nd\ne\n"
        new = "a\nB\nc\nD\ne\n"
        diff = compute_diff(old, new)
        result = apply_diff(old, diff)
        assert result == new


class TestReconstructVersion:
    def test_single_snapshot(self):
        """Just a snapshot returns its content."""
        content = "snapshot content\n"
        result = reconstruct_version([(True, content)])
        assert result == content

    def test_snapshot_with_diffs(self):
        """Snapshot + 3 forward diffs reconstructs correctly."""
        v1 = "line1\nline2\nline3\n"
        v2 = "line1\nchanged2\nline3\n"
        v3 = "line1\nchanged2\nchanged3\n"
        v4 = "line1\nchanged2\nchanged3\nline4\n"

        diff_1_2 = compute_diff(v1, v2)
        diff_2_3 = compute_diff(v2, v3)
        diff_3_4 = compute_diff(v3, v4)

        entries = [
            (True, v1),
            (False, diff_1_2),
            (False, diff_2_3),
            (False, diff_3_4),
        ]
        result = reconstruct_version(entries)
        assert result == v4

    def test_empty_list_returns_empty(self):
        """Empty list returns empty string."""
        assert reconstruct_version([]) == ""

    def test_no_snapshot_first_raises(self):
        """First entry not snapshot raises ValueError."""
        with pytest.raises(ValueError, match="snapshot"):
            reconstruct_version([(False, "diff data")])

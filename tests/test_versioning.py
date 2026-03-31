"""Unit tests for the versioning module — diffs, snapshots, reconstruction."""

from __future__ import annotations

import pytest

from grover.versioning import (
    SNAPSHOT_INTERVAL,
    VersionRecord,
    apply_diff,
    compute_diff,
    create_version,
    reconstruct_version,
)

# ------------------------------------------------------------------
# compute_diff
# ------------------------------------------------------------------


class TestComputeDiff:
    def test_identical_content_returns_empty(self):
        assert compute_diff("hello\n", "hello\n") == ""

    def test_simple_change(self):
        diff = compute_diff("line1\n", "line2\n")
        assert diff != ""
        assert "-line1" in diff
        assert "+line2" in diff

    def test_add_lines(self):
        diff = compute_diff("a\n", "a\nb\n")
        assert "+b" in diff

    def test_remove_lines(self):
        diff = compute_diff("a\nb\n", "a\n")
        assert "-b" in diff

    def test_multiline_change(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = compute_diff(old, new)
        assert "-line2" in diff
        assert "+changed" in diff

    def test_empty_to_content(self):
        diff = compute_diff("", "hello\n")
        assert "+hello" in diff

    def test_content_to_empty(self):
        diff = compute_diff("hello\n", "")
        assert "-hello" in diff

    def test_no_trailing_newline(self):
        """Files without trailing newlines get the 'no newline' marker."""
        diff = compute_diff("a", "b")
        assert diff != ""
        assert "No newline at end of file" in diff


# ------------------------------------------------------------------
# apply_diff
# ------------------------------------------------------------------


class TestApplyDiff:
    def test_empty_diff_returns_base(self):
        assert apply_diff("hello\n", "") == "hello\n"

    def test_simple_patch(self):
        old = "line1\nline2\n"
        new = "line1\nchanged\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_add_lines(self):
        old = "a\n"
        new = "a\nb\nc\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_remove_lines(self):
        old = "a\nb\nc\n"
        new = "a\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_insert_into_empty_file(self):
        old = ""
        new = "hello\nworld\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_empty_file_from_content(self):
        old = "hello\nworld\n"
        new = ""
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_no_trailing_newline_roundtrip(self):
        old = "line1\nline2"
        new = "line1\nchanged"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_add_trailing_newline(self):
        old = "hello"
        new = "hello\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_remove_trailing_newline(self):
        old = "hello\n"
        new = "hello"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_multiline_complex_edit(self):
        old = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        new = "def foo():\n    return 42\n\ndef bar():\n    return 2\n\ndef baz():\n    return 3\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_parseable_but_empty_patch_returns_base(self):
        """A diff with headers but no hunks returns the base unchanged."""
        # unidiff parses this as a PatchSet with zero files
        assert apply_diff("hello\n", "not a real diff\n") == "hello\n"

    def test_hunk_out_of_bounds_raises(self):
        # Craft a diff that references lines beyond the base content
        bad_diff = (
            "--- a\n"
            "+++ b\n"
            "@@ -50,1 +50,1 @@\n"
            "-old\n"
            "+new\n"
        )
        with pytest.raises(ValueError, match="out of bounds"):
            apply_diff("short\n", bad_diff)


# ------------------------------------------------------------------
# reconstruct_version
# ------------------------------------------------------------------


class TestReconstructVersion:
    def test_empty_list_returns_empty(self):
        assert reconstruct_version([]) == ""

    def test_single_snapshot(self):
        assert reconstruct_version([(True, "hello\n")]) == "hello\n"

    def test_snapshot_then_diff(self):
        v1 = "line1\n"
        v2 = "line1\nline2\n"
        diff = compute_diff(v1, v2)
        result = reconstruct_version([(True, v1), (False, diff)])
        assert result == v2

    def test_chain_of_diffs(self):
        contents = [f"v{i}\n" for i in range(1, 5)]
        chain: list[tuple[bool, str]] = [(True, contents[0])]
        for i in range(1, len(contents)):
            diff = compute_diff(contents[i - 1], contents[i])
            chain.append((False, diff))
        result = reconstruct_version(chain)
        assert result == contents[-1]

    def test_mid_chain_snapshot(self):
        """A snapshot in the middle resets the base."""
        v1 = "original\n"
        v2 = "modified\n"
        v3 = "snapshot content\n"  # snapshot
        v4 = "final\n"

        chain = [
            (True, v1),
            (False, compute_diff(v1, v2)),
            (True, v3),  # snapshot replaces running content
            (False, compute_diff(v3, v4)),
        ]
        assert reconstruct_version(chain) == v4

    def test_first_entry_not_snapshot_raises(self):
        with pytest.raises(ValueError, match="First entry must be a snapshot"):
            reconstruct_version([(False, "some diff")])

    def test_large_chain(self):
        """Reconstruct through many diffs to verify cumulative correctness."""
        base = "line 0\n"
        chain: list[tuple[bool, str]] = [(True, base)]
        prev = base
        for i in range(1, 25):
            new = f"line {i}\n"
            if i % SNAPSHOT_INTERVAL == 0:
                chain.append((True, new))
            else:
                chain.append((False, compute_diff(prev, new)))
            prev = new
        assert reconstruct_version(chain) == "line 24\n"


# ------------------------------------------------------------------
# create_version
# ------------------------------------------------------------------


class TestCreateVersion:
    def test_first_version_is_snapshot(self):
        r = create_version(None, "hello\n", 1)
        assert isinstance(r, VersionRecord)
        assert r.is_snapshot is True
        assert r.content == "hello\n"
        assert r.version_diff is None

    def test_second_version_is_diff(self):
        r = create_version("hello\n", "world\n", 2)
        assert r.is_snapshot is False
        assert r.content is None
        assert r.version_diff is not None
        assert "-hello" in r.version_diff
        assert "+world" in r.version_diff

    def test_snapshot_interval(self):
        r = create_version("old\n", "new\n", SNAPSHOT_INTERVAL)
        assert r.is_snapshot is True
        assert r.content == "new\n"
        assert r.version_diff is None

    def test_non_snapshot_interval(self):
        r = create_version("old\n", "new\n", SNAPSHOT_INTERVAL + 1)
        assert r.is_snapshot is False

    def test_force_snapshot(self):
        r = create_version("old\n", "new\n", 5, force_snapshot=True)
        assert r.is_snapshot is True
        assert r.content == "new\n"

    def test_none_prev_content_forces_snapshot(self):
        r = create_version(None, "content\n", 7)
        assert r.is_snapshot is True

    def test_diff_roundtrip(self):
        """A diff version can reconstruct the content via apply_diff."""
        prev = "old content\nline two\n"
        new = "new content\nline two\nline three\n"
        r = create_version(prev, new, 3)
        assert r.is_snapshot is False
        assert apply_diff(prev, r.version_diff) == new

    def test_identical_content_produces_empty_diff(self):
        r = create_version("same\n", "same\n", 3)
        assert r.is_snapshot is False
        assert r.version_diff == ""

    def test_create_then_reconstruct(self):
        """Full cycle: create versions, then reconstruct any target."""
        contents = [
            "def main():\n    pass\n",
            "def main():\n    print('hello')\n",
            "def main():\n    print('hello')\n    return 0\n",
        ]
        records: list[VersionRecord] = []
        for i, content in enumerate(contents):
            prev = contents[i - 1] if i > 0 else None
            records.append(create_version(prev, content, i + 1))

        # Reconstruct version 3 from full chain
        chain = [
            (r.is_snapshot, r.content if r.is_snapshot else r.version_diff)
            for r in records
        ]
        assert reconstruct_version(chain) == contents[-1]

        # Reconstruct version 2 from first two
        chain2 = [
            (r.is_snapshot, r.content if r.is_snapshot else r.version_diff)
            for r in records[:2]
        ]
        assert reconstruct_version(chain2) == contents[1]

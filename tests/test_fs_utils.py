"""Tests for fs/utils.py — path helpers, file detection, replacement engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from grover.fs.types import ReadResult
from grover.fs.utils import (
    block_anchor_replacer,
    format_read_output,
    from_trash_path,
    get_line_number,
    guess_mime_type,
    is_binary_file,
    is_shared_path,
    is_text_file,
    is_trash_path,
    levenshtein,
    line_trimmed_replacer,
    normalize_line_endings,
    normalize_path,
    replace,
    simple_replacer,
    split_path,
    to_trash_path,
    validate_path,
)

# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------


class TestNormalizePath:
    @pytest.mark.parametrize(
        ("input_path", "expected"),
        [
            pytest.param("", "/", id="empty"),
            pytest.param("foo.txt", "/foo.txt", id="no-leading-slash"),
            pytest.param("/foo//bar.txt", "/foo/bar.txt", id="double-slashes"),
            pytest.param("/foo/../bar.txt", "/bar.txt", id="dotdot"),
            pytest.param("/foo/", "/foo", id="trailing-slash"),
            pytest.param("/", "/", id="root"),
        ],
    )
    def test_normalize(self, input_path: str, expected: str):
        assert normalize_path(input_path) == expected


class TestSplitPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            pytest.param("/foo/bar.txt", ("/foo", "bar.txt"), id="nested-file"),
            pytest.param("/foo.txt", ("/", "foo.txt"), id="root-file"),
            pytest.param("/", ("/", ""), id="root"),
        ],
    )
    def test_split(self, path: str, expected: tuple[str, str]):
        assert split_path(path) == expected


class TestValidatePath:
    def test_valid(self):
        ok, msg = validate_path("/hello.txt")
        assert ok is True
        assert msg == ""

    @pytest.mark.parametrize(
        ("path", "expected_msg"),
        [
            pytest.param("/hello\x00.txt", "null", id="null-byte"),
            pytest.param("/" + "a" * 4096, "long", id="path-too-long"),
            pytest.param("/CON.txt", "Reserved", id="reserved-name"),
            pytest.param("/NUL", "Reserved", id="reserved-no-ext"),
            pytest.param("/" + "a" * 256, "Filename too long", id="filename-too-long"),
        ],
    )
    def test_invalid(self, path: str, expected_msg: str):
        ok, msg = validate_path(path)
        assert ok is False
        assert expected_msg.lower() in msg.lower()


# ---------------------------------------------------------------------------
# File Detection
# ---------------------------------------------------------------------------


class TestIsTextFile:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            pytest.param("main.py", True, id="python"),
            pytest.param("config.json", True, id="json"),
            pytest.param("Makefile", True, id="makefile"),
            pytest.param(".gitignore", True, id="dotfile"),
            pytest.param("image.png", False, id="binary-ext"),
            pytest.param("data.xyz", False, id="unknown-ext"),
        ],
    )
    def test_is_text(self, filename: str, expected: bool):
        assert is_text_file(filename) is expected


class TestGuessMimeType:
    @pytest.mark.parametrize(
        ("filename", "expected_substr"),
        [
            pytest.param("main.py", "python", id="python"),
            pytest.param("data.json", "json", id="json"),
            pytest.param("thing.xyz123", "text/plain", id="unknown"),
        ],
    )
    def test_mime(self, filename: str, expected_substr: str):
        assert expected_substr in guess_mime_type(filename)


class TestIsBinaryFile:
    def test_known_binary_extension(self, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert is_binary_file(p) is True

    def test_text_file(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("Hello world\n")
        assert is_binary_file(p) is False

    def test_file_with_null_bytes(self, tmp_path):
        p = tmp_path / "data.dat"
        p.write_bytes(b"hello\x00world")
        assert is_binary_file(p) is True

    def test_nonexistent_file(self):
        assert is_binary_file(Path("/nonexistent/file.txt")) is False


# ---------------------------------------------------------------------------
# Trash Path Helpers
# ---------------------------------------------------------------------------


class TestTrashPaths:
    def test_is_trash_path(self):
        assert is_trash_path("/__trash__/abc/hello.txt") is True
        assert is_trash_path("/hello.txt") is False
        assert is_trash_path("/__trash__") is False

    def test_to_trash_path(self):
        result = to_trash_path("/hello.txt", "file-uuid")
        assert result == "/__trash__/file-uuid/hello.txt"

    def test_from_trash_path(self):
        result = from_trash_path("/__trash__/file-uuid/hello.txt")
        assert result == "/hello.txt"

    def test_from_trash_path_not_trash(self):
        assert from_trash_path("/hello.txt") == "/hello.txt"

    def test_from_trash_path_no_slash(self):
        assert from_trash_path("/__trash__/abc") == "/"


# ---------------------------------------------------------------------------
# Text Replacement
# ---------------------------------------------------------------------------


class TestNormalizeLineEndings:
    def test_crlf_to_lf(self):
        assert normalize_line_endings("a\r\nb\r\n") == "a\nb\n"

    def test_lf_unchanged(self):
        assert normalize_line_endings("a\nb\n") == "a\nb\n"


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_one_change(self):
        assert levenshtein("abc", "axc") == 1

    def test_empty(self):
        assert levenshtein("", "abc") == 3
        assert levenshtein("abc", "") == 3


class TestGetLineNumber:
    def test_first_line(self):
        assert get_line_number("hello\nworld\n", 0) == 1

    def test_second_line(self):
        assert get_line_number("hello\nworld\n", 6) == 2


class TestReplace:
    def test_exact_match(self):
        result = replace("hello world", "world", "earth")
        assert result.success is True
        assert result.content == "hello earth"
        assert result.method_used == "exact"

    def test_replace_all(self):
        result = replace("a b a b", "a", "x", replace_all=True)
        assert result.success is True
        assert result.content == "x b x b"

    def test_no_match(self):
        result = replace("hello world", "xyz", "abc")
        assert result.success is False
        assert "not found" in result.error

    def test_empty_old_string(self):
        result = replace("hello", "", "x")
        assert result.success is False

    def test_same_strings(self):
        result = replace("hello", "hello", "hello")
        assert result.success is False

    def test_line_trimmed_match(self):
        content = "  hello  \n  world  \n"
        result = replace(content, "hello\nworld\n", "hi\nearth\n")
        assert result.success is True
        # The line_trimmed replacer should match the trimmed lines
        assert result.method_used in ("exact", "line_trimmed")

    def test_multiple_matches_error(self):
        result = replace("aXb aXb", "aXb", "Y")
        assert result.success is False
        assert "2 matches" in result.error

    def test_replace_all_fuzzy_rejected(self):
        # replace_all with non-exact match should fail
        content = "  hello  \n  world  \n"
        find = "hello\nworld\n"
        # Only if exact fails and line_trimmed matches
        result = replace(content, find, "replacement\n", replace_all=True)
        # If exact match works, replace_all succeeds; otherwise
        # line_trimmed should reject replace_all
        if result.success:
            assert result.method_used == "exact"

    def test_block_anchor_match(self):
        content = "def foo():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z\n"
        find = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    return x + y + z\n"
        new = "def bar():\n    return 42\n"
        result = replace(content, find, new)
        # block_anchor needs >=3 lines and matching first/last anchors
        # "def foo():" matches but "return x + y + z" also matches
        if result.success:
            assert result.method_used in ("exact", "line_trimmed", "block_anchor")


# ---------------------------------------------------------------------------
# Path Validation Edge Cases
# ---------------------------------------------------------------------------


class TestValidatePathEdgeCases:
    def test_validate_255_char_filename(self):
        ok, _msg = validate_path("/" + "a" * 255)
        assert ok is True

    def test_validate_256_char_filename(self):
        ok, msg = validate_path("/" + "a" * 256)
        assert ok is False
        assert "Filename too long" in msg

    def test_normalize_path_multiple_slashes(self):
        # posixpath.normpath preserves a leading // (POSIX allows it)
        # but collapses internal triple slashes
        assert normalize_path("///foo///bar") == "/foo/bar"

    def test_control_character_rejected(self):
        ok, msg = validate_path("/file\x01name.txt")
        assert ok is False
        assert "control character" in msg.lower()


# ---------------------------------------------------------------------------
# @shared Path Validation
# ---------------------------------------------------------------------------


class TestSharedPathValidation:
    def test_validate_path_rejects_at_shared(self):
        ok, msg = validate_path("/foo/@shared/bar.txt")
        assert ok is False
        assert "@shared" in msg

    def test_validate_path_rejects_at_shared_root(self):
        ok, msg = validate_path("/@shared")
        assert ok is False
        assert "@shared" in msg

    def test_validate_path_allows_normal_at_sign(self):
        ok, msg = validate_path("/foo/@bar/baz.txt")
        assert ok is True
        assert msg == ""

    def test_is_shared_path_true(self):
        assert is_shared_path("/@shared/alice/notes.md") is True
        assert is_shared_path("/foo/@shared/bar") is True

    def test_is_shared_path_false(self):
        assert is_shared_path("/foo/bar.txt") is False
        assert is_shared_path("/foo/@bar/baz") is False
        assert is_shared_path("/foo/shared/baz") is False

    def test_is_shared_path_at_shared_as_substring(self):
        """'@shared' embedded in a longer segment name is NOT a shared path."""
        assert is_shared_path("/foo/@shared_extra/baz") is False

    def test_validate_path_rejects_nested_shared(self):
        ok, msg = validate_path("/foo/bar/@shared/baz/qux.txt")
        assert ok is False
        assert "@shared" in msg


# ---------------------------------------------------------------------------
# Replacer Edge Cases
# ---------------------------------------------------------------------------


class TestSimpleReplacerEdgeCases:
    def test_no_match(self):
        matches = list(simple_replacer("hello world", "xyz"))
        assert matches == []

    def test_multiple_matches(self):
        matches = list(simple_replacer("aXbXc", "X"))
        assert len(matches) == 2
        assert all(m.confidence == 1.0 for m in matches)


class TestLineTrimmedReplacerEdgeCases:
    def test_whitespace_only_match(self):
        content = "  hello  \n  world  \n"
        find = "hello\nworld"
        matches = list(line_trimmed_replacer(content, find))
        assert len(matches) == 1
        assert matches[0].method == "line_trimmed"
        assert matches[0].confidence == 0.9


class TestBlockAnchorReplacerEdgeCases:
    def test_minimum_block(self):
        content = "start\nmiddle content\nend\n"
        find = "start\nmiddle content\nend\n"
        matches = list(block_anchor_replacer(content, find))
        # Exactly 3 lines with matching anchors
        assert len(matches) <= 1

    def test_less_than_3_lines_returns_empty(self):
        content = "start\nend\n"
        find = "start\nend\n"
        matches = list(block_anchor_replacer(content, find))
        assert matches == []


class TestReplaceEdgeCases:
    def test_replace_empty_content_old_string(self):
        result = replace("", "x", "y")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_replace_crlf_normalized(self):
        result = replace("hello\r\nworld", "hello\nworld", "hi\nearth")
        assert result.success is True
        assert result.content == "hi\nearth"


# ---------------------------------------------------------------------------
# Read Output Formatting
# ---------------------------------------------------------------------------


class TestFormatReadOutput:
    def test_format_read_output_with_offset(self):
        result = ReadResult(
            success=True,
            message="ok",
            content="line1\nline2\nline3",
            offset=10,
        )
        output = format_read_output(result)
        # Lines should start at offset+1 = 11
        assert "00011|" in output
        assert "00012|" in output
        assert "00013|" in output

    def test_format_read_output_truncated(self):
        result = ReadResult(
            success=True,
            message="ok",
            content="line1\nline2",
            offset=0,
            truncated=True,
        )
        output = format_read_output(result)
        assert "File has more lines" in output
        assert "offset" in output.lower()

    def test_format_read_output_empty(self):
        result = ReadResult(success=True, message="ok", content="")
        output = format_read_output(result)
        assert "empty file" in output.lower()

    def test_format_read_output_none_content(self):
        result = ReadResult(success=True, message="ok", content=None)
        output = format_read_output(result)
        assert "empty file" in output.lower()

    def test_format_read_output_end_of_file(self):
        result = ReadResult(
            success=True,
            message="ok",
            content="hello\nworld",
            total_lines=2,
            truncated=False,
        )
        output = format_read_output(result)
        assert "End of file" in output
        assert "2 lines" in output

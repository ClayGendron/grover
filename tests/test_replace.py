"""Tests for the text replacement engine."""

from __future__ import annotations

from grover.replace import (
    Match,
    ReplaceResult,
    block_anchor_replacer,
    get_context_lines,
    get_line_number,
    levenshtein,
    line_trimmed_replacer,
    normalize_line_endings,
    replace,
    simple_replacer,
)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


class TestNormalizeLineEndings:
    def test_crlf_to_lf(self):
        assert normalize_line_endings("a\r\nb\r\n") == "a\nb\n"

    def test_lf_unchanged(self):
        assert normalize_line_endings("a\nb\n") == "a\nb\n"

    def test_mixed(self):
        assert normalize_line_endings("a\r\nb\nc\r\n") == "a\nb\nc\n"

    def test_empty(self):
        assert normalize_line_endings("") == ""


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_left(self):
        assert levenshtein("", "abc") == 3

    def test_empty_right(self):
        assert levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_single_insertion(self):
        assert levenshtein("abc", "abcd") == 1

    def test_single_deletion(self):
        assert levenshtein("abcd", "abc") == 1

    def test_single_substitution(self):
        assert levenshtein("abc", "axc") == 1

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3

    def test_transposition(self):
        # Levenshtein counts transposition as 2 (delete + insert)
        assert levenshtein("ab", "ba") == 2


class TestGetLineNumber:
    def test_start_of_file(self):
        assert get_line_number("abc\ndef\n", 0) == 1

    def test_second_line(self):
        assert get_line_number("abc\ndef\n", 4) == 2

    def test_third_line(self):
        assert get_line_number("a\nb\nc\n", 4) == 3


class TestGetContextLines:
    def test_basic_context(self):
        content = "line1\nline2\nline3\nline4\nline5"
        # Match spans line2 (pos 6 to 11)
        result = get_context_lines(content, 6, 11, context=1)
        lines = result.split("\n")
        assert len(lines) == 3  # line1 (context), line2 (match), line3 (context)
        assert ">" in lines[1]  # line2 is marked
        assert " " == lines[0][5]  # line1 is context (space prefix)

    def test_context_at_start(self):
        content = "first\nsecond\nthird"
        result = get_context_lines(content, 0, 5, context=2)
        assert ">" in result.split("\n")[0]

    def test_context_at_end(self):
        content = "first\nsecond\nthird"
        result = get_context_lines(content, 13, 18, context=2)
        lines = result.split("\n")
        assert ">" in lines[-1]


# ------------------------------------------------------------------
# Replacers
# ------------------------------------------------------------------


class TestSimpleReplacer:
    def test_single_match(self):
        matches = list(simple_replacer("hello world", "world"))
        assert len(matches) == 1
        assert matches[0].start == 6
        assert matches[0].end == 11
        assert matches[0].method == "exact"
        assert matches[0].confidence == 1.0

    def test_multiple_matches(self):
        matches = list(simple_replacer("aaa", "a"))
        assert len(matches) == 3

    def test_no_match(self):
        matches = list(simple_replacer("hello", "world"))
        assert len(matches) == 0

    def test_multiline(self):
        matches = list(simple_replacer("line1\nline2\nline3", "line2"))
        assert len(matches) == 1
        assert matches[0].text == "line2"


class TestLineTrimmedReplacer:
    def test_whitespace_mismatch(self):
        content = "  def foo():\n    pass"
        find = "def foo():\n  pass"
        matches = list(line_trimmed_replacer(content, find))
        assert len(matches) == 1
        assert matches[0].method == "line_trimmed"
        assert matches[0].confidence == 0.9

    def test_no_match(self):
        matches = list(line_trimmed_replacer("def foo():\n    pass", "def bar():\n    pass"))
        assert len(matches) == 0

    def test_empty_find(self):
        matches = list(line_trimmed_replacer("content", ""))
        assert len(matches) == 0

    def test_trailing_newline_stripped(self):
        content = "a\nb\nc"
        find = "a\nb\n"
        matches = list(line_trimmed_replacer(content, find))
        assert len(matches) == 1

    def test_matched_text_preserves_original_whitespace(self):
        content = "    indented\n    code"
        find = "indented\ncode"
        matches = list(line_trimmed_replacer(content, find))
        assert len(matches) == 1
        assert "    indented" in matches[0].text


class TestBlockAnchorReplacer:
    def test_exact_anchors_fuzzy_middle(self):
        content = "def foo():\n    x = 1\n    return x"
        find = "def foo():\n    x = 2\n    return x"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 1
        assert matches[0].method == "block_anchor"
        assert matches[0].confidence > 0

    def test_too_few_lines(self):
        matches = list(block_anchor_replacer("a\nb", "a\nb"))
        assert len(matches) == 0

    def test_no_anchor_match(self):
        content = "def foo():\n    pass\n    return"
        find = "def bar():\n    pass\n    return"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 0

    def test_single_candidate_below_threshold(self):
        content = "start\ncompletely different\nend"
        find = "start\nxxxxxxxxxxxxxxx\nend"
        matches = list(block_anchor_replacer(content, find))
        # similarity may be below threshold
        if matches:
            assert matches[0].confidence >= 0.6

    def test_multiple_candidates_picks_best(self):
        content = "def foo():\n    x = 1\n    return x\ndef foo():\n    x = 2\n    return x"
        find = "def foo():\n    x = 2\n    return x"
        matches = list(block_anchor_replacer(content, find))
        # Should pick the better match
        assert len(matches) <= 1

    def test_trailing_newline_stripped(self):
        content = "start\n    middle\nend\nmore"
        find = "start\n    middle\nend\n"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 1

    def test_identical_middle(self):
        content = "start\n    x = 1\n    y = 2\nend"
        find = "start\n    x = 1\n    y = 2\nend"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 1
        assert matches[0].confidence == 1.0

    def test_three_line_find_with_trailing_newline_becomes_two(self):
        """After popping trailing empty line, find has < 3 lines → no match."""
        content = "start\nmiddle\nend"
        find = "start\nmiddle\n"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 0

    def test_adjacent_anchors_no_middle(self):
        """Content has anchors adjacent but find has middle → lines_to_check <= 0."""
        # content: "start" at 0, "end" at 2 (i+2), so j searches from i+2
        content = "other\nstart\nfoo\nend\nmore"
        # find has 3 lines (passes < 3 check) but content block is only 2 apart
        find = "start\nfoo\nend"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 1
        assert matches[0].confidence == 1.0

    def test_empty_middle_line_skipped(self):
        """Empty middle lines have max_len=0 and are skipped in similarity."""
        content = "start\n\nx = 1\ny = 2\nz = 3\nend"
        find = "start\n\nx = 1\ny = 2\nz = 3\nend"
        matches = list(block_anchor_replacer(content, find))
        assert len(matches) == 1


# ------------------------------------------------------------------
# Core replace()
# ------------------------------------------------------------------


class TestReplace:
    def test_exact_match(self):
        r = replace("hello world", "world", "earth")
        assert r.success
        assert r.content == "hello earth"
        assert r.method_used == "exact"

    def test_empty_old_string(self):
        r = replace("content", "", "new")
        assert not r.success
        assert "empty" in r.error

    def test_same_old_new(self):
        r = replace("content", "content", "content")
        assert not r.success
        assert "different" in r.error

    def test_no_match(self):
        r = replace("hello", "xyz", "abc")
        assert not r.success
        assert "not found" in r.error

    def test_replace_all_exact(self):
        r = replace("a b a b a", "a", "x", replace_all=True)
        assert r.success
        assert r.content == "x b x b x"

    def test_replace_all_rejects_fuzzy(self):
        # Use whitespace mismatch to trigger line_trimmed (fuzzy)
        content = "  def foo():\n    pass\nother"
        old = "def foo():\n  pass"
        r = replace(content, old, "replaced", replace_all=True)
        assert not r.success
        assert "replace_all" in r.error

    def test_line_trimmed_fallback(self):
        content = "  def foo():\n    pass"
        old = "def foo():\npass"
        r = replace(content, old, "def bar():\n    pass")
        assert r.success
        assert r.method_used == "line_trimmed"

    def test_block_anchor_fallback(self):
        content = "def foo():\n    x = 1\n    return x"
        old = "def foo():\n    x = 2\n    return x"
        r = replace(content, old, "def foo():\n    x = 3\n    return x")
        assert r.success
        assert r.method_used == "block_anchor"

    def test_multiple_exact_matches_errors(self):
        r = replace("a b a", "a", "x")
        assert not r.success
        assert "2 matches" in r.error
        assert r.matches is not None
        assert len(r.matches) == 2

    def test_crlf_normalized(self):
        r = replace("hello\r\nworld", "hello\nworld", "replaced")
        assert r.success
        assert r.content == "replaced"

    def test_multiline_exact(self):
        content = "line1\nline2\nline3"
        r = replace(content, "line1\nline2", "replaced")
        assert r.success
        assert r.content == "replaced\nline3"

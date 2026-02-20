"""Tests for Phase 4: New query response types (pure type definitions)."""

from __future__ import annotations

import dataclasses

import pytest

from grover.fs.query_types import (
    ChunkMatch,
    GlobHit,
    GlobQueryResult,
    GrepHit,
    GrepQueryResult,
    LineMatch,
    SearchHit,
    SearchQueryResult,
)

# ==================================================================
# Frozen immutability
# ==================================================================


class TestFrozenImmutability:
    """All 8 types must be frozen dataclasses."""

    @pytest.mark.parametrize(
        "cls",
        [
            LineMatch,
            ChunkMatch,
            GlobHit,
            GrepHit,
            SearchHit,
            GlobQueryResult,
            GrepQueryResult,
            SearchQueryResult,
        ],
    )
    def test_is_frozen_dataclass(self, cls):
        assert dataclasses.is_dataclass(cls)
        # Frozen dataclasses have __dataclass_params__.frozen == True
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_line_match_immutable(self):
        lm = LineMatch(line_number=10, line_content="x = 1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            lm.line_number = 20  # type: ignore[misc]

    def test_chunk_match_immutable(self):
        cm = ChunkMatch(name="foo", line_start=1, line_end=5, score=0.9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cm.score = 0.5  # type: ignore[misc]

    def test_glob_hit_immutable(self):
        gh = GlobHit(path="/a.py")
        with pytest.raises(dataclasses.FrozenInstanceError):
            gh.path = "/b.py"  # type: ignore[misc]

    def test_grep_hit_immutable(self):
        gh = GrepHit(path="/a.py")
        with pytest.raises(dataclasses.FrozenInstanceError):
            gh.path = "/b.py"  # type: ignore[misc]

    def test_search_hit_immutable(self):
        sh = SearchHit(path="/a.py")
        with pytest.raises(dataclasses.FrozenInstanceError):
            sh.path = "/b.py"  # type: ignore[misc]

    def test_glob_query_result_immutable(self):
        r = GlobQueryResult(success=True, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.success = False  # type: ignore[misc]

    def test_grep_query_result_immutable(self):
        r = GrepQueryResult(success=True, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.success = False  # type: ignore[misc]

    def test_search_query_result_immutable(self):
        r = SearchQueryResult(success=True, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.success = False  # type: ignore[misc]


# ==================================================================
# Default values
# ==================================================================


class TestDefaults:
    """All types should have sensible defaults for optional fields."""

    def test_line_match_defaults(self):
        lm = LineMatch(line_number=1, line_content="hello")
        assert lm.context_before == ()
        assert lm.context_after == ()

    def test_chunk_match_defaults(self):
        cm = ChunkMatch(name="foo", line_start=1, line_end=5, score=0.9)
        assert cm.snippet == ""

    def test_glob_hit_defaults(self):
        gh = GlobHit(path="/a.py")
        assert gh.is_directory is False
        assert gh.size_bytes is None
        assert gh.mime_type is None

    def test_grep_hit_defaults(self):
        gh = GrepHit(path="/a.py")
        assert gh.line_matches == ()

    def test_search_hit_defaults(self):
        sh = SearchHit(path="/a.py")
        assert sh.score == 0.0
        assert sh.chunk_matches == ()

    def test_glob_query_result_defaults(self):
        r = GlobQueryResult(success=True, message="ok")
        assert r.hits == ()
        assert r.pattern == ""
        assert r.path == "/"

    def test_grep_query_result_defaults(self):
        r = GrepQueryResult(success=True, message="ok")
        assert r.hits == ()
        assert r.pattern == ""
        assert r.path == "/"
        assert r.files_searched == 0
        assert r.files_matched == 0
        assert r.truncated is False

    def test_search_query_result_defaults(self):
        r = SearchQueryResult(success=True, message="ok")
        assert r.hits == ()
        assert r.query == ""
        assert r.path == "/"
        assert r.files_matched == 0
        assert r.truncated is False


# ==================================================================
# Evidence types construction
# ==================================================================


class TestEvidenceTypes:
    def test_line_match_with_context(self):
        lm = LineMatch(
            line_number=10,
            line_content="def foo():",
            context_before=("", "# comment"),
            context_after=("    pass",),
        )
        assert lm.line_number == 10
        assert lm.line_content == "def foo():"
        assert len(lm.context_before) == 2
        assert len(lm.context_after) == 1

    def test_chunk_match_with_snippet(self):
        cm = ChunkMatch(
            name="MyClass.my_method",
            line_start=15,
            line_end=30,
            score=0.95,
            snippet="def my_method(self):\n    return 42",
        )
        assert cm.name == "MyClass.my_method"
        assert cm.line_start == 15
        assert cm.line_end == 30
        assert cm.score == 0.95
        assert "my_method" in cm.snippet


# ==================================================================
# Hit types construction
# ==================================================================


class TestHitTypes:
    def test_glob_hit_file(self):
        gh = GlobHit(path="/src/main.py", size_bytes=1234, mime_type="text/x-python")
        assert gh.path == "/src/main.py"
        assert gh.is_directory is False
        assert gh.size_bytes == 1234
        assert gh.mime_type == "text/x-python"

    def test_glob_hit_directory(self):
        gh = GlobHit(path="/src/", is_directory=True)
        assert gh.is_directory is True
        assert gh.size_bytes is None

    def test_grep_hit_with_matches(self):
        matches = (
            LineMatch(line_number=5, line_content="def alpha():"),
            LineMatch(line_number=15, line_content="def beta():"),
        )
        gh = GrepHit(path="/funcs.py", line_matches=matches)
        assert gh.path == "/funcs.py"
        assert len(gh.line_matches) == 2
        assert gh.line_matches[0].line_number == 5

    def test_search_hit_with_chunks(self):
        chunks = (
            ChunkMatch(name="alpha", line_start=1, line_end=3, score=0.95, snippet="def alpha():"),
            ChunkMatch(name="beta", line_start=5, line_end=7, score=0.80, snippet="def beta():"),
        )
        sh = SearchHit(path="/funcs.py", score=0.95, chunk_matches=chunks)
        assert sh.path == "/funcs.py"
        assert sh.score == 0.95
        assert len(sh.chunk_matches) == 2


# ==================================================================
# Result types construction
# ==================================================================


class TestResultTypes:
    def test_glob_query_result_with_hits(self):
        hits = (
            GlobHit(path="/a.py", size_bytes=100),
            GlobHit(path="/b.py", size_bytes=200),
        )
        r = GlobQueryResult(
            success=True,
            message="Found 2 files",
            hits=hits,
            pattern="*.py",
            path="/",
        )
        assert r.success is True
        assert len(r.hits) == 2
        assert r.pattern == "*.py"

    def test_glob_query_result_failure(self):
        r = GlobQueryResult(success=False, message="No mount found")
        assert r.success is False
        assert r.hits == ()

    def test_grep_query_result_with_hits(self):
        hits = (
            GrepHit(
                path="/a.py",
                line_matches=(LineMatch(line_number=1, line_content="def foo():"),),
            ),
        )
        r = GrepQueryResult(
            success=True,
            message="1 match in 1 file",
            hits=hits,
            pattern="def ",
            files_searched=10,
            files_matched=1,
        )
        assert r.files_searched == 10
        assert r.files_matched == 1
        assert len(r.hits) == 1

    def test_grep_query_result_truncated(self):
        r = GrepQueryResult(
            success=True,
            message="Results truncated",
            truncated=True,
        )
        assert r.truncated is True

    def test_search_query_result_with_hits(self):
        hits = (
            SearchHit(
                path="/funcs.py",
                score=0.95,
                chunk_matches=(ChunkMatch(name="alpha", line_start=1, line_end=3, score=0.95),),
            ),
        )
        r = SearchQueryResult(
            success=True,
            message="1 result",
            hits=hits,
            query="authentication function",
            files_matched=1,
        )
        assert r.query == "authentication function"
        assert r.files_matched == 1
        assert r.hits[0].score == 0.95

    def test_search_query_result_failure(self):
        r = SearchQueryResult(success=False, message="No search engine")
        assert r.success is False
        assert r.hits == ()

"""Tests for query/render.py — all render modes and helpers."""

from __future__ import annotations

import pytest

from grover.query.render import render_query_result
from grover.results import Candidate, Detail, GroverResult


def _candidate(
    path: str,
    *,
    content: str | None = None,
    kind: str | None = None,
    score: float | None = None,
    details: tuple[Detail, ...] = (),
    lines: int | None = None,
    size_bytes: int | None = None,
    tokens: int | None = None,
    mime_type: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> Candidate:
    return Candidate(
        path=path, content=content, kind=kind, score=score, details=details,
        lines=lines, size_bytes=size_bytes, tokens=tokens, mime_type=mime_type,
        created_at=created_at, updated_at=updated_at,
    )


# ===========================================================================
# Error rendering
# ===========================================================================


class TestErrorRendering:
    def test_errors_only(self):
        result = GroverResult(success=False, errors=["file not found", "bad path"])
        output = render_query_result(result, mode="content")
        assert "Error: file not found" in output
        assert "Error: bad path" in output

    def test_body_with_errors(self):
        result = GroverResult(
            success=True,
            candidates=[_candidate("/a.py", content="hello")],
            errors=["warning: deprecated"],
        )
        output = render_query_result(result, mode="content")
        assert "hello" in output
        assert "Error: warning: deprecated" in output


# ===========================================================================
# Content mode
# ===========================================================================


class TestContentMode:
    def test_single_file(self):
        result = GroverResult(candidates=[_candidate("/a.py", content="hello")])
        output = render_query_result(result, mode="content")
        assert output == "hello"

    def test_empty_candidates(self):
        result = GroverResult(candidates=[])
        output = render_query_result(result, mode="content")
        assert output == ""

    def test_multi_file_sorted_with_headers(self):
        result = GroverResult(candidates=[
            _candidate("/b.py", content="bravo"),
            _candidate("/a.py", content="alpha"),
        ])
        output = render_query_result(result, mode="content")
        assert "==> /a.py <==" in output
        assert "==> /b.py <==" in output
        # a.py should come before b.py (sorted)
        assert output.index("/a.py") < output.index("/b.py")


# ===========================================================================
# Action mode
# ===========================================================================


class TestActionMode:
    def test_no_changes(self):
        result = GroverResult(candidates=[])
        output = render_query_result(result, mode="action")
        assert output == "No changes"

    def test_single_path(self):
        detail = Detail(operation="write")
        result = GroverResult(candidates=[_candidate("/a.py", details=(detail,))])
        output = render_query_result(result, mode="action")
        assert output == "Wrote /a.py"

    def test_multi_path(self):
        detail = Detail(operation="delete")
        result = GroverResult(candidates=[
            _candidate("/a.py", details=(detail,)),
            _candidate("/b.py", details=(detail,)),
        ])
        output = render_query_result(result, mode="action")
        assert output == "Deleted 2 paths"

    def test_errors_only_action(self):
        # success=True with errors but no candidates → _render_action error path
        result = GroverResult(
            success=True, errors=["permission denied"], candidates=[],
        )
        output = render_query_result(result, mode="action")
        assert "Error: permission denied" in output

    def test_all_verbs(self):
        for op, verb in [
            ("write", "Wrote"), ("edit", "Edited"), ("delete", "Deleted"),
            ("move", "Moved"), ("copy", "Copied"), ("mkdir", "Created"),
            ("mkconn", "Connected"), ("custom_op", "Custom op"),
        ]:
            detail = Detail(operation=op)
            result = GroverResult(candidates=[_candidate("/x", details=(detail,))])
            output = render_query_result(result, mode="action")
            assert verb in output


# ===========================================================================
# Stat mode
# ===========================================================================


class TestStatMode:
    def test_stat_with_metadata(self):
        result = GroverResult(candidates=[
            _candidate(
                "/a.py", kind="file", lines=42, size_bytes=1024,
                tokens=100, mime_type="text/x-python",
                created_at="2024-01-01", updated_at="2024-06-01",
            ),
        ])
        output = render_query_result(result, mode="stat")
        assert "/a.py" in output
        assert "kind: file" in output
        assert "lines: 42" in output
        assert "size_bytes: 1024" in output

    def test_stat_with_none_values(self):
        result = GroverResult(candidates=[_candidate("/a.py", kind="file")])
        output = render_query_result(result, mode="stat")
        assert "/a.py" in output
        assert "kind: file" in output
        assert "lines:" not in output


# ===========================================================================
# Query list mode
# ===========================================================================


class TestQueryListMode:
    def test_unranked(self):
        result = GroverResult(candidates=[
            _candidate("/b.py"),
            _candidate("/a.py"),
        ])
        output = render_query_result(result, mode="query_list")
        assert "/a.py" in output
        assert "/b.py" in output
        assert "\t" not in output

    def test_ranked(self):
        # score is a property derived from details[-1].score
        result = GroverResult(candidates=[
            _candidate("/a.py", details=(Detail(operation="search", score=0.95),)),
            _candidate("/b.py", details=(Detail(operation="search", score=0.80),)),
        ])
        output = render_query_result(result, mode="query_list")
        assert "/a.py\t0.9500" in output
        assert "/b.py\t0.8000" in output


# ===========================================================================
# Unhandled mode
# ===========================================================================


class TestUnhandledMode:
    def test_raises_assertion_error(self):
        result = GroverResult(candidates=[_candidate("/a.py")])
        with pytest.raises(AssertionError, match="Unhandled render mode"):
            render_query_result(result, mode="bogus_mode")  # type: ignore[arg-type]

"""Tests for Grover v2 result types — Detail, Candidate, GroverResult."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from grover.results import Candidate, Detail, GroverResult


def _c(path: str, kind: str = "file", **kwargs: object) -> Candidate:
    """Shorthand for building test candidates with auto-generated id."""
    return Candidate(id=str(uuid.uuid4()), path=path, kind=kind, **kwargs)


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


class TestDetail:
    def test_construction(self):
        d = Detail(operation="semantic_search", score=0.95)
        assert d.operation == "semantic_search"
        assert d.score == 0.95
        assert d.success is True

    def test_defaults(self):
        d = Detail(operation="read")
        assert d.score is None
        assert d.success is True
        assert d.message == ""
        assert d.metadata is None

    def test_json_excludes_none(self):
        d = Detail(
            operation="grep",
            metadata={"line_number": 42, "line_content": "def login():"},
        )
        data = d.model_dump(exclude_none=True)
        assert data["operation"] == "grep"
        assert data["metadata"]["line_number"] == 42

    def test_json_round_trip(self):
        d = Detail(operation="pagerank", score=0.42)
        data = d.model_dump()
        restored = Detail.model_validate(data)
        assert restored == d

    def test_frozen(self):
        d = Detail(operation="read")
        with pytest.raises(ValidationError):
            d.operation = "write"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_construction(self):
        c = _c("/src/auth.py")
        assert c.path == "/src/auth.py"
        assert c.kind == "file"
        assert c.name == "auth.py"
        assert c.content is None
        assert c.lines == 0

    def test_score_property_empty_details(self):
        c = _c("/a.py")
        assert c.score == 0.0

    def test_score_property_from_last_detail(self):
        c = _c(
            "/a.py",
            details=[
                Detail(operation="search", score=0.9),
                Detail(operation="pagerank", score=0.42),
            ],
        )
        assert c.score == 0.42

    def test_json_excludes_none(self):
        c = _c("/a.py", lines=100, size_bytes=4096)
        data = c.model_dump(exclude_none=True)
        assert "content" not in data
        assert "weight" not in data
        assert data["path"] == "/a.py"
        assert data["lines"] == 100

    def test_connection_candidate(self):
        c = _c(
            "/a.py/.connections/imports/b.py",
            kind="connection",
            weight=1.0,
            distance=0.5,
        )
        data = c.model_dump(exclude_none=True)
        assert data["weight"] == 1.0
        assert data["distance"] == 0.5
        assert c.kind == "connection"

    def test_version_candidate(self):
        c = _c(
            "/a.py/.versions/3",
            kind="version",
        )
        assert c.name == "3"

    def test_json_round_trip(self):
        c = _c(
            "/a.py",
            lines=50,
            details=[Detail(operation="read", score=1.0)],
        )
        data = c.model_dump()
        restored = Candidate.model_validate(data)
        assert restored.path == c.path
        assert restored.score == c.score

    def test_frozen(self):
        c = _c("/a.py")
        with pytest.raises(ValidationError):
            c.path = "/b.py"

    def test_zero_metrics_included_in_json(self):
        """0 is not None — zero metrics should be present in JSON."""
        c = _c("/a.py", lines=0, size_bytes=0)
        data = c.model_dump(exclude_none=True)
        assert "lines" in data
        assert data["lines"] == 0

    def test_details_is_immutable_tuple(self):
        """details is a tuple — truly immutable, not just frozen field assignment."""
        c = _c("/a.py")
        assert isinstance(c.details, tuple)


# ---------------------------------------------------------------------------
# GroverResult — construction & data access
# ---------------------------------------------------------------------------


class TestGroverResultBasics:
    def test_empty_result(self):
        r = GroverResult()
        assert r.success is True
        assert r.errors == []
        assert r.error_message == ""
        assert r.candidates == []
        assert r.paths == ()
        assert r.file is None
        assert r.content is None
        assert len(r) == 0
        assert not r  # empty + success = falsy (no candidates)

    def test_with_candidates(self):
        r = GroverResult(
            candidates=[
                _c("/a.py"),
                _c("/b.py"),
            ]
        )
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.file.path == "/a.py"
        assert r

    def test_failed_result_is_falsy(self):
        r = GroverResult(
            success=False,
            candidates=[_c("/a.py")],
        )
        assert not r

    def test_contains(self):
        r = GroverResult(candidates=[_c("/a.py")])
        assert "/a.py" in r
        assert "/b.py" not in r

    def test_iteration(self):
        candidates = [_c("/a.py"), _c("/b.py")]
        r = GroverResult(candidates=candidates)
        paths = [c.path for c in r]
        assert paths == ["/a.py", "/b.py"]

    def test_content_shorthand(self):
        r = GroverResult(
            candidates=[_c("/a.py", content="print('hello')")]
        )
        assert r.content == "print('hello')"

    def test_explain(self):
        d1 = Detail(operation="search", score=0.9)
        d2 = Detail(operation="pagerank", score=0.4)
        r = GroverResult(
            candidates=[
                _c("/a.py", details=[d1, d2]),
                _c("/b.py", details=[d1]),
            ]
        )
        assert len(r.explain("/a.py")) == 2
        assert len(r.explain("/b.py")) == 1
        assert r.explain("/c.py") == []


# ---------------------------------------------------------------------------
# GroverResult — factories
# ---------------------------------------------------------------------------


class TestGroverResultConstruction:
    def test_direct_construction(self):
        r = GroverResult(
            candidates=[
                _c("/a.py"),
                _c("/b.py"),
            ],
            errors=["test"],
        )
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.errors == ["test"]
        assert r.error_message == "test"


# ---------------------------------------------------------------------------
# GroverResult — set algebra
# ---------------------------------------------------------------------------


class TestGroverResultSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> GroverResult:
        return GroverResult(
            candidates=[
                _c(p, details=[Detail(operation=operation)])
                for p in paths
            ]
        )

    def test_intersection(self):
        a = self._make(["/a.py", "/b.py", "/c.py"], "search")
        b = self._make(["/b.py", "/c.py", "/d.py"], "grep")
        result = a & b
        assert set(result.paths) == {"/b.py", "/c.py"}

    def test_intersection_merges_details(self):
        a = self._make(["/a.py"], "search")
        b = self._make(["/a.py"], "grep")
        result = a & b
        assert len(result.candidates[0].details) == 2
        ops = [d.operation for d in result.candidates[0].details]
        assert ops == ["search", "grep"]

    def test_intersection_empty(self):
        a = self._make(["/a.py"])
        b = self._make(["/b.py"])
        result = a & b
        assert len(result) == 0

    def test_union(self):
        a = self._make(["/a.py", "/b.py"], "search")
        b = self._make(["/b.py", "/c.py"], "grep")
        result = a | b
        assert set(result.paths) == {"/a.py", "/b.py", "/c.py"}

    def test_union_merges_overlapping_details(self):
        a = self._make(["/a.py"], "search")
        b = self._make(["/a.py"], "grep")
        result = a | b
        assert len(result.candidates[0].details) == 2

    def test_difference(self):
        a = self._make(["/a.py", "/b.py", "/c.py"])
        b = self._make(["/b.py"])
        result = a - b
        assert set(result.paths) == {"/a.py", "/c.py"}

    def test_difference_empty_right(self):
        a = self._make(["/a.py", "/b.py"])
        b = GroverResult()
        result = a - b
        assert set(result.paths) == {"/a.py", "/b.py"}

    def test_success_propagation_and(self):
        a = GroverResult(success=True, candidates=[_c("/a.py")])
        b = GroverResult(success=False, candidates=[_c("/a.py")])
        result = a & b
        assert result.success is False

    def test_success_propagation_or(self):
        a = GroverResult(success=True, candidates=[_c("/a.py")])
        b = GroverResult(success=False, candidates=[_c("/b.py")])
        result = a | b
        assert result.success is False

    def test_grover_propagation_and(self):
        """_grover propagates from left operand in &."""
        a = self._make(["/a.py"])
        b = self._make(["/a.py"])
        sentinel = object()
        a._grover = sentinel
        result = a & b
        assert result._grover is sentinel

    def test_grover_propagation_or(self):
        """_grover propagates from left operand in |."""
        a = self._make(["/a.py"])
        b = self._make(["/b.py"])
        sentinel = object()
        a._grover = sentinel
        result = a | b
        assert result._grover is sentinel

    def test_grover_propagation_sub(self):
        """_grover propagates from left operand in -."""
        a = self._make(["/a.py", "/b.py"])
        b = self._make(["/b.py"])
        sentinel = object()
        a._grover = sentinel
        result = a - b
        assert result._grover is sentinel


# ---------------------------------------------------------------------------
# GroverResult — enrichment chains
# ---------------------------------------------------------------------------


class TestGroverResultEnrichment:
    def test_sort_by_last_operation(self):
        r = GroverResult(
            candidates=[
                _c("/low.py", details=[Detail(operation="search", score=0.1)]),
                _c("/high.py", details=[Detail(operation="search", score=0.9)]),
                _c("/mid.py", details=[Detail(operation="search", score=0.5)]),
            ],
        )
        sorted_r = r.sort()
        assert [c.path for c in sorted_r] == ["/high.py", "/mid.py", "/low.py"]

    def test_sort_by_explicit_operation(self):
        r = GroverResult(
            candidates=[
                _c("/a.py", details=[
                    Detail(operation="search", score=0.9),
                    Detail(operation="pagerank", score=0.2),
                ]),
                _c("/b.py", details=[
                    Detail(operation="search", score=0.1),
                    Detail(operation="pagerank", score=0.8),
                ]),
            ],
        )
        # Default: sort by last operation (pagerank)
        sorted_r = r.sort()
        assert sorted_r.candidates[0].path == "/b.py"
        # Explicit: sort by search
        sorted_r = r.sort(operation="search")
        assert sorted_r.candidates[0].path == "/a.py"

    def test_sort_ascending(self):
        r = GroverResult(
            candidates=[
                _c("/high.py", details=[Detail(operation="s", score=0.9)]),
                _c("/low.py", details=[Detail(operation="s", score=0.1)]),
            ],
        )
        sorted_r = r.sort(reverse=False)
        assert [c.path for c in sorted_r] == ["/low.py", "/high.py"]

    def test_sort_custom_key(self):
        r = GroverResult(
            candidates=[
                _c("/small.py", size_bytes=100),
                _c("/big.py", size_bytes=9000),
            ]
        )
        sorted_r = r.sort(key=lambda c: c.size_bytes)
        assert sorted_r.candidates[0].path == "/big.py"

    def test_sort_no_args_uses_last_score(self):
        """sort() with no args uses candidate.score (last detail's score)."""
        r = GroverResult(
            candidates=[
                _c("/a.py", details=[Detail(operation="s", score=0.1)]),
                _c("/b.py", details=[Detail(operation="s", score=0.9)]),
            ]
        )
        sorted_r = r.sort()
        assert sorted_r.candidates[0].path == "/b.py"

    def test_top(self):
        r = GroverResult(
            candidates=[
                _c("/a.py", details=[Detail(operation="s", score=0.1)]),
                _c("/b.py", details=[Detail(operation="s", score=0.9)]),
                _c("/c.py", details=[Detail(operation="s", score=0.5)]),
            ],
        )
        top2 = r.top(2)
        assert len(top2) == 2
        assert top2.candidates[0].path == "/b.py"
        assert top2.candidates[1].path == "/c.py"

    def test_top_more_than_available(self):
        r = GroverResult(
            candidates=[_c("/a.py", details=[Detail(operation="s")])],
        )
        top5 = r.top(5)
        assert len(top5) == 1

    def test_filter(self):
        r = GroverResult(
            candidates=[
                _c("/a.py", size_bytes=100),
                _c("/b/", kind="directory"),
                _c("/c.py", size_bytes=0),
            ]
        )
        files_with_content = r.filter(lambda c: c.kind == "file" and c.size_bytes > 0)
        assert len(files_with_content) == 1
        assert files_with_content.candidates[0].path == "/a.py"

    def test_kinds(self):
        r = GroverResult(
            candidates=[
                _c("/a.py"),
                _c("/b/", kind="directory"),
                _c("/a.py/.chunks/login", kind="chunk"),
            ]
        )
        files_only = r.kinds("file")
        assert len(files_only) == 1
        files_and_chunks = r.kinds("file", "chunk")
        assert len(files_and_chunks) == 2

    def test_enrichment_preserves_grover(self):
        """Enrichment chains propagate _grover."""
        r = GroverResult(
            candidates=[_c("/a.py", details=[Detail(operation="s")])],
        )
        sentinel = object()
        r._grover = sentinel
        assert r.sort()._grover is sentinel
        assert r.filter(lambda c: True)._grover is sentinel
        assert r.kinds("file")._grover is sentinel
        assert r.top(10)._grover is sentinel


# ---------------------------------------------------------------------------
# GroverResult — chain stubs (without bound grover)
# ---------------------------------------------------------------------------


class TestScoreFor:
    def test_score_for(self):
        c = _c(
            "/a.py",
            details=[
                Detail(operation="search", score=0.9),
                Detail(operation="pagerank", score=0.4),
            ],
        )
        assert c.score_for("search") == 0.9
        assert c.score_for("pagerank") == 0.4
        assert c.score_for("nonexistent") == 0.0


class TestGroverResultChainStubs:
    def test_chain_without_grover_raises(self):
        r = GroverResult(candidates=[_c("/a.py")])
        with pytest.raises(RuntimeError, match="bound Grover instance"):
            r.read()

    def test_all_crud_stubs_raise_without_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        for method_name in ("read", "delete", "stat", "ls"):
            with pytest.raises(RuntimeError):
                getattr(r, method_name)()

    def test_edit_raises_without_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        with pytest.raises(RuntimeError):
            r.edit("old", "new")

    def test_all_query_stubs_raise_without_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        with pytest.raises(RuntimeError):
            r.glob("*.py")
        with pytest.raises(RuntimeError):
            r.grep("pattern")
        with pytest.raises(RuntimeError):
            r.semantic_search("query")
        with pytest.raises(RuntimeError):
            r.vector_search([0.1, 0.2])
        with pytest.raises(RuntimeError):
            r.lexical_search("query")

    def test_all_graph_stubs_raise_without_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        graph_methods = [
            "predecessors", "successors", "ancestors", "descendants",
            "meeting_subgraph", "min_meeting_subgraph",
            "pagerank", "betweenness_centrality", "closeness_centrality",
            "degree_centrality", "in_degree_centrality", "out_degree_centrality",
            "hits",
        ]
        for method_name in graph_methods:
            with pytest.raises(RuntimeError):
                getattr(r, method_name)()

    def test_neighborhood_raises_without_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        with pytest.raises(RuntimeError):
            r.neighborhood(depth=2)


# ---------------------------------------------------------------------------
# GroverResult — JSON serialization
# ---------------------------------------------------------------------------


class TestGroverResultJSON:
    def test_model_dump_excludes_grover(self):
        r = GroverResult(candidates=[_c("/a.py")])
        r._grover = object()
        data = r.model_dump()
        assert "_grover" not in data

    def test_model_dump_exclude_none(self):
        r = GroverResult(
            candidates=[
                _c("/a.py", lines=142, details=[
                    Detail(operation="semantic_search", score=0.95)
                ])
            ]
        )
        data = r.model_dump(exclude_none=True)
        candidate = data["candidates"][0]
        assert "content" not in candidate
        assert "weight" not in candidate
        assert candidate["path"] == "/a.py"
        assert candidate["lines"] == 142
        detail = candidate["details"][0]
        assert "metadata" not in detail
        assert detail["operation"] == "semantic_search"
        assert detail["score"] == 0.95

    def test_json_round_trip(self):
        r = GroverResult(
            success=True,
            errors=["Found 2 files"],
            candidates=[
                _c("/a.py", details=[Detail(operation="glob", score=0.0)]),
                _c("/b.py", details=[Detail(operation="glob", score=0.0)]),
            ],
        )
        data = r.model_dump()
        restored = GroverResult.model_validate(data)
        assert restored.paths == r.paths
        assert restored.success == r.success
        assert restored.errors == r.errors
        assert len(restored.candidates) == 2
        assert restored.candidates[0].details[0].operation == "glob"

    def test_independent_candidate_lists(self):
        """Pydantic v2 should give each instance its own candidates list."""
        r1 = GroverResult()
        r2 = GroverResult()
        assert r1.candidates is not r2.candidates

    def test_json_string_round_trip(self):
        """FastAPI uses model_dump_json, not model_dump."""
        r = GroverResult(
            candidates=[_c("/a.py", details=[Detail(operation="read")])],
        )
        json_str = r.model_dump_json(exclude_none=True)
        restored = GroverResult.model_validate_json(json_str)
        assert restored.paths == r.paths
        assert restored.success == r.success


# ---------------------------------------------------------------------------
# Merge edge cases
# ---------------------------------------------------------------------------


class TestMergeEdgeCases:
    def test_merge_preserves_zero_metrics_from_left(self):
        """lines=0 on left should NOT be replaced by right's value."""
        a = GroverResult(candidates=[
            Candidate(id="1", path="/a.py", kind="file", lines=0, size_bytes=0),
        ])
        b = GroverResult(candidates=[
            Candidate(id="2", path="/a.py", kind="file", lines=50, size_bytes=4096),
        ])
        result = a & b
        assert result.candidates[0].lines == 0
        assert result.candidates[0].size_bytes == 0

    def test_merge_preserves_empty_string_content(self):
        """content='' (empty file) should NOT be replaced by right's content."""
        a = GroverResult(candidates=[
            Candidate(id="1", path="/a.py", kind="file", content=""),
        ])
        b = GroverResult(candidates=[
            Candidate(id="2", path="/a.py", kind="file", content="real content"),
        ])
        result = a & b
        assert result.candidates[0].content == ""

    def test_merge_preserves_left_id(self):
        """Left candidate's id wins in a merge."""
        a = GroverResult(candidates=[
            Candidate(id="left-id", path="/a.py", kind="file"),
        ])
        b = GroverResult(candidates=[
            Candidate(id="right-id", path="/a.py", kind="file"),
        ])
        result = a & b
        assert result.candidates[0].id == "left-id"

    def test_merge_falls_back_to_right_for_none(self):
        """When left has None, right's value is used."""
        a = GroverResult(candidates=[
            Candidate(id="1", path="/a.py", kind="file", content=None, mime_type=None),
        ])
        b = GroverResult(candidates=[
            Candidate(id="2", path="/a.py", kind="file", content="hello", mime_type="text/python"),
        ])
        result = a & b
        assert result.candidates[0].content == "hello"
        assert result.candidates[0].mime_type == "text/python"


# ---------------------------------------------------------------------------
# Top edge cases
# ---------------------------------------------------------------------------


class TestTopEdgeCases:
    def test_top_zero_raises(self):
        r = GroverResult(
            candidates=[_c("/a.py", details=[Detail(operation="s", score=0.5)])],
        )
        with pytest.raises(ValueError, match="k must be >= 1"):
            r.top(0)

    def test_top_negative_raises(self):
        r = GroverResult(
            candidates=[_c("/a.py", details=[Detail(operation="s", score=0.5)])],
        )
        with pytest.raises(ValueError, match="k must be >= 1"):
            r.top(-1)


# ---------------------------------------------------------------------------
# Additional coverage from review
# ---------------------------------------------------------------------------


class TestScoreEdgeCases:
    def test_score_property_with_none_score(self):
        """score=None on last detail should return 0.0."""
        c = _c("/a.py", details=[Detail(operation="read", score=None)])
        assert c.score == 0.0

    def test_score_for_with_none_score(self):
        """score_for returns 0.0 when the matching detail has score=None."""
        c = _c("/a.py", details=[Detail(operation="read", score=None)])
        assert c.score_for("read") == 0.0

    def test_score_for_returns_most_recent_duplicate(self):
        """When multiple details share an operation, most recent wins."""
        c = _c("/a.py", details=[
            Detail(operation="search", score=0.3),
            Detail(operation="search", score=0.9),
        ])
        assert c.score_for("search") == 0.9


class TestSubSuccessPropagation:
    def test_sub_preserves_left_success(self):
        a = GroverResult(success=True, candidates=[_c("/a.py")])
        b = GroverResult(success=False, candidates=[_c("/b.py")])
        result = a - b
        assert result.success is True


class TestFirstSet:
    def test_returns_a_when_not_none(self):
        assert GroverResult._first_set("a", "b") == "a"

    def test_returns_b_when_a_is_none(self):
        assert GroverResult._first_set(None, "b") == "b"

    def test_returns_default_when_both_none(self):
        assert GroverResult._first_set(None, None, "default") == "default"

    def test_returns_none_when_all_none(self):
        assert GroverResult._first_set(None, None) is None

    def test_preserves_zero(self):
        assert GroverResult._first_set(0, 99) == 0

    def test_preserves_empty_string(self):
        assert GroverResult._first_set("", "fallback") == ""

    def test_preserves_zero_float(self):
        assert GroverResult._first_set(0.0, 1.0) == 0.0


class TestRequiredFields:
    def test_candidate_requires_id(self):
        with pytest.raises(ValidationError):
            Candidate(path="/a.py", kind="file")

    def test_candidate_requires_kind(self):
        with pytest.raises(ValidationError):
            Candidate(id="1", path="/a.py")

    def test_candidate_requires_path(self):
        with pytest.raises(ValidationError):
            Candidate(id="1", kind="file")


class TestDatetimeRoundTrip:
    def test_candidate_datetime_json_round_trip(self):
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        c = Candidate(
            id="1", path="/a.py", kind="file", created_at=now, updated_at=now
        )
        json_str = c.model_dump_json()
        restored = Candidate.model_validate_json(json_str)
        assert restored.created_at == now
        assert restored.updated_at == now


class TestDuplicatePaths:
    def test_as_dict_last_wins_on_duplicate_paths(self):
        """If candidates have duplicate paths, _as_dict keeps the last one."""
        c1 = Candidate(id="1", path="/a.py", kind="file", content="first")
        c2 = Candidate(id="2", path="/a.py", kind="file", content="second")
        r = GroverResult(candidates=[c1, c2])
        d = r._as_dict()
        assert len(d) == 1
        assert d["/a.py"].content == "second"

    def test_intersection_with_duplicates_on_one_side(self):
        c1 = Candidate(id="1", path="/a.py", kind="file", content="first")
        c2 = Candidate(id="2", path="/a.py", kind="file", content="second")
        a = GroverResult(candidates=[c1, c2])
        b = GroverResult(candidates=[_c("/a.py")])
        result = a & b
        assert len(result) == 1

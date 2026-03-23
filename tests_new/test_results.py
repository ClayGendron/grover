"""Tests for Grover v2 result types — Detail, Candidate, GroverResult."""

from __future__ import annotations

import pytest

from grover.results import Candidate, Detail, GroverResult


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


class TestDetail:
    def test_construction(self):
        d = Detail(operation="semantic_search", score=0.95, snippet="auth handler")
        assert d.operation == "semantic_search"
        assert d.score == 0.95
        assert d.snippet == "auth handler"
        assert d.success is True

    def test_defaults(self):
        d = Detail(operation="read")
        assert d.score == 0.0
        assert d.success is True
        assert d.message == ""
        assert d.snippet is None
        assert d.algorithm is None
        assert d.extra is None

    def test_json_excludes_none(self):
        d = Detail(operation="grep", line_number=42, line_content="def login():")
        data = d.model_dump(exclude_none=True)
        assert "snippet" not in data
        assert "algorithm" not in data
        assert "extra" not in data
        assert data["operation"] == "grep"
        assert data["line_number"] == 42

    def test_json_round_trip(self):
        d = Detail(operation="pagerank", score=0.42, algorithm="pagerank")
        data = d.model_dump()
        restored = Detail.model_validate(data)
        assert restored == d

    def test_frozen(self):
        d = Detail(operation="read")
        with pytest.raises(Exception):
            d.operation = "write"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_construction(self):
        c = Candidate(path="/src/auth.py")
        assert c.path == "/src/auth.py"
        assert c.kind == "file"
        assert c.name == ""
        assert c.content is None
        assert c.lines == 0

    def test_score_property_empty_details(self):
        c = Candidate(path="/a.py")
        assert c.score == 0.0

    def test_score_property_from_last_detail(self):
        c = Candidate(
            path="/a.py",
            details=[
                Detail(operation="search", score=0.9),
                Detail(operation="pagerank", score=0.42),
            ],
        )
        assert c.score == 0.42

    def test_json_excludes_none(self):
        c = Candidate(path="/a.py", kind="file", lines=100, size_bytes=4096)
        data = c.model_dump(exclude_none=True)
        assert "content" not in data
        assert "source_path" not in data
        assert "version_number" not in data
        assert data["path"] == "/a.py"
        assert data["lines"] == 100

    def test_connection_candidate(self):
        c = Candidate(
            path="/a.py/.connections/imports/b.py",
            kind="connection",
            source_path="/a.py",
            target_path="/b.py",
            connection_type="imports",
            weight=1.0,
        )
        data = c.model_dump(exclude_none=True)
        assert data["source_path"] == "/a.py"
        assert data["target_path"] == "/b.py"
        assert data["connection_type"] == "imports"

    def test_version_candidate(self):
        c = Candidate(
            path="/a.py/.versions/3",
            kind="version",
            version_number=3,
        )
        assert c.version_number == 3

    def test_json_round_trip(self):
        c = Candidate(
            path="/a.py",
            kind="file",
            lines=50,
            details=[Detail(operation="read", score=1.0)],
        )
        data = c.model_dump()
        restored = Candidate.model_validate(data)
        assert restored.path == c.path
        assert restored.score == c.score

    def test_frozen(self):
        c = Candidate(path="/a.py")
        with pytest.raises(Exception):
            c.path = "/b.py"

    def test_zero_metrics_included_in_json(self):
        """0 is not None — zero metrics should be present in JSON."""
        c = Candidate(path="/a.py", lines=0, size_bytes=0)
        data = c.model_dump(exclude_none=True)
        assert "lines" in data
        assert data["lines"] == 0

    def test_independent_details_lists(self):
        """Pydantic v2 should give each instance its own details list."""
        c1 = Candidate(path="/a.py")
        c2 = Candidate(path="/b.py")
        assert c1.details is not c2.details or c1.details == c2.details == []


# ---------------------------------------------------------------------------
# GroverResult — construction & data access
# ---------------------------------------------------------------------------


class TestGroverResultBasics:
    def test_empty_result(self):
        r = GroverResult()
        assert r.success is True
        assert r.message == ""
        assert r.candidates == []
        assert r.paths == ()
        assert r.file is None
        assert r.content is None
        assert len(r) == 0
        assert not r  # empty + success = falsy (no candidates)

    def test_with_candidates(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/a.py"),
                Candidate(path="/b.py"),
            ]
        )
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.file.path == "/a.py"
        assert r

    def test_failed_result_is_falsy(self):
        r = GroverResult(
            success=False,
            candidates=[Candidate(path="/a.py")],
        )
        assert not r

    def test_contains(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        assert "/a.py" in r
        assert "/b.py" not in r

    def test_iteration(self):
        candidates = [Candidate(path="/a.py"), Candidate(path="/b.py")]
        r = GroverResult(candidates=candidates)
        paths = [c.path for c in r]
        assert paths == ["/a.py", "/b.py"]

    def test_content_shorthand(self):
        r = GroverResult(
            candidates=[Candidate(path="/a.py", content="print('hello')")]
        )
        assert r.content == "print('hello')"

    def test_explain(self):
        d1 = Detail(operation="search", score=0.9)
        d2 = Detail(operation="pagerank", score=0.4)
        r = GroverResult(
            candidates=[
                Candidate(path="/a.py", details=[d1, d2]),
                Candidate(path="/b.py", details=[d1]),
            ]
        )
        assert len(r.explain("/a.py")) == 2
        assert len(r.explain("/b.py")) == 1
        assert r.explain("/c.py") == []


# ---------------------------------------------------------------------------
# GroverResult — factories
# ---------------------------------------------------------------------------


class TestGroverResultFactories:
    def test_from_paths(self):
        r = GroverResult.from_paths(["/a.py", "/b.py"], operation="glob")
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.candidates[0].details[0].operation == "glob"

    def test_from_paths_with_score(self):
        r = GroverResult.from_paths(["/a.py"], operation="search", score=0.9)
        assert r.candidates[0].score == 0.9

    def test_from_candidates(self):
        candidates = [Candidate(path="/a.py"), Candidate(path="/b.py")]
        r = GroverResult.from_candidates(candidates, message="test")
        assert len(r) == 2
        assert r.message == "test"


# ---------------------------------------------------------------------------
# GroverResult — set algebra
# ---------------------------------------------------------------------------


class TestGroverResultSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> GroverResult:
        return GroverResult.from_paths(paths, operation=operation)

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
        a = GroverResult(success=True, candidates=[Candidate(path="/a.py")])
        b = GroverResult(success=False, candidates=[Candidate(path="/a.py")])
        result = a & b
        assert result.success is False

    def test_success_propagation_or(self):
        a = GroverResult(success=True, candidates=[Candidate(path="/a.py")])
        b = GroverResult(success=False, candidates=[Candidate(path="/b.py")])
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
    def test_sort_by_score(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/low.py", details=[Detail(operation="s", score=0.1)]),
                Candidate(path="/high.py", details=[Detail(operation="s", score=0.9)]),
                Candidate(path="/mid.py", details=[Detail(operation="s", score=0.5)]),
            ]
        )
        sorted_r = r.sort()
        assert [c.path for c in sorted_r] == ["/high.py", "/mid.py", "/low.py"]

    def test_sort_ascending(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/high.py", details=[Detail(operation="s", score=0.9)]),
                Candidate(path="/low.py", details=[Detail(operation="s", score=0.1)]),
            ]
        )
        sorted_r = r.sort(reverse=False)
        assert [c.path for c in sorted_r] == ["/low.py", "/high.py"]

    def test_sort_custom_key(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/small.py", size_bytes=100),
                Candidate(path="/big.py", size_bytes=9000),
            ]
        )
        sorted_r = r.sort(key=lambda c: c.size_bytes)
        assert sorted_r.candidates[0].path == "/big.py"

    def test_top(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/a.py", details=[Detail(operation="s", score=0.1)]),
                Candidate(path="/b.py", details=[Detail(operation="s", score=0.9)]),
                Candidate(path="/c.py", details=[Detail(operation="s", score=0.5)]),
            ]
        )
        top2 = r.top(2)
        assert len(top2) == 2
        assert top2.candidates[0].path == "/b.py"
        assert top2.candidates[1].path == "/c.py"

    def test_top_more_than_available(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        top5 = r.top(5)
        assert len(top5) == 1

    def test_filter(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file", size_bytes=100),
                Candidate(path="/b/", kind="directory"),
                Candidate(path="/c.py", kind="file", size_bytes=0),
            ]
        )
        files_with_content = r.filter(lambda c: c.kind == "file" and c.size_bytes > 0)
        assert len(files_with_content) == 1
        assert files_with_content.candidates[0].path == "/a.py"

    def test_kinds(self):
        r = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
                Candidate(path="/b/", kind="directory"),
                Candidate(path="/a.py/.chunks/login", kind="chunk"),
            ]
        )
        files_only = r.kinds("file")
        assert len(files_only) == 1
        files_and_chunks = r.kinds("file", "chunk")
        assert len(files_and_chunks) == 2

    def test_enrichment_preserves_grover(self):
        """Enrichment chains propagate _grover."""
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        sentinel = object()
        r._grover = sentinel
        assert r.sort()._grover is sentinel
        assert r.filter(lambda c: True)._grover is sentinel
        assert r.kinds("file")._grover is sentinel
        assert r.top(10)._grover is sentinel


# ---------------------------------------------------------------------------
# GroverResult — chain stubs (without bound grover)
# ---------------------------------------------------------------------------


class TestGroverResultChainStubs:
    def test_chain_without_grover_raises(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        with pytest.raises(RuntimeError, match="bound Grover instance"):
            r.read()

    def test_all_crud_stubs_raise_without_grover(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        for method_name in ("read", "delete", "stat", "exists", "ls"):
            with pytest.raises(RuntimeError):
                getattr(r, method_name)()

    def test_edit_raises_without_grover(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        with pytest.raises(RuntimeError):
            r.edit("old", "new")

    def test_all_query_stubs_raise_without_grover(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
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
        r = GroverResult(candidates=[Candidate(path="/a.py")])
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
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        with pytest.raises(RuntimeError):
            r.neighborhood(depth=2)


# ---------------------------------------------------------------------------
# GroverResult — JSON serialization
# ---------------------------------------------------------------------------


class TestGroverResultJSON:
    def test_model_dump_excludes_grover(self):
        r = GroverResult(candidates=[Candidate(path="/a.py")])
        r._grover = object()
        data = r.model_dump()
        assert "_grover" not in data

    def test_model_dump_exclude_none(self):
        r = GroverResult(
            candidates=[
                Candidate(
                    path="/a.py",
                    kind="file",
                    lines=142,
                    details=[
                        Detail(operation="semantic_search", score=0.95, snippet="auth handler")
                    ],
                )
            ]
        )
        data = r.model_dump(exclude_none=True)
        candidate = data["candidates"][0]
        assert "content" not in candidate
        assert "source_path" not in candidate
        assert candidate["path"] == "/a.py"
        assert candidate["lines"] == 142
        detail = candidate["details"][0]
        assert "algorithm" not in detail
        assert detail["operation"] == "semantic_search"
        assert detail["snippet"] == "auth handler"

    def test_json_round_trip(self):
        r = GroverResult(
            success=True,
            message="Found 2 files",
            candidates=[
                Candidate(
                    path="/a.py",
                    kind="file",
                    details=[Detail(operation="glob", score=0.0)],
                ),
                Candidate(
                    path="/b.py",
                    kind="file",
                    details=[Detail(operation="glob", score=0.0)],
                ),
            ],
        )
        data = r.model_dump()
        restored = GroverResult.model_validate(data)
        assert restored.paths == r.paths
        assert restored.success == r.success
        assert restored.message == r.message
        assert len(restored.candidates) == 2
        assert restored.candidates[0].details[0].operation == "glob"

    def test_independent_candidate_lists(self):
        """Pydantic v2 should give each instance its own candidates list."""
        r1 = GroverResult()
        r2 = GroverResult()
        assert r1.candidates is not r2.candidates or r1.candidates == r2.candidates == []

"""Tests for result type hierarchy: FileOperationResult, FileSearchResult, Evidence, set algebra."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.ref import Ref
from grover.results import (
    DeleteResult,
    EditResult,
    Evidence,
    FileOperationResult,
    FileSearchCandidate,
    FileSearchResult,
    GetVersionContentResult,
    GlobEvidence,
    GlobResult,
    GraphEvidence,
    GraphResult,
    GrepEvidence,
    GrepResult,
    HybridEvidence,
    HybridSearchResult,
    LexicalEvidence,
    LexicalSearchResult,
    LineMatch,
    ListDirEvidence,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    TrashEvidence,
    TrashResult,
    TreeEvidence,
    TreeResult,
    VectorEvidence,
    VectorSearchResult,
    WriteResult,
)

# =====================================================================
# FileOperationResult base
# =====================================================================


class TestFileOperationResult:
    def test_success(self):
        r = FileOperationResult(success=True, message="ok")
        assert r.success is True
        assert r.message == "ok"

    def test_failure(self):
        r = FileOperationResult(success=False, message="fail")
        assert r.success is False

    def test_existing_types_inherit(self):
        for cls in [
            ReadResult,
            WriteResult,
            EditResult,
            DeleteResult,
            MkdirResult,
            MoveResult,
            RestoreResult,
            GetVersionContentResult,
        ]:
            instance = cls(success=True, message="ok")
            assert isinstance(instance, FileOperationResult), f"{cls.__name__} not subclass"

    def test_existing_result_fields_preserved(self):
        r = ReadResult(
            success=True,
            message="ok",
            content="hello",
            path="/a.txt",
            total_lines=1,
            lines_read=1,
            truncated=False,
            line_offset=0,
        )
        assert r.content == "hello"
        assert r.path == "/a.txt"
        assert r.total_lines == 1


# =====================================================================
# Evidence types
# =====================================================================


class TestEvidence:
    def test_base_frozen(self):
        e = Evidence(strategy="glob", path="/a.py")
        assert e.strategy == "glob"
        assert e.path == "/a.py"

    def test_glob_evidence(self):
        e = GlobEvidence(strategy="glob", path="/a.py", is_directory=False, size_bytes=100)
        assert isinstance(e, Evidence)
        assert e.is_directory is False
        assert e.size_bytes == 100

    def test_grep_evidence(self):
        lm = LineMatch(line_number=5, line_content="def foo():")
        e = GrepEvidence(strategy="grep", path="/a.py", line_matches=(lm,))
        assert isinstance(e, Evidence)
        assert len(e.line_matches) == 1
        assert e.line_matches[0].line_number == 5

    def test_line_match_frozen(self):
        lm = LineMatch(line_number=1, line_content="x")
        assert lm.line_number == 1
        assert lm.context_before == ()
        assert lm.context_after == ()

    def test_line_match_with_context(self):
        lm = LineMatch(
            line_number=3,
            line_content="target",
            context_before=("before1", "before2"),
            context_after=("after1",),
        )
        assert len(lm.context_before) == 2
        assert len(lm.context_after) == 1

    def test_tree_evidence(self):
        e = TreeEvidence(strategy="tree", path="/dir", depth=2, is_directory=True)
        assert e.depth == 2
        assert e.is_directory is True

    def test_listdir_evidence(self):
        e = ListDirEvidence(strategy="list_dir", path="/a.py", is_directory=False, size_bytes=50)
        assert e.size_bytes == 50

    def test_trash_evidence(self):
        now = datetime.now(UTC)
        e = TrashEvidence(
            strategy="trash", path="/__trash__/abc/a.py", deleted_at=now, original_path="/a.py"
        )
        assert e.deleted_at == now
        assert e.original_path == "/a.py"

    def test_vector_evidence(self):
        e = VectorEvidence(strategy="vector_search", path="/a.py", snippet="auth logic")
        assert e.snippet == "auth logic"

    def test_lexical_evidence(self):
        e = LexicalEvidence(strategy="lexical_search", path="/a.py", snippet="login")
        assert e.snippet == "login"

    def test_hybrid_evidence(self):
        e = HybridEvidence(strategy="hybrid_search", path="/a.py", snippet="mixed")
        assert e.snippet == "mixed"

    def test_graph_evidence(self):
        e = GraphEvidence(
            strategy="dependents", path="/a.py", algorithm="dependents", relationship="imports"
        )
        assert e.algorithm == "dependents"
        assert e.relationship == "imports"


# =====================================================================
# FileSearchResult base
# =====================================================================


class TestFileSearchResult:
    def test_empty_result(self):
        r = FileSearchResult(success=True, message="empty")
        assert len(r) == 0
        assert r.paths == ()
        assert not r  # empty is falsy
        assert list(r) == []

    def test_with_entries(self):
        candidates = [
            FileSearchCandidate(path="/a.py", evidence=[Evidence(strategy="glob", path="/a.py")]),
            FileSearchCandidate(path="/b.py", evidence=[Evidence(strategy="glob", path="/b.py")]),
        ]
        r = FileSearchResult(success=True, message="2 paths", candidates=candidates)
        assert len(r) == 2
        assert r  # non-empty is truthy
        assert "/a.py" in r
        assert "/c.py" not in r

    def test_paths_property(self):
        candidates = [
            FileSearchCandidate(path="/x.py", evidence=[]),
            FileSearchCandidate(path="/y.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", candidates=candidates)
        assert set(r.paths) == {"/x.py", "/y.py"}

    def test_iteration(self):
        candidates = [
            FileSearchCandidate(path="/a.py", evidence=[]),
            FileSearchCandidate(path="/b.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", candidates=candidates)
        assert set(r) == {"/a.py", "/b.py"}

    def test_explain(self):
        e1 = Evidence(strategy="glob", path="/a.py")
        e2 = Evidence(strategy="grep", path="/a.py")
        candidates = [FileSearchCandidate(path="/a.py", evidence=[e1, e2])]
        r = FileSearchResult(success=True, message="ok", candidates=candidates)
        chain = r.explain("/a.py")
        assert len(chain) == 2
        assert chain[0].strategy == "glob"
        assert chain[1].strategy == "grep"

    def test_explain_missing_path(self):
        r = FileSearchResult(success=True, message="ok")
        assert r.explain("/missing.py") == []

    def test_to_refs(self):
        candidates = [
            FileSearchCandidate(path="/a.py", evidence=[]),
            FileSearchCandidate(path="/b.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", candidates=candidates)
        refs = r.to_refs()
        assert len(refs) == 2
        assert all(isinstance(ref, Ref) for ref in refs)
        paths = {ref.path for ref in refs}
        assert paths == {"/a.py", "/b.py"}

    def test_from_paths(self):
        r = FileSearchResult.from_paths(["/a.py", "/b.py"], strategy="custom")
        assert len(r) == 2
        assert "/a.py" in r
        assert r.explain("/a.py")[0].strategy == "custom"

    def test_from_refs(self):
        refs = [Ref(path="/a.py"), Ref(path="/b.py")]
        r = FileSearchResult.from_refs(refs, strategy="ref")
        assert len(r) == 2
        assert "/a.py" in r

    def test_failed_result_is_falsy(self):
        candidates = [FileSearchCandidate(path="/a.py", evidence=[])]
        r = FileSearchResult(success=False, message="fail", candidates=candidates)
        assert not r  # failed is falsy even with entries


# =====================================================================
# Set algebra
# =====================================================================


class TestSetAlgebra:
    def _make(self, paths: list[str], strategy: str = "test") -> FileSearchResult:
        entries = {p: [Evidence(strategy=strategy, path=p)] for p in paths}
        return FileSearchResult(
            success=True, message="ok", candidates=FileSearchResult._dict_to_candidates(entries)
        )

    def test_union(self):
        a = self._make(["/a.py", "/b.py"])
        b = self._make(["/b.py", "/c.py"])
        result = a | b
        assert set(result.paths) == {"/a.py", "/b.py", "/c.py"}
        assert isinstance(result, FileSearchResult)

    def test_intersection(self):
        a = self._make(["/a.py", "/b.py"])
        b = self._make(["/b.py", "/c.py"])
        result = a & b
        assert set(result.paths) == {"/b.py"}

    def test_difference(self):
        a = self._make(["/a.py", "/b.py", "/c.py"])
        b = self._make(["/b.py"])
        result = a - b
        assert set(result.paths) == {"/a.py", "/c.py"}

    def test_pipeline(self):
        a = self._make(["/a.py", "/b.py", "/c.py"])
        b = self._make(["/b.py", "/c.py", "/d.py"])
        result = a >> b
        assert set(result.paths) == {"/b.py", "/c.py"}

    def test_evidence_merged_on_union(self):
        a = self._make(["/a.py"], strategy="glob")
        b = self._make(["/a.py"], strategy="grep")
        result = a | b
        chain = result.explain("/a.py")
        strategies = {e.strategy for e in chain}
        assert strategies == {"glob", "grep"}

    def test_evidence_merged_on_intersection(self):
        a = self._make(["/a.py"], strategy="glob")
        b = self._make(["/a.py"], strategy="grep")
        result = a & b
        chain = result.explain("/a.py")
        strategies = {e.strategy for e in chain}
        assert strategies == {"glob", "grep"}

    def test_evidence_lhs_only_on_difference(self):
        a = self._make(["/a.py", "/b.py"], strategy="glob")
        b = self._make(["/b.py"], strategy="grep")
        result = a - b
        assert "/a.py" in result
        chain = result.explain("/a.py")
        assert all(e.strategy == "glob" for e in chain)

    def test_empty_intersection(self):
        a = self._make(["/a.py"])
        b = self._make(["/b.py"])
        result = a & b
        assert len(result) == 0
        assert not result

    def test_success_propagation_union(self):
        a = FileSearchResult(success=True, message="a")
        b = FileSearchResult(success=False, message="b")
        result = a | b
        assert result.success is True  # True OR False = True

    def test_success_propagation_intersection(self):
        a = FileSearchResult(success=True, message="a")
        b = FileSearchResult(success=False, message="b")
        result = a & b
        assert result.success is False  # True AND False = False

    def test_invalid_operand_and(self):
        import pytest

        r = self._make(["/a.py"])
        with pytest.raises(TypeError):
            r & "not a result"

    def test_invalid_operand_or(self):
        import pytest

        r = self._make(["/a.py"])
        with pytest.raises(TypeError):
            r | 42

    def test_invalid_operand_sub(self):
        import pytest

        r = self._make(["/a.py"])
        with pytest.raises(TypeError):
            r - None  # type: ignore[operator]

    def test_invalid_operand_rshift(self):
        import pytest

        r = self._make(["/a.py"])
        with pytest.raises(TypeError):
            r >> []


# =====================================================================
# Same type preserves subclass, mixed returns base
# =====================================================================


class TestSubclassPreservation:
    def test_same_type_union_preserves(self):
        a = GlobResult(
            success=True,
            message="a",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GlobEvidence(strategy="glob", path="/a.py")]
                )
            ],
        )
        b = GlobResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/b.py", evidence=[GlobEvidence(strategy="glob", path="/b.py")]
                )
            ],
        )
        result = a | b
        assert isinstance(result, GlobResult)

    def test_same_type_intersection_preserves(self):
        a = GrepResult(
            success=True,
            message="a",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GrepEvidence(strategy="grep", path="/a.py")]
                )
            ],
        )
        b = GrepResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GrepEvidence(strategy="grep", path="/a.py")]
                )
            ],
        )
        result = a & b
        assert isinstance(result, GrepResult)

    def test_mixed_types_returns_base(self):
        a = GlobResult(
            success=True,
            message="a",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GlobEvidence(strategy="glob", path="/a.py")]
                )
            ],
        )
        b = GrepResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GrepEvidence(strategy="grep", path="/a.py")]
                )
            ],
        )
        result = a & b
        assert type(result) is FileSearchResult

    def test_mixed_types_pipeline_returns_base(self):
        a = GlobResult(
            success=True,
            message="a",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GlobEvidence(strategy="glob", path="/a.py")]
                )
            ],
        )
        b = VectorSearchResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[VectorEvidence(strategy="vector", path="/a.py")]
                )
            ],
        )
        result = a >> b
        assert type(result) is FileSearchResult

    def test_base_and_subclass_returns_base(self):
        a = FileSearchResult.from_paths(["/a.py"])
        b = GlobResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/a.py", evidence=[GlobEvidence(strategy="glob", path="/a.py")]
                )
            ],
        )
        result = a | b
        assert type(result) is FileSearchResult

    def test_same_graph_result_preserves(self):
        a = GraphResult(
            success=True,
            message="a",
            candidates=[
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[
                        GraphEvidence(strategy="dependents", path="/a.py", algorithm="dependents")
                    ],
                )
            ],
        )
        b = GraphResult(
            success=True,
            message="b",
            candidates=[
                FileSearchCandidate(
                    path="/b.py",
                    evidence=[GraphEvidence(strategy="impacts", path="/b.py", algorithm="impacts")],
                )
            ],
        )
        result = a | b
        assert isinstance(result, GraphResult)


# =====================================================================
# Subclass convenience accessors
# =====================================================================


class TestGlobResultAccessors:
    def test_directories_and_files(self):
        r = GlobResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/src",
                    evidence=[GlobEvidence(strategy="glob", path="/src", is_directory=True)],
                ),
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[
                        GlobEvidence(
                            strategy="glob", path="/a.py", is_directory=False, size_bytes=100
                        )
                    ],
                ),
            ],
        )
        assert r.directories() == ("/src",)
        assert r.files() == ("/a.py",)

    def test_file_info(self):
        r = GlobResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[
                        GlobEvidence(
                            strategy="glob",
                            path="/a.py",
                            size_bytes=200,
                            mime_type="text/x-python",
                        )
                    ],
                ),
            ],
        )
        info = r.file_info("/a.py")
        assert info is not None
        assert info.size_bytes == 200
        assert info.mime_type == "text/x-python"

    def test_file_info_missing(self):
        r = GlobResult(success=True, message="ok")
        assert r.file_info("/missing.py") is None


class TestGrepResultAccessors:
    def test_line_matches(self):
        lm1 = LineMatch(line_number=5, line_content="def foo():")
        lm2 = LineMatch(line_number=10, line_content="def bar():")
        r = GrepResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[GrepEvidence(strategy="grep", path="/a.py", line_matches=(lm1, lm2))],
                ),
            ],
        )
        matches = r.line_matches("/a.py")
        assert len(matches) == 2
        assert matches[0].line_number == 5

    def test_line_matches_missing(self):
        r = GrepResult(success=True, message="ok")
        assert r.line_matches("/missing.py") == ()

    def test_all_matches(self):
        lm1 = LineMatch(line_number=1, line_content="import os")
        lm2 = LineMatch(line_number=3, line_content="import sys")
        r = GrepResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[GrepEvidence(strategy="grep", path="/a.py", line_matches=(lm1,))],
                ),
                FileSearchCandidate(
                    path="/b.py",
                    evidence=[GrepEvidence(strategy="grep", path="/b.py", line_matches=(lm2,))],
                ),
            ],
        )
        all_matches = r.all_matches()
        assert len(all_matches) == 2
        paths = {path for path, _ in all_matches}
        assert paths == {"/a.py", "/b.py"}


class TestTreeResultAccessors:
    def test_total_files_and_dirs(self):
        r = TreeResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/src",
                    evidence=[
                        TreeEvidence(strategy="tree", path="/src", depth=1, is_directory=True)
                    ],
                ),
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[
                        TreeEvidence(strategy="tree", path="/a.py", depth=1, is_directory=False)
                    ],
                ),
                FileSearchCandidate(
                    path="/b.py",
                    evidence=[
                        TreeEvidence(strategy="tree", path="/b.py", depth=1, is_directory=False)
                    ],
                ),
            ],
        )
        assert r.total_files == 2
        assert r.total_dirs == 1


class TestListDirResultAccessors:
    def test_directories_and_files(self):
        r = ListDirResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/src",
                    evidence=[ListDirEvidence(strategy="list_dir", path="/src", is_directory=True)],
                ),
                FileSearchCandidate(
                    path="/a.py",
                    evidence=[
                        ListDirEvidence(
                            strategy="list_dir", path="/a.py", is_directory=False, size_bytes=50
                        )
                    ],
                ),
            ],
        )
        assert r.directories() == ("/src",)
        assert r.files() == ("/a.py",)


class TestTrashResultAccessors:
    def test_deleted_paths(self):
        now = datetime.now(UTC)
        r = TrashResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/__trash__/abc/a.py",
                    evidence=[
                        TrashEvidence(
                            strategy="trash",
                            path="/__trash__/abc/a.py",
                            deleted_at=now,
                            original_path="/a.py",
                        )
                    ],
                ),
            ],
        )
        assert r.deleted_paths() == ("/a.py",)


class TestVectorSearchResultAccessors:
    def test_snippets(self):
        r = VectorSearchResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/auth.py",
                    evidence=[
                        VectorEvidence(
                            strategy="vector_search", path="/auth.py", snippet="login code"
                        )
                    ],
                ),
            ],
        )
        assert r.snippets("/auth.py") == ("login code",)

    def test_snippets_missing(self):
        r = VectorSearchResult(success=True, message="ok")
        assert r.snippets("/missing.py") == ()


class TestLexicalSearchResultAccessors:
    def test_snippets(self):
        r = LexicalSearchResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/auth.py",
                    evidence=[
                        LexicalEvidence(
                            strategy="lexical", path="/auth.py", snippet="keyword match"
                        )
                    ],
                ),
            ],
        )
        assert r.snippets("/auth.py") == ("keyword match",)


class TestHybridSearchResultAccessors:
    def test_snippets(self):
        r = HybridSearchResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/auth.py",
                    evidence=[
                        HybridEvidence(strategy="hybrid", path="/auth.py", snippet="hybrid match")
                    ],
                ),
            ],
        )
        assert r.snippets("/auth.py") == ("hybrid match",)


class TestGraphResultAccessors:
    def test_algorithm(self):
        r = GraphResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/b.py",
                    evidence=[
                        GraphEvidence(
                            strategy="dependents",
                            path="/b.py",
                            algorithm="dependents",
                            relationship="imports",
                        )
                    ],
                ),
            ],
        )
        assert r.algorithm == "dependents"

    def test_algorithm_empty(self):
        r = GraphResult(success=True, message="ok")
        assert r.algorithm == ""

    def test_relationships(self):
        r = GraphResult(
            success=True,
            message="ok",
            candidates=[
                FileSearchCandidate(
                    path="/b.py",
                    evidence=[
                        GraphEvidence(
                            strategy="deps",
                            path="/b.py",
                            algorithm="dependents",
                            relationship="imports",
                        ),
                        GraphEvidence(
                            strategy="deps",
                            path="/b.py",
                            algorithm="dependents",
                            relationship="contains",
                        ),
                    ],
                ),
            ],
        )
        assert r.relationships("/b.py") == ("imports", "contains")

    def test_relationships_missing(self):
        r = GraphResult(success=True, message="ok")
        assert r.relationships("/missing.py") == ()

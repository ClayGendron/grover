"""Tests for result type hierarchy: FileOperationResult, FileSearchResult, Evidence, set algebra."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.ref import Ref
from grover.results import (
    AncestorsResult,
    BetweennessResult,
    ClosenessResult,
    CommonNeighborsResult,
    ConnectionCandidate,
    DegreeResult,
    DeleteResult,
    DescendantsResult,
    DiffVersionsResult,
    EditResult,
    EgoGraphResult,
    Evidence,
    FileCandidate,
    FileOperationResult,
    FileSearchResult,
    GetVersionContentResult,
    GlobEvidence,
    GlobResult,
    GraphEvidence,
    GraphResult,
    GrepEvidence,
    GrepResult,
    HarmonicResult,
    HasPathResult,
    HitsResult,
    HybridEvidence,
    HybridSearchResult,
    KatzResult,
    LexicalEvidence,
    LexicalSearchResult,
    LineMatch,
    ListDirEvidence,
    ListDirResult,
    MeetingSubgraphResult,
    MkdirResult,
    MoveResult,
    PageRankResult,
    PredecessorsResult,
    ReadResult,
    RestoreResult,
    ShortestPathResult,
    SubgraphSearchResult,
    SuccessorsResult,
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
        e = Evidence(operation="glob")
        assert e.operation == "glob"

    def test_evidence_no_path(self):
        e = Evidence(operation="test")
        assert not hasattr(e, "path") or "path" not in e.__dataclass_fields__

    def test_evidence_score_default(self):
        e = Evidence(operation="test")
        assert e.score == 0.0

    def test_evidence_query_args_default(self):
        e = Evidence(operation="test")
        assert e.query_args == {}

    def test_evidence_with_score_and_query_args(self):
        e = Evidence(operation="search", score=0.85, query_args={"k": 10, "q": "hello"})
        assert e.score == 0.85
        assert e.query_args == {"k": 10, "q": "hello"}

    def test_glob_evidence(self):
        e = GlobEvidence(operation="glob", is_directory=False, size_bytes=100)
        assert isinstance(e, Evidence)
        assert e.is_directory is False
        assert e.size_bytes == 100

    def test_grep_evidence(self):
        lm = LineMatch(line_number=5, line_content="def foo():")
        e = GrepEvidence(operation="grep", line_matches=(lm,))
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
        e = TreeEvidence(operation="tree", depth=2, is_directory=True)
        assert e.depth == 2
        assert e.is_directory is True

    def test_listdir_evidence(self):
        e = ListDirEvidence(operation="list_dir", is_directory=False, size_bytes=50)
        assert e.size_bytes == 50

    def test_trash_evidence(self):
        now = datetime.now(UTC)
        e = TrashEvidence(operation="trash", deleted_at=now, original_path="/a.py")
        assert e.deleted_at == now
        assert e.original_path == "/a.py"

    def test_vector_evidence(self):
        e = VectorEvidence(operation="vector_search", snippet="auth logic")
        assert e.snippet == "auth logic"

    def test_lexical_evidence(self):
        e = LexicalEvidence(operation="lexical_search", snippet="login")
        assert e.snippet == "login"

    def test_hybrid_evidence(self):
        e = HybridEvidence(operation="hybrid_search", snippet="mixed")
        assert e.snippet == "mixed"

    def test_graph_evidence(self):
        e = GraphEvidence(
            operation="predecessors", algorithm="predecessors", relationship="imports"
        )
        assert e.algorithm == "predecessors"
        assert e.relationship == "imports"


# =====================================================================
# FileCandidate
# =====================================================================


class TestFileCandidate:
    def test_file_candidate_scores(self):
        fc = FileCandidate(
            path="/a.py",
            evidence=[
                Evidence(operation="pagerank", score=0.9),
                Evidence(operation="vector_search", score=0.7),
                Evidence(operation="grep"),  # score=0.0 → excluded
            ],
        )
        assert fc.scores == {"pagerank": 0.9, "vector_search": 0.7}

    def test_file_candidate_scores_empty(self):
        fc = FileCandidate(path="/a.py", evidence=[])
        assert fc.scores == {}


# =====================================================================
# ConnectionCandidate
# =====================================================================


class TestConnectionCandidate:
    def test_connection_candidate_frozen(self):
        cc = ConnectionCandidate(
            source_path="/a.py",
            target_path="/b.py",
            connection_type="imports",
        )
        assert cc.source_path == "/a.py"
        assert cc.target_path == "/b.py"
        assert cc.connection_type == "imports"

    def test_connection_candidate_path_format(self):
        cc = ConnectionCandidate(
            source_path="/a.py",
            target_path="/b.py",
            connection_type="imports",
        )
        assert cc.path == "/a.py[imports]/b.py"

    def test_connection_candidate_default_weight(self):
        cc = ConnectionCandidate(
            source_path="/a.py",
            target_path="/b.py",
            connection_type="imports",
        )
        assert cc.weight == 1.0

    def test_connection_candidate_scores(self):
        cc = ConnectionCandidate(
            source_path="/a.py",
            target_path="/b.py",
            connection_type="imports",
            evidence=[
                Evidence(operation="pagerank", score=0.9),
                Evidence(operation="graph", score=0.5),
            ],
        )
        assert cc.scores == {"pagerank": 0.9, "graph": 0.5}


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
            FileCandidate(path="/a.py", evidence=[Evidence(operation="glob")]),
            FileCandidate(path="/b.py", evidence=[Evidence(operation="glob")]),
        ]
        r = FileSearchResult(success=True, message="2 paths", file_candidates=candidates)
        assert len(r) == 2
        assert r  # non-empty is truthy
        assert "/a.py" in r
        assert "/c.py" not in r

    def test_paths_property(self):
        candidates = [
            FileCandidate(path="/x.py", evidence=[]),
            FileCandidate(path="/y.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", file_candidates=candidates)
        assert set(r.paths) == {"/x.py", "/y.py"}

    def test_iteration(self):
        candidates = [
            FileCandidate(path="/a.py", evidence=[]),
            FileCandidate(path="/b.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", file_candidates=candidates)
        assert set(r) == {"/a.py", "/b.py"}

    def test_explain(self):
        e1 = Evidence(operation="glob")
        e2 = Evidence(operation="grep")
        candidates = [FileCandidate(path="/a.py", evidence=[e1, e2])]
        r = FileSearchResult(success=True, message="ok", file_candidates=candidates)
        chain = r.explain("/a.py")
        assert len(chain) == 2
        assert chain[0].operation == "glob"
        assert chain[1].operation == "grep"

    def test_explain_missing_path(self):
        r = FileSearchResult(success=True, message="ok")
        assert r.explain("/missing.py") == []

    def test_to_refs(self):
        candidates = [
            FileCandidate(path="/a.py", evidence=[]),
            FileCandidate(path="/b.py", evidence=[]),
        ]
        r = FileSearchResult(success=True, message="ok", file_candidates=candidates)
        refs = r.to_refs()
        assert len(refs) == 2
        assert all(isinstance(ref, Ref) for ref in refs)
        paths = {ref.path for ref in refs}
        assert paths == {"/a.py", "/b.py"}

    def test_from_paths(self):
        r = FileSearchResult.from_paths(["/a.py", "/b.py"], operation="custom")
        assert len(r) == 2
        assert "/a.py" in r
        assert r.explain("/a.py")[0].operation == "custom"

    def test_from_refs(self):
        refs = [Ref(path="/a.py"), Ref(path="/b.py")]
        r = FileSearchResult.from_refs(refs, operation="ref")
        assert len(r) == 2
        assert "/a.py" in r

    def test_failed_result_is_falsy(self):
        candidates = [FileCandidate(path="/a.py", evidence=[])]
        r = FileSearchResult(success=False, message="fail", file_candidates=candidates)
        assert not r  # failed is falsy even with entries

    def test_connection_candidates_default_empty(self):
        r = FileSearchResult(success=True, message="ok")
        assert r.connection_candidates == []

    def test_connection_paths_property(self):
        cc1 = ConnectionCandidate(
            source_path="/a.py", target_path="/b.py", connection_type="imports"
        )
        cc2 = ConnectionCandidate(source_path="/c.py", target_path="/d.py", connection_type="calls")
        r = FileSearchResult(success=True, message="ok", connection_candidates=[cc1, cc2])
        assert r.connection_paths == ("/a.py[imports]/b.py", "/c.py[calls]/d.py")

    def test_len_counts_file_candidates_only(self):
        fc = [FileCandidate(path="/a.py", evidence=[])]
        cc = [
            ConnectionCandidate(source_path="/a.py", target_path="/b.py", connection_type="imports")
        ]
        r = FileSearchResult(
            success=True, message="ok", file_candidates=fc, connection_candidates=cc
        )
        assert len(r) == 1  # only file_candidates count

    def test_bool_checks_file_candidates_only(self):
        cc = [
            ConnectionCandidate(source_path="/a.py", target_path="/b.py", connection_type="imports")
        ]
        r = FileSearchResult(success=True, message="ok", connection_candidates=cc)
        assert not r  # no file_candidates → falsy

    def test_iter_yields_file_paths_only(self):
        fc = [FileCandidate(path="/a.py", evidence=[])]
        cc = [
            ConnectionCandidate(source_path="/c.py", target_path="/d.py", connection_type="imports")
        ]
        r = FileSearchResult(
            success=True, message="ok", file_candidates=fc, connection_candidates=cc
        )
        assert list(r) == ["/a.py"]


# =====================================================================
# Set algebra
# =====================================================================


class TestSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> FileSearchResult:
        entries = {p: [Evidence(operation=operation)] for p in paths}
        return FileSearchResult(
            success=True,
            message="ok",
            file_candidates=FileSearchResult._dict_to_candidates(entries),
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
        a = self._make(["/a.py"], operation="glob")
        b = self._make(["/a.py"], operation="grep")
        result = a | b
        chain = result.explain("/a.py")
        operations = {e.operation for e in chain}
        assert operations == {"glob", "grep"}

    def test_evidence_merged_on_intersection(self):
        a = self._make(["/a.py"], operation="glob")
        b = self._make(["/a.py"], operation="grep")
        result = a & b
        chain = result.explain("/a.py")
        operations = {e.operation for e in chain}
        assert operations == {"glob", "grep"}

    def test_evidence_lhs_only_on_difference(self):
        a = self._make(["/a.py", "/b.py"], operation="glob")
        b = self._make(["/b.py"], operation="grep")
        result = a - b
        assert "/a.py" in result
        chain = result.explain("/a.py")
        assert all(e.operation == "glob" for e in chain)

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
# Set algebra with connections
# =====================================================================


class TestSetAlgebraConnections:
    def _make_conn(
        self, src: str, tgt: str, conn_type: str = "imports", operation: str = "test"
    ) -> ConnectionCandidate:
        return ConnectionCandidate(
            source_path=src,
            target_path=tgt,
            connection_type=conn_type,
            evidence=[Evidence(operation=operation)],
        )

    def test_union_merges_connections(self):
        cc1 = self._make_conn("/a.py", "/b.py")
        cc2 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connection_candidates=[cc1])
        b = FileSearchResult(success=True, message="b", connection_candidates=[cc2])
        result = a | b
        assert len(result.connection_candidates) == 2

    def test_intersection_connections(self):
        cc1 = self._make_conn("/a.py", "/b.py", operation="glob")
        cc2 = self._make_conn("/a.py", "/b.py", operation="grep")
        cc3 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connection_candidates=[cc1, cc3])
        b = FileSearchResult(success=True, message="b", connection_candidates=[cc2])
        result = a & b
        assert len(result.connection_candidates) == 1
        assert result.connection_candidates[0].path == "/a.py[imports]/b.py"

    def test_difference_connections(self):
        cc1 = self._make_conn("/a.py", "/b.py")
        cc2 = self._make_conn("/a.py", "/b.py")
        cc3 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connection_candidates=[cc1, cc3])
        b = FileSearchResult(success=True, message="b", connection_candidates=[cc2])
        result = a - b
        assert len(result.connection_candidates) == 1
        assert result.connection_candidates[0].path == "/c.py[imports]/d.py"

    def test_pipeline_connections(self):
        cc1 = self._make_conn("/a.py", "/b.py", operation="glob")
        cc2 = self._make_conn("/a.py", "/b.py", operation="grep")
        a = FileSearchResult(success=True, message="a", connection_candidates=[cc1])
        b = FileSearchResult(success=True, message="b", connection_candidates=[cc2])
        result = a >> b
        assert len(result.connection_candidates) == 1

    def test_connection_evidence_merged_on_union(self):
        cc1 = self._make_conn("/a.py", "/b.py", operation="glob")
        cc2 = self._make_conn("/a.py", "/b.py", operation="grep")
        a = FileSearchResult(success=True, message="a", connection_candidates=[cc1])
        b = FileSearchResult(success=True, message="b", connection_candidates=[cc2])
        result = a | b
        assert len(result.connection_candidates) == 1
        ops = {e.operation for e in result.connection_candidates[0].evidence}
        assert ops == {"glob", "grep"}


# =====================================================================
# Same type preserves subclass, mixed returns base
# =====================================================================


class TestSubclassPreservation:
    def test_same_type_union_preserves(self):
        a = GlobResult(
            success=True,
            message="a",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GlobEvidence(operation="glob")])
            ],
        )
        b = GlobResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(path="/b.py", evidence=[GlobEvidence(operation="glob")])
            ],
        )
        result = a | b
        assert isinstance(result, GlobResult)

    def test_same_type_intersection_preserves(self):
        a = GrepResult(
            success=True,
            message="a",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GrepEvidence(operation="grep")])
            ],
        )
        b = GrepResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GrepEvidence(operation="grep")])
            ],
        )
        result = a & b
        assert isinstance(result, GrepResult)

    def test_mixed_types_returns_base(self):
        a = GlobResult(
            success=True,
            message="a",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GlobEvidence(operation="glob")])
            ],
        )
        b = GrepResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GrepEvidence(operation="grep")])
            ],
        )
        result = a & b
        assert type(result) is FileSearchResult

    def test_mixed_types_pipeline_returns_base(self):
        a = GlobResult(
            success=True,
            message="a",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GlobEvidence(operation="glob")])
            ],
        )
        b = VectorSearchResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[VectorEvidence(operation="vector")])
            ],
        )
        result = a >> b
        assert type(result) is FileSearchResult

    def test_base_and_subclass_returns_base(self):
        a = FileSearchResult.from_paths(["/a.py"])
        b = GlobResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GlobEvidence(operation="glob")])
            ],
        )
        result = a | b
        assert type(result) is FileSearchResult

    def test_same_graph_result_preserves(self):
        a = GraphResult(
            success=True,
            message="a",
            file_candidates=[
                FileCandidate(
                    path="/a.py",
                    evidence=[GraphEvidence(operation="predecessors", algorithm="predecessors")],
                )
            ],
        )
        b = GraphResult(
            success=True,
            message="b",
            file_candidates=[
                FileCandidate(
                    path="/b.py",
                    evidence=[GraphEvidence(operation="successors", algorithm="successors")],
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
            file_candidates=[
                FileCandidate(
                    path="/src",
                    evidence=[GlobEvidence(operation="glob", is_directory=True)],
                ),
                FileCandidate(
                    path="/a.py",
                    evidence=[GlobEvidence(operation="glob", is_directory=False, size_bytes=100)],
                ),
            ],
        )
        assert r.directories() == ("/src",)
        assert r.files() == ("/a.py",)

    def test_file_info(self):
        r = GlobResult(
            success=True,
            message="ok",
            file_candidates=[
                FileCandidate(
                    path="/a.py",
                    evidence=[
                        GlobEvidence(
                            operation="glob",
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
            file_candidates=[
                FileCandidate(
                    path="/a.py",
                    evidence=[GrepEvidence(operation="grep", line_matches=(lm1, lm2))],
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
            file_candidates=[
                FileCandidate(
                    path="/a.py",
                    evidence=[GrepEvidence(operation="grep", line_matches=(lm1,))],
                ),
                FileCandidate(
                    path="/b.py",
                    evidence=[GrepEvidence(operation="grep", line_matches=(lm2,))],
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
            file_candidates=[
                FileCandidate(
                    path="/src",
                    evidence=[TreeEvidence(operation="tree", depth=1, is_directory=True)],
                ),
                FileCandidate(
                    path="/a.py",
                    evidence=[TreeEvidence(operation="tree", depth=1, is_directory=False)],
                ),
                FileCandidate(
                    path="/b.py",
                    evidence=[TreeEvidence(operation="tree", depth=1, is_directory=False)],
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
            file_candidates=[
                FileCandidate(
                    path="/src",
                    evidence=[ListDirEvidence(operation="list_dir", is_directory=True)],
                ),
                FileCandidate(
                    path="/a.py",
                    evidence=[
                        ListDirEvidence(operation="list_dir", is_directory=False, size_bytes=50)
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
            file_candidates=[
                FileCandidate(
                    path="/__trash__/abc/a.py",
                    evidence=[
                        TrashEvidence(
                            operation="trash",
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
            file_candidates=[
                FileCandidate(
                    path="/auth.py",
                    evidence=[VectorEvidence(operation="vector_search", snippet="login code")],
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
            file_candidates=[
                FileCandidate(
                    path="/auth.py",
                    evidence=[LexicalEvidence(operation="lexical", snippet="keyword match")],
                ),
            ],
        )
        assert r.snippets("/auth.py") == ("keyword match",)


class TestHybridSearchResultAccessors:
    def test_snippets(self):
        r = HybridSearchResult(
            success=True,
            message="ok",
            file_candidates=[
                FileCandidate(
                    path="/auth.py",
                    evidence=[HybridEvidence(operation="hybrid", snippet="hybrid match")],
                ),
            ],
        )
        assert r.snippets("/auth.py") == ("hybrid match",)


class TestGraphResultAccessors:
    def test_algorithm(self):
        r = GraphResult(
            success=True,
            message="ok",
            file_candidates=[
                FileCandidate(
                    path="/b.py",
                    evidence=[
                        GraphEvidence(
                            operation="predecessors",
                            algorithm="predecessors",
                            relationship="imports",
                        )
                    ],
                ),
            ],
        )
        assert r.algorithm == "predecessors"

    def test_algorithm_empty(self):
        r = GraphResult(success=True, message="ok")
        assert r.algorithm == ""

    def test_relationships(self):
        r = GraphResult(
            success=True,
            message="ok",
            file_candidates=[
                FileCandidate(
                    path="/b.py",
                    evidence=[
                        GraphEvidence(
                            operation="predecessors",
                            algorithm="predecessors",
                            relationship="imports",
                        ),
                        GraphEvidence(
                            operation="predecessors",
                            algorithm="predecessors",
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


# =====================================================================
# DiffVersionsResult
# =====================================================================


class TestDiffVersionsResult:
    def test_diff_versions_result_defaults(self):
        r = DiffVersionsResult()
        assert isinstance(r, FileOperationResult)
        assert r.version_a == 0
        assert r.version_b == 0
        assert r.diff == ""
        assert r.content_a == ""
        assert r.content_b == ""
        assert r.success is True

    def test_diff_versions_result_with_data(self):
        r = DiffVersionsResult(
            success=True,
            path="/a.py",
            version_a=1,
            version_b=3,
            diff="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new",
            content_a="old",
            content_b="new",
        )
        assert r.version_a == 1
        assert r.version_b == 3
        assert r.diff.startswith("---")
        assert r.content_a == "old"
        assert r.content_b == "new"
        assert r.path == "/a.py"


# =====================================================================
# GraphResult set algebra with connection candidates
# =====================================================================


class TestGraphResultSetAlgebra:
    def test_graph_result_set_algebra_preserves_connections(self):
        r1 = GraphResult(
            success=True,
            message="2 node(s)",
            file_candidates=[
                FileCandidate(path="/a.py", evidence=[GraphEvidence(operation="op1")]),
                FileCandidate(path="/b.py", evidence=[GraphEvidence(operation="op1")]),
            ],
            connection_candidates=[
                ConnectionCandidate(
                    source_path="/a.py",
                    target_path="/b.py",
                    connection_type="imports",
                    evidence=[GraphEvidence(operation="op1")],
                )
            ],
        )
        r2 = GraphResult(
            success=True,
            message="2 node(s)",
            file_candidates=[
                FileCandidate(path="/b.py", evidence=[GraphEvidence(operation="op2")]),
                FileCandidate(path="/c.py", evidence=[GraphEvidence(operation="op2")]),
            ],
            connection_candidates=[
                ConnectionCandidate(
                    source_path="/b.py",
                    target_path="/c.py",
                    connection_type="calls",
                    evidence=[GraphEvidence(operation="op2")],
                )
            ],
        )
        union = r1 | r2
        assert len(union.connection_candidates) == 2
        conn_paths = {cc.path for cc in union.connection_candidates}
        assert "/a.py[imports]/b.py" in conn_paths
        assert "/b.py[calls]/c.py" in conn_paths


# =====================================================================
# HitsResult convenience methods
# =====================================================================


class TestHitsResult:
    def test_hits_result_hub_and_authority_scores(self):
        r = HitsResult(
            success=True,
            message="ok",
            file_candidates=[
                FileCandidate(
                    path="/a.py",
                    evidence=[
                        GraphEvidence(operation="hits_authority", algorithm="hits", score=0.8),
                        GraphEvidence(operation="hits_hub", algorithm="hits", score=0.3),
                    ],
                ),
                FileCandidate(
                    path="/b.py",
                    evidence=[
                        GraphEvidence(operation="hits_authority", algorithm="hits", score=0.2),
                        GraphEvidence(operation="hits_hub", algorithm="hits", score=0.9),
                    ],
                ),
            ],
        )
        assert r.authority_score("/a.py") == 0.8
        assert r.hub_score("/a.py") == 0.3
        assert r.authority_score("/b.py") == 0.2
        assert r.hub_score("/b.py") == 0.9
        # Missing path returns 0.0
        assert r.authority_score("/missing.py") == 0.0
        assert r.hub_score("/missing.py") == 0.0


# =====================================================================
# Typed graph result subclasses inherit GraphResult
# =====================================================================


class TestTypedGraphResultSubclasses:
    def test_typed_graph_results_inherit_graph_result(self):
        subclasses = [
            PredecessorsResult,
            SuccessorsResult,
            AncestorsResult,
            DescendantsResult,
            ShortestPathResult,
            HasPathResult,
            SubgraphSearchResult,
            MeetingSubgraphResult,
            EgoGraphResult,
            PageRankResult,
            HitsResult,
            BetweennessResult,
            ClosenessResult,
            HarmonicResult,
            KatzResult,
            DegreeResult,
            CommonNeighborsResult,
        ]
        for cls in subclasses:
            instance = cls(success=True, message="test")
            assert isinstance(instance, GraphResult), f"{cls.__name__} not a GraphResult"
            assert isinstance(instance, FileSearchResult), f"{cls.__name__} not a FileSearchResult"

"""Tests for result type hierarchy: FileOperationResult, FileSearchResult, Evidence, set algebra.

These tests exercise the internal types in ``grover.models.internal``
(File, FileConnection, FileSearchResult, FileOperationResult, Evidence).
"""

from __future__ import annotations

from datetime import UTC, datetime

from grover.models.internal.evidence import (
    Evidence,
    GlobEvidence,
    GraphRelationshipEvidence,
    GrepEvidence,
    HybridEvidence,
    LexicalEvidence,
    LineMatch,
    ListDirEvidence,
    TrashEvidence,
    VectorEvidence,
    VersionEvidence,
)
from grover.models.internal.ref import File, FileConnection, Ref
from grover.models.internal.results import FileOperationResult, FileSearchResult

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


# =====================================================================
# Evidence types
# =====================================================================


class TestEvidence:
    def test_base_frozen(self):
        e = Evidence(operation="glob")
        assert e.operation == "glob"

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
        from grover.models.internal.evidence import TreeEvidence

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

    def test_graph_relationship_evidence(self):
        e = GraphRelationshipEvidence(operation="predecessors", paths=["/a.py", "/b.py"])
        assert e.operation == "predecessors"
        assert e.paths == ["/a.py", "/b.py"]


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
        files = [
            File(path="/a.py", evidence=[Evidence(operation="glob")]),
            File(path="/b.py", evidence=[Evidence(operation="glob")]),
        ]
        r = FileSearchResult(success=True, message="2 paths", files=files)
        assert len(r) == 2
        assert r  # non-empty is truthy
        assert "/a.py" in r
        assert "/c.py" not in r

    def test_paths_property(self):
        files = [
            File(path="/x.py"),
            File(path="/y.py"),
        ]
        r = FileSearchResult(success=True, message="ok", files=files)
        assert set(r.paths) == {"/x.py", "/y.py"}

    def test_iteration(self):
        files = [
            File(path="/a.py"),
            File(path="/b.py"),
        ]
        r = FileSearchResult(success=True, message="ok", files=files)
        assert set(r) == {"/a.py", "/b.py"}

    def test_explain(self):
        e1 = Evidence(operation="glob")
        e2 = Evidence(operation="grep")
        files = [File(path="/a.py", evidence=[e1, e2])]
        r = FileSearchResult(success=True, message="ok", files=files)
        chain = r.explain("/a.py")
        assert len(chain) == 2
        assert chain[0].operation == "glob"
        assert chain[1].operation == "grep"

    def test_explain_missing_path(self):
        r = FileSearchResult(success=True, message="ok")
        assert r.explain("/missing.py") == []

    def test_to_refs(self):
        files = [
            File(path="/a.py"),
            File(path="/b.py"),
        ]
        r = FileSearchResult(success=True, message="ok", files=files)
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
        files = [File(path="/a.py")]
        r = FileSearchResult(success=False, message="fail", files=files)
        assert not r  # failed is falsy even with entries

    def test_connections_default_empty(self):
        r = FileSearchResult(success=True, message="ok")
        assert r.connections == []

    def test_connection_paths_property(self):
        conn1 = FileConnection(path="/a.py[imports]/b.py", source_path="/a.py", target_path="/b.py", type="imports")
        conn2 = FileConnection(path="/c.py[calls]/d.py", source_path="/c.py", target_path="/d.py", type="calls")
        r = FileSearchResult(success=True, message="ok", connections=[conn1, conn2])
        assert r.connection_paths == ("/a.py[imports]/b.py", "/c.py[calls]/d.py")

    def test_len_counts_files_only(self):
        files = [File(path="/a.py")]
        conns = [FileConnection(path="/a.py[imports]/b.py", source_path="/a.py", target_path="/b.py", type="imports")]
        r = FileSearchResult(success=True, message="ok", files=files, connections=conns)
        assert len(r) == 1  # only files count

    def test_bool_checks_files_only(self):
        conns = [FileConnection(path="/a.py[imports]/b.py", source_path="/a.py", target_path="/b.py", type="imports")]
        r = FileSearchResult(success=True, message="ok", connections=conns)
        assert not r  # no files → falsy

    def test_iter_yields_file_paths_only(self):
        files = [File(path="/a.py")]
        conns = [FileConnection(path="/c.py[imports]/d.py", source_path="/c.py", target_path="/d.py", type="imports")]
        r = FileSearchResult(success=True, message="ok", files=files, connections=conns)
        assert list(r) == ["/a.py"]


# =====================================================================
# Set algebra
# =====================================================================


class TestSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> FileSearchResult:
        return FileSearchResult(
            success=True,
            message="ok",
            files=[File(path=p, evidence=[Evidence(operation=operation)]) for p in paths],
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
    def _make_conn(self, src: str, tgt: str, conn_type: str = "imports", operation: str = "test") -> FileConnection:
        return FileConnection(
            path=f"{src}[{conn_type}]{tgt}",
            source_path=src,
            target_path=tgt,
            type=conn_type,
            evidence=[Evidence(operation=operation)],
        )

    def test_union_merges_connections(self):
        conn1 = self._make_conn("/a.py", "/b.py")
        conn2 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connections=[conn1])
        b = FileSearchResult(success=True, message="b", connections=[conn2])
        result = a | b
        assert len(result.connections) == 2

    def test_intersection_connections(self):
        conn1 = self._make_conn("/a.py", "/b.py", operation="glob")
        conn2 = self._make_conn("/a.py", "/b.py", operation="grep")
        conn3 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connections=[conn1, conn3])
        b = FileSearchResult(success=True, message="b", connections=[conn2])
        result = a & b
        assert len(result.connections) == 1
        c = result.connections[0]
        assert f"{c.source_path}[{c.type}]{c.target_path}" == "/a.py[imports]/b.py"

    def test_difference_connections(self):
        conn1 = self._make_conn("/a.py", "/b.py")
        conn2 = self._make_conn("/a.py", "/b.py")
        conn3 = self._make_conn("/c.py", "/d.py")
        a = FileSearchResult(success=True, message="a", connections=[conn1, conn3])
        b = FileSearchResult(success=True, message="b", connections=[conn2])
        result = a - b
        assert len(result.connections) == 1
        c = result.connections[0]
        assert f"{c.source_path}[{c.type}]{c.target_path}" == "/c.py[imports]/d.py"

    def test_pipeline_connections(self):
        conn1 = self._make_conn("/a.py", "/b.py", operation="glob")
        conn2 = self._make_conn("/a.py", "/b.py", operation="grep")
        a = FileSearchResult(success=True, message="a", connections=[conn1])
        b = FileSearchResult(success=True, message="b", connections=[conn2])
        result = a >> b
        assert len(result.connections) == 1

    def test_connection_evidence_merged_on_union(self):
        conn1 = self._make_conn("/a.py", "/b.py", operation="glob")
        conn2 = self._make_conn("/a.py", "/b.py", operation="grep")
        a = FileSearchResult(success=True, message="a", connections=[conn1])
        b = FileSearchResult(success=True, message="b", connections=[conn2])
        result = a | b
        assert len(result.connections) == 1
        ops = {e.operation for e in result.connections[0].evidence}
        assert ops == {"glob", "grep"}


# =====================================================================
# Graph-style set algebra with connections
# =====================================================================


class TestGraphStyleSetAlgebra:
    def test_set_algebra_preserves_connections(self):
        r1 = FileSearchResult(
            success=True,
            message="2 node(s)",
            files=[
                File(path="/a.py", evidence=[GraphRelationshipEvidence(operation="op1")]),
                File(path="/b.py", evidence=[GraphRelationshipEvidence(operation="op1")]),
            ],
            connections=[
                FileConnection(
                    path="/a.py[imports]/b.py",
                    source_path="/a.py",
                    target_path="/b.py",
                    type="imports",
                    evidence=[GraphRelationshipEvidence(operation="op1")],
                )
            ],
        )
        r2 = FileSearchResult(
            success=True,
            message="2 node(s)",
            files=[
                File(path="/b.py", evidence=[GraphRelationshipEvidence(operation="op2")]),
                File(path="/c.py", evidence=[GraphRelationshipEvidence(operation="op2")]),
            ],
            connections=[
                FileConnection(
                    path="/b.py[calls]/c.py",
                    source_path="/b.py",
                    target_path="/c.py",
                    type="calls",
                    evidence=[GraphRelationshipEvidence(operation="op2")],
                )
            ],
        )
        union = r1 | r2
        assert len(union.connections) == 2
        conn_paths = union.connection_paths
        assert "/a.py[imports]/b.py" in conn_paths
        assert "/b.py[calls]/c.py" in conn_paths


# =====================================================================
# Version evidence
# =====================================================================


class TestVersionEvidence:
    def test_version_evidence_fields(self):
        now = datetime.now(UTC)
        ve = VersionEvidence(
            operation="version",
            version=1,
            content_hash="abc",
            size_bytes=10,
            created_at=now,
        )
        assert ve.version == 1
        assert ve.content_hash == "abc"
        assert ve.size_bytes == 10
        assert ve.created_at == now

    def test_version_evidence_defaults(self):
        ve = VersionEvidence(operation="version")
        assert ve.created_by is None
        assert ve.version == 0
        assert ve.content_hash == ""
        assert ve.size_bytes == 0

"""Tests for internal result types (FileSearchSet, FileSearchResult, FileOperationResult)."""

from __future__ import annotations

import dataclasses

from grover.models.internal.evidence import Evidence, GlobEvidence, VectorEvidence
from grover.models.internal.ref import File, FileConnection, Ref
from grover.models.internal.results import (
    FileOperationResult,
    FileSearchResult,
    FileSearchSet,
)


class TestFileOperationResult:
    def test_construction(self):
        f = File(path="/a.py", content="x = 1")
        r = FileOperationResult(file=f)
        assert r.file.path == "/a.py"
        assert r.file.content == "x = 1"
        assert r.message == ""
        assert r.success is True

    def test_with_message(self):
        f = File(path="/a.py")
        r = FileOperationResult(file=f, message="Created", success=True)
        assert r.message == "Created"

    def test_failure(self):
        f = File(path="/a.py")
        r = FileOperationResult(file=f, success=False, message="Not found")
        assert r.success is False
        assert r.message == "Not found"


class TestFileSearchResultConstruction:
    def test_defaults(self):
        r = FileSearchResult()
        assert r.files == []
        assert r.connections == []
        assert r.message == ""
        assert r.success is True

    def test_with_files(self):
        f1 = File(path="/a.py")
        f2 = File(path="/b.py")
        r = FileSearchResult(files=[f1, f2])
        assert len(r.files) == 2

    def test_with_connections(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        r = FileSearchResult(connections=[conn])
        assert len(r.connections) == 1


class TestFileSearchResultPaths:
    def test_paths_property(self):
        r = FileSearchResult(files=[File(path="/a.py"), File(path="/b.py"), File(path="/c.py")])
        assert r.paths == ("/a.py", "/b.py", "/c.py")

    def test_paths_empty(self):
        r = FileSearchResult()
        assert r.paths == ()


class TestFileSearchResultIteration:
    def test_len(self):
        r = FileSearchResult(files=[File(path="/a.py"), File(path="/b.py")])
        assert len(r) == 2

    def test_len_empty(self):
        assert len(FileSearchResult()) == 0

    def test_bool_true(self):
        r = FileSearchResult(files=[File(path="/a.py")])
        assert bool(r) is True

    def test_bool_false_empty(self):
        r = FileSearchResult()
        assert bool(r) is False

    def test_bool_false_not_success(self):
        r = FileSearchResult(files=[File(path="/a.py")], success=False)
        assert bool(r) is False

    def test_iter(self):
        r = FileSearchResult(files=[File(path="/a.py"), File(path="/b.py")])
        assert list(r) == ["/a.py", "/b.py"]

    def test_contains(self):
        r = FileSearchResult(files=[File(path="/a.py"), File(path="/b.py")])
        assert "/a.py" in r
        assert "/c.py" not in r


class TestFileSearchResultFromPaths:
    def test_from_paths(self):
        r = FileSearchResult.from_paths(["/a.py", "/b.py"], operation="glob")
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.success is True
        assert r.files[0].evidence[0].operation == "glob"

    def test_from_paths_default_operation(self):
        r = FileSearchResult.from_paths(["/a.py"])
        assert r.files[0].evidence[0].operation == "unknown"

    def test_from_paths_empty(self):
        r = FileSearchResult.from_paths([])
        assert len(r) == 0


class TestFileSearchResultSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> FileSearchResult:
        return FileSearchResult(files=[File(path=p, evidence=[Evidence(operation=operation)]) for p in paths])

    def test_intersection(self):
        r1 = self._make(["/a.py", "/b.py", "/c.py"], "glob")
        r2 = self._make(["/b.py", "/c.py", "/d.py"], "grep")
        result = r1 & r2
        assert set(result.paths) == {"/b.py", "/c.py"}
        # Evidence merged
        for f in result.files:
            assert len(f.evidence) == 2

    def test_intersection_empty(self):
        r1 = self._make(["/a.py"])
        r2 = self._make(["/b.py"])
        result = r1 & r2
        assert len(result) == 0

    def test_union(self):
        r1 = self._make(["/a.py", "/b.py"], "glob")
        r2 = self._make(["/b.py", "/c.py"], "grep")
        result = r1 | r2
        assert set(result.paths) == {"/a.py", "/b.py", "/c.py"}
        # Overlapping path has merged evidence
        b_file = next(f for f in result.files if f.path == "/b.py")
        assert len(b_file.evidence) == 2

    def test_difference(self):
        r1 = self._make(["/a.py", "/b.py", "/c.py"])
        r2 = self._make(["/b.py"])
        result = r1 - r2
        assert set(result.paths) == {"/a.py", "/c.py"}

    def test_pipeline(self):
        r1 = self._make(["/a.py", "/b.py", "/c.py"], "glob")
        r2 = self._make(["/b.py", "/d.py"], "grep")
        result = r1 >> r2
        assert set(result.paths) == {"/b.py"}
        b_file = result.files[0]
        assert len(b_file.evidence) == 2

    def test_not_implemented_with_non_result(self):
        r = self._make(["/a.py"])
        assert r.__and__(42) is NotImplemented
        assert r.__or__(42) is NotImplemented
        assert r.__sub__(42) is NotImplemented
        assert r.__rshift__(42) is NotImplemented

    def test_intersection_success_both_true(self):
        r1 = FileSearchResult(files=[File(path="/a.py")], success=True)
        r2 = FileSearchResult(files=[File(path="/a.py")], success=True)
        assert (r1 & r2).success is True

    def test_intersection_success_one_false(self):
        r1 = FileSearchResult(files=[File(path="/a.py")], success=True)
        r2 = FileSearchResult(files=[File(path="/a.py")], success=False)
        assert (r1 & r2).success is False

    def test_union_success_one_true(self):
        r1 = FileSearchResult(files=[File(path="/a.py")], success=True)
        r2 = FileSearchResult(success=False)
        assert (r1 | r2).success is True

    def test_connection_intersection(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            evidence=[Evidence(operation="graph")],
        )
        r1 = FileSearchResult(connections=[conn])
        r2 = FileSearchResult(connections=[conn])
        result = r1 & r2
        assert len(result.connections) == 1
        assert len(result.connections[0].evidence) == 2

    def test_connection_union(self):
        conn1 = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        conn2 = FileConnection(
            path="/c.py[imports]/d.py",
            source_path="/c.py",
            target_path="/d.py",
            type="imports",
        )
        r1 = FileSearchResult(connections=[conn1])
        r2 = FileSearchResult(connections=[conn2])
        result = r1 | r2
        assert len(result.connections) == 2


class TestFileSearchResultRebase:
    def test_rebase_files(self):
        r = FileSearchResult(files=[File(path="/a.py"), File(path="/b.py")])
        rebased = r.rebase("/mount")
        assert rebased.paths == ("/mount/a.py", "/mount/b.py")

    def test_rebase_root(self):
        r = FileSearchResult(files=[File(path="/")])
        rebased = r.rebase("/mount")
        assert rebased.paths == ("/mount",)

    def test_rebase_connections(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        r = FileSearchResult(connections=[conn])
        rebased = r.rebase("/mount")
        assert rebased.connections[0].source_path == "/mount/a.py"
        assert rebased.connections[0].target_path == "/mount/b.py"

    def test_rebase_preserves_evidence(self):
        r = FileSearchResult(files=[File(path="/a.py", evidence=[Evidence(operation="glob")])])
        rebased = r.rebase("/m")
        assert len(rebased.files[0].evidence) == 1
        assert rebased.files[0].evidence[0].operation == "glob"

    def test_rebase_preserves_original(self):
        r = FileSearchResult(files=[File(path="/a.py")])
        rebased = r.rebase("/m")
        assert r.paths == ("/a.py",)
        assert rebased.paths == ("/m/a.py",)


class TestFileSearchResultRemapPaths:
    def test_remap(self):
        r = FileSearchResult(files=[File(path="/src/a.py"), File(path="/src/b.py")])
        result = r.remap_paths(lambda p: p.replace("/src", "/dst"))
        assert set(result.paths) == {"/dst/a.py", "/dst/b.py"}

    def test_remap_merges_collisions(self):
        r = FileSearchResult(
            files=[
                File(path="/a.py", evidence=[Evidence(operation="glob")]),
                File(path="/b.py", evidence=[Evidence(operation="grep")]),
            ]
        )
        # Both map to same path
        result = r.remap_paths(lambda _: "/merged.py")
        assert len(result) == 1
        assert result.files[0].path == "/merged.py"
        assert len(result.files[0].evidence) == 2

    def test_remap_connections(self):
        conn = FileConnection(
            path="/src/a.py[imports]/src/b.py",
            source_path="/src/a.py",
            target_path="/src/b.py",
            type="imports",
        )
        r = FileSearchResult(connections=[conn])
        result = r.remap_paths(lambda p: p.replace("/src", "/dst"))
        assert result.connections[0].source_path == "/dst/a.py"
        assert result.connections[0].target_path == "/dst/b.py"


class TestFileSearchResultWithEvidence:
    def test_files_carry_evidence(self):
        glob_ev = GlobEvidence(operation="glob", is_directory=False, size_bytes=100)
        vec_ev = VectorEvidence(operation="vector_search", score=0.95, snippet="match")
        f = File(path="/a.py", evidence=[glob_ev, vec_ev])
        r = FileSearchResult(files=[f])
        assert len(r.files[0].evidence) == 2
        assert isinstance(r.files[0].evidence[0], GlobEvidence)
        assert isinstance(r.files[0].evidence[1], VectorEvidence)

    def test_serialization_round_trip(self):
        f = File(path="/a.py", evidence=[Evidence(operation="glob", score=0.5)])
        r = FileSearchResult(files=[f], message="1 path")
        # Dataclass round-trip: reconstruct from fields
        r2 = FileSearchResult(
            files=[File(path=fi.path, evidence=list(fi.evidence)) for fi in r.files],
            connections=list(r.connections),
            message=r.message,
            success=r.success,
        )
        assert r2.paths == ("/a.py",)
        assert r2.files[0].evidence[0].operation == "glob"


# =====================================================================
# FileSearchSet tests
# =====================================================================


class TestFileSearchSetConstruction:
    def test_defaults(self):
        s = FileSearchSet()
        assert s.files == []
        assert s.connections == []

    def test_has_no_success_or_message(self):
        fields = {f.name for f in dataclasses.fields(FileSearchSet)}
        assert "success" not in fields
        assert "message" not in fields

    def test_with_files(self):
        s = FileSearchSet(files=[File(path="/a.py"), File(path="/b.py")])
        assert len(s) == 2
        assert s.paths == ("/a.py", "/b.py")


class TestFileSearchSetBool:
    def test_bool_true_with_files(self):
        s = FileSearchSet(files=[File(path="/a.py")])
        assert bool(s) is True

    def test_bool_false_empty(self):
        s = FileSearchSet()
        assert bool(s) is False


class TestFileSearchSetSetAlgebra:
    def _make(self, paths: list[str], operation: str = "test") -> FileSearchSet:
        return FileSearchSet(files=[File(path=p, evidence=[Evidence(operation=operation)]) for p in paths])

    def test_intersection(self):
        s1 = self._make(["/a.py", "/b.py"], "glob")
        s2 = self._make(["/b.py", "/c.py"], "grep")
        result = s1 & s2
        assert isinstance(result, FileSearchSet)
        assert set(result.paths) == {"/b.py"}

    def test_union(self):
        s1 = self._make(["/a.py"])
        s2 = self._make(["/b.py"])
        result = s1 | s2
        assert set(result.paths) == {"/a.py", "/b.py"}

    def test_difference(self):
        s1 = self._make(["/a.py", "/b.py"])
        s2 = self._make(["/b.py"])
        result = s1 - s2
        assert set(result.paths) == {"/a.py"}

    def test_pipeline(self):
        s1 = self._make(["/a.py", "/b.py"])
        s2 = self._make(["/b.py", "/c.py"])
        result = s1 >> s2
        assert set(result.paths) == {"/b.py"}

    def test_not_implemented_with_non_set(self):
        s = self._make(["/a.py"])
        assert s.__and__(42) is NotImplemented
        assert s.__or__(42) is NotImplemented
        assert s.__sub__(42) is NotImplemented
        assert s.__rshift__(42) is NotImplemented


class TestFileSearchSetExplain:
    def test_explain_present(self):
        ev = Evidence(operation="glob", score=0.9)
        s = FileSearchSet(files=[File(path="/a.py", evidence=[ev])])
        chain = s.explain("/a.py")
        assert len(chain) == 1
        assert chain[0].operation == "glob"

    def test_explain_absent(self):
        s = FileSearchSet(files=[File(path="/a.py")])
        assert s.explain("/missing.py") == []


class TestFileSearchSetToRefs:
    def test_to_refs(self):
        s = FileSearchSet(files=[File(path="/a.py"), File(path="/b.py")])
        refs = s.to_refs()
        assert len(refs) == 2
        assert refs[0].path == "/a.py"
        assert refs[1].path == "/b.py"
        assert isinstance(refs[0], Ref)

    def test_to_refs_empty(self):
        s = FileSearchSet()
        assert s.to_refs() == []


class TestFileSearchSetConnectionPaths:
    def test_connection_paths(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        s = FileSearchSet(connections=[conn])
        assert s.connection_paths == ("/a.py[imports]/b.py",)

    def test_connection_paths_empty(self):
        s = FileSearchSet()
        assert s.connection_paths == ()


class TestFileSearchResultIsAFileSearchSet:
    def test_isinstance(self):
        r = FileSearchResult()
        assert isinstance(r, FileSearchSet)

    def test_result_has_success_and_message(self):
        r = FileSearchResult()
        assert r.success is True
        assert r.message == ""

    def test_result_inherits_explain(self):
        ev = Evidence(operation="glob")
        r = FileSearchResult(files=[File(path="/a.py", evidence=[ev])])
        assert r.explain("/a.py") == [ev]

    def test_result_inherits_to_refs(self):
        r = FileSearchResult(files=[File(path="/a.py")])
        refs = r.to_refs()
        assert len(refs) == 1

    def test_result_connection_paths(self):
        conn = FileConnection(
            path="/x.py[calls]/y.py",
            source_path="/x.py",
            target_path="/y.py",
            type="calls",
        )
        r = FileSearchResult(connections=[conn])
        assert r.connection_paths == ("/x.py[calls]/y.py",)


class TestFileSearchResultFromRefs:
    def test_from_refs(self):
        refs = [Ref(path="/a.py"), Ref(path="/b.py")]
        r = FileSearchResult.from_refs(refs, operation="graph")
        assert len(r) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert r.success is True
        assert r.files[0].evidence[0].operation == "graph"
        assert r.message == "2 refs"

    def test_from_refs_empty(self):
        r = FileSearchResult.from_refs([])
        assert len(r) == 0


class TestSetAlgebraAcceptsFileSearchSet:
    """FileSearchResult set algebra should accept FileSearchSet operands."""

    def _result(self, paths: list[str]) -> FileSearchResult:
        return FileSearchResult(files=[File(path=p, evidence=[Evidence(operation="r")]) for p in paths])

    def _set(self, paths: list[str]) -> FileSearchSet:
        return FileSearchSet(files=[File(path=p, evidence=[Evidence(operation="s")]) for p in paths])

    def test_result_and_set(self):
        r = self._result(["/a.py", "/b.py"])
        s = self._set(["/b.py", "/c.py"])
        result = r & s
        assert isinstance(result, FileSearchResult)
        assert set(result.paths) == {"/b.py"}
        assert result.success is True

    def test_result_or_set(self):
        r = self._result(["/a.py"])
        s = self._set(["/b.py"])
        result = r | s
        assert isinstance(result, FileSearchResult)
        assert set(result.paths) == {"/a.py", "/b.py"}

    def test_result_sub_set(self):
        r = self._result(["/a.py", "/b.py"])
        s = self._set(["/b.py"])
        result = r - s
        assert isinstance(result, FileSearchResult)
        assert set(result.paths) == {"/a.py"}

    def test_result_rshift_set(self):
        r = self._result(["/a.py", "/b.py"])
        s = self._set(["/b.py", "/c.py"])
        result = r >> s
        assert isinstance(result, FileSearchResult)
        assert set(result.paths) == {"/b.py"}

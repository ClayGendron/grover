"""Tests for GroverFileSystem base class — constructor, session, helpers, routing, public methods.

Covers everything in ``base.py`` not already tested by ``test_routing.py``
(which focuses on mount management and basic candidate routing).
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from grover.base import GroverFileSystem
from grover.results import EditOperation, GroverResult, TwoPathOperation
from tests.conftest import (
    candidate as _candidate,
)
from tests.conftest import (
    dummy_session_factory,
    tracking_session_factory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FullRoutingFS(GroverFileSystem):
    """GroverFileSystem with all ``_*_impl`` methods replaced by AsyncMocks."""

    _ALL_OPS = (
        "read", "stat", "edit", "ls", "delete", "write", "mkdir",
        "move", "copy", "mkconn", "tree",
        "glob", "grep", "semantic_search", "vector_search", "lexical_search",
        "predecessors", "successors", "ancestors", "descendants", "neighborhood",
        "meeting_subgraph", "min_meeting_subgraph",
        "pagerank", "betweenness_centrality", "closeness_centrality",
        "degree_centrality", "in_degree_centrality", "out_degree_centrality", "hits",
    )

    def __init__(self, name: str = "test") -> None:
        super().__init__(session_factory=dummy_session_factory())
        self._name = name
        for op in self._ALL_OPS:
            mock = AsyncMock(return_value=GroverResult())
            setattr(self, f"_{op}_impl", mock)
            setattr(self, f"{op}_mock", mock)

    def __getattr__(self, name: str) -> AsyncMock:
        return cast("AsyncMock", object.__getattribute__(self, name))


# =========================================================================
# Constructor validation
# =========================================================================


class TestConstructorValidation:
    def test_no_engine_no_session_factory_raises(self):
        with pytest.raises(ValueError, match="requires either engine or session_factory"):
            GroverFileSystem()

    def test_engine_only_creates_session_factory(self):
        engine = MagicMock()
        fs = GroverFileSystem(engine=engine)
        assert fs._session_factory is not None
        assert fs._engine is engine

    def test_session_factory_only(self):
        factory = dummy_session_factory()
        fs = GroverFileSystem(session_factory=factory)
        assert fs._session_factory is factory
        assert fs._engine is None

    def test_both_engine_and_session_factory_prefers_factory(self):
        engine = MagicMock()
        factory = dummy_session_factory()
        fs = GroverFileSystem(engine=engine, session_factory=factory)
        assert fs._session_factory is factory
        assert fs._engine is engine


# =========================================================================
# Session management
# =========================================================================


class TestSessionManagement:
    async def test_use_session_commits_on_success(self):
        factory, sessions = tracking_session_factory()
        fs = _FullRoutingFS()
        fs._session_factory = factory

        async with fs._use_session():
            pass

        assert len(sessions) == 1
        assert sessions[0].committed is True
        assert sessions[0].rolled_back is False

    async def test_use_session_rolls_back_on_error(self):
        factory, sessions = tracking_session_factory()
        fs = _FullRoutingFS()
        fs._session_factory = factory

        with pytest.raises(RuntimeError, match="boom"):
            async with fs._use_session():
                raise RuntimeError("boom")

        assert len(sessions) == 1
        assert sessions[0].committed is False
        assert sessions[0].rolled_back is True


# =========================================================================
# Result helpers
# =========================================================================


class TestError:
    def test_error_returns_failed_result(self):
        fs = _FullRoutingFS()
        r = fs._error("something broke")
        assert r.success is False
        assert r.errors == ["something broke"]
        assert r.candidates == []


class TestAddPrefix:
    def test_empty_prefix_returns_result_unchanged(self):
        r = GroverResult(candidates=[_candidate("/file.py")])
        rebased = r.add_prefix("")
        assert rebased is r

    def test_prefix_prepends_to_candidate_paths(self):
        r = GroverResult(candidates=[_candidate("/file.py"), _candidate("/dir/a.py")])
        r.add_prefix("/mount")
        assert r.candidates[0].path == "/mount/file.py"
        assert r.candidates[1].path == "/mount/dir/a.py"

    def test_root_path_gets_prefix_without_trailing_slash(self):
        r = GroverResult(candidates=[_candidate("/")])
        r.add_prefix("/data")
        assert r.candidates[0].path == "/data"

    def test_preserves_success_and_errors(self):
        r = GroverResult(success=False, errors=["err"], candidates=[_candidate("/x.py")])
        r.add_prefix("/m")
        assert r.success is False
        assert r.errors == ["err"]


class TestExcludeMountedPaths:
    async def test_no_mounts_returns_unchanged(self):
        fs = _FullRoutingFS()
        r = GroverResult(candidates=[_candidate("/a.py")])
        result = fs._exclude_mounted_paths(r)
        assert len(result.candidates) == 1

    async def test_filters_candidates_under_mount(self):
        fs = _FullRoutingFS()
        child = _FullRoutingFS("child")
        await fs.add_mount("/data", child)
        r = GroverResult(candidates=[
            _candidate("/local.py"),
            _candidate("/data/file.py"),
            _candidate("/data"),
        ])
        result = fs._exclude_mounted_paths(r)
        assert [c.path for c in result.candidates] == ["/local.py"]

    async def test_does_not_filter_prefix_substring(self):
        fs = _FullRoutingFS()
        await fs.add_mount("/web", _FullRoutingFS("child"))
        r = GroverResult(candidates=[_candidate("/webinar/page.html")])
        result = fs._exclude_mounted_paths(r)
        assert len(result.candidates) == 1


class TestRequireSameMount:
    def test_single_resolved_succeeds(self):
        fs = _FullRoutingFS()
        resolved = [(fs, "/file.py", "/mount")]
        result = GroverFileSystem._require_same_mount(resolved, "test ops")
        assert result == (fs, "/mount")

    def test_same_mount_succeeds(self):
        fs = _FullRoutingFS()
        resolved = [(fs, "/a.py", "/m"), (fs, "/b.py", "/m")]
        result = GroverFileSystem._require_same_mount(resolved, "ops")
        assert result == (fs, "/m")

    def test_different_filesystem_returns_error(self):
        fs1 = _FullRoutingFS("a")
        fs2 = _FullRoutingFS("b")
        resolved = [(fs1, "/a.py", "/m"), (fs2, "/b.py", "/m")]
        result = GroverFileSystem._require_same_mount(resolved, "move sources")
        assert isinstance(result, str)
        assert "move sources" in result

    def test_different_prefix_returns_error(self):
        fs = _FullRoutingFS()
        resolved = [(fs, "/a.py", "/m1"), (fs, "/b.py", "/m2")]
        result = GroverFileSystem._require_same_mount(resolved, "copy dests")
        assert isinstance(result, str)
        assert "copy dests" in result


class TestMergeResults:
    def test_empty_list_returns_success(self):
        r = GroverFileSystem._merge_results([])
        assert r.success is True
        assert r.candidates == []

    def test_single_result_returned(self):
        r1 = GroverResult(candidates=[_candidate("/a.py")])
        merged = GroverFileSystem._merge_results([r1])
        assert merged.candidates[0].path == "/a.py"

    def test_union_of_two_results(self):
        r1 = GroverResult(candidates=[_candidate("/a.py")])
        r2 = GroverResult(candidates=[_candidate("/b.py")])
        merged = GroverFileSystem._merge_results([r1, r2])
        paths = {c.path for c in merged.candidates}
        assert paths == {"/a.py", "/b.py"}

    def test_failure_propagates(self):
        r1 = GroverResult(candidates=[_candidate("/a.py")])
        r2 = GroverResult(success=False, errors=["fail"], candidates=[])
        merged = GroverFileSystem._merge_results([r1, r2])
        assert merged.success is False
        assert "fail" in merged.errors


# =========================================================================
# Candidate dispatch
# =========================================================================


class TestGroupByTerminal:
    async def test_groups_by_filesystem(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        candidates = GroverResult(candidates=[
            _candidate("/local.py"),
            _candidate("/data/remote.py"),
        ])
        groups = root._group_candidates_by_terminal(candidates)

        assert len(groups) == 2
        fs_names = {g[0]._name for g in groups}
        assert fs_names == {"root", "child"}

        for fs, prefix, result in groups:
            if fs._name == "root":
                assert prefix == ""
                assert result.candidates[0].path == "/local.py"
            else:
                assert prefix == "/data"
                assert result.candidates[0].path == "/remote.py"

    async def test_single_mount_groups_together(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        candidates = GroverResult(candidates=[
            _candidate("/data/a.py"),
            _candidate("/data/b.py"),
        ])
        groups = root._group_candidates_by_terminal(candidates)
        assert len(groups) == 1
        assert len(groups[0][2].candidates) == 2

    def test_empty_candidates(self):
        root = _FullRoutingFS()
        groups = root._group_candidates_by_terminal(GroverResult())
        assert groups == []


class TestDispatchCandidates:
    async def test_dispatches_to_correct_filesystem(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.read_mock.return_value = GroverResult(
            candidates=[_candidate("/remote.py", content="hello")],
        )

        candidates = GroverResult(candidates=[_candidate("/data/remote.py")])
        result = await root._dispatch_candidates("read", candidates)

        child.read_mock.assert_awaited_once()
        assert result.candidates[0].path == "/data/remote.py"
        assert result.candidates[0].content == "hello"

    async def test_empty_candidates_returns_empty(self):
        root = _FullRoutingFS()
        result = await root._dispatch_candidates("read", GroverResult())
        assert result.candidates == []
        assert result.success is True

    async def test_merges_results_from_multiple_mounts(self):
        root = _FullRoutingFS("root")
        c1 = _FullRoutingFS("c1")
        c2 = _FullRoutingFS("c2")
        await root.add_mount("/m1", c1)
        await root.add_mount("/m2", c2)

        c1.stat_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        c2.stat_mock.return_value = GroverResult(candidates=[_candidate("/b.py")])

        candidates = GroverResult(candidates=[
            _candidate("/m1/a.py"),
            _candidate("/m2/b.py"),
        ])
        result = await root._dispatch_candidates("stat", candidates)

        paths = {c.path for c in result.candidates}
        assert paths == {"/m1/a.py", "/m2/b.py"}


# =========================================================================
# _route_single
# =========================================================================


class TestRouteSingle:
    async def test_requires_exactly_one_of_path_or_candidates(self):
        root = _FullRoutingFS("root")

        with pytest.raises(ValueError, match="Exactly one of path or candidates"):
            await root._route_single("read", None, None)

        with pytest.raises(ValueError, match="Exactly one of path or candidates"):
            await root._route_single(
                "read",
                "/local.py",
                GroverResult(candidates=[_candidate("/remote.py")]),
            )

    async def test_with_path_resolves_and_calls_impl(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.read_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py", content="data")],
        )
        result = await root._route_single("read", "/data/file.py", None)

        child.read_mock.assert_awaited_once()
        assert result.candidates[0].path == "/data/file.py"

    async def test_with_candidates_dispatches(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.ls_mock.return_value = GroverResult(
            candidates=[_candidate("/x.py")],
        )

        candidates = GroverResult(candidates=[_candidate("/data/x.py")])
        result = await root._route_single("ls", None, candidates)

        child.ls_mock.assert_awaited_once()
        assert result.candidates[0].path == "/data/x.py"

    async def test_unmounted_path_stays_on_self(self):
        root = _FullRoutingFS("root")
        root.stat_mock.return_value = GroverResult(
            candidates=[_candidate("/local.py")],
        )
        result = await root._route_single("stat", "/local.py", None)

        root.stat_mock.assert_awaited_once()
        assert result.candidates[0].path == "/local.py"


# =========================================================================
# _route_two_path
# =========================================================================


class TestRouteTwoPath:
    async def test_empty_ops_returns_success(self):
        root = _FullRoutingFS()
        result = await root._route_two_path("move", [])
        assert result.success is True
        assert result.candidates == []

    async def test_same_mount_calls_impl(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.move_mock.return_value = GroverResult(
            candidates=[_candidate("/b.py")],
        )

        ops = [TwoPathOperation(src="/data/a.py", dest="/data/b.py")]
        result = await root._route_two_path("move", ops)

        child.move_mock.assert_awaited_once()
        assert result.candidates[0].path == "/data/b.py"

    async def test_same_mount_rebases_ops(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.copy_mock.return_value = GroverResult(candidates=[_candidate("/b.py")])

        ops = [TwoPathOperation(src="/data/a.py", dest="/data/b.py")]
        await root._route_two_path("copy", ops)

        call_kwargs = child.copy_mock.call_args
        batch = call_kwargs.kwargs.get("ops") or call_kwargs[1].get("ops")
        assert batch[0].src == "/a.py"
        assert batch[0].dest == "/b.py"

    async def test_different_source_mounts_returns_error(self):
        root = _FullRoutingFS("root")
        m1 = _FullRoutingFS("m1")
        m2 = _FullRoutingFS("m2")
        await root.add_mount("/m1", m1)
        await root.add_mount("/m2", m2)

        ops = [
            TwoPathOperation(src="/m1/a.py", dest="/m1/b.py"),
            TwoPathOperation(src="/m2/c.py", dest="/m1/d.py"),
        ]
        result = await root._route_two_path("move", ops)
        assert result.success is False
        assert "move sources" in result.errors[0]

    async def test_different_dest_mounts_returns_error(self):
        root = _FullRoutingFS("root")
        m1 = _FullRoutingFS("m1")
        m2 = _FullRoutingFS("m2")
        await root.add_mount("/m1", m1)
        await root.add_mount("/m2", m2)

        ops = [
            TwoPathOperation(src="/m1/a.py", dest="/m1/b.py"),
            TwoPathOperation(src="/m1/c.py", dest="/m2/d.py"),
        ]
        result = await root._route_two_path("move", ops)
        assert result.success is False
        assert "move destinations" in result.errors[0]


# =========================================================================
# Cross-mount transfer
# =========================================================================


class TestCrossMountTransfer:
    async def test_cross_mount_copy_reads_and_writes(self):
        root = _FullRoutingFS("root")
        src = _FullRoutingFS("src")
        dst = _FullRoutingFS("dst")
        await root.add_mount("/src", src)
        await root.add_mount("/dst", dst)

        src.read_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py", content="hello")],
        )
        dst.write_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py")],
        )

        result = await root.copy("/src/file.py", "/dst/file.py")

        src.read_mock.assert_awaited_once()
        dst.write_mock.assert_awaited_once()
        assert result.success is True
        assert result.candidates[0].path == "/dst/file.py"

    async def test_cross_mount_move_also_deletes_source(self):
        root = _FullRoutingFS("root")
        src = _FullRoutingFS("src")
        dst = _FullRoutingFS("dst")
        await root.add_mount("/src", src)
        await root.add_mount("/dst", dst)

        src.read_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py", content="data")],
        )
        dst.write_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py")],
        )
        src.delete_mock.return_value = GroverResult(
            candidates=[_candidate("/file.py")],
        )

        result = await root.move("/src/file.py", "/dst/file.py")

        src.delete_mock.assert_awaited_once()
        assert result.success is True


# =========================================================================
# _route_fanout
# =========================================================================


class TestRouteFanout:
    async def test_with_candidates_dispatches(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.grep_mock.return_value = GroverResult(
            candidates=[_candidate("/match.py")],
        )

        candidates = GroverResult(candidates=[_candidate("/data/match.py")])
        result = await root._route_fanout("grep", candidates, pattern="test")

        child.grep_mock.assert_awaited_once()
        assert result.candidates[0].path == "/data/match.py"

    async def test_without_candidates_fans_out(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        root.glob_mock.return_value = GroverResult(
            candidates=[_candidate("/local.py")],
        )
        child.glob_mock.return_value = GroverResult(
            candidates=[_candidate("/remote.py")],
        )

        result = await root._route_fanout("glob", None, pattern="*.py")

        root.glob_mock.assert_awaited_once()
        child.glob_mock.assert_awaited_once()
        paths = {c.path for c in result.candidates}
        assert "/local.py" in paths
        assert "/data/remote.py" in paths

    async def test_fanout_excludes_mounted_paths_from_self(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        root.glob_mock.return_value = GroverResult(
            candidates=[_candidate("/local.py"), _candidate("/data/shadow.py")],
        )
        child.glob_mock.return_value = GroverResult(
            candidates=[_candidate("/real.py")],
        )

        result = await root._route_fanout("glob", None, pattern="*.py")

        paths = {c.path for c in result.candidates}
        assert "/local.py" in paths
        assert "/data/real.py" in paths
        assert "/data/shadow.py" not in paths

    async def test_fanout_rebases_mount_results(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/web", child)

        root.pagerank_mock.return_value = GroverResult()
        child.pagerank_mock.return_value = GroverResult(
            candidates=[_candidate("/index.html")],
        )

        result = await root._route_fanout("pagerank", None)
        assert result.candidates[0].path == "/web/index.html"


# =========================================================================
# Public methods — CRUD
# =========================================================================


class TestPublicCRUD:
    async def test_read_rejects_invalid_input_combinations(self):
        fs = _FullRoutingFS()

        with pytest.raises(ValueError, match="Exactly one of path or candidates"):
            await fs.read()

        with pytest.raises(ValueError, match="Exactly one of path or candidates"):
            await fs.read("/f.py", candidates=GroverResult(candidates=[_candidate("/f.py")]))

    @pytest.mark.parametrize("method", ["read", "stat", "ls", "mkdir"])
    async def test_single_path_ops_route_to_impl(self, method):
        fs = _FullRoutingFS()
        mock = getattr(fs, f"{method}_mock")
        mock.return_value = GroverResult(candidates=[_candidate("/f.py")])
        result = await getattr(fs, method)("/f.py")
        mock.assert_awaited_once()
        assert result.candidates[0].path == "/f.py"

    async def test_edit_routes(self):
        fs = _FullRoutingFS()
        fs.edit_mock.return_value = GroverResult(candidates=[_candidate("/f.py")])
        result = await fs.edit("/f.py", old="x", new="y")
        fs.edit_mock.assert_awaited_once()
        assert result.candidates[0].path == "/f.py"

    async def test_edit_creates_edit_operation(self):
        fs = _FullRoutingFS()
        fs.edit_mock.return_value = GroverResult()
        await fs.edit("/f.py", old="a", new="b", replace_all=True)
        call_kwargs = fs.edit_mock.call_args
        edits = call_kwargs.kwargs.get("edits") or call_kwargs[1].get("edits")
        assert len(edits) == 1
        assert edits[0].old == "a"
        assert edits[0].new == "b"
        assert edits[0].replace_all is True

    async def test_edit_with_explicit_edits(self):
        fs = _FullRoutingFS()
        fs.edit_mock.return_value = GroverResult()
        ops = [EditOperation(old="x", new="y"), EditOperation(old="a", new="b")]
        await fs.edit("/f.py", edits=ops)
        call_kwargs = fs.edit_mock.call_args
        edits = call_kwargs.kwargs.get("edits") or call_kwargs[1].get("edits")
        assert len(edits) == 2

    async def test_delete_routes(self):
        fs = _FullRoutingFS()
        fs.delete_mock.return_value = GroverResult(candidates=[_candidate("/f.py")])
        result = await fs.delete("/f.py")
        fs.delete_mock.assert_awaited_once()
        assert result.candidates[0].path == "/f.py"

    async def test_delete_permanent_kwarg(self):
        fs = _FullRoutingFS()
        fs.delete_mock.return_value = GroverResult()
        await fs.delete("/f.py", permanent=True)
        call_kwargs = fs.delete_mock.call_args
        assert call_kwargs.kwargs.get("permanent") is True

    async def test_write_routes(self):
        fs = _FullRoutingFS()
        fs.write_mock.return_value = GroverResult(candidates=[_candidate("/f.py")])
        result = await fs.write("/f.py", "hello")
        fs.write_mock.assert_awaited_once()
        assert result.candidates[0].path == "/f.py"

    async def test_write_passes_content_and_overwrite(self):
        fs = _FullRoutingFS()
        fs.write_mock.return_value = GroverResult()
        await fs.write("/f.py", "data", overwrite=False)
        call_kwargs = fs.write_mock.call_args
        assert call_kwargs.kwargs.get("content") == "data"
        assert call_kwargs.kwargs.get("overwrite") is False

    async def test_tree_passes_max_depth(self):
        fs = _FullRoutingFS()
        fs.tree_mock.return_value = GroverResult(candidates=[_candidate("/dir")])
        await fs.tree("/dir", max_depth=3)
        call_kwargs = fs.tree_mock.call_args
        assert call_kwargs.kwargs.get("max_depth") == 3


# =========================================================================
# Public methods — two-path ops
# =========================================================================


class TestPublicTwoPath:
    async def test_move_routes(self):
        fs = _FullRoutingFS()
        fs.move_mock.return_value = GroverResult(candidates=[_candidate("/b.py")])
        result = await fs.move("/a.py", "/b.py")
        fs.move_mock.assert_awaited_once()
        assert result.candidates[0].path == "/b.py"

    async def test_move_with_batch(self):
        fs = _FullRoutingFS()
        fs.move_mock.return_value = GroverResult()
        moves = [TwoPathOperation(src="/a.py", dest="/b.py")]
        await fs.move(moves=moves)
        fs.move_mock.assert_awaited_once()

    async def test_copy_routes(self):
        fs = _FullRoutingFS()
        fs.copy_mock.return_value = GroverResult(candidates=[_candidate("/b.py")])
        result = await fs.copy("/a.py", "/b.py")
        fs.copy_mock.assert_awaited_once()
        assert result.candidates[0].path == "/b.py"

    async def test_copy_with_batch(self):
        fs = _FullRoutingFS()
        fs.copy_mock.return_value = GroverResult()
        copies = [TwoPathOperation(src="/a.py", dest="/b.py")]
        await fs.copy(copies=copies)
        fs.copy_mock.assert_awaited_once()


# =========================================================================
# Public methods — mkconn
# =========================================================================


class TestPublicMkconn:
    async def test_mkconn_same_mount(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        child.mkconn_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await root.mkconn("/data/a.py", "/data/b.py", "imports")
        child.mkconn_mock.assert_awaited_once()
        assert result.success is True

    async def test_mkconn_cross_mount_returns_error(self):
        root = _FullRoutingFS("root")
        m1 = _FullRoutingFS("m1")
        m2 = _FullRoutingFS("m2")
        await root.add_mount("/m1", m1)
        await root.add_mount("/m2", m2)

        result = await root.mkconn("/m1/a.py", "/m2/b.py", "imports")
        assert result.success is False
        assert "Cross-mount" in result.errors[0]

    async def test_mkconn_on_root_filesystem(self):
        fs = _FullRoutingFS()
        fs.mkconn_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await fs.mkconn("/a.py", "/b.py", "imports")
        fs.mkconn_mock.assert_awaited_once()
        assert result.success is True


# =========================================================================
# Public methods — search (fanout)
# =========================================================================


class TestPublicSearch:
    async def test_glob_routes(self):
        fs = _FullRoutingFS()
        fs.glob_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await fs.glob("*.py")
        fs.glob_mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"

    async def test_glob_with_candidates(self):
        fs = _FullRoutingFS()
        fs.glob_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        cands = GroverResult(candidates=[_candidate("/a.py")])
        await fs.glob("*.py", candidates=cands)
        fs.glob_mock.assert_awaited_once()

    async def test_grep_routes(self):
        fs = _FullRoutingFS()
        fs.grep_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await fs.grep("pattern")
        fs.grep_mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"

    async def test_grep_passes_kwargs(self):
        fs = _FullRoutingFS()
        fs.grep_mock.return_value = GroverResult()
        await fs.grep("test", case_sensitive=False, max_results=5)
        call_kwargs = fs.grep_mock.call_args
        assert call_kwargs.kwargs.get("case_sensitive") is False
        assert call_kwargs.kwargs.get("max_results") == 5

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("semantic_search", ("auth logic",)),
            ("vector_search", ([0.1, 0.2, 0.3],)),
            ("lexical_search", ("keyword",)),
        ],
    )
    async def test_search_variants_route_to_impl(self, method, args):
        fs = _FullRoutingFS()
        mock = getattr(fs, f"{method}_mock")
        mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await getattr(fs, method)(*args)
        mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"

    async def test_semantic_search_passes_k(self):
        fs = _FullRoutingFS()
        fs.semantic_search_mock.return_value = GroverResult()
        await fs.semantic_search("query", k=5)
        call_kwargs = fs.semantic_search_mock.call_args
        assert call_kwargs.kwargs.get("k") == 5


# =========================================================================
# Public methods — graph traversal (route_single)
# =========================================================================


class TestPublicGraphTraversal:
    @pytest.mark.parametrize("method", [
        "predecessors", "successors", "ancestors", "descendants",
    ])
    async def test_traversal_ops_route_to_impl(self, method):
        fs = _FullRoutingFS()
        mock = getattr(fs, f"{method}_mock")
        mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await getattr(fs, method)("/a.py")
        mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"

    async def test_neighborhood_passes_depth(self):
        fs = _FullRoutingFS()
        fs.neighborhood_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        await fs.neighborhood("/a.py", depth=3)
        call_kwargs = fs.neighborhood_mock.call_args
        assert call_kwargs.kwargs.get("depth") == 3

    async def test_predecessors_with_candidates(self):
        fs = _FullRoutingFS()
        fs.predecessors_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        cands = GroverResult(candidates=[_candidate("/a.py")])
        await fs.predecessors(candidates=cands)
        fs.predecessors_mock.assert_awaited_once()


# =========================================================================
# Public methods — graph candidate-only (dispatch)
# =========================================================================


class TestPublicGraphCandidateOnly:
    @pytest.mark.parametrize("method", ["meeting_subgraph", "min_meeting_subgraph"])
    async def test_subgraph_ops_dispatch(self, method):
        fs = _FullRoutingFS()
        mock = getattr(fs, f"{method}_mock")
        mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        cands = GroverResult(candidates=[_candidate("/a.py")])
        result = await getattr(fs, method)(cands)
        mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"


# =========================================================================
# Public methods — graph algorithms (fanout)
# =========================================================================


class TestPublicGraphAlgorithms:
    @pytest.mark.parametrize("method", [
        "pagerank", "betweenness_centrality", "closeness_centrality",
        "degree_centrality", "in_degree_centrality", "out_degree_centrality",
        "hits",
    ])
    async def test_algorithm_routes_to_impl(self, method):
        fs = _FullRoutingFS()
        mock = getattr(fs, f"{method}_mock")
        mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        result = await getattr(fs, method)()
        mock.assert_awaited_once()
        assert result.candidates[0].path == "/a.py"

    async def test_pagerank_with_candidates(self):
        fs = _FullRoutingFS()
        fs.pagerank_mock.return_value = GroverResult(candidates=[_candidate("/a.py")])
        cands = GroverResult(candidates=[_candidate("/a.py")])
        await fs.pagerank(candidates=cands)
        fs.pagerank_mock.assert_awaited_once()

    async def test_algorithms_fan_out_across_mounts(self):
        root = _FullRoutingFS("root")
        child = _FullRoutingFS("child")
        await root.add_mount("/data", child)

        root.hits_mock.return_value = GroverResult(
            candidates=[_candidate("/local.py")],
        )
        child.hits_mock.return_value = GroverResult(
            candidates=[_candidate("/remote.py")],
        )

        result = await root.hits()
        paths = {c.path for c in result.candidates}
        assert "/local.py" in paths
        assert "/data/remote.py" in paths


# ===========================================================================
# Edge cases — empty/missing args
# ===========================================================================


class TestEdgeCaseMissingArgs:
    async def test_route_write_batch_empty(self):
        fs = _FullRoutingFS()
        result = await fs._route_write_batch([], overwrite=True)
        assert result.success is True
        assert result.candidates == []

    async def test_edit_without_old_new_returns_error(self):
        fs = _FullRoutingFS()
        result = await fs.edit(path="/a.py")
        assert result.success is False
        assert "edit requires" in result.errors[0]

    async def test_move_without_src_dest_returns_error(self):
        fs = _FullRoutingFS()
        result = await fs.move()
        assert result.success is False
        assert "move requires" in result.errors[0]

    async def test_copy_without_src_dest_returns_error(self):
        fs = _FullRoutingFS()
        result = await fs.copy()
        assert result.success is False
        assert "copy requires" in result.errors[0]

"""Tests for query/executor.py — execution engine for CLI query AST."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from grover.query.ast import (
    CopyCommand,
    EditCommand,
    ExceptStage,
    GlobCommand,
    GraphTraversalCommand,
    GrepCommand,
    IntersectStage,
    KindsCommand,
    LexicalSearchCommand,
    LsCommand,
    MeetingGraphCommand,
    MkconnCommand,
    MkdirCommand,
    MoveCommand,
    PipelineNode,
    QueryPlan,
    RankCommand,
    ReadCommand,
    SemanticSearchCommand,
    SortCommand,
    TopCommand,
    TreeCommand,
    UnionNode,
    VectorSearchCommand,
    Visibility,
    WriteCommand,
)
from grover.query.executor import _apply_visibility, _preserve_under_root, execute_query
from grover.results import Candidate, GroverResult

# ===========================================================================
# Helpers
# ===========================================================================


def _fs():
    """Build a mock GroverFileSystem with default return values."""
    fs = AsyncMock()
    fs._merge_results = lambda results: GroverResult(
        candidates=[c for r in results for c in r.candidates],
    )
    empty = GroverResult(candidates=[])
    for method in (
        "read",
        "stat",
        "delete",
        "write",
        "edit",
        "ls",
        "tree",
        "mkdir",
        "move",
        "copy",
        "mkconn",
        "glob",
        "grep",
        "semantic_search",
        "vector_search",
        "lexical_search",
        "predecessors",
        "successors",
        "ancestors",
        "descendants",
        "neighborhood",
        "meeting_subgraph",
        "min_meeting_subgraph",
        "pagerank",
        "betweenness_centrality",
        "closeness_centrality",
        "degree_centrality",
        "in_degree_centrality",
        "out_degree_centrality",
        "hits",
    ):
        getattr(fs, method).return_value = empty
    return fs


def _plan(node) -> QueryPlan:
    return QueryPlan(ast=node, methods=(), render_mode="query_list")


def _result(*paths: str) -> GroverResult:
    return GroverResult(candidates=[Candidate(path=p) for p in paths])


NO_VIS = Visibility(include_all=False, include_kinds=())


# ===========================================================================
# Edit command
# ===========================================================================


class TestEditCommand:
    async def test_edit_with_piped_input(self):
        fs = _fs()
        piped = _result("/a.py")
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(EditCommand(old="x", new="y", paths=(), replace_all=False),),
        )
        fs.read.return_value = piped
        fs.edit.return_value = GroverResult(candidates=[Candidate(path="/a.py")])
        await execute_query(fs, _plan(node))
        fs.edit.assert_called_once()

    async def test_edit_piped_plus_paths_raises(self):
        fs = _fs()
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(EditCommand(old="x", new="y", paths=("/b.py",), replace_all=False),),
        )
        fs.read.return_value = _result("/a.py")
        with pytest.raises(ValueError, match="cannot combine piped input"):
            await execute_query(fs, _plan(node))

    async def test_edit_no_path_no_pipe_raises(self):
        fs = _fs()
        node = EditCommand(old="x", new="y", paths=(), replace_all=False)
        with pytest.raises(ValueError, match="requires a path"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Write command
# ===========================================================================


class TestWriteCommand:
    async def test_write_piped_raises(self):
        fs = _fs()
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(WriteCommand(path="/b.py", content="x", overwrite=True),),
        )
        fs.read.return_value = _result("/a.py")
        with pytest.raises(ValueError, match="cannot be used in a pipeline"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Mkdir command
# ===========================================================================


class TestMkdirCommand:
    async def test_mkdir_piped_raises(self):
        fs = _fs()
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(MkdirCommand(paths=("/dir",)),),
        )
        fs.read.return_value = _result("/a.py")
        with pytest.raises(ValueError, match="cannot be used in a pipeline"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Transfer commands (move/copy)
# ===========================================================================


class TestTransferCommands:
    async def test_move_no_source_no_pipe_raises(self):
        fs = _fs()
        node = MoveCommand(dest="/dest", overwrite=True)
        with pytest.raises(ValueError, match="requires a source"):
            await execute_query(fs, _plan(node))

    async def test_move_piped_plus_src_raises(self):
        fs = _fs()
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(MoveCommand(src="/a.py", dest="/b.py", overwrite=True),),
        )
        fs.read.return_value = _result("/a.py")
        with pytest.raises(ValueError, match="cannot combine piped input"):
            await execute_query(fs, _plan(node))

    async def test_move_piped_with_candidates(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        fs.move.return_value = GroverResult(candidates=[Candidate(path="/dest/a.py")])
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(MoveCommand(dest="/dest", overwrite=True),),
        )
        await execute_query(fs, _plan(node))
        fs.move.assert_called_once()

    async def test_move_piped_empty_candidates(self):
        fs = _fs()
        fs.glob.return_value = GroverResult(candidates=[])
        node = PipelineNode(
            source=GlobCommand(pattern="*.xyz", visibility=NO_VIS),
            stages=(MoveCommand(dest="/dest", overwrite=True),),
        )
        result = await execute_query(fs, _plan(node))
        assert result.candidates == []

    async def test_copy_no_source_no_pipe_raises(self):
        fs = _fs()
        node = CopyCommand(dest="/dest", overwrite=True)
        with pytest.raises(ValueError, match="requires a source"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Mkconn command
# ===========================================================================


class TestMkconnCommand:
    async def test_mkconn_no_source_no_pipe_raises(self):
        fs = _fs()
        node = MkconnCommand(connection_type="imports", target="/b.py")
        with pytest.raises(ValueError, match="requires a source"):
            await execute_query(fs, _plan(node))

    async def test_mkconn_piped_plus_source_raises(self):
        fs = _fs()
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(MkconnCommand(source="/x.py", connection_type="imports", target="/b.py"),),
        )
        fs.read.return_value = _result("/a.py")
        with pytest.raises(ValueError, match="cannot combine piped input"):
            await execute_query(fs, _plan(node))

    async def test_mkconn_piped_multi_source(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py", "/c.py")
        fs.mkconn.return_value = GroverResult(candidates=[Candidate(path="/conn")])
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py", "/c.py")),
            stages=(MkconnCommand(connection_type="imports", target="/b.py"),),
        )
        await execute_query(fs, _plan(node))
        assert fs.mkconn.call_count == 2


# ===========================================================================
# Tree / glob / grep with visibility
# ===========================================================================


class TestTreeGlobGrepVisibility:
    async def test_tree_piped_input(self):
        fs = _fs()
        fs.read.return_value = _result("/dir")
        fs.tree.return_value = GroverResult(candidates=[Candidate(path="/dir/a.py")])
        node = PipelineNode(
            source=ReadCommand(paths=("/dir",)),
            stages=(TreeCommand(paths=(), max_depth=None, visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        fs.tree.assert_called_once()

    async def test_tree_with_visibility(self):
        fs = _fs()
        vis = Visibility(include_all=True, include_kinds=())
        node = TreeCommand(paths=(), max_depth=None, visibility=vis)
        fs.ls.return_value = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
            ]
        )
        await execute_query(fs, _plan(node))
        fs.ls.assert_called()

    async def test_glob_piped(self):
        fs = _fs()
        fs.read.return_value = _result("/dir")
        fs.glob.return_value = GroverResult(candidates=[])
        node = PipelineNode(
            source=ReadCommand(paths=("/dir",)),
            stages=(GlobCommand(pattern="*.py", visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        fs.glob.assert_called_once()

    async def test_glob_with_visibility(self):
        fs = _fs()
        vis = Visibility(include_all=True, include_kinds=())
        node = GlobCommand(pattern="*.py", visibility=vis)
        fs.ls.return_value = GroverResult(candidates=[Candidate(path="/a.py", kind="file")])
        fs.glob.return_value = GroverResult(candidates=[Candidate(path="/a.py", kind="file")])
        await execute_query(fs, _plan(node))
        fs.ls.assert_called()

    async def test_grep_with_visibility(self):
        fs = _fs()
        vis = Visibility(include_all=True, include_kinds=())
        node = GrepCommand(pattern="test", case_sensitive=True, max_results=None, visibility=vis)
        fs.ls.return_value = GroverResult(candidates=[Candidate(path="/a.py", kind="file")])
        fs.read.return_value = GroverResult(candidates=[Candidate(path="/a.py", kind="file", content="test")])
        fs.grep.return_value = GroverResult(candidates=[])
        await execute_query(fs, _plan(node))
        fs.ls.assert_called()


# ===========================================================================
# Lexical search with visibility
# ===========================================================================


class TestLexicalSearchVisibility:
    async def test_lexical_with_visibility(self):
        fs = _fs()
        vis = Visibility(include_all=True, include_kinds=())
        node = LexicalSearchCommand(query="test", k=10, visibility=vis)
        fs.ls.return_value = GroverResult(candidates=[Candidate(path="/a.py", kind="file")])
        fs.lexical_search.return_value = GroverResult(candidates=[])
        await execute_query(fs, _plan(node))
        fs.ls.assert_called()


# ===========================================================================
# Graph traversal
# ===========================================================================


class TestGraphTraversal:
    async def test_single_path(self):
        fs = _fs()
        fs.predecessors.return_value = GroverResult(candidates=[Candidate(path="/b.py")])
        node = GraphTraversalCommand(method_name="predecessors", paths=("/a.py",), depth=2, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.predecessors.assert_called_once()

    async def test_multi_path(self):
        fs = _fs()
        fs.successors.return_value = GroverResult(candidates=[])
        node = GraphTraversalCommand(method_name="successors", paths=("/a.py", "/b.py"), depth=2, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.successors.assert_called_once()

    async def test_piped_input(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        fs.ancestors.return_value = GroverResult(candidates=[])
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(GraphTraversalCommand(method_name="ancestors", paths=(), depth=2, visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        fs.ancestors.assert_called_once()

    async def test_neighborhood_single_path_with_depth(self):
        fs = _fs()
        fs.neighborhood.return_value = GroverResult(candidates=[])
        node = GraphTraversalCommand(method_name="neighborhood", paths=("/a.py",), depth=3, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        call_kwargs = fs.neighborhood.call_args[1]
        assert call_kwargs["depth"] == 3

    async def test_neighborhood_multi_path(self):
        fs = _fs()
        fs.neighborhood.return_value = GroverResult(candidates=[])
        node = GraphTraversalCommand(
            method_name="neighborhood",
            paths=("/a.py", "/b.py"),
            depth=2,
            visibility=NO_VIS,
        )
        await execute_query(fs, _plan(node))
        fs.neighborhood.assert_called_once()

    async def test_neighborhood_piped(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        fs.neighborhood.return_value = GroverResult(candidates=[])
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(
                GraphTraversalCommand(
                    method_name="neighborhood",
                    paths=(),
                    depth=2,
                    visibility=NO_VIS,
                ),
            ),
        )
        await execute_query(fs, _plan(node))
        call_kwargs = fs.neighborhood.call_args[1]
        assert "depth" in call_kwargs

    async def test_no_paths_no_pipe_raises(self):
        fs = _fs()
        node = GraphTraversalCommand(method_name="predecessors", paths=(), depth=2, visibility=NO_VIS)
        with pytest.raises(ValueError, match="requires explicit paths"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Rank
# ===========================================================================


class TestRankCommand:
    async def test_rank_piped(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        fs.pagerank.return_value = GroverResult(candidates=[Candidate(path="/a.py", score=0.5)])
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(RankCommand(method_name="pagerank", paths=(), visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        fs.pagerank.assert_called_once()

    async def test_rank_explicit_paths(self):
        fs = _fs()
        fs.pagerank.return_value = GroverResult(candidates=[])
        node = RankCommand(method_name="pagerank", paths=("/a.py",), visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.pagerank.assert_called_once()

    async def test_rank_no_paths(self):
        fs = _fs()
        fs.pagerank.return_value = GroverResult(candidates=[])
        node = RankCommand(method_name="pagerank", paths=(), visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.pagerank.assert_called_once()


# ===========================================================================
# Set operations (sort, top, kinds)
# ===========================================================================


class TestSetOperations:
    async def test_sort_requires_piped(self):
        fs = _fs()
        node = SortCommand(operation=None, reverse=True)
        with pytest.raises(ValueError, match="sort requires piped input"):
            await execute_query(fs, _plan(node))

    async def test_top_requires_piped(self):
        fs = _fs()
        node = TopCommand(k=5)
        with pytest.raises(ValueError, match="top requires piped input"):
            await execute_query(fs, _plan(node))

    async def test_kinds_requires_piped(self):
        fs = _fs()
        node = KindsCommand(kinds=("file",))
        with pytest.raises(ValueError, match="kinds requires piped input"):
            await execute_query(fs, _plan(node))

    async def test_intersect(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        fs.glob.return_value = _result("/a.py")
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(IntersectStage(query=GlobCommand(pattern="*.py", visibility=NO_VIS)),),
        )
        result = await execute_query(fs, _plan(node))
        assert result is not None

    async def test_intersect_requires_piped(self):
        fs = _fs()
        node = IntersectStage(query=GlobCommand(pattern="*.py", visibility=NO_VIS))
        with pytest.raises(ValueError, match="intersect requires piped input"):
            await execute_query(fs, _plan(node))

    async def test_except_requires_piped(self):
        fs = _fs()
        node = ExceptStage(query=GlobCommand(pattern="*.py", visibility=NO_VIS))
        with pytest.raises(ValueError, match="except requires piped input"):
            await execute_query(fs, _plan(node))


# ===========================================================================
# Meeting graph
# ===========================================================================


class TestMeetingGraphCommand:
    async def test_meeting_explicit_paths(self):
        fs = _fs()
        fs.meeting_subgraph.return_value = GroverResult(candidates=[])
        node = MeetingGraphCommand(paths=("/a.py", "/b.py"), minimal=False, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.meeting_subgraph.assert_called_once()

    async def test_min_meeting(self):
        fs = _fs()
        fs.min_meeting_subgraph.return_value = GroverResult(candidates=[])
        node = MeetingGraphCommand(paths=("/a.py", "/b.py"), minimal=True, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.min_meeting_subgraph.assert_called_once()


# ===========================================================================
# Semantic / vector search
# ===========================================================================


class TestSearchCommands:
    async def test_semantic_search(self):
        fs = _fs()
        node = SemanticSearchCommand(query="test", k=10, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.semantic_search.assert_called_once()

    async def test_vector_search(self):
        fs = _fs()
        node = VectorSearchCommand(vector=(0.1, 0.2), k=10, visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        fs.vector_search.assert_called_once()


# ===========================================================================
# _apply_visibility
# ===========================================================================


class TestApplyVisibility:
    def test_include_all_returns_unchanged(self):
        vis = Visibility(include_all=True, include_kinds=())
        result = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
                Candidate(path="/a.py@1", kind="version"),
            ]
        )
        filtered = _apply_visibility(result, vis, {"file"})
        assert len(filtered.candidates) == 2

    def test_default_kinds_filter(self):
        vis = Visibility(include_all=False, include_kinds=())
        result = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
                Candidate(path="/a.py@1", kind="version"),
            ]
        )
        filtered = _apply_visibility(result, vis, {"file"})
        assert len(filtered.candidates) == 1
        assert filtered.candidates[0].kind == "file"

    def test_include_kinds_extends_defaults(self):
        vis = Visibility(include_all=False, include_kinds=("version",))
        result = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
                Candidate(path="/a.py@1", kind="version"),
            ]
        )
        filtered = _apply_visibility(result, vis, {"file"})
        assert len(filtered.candidates) == 2


# ===========================================================================
# Union node
# ===========================================================================


class TestUnionNode:
    async def test_union_merges_results(self):
        fs = _fs()
        fs.read.side_effect = [_result("/a.py"), _result("/b.py")]
        node = UnionNode(
            operands=(
                ReadCommand(paths=("/a.py",)),
                ReadCommand(paths=("/b.py",)),
            )
        )
        result = await execute_query(fs, _plan(node))
        assert len(result.candidates) == 2


# ===========================================================================
# Ls command edge cases
# ===========================================================================


class TestLsCommand:
    async def test_ls_piped_plus_paths_raises(self):
        fs = _fs()
        fs.read.return_value = _result("/a.py")
        node = PipelineNode(
            source=ReadCommand(paths=("/a.py",)),
            stages=(LsCommand(paths=("/b",), visibility=NO_VIS),),
        )
        with pytest.raises(ValueError, match="cannot combine piped input"):
            await execute_query(fs, _plan(node))

    async def test_ls_piped_delegates(self):
        fs = _fs()
        fs.read.return_value = _result("/dir")
        fs.ls.return_value = GroverResult(candidates=[Candidate(path="/dir/a.py")])
        node = PipelineNode(
            source=ReadCommand(paths=("/dir",)),
            stages=(LsCommand(paths=(), visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        call_kwargs = fs.ls.call_args[1]
        assert "candidates" in call_kwargs

    async def test_ls_no_paths_defaults_to_root(self):
        fs = _fs()
        fs.ls.return_value = GroverResult(candidates=[])
        node = LsCommand(paths=(), visibility=NO_VIS)
        await execute_query(fs, _plan(node))
        call_kwargs = fs.ls.call_args[1]
        assert call_kwargs.get("path") == "/"


# ===========================================================================
# Stat / delete commands (line 95, 97)
# ===========================================================================


class TestStatDeleteCommands:
    async def test_stat_explicit_path(self):
        fs = _fs()
        from grover.query.ast import StatCommand

        fs.stat.return_value = GroverResult(candidates=[Candidate(path="/a.py")])
        node = StatCommand(paths=("/a.py",))
        await execute_query(fs, _plan(node))
        fs.stat.assert_called_once()

    async def test_delete_explicit_path(self):
        fs = _fs()
        from grover.query.ast import DeleteCommand

        fs.delete.return_value = GroverResult(candidates=[Candidate(path="/a.py")])
        node = DeleteCommand(paths=("/a.py",))
        await execute_query(fs, _plan(node))
        fs.delete.assert_called_once()


# ===========================================================================
# Mkdir execution (lines 120-123)
# ===========================================================================


class TestMkdirExecution:
    async def test_mkdir_executes(self):
        fs = _fs()
        fs.mkdir.return_value = GroverResult(candidates=[Candidate(path="/dir")])
        node = MkdirCommand(paths=("/dir", "/dir2"))
        await execute_query(fs, _plan(node))
        assert fs.mkdir.call_count == 2


# ===========================================================================
# Kinds execution (line 189)
# ===========================================================================


class TestKindsExecution:
    async def test_kinds_filters(self):
        fs = _fs()
        fs.glob.return_value = GroverResult(
            candidates=[
                Candidate(path="/a.py", kind="file"),
                Candidate(path="/dir", kind="directory"),
            ]
        )
        node = PipelineNode(
            source=GlobCommand(pattern="**/*", visibility=NO_VIS),
            stages=(KindsCommand(kinds=("file",)),),
        )
        result = await execute_query(fs, _plan(node))
        assert all(c.kind == "file" for c in result.candidates)


# ===========================================================================
# Move/copy with explicit src (lines 237-238)
# ===========================================================================


class TestExplicitTransfer:
    async def test_move_explicit_src_dest(self):
        fs = _fs()
        fs.move.return_value = GroverResult(candidates=[Candidate(path="/b.py")])
        node = MoveCommand(src="/a.py", dest="/b.py", overwrite=True)
        await execute_query(fs, _plan(node))
        fs.move.assert_called_once()

    async def test_copy_explicit_src_dest(self):
        fs = _fs()
        fs.copy.return_value = GroverResult(candidates=[Candidate(path="/b.py")])
        node = CopyCommand(src="/a.py", dest="/b.py", overwrite=True)
        await execute_query(fs, _plan(node))
        fs.copy.assert_called_once()

    async def test_copy_piped_with_candidates(self):
        fs = _fs()
        fs.glob.return_value = _result("/a.py")
        fs.copy.return_value = GroverResult(candidates=[Candidate(path="/dest/a.py")])
        node = PipelineNode(
            source=GlobCommand(pattern="*.py", visibility=NO_VIS),
            stages=(CopyCommand(dest="/dest", overwrite=True),),
        )
        await execute_query(fs, _plan(node))
        fs.copy.assert_called_once()


# ===========================================================================
# Mkconn with explicit source (line 267)
# ===========================================================================


class TestMkconnExplicitSource:
    async def test_mkconn_explicit_source(self):
        fs = _fs()
        fs.mkconn.return_value = GroverResult(candidates=[Candidate(path="/conn")])
        node = MkconnCommand(source="/a.py", connection_type="imports", target="/b.py")
        await execute_query(fs, _plan(node))
        fs.mkconn.assert_called_once()


# ===========================================================================
# Grep reading non-file candidates (line 348)
# ===========================================================================


class TestGrepNonFileRead:
    async def test_grep_reads_non_file_candidates_first(self):
        fs = _fs()
        piped = GroverResult(
            candidates=[
                Candidate(path="/dir", kind="directory", content=None),
            ]
        )
        fs.glob.return_value = piped
        fs.read.return_value = GroverResult(
            candidates=[
                Candidate(path="/dir", kind="directory", content="readme"),
            ]
        )
        fs.grep.return_value = GroverResult(candidates=[])
        node = PipelineNode(
            source=GlobCommand(pattern="*", visibility=NO_VIS),
            stages=(GrepCommand(pattern="test", case_sensitive=True, max_results=None, visibility=NO_VIS),),
        )
        await execute_query(fs, _plan(node))
        fs.read.assert_called_once()


# ===========================================================================
# Coverage: line 111 — edit with explicit paths (no pipe)
# ===========================================================================


class TestEditExplicitPaths:
    async def test_edit_with_explicit_paths_no_pipe(self):
        """Line 111: edit command with paths and no piped input calls edit."""
        fs = _fs()
        fs.edit.return_value = GroverResult(candidates=[Candidate(path="/a.py")])
        node = EditCommand(old="x", new="y", paths=("/a.py",), replace_all=False)
        await execute_query(fs, _plan(node))
        fs.edit.assert_called_once()


# ===========================================================================
# Coverage: line 461 — _preserve_under_root with root path
# ===========================================================================


class TestPreserveUnderRoot:
    def test_root_path_raises(self):
        """Line 461: passing root '/' as source raises ValueError."""
        with pytest.raises(ValueError, match="Cannot preserve the root path"):
            _preserve_under_root("/dest", "/")

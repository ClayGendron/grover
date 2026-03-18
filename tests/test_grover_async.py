"""Tests for the GroverAsync class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.internal.results import FileSearchResult, FileSearchSet
from grover.providers.graph import RustworkxGraph
from grover.providers.search.local import LocalVectorStore

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def workspace2(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace2"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    await g.add_mount(
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Lifecycle
# ==================================================================


class TestGroverAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_construction_no_args(self):
        g = GroverAsync()
        assert not g._ctx.initialized  # No mounts yet

    @pytest.mark.asyncio
    async def test_mount_sets_initialized(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "app",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        assert g._ctx.initialized
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "app",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        await g.write("/app/test.txt", "hello")
        assert (await g.exists("/app/test.txt")).message == "exists"
        await g.unmount("/app")
        # Mount should be gone
        assert not g._ctx.registry.has_mount("/app")
        await g.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "app",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        await g.close()
        await g.close()  # Should not raise


# ==================================================================
# Direct Access Mode
# ==================================================================


class TestGroverAsyncDirectAccess:
    @pytest.mark.asyncio
    async def test_write_and_read(self, grover: GroverAsync):
        assert await grover.write("/project/hello.txt", "hello world")
        result = await grover.read("/project/hello.txt")
        assert result.success
        assert result.file.content == "hello world"

    @pytest.mark.asyncio
    async def test_edit(self, grover: GroverAsync):
        await grover.write("/project/doc.txt", "old text here")
        assert await grover.edit("/project/doc.txt", "old", "new")
        result = await grover.read("/project/doc.txt")
        assert result.file.content == "new text here"

    @pytest.mark.asyncio
    async def test_delete(self, grover: GroverAsync):
        await grover.write("/project/tmp.txt", "temporary")
        assert await grover.delete("/project/tmp.txt")
        result = await grover.read("/project/tmp.txt")
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_dir(self, grover: GroverAsync):
        await grover.write("/project/a.txt", "a")
        await grover.write("/project/b.txt", "b")
        result = await grover.list_dir("/project")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "a.txt" in names
        assert "b.txt" in names

    @pytest.mark.asyncio
    async def test_exists(self, grover: GroverAsync):
        assert (await grover.exists("/project/nope.txt")).message != "exists"
        await grover.write("/project/yes.txt", "yes")
        assert (await grover.exists("/project/yes.txt")).message == "exists"

    @pytest.mark.asyncio
    async def test_write_overwrite_false_fails_when_exists(self, grover: GroverAsync):
        await grover.write("/project/exists.txt", "original")
        result = await grover.write("/project/exists.txt", "new", overwrite=False)
        assert not result.success
        # Original content should be unchanged
        assert (await grover.read("/project/exists.txt")).file.content == "original"

    @pytest.mark.asyncio
    async def test_write_overwrite_false_succeeds_for_new(self, grover: GroverAsync):
        result = await grover.write("/project/brand_new.txt", "content", overwrite=False)
        assert result.success
        assert (await grover.read("/project/brand_new.txt")).file.content == "content"

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, grover: GroverAsync):
        await grover.write("/project/multi.txt", "foo bar foo baz foo")
        result = await grover.edit("/project/multi.txt", "foo", "qux", replace_all=True)
        assert result.success
        assert (await grover.read("/project/multi.txt")).file.content == "qux bar qux baz qux"

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, grover: GroverAsync):
        lines = "\n".join(f"line {i}" for i in range(20))
        await grover.write("/project/lines.txt", lines)
        result = await grover.read("/project/lines.txt", offset=5, limit=3)
        assert result.success
        content = result.file.content
        assert content is not None
        assert "line 5" in content
        assert "line 7" in content
        assert "line 8" not in content


# ==================================================================
# Multi-Mount CRUD
# ==================================================================


class TestGroverAsyncMultiMount:
    @pytest.mark.asyncio
    async def test_two_mounts(self, workspace: Path, workspace2: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "app",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_app"),
            embedding_provider=FakeProvider(),
        )
        await g.add_mount(
            "data",
            filesystem=LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_data"),
            embedding_provider=FakeProvider(),
        )

        await g.write("/app/code.txt", "code content")
        await g.write("/data/doc.txt", "doc content")

        assert (await g.read("/app/code.txt")).file.content == "code content"
        assert (await g.read("/data/doc.txt")).file.content == "doc content"

        # List root should show both mounts (but not .grover)
        result = await g.list_dir()
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "app" in names
        assert "data" in names
        assert ".grover" not in names

        await g.close()

    @pytest.mark.asyncio
    async def test_isolation_between_mounts(self, workspace: Path, workspace2: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        lfs_a = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a")
        lfs_b = LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b")
        await g.add_mount("a", filesystem=lfs_a, embedding_provider=FakeProvider())
        await g.add_mount("b", filesystem=lfs_b, embedding_provider=FakeProvider())

        await g.write("/a/file.txt", "in mount a")
        assert (await g.exists("/a/file.txt")).message == "exists"
        assert (await g.exists("/b/file.txt")).message != "exists"

        await g.close()


# ==================================================================
# Index
# ==================================================================


class TestGroverAsyncIndex:
    @pytest.mark.asyncio
    async def test_index_scans_files(self, grover: GroverAsync, workspace: Path):
        (workspace / "one.py").write_text("def one():\n    return 1\n")
        (workspace / "two.py").write_text("def two():\n    return 2\n")
        stats = await grover.index()
        assert stats["files_scanned"] >= 2

    @pytest.mark.asyncio
    async def test_index_specific_mount(
        self,
        workspace: Path,
        workspace2: Path,
        tmp_path: Path,
    ):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        lfs_a = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a")
        lfs_b = LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b")
        await g.add_mount("a", filesystem=lfs_a, embedding_provider=FakeProvider())
        await g.add_mount("b", filesystem=lfs_b, embedding_provider=FakeProvider())

        (workspace / "a.py").write_text("def a():\n    pass\n")
        (workspace2 / "b.py").write_text("def b():\n    pass\n")

        stats = await g.index("/a")
        # Should only index mount /a
        assert stats["files_scanned"] >= 1
        assert g.get_graph().has_node("/a/a.py")

        await g.close()


# ==================================================================
# Persistence
# ==================================================================


# ==================================================================
# Properties
# ==================================================================


class TestGroverAsyncProperties:
    @pytest.mark.asyncio
    async def test_get_graph(self, grover: GroverAsync):
        assert isinstance(grover.get_graph(), RustworkxGraph)


# ==================================================================
# Graph Query Wrappers
# ==================================================================


class TestGroverAsyncGraphQueries:
    @pytest.mark.asyncio
    async def test_predecessors_returns_graph_result(self, grover: GroverAsync):
        await grover.write("/project/lib.py", "def helper():\n    return 42\n")
        await grover.write(
            "/project/main.py",
            "from lib import helper\n\ndef run():\n    return helper()\n",
        )
        await grover.flush()
        result = await grover.predecessors(FileSearchSet.from_paths(["/project/lib.py"]))
        # main.py imports lib.py, so main.py is a predecessor
        # The graph stores "imports" edges from analyzer
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_successors_returns_graph_result(self, grover: GroverAsync):
        await grover.write("/project/dep.py", "def util():\n    pass\n")
        await grover.write(
            "/project/consumer.py",
            "from dep import util\n\ndef main():\n    util()\n",
        )
        await grover.flush()
        result = await grover.successors(FileSearchSet.from_paths(["/project/consumer.py"]))
        assert isinstance(result, FileSearchResult)
        assert result.success is True


# ==================================================================
# Version operations (file_ops)
# ==================================================================


class TestGroverAsyncVersionOps:
    @pytest.mark.asyncio
    async def test_read_version(self, grover: GroverAsync):
        """read_version returns the content of a specific version."""
        await grover.write("/project/doc.txt", "version one")
        await grover.write("/project/doc.txt", "version two")
        result = await grover.read_version("/project/doc.txt", 1)
        assert result.success is True
        assert result.file.content == "version one"

    @pytest.mark.asyncio
    async def test_diff_versions_basic(self, grover: GroverAsync):
        """diff_versions computes a unified diff between two versions."""
        await grover.write("/project/doc.txt", "hello\n")
        await grover.write("/project/doc.txt", "hello world\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 2)
        assert result.success is True
        assert result.file.content is not None
        assert result.file.content != ""
        assert "-hello" in result.file.content or "+hello world" in result.file.content

    @pytest.mark.asyncio
    async def test_diff_versions_same_version(self, grover: GroverAsync):
        """diff_versions with same version returns empty diff."""
        await grover.write("/project/doc.txt", "content\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 1)
        assert result.success is True
        assert result.file.content == ""

    @pytest.mark.asyncio
    async def test_diff_versions_invalid_version(self, grover: GroverAsync):
        """diff_versions with nonexistent version returns failure."""
        await grover.write("/project/doc.txt", "content\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 999)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_tree_still_works(self, grover: GroverAsync):
        """tree() still works after moving from search_ops to file_ops."""
        await grover.write("/project/a.py", "a\n")
        await grover.write("/project/sub/b.py", "b\n")
        result = await grover.tree("/project")
        assert result.success is True
        assert len(result) >= 2


# ==================================================================
# Phase 3 — Graph Operations (consolidated)
# ==================================================================


class TestGroverAsyncGraphAlgorithms:
    """Tests for new graph algorithm facades."""

    @pytest.mark.asyncio
    async def test_ancestors(self, grover: GroverAsync):
        await grover.write("/project/base.py", "X = 1\n")
        await grover.write(
            "/project/mid.py",
            "from base import X\n\ndef mid():\n    return X\n",
        )
        await grover.write(
            "/project/top.py",
            "from mid import mid\n\ndef top():\n    return mid()\n",
        )
        await grover.flush()
        result = await grover.ancestors(FileSearchSet.from_paths(["/project/base.py"]))
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_descendants(self, grover: GroverAsync):
        await grover.write("/project/root.py", "X = 1\n")
        await grover.write(
            "/project/child.py",
            "from root import X\n\ndef child():\n    return X\n",
        )
        await grover.flush()
        result = await grover.descendants(FileSearchSet.from_paths(["/project/root.py"]))
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_ego_graph(self, grover: GroverAsync):
        await grover.write("/project/center.py", "X = 1\n")
        await grover.write(
            "/project/near.py",
            "from center import X\n\ndef near():\n    return X\n",
        )
        await grover.flush()
        result = await grover.ego_graph(FileSearchSet.from_paths(["/project/center.py"]), max_depth=1)
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_min_meeting_subgraph_from_candidates(self, grover: GroverAsync):
        await grover.write("/project/x.py", "X = 1\n")
        await grover.write(
            "/project/y.py",
            "from x import X\n\ndef y():\n    return X\n",
        )
        await grover.flush()
        candidates = await grover.glob("*.py", "/project")
        result = await grover.min_meeting_subgraph(candidates)
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_hits_two_evidence_records(self, grover: GroverAsync):
        await grover.write("/project/h1.py", "X = 1\n")
        await grover.write(
            "/project/h2.py",
            "from h1 import X\n\ndef h2():\n    return X\n",
        )
        await grover.flush()
        result = await grover.hits(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True
        if len(result) > 0:
            # Each candidate should have one evidence record with authority and hub scores
            for f in result.files:
                ops = [e.operation for e in f.evidence]
                assert "hits" in ops
                hits_ev = next(e for e in f.evidence if e.operation == "hits")
                assert "authority" in hits_ev.scores
                assert "hub" in hits_ev.scores

    @pytest.mark.asyncio
    async def test_betweenness_centrality(self, grover: GroverAsync):
        await grover.write("/project/bc.py", "X = 1\n")
        await grover.flush()
        result = await grover.betweenness_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_closeness_centrality(self, grover: GroverAsync):
        await grover.write("/project/cc.py", "X = 1\n")
        await grover.flush()
        result = await grover.closeness_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_katz_centrality(self, grover: GroverAsync):
        await grover.write("/project/kc.py", "X = 1\n")
        await grover.flush()
        result = await grover.katz_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_degree_centrality(self, grover: GroverAsync):
        await grover.write("/project/dc.py", "X = 1\n")
        await grover.flush()
        result = await grover.degree_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_in_degree_centrality(self, grover: GroverAsync):
        await grover.write("/project/idc.py", "X = 1\n")
        await grover.flush()
        result = await grover.in_degree_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_out_degree_centrality(self, grover: GroverAsync):
        await grover.write("/project/odc.py", "X = 1\n")
        await grover.flush()
        result = await grover.out_degree_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_add_connection_on_graph_ops(self, grover: GroverAsync):
        """add_connection works from GraphOpsMixin (formerly ConnectionMixin)."""
        await grover.write("/project/ac_src.py", "X = 1\n")
        await grover.write("/project/ac_tgt.py", "Y = 2\n")
        await grover.flush()
        result = await grover.add_connection("/project/ac_src.py", "/project/ac_tgt.py", "imports")
        assert result.success is True
        await grover.flush()
        graph = grover.get_graph()
        assert graph.has_edge("/project/ac_src.py", "/project/ac_tgt.py")

    @pytest.mark.asyncio
    async def test_delete_connection_on_graph_ops(self, grover: GroverAsync):
        """delete_connection works from GraphOpsMixin (formerly ConnectionMixin)."""
        await grover.write("/project/dc_src.py", "X = 1\n")
        await grover.write("/project/dc_tgt.py", "Y = 2\n")
        await grover.flush()
        await grover.add_connection("/project/dc_src.py", "/project/dc_tgt.py", "imports")
        await grover.flush()
        result = await grover.delete_connection("/project/dc_src.py", "/project/dc_tgt.py", connection_type="imports")
        assert result.success is True
        await grover.flush()
        graph = grover.get_graph()
        assert not graph.has_edge("/project/dc_src.py", "/project/dc_tgt.py")


# ------------------------------------------------------------------
# Phase 4 - candidates filtering on search methods
# ------------------------------------------------------------------


class TestGroverAsyncSearchCandidates:
    """Tests for candidates filtering on glob, grep, vector_search, etc."""

    @pytest.fixture(autouse=True)
    async def _setup(self, grover: GroverAsync):
        """Write three files so glob/grep have something to match."""
        await grover.write("/project/alpha.py", "HELLO = 1\n")
        await grover.write("/project/beta.py", "WORLD = 2\n")
        await grover.write("/project/gamma.py", "HELLO = 3\n")
        await grover.flush()
        self.grover = grover

    @pytest.mark.asyncio
    async def test_glob_with_candidates_filter(self):
        """glob with candidates returns only files in the candidate set."""
        # Build a candidate set with only alpha.py
        full = await self.grover.glob("*.py", "/project")
        assert len(full) >= 3
        candidates = await self.grover.glob("alpha*", "/project")
        assert len(candidates) >= 1

        filtered = await self.grover.glob("*.py", "/project", candidates=candidates)
        assert isinstance(filtered, FileSearchResult)
        paths = {f.path for f in filtered.files}
        assert "/project/alpha.py" in paths
        assert "/project/beta.py" not in paths
        assert "/project/gamma.py" not in paths

    @pytest.mark.asyncio
    async def test_glob_without_candidates(self):
        """glob without candidates returns all matches (backward compat)."""
        result = await self.grover.glob("*.py", "/project")
        assert isinstance(result, FileSearchResult)
        assert len(result) >= 3

    @pytest.mark.asyncio
    async def test_grep_with_candidates_filter(self):
        """grep with candidates filters results to candidate paths."""
        # HELLO appears in alpha.py and gamma.py
        full = await self.grover.grep("HELLO", "/project")
        full_paths = {f.path for f in full.files}
        assert "/project/alpha.py" in full_paths
        assert "/project/gamma.py" in full_paths

        # Filter to only alpha.py
        candidates = await self.grover.glob("alpha*", "/project")
        filtered = await self.grover.grep("HELLO", "/project", candidates=candidates)
        assert isinstance(filtered, FileSearchResult)
        filtered_paths = {f.path for f in filtered.files}
        assert "/project/alpha.py" in filtered_paths
        assert "/project/gamma.py" not in filtered_paths

    @pytest.mark.asyncio
    async def test_candidates_preserves_result_type(self):
        """Filtered GlobResult is still a GlobResult instance."""
        candidates = await self.grover.glob("alpha*", "/project")
        result = await self.grover.glob("*.py", "/project", candidates=candidates)
        assert isinstance(result, FileSearchResult)
        # GrepResult type preserved too
        grep_result = await self.grover.grep("HELLO", "/project", candidates=candidates)
        assert isinstance(grep_result, FileSearchResult)

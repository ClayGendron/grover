"""Tests for the Grover integration class (sync wrapper)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover
from grover.models.internal.results import FileSearchResult, FileSearchSet
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
def grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover()
    g.add_mount(
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g
    g.close()


# ==================================================================
# Construction
# ==================================================================


class TestGroverConstruction:
    def test_mount_first_api(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = Grover()
        g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        try:
            assert g._async._ctx.initialized
        finally:
            g.close()

    def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = Grover()
        g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        g.close()
        g.close()  # Should not raise
        assert g._closed


# ==================================================================
# Filesystem
# ==================================================================


class TestGroverFilesystem:
    def test_write_and_read(self, grover: Grover):
        assert grover.write("/project/hello.txt", "hello world")
        result = grover.read("/project/hello.txt")
        assert result.success
        assert result.file.content == "hello world"

    def test_edit(self, grover: Grover):
        grover.write("/project/doc.txt", "old text here")
        assert grover.edit("/project/doc.txt", "old", "new")
        assert grover.read("/project/doc.txt").file.content == "new text here"

    def test_delete(self, grover: Grover):
        grover.write("/project/tmp.txt", "temporary")
        assert grover.delete("/project/tmp.txt")
        assert not grover.read("/project/tmp.txt").success

    def test_list_dir(self, grover: Grover):
        grover.write("/project/a.txt", "a")
        grover.write("/project/b.txt", "b")
        result = grover.list_dir("/project")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "a.txt" in names
        assert "b.txt" in names

    def test_exists(self, grover: Grover):
        assert grover.exists("/project/nope.txt").message != "exists"
        grover.write("/project/yes.txt", "yes")
        assert grover.exists("/project/yes.txt").message == "exists"

    def test_write_overwrite_false_fails_when_exists(self, grover: Grover):
        grover.write("/project/exists.txt", "original")
        result = grover.write("/project/exists.txt", "new", overwrite=False)
        assert not result.success
        # Original content should be unchanged
        assert grover.read("/project/exists.txt").file.content == "original"

    def test_write_overwrite_false_succeeds_for_new(self, grover: Grover):
        result = grover.write("/project/brand_new.txt", "content", overwrite=False)
        assert result.success
        assert grover.read("/project/brand_new.txt").file.content == "content"

    def test_edit_replace_all(self, grover: Grover):
        grover.write("/project/multi.txt", "foo bar foo baz foo")
        result = grover.edit("/project/multi.txt", "foo", "qux", replace_all=True)
        assert result.success
        assert grover.read("/project/multi.txt").file.content == "qux bar qux baz qux"


# ==================================================================
# Index
# ==================================================================


class TestGroverIndex:
    def test_index_scans_files(self, grover: Grover, workspace: Path):
        # Write files directly to disk so index() discovers them
        (workspace / "one.py").write_text("def one():\n    return 1\n")
        (workspace / "two.py").write_text("def two():\n    return 2\n")
        stats = grover.index()
        assert stats["files_scanned"] >= 2

    def test_index_creates_chunks(self, grover: Grover, workspace: Path):
        (workspace / "funcs.py").write_text("def alpha():\n    pass\n\ndef beta():\n    pass\n")
        stats = grover.index()
        assert stats["chunks_created"] >= 2

    def test_index_builds_graph(self, grover: Grover, workspace: Path):
        (workspace / "main.py").write_text("def main():\n    pass\n")
        grover.index()
        assert grover.get_graph().has_node("/project/main.py")

    def test_index_returns_stats(self, grover: Grover, workspace: Path):
        (workspace / "a.py").write_text("def a():\n    pass\n")
        stats = grover.index()
        assert "files_scanned" in stats
        assert "chunks_created" in stats
        assert "edges_added" in stats

    def test_index_skips_grover_dir(self, grover: Grover, workspace: Path):
        # Create a .grover subdirectory with a file
        grover_dir = workspace / ".grover" / "chunks"
        grover_dir.mkdir(parents=True)
        (grover_dir / "stale.txt").write_text("stale chunk")
        (workspace / "real.py").write_text("def real():\n    pass\n")

        grover.index()
        # The .grover file should NOT be indexed
        assert not grover.get_graph().has_node("/project/.grover/chunks/stale.txt")
        # But the real file should be
        assert grover.get_graph().has_node("/project/real.py")


# ==================================================================
# Tree (sync)
# ==================================================================


def test_tree(grover: Grover):
    """tree() works on sync facade."""
    grover.write("/project/a.py", "a\n")
    result = grover.tree("/project")
    assert result.success is True
    assert len(result) >= 1


# ==================================================================
# Phase 3 — Graph Operations (consolidated)
# ==================================================================


class TestGroverGraphAlgorithms:
    """Tests for sync graph algorithm facades."""

    def test_ancestors(self, grover: Grover):
        grover.write("/project/base.py", "X = 1\n")
        grover.write(
            "/project/child.py",
            "from base import X\n\ndef child():\n    return X\n",
        )
        grover.flush()
        result = grover.ancestors(FileSearchSet.from_paths(["/project/base.py"]))
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_descendants(self, grover: Grover):
        grover.write("/project/root.py", "X = 1\n")
        grover.write(
            "/project/leaf.py",
            "from root import X\n\ndef leaf():\n    return X\n",
        )
        grover.flush()
        result = grover.descendants(FileSearchSet.from_paths(["/project/root.py"]))
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_ego_graph(self, grover: Grover):
        grover.write("/project/ego.py", "X = 1\n")
        grover.flush()
        result = grover.ego_graph(FileSearchSet.from_paths(["/project/ego.py"]), max_depth=1)
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_pagerank(self, grover: Grover):
        grover.write("/project/pr.py", "X = 1\n")
        grover.flush()
        result = grover.pagerank(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_betweenness_centrality(self, grover: Grover):
        grover.write("/project/bw.py", "X = 1\n")
        grover.flush()
        result = grover.betweenness_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_hits(self, grover: Grover):
        grover.write("/project/ht.py", "X = 1\n")
        grover.flush()
        result = grover.hits(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_degree_centrality(self, grover: Grover):
        grover.write("/project/deg.py", "X = 1\n")
        grover.flush()
        result = grover.degree_centrality(FileSearchSet())
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    def test_add_connection_sync(self, grover: Grover):
        grover.write("/project/conn_a.py", "X = 1\n")
        grover.write("/project/conn_b.py", "Y = 2\n")
        grover.flush()
        result = grover.add_connection("/project/conn_a.py", "/project/conn_b.py", "imports")
        assert result.success is True
        grover.flush()
        assert grover.get_graph().has_edge("/project/conn_a.py", "/project/conn_b.py")

    def test_delete_connection_sync(self, grover: Grover):
        grover.write("/project/dconn_a.py", "X = 1\n")
        grover.write("/project/dconn_b.py", "Y = 2\n")
        grover.flush()
        grover.add_connection("/project/dconn_a.py", "/project/dconn_b.py", "imports")
        grover.flush()
        result = grover.delete_connection("/project/dconn_a.py", "/project/dconn_b.py", connection_type="imports")
        assert result.success is True
        grover.flush()
        assert not grover.get_graph().has_edge("/project/dconn_a.py", "/project/dconn_b.py")


# ------------------------------------------------------------------
# Phase 4 - candidates filtering on search methods (sync)
# ------------------------------------------------------------------


class TestGroverSearchCandidates:
    """Tests for candidates filtering on glob/grep (sync wrapper)."""

    def test_glob_with_candidates_filter(self, grover: Grover):
        grover.write("/project/alpha.py", "HELLO = 1\n")
        grover.write("/project/beta.py", "WORLD = 2\n")
        grover.write("/project/gamma.py", "HELLO = 3\n")
        grover.flush()

        candidates = grover.glob("alpha*", "/project")
        filtered = grover.glob("*.py", "/project", candidates=candidates)
        assert isinstance(filtered, FileSearchResult)
        paths = {f.path for f in filtered.files}
        assert "/project/alpha.py" in paths
        assert "/project/beta.py" not in paths

    def test_grep_with_candidates_filter(self, grover: Grover):
        grover.write("/project/alpha.py", "HELLO = 1\n")
        grover.write("/project/gamma.py", "HELLO = 3\n")
        grover.flush()

        candidates = grover.glob("alpha*", "/project")
        filtered = grover.grep("HELLO", "/project", candidates=candidates)
        assert isinstance(filtered, FileSearchResult)
        paths = {f.path for f in filtered.files}
        assert "/project/alpha.py" in paths
        assert "/project/gamma.py" not in paths

    def test_candidates_preserves_result_type(self, grover: Grover):
        grover.write("/project/alpha.py", "X = 1\n")
        grover.flush()
        candidates = grover.glob("alpha*", "/project")
        result = grover.glob("*.py", "/project", candidates=candidates)
        assert isinstance(result, FileSearchResult)

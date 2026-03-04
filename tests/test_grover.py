"""Tests for the Grover integration class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover
from grover.providers.graph import RustworkxGraph
from grover.providers.search.local import LocalVectorStore
from grover.results import (
    AncestorsResult,
    BetweennessResult,
    CommonNeighborsResult,
    DegreeResult,
    DescendantsResult,
    EgoGraphResult,
    GlobResult,
    GrepResult,
    HasPathResult,
    HitsResult,
    PageRankResult,
    PredecessorsResult,
    ShortestPathResult,
    SuccessorsResult,
    VectorSearchResult,
)

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
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g
    g.close()


@pytest.fixture
def grover_no_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    """Grover without search to test graceful degradation."""
    data = tmp_path / "grover_data"
    g = Grover()
    g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
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
            "/project",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
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
            "/project",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
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
        assert result.content == "hello world"

    def test_edit(self, grover: Grover):
        grover.write("/project/doc.txt", "old text here")
        assert grover.edit("/project/doc.txt", "old", "new")
        assert grover.read("/project/doc.txt").content == "new text here"

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
        assert not grover.exists("/project/nope.txt").exists
        grover.write("/project/yes.txt", "yes")
        assert grover.exists("/project/yes.txt").exists

    def test_write_overwrite_false_fails_when_exists(self, grover: Grover):
        grover.write("/project/exists.txt", "original")
        result = grover.write("/project/exists.txt", "new", overwrite=False)
        assert not result.success
        # Original content should be unchanged
        assert grover.read("/project/exists.txt").content == "original"

    def test_write_overwrite_false_succeeds_for_new(self, grover: Grover):
        result = grover.write("/project/brand_new.txt", "content", overwrite=False)
        assert result.success
        assert grover.read("/project/brand_new.txt").content == "content"

    def test_edit_replace_all(self, grover: Grover):
        grover.write("/project/multi.txt", "foo bar foo baz foo")
        result = grover.edit("/project/multi.txt", "foo", "qux", replace_all=True)
        assert result.success
        assert grover.read("/project/multi.txt").content == "qux bar qux baz qux"

    def test_read_with_offset_and_limit(self, grover: Grover):
        lines = "\n".join(f"line {i}" for i in range(20))
        grover.write("/project/lines.txt", lines)
        result = grover.read("/project/lines.txt", offset=5, limit=3)
        assert result.success
        content = result.content
        assert content is not None
        assert "line 5" in content
        assert "line 7" in content
        assert "line 8" not in content


# ==================================================================
# Graph
# ==================================================================


class TestGroverGraph:
    def test_get_graph(self, grover: Grover):
        assert isinstance(grover.get_graph(), RustworkxGraph)
        assert grover.get_graph() is grover._async.get_graph()

    def test_predecessors_after_write(self, grover: Grover):
        code = 'import os\n\ndef hello():\n    return "hi"\n'
        grover.write("/project/app.py", code)
        grover.flush()
        # File should be in graph now
        assert grover.get_graph().has_node("/project/app.py")
        # Check predecessors doesn't crash (may be empty if no other file points to it)
        result = grover.predecessors("/project/app.py")
        assert isinstance(result, PredecessorsResult)
        assert result.success is True

    def test_successors_after_write(self, grover: Grover):
        code = 'def greet():\n    return "hello"\n'
        grover.write("/project/greet.py", code)
        grover.flush()
        # The file should have "contains" edges to its chunks
        result = grover.successors("/project/greet.py")
        assert isinstance(result, SuccessorsResult)
        assert result.success is True
        # Should contain the greet function chunk
        assert len(result) >= 1

    def test_contains_via_graph_provider(self, grover: Grover):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        grover.write("/project/funcs.py", code)
        grover.flush()
        refs = grover.get_graph().contains("/project/funcs.py")
        assert len(refs) >= 2
        assert any("foo" in r.path for r in refs)
        assert any("bar" in r.path for r in refs)


# ==================================================================
# Search
# ==================================================================


class TestGroverSearch:
    def test_vector_search_after_write(self, grover: Grover):
        code = 'def authenticate_user():\n    """Verify user credentials."""\n    pass\n'
        grover.write("/project/auth.py", code)
        grover.flush()
        result = grover.vector_search("authenticate")
        assert isinstance(result, VectorSearchResult)
        assert result.success is True
        assert len(result) >= 1

    def test_vector_search_returns_vector_search_result(self, grover: Grover):
        grover.write("/project/data.txt", "important data content")
        result = grover.vector_search("data")
        assert isinstance(result, VectorSearchResult)
        assert result.success is True

    def test_vector_search_empty(self, grover: Grover):
        result = grover.vector_search("nonexistent query")
        assert isinstance(result, VectorSearchResult)
        assert len(result) == 0

    def test_vector_search_returns_failure_without_provider(self, grover_no_search: Grover):
        mounts = grover_no_search._async._ctx.registry.list_visible_mounts()
        has_search = any(getattr(m.filesystem, "search_provider", None) is not None for m in mounts)
        if has_search:
            pytest.skip("search provider is installed; search available")
        result = grover_no_search.vector_search("anything")
        assert result.success is False
        assert "not available" in result.message


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
# Event Handlers
# ==================================================================


class TestGroverEventHandlers:
    def test_write_updates_graph(self, grover: Grover):
        grover.write("/project/mod.py", "def work():\n    pass\n")
        grover.flush()
        assert grover.get_graph().has_node("/project/mod.py")

    def test_write_updates_search(self, grover: Grover):
        grover.write(
            "/project/search_me.py",
            'def searchable():\n    """A unique searchable function."""\n    pass\n',
        )
        grover.flush()
        result = grover.vector_search("searchable")
        assert len(result) >= 1

    def test_delete_removes_from_graph(self, grover: Grover):
        grover.write("/project/gone.py", "def gone():\n    pass\n")
        grover.flush()
        assert grover.get_graph().has_node("/project/gone.py")
        grover.delete("/project/gone.py")
        grover.flush()
        assert not grover.get_graph().has_node("/project/gone.py")

    def test_delete_removes_from_search(self, grover: Grover):
        grover.write(
            "/project/vanish.py",
            "def vanishing_function():\n    pass\n",
        )
        grover.flush()
        # Verify it's in search (search provider is now on the filesystem)
        mount = next(
            m for m in grover._async._ctx.registry.list_visible_mounts() if m.path == "/project"
        )
        sp = mount.filesystem.search_provider
        assert sp is not None
        assert sp.has("/project/vanish.py#vanishing_function")
        grover.delete("/project/vanish.py")
        grover.flush()
        # Should be removed from search
        assert not sp.has("/project/vanish.py#vanishing_function")


# ==================================================================
# Persistence
# ==================================================================


class TestGroverPersistence:
    def test_save_persists_graph(self, grover: Grover, workspace: Path):
        grover.write("/project/persist.py", "def persist():\n    pass\n")
        grover.save()

        # Graph node should exist after save
        assert grover.get_graph().has_node("/project/persist.py")


# ==================================================================
# Edge Cases
# ==================================================================


class TestGroverEdgeCases:
    def test_unsupported_file_type_embedded(self, grover: Grover):
        """Non-analyzable files should be embedded as whole files."""
        grover.write("/project/readme.txt", "This is a readme file")
        grover.flush()
        assert grover.get_graph().has_node("/project/readme.txt")
        # Should be searchable as whole file
        result = grover.vector_search("readme")
        assert len(result) >= 1

    def test_empty_file_no_crash(self, grover: Grover):
        """Empty files should not crash the pipeline."""
        grover.write("/project/empty.py", "")
        # Should not raise — file may or may not be in graph

    def test_syntax_error_no_crash(self, grover: Grover):
        """Files with syntax errors should not crash the pipeline."""
        bad_code = "def broken(\n    # missing close paren and body"
        grover.write("/project/bad.py", bad_code)
        grover.flush()
        # Should not raise
        assert grover.get_graph().has_node("/project/bad.py")


# ==================================================================
# Sync authenticated mount + sharing
# ==================================================================


@pytest.fixture
def auth_grover(tmp_path: Path) -> Iterator[Grover]:
    """Sync Grover with a UserScopedFileSystem backend."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from grover.backends.user_scoped import UserScopedFileSystem
    from grover.models.share import FileShare

    g = Grover()
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    backend = UserScopedFileSystem(share_model=FileShare)
    g.add_mount("/ws", backend, engine=engine, embedding_provider=FakeProvider())
    yield g
    g.close()


class TestGroverSyncAuthenticated:
    def test_authenticated_mount(self, auth_grover: Grover):
        auth_grover.write("/ws/notes.md", "hello", user_id="alice")
        result = auth_grover.read("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert result.content == "hello"

    def test_share_unshare(self, auth_grover: Grover):
        auth_grover.write("/ws/notes.md", "data", user_id="alice")
        share_result = auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        assert share_result.success is True

        unshare_result = auth_grover.unshare("/ws/notes.md", "bob", user_id="alice")
        assert unshare_result.success is True

    def test_list_shares(self, auth_grover: Grover):
        auth_grover.write("/ws/notes.md", "data", user_id="alice")
        auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        result = auth_grover.list_shares("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert len(result) == 1

    def test_list_shared_with_me(self, auth_grover: Grover):
        auth_grover.write("/ws/a.md", "a", user_id="alice")
        auth_grover.share("/ws/a.md", "bob", "read", user_id="alice")
        result = auth_grover.list_shared_with_me(user_id="bob")
        assert result.success is True
        assert len(result) == 1
        # Path should be an @shared path, not a raw stored path
        assert result.file_candidates[0].path == "/ws/@shared/alice/a.md"

    def test_move_and_copy(self, auth_grover: Grover):
        auth_grover.write("/ws/src.md", "content", user_id="alice")
        copy_result = auth_grover.copy("/ws/src.md", "/ws/copy.md", user_id="alice")
        assert copy_result.success is True
        move_result = auth_grover.move("/ws/src.md", "/ws/moved.md", user_id="alice")
        assert move_result.success is True


# ==================================================================
# Version operations (sync)
# ==================================================================


class TestGroverVersionOps:
    def test_read_version(self, grover: Grover):
        """Sync read_version returns the content of a specific version."""
        grover.write("/project/doc.txt", "version one")
        grover.write("/project/doc.txt", "version two")
        result = grover.read_version("/project/doc.txt", 1)
        assert result.success is True
        assert result.content == "version one"

    def test_diff_versions_basic(self, grover: Grover):
        """Sync diff_versions computes a unified diff."""
        grover.write("/project/doc.txt", "hello\n")
        grover.write("/project/doc.txt", "hello world\n")
        result = grover.diff_versions("/project/doc.txt", 1, 2)
        assert result.success is True
        assert result.version_a == 1
        assert result.version_b == 2
        assert result.diff != ""

    def test_diff_versions_invalid_version(self, grover: Grover):
        """Sync diff_versions with nonexistent version returns failure."""
        grover.write("/project/doc.txt", "content\n")
        result = grover.diff_versions("/project/doc.txt", 1, 999)
        assert result.success is False

    def test_tree(self, grover: Grover):
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
        result = grover.ancestors("/project/base.py")
        assert isinstance(result, AncestorsResult)
        assert result.success is True

    def test_descendants(self, grover: Grover):
        grover.write("/project/root.py", "X = 1\n")
        grover.write(
            "/project/leaf.py",
            "from root import X\n\ndef leaf():\n    return X\n",
        )
        grover.flush()
        result = grover.descendants("/project/root.py")
        assert isinstance(result, DescendantsResult)
        assert result.success is True

    def test_shortest_path(self, grover: Grover):
        grover.write("/project/sp_a.py", "X = 1\n")
        grover.write(
            "/project/sp_b.py",
            "from sp_a import X\n\ndef sp_b():\n    return X\n",
        )
        grover.flush()
        result = grover.shortest_path("/project/sp_b.py", "/project/sp_a.py")
        assert isinstance(result, ShortestPathResult)
        assert result.success is True

    def test_has_path_no_path(self, grover: Grover):
        grover.write("/project/island_x.py", "X = 1\n")
        grover.write("/project/island_y.py", "Y = 2\n")
        grover.flush()
        result = grover.has_path("/project/island_x.py", "/project/island_y.py")
        assert isinstance(result, HasPathResult)
        assert bool(result) is False

    def test_ego_graph(self, grover: Grover):
        grover.write("/project/ego.py", "X = 1\n")
        grover.flush()
        result = grover.ego_graph("/project/ego.py", max_depth=1)
        assert isinstance(result, EgoGraphResult)
        assert result.success is True

    def test_pagerank(self, grover: Grover):
        grover.write("/project/pr.py", "X = 1\n")
        grover.flush()
        result = grover.pagerank()
        assert isinstance(result, PageRankResult)
        assert result.success is True

    def test_betweenness_centrality(self, grover: Grover):
        grover.write("/project/bw.py", "X = 1\n")
        grover.flush()
        result = grover.betweenness_centrality()
        assert isinstance(result, BetweennessResult)
        assert result.success is True

    def test_hits(self, grover: Grover):
        grover.write("/project/ht.py", "X = 1\n")
        grover.flush()
        result = grover.hits()
        assert isinstance(result, HitsResult)
        assert result.success is True

    def test_degree_centrality(self, grover: Grover):
        grover.write("/project/deg.py", "X = 1\n")
        grover.flush()
        result = grover.degree_centrality()
        assert isinstance(result, DegreeResult)
        assert result.success is True

    def test_common_neighbors(self, grover: Grover):
        grover.write("/project/cn_shared.py", "S = 1\n")
        grover.write(
            "/project/cn_a.py",
            "from cn_shared import S\n\ndef a():\n    return S\n",
        )
        grover.write(
            "/project/cn_b.py",
            "from cn_shared import S\n\ndef b():\n    return S\n",
        )
        grover.flush()
        result = grover.common_neighbors("/project/cn_a.py", "/project/cn_b.py")
        assert isinstance(result, CommonNeighborsResult)
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
        result = grover.delete_connection(
            "/project/dconn_a.py", "/project/dconn_b.py", connection_type="imports"
        )
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
        assert isinstance(filtered, GlobResult)
        paths = {c.path for c in filtered.file_candidates}
        assert "/project/alpha.py" in paths
        assert "/project/beta.py" not in paths

    def test_grep_with_candidates_filter(self, grover: Grover):
        grover.write("/project/alpha.py", "HELLO = 1\n")
        grover.write("/project/gamma.py", "HELLO = 3\n")
        grover.flush()

        candidates = grover.glob("alpha*", "/project")
        filtered = grover.grep("HELLO", "/project", candidates=candidates)
        assert isinstance(filtered, GrepResult)
        paths = {c.path for c in filtered.file_candidates}
        assert "/project/alpha.py" in paths
        assert "/project/gamma.py" not in paths

    def test_candidates_preserves_result_type(self, grover: Grover):
        grover.write("/project/alpha.py", "X = 1\n")
        grover.flush()
        candidates = grover.glob("alpha*", "/project")
        result = grover.glob("*.py", "/project", candidates=candidates)
        assert type(result) is GlobResult

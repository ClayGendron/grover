"""Tests for Phase 3: Chunk storage migration (DB rows, not VFS files)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.fs.local_fs import LocalFileSystem
from grover.fs.protocol import GroverFileSystem
from grover.fs.providers.search.local import LocalVectorStore
from grover.grover_async import GroverAsync

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _get_mount(g: GroverAsync, mount_path: str):
    return next(m for m in g._ctx.registry.list_visible_mounts() if m.path == mount_path)


PYTHON_CODE = """\
def alpha():
    \"\"\"Alpha function.\"\"\"
    return 1


def beta():
    \"\"\"Beta function.\"\"\"
    return 2
"""


# ==================================================================
# Chunk rows written to DB
# ==================================================================


class TestAnalyzeWritesChunkRows:
    @pytest.mark.asyncio
    async def test_analyze_writes_chunk_rows(self, grover: GroverAsync):
        """Writing a .py file should create chunk rows in the DB."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        mount = _get_mount(grover, "/project")
        backend = mount.filesystem
        assert isinstance(backend, GroverFileSystem)

        async with grover._ctx.session_for(mount) as sess:
            result = await backend.list_file_chunks("/project/funcs.py", session=sess)

        assert len(result.chunks) >= 2
        paths = [c.path for c in result.chunks]
        assert any("alpha" in p for p in paths)
        assert any("beta" in p for p in paths)


# ==================================================================
# Graph nodes and edges still created
# ==================================================================


class TestAnalyzeCreatesGraphState:
    @pytest.mark.asyncio
    async def test_analyze_creates_graph_nodes(self, grover: GroverAsync):
        """Chunk nodes should have parent_path attribute pointing to parent file."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        graph = grover.get_graph("/project/funcs.py")
        assert graph.has_node("/project/funcs.py")

        # Find chunk nodes
        chunk_nodes = graph.find_nodes(parent_path="/project/funcs.py")
        assert len(chunk_nodes) >= 2

        for node_path in chunk_nodes:
            data = graph.get_node(node_path)
            assert data["parent_path"] == "/project/funcs.py"
            assert "line_start" in data
            assert "line_end" in data

    @pytest.mark.asyncio
    async def test_analyze_creates_contains_edges(self, grover: GroverAsync):
        """'contains' edges should connect parent file to chunk nodes."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        graph = grover.get_graph("/project/funcs.py")
        contains = graph.contains("/project/funcs.py")
        assert len(contains) >= 2


# ==================================================================
# Re-analysis replaces chunks
# ==================================================================


class TestReAnalyzeReplacesChunks:
    @pytest.mark.asyncio
    async def test_re_analyze_replaces_chunks(self, grover: GroverAsync):
        """Editing a file should replace old chunk rows with new ones."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        mount = _get_mount(grover, "/project")
        backend = mount.filesystem
        assert isinstance(backend, GroverFileSystem)

        # Check initial chunks
        async with grover._ctx.session_for(mount) as sess:
            result_before = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(result_before.chunks) >= 2

        # Edit: replace beta with gamma
        new_code = """\
def alpha():
    \"\"\"Alpha function.\"\"\"
    return 1


def gamma():
    \"\"\"Gamma function.\"\"\"
    return 3
"""
        await grover.write("/project/funcs.py", new_code)
        await grover.flush()

        async with grover._ctx.session_for(mount) as sess:
            result_after = await backend.list_file_chunks("/project/funcs.py", session=sess)
        paths = [c.path for c in result_after.chunks]
        assert any("alpha" in p for p in paths)
        assert any("gamma" in p for p in paths)
        assert not any("beta" in p for p in paths)


# ==================================================================
# Delete cleans chunks
# ==================================================================


class TestDeleteCleansChunks:
    @pytest.mark.asyncio
    async def test_delete_cleans_chunks(self, grover: GroverAsync):
        """Deleting a file should remove its chunk rows."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        mount = _get_mount(grover, "/project")
        backend = mount.filesystem
        assert isinstance(backend, GroverFileSystem)

        # Verify chunks exist
        async with grover._ctx.session_for(mount) as sess:
            result = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(result.chunks) >= 2

        # Delete the file
        await grover.delete("/project/funcs.py")
        await grover.flush()

        async with grover._ctx.session_for(mount) as sess:
            result_after = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(result_after.chunks) == 0


# ==================================================================
# Move re-indexes chunks
# ==================================================================


class TestMoveReIndexesChunks:
    @pytest.mark.asyncio
    async def test_move_re_indexes_chunks(self, grover: GroverAsync):
        """Moving a file should delete old chunks and create new ones at new path."""
        await grover.write("/project/old.py", PYTHON_CODE)
        await grover.flush()

        mount = _get_mount(grover, "/project")
        backend = mount.filesystem
        assert isinstance(backend, GroverFileSystem)

        # Verify old chunks
        async with grover._ctx.session_for(mount) as sess:
            result = await backend.list_file_chunks("/project/old.py", session=sess)
        assert len(result.chunks) >= 2

        # Move
        await grover.move("/project/old.py", "/project/new.py")
        await grover.flush()

        # Old chunks should be gone
        async with grover._ctx.session_for(mount) as sess:
            result_old = await backend.list_file_chunks("/project/old.py", session=sess)
        assert len(result_old.chunks) == 0

        # New chunks should exist
        async with grover._ctx.session_for(mount) as sess:
            result_new = await backend.list_file_chunks("/project/new.py", session=sess)
        assert len(result_new.chunks) >= 2


# ==================================================================
# Hardened graph cleanup
# ==================================================================


class TestHardenedGraphCleanup:
    @pytest.mark.asyncio
    async def test_removes_children_by_contains_edge(self, grover: GroverAsync):
        """remove_file_subgraph should find children via 'contains' edges."""
        graph = grover.get_graph()

        # Manually create nodes with only a contains edge (no parent_path attr)
        graph.add_node("/test/parent.py")
        graph.add_node("/test/child1")
        graph.add_edge("/test/parent.py", "/test/child1", edge_type="contains")

        removed = graph.remove_file_subgraph("/test/parent.py")
        assert "/test/parent.py" in removed
        assert "/test/child1" in removed
        assert not graph.has_node("/test/child1")

    @pytest.mark.asyncio
    async def test_removes_children_by_parent_path_attr(self, grover: GroverAsync):
        """remove_file_subgraph should find children via parent_path attribute."""
        graph = grover.get_graph()

        # Create nodes with parent_path attr but no contains edge
        graph.add_node("/test/parent.py")
        graph.add_node("/test/child2", parent_path="/test/parent.py")

        removed = graph.remove_file_subgraph("/test/parent.py")
        assert "/test/parent.py" in removed
        assert "/test/child2" in removed
        assert not graph.has_node("/test/child2")

    @pytest.mark.asyncio
    async def test_removes_children_by_both_methods(self, grover: GroverAsync):
        """Children found by either method should be removed (union)."""
        graph = grover.get_graph()

        graph.add_node("/test/parent.py")
        # child_a: only via contains edge
        graph.add_node("/test/child_a")
        graph.add_edge("/test/parent.py", "/test/child_a", edge_type="contains")
        # child_b: only via parent_path attr
        graph.add_node("/test/child_b", parent_path="/test/parent.py")
        # child_c: via both
        graph.add_node("/test/child_c", parent_path="/test/parent.py")
        graph.add_edge("/test/parent.py", "/test/child_c", edge_type="contains")

        removed = graph.remove_file_subgraph("/test/parent.py")
        assert "/test/child_a" in removed
        assert "/test/child_b" in removed
        assert "/test/child_c" in removed
        assert not graph.has_node("/test/child_a")
        assert not graph.has_node("/test/child_b")
        assert not graph.has_node("/test/child_c")


# ==================================================================
# Vector metadata has chunk fields
# ==================================================================


class TestVectorMetadata:
    @pytest.mark.asyncio
    async def test_vector_metadata_has_chunk_fields(self, grover: GroverAsync):
        """Vector entries should include chunk_name, line_start, line_end in metadata."""
        await grover.write("/project/funcs.py", PYTHON_CODE)
        await grover.flush()

        mount = _get_mount(grover, "/project")
        local_store = mount.filesystem.search_provider
        assert local_store is not None

        # Check metadata of stored vectors
        found = False
        for meta in local_store._key_to_meta.values():
            parent = meta.get("parent_path")
            if parent == "/project/funcs.py":
                assert "chunk_name" in meta
                assert "line_start" in meta
                assert "line_end" in meta
                assert meta["chunk_name"] is not None
                found = True
                break
        assert found, "No chunk vector found with parent_path=/project/funcs.py"

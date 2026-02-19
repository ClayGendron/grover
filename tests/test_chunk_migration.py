"""Tests for Phase 3: Chunk storage migration (DB rows, not VFS files)."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover._grover_async import GroverAsync
from grover.events import EventType, FileEvent
from grover.fs.local_fs import LocalFileSystem
from grover.fs.protocol import SupportsFileChunks

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing."""

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


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
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _get_mount(g: GroverAsync, mount_path: str):
    return next(m for m in g._registry.list_visible_mounts() if m.mount_path == mount_path)


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

        mount = _get_mount(grover, "/project")
        backend = mount.backend
        assert isinstance(backend, SupportsFileChunks)

        async with grover._vfs.session_for(mount) as sess:
            chunks = await backend.list_file_chunks("/project/funcs.py", session=sess)

        assert len(chunks) >= 2
        names = [c.name for c in chunks]
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.asyncio
    async def test_analyze_no_vfs_chunk_files(self, grover: GroverAsync):
        """Writing a .py file should NOT create files under /.grover/chunks/."""
        await grover.write("/project/funcs.py", PYTHON_CODE)

        result = await grover.glob("*", path="/.grover/chunks")
        # Either the glob fails (path doesn't exist) or returns empty
        if result.success:
            assert len(result.entries) == 0, (
                f"Expected no chunk files, found: {[e.path for e in result.entries]}"
            )


# ==================================================================
# Graph nodes and edges still created
# ==================================================================


class TestAnalyzeCreatesGraphState:
    @pytest.mark.asyncio
    async def test_analyze_creates_graph_nodes(self, grover: GroverAsync):
        """Chunk nodes should have parent_path attribute pointing to parent file."""
        await grover.write("/project/funcs.py", PYTHON_CODE)

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

        mount = _get_mount(grover, "/project")
        backend = mount.backend
        assert isinstance(backend, SupportsFileChunks)

        # Check initial chunks
        async with grover._vfs.session_for(mount) as sess:
            chunks_before = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(chunks_before) >= 2

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

        async with grover._vfs.session_for(mount) as sess:
            chunks_after = await backend.list_file_chunks("/project/funcs.py", session=sess)
        names = [c.name for c in chunks_after]
        assert "alpha" in names
        assert "gamma" in names
        assert "beta" not in names


# ==================================================================
# Delete cleans chunks
# ==================================================================


class TestDeleteCleansChunks:
    @pytest.mark.asyncio
    async def test_delete_cleans_chunks(self, grover: GroverAsync):
        """Deleting a file should remove its chunk rows."""
        await grover.write("/project/funcs.py", PYTHON_CODE)

        mount = _get_mount(grover, "/project")
        backend = mount.backend
        assert isinstance(backend, SupportsFileChunks)

        # Verify chunks exist
        async with grover._vfs.session_for(mount) as sess:
            chunks = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(chunks) >= 2

        # Delete the file
        await grover.delete("/project/funcs.py")

        async with grover._vfs.session_for(mount) as sess:
            chunks_after = await backend.list_file_chunks("/project/funcs.py", session=sess)
        assert len(chunks_after) == 0


# ==================================================================
# Move re-indexes chunks
# ==================================================================


class TestMoveReIndexesChunks:
    @pytest.mark.asyncio
    async def test_move_re_indexes_chunks(self, grover: GroverAsync):
        """Moving a file should delete old chunks and create new ones at new path."""
        await grover.write("/project/old.py", PYTHON_CODE)

        mount = _get_mount(grover, "/project")
        backend = mount.backend
        assert isinstance(backend, SupportsFileChunks)

        # Verify old chunks
        async with grover._vfs.session_for(mount) as sess:
            old_chunks = await backend.list_file_chunks("/project/old.py", session=sess)
        assert len(old_chunks) >= 2

        # Move
        await grover.move("/project/old.py", "/project/new.py")

        # Old chunks should be gone
        async with grover._vfs.session_for(mount) as sess:
            old_chunks_after = await backend.list_file_chunks("/project/old.py", session=sess)
        assert len(old_chunks_after) == 0

        # New chunks should exist
        async with grover._vfs.session_for(mount) as sess:
            new_chunks = await backend.list_file_chunks("/project/new.py", session=sess)
        assert len(new_chunks) >= 2


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

        mount = _get_mount(grover, "/project")
        search_engine = mount.backend._search_engine
        assert search_engine is not None

        local_store = search_engine._get_local_store()
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


# ==================================================================
# FileEvent user_id
# ==================================================================


class TestFileEventUserId:
    def test_file_event_user_id_default(self):
        """FileEvent.user_id should default to None."""
        event = FileEvent(event_type=EventType.FILE_WRITTEN, path="/test.py")
        assert event.user_id is None

    def test_file_event_user_id_set(self):
        """FileEvent.user_id should be settable."""
        event = FileEvent(event_type=EventType.FILE_WRITTEN, path="/test.py", user_id="alice")
        assert event.user_id == "alice"

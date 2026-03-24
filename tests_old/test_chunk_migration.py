"""Tests for chunk-related graph cleanup (hardened remove_node)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.providers.search.local import LocalVectorStore

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
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Hardened graph cleanup
# ==================================================================


class TestHardenedGraphCleanup:
    @pytest.mark.asyncio
    async def test_remove_node_removes_parent(self, grover: GroverAsync):
        """remove_node removes the node and cleans incident edges."""
        graph = grover.get_graph()

        # Manually create nodes with only a contains edge (no parent_path attr)
        graph.add_node("/test/parent.py")
        graph.add_node("/test/child1")
        graph.add_edge("/test/parent.py", "/test/child1", edge_type="contains")

        graph.remove_node("/test/parent.py")
        assert not graph.has_node("/test/parent.py")
        # Child still exists (remove_node only removes the specified node)
        assert graph.has_node("/test/child1")
        # But the edge is cleaned up
        assert not graph.has_edge("/test/parent.py", "/test/child1")

    @pytest.mark.asyncio
    async def test_remove_node_cleans_edges(self, grover: GroverAsync):
        """Removing a node cleans up all incident edges."""
        graph = grover.get_graph()

        graph.add_node("/test/parent.py")
        graph.add_node("/test/child_a")
        graph.add_edge("/test/parent.py", "/test/child_a", edge_type="contains")
        graph.add_node("/test/child_b")
        graph.add_edge("/test/parent.py", "/test/child_b", edge_type="imports")

        graph.remove_node("/test/parent.py")
        assert not graph.has_node("/test/parent.py")
        assert not graph.has_edge("/test/parent.py", "/test/child_a")
        assert not graph.has_edge("/test/parent.py", "/test/child_b")

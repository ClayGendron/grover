"""Tests for background indexing: debounce, cancellation, lifecycle, and manual mode."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover, GroverAsync
from grover.providers.search.local import LocalVectorStore
from grover.worker import BackgroundWorker, IndexingMode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ==================================================================
# TestBackgroundModeAsync (GroverAsync integration)
# ==================================================================


class TestBackgroundModeAsync:
    """Integration tests for background indexing through GroverAsync."""

    @pytest.fixture
    async def grover(self, tmp_path: Path) -> GroverAsync:
        data = tmp_path / "grover_data"
        ws = tmp_path / "workspace"
        ws.mkdir()
        g = GroverAsync()
        await g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
            search_provider=LocalVectorStore(dimension=FAKE_DIM),
        )
        yield g  # type: ignore[misc]
        await g.close()

    @pytest.mark.asyncio
    async def test_write_returns_before_indexing(self, grover: GroverAsync) -> None:
        """Write should return before graph is updated."""
        await grover.write("/project/mod.py", "def foo():\n    pass\n")
        # Graph should NOT have the node yet
        assert not grover.get_graph().has_node("/project/mod.py")
        await grover.flush()
        assert grover.get_graph().has_node("/project/mod.py")

    @pytest.mark.asyncio
    async def test_write_then_flush_then_search(self, grover: GroverAsync) -> None:
        """After flush, vector search should find the written content."""
        await grover.write(
            "/project/auth.py",
            'def authenticate():\n    """Auth logic."""\n    pass\n',
        )
        await grover.flush()
        result = await grover.vector_search("authenticate")
        assert result.success is True
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_rapid_writes_debounced(self, grover: GroverAsync) -> None:
        """Rapid writes to the same file should debounce to final content."""
        for i in range(5):
            await grover.write("/project/rapid.py", f"VERSION = {i}\n")
        await grover.flush()
        graph = grover.get_graph()
        assert graph.has_node("/project/rapid.py")
        result = await grover.read("/project/rapid.py")
        assert result.file.content == "VERSION = 4\n"

    @pytest.mark.asyncio
    async def test_write_then_delete_cancels(self, grover: GroverAsync) -> None:
        """Write followed by delete (no flush between) — node should NOT be in graph."""
        await grover.write("/project/temp.py", "def temp():\n    pass\n")
        await grover.delete("/project/temp.py")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/temp.py")

    @pytest.mark.asyncio
    async def test_move_in_background(self, grover: GroverAsync) -> None:
        """Move should update graph after flush."""
        await grover.write("/project/old.py", "def foo():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/old.py")

        await grover.move("/project/old.py", "/project/new.py")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/old.py")
        assert grover.get_graph().has_node("/project/new.py")

    @pytest.mark.asyncio
    async def test_edit_triggers_reindex(self, grover: GroverAsync) -> None:
        """Edit should trigger re-indexing in background."""
        await grover.write("/project/edit_me.py", "def alpha():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/edit_me.py")

        await grover.edit("/project/edit_me.py", "alpha", "beta")
        await grover.flush()
        # Graph should still have the file (re-analyzed)
        assert grover.get_graph().has_node("/project/edit_me.py")

    @pytest.mark.asyncio
    async def test_copy_triggers_index_at_dest(self, grover: GroverAsync) -> None:
        """Copy should index the destination file."""
        await grover.write("/project/src.py", "def src():\n    pass\n")
        await grover.flush()

        await grover.copy("/project/src.py", "/project/dst.py")
        await grover.flush()
        assert grover.get_graph().has_node("/project/src.py")
        assert grover.get_graph().has_node("/project/dst.py")

    @pytest.mark.asyncio
    async def test_restore_triggers_reindex(self, grover: GroverAsync) -> None:
        """Restore from trash should re-index the file."""
        await grover.write("/project/restore_me.py", "def restore():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/restore_me.py")

        await grover.delete("/project/restore_me.py")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/restore_me.py")

        result = await grover.restore_from_trash("/project/restore_me.py")
        if result.success:
            await grover.flush()
            assert grover.get_graph().has_node("/project/restore_me.py")

    @pytest.mark.asyncio
    async def test_index_still_synchronous(self, grover: GroverAsync, tmp_path: Path) -> None:
        """index() should return with graph populated (no flush needed)."""
        ws = tmp_path / "workspace"
        (ws / "indexed.py").write_text("def indexed():\n    pass\n")
        await grover.index()
        assert grover.get_graph().has_node("/project/indexed.py")

    @pytest.mark.asyncio
    async def test_close_drains_pending(self, tmp_path: Path) -> None:
        """close() should drain pending work before shutting down."""
        data = tmp_path / "grover_data"
        ws = tmp_path / "workspace"
        ws.mkdir()
        g = GroverAsync()
        await g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        await g.write("/project/close_test.py", "def test():\n    pass\n")
        # No explicit flush — close should drain
        await g.close()
        # Can't check graph after close, but no errors means drain completed

    @pytest.mark.asyncio
    async def test_save_drains_pending(self, grover: GroverAsync) -> None:
        """save() should drain pending work before persisting."""
        await grover.write("/project/save_test.py", "def save():\n    pass\n")
        # No explicit flush — save should drain
        await grover.save()
        # Graph should now have the node (drain happened)
        assert grover.get_graph().has_node("/project/save_test.py")


# ==================================================================
# TestBackgroundModeSync (Grover sync wrapper)
# ==================================================================


class TestBackgroundModeSync:
    """Integration tests for background indexing through sync Grover."""

    @pytest.fixture
    def grover(self, tmp_path: Path) -> Iterator[Grover]:
        data = tmp_path / "grover_data"
        ws = tmp_path / "workspace"
        ws.mkdir()
        g = Grover()
        g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
            search_provider=LocalVectorStore(dimension=FAKE_DIM),
        )
        yield g
        g.close()

    def test_sync_write_flush_search(self, grover: Grover) -> None:
        """Sync write, flush, then search should find results."""
        grover.write(
            "/project/sync.py",
            'def sync_func():\n    """Sync function."""\n    pass\n',
        )
        grover.flush()
        result = grover.vector_search("sync_func")
        assert result.success is True
        assert len(result) >= 1

    def test_sync_close_drains(self, tmp_path: Path) -> None:
        """Sync close should drain pending work."""
        data = tmp_path / "grover_data2"
        ws = tmp_path / "workspace2"
        ws.mkdir()
        g = Grover()
        g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        g.write("/project/drain.py", "def drain():\n    pass\n")
        # close should drain and not crash
        g.close()


# ==================================================================
# TestManualMode
# ==================================================================


class TestManualMode:
    """Tests for MANUAL indexing mode."""

    @pytest.fixture
    async def grover(self, tmp_path: Path) -> GroverAsync:
        data = tmp_path / "grover_data"
        ws = tmp_path / "workspace"
        ws.mkdir()
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        yield g  # type: ignore[misc]
        await g.close()

    @pytest.mark.asyncio
    async def test_write_no_graph_update(self, grover: GroverAsync) -> None:
        """In manual mode, write should NOT update the graph."""
        await grover.write("/project/manual.py", "def manual():\n    pass\n")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/manual.py")

    @pytest.mark.asyncio
    async def test_explicit_index_works(self, grover: GroverAsync, tmp_path: Path) -> None:
        """In manual mode, explicit index() should still populate the graph."""
        ws = tmp_path / "workspace"
        (ws / "indexed.py").write_text("def indexed():\n    pass\n")
        await grover.index()
        assert grover.get_graph().has_node("/project/indexed.py")

    @pytest.mark.asyncio
    async def test_manual_flush_is_noop(self, grover: GroverAsync) -> None:
        """In manual mode, flush should be a harmless no-op."""
        await grover.write("/project/noop.py", "x = 1\n")
        await grover.flush()  # Should not raise
        assert not grover.get_graph().has_node("/project/noop.py")


# ==================================================================
# TestDrainSafety (timeout + iteration guard)
# ==================================================================


class TestDrainSafety:
    """Tests for drain() timeout and iteration guard."""

    @pytest.mark.asyncio
    async def test_drain_timeout_cancels_hung_task(self) -> None:
        """drain() should cancel tasks that exceed the timeout."""
        worker = BackgroundWorker(drain_timeout=0.5)
        worker.schedule_immediate(asyncio.sleep(999))
        await worker.drain()
        assert len(worker._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_drain_max_iterations_guard(self) -> None:
        """drain() should bail after _MAX_DRAIN_ITERATIONS even if tasks reschedule."""
        worker = BackgroundWorker(drain_timeout=10.0)

        async def self_rescheduler() -> None:
            worker.schedule("loop", lambda: self_rescheduler())

        worker.schedule("loop", lambda: self_rescheduler())
        await worker.drain()
        # Should return without hanging
        assert len(worker._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_drain_explicit_timeout_override(self) -> None:
        """Explicit timeout= kwarg should override constructor default."""
        worker = BackgroundWorker(drain_timeout=30.0)
        worker.schedule_immediate(asyncio.sleep(999))
        await worker.drain(timeout=0.5)
        assert len(worker._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_drain_normal_operation_unchanged(self) -> None:
        """Fast tasks should complete normally under drain()."""
        results: list[int] = []

        async def quick(n: int) -> None:
            results.append(n)

        worker = BackgroundWorker(drain_timeout=5.0)
        worker.schedule_immediate(quick(1))
        worker.schedule_immediate(quick(2))
        worker.schedule_immediate(quick(3))
        await worker.drain()
        assert sorted(results) == [1, 2, 3]
        assert len(worker._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_drain_empty_is_noop(self) -> None:
        """drain() with nothing pending should return immediately."""
        worker = BackgroundWorker()
        await worker.drain()
        assert len(worker._active_tasks) == 0
        assert len(worker._pending) == 0

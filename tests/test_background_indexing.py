"""Tests for background indexing: debounce, cancellation, lifecycle, and manual mode."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.events import EventBus, EventType, FileEvent, IndexingMode
from grover.fs.local_fs import LocalFileSystem

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ------------------------------------------------------------------
# Helpers
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


# ==================================================================
# TestBackgroundDispatch (EventBus level)
# ==================================================================


class TestBackgroundDispatch:
    """EventBus-level tests for background dispatch, debounce, and drain."""

    async def test_emit_returns_before_handler_runs(self) -> None:
        """Emit should return before the handler has executed."""
        bus = EventBus()
        barrier = asyncio.Event()
        ran = []

        async def handler(event: FileEvent) -> None:
            ran.append(True)
            barrier.set()

        bus.register(EventType.FILE_WRITTEN, handler)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        # Handler hasn't run yet (still in debounce timer)
        assert len(ran) == 0
        await bus.drain()
        assert len(ran) == 1

    async def test_debounce_same_path(self) -> None:
        """Multiple rapid writes to the same path should debounce to one handler call."""
        bus = EventBus(debounce_delay=0.05)
        calls: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            calls.append(event)

        bus.register(EventType.FILE_WRITTEN, handler)
        for i in range(5):
            await bus.emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py", content=f"v{i}")
            )
        await bus.drain()
        assert len(calls) == 1
        assert calls[0].content == "v4"  # Last event wins

    async def test_debounce_different_paths(self) -> None:
        """Writes to different paths should each fire."""
        bus = EventBus(debounce_delay=0.05)
        calls: list[str] = []

        async def handler(event: FileEvent) -> None:
            calls.append(event.path)

        bus.register(EventType.FILE_WRITTEN, handler)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/b.py"))
        await bus.drain()
        assert sorted(calls) == ["/a.py", "/b.py"]

    async def test_delete_cancels_pending_write(self) -> None:
        """DELETE should cancel any pending WRITTEN for the same path."""
        bus = EventBus(debounce_delay=0.5)
        written_calls: list[str] = []
        deleted_calls: list[str] = []

        async def on_write(event: FileEvent) -> None:
            written_calls.append(event.path)

        async def on_delete(event: FileEvent) -> None:
            deleted_calls.append(event.path)

        bus.register(EventType.FILE_WRITTEN, on_write)
        bus.register(EventType.FILE_DELETED, on_delete)

        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await bus.emit(FileEvent(event_type=EventType.FILE_DELETED, path="/a.py"))
        await bus.drain()

        assert len(written_calls) == 0  # Write was cancelled
        assert len(deleted_calls) == 1

    async def test_move_cancels_pending_write_for_old_path(self) -> None:
        """MOVE should cancel any pending event for old_path."""
        bus = EventBus(debounce_delay=0.5)
        written_calls: list[str] = []
        moved_calls: list[str] = []

        async def on_write(event: FileEvent) -> None:
            written_calls.append(event.path)

        async def on_move(event: FileEvent) -> None:
            moved_calls.append(event.path)

        bus.register(EventType.FILE_WRITTEN, on_write)
        bus.register(EventType.FILE_MOVED, on_move)

        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/old.py"))
        await bus.emit(
            FileEvent(event_type=EventType.FILE_MOVED, path="/new.py", old_path="/old.py")
        )
        await bus.drain()

        assert len(written_calls) == 0  # Write for old path was cancelled
        assert len(moved_calls) == 1

    async def test_connection_events_fire_immediately(self) -> None:
        """Connection events should fire without waiting for debounce delay."""
        bus = EventBus(debounce_delay=10.0)  # Very long delay
        calls: list[str] = []

        async def handler(event: FileEvent) -> None:
            calls.append(event.path)

        bus.register(EventType.CONNECTION_ADDED, handler)
        await bus.emit(
            FileEvent(
                event_type=EventType.CONNECTION_ADDED,
                path="/a.py[imports]/b.py",
                source_path="/a.py",
                target_path="/b.py",
                connection_type="imports",
            )
        )
        await bus.drain()
        assert len(calls) == 1

    async def test_nested_emit_dispatches_inline(self) -> None:
        """An emit from within a handler should dispatch inline (no new background task)."""
        bus = EventBus()
        order: list[str] = []

        async def outer_handler(event: FileEvent) -> None:
            order.append("outer_start")
            await bus.emit(
                FileEvent(
                    event_type=EventType.CONNECTION_ADDED,
                    path="/conn",
                    source_path="/a.py",
                    target_path="/b.py",
                    connection_type="test",
                )
            )
            order.append("outer_end")

        async def inner_handler(event: FileEvent) -> None:
            order.append("inner")

        bus.register(EventType.FILE_WRITTEN, outer_handler)
        bus.register(EventType.CONNECTION_ADDED, inner_handler)

        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await bus.drain()

        # Inner should have run between outer_start and outer_end (inline)
        assert order == ["outer_start", "inner", "outer_end"]

    async def test_drain_settles_completely(self) -> None:
        """After drain, pending_count should be 0."""
        bus = EventBus(debounce_delay=0.05)

        async def noop_handler(event: FileEvent) -> None:
            pass

        bus.register(EventType.FILE_WRITTEN, noop_handler)
        for i in range(10):
            await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path=f"/f{i}.py"))
        assert bus.pending_count > 0
        await bus.drain()
        assert bus.pending_count == 0

    async def test_error_isolation_in_background(self) -> None:
        """A failing handler should not prevent other handlers from running."""
        bus = EventBus()
        calls: list[str] = []

        async def failing(event: FileEvent) -> None:
            raise RuntimeError("boom")

        async def good(event: FileEvent) -> None:
            calls.append("ok")

        bus.register(EventType.FILE_WRITTEN, failing)
        bus.register(EventType.FILE_WRITTEN, good)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await bus.drain()
        assert calls == ["ok"]

    async def test_manual_mode_suppresses_all(self) -> None:
        """In MANUAL mode, emit should not call any handlers."""
        bus = EventBus(indexing_mode=IndexingMode.MANUAL)
        calls: list[str] = []

        async def handler(event: FileEvent) -> None:
            calls.append("called")

        bus.register(EventType.FILE_WRITTEN, handler)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await bus.drain()
        assert len(calls) == 0

    async def test_pending_count(self) -> None:
        """pending_count should track debounced + active background events."""
        bus = EventBus(debounce_delay=10.0)

        async def slow_handler(event: FileEvent) -> None:
            await asyncio.sleep(0.1)

        bus.register(EventType.FILE_WRITTEN, slow_handler)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        assert bus.pending_count >= 1
        await bus.drain()
        assert bus.pending_count == 0

    async def test_configurable_debounce_delay(self) -> None:
        """Events should not fire before the debounce delay."""
        bus = EventBus(debounce_delay=0.3)
        calls: list[str] = []

        async def handler(event: FileEvent) -> None:
            calls.append("done")

        bus.register(EventType.FILE_WRITTEN, handler)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.py"))
        await asyncio.sleep(0.05)
        assert len(calls) == 0  # Not yet fired
        await bus.drain()
        assert len(calls) == 1


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
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
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
        assert result.content == "VERSION = 4\n"

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
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
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
        g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
        g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
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
        g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
        g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
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
        g = GroverAsync(
            data_dir=str(data),
            embedding_provider=FakeProvider(),
            indexing_mode=IndexingMode.MANUAL,
        )
        await g.add_mount("/project", LocalFileSystem(workspace_dir=ws, data_dir=data / "local"))
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

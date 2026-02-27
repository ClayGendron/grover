"""Tests for EventBus and event types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover._grover_async import GroverAsync
from grover.events import EventBus, EventType, FileEvent
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_fs import LocalFileSystem

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Helpers
# =========================================================================


async def _collecting_handler(events: list[FileEvent], event: FileEvent) -> None:
    """Append event to a list for assertion."""
    events.append(event)


async def _failing_handler(event: FileEvent) -> None:
    """Handler that always raises."""
    raise RuntimeError(f"boom on {event.path}")


# =========================================================================
# EventType
# =========================================================================


class TestEventType:
    def test_member_count(self) -> None:
        assert len(EventType) == 6

    def test_values(self) -> None:
        assert EventType.FILE_WRITTEN.value == "file_written"
        assert EventType.FILE_DELETED.value == "file_deleted"
        assert EventType.FILE_MOVED.value == "file_moved"
        assert EventType.FILE_RESTORED.value == "file_restored"

    def test_unique_values(self) -> None:
        values = [et.value for et in EventType]
        assert len(values) == len(set(values))


# =========================================================================
# FileEvent
# =========================================================================


class TestFileEvent:
    def test_construction(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt")
        assert ev.event_type is EventType.FILE_WRITTEN
        assert ev.path == "/a.txt"
        assert ev.old_path is None
        assert ev.content is None

    def test_with_content(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt", content="hello")
        assert ev.content == "hello"

    def test_move_event(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_MOVED, path="/b.txt", old_path="/a.txt")
        assert ev.old_path == "/a.txt"
        assert ev.path == "/b.txt"

    def test_deleted_event(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_DELETED, path="/gone.txt")
        assert ev.event_type is EventType.FILE_DELETED

    def test_restored_event(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_RESTORED, path="/back.txt")
        assert ev.event_type is EventType.FILE_RESTORED

    def test_immutable(self) -> None:
        ev = FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt")
        with pytest.raises(AttributeError):
            ev.path = "/changed.txt"  # type: ignore[misc]


# =========================================================================
# EventBus Registration
# =========================================================================


class TestEventBusRegistration:
    def test_initial_handler_count(self) -> None:
        bus = EventBus()
        assert bus.handler_count == 0

    def test_register_increments_count(self) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)
        assert bus.handler_count == 1

    def test_register_multiple_types(self) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)
        bus.register(EventType.FILE_DELETED, _failing_handler)
        assert bus.handler_count == 2

    def test_register_multiple_handlers_same_type(self) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)

        async def another(event: FileEvent) -> None:
            pass

        bus.register(EventType.FILE_WRITTEN, another)
        assert bus.handler_count == 2

    def test_unregister_returns_true(self) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)
        assert bus.unregister(EventType.FILE_WRITTEN, _failing_handler) is True
        assert bus.handler_count == 0

    def test_unregister_missing_returns_false(self) -> None:
        bus = EventBus()
        assert bus.unregister(EventType.FILE_WRITTEN, _failing_handler) is False

    def test_clear(self) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)
        bus.register(EventType.FILE_DELETED, _failing_handler)
        bus.clear()
        assert bus.handler_count == 0


# =========================================================================
# EventBus Emit
# =========================================================================


class TestEventBusEmit:
    async def test_handler_called_with_event(self) -> None:
        bus = EventBus()
        collected: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        bus.register(EventType.FILE_WRITTEN, handler)
        ev = FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt")
        await bus.emit(ev)
        await bus.drain()
        assert collected == [ev]

    async def test_multiple_handlers_called_in_order(self) -> None:
        bus = EventBus()
        order: list[int] = []

        async def first(event: FileEvent) -> None:
            order.append(1)

        async def second(event: FileEvent) -> None:
            order.append(2)

        bus.register(EventType.FILE_WRITTEN, first)
        bus.register(EventType.FILE_WRITTEN, second)
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt"))
        await bus.drain()
        assert order == [1, 2]

    async def test_type_filtering(self) -> None:
        bus = EventBus()
        written: list[FileEvent] = []
        deleted: list[FileEvent] = []

        async def on_write(event: FileEvent) -> None:
            await _collecting_handler(written, event)

        async def on_delete(event: FileEvent) -> None:
            await _collecting_handler(deleted, event)

        bus.register(EventType.FILE_WRITTEN, on_write)
        bus.register(EventType.FILE_DELETED, on_delete)

        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt"))
        await bus.drain()
        assert len(written) == 1
        assert len(deleted) == 0

    async def test_no_handler_noop(self) -> None:
        bus = EventBus()
        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt"))

    async def test_error_isolation(self) -> None:
        bus = EventBus()
        collected: list[FileEvent] = []

        async def good_handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        bus.register(EventType.FILE_WRITTEN, _failing_handler)
        bus.register(EventType.FILE_WRITTEN, good_handler)

        await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt"))
        await bus.drain()
        assert len(collected) == 1

    async def test_error_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)

        with caplog.at_level(logging.WARNING, logger="grover.events"):
            await bus.emit(FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt"))
            await bus.drain()

        assert "failed" in caplog.text
        assert "file_written" in caplog.text
        assert "/a.txt" in caplog.text


# =========================================================================
# Integration: LocalFileSystem + EventBus via GroverAsync
# =========================================================================


class TestEventBusIntegration:
    """EventBus wired through GroverAsync with a local filesystem backend."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, list[FileEvent]]:
        collected: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        ws = tmp_path / "workspace"
        ws.mkdir()
        lfs = LocalFileSystem(workspace_dir=ws, data_dir=tmp_path / ".grover_test")
        await g.add_mount("/local", lfs)

        for et in EventType:
            g._ctx.event_bus.register(et, handler)

        yield g, collected
        await g.close()

    async def test_write_emits_file_written(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup
        result = await grover.write("/local/hello.txt", "hello world")
        assert result.success
        await grover.flush()
        written_events = [e for e in collected if e.event_type is EventType.FILE_WRITTEN]
        assert len(written_events) >= 1
        ev = written_events[-1]
        assert ev.path == "/local/hello.txt"
        assert ev.content == "hello world"

    async def test_edit_emits_file_written(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup
        await grover.write("/local/hello.txt", "hello world")
        await grover.flush()
        collected.clear()

        result = await grover.edit("/local/hello.txt", "hello", "goodbye")
        assert result.success
        await grover.flush()
        written_events = [e for e in collected if e.event_type is EventType.FILE_WRITTEN]
        assert len(written_events) >= 1
        ev = written_events[-1]
        assert ev.path == "/local/hello.txt"
        assert ev.content is None

    async def test_delete_emits_file_deleted(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup
        await grover.write("/local/hello.txt", "hello")
        await grover.flush()
        collected.clear()

        result = await grover.delete("/local/hello.txt", permanent=True)
        assert result.success
        await grover.flush()
        delete_events = [e for e in collected if e.event_type is EventType.FILE_DELETED]
        assert len(delete_events) >= 1
        ev = delete_events[-1]
        assert ev.path == "/local/hello.txt"

    async def test_move_emits_file_moved(self, setup: tuple[GroverAsync, list[FileEvent]]) -> None:
        grover, collected = setup
        await grover.write("/local/a.txt", "content")
        await grover.flush()
        collected.clear()

        result = await grover.move("/local/a.txt", "/local/b.txt")
        assert result.success
        await grover.flush()
        move_events = [e for e in collected if e.event_type is EventType.FILE_MOVED]
        assert len(move_events) >= 1
        ev = move_events[-1]
        assert ev.path == "/local/b.txt"
        assert ev.old_path == "/local/a.txt"

    async def test_copy_emits_file_written(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup
        await grover.write("/local/a.txt", "content")
        await grover.flush()
        collected.clear()

        result = await grover.copy("/local/a.txt", "/local/b.txt")
        assert result.success
        await grover.flush()
        written_events = [e for e in collected if e.event_type is EventType.FILE_WRITTEN]
        assert len(written_events) >= 1
        ev = written_events[-1]
        assert ev.path == "/local/b.txt"

    async def test_failed_operation_does_not_emit(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup
        result = await grover.delete("/local/nonexistent.txt", permanent=True)
        assert not result.success
        assert len(collected) == 0

    async def test_no_extra_event_bus_still_works(self, tmp_path: Path) -> None:
        """GroverAsync works without extra event handlers."""
        g = GroverAsync(data_dir=str(tmp_path / "grover_data2"))
        ws = tmp_path / "ws2"
        ws.mkdir()
        lfs = LocalFileSystem(workspace_dir=ws, data_dir=tmp_path / ".grover_test2")
        await g.add_mount("/local", lfs)

        result = await g.write("/local/hello.txt", "hello")
        assert result.success
        await g.close()


# =========================================================================
# Integration: DatabaseFileSystem + EventBus (restore operations)
# =========================================================================


class TestEventBusRestoreIntegration:
    """EventBus with DatabaseFileSystem for restore_version/restore_from_trash."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[GroverAsync, list[FileEvent]]:
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem(dialect="sqlite")

        collected: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/vfs", fs, session_factory=factory)

        for et in EventType:
            g._ctx.event_bus.register(et, handler)

        yield g, collected
        await g.close()
        await engine.dispose()

    async def test_restore_version_emits_file_restored(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup

        await grover.write("/vfs/hello.txt", "v1")
        await grover.write("/vfs/hello.txt", "v2")
        await grover.flush()
        collected.clear()

        result = await grover.restore_version("/vfs/hello.txt", 1)
        assert result.success
        await grover.flush()
        restore_events = [e for e in collected if e.event_type is EventType.FILE_RESTORED]
        assert len(restore_events) >= 1
        ev = restore_events[-1]
        assert ev.path == "/vfs/hello.txt"

    async def test_restore_from_trash_emits_file_restored(
        self, setup: tuple[GroverAsync, list[FileEvent]]
    ) -> None:
        grover, collected = setup

        await grover.write("/vfs/hello.txt", "content")
        await grover.flush()
        await grover.delete("/vfs/hello.txt")
        await grover.flush()
        collected.clear()

        result = await grover.restore_from_trash("/vfs/hello.txt")
        assert result.success
        await grover.flush()
        restore_events = [e for e in collected if e.event_type is EventType.FILE_RESTORED]
        assert len(restore_events) >= 1
        ev = restore_events[-1]
        assert ev.path == "/vfs/hello.txt"

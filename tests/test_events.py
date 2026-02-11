"""Tests for EventBus and event types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.events import EventBus, EventType, FileEvent
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.vfs import VFS

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Helpers
# =========================================================================


async def _collecting_handler(
    events: list[FileEvent], event: FileEvent
) -> None:
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
        assert len(EventType) == 4

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
        ev = FileEvent(
            event_type=EventType.FILE_WRITTEN, path="/a.txt", content="hello"
        )
        assert ev.content == "hello"

    def test_move_event(self) -> None:
        ev = FileEvent(
            event_type=EventType.FILE_MOVED, path="/b.txt", old_path="/a.txt"
        )
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
        assert len(collected) == 1

    async def test_error_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = EventBus()
        bus.register(EventType.FILE_WRITTEN, _failing_handler)

        with caplog.at_level(logging.WARNING, logger="grover.events"):
            await bus.emit(
                FileEvent(event_type=EventType.FILE_WRITTEN, path="/a.txt")
            )

        assert "failed" in caplog.text
        assert "file_written" in caplog.text
        assert "/a.txt" in caplog.text


# =========================================================================
# Integration: LocalFileSystem + EventBus
# =========================================================================


class TestEventBusIntegration:
    """EventBus wired through VFS with a local filesystem backend."""

    @pytest.fixture
    async def setup(self, tmp_path: Path) -> tuple[VFS, EventBus, list[FileEvent]]:
        bus = EventBus()
        collected: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        for et in EventType:
            bus.register(et, handler)

        backend = LocalFileSystem(
            workspace_dir=tmp_path,
            data_dir=tmp_path / ".grover_test",
        )
        await backend._ensure_db()
        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/local",
                backend=backend,
                session_factory=backend._session_factory,
                mount_type="local",
            )
        )
        ufs = VFS(registry, event_bus=bus)

        async with ufs:
            yield ufs, bus, collected

    async def test_write_emits_file_written(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        result = await ufs.write("/local/hello.txt", "hello world")
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_WRITTEN
        assert ev.path == "/local/hello.txt"
        assert ev.content == "hello world"

    async def test_edit_emits_file_written(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        await ufs.write("/local/hello.txt", "hello world")
        collected.clear()

        result = await ufs.edit("/local/hello.txt", "hello", "goodbye")
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_WRITTEN
        assert ev.path == "/local/hello.txt"
        assert ev.content is None

    async def test_delete_emits_file_deleted(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        await ufs.write("/local/hello.txt", "hello")
        collected.clear()

        result = await ufs.delete("/local/hello.txt", permanent=True)
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_DELETED
        assert ev.path == "/local/hello.txt"

    async def test_move_emits_file_moved(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        await ufs.write("/local/a.txt", "content")
        collected.clear()

        result = await ufs.move("/local/a.txt", "/local/b.txt")
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_MOVED
        assert ev.path == "/local/b.txt"
        assert ev.old_path == "/local/a.txt"

    async def test_copy_emits_file_written(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        await ufs.write("/local/a.txt", "content")
        collected.clear()

        result = await ufs.copy("/local/a.txt", "/local/b.txt")
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_WRITTEN
        assert ev.path == "/local/b.txt"

    async def test_failed_operation_does_not_emit(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup
        result = await ufs.delete("/local/nonexistent.txt", permanent=True)
        assert not result.success
        assert len(collected) == 0

    async def test_no_event_bus_still_works(self, tmp_path: Path) -> None:
        backend = LocalFileSystem(
            workspace_dir=tmp_path,
            data_dir=tmp_path / ".grover_test2",
        )
        await backend._ensure_db()
        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/local",
                backend=backend,
                session_factory=backend._session_factory,
                mount_type="local",
            )
        )
        ufs = VFS(registry)

        async with ufs:
            result = await ufs.write("/local/hello.txt", "hello")
            assert result.success


# =========================================================================
# Integration: DatabaseFileSystem + EventBus (VFS restore operations)
# =========================================================================


class TestEventBusVFSIntegration:
    """EventBus with DatabaseFileSystem for restore_version/restore_from_trash."""

    @pytest.fixture
    async def setup(self) -> tuple[VFS, EventBus, list[FileEvent]]:
        from grover.fs.database_fs import DatabaseFileSystem

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        fs = DatabaseFileSystem(dialect="sqlite")

        bus = EventBus()
        collected: list[FileEvent] = []

        async def handler(event: FileEvent) -> None:
            await _collecting_handler(collected, event)

        for et in EventType:
            bus.register(et, handler)

        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/vfs", backend=fs, session_factory=factory, mount_type="vfs",
            )
        )
        ufs = VFS(registry, event_bus=bus)

        async with ufs:
            yield ufs, bus, collected

        await engine.dispose()

    async def test_restore_version_emits_file_restored(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup

        await ufs.write("/vfs/hello.txt", "v1")
        await ufs.write("/vfs/hello.txt", "v2")
        collected.clear()

        result = await ufs.restore_version("/vfs/hello.txt", 1)
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_RESTORED
        assert ev.path == "/vfs/hello.txt"

    async def test_restore_from_trash_emits_file_restored(
        self, setup: tuple[VFS, EventBus, list[FileEvent]]
    ) -> None:
        ufs, _, collected = setup

        await ufs.write("/vfs/hello.txt", "content")
        await ufs.delete("/vfs/hello.txt")
        collected.clear()

        result = await ufs.restore_from_trash("/vfs/hello.txt")
        assert result.success
        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type is EventType.FILE_RESTORED
        assert ev.path == "/vfs/hello.txt"

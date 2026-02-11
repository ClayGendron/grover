"""Tests for UnifiedFileSystem — routing, permissions, cross-mount ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.events import EventBus, EventType, FileEvent
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission
from grover.fs.unified import UnifiedFileSystem

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Helpers
# =========================================================================


async def _collecting_handler(
    events: list[FileEvent], event: FileEvent
) -> None:
    events.append(event)


async def _make_local_setup(
    tmp_path: Path,
    *,
    second_mount: bool = False,
    ro_mount: bool = False,
) -> tuple[UnifiedFileSystem, EventBus, list[FileEvent], MountRegistry]:
    """Build registry with /local (and optionally /other or /ro) mounts."""
    bus = EventBus()
    collected: list[FileEvent] = []

    async def handler(event: FileEvent) -> None:
        await _collecting_handler(collected, event)

    for et in EventType:
        bus.register(et, handler)

    registry = MountRegistry()

    backend_local = LocalFileSystem(
        workspace_dir=tmp_path / "ws_local",
        data_dir=tmp_path / ".grover_local",
    )
    (tmp_path / "ws_local").mkdir(exist_ok=True)
    await backend_local._ensure_db()
    registry.add_mount(
        MountConfig(
            mount_path="/local",
            backend=backend_local,
            session_factory=backend_local._session_factory,
            mount_type="local",
        )
    )

    if second_mount:
        (tmp_path / "ws_other").mkdir(exist_ok=True)
        backend_other = LocalFileSystem(
            workspace_dir=tmp_path / "ws_other",
            data_dir=tmp_path / ".grover_other",
        )
        await backend_other._ensure_db()
        registry.add_mount(
            MountConfig(
                mount_path="/other",
                backend=backend_other,
                session_factory=backend_other._session_factory,
                mount_type="local",
            )
        )

    if ro_mount:
        (tmp_path / "ws_ro").mkdir(exist_ok=True)
        backend_ro = LocalFileSystem(
            workspace_dir=tmp_path / "ws_ro",
            data_dir=tmp_path / ".grover_ro",
        )
        await backend_ro._ensure_db()
        registry.add_mount(
            MountConfig(
                mount_path="/ro",
                backend=backend_ro,
                session_factory=backend_ro._session_factory,
                mount_type="local",
                permission=Permission.READ_ONLY,
            )
        )

    ufs = UnifiedFileSystem(registry, event_bus=bus)
    return ufs, bus, collected, registry


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
async def rw_ufs(tmp_path: Path):
    """Single RW mount at /local."""
    ufs, _bus, collected, _registry = await _make_local_setup(tmp_path)
    async with ufs:
        yield ufs, collected


@pytest.fixture
async def rw_ro_ufs(tmp_path: Path):
    """RW mount at /local + RO mount at /ro."""
    ufs, _bus, collected, _registry = await _make_local_setup(tmp_path, ro_mount=True)
    async with ufs:
        yield ufs, collected


@pytest.fixture
async def two_mount_ufs(tmp_path: Path):
    """Two RW mounts at /local and /other."""
    ufs, _bus, collected, _registry = await _make_local_setup(tmp_path, second_mount=True)
    async with ufs:
        yield ufs, collected


# =========================================================================
# Permission Checks
# =========================================================================


class TestPermissionChecks:
    async def test_write_read_only_mount(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.write("/ro/file.txt", "content")
        assert result.success is False
        assert "read-only" in result.message.lower()

    async def test_edit_read_only_mount(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.edit("/ro/file.txt", "old", "new")
        assert result.success is False

    async def test_delete_read_only_mount(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.delete("/ro/file.txt")
        assert result.success is False

    async def test_mkdir_read_only_mount(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.mkdir("/ro/subdir")
        assert result.success is False

    async def test_move_from_read_only_rejected(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.move("/ro/file.txt", "/local/file.txt")
        assert result.success is False

    async def test_move_to_read_only_rejected(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        await ufs.write("/local/file.txt", "content")
        result = await ufs.move("/local/file.txt", "/ro/file.txt")
        assert result.success is False

    async def test_copy_to_read_only_rejected(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        await ufs.write("/local/file.txt", "content")
        result = await ufs.copy("/local/file.txt", "/ro/file.txt")
        assert result.success is False

    async def test_restore_version_read_only(self, tmp_path: Path):
        """restore_version on a RO mount returns success=False."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        db = DatabaseFileSystem(dialect="sqlite")

        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/vfs",
                backend=db,
                session_factory=factory,
                mount_type="vfs",
                permission=Permission.READ_ONLY,
            )
        )
        ufs = UnifiedFileSystem(registry)
        async with ufs:
            result = await ufs.restore_version("/vfs/file.txt", 1)
            assert result.success is False
        await engine.dispose()

    async def test_restore_from_trash_read_only(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        result = await ufs.restore_from_trash("/ro/file.txt")
        assert result.success is False

    async def test_get_permission_info(self, rw_ro_ufs):
        ufs, _ = rw_ro_ufs
        perm, is_override = ufs.get_permission_info("/ro/file.txt")
        assert perm == Permission.READ_ONLY.value
        assert isinstance(is_override, bool)


# =========================================================================
# Read Operations
# =========================================================================


class TestReadOperations:
    async def test_read_prefixes_path(self, rw_ufs):
        ufs, _ = rw_ufs
        await ufs.write("/local/hello.txt", "hello")
        result = await ufs.read("/local/hello.txt")
        assert result.success
        assert result.file_path is not None
        assert result.file_path.startswith("/local")

    async def test_list_dir_root(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        result = await ufs.list_dir("/")
        names = {e.name for e in result.entries}
        assert "local" in names
        assert "other" in names

    async def test_list_dir_nonroot(self, rw_ufs):
        ufs, _ = rw_ufs
        await ufs.write("/local/a.txt", "a")
        result = await ufs.list_dir("/local")
        paths = {e.path for e in result.entries}
        assert any(p.startswith("/local/") for p in paths)

    async def test_exists_root(self, rw_ufs):
        ufs, _ = rw_ufs
        assert await ufs.exists("/") is True

    async def test_exists_mount_path(self, rw_ufs):
        ufs, _ = rw_ufs
        assert await ufs.exists("/local") is True

    async def test_exists_no_mount(self, rw_ufs):
        ufs, _ = rw_ufs
        assert await ufs.exists("/nonexistent/file.txt") is False

    async def test_get_info_mount_path(self, rw_ufs):
        ufs, _ = rw_ufs
        info = await ufs.get_info("/local")
        assert info is not None
        assert info.is_directory is True
        assert info.path == "/local"

    async def test_get_info_no_mount(self, rw_ufs):
        ufs, _ = rw_ufs
        info = await ufs.get_info("/nonexistent")
        assert info is None

    async def test_get_info_file(self, rw_ufs):
        ufs, _ = rw_ufs
        await ufs.write("/local/file.txt", "content")
        info = await ufs.get_info("/local/file.txt")
        assert info is not None
        assert info.path.startswith("/local/")


# =========================================================================
# Cross-Mount Operations
# =========================================================================


class TestCrossMountOperations:
    async def test_cross_mount_move_success(self, two_mount_ufs):
        ufs, collected = two_mount_ufs
        await ufs.write("/local/a.txt", "cross-mount data")
        collected.clear()

        result = await ufs.move("/local/a.txt", "/other/a.txt")
        assert result.success
        # Content exists at new location
        read = await ufs.read("/other/a.txt")
        assert read.success
        assert read.content == "cross-mount data"
        # Source deleted
        assert await ufs.exists("/local/a.txt") is False
        # Move event emitted
        move_events = [e for e in collected if e.event_type is EventType.FILE_MOVED]
        assert len(move_events) == 1
        assert move_events[0].path == "/other/a.txt"
        assert move_events[0].old_path == "/local/a.txt"

    async def test_cross_mount_move_read_fails(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        result = await ufs.move("/local/nonexistent.txt", "/other/dest.txt")
        assert result.success is False

    async def test_cross_mount_copy_success(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        await ufs.write("/local/src.txt", "copy me")
        result = await ufs.copy("/local/src.txt", "/other/dest.txt")
        assert result.success
        # Both should exist
        assert (await ufs.read("/other/dest.txt")).content == "copy me"
        assert await ufs.exists("/local/src.txt")

    async def test_cross_mount_copy_read_fails(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        result = await ufs.copy("/local/nonexistent.txt", "/other/dest.txt")
        assert result.success is False


# =========================================================================
# Trash Operations
# =========================================================================


class TestTrashOperations:
    async def test_list_trash_aggregates_mounts(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        await ufs.write("/local/a.txt", "a")
        await ufs.write("/other/b.txt", "b")
        await ufs.delete("/local/a.txt")
        await ufs.delete("/other/b.txt")
        result = await ufs.list_trash()
        assert result.success
        assert len(result.entries) >= 2

    async def test_empty_trash_aggregates_mounts(self, two_mount_ufs):
        ufs, _ = two_mount_ufs
        await ufs.write("/local/a.txt", "a")
        await ufs.write("/other/b.txt", "b")
        await ufs.delete("/local/a.txt")
        await ufs.delete("/other/b.txt")
        result = await ufs.empty_trash()
        assert result.success
        assert result.total_deleted is not None
        assert result.total_deleted >= 2


# =========================================================================
# Context Manager / Backend Lifecycle
# =========================================================================


class TestContextManager:
    async def test_close_all_backends(self, tmp_path: Path):
        ufs, _, _, _ = await _make_local_setup(tmp_path)
        async with ufs:
            assert len(ufs._entered_backends) > 0
        assert len(ufs._entered_backends) == 0

    async def test_enter_exit_backend(self, tmp_path: Path):
        (tmp_path / "ws_manual").mkdir(exist_ok=True)
        backend = LocalFileSystem(
            workspace_dir=tmp_path / "ws_manual",
            data_dir=tmp_path / ".grover_manual",
        )
        registry = MountRegistry()
        ufs = UnifiedFileSystem(registry)

        await ufs.enter_backend(backend)
        assert backend in ufs._entered_backends

        await ufs.exit_backend(backend)
        assert backend not in ufs._entered_backends


# =========================================================================
# Version Operations (delegates correctly)
# =========================================================================


class TestVersionOperations:
    @pytest.fixture
    async def vfs_ufs(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        db = DatabaseFileSystem(dialect="sqlite")

        registry = MountRegistry()
        registry.add_mount(MountConfig(
            mount_path="/vfs", backend=db, session_factory=factory, mount_type="vfs",
        ))
        ufs = UnifiedFileSystem(registry)
        async with ufs:
            yield ufs
        await engine.dispose()

    async def test_list_versions_routes(self, vfs_ufs):
        ufs = vfs_ufs
        await ufs.write("/vfs/doc.txt", "v1")
        await ufs.write("/vfs/doc.txt", "v2")
        versions = await ufs.list_versions("/vfs/doc.txt")
        assert len(versions) >= 2

    async def test_get_version_content_routes(self, vfs_ufs):
        ufs = vfs_ufs
        await ufs.write("/vfs/doc.txt", "first")
        await ufs.write("/vfs/doc.txt", "second")
        content = await ufs.get_version_content("/vfs/doc.txt", 1)
        assert content == "first"

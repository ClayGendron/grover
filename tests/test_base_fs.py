"""Tests for BaseFileSystem — versioning, CRUD, trash via DatabaseFileSystem."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.diff import SNAPSHOT_INTERVAL


async def _make_fs() -> tuple[DatabaseFileSystem, object]:
    """Create a DatabaseFileSystem backed by in-memory SQLite."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(session_factory=factory, dialect="sqlite")
    return fs, engine


# ---------------------------------------------------------------------------
# Write + Read
# ---------------------------------------------------------------------------


class TestWriteRead:
    async def test_write_creates_file(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.write("/hello.py", "print('hi')\n")
            assert result.success is True
            assert result.created is True

            read = await fs.read("/hello.py")
            assert read.success is True
            assert "print('hi')" in read.content
        await engine.dispose()

    async def test_write_updates_file(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "v1\n")
            result = await fs.write("/f.py", "v2\n")
            assert result.success is True
            assert result.created is False
            assert result.version == 2
        await engine.dispose()

    async def test_write_non_text_rejected(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.write("/image.png", "data")
            assert result.success is False
        await engine.dispose()

    async def test_read_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.read("/nope.py")
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


class TestEdit:
    async def test_edit_exact_match(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "hello world\n")
            result = await fs.edit("/f.py", "world", "earth")
            assert result.success is True

            read = await fs.read("/f.py")
            assert "earth" in read.content
        await engine.dispose()

    async def test_edit_increments_version(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "line1\n")
            result = await fs.edit("/f.py", "line1", "line2")
            assert result.version == 2
        await engine.dispose()

    async def test_edit_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.edit("/nope.py", "a", "b")
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Diff-based Versioning
# ---------------------------------------------------------------------------


class TestVersioning:
    async def test_version_content_round_trip(self):
        """Write + 5 edits, then retrieve each intermediate version."""
        fs, engine = await _make_fs()
        async with fs:
            content_v1 = "line1\nline2\nline3\n"
            await fs.write("/f.py", content_v1)

            contents = [content_v1]
            for i in range(5):
                old_line = f"line{i + 1}" if i == 0 else f"edited_{i}"
                new_line = f"edited_{i + 1}"
                prev = contents[-1]
                new = prev.replace(old_line, new_line, 1)
                await fs.edit("/f.py", old_line, new_line)
                contents.append(new)

            # Verify each version can be reconstructed
            for version_num in range(1, len(contents) + 1):
                vc = await fs.get_version_content("/f.py", version_num)
                assert vc is not None, f"Version {version_num} returned None"
                assert vc == contents[version_num - 1], (
                    f"Version {version_num} mismatch"
                )
        await engine.dispose()

    async def test_snapshot_interval(self):
        """Verify snapshots are stored at the configured interval."""
        fs, engine = await _make_fs()
        async with fs:
            # Write initial (version 1 = snapshot)
            await fs.write("/f.py", "version 0\n")

            # Edit up to SNAPSHOT_INTERVAL writes
            for i in range(1, SNAPSHOT_INTERVAL + 1):
                old = f"version {i - 1}"
                new = f"version {i}"
                await fs.edit("/f.py", old, new)

            # List versions and check
            versions = await fs.list_versions("/f.py")
            assert len(versions) > 0

            # Version 1 (initial write) should be a snapshot
            vc1 = await fs.get_version_content("/f.py", 1)
            assert vc1 is not None
        await engine.dispose()

    async def test_get_version_content_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.get_version_content("/nope.py", 1)
            assert result is None
        await engine.dispose()


# ---------------------------------------------------------------------------
# List Versions
# ---------------------------------------------------------------------------


class TestListVersions:
    async def test_list_versions(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "v1\n")
            await fs.edit("/f.py", "v1", "v2")

            versions = await fs.list_versions("/f.py")
            assert len(versions) >= 1
            # Versions are ordered descending
            assert versions[0].version >= versions[-1].version
        await engine.dispose()

    async def test_list_versions_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            versions = await fs.list_versions("/nope.py")
            assert versions == []
        await engine.dispose()


# ---------------------------------------------------------------------------
# Restore Version
# ---------------------------------------------------------------------------


class TestRestoreVersion:
    async def test_restore_version(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "original\n")
            await fs.edit("/f.py", "original", "modified")

            result = await fs.restore_version("/f.py", 1)
            assert result.success is True
            assert result.restored_version == 1

            read = await fs.read("/f.py")
            assert "original" in read.content
        await engine.dispose()

    async def test_restore_nonexistent_version(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "content\n")
            result = await fs.restore_version("/f.py", 999)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Delete (soft / permanent)
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_soft_delete(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "content\n")
            result = await fs.delete("/f.py")
            assert result.success is True
            assert result.permanent is False

            # File should no longer be readable
            read = await fs.read("/f.py")
            assert read.success is False
        await engine.dispose()

    async def test_permanent_delete(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "content\n")
            result = await fs.delete("/f.py", permanent=True)
            assert result.success is True
            assert result.permanent is True
        await engine.dispose()

    async def test_permanent_delete_cleans_versions(self):
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "v1\n")
            await fs.write("/f.py", "v2\n")
            versions = await fs.list_versions("/f.py")
            assert len(versions) == 2

            await fs.delete("/f.py", permanent=True)

        # Verify no orphaned version records remain
        factory = fs.session_factory
        async with factory() as session:
            result = await session.execute(select(FileVersion))
            assert result.scalars().all() == [], "Version records should be deleted"
        await engine.dispose()

    async def test_delete_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.delete("/nope.py")
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Trash Operations
# ---------------------------------------------------------------------------


class TestTrash:
    async def test_list_trash(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "content\n")
            await fs.delete("/f.py")

            trash = await fs.list_trash()
            assert trash.success is True
            assert len(trash.entries) == 1
        await engine.dispose()

    async def test_restore_from_trash(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "content\n")
            await fs.delete("/f.py")

            result = await fs.restore_from_trash("/f.py")
            assert result.success is True

            read = await fs.read("/f.py")
            assert read.success is True
            assert "content" in read.content
        await engine.dispose()

    async def test_restore_not_in_trash(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.restore_from_trash("/nope.py")
            assert result.success is False
        await engine.dispose()

    async def test_empty_trash(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/a.py", "a\n")
            await fs.write("/b.py", "b\n")
            await fs.delete("/a.py")
            await fs.delete("/b.py")

            result = await fs.empty_trash()
            assert result.success is True
            assert result.total_deleted == 2

            trash = await fs.list_trash()
            assert len(trash.entries) == 0
        await engine.dispose()

    async def test_empty_trash_cleans_versions(self):
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/a.py", "v1\n")
            await fs.write("/a.py", "v2\n")
            await fs.delete("/a.py")
            await fs.empty_trash()

        factory = fs.session_factory
        async with factory() as session:
            result = await session.execute(select(FileVersion))
            assert result.scalars().all() == [], "Version records should be deleted"
        await engine.dispose()


# ---------------------------------------------------------------------------
# Directory Operations
# ---------------------------------------------------------------------------


class TestDirectoryOps:
    async def test_mkdir(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.mkdir("/src")
            assert result.success is True
            assert "/src" in result.created_dirs
        await engine.dispose()

    async def test_mkdir_parents(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.mkdir("/a/b/c")
            assert result.success is True
            assert len(result.created_dirs) >= 1
        await engine.dispose()

    async def test_mkdir_existing(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/src")
            result = await fs.mkdir("/src")
            assert result.success is True
            assert result.created_dirs == []
        await engine.dispose()

    async def test_list_dir(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/hello.py", "content\n")
            await fs.mkdir("/src")

            result = await fs.list_dir("/")
            assert result.success is True
            names = [e.name for e in result.entries]
            assert "hello.py" in names
            assert "src" in names
        await engine.dispose()


# ---------------------------------------------------------------------------
# Move / Copy
# ---------------------------------------------------------------------------


class TestMoveCopy:
    async def test_move_file(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/a.py", "content\n")
            result = await fs.move("/a.py", "/b.py")
            assert result.success is True

            assert await fs.exists("/a.py") is False
            assert await fs.exists("/b.py") is True
        await engine.dispose()

    async def test_copy_file(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/a.py", "content\n")
            result = await fs.copy("/a.py", "/b.py")
            assert result.success is True

            assert await fs.exists("/a.py") is True
            assert await fs.exists("/b.py") is True
        await engine.dispose()


# ---------------------------------------------------------------------------
# Exists / GetInfo
# ---------------------------------------------------------------------------


class TestExistsGetInfo:
    async def test_exists(self):
        fs, engine = await _make_fs()
        async with fs:
            assert await fs.exists("/nope.py") is False
            await fs.write("/f.py", "x\n")
            assert await fs.exists("/f.py") is True
        await engine.dispose()

    async def test_get_info(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "hello\n")
            info = await fs.get_info("/f.py")
            assert info is not None
            assert info.name == "f.py"
            assert info.is_directory is False
        await engine.dispose()

    async def test_get_info_nonexistent(self):
        fs, engine = await _make_fs()
        async with fs:
            info = await fs.get_info("/nope.py")
            assert info is None
        await engine.dispose()

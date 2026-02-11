"""Tests for BaseFileSystem — versioning, CRUD, trash via DatabaseFileSystem."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.diff import SNAPSHOT_INTERVAL
from grover.fs.exceptions import ConsistencyError


async def _make_fs():
    """Create a stateless DatabaseFileSystem + session factory + engine."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(dialect="sqlite")
    return fs, factory, engine


# ---------------------------------------------------------------------------
# Write + Read
# ---------------------------------------------------------------------------


class TestWriteRead:
    async def test_write_creates_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.write("/hello.py", "print('hi')\n", session=session)
            assert result.success is True
            assert result.created is True

            read = await fs.read("/hello.py", session=session)
            assert read.success is True
            assert "print('hi')" in read.content
        await engine.dispose()

    async def test_write_updates_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            result = await fs.write("/f.py", "v2\n", session=session)
            assert result.success is True
            assert result.created is False
            assert result.version == 2
        await engine.dispose()

    async def test_write_non_text_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.write("/image.png", "data", session=session)
            assert result.success is False
        await engine.dispose()

    async def test_read_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.read("/nope.py", session=session)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


class TestEdit:
    async def test_edit_exact_match(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "hello world\n", session=session)
            result = await fs.edit("/f.py", "world", "earth", session=session)
            assert result.success is True

            read = await fs.read("/f.py", session=session)
            assert "earth" in read.content
        await engine.dispose()

    async def test_edit_increments_version(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "line1\n", session=session)
            result = await fs.edit("/f.py", "line1", "line2", session=session)
            assert result.version == 2
        await engine.dispose()

    async def test_edit_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.edit("/nope.py", "a", "b", session=session)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Diff-based Versioning
# ---------------------------------------------------------------------------


class TestVersioning:
    async def test_version_content_round_trip(self):
        """Write + 5 edits, then retrieve each intermediate version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            content_v1 = "line1\nline2\nline3\n"
            await fs.write("/f.py", content_v1, session=session)

            contents = [content_v1]
            for i in range(5):
                old_line = f"line{i + 1}" if i == 0 else f"edited_{i}"
                new_line = f"edited_{i + 1}"
                prev = contents[-1]
                new = prev.replace(old_line, new_line, 1)
                await fs.edit("/f.py", old_line, new_line, session=session)
                contents.append(new)

            # Verify each version can be reconstructed
            for version_num in range(1, len(contents) + 1):
                vc = await fs.get_version_content("/f.py", version_num, session=session)
                assert vc.success, f"Version {version_num} failed: {vc.message}"
                assert vc.content == contents[version_num - 1], (
                    f"Version {version_num} mismatch"
                )
        await engine.dispose()

    async def test_snapshot_interval(self):
        """Verify snapshots are stored at the configured interval."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "version 0\n", session=session)

            for i in range(1, SNAPSHOT_INTERVAL + 1):
                old = f"version {i - 1}"
                new = f"version {i}"
                await fs.edit("/f.py", old, new, session=session)

            result = await fs.list_versions("/f.py", session=session)
            assert len(result.versions) > 0

            vc1 = await fs.get_version_content("/f.py", 1, session=session)
            assert vc1.success
        await engine.dispose()

    async def test_get_version_content_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.get_version_content("/nope.py", 1, session=session)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# List Versions
# ---------------------------------------------------------------------------


class TestListVersions:
    async def test_list_versions(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            await fs.edit("/f.py", "v1", "v2", session=session)

            result = await fs.list_versions("/f.py", session=session)
            assert len(result.versions) >= 1
            assert result.versions[0].version >= result.versions[-1].version
        await engine.dispose()

    async def test_list_versions_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.list_versions("/nope.py", session=session)
            assert result.versions == []
        await engine.dispose()


# ---------------------------------------------------------------------------
# Restore Version
# ---------------------------------------------------------------------------


class TestRestoreVersion:
    async def test_restore_version(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "original\n", session=session)
            await fs.edit("/f.py", "original", "modified", session=session)

            result = await fs.restore_version("/f.py", 1, session=session)
            assert result.success is True
            assert result.restored_version == 1

            read = await fs.read("/f.py", session=session)
            assert "original" in read.content
        await engine.dispose()

    async def test_restore_nonexistent_version(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            result = await fs.restore_version("/f.py", 999, session=session)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Delete (soft / permanent)
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_soft_delete(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            result = await fs.delete("/f.py", session=session)
            assert result.success is True
            assert result.permanent is False

            read = await fs.read("/f.py", session=session)
            assert read.success is False
        await engine.dispose()

    async def test_permanent_delete(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            result = await fs.delete("/f.py", permanent=True, session=session)
            assert result.success is True
            assert result.permanent is True
        await engine.dispose()

    async def test_permanent_delete_cleans_versions(self):
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            await fs.write("/f.py", "v2\n", session=session)
            ver_result = await fs.list_versions("/f.py", session=session)
            assert len(ver_result.versions) == 2

            await fs.delete("/f.py", permanent=True, session=session)
            await session.commit()

        # Verify no orphaned version records remain
        async with factory() as session:
            db_result = await session.execute(select(FileVersion))
            assert db_result.scalars().all() == [], "Version records should be deleted"
        await engine.dispose()

    async def test_delete_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.delete("/nope.py", session=session)
            assert result.success is False
        await engine.dispose()


# ---------------------------------------------------------------------------
# Trash Operations
# ---------------------------------------------------------------------------


class TestTrash:
    async def test_list_trash(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            await fs.delete("/f.py", session=session)

            trash = await fs.list_trash(session=session)
            assert trash.success is True
            assert len(trash.entries) == 1
        await engine.dispose()

    async def test_restore_from_trash(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            await fs.delete("/f.py", session=session)

            result = await fs.restore_from_trash("/f.py", session=session)
            assert result.success is True

            read = await fs.read("/f.py", session=session)
            assert read.success is True
            assert "content" in read.content
        await engine.dispose()

    async def test_restore_not_in_trash(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.restore_from_trash("/nope.py", session=session)
            assert result.success is False
        await engine.dispose()

    async def test_empty_trash(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "a\n", session=session)
            await fs.write("/b.py", "b\n", session=session)
            await fs.delete("/a.py", session=session)
            await fs.delete("/b.py", session=session)

            result = await fs.empty_trash(session=session)
            assert result.success is True
            assert result.total_deleted == 2

            trash = await fs.list_trash(session=session)
            assert len(trash.entries) == 0
        await engine.dispose()

    async def test_empty_trash_cleans_versions(self):
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "v1\n", session=session)
            await fs.write("/a.py", "v2\n", session=session)
            await fs.delete("/a.py", session=session)
            await fs.empty_trash(session=session)
            await session.commit()

        async with factory() as session:
            result = await session.execute(select(FileVersion))
            assert result.scalars().all() == [], "Version records should be deleted"
        await engine.dispose()

    async def test_soft_delete_directory_trashes_children(self):
        """H3: Soft-deleting a directory should also trash all children."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/mydir", session=session)
            await fs.write("/mydir/a.py", "a\n", session=session)
            await fs.write("/mydir/b.py", "b\n", session=session)

            result = await fs.delete("/mydir", session=session)
            assert result.success is True

            assert await fs.exists("/mydir/a.py", session=session) is False
            assert await fs.exists("/mydir/b.py", session=session) is False

            trash = await fs.list_trash(session=session)
            paths = [e.path for e in trash.entries]
            assert "/mydir" in paths
            assert "/mydir/a.py" in paths
            assert "/mydir/b.py" in paths
        await engine.dispose()

    async def test_restore_directory_restores_children(self):
        """H3: Restoring a directory from trash also restores children."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/mydir", session=session)
            await fs.write("/mydir/a.py", "a content\n", session=session)
            await fs.write("/mydir/b.py", "b content\n", session=session)

            await fs.delete("/mydir", session=session)

            result = await fs.restore_from_trash("/mydir", session=session)
            assert result.success is True

            assert await fs.exists("/mydir/a.py", session=session) is True
            assert await fs.exists("/mydir/b.py", session=session) is True

            read_a = await fs.read("/mydir/a.py", session=session)
            assert "a content" in read_a.content
            read_b = await fs.read("/mydir/b.py", session=session)
            assert "b content" in read_b.content
        await engine.dispose()


# ---------------------------------------------------------------------------
# Directory Operations
# ---------------------------------------------------------------------------


class TestDirectoryOps:
    async def test_mkdir(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.mkdir("/src", session=session)
            assert result.success is True
            assert "/src" in result.created_dirs
        await engine.dispose()

    async def test_mkdir_parents(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.mkdir("/a/b/c", session=session)
            assert result.success is True
            assert len(result.created_dirs) >= 1
        await engine.dispose()

    async def test_mkdir_existing(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/src", session=session)
            result = await fs.mkdir("/src", session=session)
            assert result.success is True
            assert result.created_dirs == []
        await engine.dispose()

    async def test_list_dir(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/hello.py", "content\n", session=session)
            await fs.mkdir("/src", session=session)

            result = await fs.list_dir("/", session=session)
            assert result.success is True
            names = [e.name for e in result.entries]
            assert "hello.py" in names
            assert "src" in names
        await engine.dispose()

    async def test_list_dir_only_direct_children(self):
        """H2: list_dir should only return direct children, not nested files."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/src", session=session)
            await fs.write("/src/main.py", "main\n", session=session)
            await fs.write("/src/lib/helper.py", "helper\n", session=session)
            await fs.write("/readme.md", "# readme\n", session=session)

            root_result = await fs.list_dir("/", session=session)
            root_names = [e.name for e in root_result.entries]
            assert "src" in root_names
            assert "readme.md" in root_names
            assert "main.py" not in root_names
            assert "helper.py" not in root_names

            src_result = await fs.list_dir("/src", session=session)
            src_names = [e.name for e in src_result.entries]
            assert "main.py" in src_names
            assert "lib" in src_names
            assert "helper.py" not in src_names
        await engine.dispose()

    async def test_list_dir_excludes_deleted(self):
        """H2: Deleted files should not appear in list_dir."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "a\n", session=session)
            await fs.write("/b.py", "b\n", session=session)
            await fs.delete("/a.py", session=session)

            result = await fs.list_dir("/", session=session)
            names = [e.name for e in result.entries]
            assert "a.py" not in names
            assert "b.py" in names
        await engine.dispose()


# ---------------------------------------------------------------------------
# Move / Copy
# ---------------------------------------------------------------------------


class TestMoveCopy:
    async def test_move_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "content\n", session=session)
            result = await fs.move("/a.py", "/b.py", session=session)
            assert result.success is True

            assert await fs.exists("/a.py", session=session) is False
            assert await fs.exists("/b.py", session=session) is True
        await engine.dispose()

    async def test_move_empty_file_to_existing(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/empty.py", "", session=session)
            await fs.write("/target.py", "old\n", session=session)

            result = await fs.move("/empty.py", "/target.py", session=session)
            assert result.success is True
            assert await fs.exists("/empty.py", session=session) is False
            assert await fs.exists("/target.py", session=session) is True
        await engine.dispose()

    async def test_copy_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "content\n", session=session)
            result = await fs.copy("/a.py", "/b.py", session=session)
            assert result.success is True

            assert await fs.exists("/a.py", session=session) is True
            assert await fs.exists("/b.py", session=session) is True
        await engine.dispose()

    async def test_move_to_existing_file_overwrites(self):
        """C3: Moving to an existing file overwrites the destination."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src.py", "source content\n", session=session)
            await fs.write("/dest.py", "old dest content\n", session=session)

            result = await fs.move("/src.py", "/dest.py", session=session)
            assert result.success is True

            assert await fs.exists("/src.py", session=session) is False

            read = await fs.read("/dest.py", session=session)
            assert read.success is True
            assert "source content" in read.content
        await engine.dispose()

    async def test_move_to_existing_directory_rejected(self):
        """C3: Moving to an existing directory is rejected."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src.py", "content\n", session=session)
            await fs.mkdir("/dest", session=session)

            result = await fs.move("/src.py", "/dest", session=session)
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_over_file_rejected(self):
        """C3: Moving a directory over a file is rejected."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/srcdir", session=session)
            await fs.write("/srcdir/child.py", "child\n", session=session)
            await fs.write("/dest.py", "content\n", session=session)

            result = await fs.move("/srcdir", "/dest.py", session=session)
            assert result.success is False
            assert "cannot move directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_to_existing_dir_rejected(self):
        """C3: Moving a directory to an existing directory is rejected."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/a", session=session)
            await fs.write("/a/child.py", "from a\n", session=session)
            await fs.mkdir("/b", session=session)
            await fs.write("/b/child.py", "from b\n", session=session)

            result = await fs.move("/a", "/b", session=session)
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_content_preserved(self):
        """C2: Directory move preserves all children's content."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/src", session=session)
            await fs.write("/src/one.py", "file one\n", session=session)
            await fs.write("/src/two.py", "file two\n", session=session)

            result = await fs.move("/src", "/dst", session=session)
            assert result.success is True

            read1 = await fs.read("/dst/one.py", session=session)
            assert read1.success is True
            assert "file one" in read1.content

            read2 = await fs.read("/dst/two.py", session=session)
            assert read2.success is True
            assert "file two" in read2.content
        await engine.dispose()


# ---------------------------------------------------------------------------
# Exists / GetInfo
# ---------------------------------------------------------------------------


class TestExistsGetInfo:
    async def test_exists(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            assert await fs.exists("/nope.py", session=session) is False
            await fs.write("/f.py", "x\n", session=session)
            assert await fs.exists("/f.py", session=session) is True
        await engine.dispose()

    async def test_get_info(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "hello\n", session=session)
            info = await fs.get_info("/f.py", session=session)
            assert info is not None
            assert info.name == "f.py"
            assert info.is_directory is False
        await engine.dispose()

    async def test_get_info_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            info = await fs.get_info("/nope.py", session=session)
            assert info is None
        await engine.dispose()


# ---------------------------------------------------------------------------
# Hash Validation
# ---------------------------------------------------------------------------


class TestHashValidation:
    async def test_version_content_hash_verified(self):
        """Corrupt a version's content_hash in DB, get_version_content raises ConsistencyError."""
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "original content\n", session=session)
            await session.flush()

            # Corrupt the stored content_hash for version 1
            result = await session.execute(
                select(FileVersion).where(FileVersion.version == 1)
            )
            ver = result.scalar_one()
            ver.content_hash = "0000000000000000000000000000000000000000000000000000000000000000"
            await session.flush()

            with pytest.raises(ConsistencyError, match="hash mismatch"):
                await fs.get_version_content("/f.py", 1, session=session)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Path Validation on exists / get_info
# ---------------------------------------------------------------------------


class TestPathValidationExistsGetInfo:
    async def test_exists_null_byte_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            assert await fs.exists("/foo\x00bar", session=session) is False
        await engine.dispose()

    async def test_get_info_null_byte_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            info = await fs.get_info("/foo\x00bar", session=session)
            assert info is None
        await engine.dispose()


# ---------------------------------------------------------------------------
# Control Characters
# ---------------------------------------------------------------------------


class TestControlChars:
    async def test_write_control_char_path_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.write("/bad\x01file.py", "content\n", session=session)
            assert result.success is False
            assert "control character" in result.message.lower()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_read_empty_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/empty.py", "", session=session)
            result = await fs.read("/empty.py", session=session)
            assert result.success is True
            assert result.content == ""
        await engine.dispose()

    async def test_write_overwrite_false(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            result = await fs.write("/f.py", "v2\n", overwrite=False, session=session)
            assert result.success is False
            assert "already exists" in result.message
        await engine.dispose()

    async def test_move_directory_with_children(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/a", session=session)
            await fs.write("/a/child.py", "child\n", session=session)

            result = await fs.move("/a", "/b", session=session)
            assert result.success is True

            assert await fs.exists("/b/child.py", session=session) is True
            assert await fs.exists("/a/child.py", session=session) is False
        await engine.dispose()

    async def test_copy_directory_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/src", session=session)
            result = await fs.copy("/src", "/dst", session=session)
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_read_directory_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/mydir", session=session)
            result = await fs.read("/mydir", session=session)
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_edit_directory_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/mydir", session=session)
            result = await fs.edit("/mydir", "old", "new", session=session)
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Parent Path
# ---------------------------------------------------------------------------


class TestParentPath:
    """H5: parent_path should be populated on write, mkdir, and move."""

    async def test_write_sets_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src/main.py", "content\n", session=session)

            info = await fs.get_info("/src/main.py", session=session)
            assert info is not None

            file = await fs.metadata.get_file(session, "/src/main.py")
            assert file.parent_path == "/src"
        await engine.dispose()

    async def test_mkdir_sets_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/a/b/c", session=session)

            b = await fs.metadata.get_file(session, "/a/b")
            assert b is not None
            assert b.parent_path == "/a"

            c = await fs.metadata.get_file(session, "/a/b/c")
            assert c is not None
            assert c.parent_path == "/a/b"
        await engine.dispose()

    async def test_move_updates_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src/file.py", "content\n", session=session)
            await fs.mkdir("/dst", session=session)
            await fs.move("/src/file.py", "/dst/file.py", session=session)

            file = await fs.metadata.get_file(session, "/dst/file.py")
            assert file is not None
            assert file.parent_path == "/dst"
        await engine.dispose()

    async def test_root_file_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/root_file.py", "content\n", session=session)

            file = await fs.metadata.get_file(session, "/root_file.py")
            assert file.parent_path == "/"
        await engine.dispose()


# ---------------------------------------------------------------------------
# Version Reconstruction Across Snapshots
# ---------------------------------------------------------------------------


class TestVersionReconstructionAcrossSnapshots:
    async def test_25_edits_spanning_2_snapshot_intervals(self):
        """25+ edits spanning 2 snapshot intervals, verify all versions."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            initial = "version_0\nstatic_line\n"
            await fs.write("/f.py", initial, session=session)
            contents = [initial]

            for i in range(1, 26):
                old_marker = f"version_{i - 1}"
                new_marker = f"version_{i}"
                prev = contents[-1]
                new = prev.replace(old_marker, new_marker, 1)
                await fs.edit("/f.py", old_marker, new_marker, session=session)
                contents.append(new)

            for version_num in range(1, len(contents) + 1):
                vc = await fs.get_version_content("/f.py", version_num, session=session)
                assert vc.success, f"Version {version_num} failed: {vc.message}"
                assert vc.content == contents[version_num - 1], (
                    f"Version {version_num} mismatch"
                )
        await engine.dispose()


# ---------------------------------------------------------------------------
# C2/C4/C5 Move Guards and Overwrite Tests
# ---------------------------------------------------------------------------


class TestMoveGuards:
    """C5: Self-move and move-into-self guards."""

    async def test_self_move_is_noop(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/file.py", "content\n", session=session)
            result = await fs.move("/file.py", "/file.py", session=session)
            assert result.success is True
            read = await fs.read("/file.py", session=session)
            assert read.content == "content\n"
        await engine.dispose()

    async def test_move_directory_into_itself_rejected(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/parent", session=session)
            await fs.write("/parent/child.py", "child\n", session=session)
            result = await fs.move("/parent", "/parent/sub", session=session)
            assert result.success is False
        await engine.dispose()

    async def test_move_file_subpath_still_works(self):
        """Moving /a/b.py to /a/c.py is fine (not moving into self)."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a/b.py", "content\n", session=session)
            result = await fs.move("/a/b.py", "/a/c.py", session=session)
            assert result.success is True
            assert await fs.exists("/a/c.py", session=session) is True
        await engine.dispose()


class TestAtomicMoveOverwrite:
    """C2: Atomic move-overwrite preserves dest history."""

    async def test_move_overwrite_is_atomic(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src.py", "source\n", session=session)
            await fs.write("/dest.py", "dest\n", session=session)
            result = await fs.move("/src.py", "/dest.py", session=session)
            assert result.success is True
            read = await fs.read("/dest.py", session=session)
            assert read.content == "source\n"
            assert await fs.exists("/src.py", session=session) is False
        await engine.dispose()

    async def test_move_overwrite_preserves_dest_history(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src.py", "source\n", session=session)
            await fs.write("/dest.py", "old dest\n", session=session)
            await fs.move("/src.py", "/dest.py", session=session)
            # dest should have version history
            ver_result = await fs.list_versions("/dest.py", session=session)
            assert len(ver_result.versions) >= 2
        await engine.dispose()


class TestRestoreOverwrite:
    """C4: restore_from_trash overwrites occupant."""

    async def test_restore_from_trash_overwrites_occupant(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/file.py", "original\n", session=session)
            await fs.delete("/file.py", session=session)
            # Write a new file at the same path
            await fs.write("/file.py", "occupant\n", session=session)
            # Restore should overwrite the occupant
            result = await fs.restore_from_trash("/file.py", session=session)
            assert result.success is True
            read = await fs.read("/file.py", session=session)
            assert read.content == "original\n"
        await engine.dispose()

    async def test_restore_from_trash_no_conflict(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/file.py", "content\n", session=session)
            await fs.delete("/file.py", session=session)
            result = await fs.restore_from_trash("/file.py", session=session)
            assert result.success is True
            read = await fs.read("/file.py", session=session)
            assert read.content == "content\n"
        await engine.dispose()

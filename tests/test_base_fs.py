"""Tests for BaseFileSystem — versioning, CRUD, trash via DatabaseFileSystem."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.diff import SNAPSHOT_INTERVAL
from grover.fs.exceptions import ConsistencyError


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

    async def test_soft_delete_directory_trashes_children(self):
        """H3: Soft-deleting a directory should also trash all children."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/mydir")
            await fs.write("/mydir/a.py", "a\n")
            await fs.write("/mydir/b.py", "b\n")

            result = await fs.delete("/mydir")
            assert result.success is True

            # Children should be gone
            assert await fs.exists("/mydir/a.py") is False
            assert await fs.exists("/mydir/b.py") is False

            # All three should be in trash
            trash = await fs.list_trash()
            paths = [e.path for e in trash.entries]
            assert "/mydir" in paths
            assert "/mydir/a.py" in paths
            assert "/mydir/b.py" in paths
        await engine.dispose()

    async def test_restore_directory_restores_children(self):
        """H3: Restoring a directory from trash also restores children."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/mydir")
            await fs.write("/mydir/a.py", "a content\n")
            await fs.write("/mydir/b.py", "b content\n")

            await fs.delete("/mydir")

            result = await fs.restore_from_trash("/mydir")
            assert result.success is True

            # Children should be back
            assert await fs.exists("/mydir/a.py") is True
            assert await fs.exists("/mydir/b.py") is True

            read_a = await fs.read("/mydir/a.py")
            assert "a content" in read_a.content
            read_b = await fs.read("/mydir/b.py")
            assert "b content" in read_b.content
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

    async def test_list_dir_only_direct_children(self):
        """H2: list_dir should only return direct children, not nested files."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/src")
            await fs.write("/src/main.py", "main\n")
            await fs.write("/src/lib/helper.py", "helper\n")
            await fs.write("/readme.md", "# readme\n")

            # Root should only list src/ and readme.md, not nested files
            root_result = await fs.list_dir("/")
            root_names = [e.name for e in root_result.entries]
            assert "src" in root_names
            assert "readme.md" in root_names
            assert "main.py" not in root_names
            assert "helper.py" not in root_names

            # /src should list main.py and lib/, not nested helper.py
            src_result = await fs.list_dir("/src")
            src_names = [e.name for e in src_result.entries]
            assert "main.py" in src_names
            assert "lib" in src_names
            assert "helper.py" not in src_names
        await engine.dispose()

    async def test_list_dir_excludes_deleted(self):
        """H2: Deleted files should not appear in list_dir."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/a.py", "a\n")
            await fs.write("/b.py", "b\n")
            await fs.delete("/a.py")

            result = await fs.list_dir("/")
            names = [e.name for e in result.entries]
            assert "a.py" not in names
            assert "b.py" in names
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

    async def test_move_empty_file_to_existing(self):
        """Moving an empty file onto an existing file should succeed."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/empty.py", "")
            await fs.write("/target.py", "old\n")

            result = await fs.move("/empty.py", "/target.py")
            assert result.success is True
            assert await fs.exists("/empty.py") is False
            assert await fs.exists("/target.py") is True
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

    async def test_move_to_existing_file_overwrites(self):
        """C3: Moving to an existing file overwrites the destination."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/src.py", "source content\n")
            await fs.write("/dest.py", "old dest content\n")

            result = await fs.move("/src.py", "/dest.py")
            assert result.success is True

            # Source should be gone
            assert await fs.exists("/src.py") is False

            # Dest should have source content
            read = await fs.read("/dest.py")
            assert read.success is True
            assert "source content" in read.content
        await engine.dispose()

    async def test_move_to_existing_directory_rejected(self):
        """C3: Moving to an existing directory is rejected."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/src.py", "content\n")
            await fs.mkdir("/dest")

            result = await fs.move("/src.py", "/dest")
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_over_file_rejected(self):
        """C3: Moving a directory over a file is rejected."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/srcdir")
            await fs.write("/srcdir/child.py", "child\n")
            await fs.write("/dest.py", "content\n")

            result = await fs.move("/srcdir", "/dest.py")
            assert result.success is False
            assert "cannot move directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_to_existing_dir_rejected(self):
        """C3: Moving a directory to an existing directory is rejected."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/a")
            await fs.write("/a/child.py", "from a\n")
            await fs.mkdir("/b")
            await fs.write("/b/child.py", "from b\n")

            result = await fs.move("/a", "/b")
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_move_directory_content_preserved(self):
        """C2: Directory move preserves all children's content."""
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/src")
            await fs.write("/src/one.py", "file one\n")
            await fs.write("/src/two.py", "file two\n")

            result = await fs.move("/src", "/dst")
            assert result.success is True

            read1 = await fs.read("/dst/one.py")
            assert read1.success is True
            assert "file one" in read1.content

            read2 = await fs.read("/dst/two.py")
            assert read2.success is True
            assert "file two" in read2.content
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


# ---------------------------------------------------------------------------
# Hash Validation
# ---------------------------------------------------------------------------


class TestHashValidation:
    async def test_version_content_hash_verified(self):
        """Corrupt a version's content_hash in DB, get_version_content raises ConsistencyError."""
        from sqlmodel import select

        from grover.models.files import FileVersion

        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "original content\n")

            # Corrupt the stored content_hash for version 1
            session = await fs._get_session()
            result = await session.execute(
                select(FileVersion).where(FileVersion.version == 1)
            )
            ver = result.scalar_one()
            ver.content_hash = "0000000000000000000000000000000000000000000000000000000000000000"
            await session.commit()

            with pytest.raises(ConsistencyError, match="hash mismatch"):
                await fs.get_version_content("/f.py", 1)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Path Validation on exists / get_info
# ---------------------------------------------------------------------------


class TestPathValidationExistsGetInfo:
    async def test_exists_null_byte_path(self):
        fs, engine = await _make_fs()
        async with fs:
            assert await fs.exists("/foo\x00bar") is False
        await engine.dispose()

    async def test_get_info_null_byte_path(self):
        fs, engine = await _make_fs()
        async with fs:
            info = await fs.get_info("/foo\x00bar")
            assert info is None
        await engine.dispose()


# ---------------------------------------------------------------------------
# Control Characters
# ---------------------------------------------------------------------------


class TestControlChars:
    async def test_write_control_char_path_rejected(self):
        fs, engine = await _make_fs()
        async with fs:
            result = await fs.write("/bad\x01file.py", "content\n")
            assert result.success is False
            assert "control character" in result.message.lower()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_read_empty_file(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/empty.py", "")
            result = await fs.read("/empty.py")
            assert result.success is True
            assert result.content == ""
        await engine.dispose()

    async def test_write_overwrite_false(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/f.py", "v1\n")
            result = await fs.write("/f.py", "v2\n", overwrite=False)
            assert result.success is False
            assert "already exists" in result.message
        await engine.dispose()

    async def test_move_directory_with_children(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/a")
            await fs.write("/a/child.py", "child\n")

            result = await fs.move("/a", "/b")
            assert result.success is True

            # Children should follow
            assert await fs.exists("/b/child.py") is True
            assert await fs.exists("/a/child.py") is False
        await engine.dispose()

    async def test_copy_directory_rejected(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/src")
            result = await fs.copy("/src", "/dst")
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_read_directory_rejected(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/mydir")
            result = await fs.read("/mydir")
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()

    async def test_edit_directory_rejected(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/mydir")
            result = await fs.edit("/mydir", "old", "new")
            assert result.success is False
            assert "directory" in result.message.lower()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Version Reconstruction Across Snapshots
# ---------------------------------------------------------------------------


class TestParentPath:
    """H5: parent_path should be populated on write, mkdir, and move."""

    async def test_write_sets_parent_path(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/src/main.py", "content\n")

            info = await fs.get_info("/src/main.py")
            assert info is not None

            # Check parent_path via DB directly
            session = await fs._get_session()
            file = await fs._get_file(session, "/src/main.py")
            assert file.parent_path == "/src"
        await engine.dispose()

    async def test_mkdir_sets_parent_path(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.mkdir("/a/b/c")

            session = await fs._get_session()
            b = await fs._get_file(session, "/a/b")
            assert b is not None
            assert b.parent_path == "/a"

            c = await fs._get_file(session, "/a/b/c")
            assert c is not None
            assert c.parent_path == "/a/b"
        await engine.dispose()

    async def test_move_updates_parent_path(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/src/file.py", "content\n")
            await fs.mkdir("/dst")
            await fs.move("/src/file.py", "/dst/file.py")

            session = await fs._get_session()
            file = await fs._get_file(session, "/dst/file.py")
            assert file is not None
            assert file.parent_path == "/dst"
        await engine.dispose()

    async def test_root_file_parent_path(self):
        fs, engine = await _make_fs()
        async with fs:
            await fs.write("/root_file.py", "content\n")

            session = await fs._get_session()
            file = await fs._get_file(session, "/root_file.py")
            assert file.parent_path == "/"
        await engine.dispose()


# ---------------------------------------------------------------------------
# Version Reconstruction Across Snapshots
# ---------------------------------------------------------------------------


class TestVersionReconstructionAcrossSnapshots:
    async def test_25_edits_spanning_2_snapshot_intervals(self):
        """25+ edits spanning 2 snapshot intervals, verify all versions."""
        fs, engine = await _make_fs()
        async with fs:
            initial = "version_0\nstatic_line\n"
            await fs.write("/f.py", initial)
            contents = [initial]

            for i in range(1, 26):
                old_marker = f"version_{i - 1}"
                new_marker = f"version_{i}"
                prev = contents[-1]
                new = prev.replace(old_marker, new_marker, 1)
                await fs.edit("/f.py", old_marker, new_marker)
                contents.append(new)

            # Verify every version is reconstructable
            for version_num in range(1, len(contents) + 1):
                vc = await fs.get_version_content("/f.py", version_num)
                assert vc is not None, f"Version {version_num} returned None"
                assert vc == contents[version_num - 1], (
                    f"Version {version_num} mismatch"
                )
        await engine.dispose()

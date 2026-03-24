"""Tests for versioning, CRUD, trash via DatabaseFileSystem."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem


async def _make_fs():
    """Create a stateless DatabaseFileSystem + session factory + engine."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem()
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
            assert "Created" in result.message

            read = await fs.read("/hello.py", session=session)
            assert read.success is True
            assert "print('hi')" in read.file.content
        await engine.dispose()

    async def test_write_updates_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            result = await fs.write("/f.py", "v2\n", session=session)
            assert result.success is True
            assert "Updated" in result.message
            assert result.file.current_version == 2
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
            assert "earth" in read.file.content
        await engine.dispose()

    async def test_edit_increments_version(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "line1\n", session=session)
            result = await fs.edit("/f.py", "line1", "line2", session=session)
            assert result.file.current_version == 2
        await engine.dispose()

    async def test_edit_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.edit("/nope.py", "a", "b", session=session)
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
            assert "trash" in result.message

            read = await fs.read("/f.py", session=session)
            assert read.success is False
        await engine.dispose()

    async def test_permanent_delete(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "content\n", session=session)
            result = await fs.delete("/f.py", permanent=True, session=session)
            assert result.success is True
            assert "Permanently" in result.message
        await engine.dispose()

    async def test_delete_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.delete("/nope.py", session=session)
            assert result.success is False
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
            assert "Created" in result.message
        await engine.dispose()

    async def test_mkdir_parents(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.mkdir("/a/b/c", session=session)
            assert result.success is True
            assert "Created" in result.message
        await engine.dispose()

    async def test_mkdir_existing(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/src", session=session)
            result = await fs.mkdir("/src", session=session)
            assert result.success is True
            assert "already exists" in result.message
        await engine.dispose()

    async def test_list_dir(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/hello.py", "content\n", session=session)
            await fs.mkdir("/src", session=session)

            result = await fs.list_dir("/", session=session)
            assert result.success is True
            names = [p.rsplit("/", 1)[-1] for p in result.paths]
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
            root_names = [p.rsplit("/", 1)[-1] for p in root_result.paths]
            assert "src" in root_names
            assert "readme.md" in root_names
            assert "main.py" not in root_names
            assert "helper.py" not in root_names

            src_result = await fs.list_dir("/src", session=session)
            src_names = [p.rsplit("/", 1)[-1] for p in src_result.paths]
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
            names = [p.rsplit("/", 1)[-1] for p in result.paths]
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

            assert (await fs.exists("/a.py", session=session)).message == "not found"
            assert (await fs.exists("/b.py", session=session)).message == "exists"
        await engine.dispose()

    async def test_move_empty_file_to_existing(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/empty.py", "", session=session)
            await fs.write("/target.py", "old\n", session=session)

            result = await fs.move("/empty.py", "/target.py", session=session)
            assert result.success is True
            assert (await fs.exists("/empty.py", session=session)).message == "not found"
            assert (await fs.exists("/target.py", session=session)).message == "exists"
        await engine.dispose()

    async def test_copy_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "content\n", session=session)
            result = await fs.copy("/a.py", "/b.py", session=session)
            assert result.success is True

            assert (await fs.exists("/a.py", session=session)).message == "exists"
            assert (await fs.exists("/b.py", session=session)).message == "exists"
        await engine.dispose()

    async def test_move_to_existing_file_overwrites(self):
        """C3: Moving to an existing file overwrites the destination."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src.py", "source content\n", session=session)
            await fs.write("/dest.py", "old dest content\n", session=session)

            result = await fs.move("/src.py", "/dest.py", session=session)
            assert result.success is True

            assert (await fs.exists("/src.py", session=session)).message == "not found"

            read = await fs.read("/dest.py", session=session)
            assert read.success is True
            assert "source content" in read.file.content
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
            assert "file one" in read1.file.content

            read2 = await fs.read("/dst/two.py", session=session)
            assert read2.success is True
            assert "file two" in read2.file.content
        await engine.dispose()


# ---------------------------------------------------------------------------
# Exists
# ---------------------------------------------------------------------------


class TestExists:
    async def test_exists(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            assert (await fs.exists("/nope.py", session=session)).message == "not found"
            await fs.write("/f.py", "x\n", session=session)
            assert (await fs.exists("/f.py", session=session)).message == "exists"
        await engine.dispose()

    async def test_exists_file(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "hello\n", session=session)
            info = await fs.exists("/f.py", session=session)
            assert info.success
            assert info.message == "exists"
        await engine.dispose()

    async def test_exists_nonexistent(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            info = await fs.exists("/nope.py", session=session)
            assert not info.success
        await engine.dispose()


# ---------------------------------------------------------------------------
# Path Validation on exists
# ---------------------------------------------------------------------------


class TestPathValidationExists:
    async def test_exists_null_byte_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            info = await fs.exists("/foo\x00bar", session=session)
            assert not info.success
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
            assert result.file.content == ""
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

            assert (await fs.exists("/b/child.py", session=session)).message == "exists"
            assert (await fs.exists("/a/child.py", session=session)).message == "not found"
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


class TestDirectoryFieldsComplete:
    """Directory upserts include all NOT NULL columns."""

    async def test_ensure_parent_dirs_populates_all_fields(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a/b/c/file.py", "x\n", session=session)

            for dir_path in ["/a", "/a/b", "/a/b/c"]:
                rec = await fs._get_file_record(session, dir_path)
                assert rec is not None, f"Missing dir: {dir_path}"
                assert rec.is_directory is True
                assert rec.mime_type is not None  # populated, not missing
                assert rec.lines is not None
                assert rec.size_bytes is not None
                assert rec.tokens is not None
                assert rec.created_at is not None
                assert rec.updated_at is not None
        await engine.dispose()

    async def test_mkdir_populates_all_fields(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/deep/nested/dir", session=session)

            for dir_path in ["/deep", "/deep/nested", "/deep/nested/dir"]:
                rec = await fs._get_file_record(session, dir_path)
                assert rec is not None, f"Missing dir: {dir_path}"
                assert rec.is_directory is True
                assert rec.mime_type is not None
                assert rec.lines is not None
                assert rec.size_bytes is not None
                assert rec.created_at is not None
                assert rec.updated_at is not None
        await engine.dispose()


class TestParentPath:
    """H5: parent_path should be populated on write, mkdir, and move."""

    async def test_write_sets_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src/main.py", "content\n", session=session)

            info = await fs.exists("/src/main.py", session=session)
            assert info.success

            file = await fs._get_file_record(session, "/src/main.py")
            assert file.parent_path == "/src"
        await engine.dispose()

    async def test_mkdir_sets_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/a/b/c", session=session)

            b = await fs._get_file_record(session, "/a/b")
            assert b is not None
            assert b.parent_path == "/a"

            c = await fs._get_file_record(session, "/a/b/c")
            assert c is not None
            assert c.parent_path == "/a/b"
        await engine.dispose()

    async def test_move_updates_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/src/file.py", "content\n", session=session)
            await fs.mkdir("/dst", session=session)
            await fs.move("/src/file.py", "/dst/file.py", session=session)

            file = await fs._get_file_record(session, "/dst/file.py")
            assert file is not None
            assert file.parent_path == "/dst"
        await engine.dispose()

    async def test_root_file_parent_path(self):
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/root_file.py", "content\n", session=session)

            file = await fs._get_file_record(session, "/root_file.py")
            assert file.parent_path == "/"
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
            assert read.file.content == "content\n"
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
            assert (await fs.exists("/a/c.py", session=session)).message == "exists"
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
            assert read.file.content == "source\n"
            assert (await fs.exists("/src.py", session=session)).message == "not found"
        await engine.dispose()

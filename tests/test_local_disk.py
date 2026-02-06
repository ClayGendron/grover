"""Tests for LocalDiskBackend — direct disk operations."""

from __future__ import annotations

import pytest

from grover.fs.local_disk import LocalDiskBackend


@pytest.fixture
def disk(tmp_path) -> LocalDiskBackend:
    """LocalDiskBackend rooted at a temporary directory."""
    return LocalDiskBackend(host_dir=tmp_path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_nonexistent_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalDiskBackend(host_dir=tmp_path / "nope")

    def test_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(NotADirectoryError):
            LocalDiskBackend(host_dir=f)


# ---------------------------------------------------------------------------
# Write / Read
# ---------------------------------------------------------------------------


class TestWriteRead:
    async def test_write_and_read(self, disk):
        result = await disk.write("/hello.py", "print('hi')\n")
        assert result.success is True
        assert result.created is True

        read = await disk.read("/hello.py")
        assert read.success is True
        assert "print('hi')" in read.content

    async def test_overwrite(self, disk):
        await disk.write("/f.py", "v1\n")
        result = await disk.write("/f.py", "v2\n")
        assert result.success is True
        assert result.created is False

    async def test_write_creates_parents(self, disk):
        result = await disk.write("/a/b/c.py", "code\n")
        assert result.success is True

    async def test_write_non_text_rejected(self, disk):
        result = await disk.write("/image.png", "data")
        assert result.success is False
        assert "non-text" in result.message.lower()

    async def test_read_nonexistent(self, disk):
        result = await disk.read("/nope.py")
        assert result.success is False

    async def test_read_directory(self, disk, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = await disk.read("/subdir")
        assert result.success is False

    async def test_read_empty_file(self, disk, tmp_path):
        (tmp_path / "empty.py").write_text("")
        result = await disk.read("/empty.py")
        assert result.success is True
        assert result.content == ""


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


class TestEdit:
    async def test_edit_exact(self, disk):
        await disk.write("/f.py", "hello world\n")
        result = await disk.edit("/f.py", "world", "earth")
        assert result.success is True

        read = await disk.read("/f.py")
        assert "earth" in read.content

    async def test_edit_nonexistent(self, disk):
        result = await disk.edit("/nope.py", "a", "b")
        assert result.success is False

    async def test_edit_directory(self, disk, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = await disk.edit("/subdir", "a", "b")
        assert result.success is False


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_file(self, disk, tmp_path):
        await disk.write("/f.py", "content\n")
        result = await disk.delete("/f.py")
        assert result.success is True
        assert result.permanent is True
        assert not (tmp_path / "f.py").exists()

    async def test_delete_directory(self, disk, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file.py").write_text("x\n")
        result = await disk.delete("/subdir")
        assert result.success is True
        assert not (tmp_path / "subdir").exists()

    async def test_delete_nonexistent(self, disk):
        result = await disk.delete("/nope.py")
        assert result.success is False


# ---------------------------------------------------------------------------
# Mkdir
# ---------------------------------------------------------------------------


class TestMkdir:
    async def test_mkdir(self, disk, tmp_path):
        result = await disk.mkdir("/new_dir")
        assert result.success is True
        assert (tmp_path / "new_dir").is_dir()

    async def test_mkdir_parents(self, disk, tmp_path):
        result = await disk.mkdir("/a/b/c")
        assert result.success is True
        assert (tmp_path / "a" / "b" / "c").is_dir()

    async def test_mkdir_exists(self, disk, tmp_path):
        (tmp_path / "existing").mkdir()
        result = await disk.mkdir("/existing")
        assert result.success is True
        assert result.created_dirs == []

    async def test_mkdir_file_conflict(self, disk):
        await disk.write("/f.py", "x\n")
        result = await disk.mkdir("/f.py")
        assert result.success is False


# ---------------------------------------------------------------------------
# List Directory
# ---------------------------------------------------------------------------


class TestListDir:
    async def test_list_root(self, disk, tmp_path):
        (tmp_path / "a.py").write_text("x\n")
        (tmp_path / "subdir").mkdir()
        result = await disk.list_dir("/")
        assert result.success is True
        assert len(result.entries) >= 2

    async def test_list_nonexistent(self, disk):
        result = await disk.list_dir("/nope")
        assert result.success is False

    async def test_list_file(self, disk):
        await disk.write("/f.py", "x\n")
        result = await disk.list_dir("/f.py")
        assert result.success is False


# ---------------------------------------------------------------------------
# Move / Copy
# ---------------------------------------------------------------------------


class TestMove:
    async def test_move_file(self, disk, tmp_path):
        await disk.write("/a.py", "content\n")
        result = await disk.move("/a.py", "/b.py")
        assert result.success is True
        assert not (tmp_path / "a.py").exists()
        assert (tmp_path / "b.py").exists()

    async def test_move_nonexistent(self, disk):
        result = await disk.move("/nope.py", "/dest.py")
        assert result.success is False


class TestCopy:
    async def test_copy_file(self, disk, tmp_path):
        await disk.write("/a.py", "content\n")
        result = await disk.copy("/a.py", "/b.py")
        assert result.success is True
        assert (tmp_path / "a.py").exists()
        assert (tmp_path / "b.py").exists()

    async def test_copy_nonexistent(self, disk):
        result = await disk.copy("/nope.py", "/dest.py")
        assert result.success is False


# ---------------------------------------------------------------------------
# Exists / GetInfo
# ---------------------------------------------------------------------------


class TestExistsAndGetInfo:
    async def test_exists(self, disk):
        await disk.write("/f.py", "x\n")
        assert await disk.exists("/f.py") is True
        assert await disk.exists("/nope.py") is False

    async def test_get_info(self, disk):
        await disk.write("/f.py", "hello\n")
        info = await disk.get_info("/f.py")
        assert info is not None
        assert info.name == "f.py"
        assert info.is_directory is False

    async def test_get_info_dir(self, disk, tmp_path):
        (tmp_path / "mydir").mkdir()
        info = await disk.get_info("/mydir")
        assert info is not None
        assert info.is_directory is True

    async def test_get_info_nonexistent(self, disk):
        info = await disk.get_info("/nope.py")
        assert info is None


# ---------------------------------------------------------------------------
# Path Traversal Security
# ---------------------------------------------------------------------------


class TestPathSecurity:
    async def test_traversal_rejected(self, disk):
        result = await disk.read("/../../../etc/passwd")
        assert result.success is False

    async def test_symlink_rejected(self, disk, tmp_path):
        target = tmp_path / "real.py"
        target.write_text("content\n")
        link = tmp_path / "link.py"
        link.symlink_to(target)

        with pytest.raises(PermissionError, match=r"[Ss]ymlink"):
            disk._resolve_path("/link.py")


# ---------------------------------------------------------------------------
# VFS Stubs
# ---------------------------------------------------------------------------


class TestVFSStubs:
    async def test_list_versions_empty(self, disk):
        assert await disk.list_versions("/anything") == []

    async def test_restore_version_unsupported(self, disk):
        result = await disk.restore_version("/x.py", 1)
        assert result.success is False

    async def test_get_version_content_none(self, disk):
        assert await disk.get_version_content("/x.py", 1) is None

    async def test_list_trash_empty(self, disk):
        result = await disk.list_trash()
        assert result.success is True
        assert result.entries == []

    async def test_restore_from_trash_unsupported(self, disk):
        result = await disk.restore_from_trash("/x.py")
        assert result.success is False

    async def test_empty_trash_noop(self, disk):
        result = await disk.empty_trash()
        assert result.success is True

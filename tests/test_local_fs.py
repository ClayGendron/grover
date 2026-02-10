"""Tests for LocalFileSystem-specific behavior — disk I/O, path security, binary detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from grover.fs.local_fs import LocalFileSystem

if TYPE_CHECKING:
    pass


async def _make_local_fs(tmp_path: Path) -> LocalFileSystem:
    """Create a LocalFileSystem with isolated workspace and data dirs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    fs = LocalFileSystem(workspace_dir=workspace, data_dir=data)
    return fs


# ---------------------------------------------------------------------------
# Path Security
# ---------------------------------------------------------------------------


class TestPathSecurity:
    async def test_symlink_rejected(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create a real file outside workspace
        target = tmp_path / "secret.txt"
        target.write_text("secret")

        # Create symlink inside workspace
        link = workspace / "link.txt"
        link.symlink_to(target)

        result = await fs.read("/link.txt")
        assert result.success is False
        assert "ymlink" in result.message or "not found" in result.message.lower()
        await fs.close()

    async def test_path_traversal_rejected(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)

        result = await fs.read("/../../etc/passwd")
        assert result.success is False
        await fs.close()

    async def test_dotdot_normalized(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)

        # Write a file, then read via a path with ..
        await fs.write("/bar.py", "content\n")
        result = await fs.read("/foo/../bar.py")
        assert result.success is True
        assert "content" in result.content
        await fs.close()


# ---------------------------------------------------------------------------
# Binary File Handling
# ---------------------------------------------------------------------------


class TestBinaryFileHandling:
    async def test_read_binary_file_rejected(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Write a PNG-like binary file directly to disk
        png_file = workspace / "image.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = await fs.read("/image.png")
        assert result.success is False
        assert "binary" in result.message.lower()
        await fs.close()


# ---------------------------------------------------------------------------
# Disk Sync
# ---------------------------------------------------------------------------


class TestDiskSync:
    async def test_list_dir_includes_disk_only_files(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create file directly on disk (no FS write)
        (workspace / "disk_only.py").write_text("# disk only\n")

        result = await fs.list_dir("/")
        names = [e.name for e in result.entries]
        assert "disk_only.py" in names
        await fs.close()

    async def test_list_dir_hides_dotfiles(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        (workspace / ".gitignore").write_text("*.pyc\n")
        (workspace / "visible.py").write_text("# visible\n")

        result = await fs.list_dir("/")
        names = [e.name for e in result.entries]
        assert ".gitignore" not in names
        assert "visible.py" in names
        await fs.close()


# ---------------------------------------------------------------------------
# Delete Backup
# ---------------------------------------------------------------------------


class TestDeleteBackup:
    async def test_delete_backs_up_disk_only_file(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create file directly on disk
        (workspace / "ephemeral.py").write_text("content\n")

        # Delete via FS — should create DB record first, then soft-delete
        result = await fs.delete("/ephemeral.py")
        assert result.success is True

        # File should be in trash (backed up to DB)
        trash = await fs.list_trash()
        paths = [e.path for e in trash.entries]
        assert "/ephemeral.py" in paths
        await fs.close()


# ---------------------------------------------------------------------------
# Atomic Writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    async def test_write_creates_file_on_disk(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        await fs.write("/hello.py", "print('hello')\n")

        disk_file = workspace / "hello.py"
        assert disk_file.exists()
        assert disk_file.read_text() == "print('hello')\n"
        await fs.close()

    async def test_write_content_atomic(self, tmp_path: Path):
        fs = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        await fs.write("/atomic.py", "content\n")

        # Verify no .tmp_ files left behind
        tmp_files = list(workspace.glob(".tmp_*"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"
        await fs.close()

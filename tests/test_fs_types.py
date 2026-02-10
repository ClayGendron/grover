"""Tests for FS result dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.fs.types import (
    DeleteResult,
    EditResult,
    FileInfo,
    ListResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    VersionInfo,
    WriteResult,
)


class TestFileInfo:
    def test_required_fields(self):
        info = FileInfo(path="/hello.txt", name="hello.txt", is_directory=False)
        assert info.path == "/hello.txt"
        assert info.name == "hello.txt"
        assert info.is_directory is False

    def test_defaults(self):
        info = FileInfo(path="/x", name="x", is_directory=True)
        assert info.size_bytes is None
        assert info.mime_type is None
        assert info.version == 1
        assert info.created_at is None
        assert info.updated_at is None
        assert info.permission is None
        assert info.mount_type is None

    def test_all_fields(self):
        now = datetime.now(UTC)
        info = FileInfo(
            path="/src",
            name="src",
            is_directory=True,
            size_bytes=4096,
            mime_type="inode/directory",
            version=3,
            created_at=now,
            updated_at=now,
            permission="read_write",
            mount_type="vfs",
        )
        assert info.size_bytes == 4096
        assert info.mount_type == "vfs"


class TestVersionInfo:
    def test_required_fields(self):
        now = datetime.now(UTC)
        vi = VersionInfo(version=1, content_hash="abc", size_bytes=10, created_at=now)
        assert vi.version == 1
        assert vi.content_hash == "abc"
        assert vi.size_bytes == 10
        assert vi.created_at == now

    def test_defaults(self):
        now = datetime.now(UTC)
        vi = VersionInfo(version=1, content_hash="abc", size_bytes=10, created_at=now)
        assert vi.created_by is None


class TestReadResult:
    def test_success(self):
        r = ReadResult(
            success=True,
            message="Read 10 lines",
            content="hello\nworld",
            file_path="/test.txt",
            total_lines=2,
            lines_read=2,
        )
        assert r.success is True
        assert r.content == "hello\nworld"
        assert r.truncated is False

    def test_failure(self):
        r = ReadResult(success=False, message="File not found")
        assert r.success is False
        assert r.content is None
        assert r.file_path is None


class TestWriteResult:
    def test_created(self):
        r = WriteResult(success=True, message="Created", file_path="/new.py", created=True)
        assert r.created is True
        assert r.version == 1

    def test_updated(self):
        r = WriteResult(success=True, message="Updated", file_path="/old.py", version=5)
        assert r.created is False
        assert r.version == 5


class TestEditResult:
    def test_success(self):
        r = EditResult(success=True, message="Applied", file_path="/x.py", version=3)
        assert r.success is True
        assert r.version == 3

    def test_defaults(self):
        r = EditResult(success=False, message="err")
        assert r.file_path is None
        assert r.version == 1


class TestDeleteResult:
    def test_soft_delete(self):
        r = DeleteResult(success=True, message="Trashed", file_path="/x.py")
        assert r.permanent is False
        assert r.total_deleted is None

    def test_permanent(self):
        r = DeleteResult(
            success=True, message="Deleted", permanent=True, total_deleted=3
        )
        assert r.permanent is True
        assert r.total_deleted == 3


class TestMkdirResult:
    def test_created(self):
        r = MkdirResult(
            success=True, message="Created", path="/a/b", created_dirs=["/a", "/a/b"]
        )
        assert r.path == "/a/b"
        assert len(r.created_dirs) == 2

    def test_defaults(self):
        r = MkdirResult(success=True, message="ok")
        assert r.path is None
        assert r.created_dirs == []


class TestListResult:
    def test_with_entries(self):
        entries = [
            FileInfo(path="/a.py", name="a.py", is_directory=False),
            FileInfo(path="/src", name="src", is_directory=True),
        ]
        r = ListResult(success=True, message="Listed 2 items", entries=entries)
        assert len(r.entries) == 2
        assert r.path == "/"

    def test_defaults(self):
        r = ListResult(success=True, message="ok")
        assert r.entries == []


class TestMoveResult:
    def test_success(self):
        r = MoveResult(
            success=True, message="Moved", old_path="/a.py", new_path="/b.py"
        )
        assert r.old_path == "/a.py"
        assert r.new_path == "/b.py"

    def test_defaults(self):
        r = MoveResult(success=False, message="err")
        assert r.old_path is None
        assert r.new_path is None


class TestRestoreResult:
    def test_success(self):
        r = RestoreResult(
            success=True,
            message="Restored",
            file_path="/x.py",
            restored_version=2,
            current_version=5,
        )
        assert r.restored_version == 2
        assert r.current_version == 5

    def test_defaults(self):
        r = RestoreResult(success=False, message="err")
        assert r.file_path is None
        assert r.restored_version == 0
        assert r.current_version == 0

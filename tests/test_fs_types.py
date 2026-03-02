"""Tests for FS result dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.types import (
    DeleteResult,
    EditResult,
    FileInfoResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    VersionEvidence,
    WriteResult,
)


class TestFileInfoResult:
    def test_required_fields(self):
        info = FileInfoResult(path="/hello.txt", is_directory=False)
        assert info.path == "/hello.txt"
        assert info.is_directory is False

    def test_defaults(self):
        info = FileInfoResult(path="/x", is_directory=True)
        assert info.size_bytes == 0
        assert info.mime_type == "text/plain"
        assert info.version == 0
        assert info.created_at is None
        assert info.updated_at is None
        assert info.permission is None
        assert info.mount_type is None

    def test_all_fields(self):
        now = datetime.now(UTC)
        info = FileInfoResult(
            path="/src",
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


class TestVersionEvidence:
    def test_required_fields(self):
        now = datetime.now(UTC)
        ve = VersionEvidence(
            strategy="version",
            path="/test.txt",
            version=1,
            content_hash="abc",
            size_bytes=10,
            created_at=now,
        )
        assert ve.version == 1
        assert ve.content_hash == "abc"
        assert ve.size_bytes == 10
        assert ve.created_at == now

    def test_defaults(self):
        ve = VersionEvidence(strategy="version", path="/test.txt")
        assert ve.created_by is None
        assert ve.version == 0
        assert ve.content_hash == ""
        assert ve.size_bytes == 0


class TestReadResult:
    def test_success(self):
        r = ReadResult(
            success=True,
            message="Read 10 lines",
            content="hello\nworld",
            path="/test.txt",
            total_lines=2,
            lines_read=2,
        )
        assert r.success is True
        assert r.content == "hello\nworld"
        assert r.truncated is False

    def test_failure(self):
        r = ReadResult(success=False, message="File not found")
        assert r.success is False
        assert r.content == ""
        assert r.path == ""


class TestWriteResult:
    def test_created(self):
        r = WriteResult(success=True, message="Created", path="/new.py", created=True)
        assert r.created is True
        assert r.version == 0

    def test_updated(self):
        r = WriteResult(success=True, message="Updated", path="/old.py", version=5)
        assert r.created is False
        assert r.version == 5


class TestEditResult:
    def test_success(self):
        r = EditResult(success=True, message="Applied", path="/x.py", version=3)
        assert r.success is True
        assert r.version == 3

    def test_defaults(self):
        r = EditResult(success=False, message="err")
        assert r.path == ""
        assert r.version == 0


class TestDeleteResult:
    def test_soft_delete(self):
        r = DeleteResult(success=True, message="Trashed", path="/x.py")
        assert r.permanent is False
        assert r.total_deleted is None

    def test_permanent(self):
        r = DeleteResult(success=True, message="Deleted", permanent=True, total_deleted=3)
        assert r.permanent is True
        assert r.total_deleted == 3


class TestMkdirResult:
    def test_created(self):
        r = MkdirResult(success=True, message="Created", path="/a/b", created_dirs=["/a", "/a/b"])
        assert r.path == "/a/b"
        assert len(r.created_dirs) == 2

    def test_defaults(self):
        r = MkdirResult(success=True, message="ok")
        assert r.path == ""
        assert r.created_dirs == []


class TestMoveResult:
    def test_success(self):
        r = MoveResult(success=True, message="Moved", old_path="/a.py", new_path="/b.py")
        assert r.old_path == "/a.py"
        assert r.new_path == "/b.py"

    def test_defaults(self):
        r = MoveResult(success=False, message="err")
        assert r.old_path == ""
        assert r.new_path == ""


class TestRestoreResult:
    def test_success(self):
        r = RestoreResult(
            success=True,
            message="Restored",
            path="/x.py",
            restored_version=2,
        )
        assert r.restored_version == 2

    def test_defaults(self):
        r = RestoreResult(success=False, message="err")
        assert r.path == ""
        assert r.restored_version == 0

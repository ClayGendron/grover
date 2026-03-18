"""Tests for internal result types and File model fields."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.models.internal.evidence import VersionEvidence
from grover.models.internal.ref import Directory, File
from grover.models.internal.results import FileOperationResult


class TestFileFields:
    """Test File model with size_bytes, mime_type, and other fields."""

    def test_required_fields(self):
        f = File(path="/hello.txt")
        assert f.path == "/hello.txt"

    def test_defaults(self):
        f = File(path="/x.py")
        assert f.size_bytes == 0
        assert f.mime_type == ""
        assert f.current_version == 0
        assert f.created_at is None
        assert f.updated_at is None

    def test_directory_type(self):
        d = Directory(path="/src")
        assert isinstance(d, Directory)
        assert d.path == "/src"

    def test_all_fields(self):
        now = datetime.now(UTC)
        f = File(
            path="/src.py",
            size_bytes=4096,
            mime_type="text/x-python",
            current_version=3,
            created_at=now,
            updated_at=now,
        )
        assert f.size_bytes == 4096
        assert f.mime_type == "text/x-python"
        assert f.current_version == 3


class TestVersionEvidence:
    def test_required_fields(self):
        now = datetime.now(UTC)
        ve = VersionEvidence(
            operation="version",
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
        ve = VersionEvidence(operation="version")
        assert ve.created_by is None
        assert ve.version == 0
        assert ve.content_hash == ""
        assert ve.size_bytes == 0


class TestFileOperationResult:
    """Test the unified FileOperationResult type."""

    def test_success(self):
        r = FileOperationResult(
            success=True,
            message="Read 10 lines",
            file=File(path="/test.txt", content="hello\nworld"),
        )
        assert r.success is True
        assert r.file.content == "hello\nworld"

    def test_failure(self):
        r = FileOperationResult(success=False, message="File not found")
        assert r.success is False
        assert r.file.path == ""

    def test_defaults(self):
        r = FileOperationResult()
        assert r.success is True
        assert r.message == ""
        assert r.file.path == ""

    def test_with_file_metadata(self):
        r = FileOperationResult(
            success=True,
            message="Created",
            file=File(path="/new.py", current_version=1),
        )
        assert r.file.path == "/new.py"
        assert r.file.current_version == 1

    def test_file_is_mutable(self):
        """FileOperationResult file field can be replaced."""
        r = FileOperationResult(file=File(path="/original"))
        r.file = File(path="/prefixed/original")
        assert r.file.path == "/prefixed/original"

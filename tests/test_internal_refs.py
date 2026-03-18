"""Tests for internal Ref types (File, FileChunk, FileVersion, FileConnection)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from grover.models.internal.ref import Directory, File, FileChunk, FileConnection, FileVersion, Ref


class TestRef:
    def test_basic_construction(self):
        r = Ref(path="/hello.py")
        assert r.path == "/hello.py"

    def test_serialization_round_trip(self):
        r = Ref(path="/a/b.py")
        data = dataclasses.asdict(r)
        assert data == {"path": "/a/b.py"}
        r2 = Ref(**data)
        assert r2.path == "/a/b.py"


class TestFile:
    def test_defaults(self):
        f = File(path="/hello.py")
        assert f.path == "/hello.py"
        assert isinstance(f, File)
        assert f.content is None
        assert f.embedding is None
        assert f.tokens == 0
        assert f.lines == 0
        assert f.current_version == 0
        assert f.chunks == []
        assert f.versions == []
        assert f.evidence == []
        assert f.created_at is None
        assert f.updated_at is None

    def test_with_content(self):
        f = File(path="/hello.py", content="print('hi')", lines=1, tokens=5)
        assert f.content == "print('hi')"
        assert f.lines == 1
        assert f.tokens == 5

    def test_with_embedding(self):
        emb = [0.1, 0.2, 0.3]
        f = File(path="/hello.py", embedding=emb)
        assert f.embedding == [0.1, 0.2, 0.3]

    def test_with_chunks(self):
        chunk = FileChunk(path="/hello.py#main", name="main", content="def main(): pass")
        f = File(path="/hello.py", chunks=[chunk])
        assert len(f.chunks) == 1
        assert f.chunks[0].name == "main"

    def test_with_versions(self):
        v = FileVersion(path="/hello.py@1", number=1)
        f = File(path="/hello.py", versions=[v])
        assert len(f.versions) == 1
        assert f.versions[0].number == 1

    def test_directory(self):
        d = Directory(path="/src")
        assert isinstance(d, Directory)
        assert d.path == "/src"

    def test_serialization(self):
        f = File(path="/a.py", content="x = 1", lines=1)
        data = dataclasses.asdict(f)
        assert data["path"] == "/a.py"
        assert data["content"] == "x = 1"
        # Round-trip reconstruction (without nested types for simplicity)
        f2 = File(path=data["path"], content=data["content"])
        assert f2.path == "/a.py"
        assert f2.content == "x = 1"

    def test_timestamps(self):
        now = datetime.now(UTC)
        f = File(path="/a.py", created_at=now, updated_at=now)
        assert f.created_at == now
        assert f.updated_at == now


class TestFileChunk:
    def test_defaults(self):
        c = FileChunk(path="/a.py#func")
        assert c.path == "/a.py#func"
        assert c.name == ""
        assert c.content == ""
        assert c.embedding is None
        assert c.tokens == 0
        assert c.line_start == 0
        assert c.line_end == 0

    def test_with_content(self):
        c = FileChunk(
            path="/a.py#login",
            name="login",
            content="def login(): pass",
            line_start=10,
            line_end=15,
            tokens=4,
        )
        assert c.name == "login"
        assert c.content == "def login(): pass"
        assert c.line_start == 10
        assert c.line_end == 15

    def test_inherits_ref(self):
        c = FileChunk(path="/a.py#login", name="login")
        assert isinstance(c, Ref)


class TestFileVersion:
    def test_defaults(self):
        v = FileVersion(path="/a.py@1", number=1)
        assert v.path == "/a.py@1"
        assert v.number == 1
        assert v.embedding is None
        assert v.created_at is None

    def test_with_timestamp(self):
        now = datetime.now(UTC)
        v = FileVersion(path="/a.py@3", number=3, created_at=now)
        assert v.created_at == now

    def test_inherits_ref(self):
        v = FileVersion(path="/a.py@1", number=1)
        assert isinstance(v, Ref)


class TestFileConnection:
    def test_construction(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        assert conn.source_path == "/a.py"
        assert conn.target_path == "/b.py"
        assert conn.type == "imports"
        assert conn.weight == 1.0
        assert conn.distance == 1.0

    def test_with_weight(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            weight=0.5,
            distance=2.0,
        )
        assert conn.weight == 0.5
        assert conn.distance == 2.0

    def test_serialization(self):
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        data = dataclasses.asdict(conn)
        assert data["source_path"] == "/a.py"
        assert data["target_path"] == "/b.py"
        # Reconstruct from dict
        conn2 = FileConnection(
            path=data["path"],
            source_path=data["source_path"],
            target_path=data["target_path"],
            type=data["type"],
        )
        assert conn2.source_path == "/a.py"

    def test_not_ref_subclass(self):
        """FileConnection is not a Ref — it's a standalone dataclass."""
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
        )
        assert not isinstance(conn, Ref)

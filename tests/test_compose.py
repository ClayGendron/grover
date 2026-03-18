"""Tests for DB model ↔ internal type composition helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from grover.models.database.chunk import FileChunkModel
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.database.vector import Vector
from grover.models.database.version import FileVersionModel
from grover.models.internal.compose import (
    chunk_to_model,
    connection_to_model,
    file_to_model,
    model_to_chunk,
    model_to_connection,
    model_to_file,
    model_to_version,
)
from grover.models.internal.ref import File, FileChunk, FileConnection


class TestModelToFile:
    def test_basic(self):
        m = FileModel(path="/a.py", is_directory=False, content="x = 1", lines=1)
        f = model_to_file(m)
        assert f.path == "/a.py"
        assert isinstance(f, File)
        assert f.content == "x = 1"
        assert f.lines == 1
        assert f.embedding is None
        assert f.chunks == []
        assert f.versions == []

    def test_with_vector(self):
        m = FileModel(path="/a.py", embedding=Vector([0.1, 0.2, 0.3]))
        f = model_to_file(m)
        assert f.embedding == [0.1, 0.2, 0.3]

    def test_none_vector(self):
        m = FileModel(path="/a.py", embedding=None)
        f = model_to_file(m)
        assert f.embedding is None

    def test_with_chunks(self):
        m = FileModel(path="/a.py")
        chunk = FileChunkModel(path="/a.py#login", file_path="/a.py", content="def login(): pass")
        f = model_to_file(m, chunks=[chunk])
        assert len(f.chunks) == 1
        assert f.chunks[0].path == "/a.py#login"
        assert f.chunks[0].name == "login"

    def test_with_versions(self):
        m = FileModel(path="/a.py")
        ver = FileVersionModel(file_id="x", file_path="/a.py", version=3, content="v3")
        f = model_to_file(m, versions=[ver])
        assert len(f.versions) == 1
        assert f.versions[0].number == 3
        assert f.versions[0].path == "/a.py@3"

    def test_timestamps(self):
        now = datetime.now(UTC)
        m = FileModel(path="/a.py", created_at=now, updated_at=now)
        f = model_to_file(m)
        assert f.created_at == now
        assert f.updated_at == now

    def test_directory(self):
        m = FileModel(path="/src", is_directory=True)
        f = model_to_file(m)
        assert f.path == "/src"

    def test_current_version(self):
        m = FileModel(path="/a.py", current_version=5)
        f = model_to_file(m)
        assert f.current_version == 5

    def test_none_content(self):
        m = FileModel(path="/a.py", content=None)
        f = model_to_file(m)
        assert f.content is None


class TestModelToChunk:
    def test_basic(self):
        m = FileChunkModel(
            path="/a.py#login",
            file_path="/a.py",
            content="def login(): pass",
            line_start=10,
            line_end=15,
        )
        c = model_to_chunk(m)
        assert c.path == "/a.py#login"
        assert c.name == "login"
        assert c.content == "def login(): pass"
        assert c.line_start == 10
        assert c.line_end == 15

    def test_with_vector(self):
        m = FileChunkModel(
            path="/a.py#func",
            file_path="/a.py",
            embedding=Vector([0.5, 0.6]),
        )
        c = model_to_chunk(m)
        assert c.embedding == [0.5, 0.6]

    def test_no_hash_in_path(self):
        m = FileChunkModel(path="", file_path="/a.py", id="abc123")
        c = model_to_chunk(m)
        assert c.path == "/a.py#abc123"
        assert c.name == ""

    def test_path_with_hash(self):
        m = FileChunkModel(path="/a.py#MyClass", file_path="/a.py")
        c = model_to_chunk(m)
        assert c.name == "MyClass"


class TestModelToVersion:
    def test_basic(self):
        m = FileVersionModel(file_id="x", file_path="/a.py", version=2)
        v = model_to_version(m)
        assert v.path == "/a.py@2"
        assert v.number == 2

    def test_with_vector(self):
        m = FileVersionModel(
            file_id="x",
            file_path="/a.py",
            version=1,
            embedding=Vector([0.1]),
        )
        v = model_to_version(m)
        assert v.embedding == [0.1]

    def test_timestamps(self):
        now = datetime.now(UTC)
        m = FileVersionModel(file_id="x", file_path="/a.py", version=1, created_at=now)
        v = model_to_version(m)
        assert v.created_at == now


class TestModelToConnection:
    def test_basic(self):
        m = FileConnectionModel(
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            weight=0.8,
        )
        c = model_to_connection(m)
        assert c.source_path == "/a.py"
        assert c.target_path == "/b.py"
        assert c.type == "imports"
        assert c.weight == 0.8

    def test_timestamps(self):
        now = datetime.now(UTC)
        m = FileConnectionModel(
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            created_at=now,
            updated_at=now,
        )
        c = model_to_connection(m)
        assert c.created_at == now
        assert c.updated_at == now


class TestFileToModel:
    def test_basic(self):
        f = File(path="/a.py", content="x = 1", lines=1, current_version=2)
        m = file_to_model(f)
        assert m.path == "/a.py"
        assert m.content == "x = 1"
        assert m.lines == 1
        assert m.current_version == 2
        assert m.embedding is None

    def test_with_embedding(self):
        f = File(path="/a.py", embedding=[0.1, 0.2, 0.3])
        m = file_to_model(f)
        assert m.embedding is not None
        assert isinstance(m.embedding, Vector)
        assert list(m.embedding) == [0.1, 0.2, 0.3]

    def test_none_embedding(self):
        f = File(path="/a.py", embedding=None)
        m = file_to_model(f)
        assert m.embedding is None

    def test_directory(self):
        f = File(path="/src")
        m = file_to_model(f)
        assert m.path == "/src"


class TestChunkToModel:
    def test_basic(self):
        c = FileChunk(
            path="/a.py#login",
            name="login",
            content="def login(): pass",
            line_start=10,
            line_end=15,
        )
        m = chunk_to_model(c, file_path="/a.py")
        assert m.path == "/a.py#login"
        assert m.file_path == "/a.py"
        assert m.content == "def login(): pass"
        assert m.line_start == 10
        assert m.line_end == 15

    def test_with_embedding(self):
        c = FileChunk(path="/a.py#f", name="f", embedding=[0.5, 0.6])
        m = chunk_to_model(c, file_path="/a.py")
        assert m.embedding is not None
        assert list(m.embedding) == [0.5, 0.6]


class TestConnectionToModel:
    def test_basic(self):
        c = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            weight=0.5,
        )
        m = connection_to_model(c)
        assert m.source_path == "/a.py"
        assert m.target_path == "/b.py"
        assert m.type == "imports"
        assert m.weight == 0.5
        assert m.path == "/a.py[imports]/b.py"


class TestRoundTrip:
    def test_file_round_trip(self):
        original = File(
            path="/a.py",
            content="hello",
            embedding=[0.1, 0.2],
            lines=1,
            current_version=3,
        )
        model = file_to_model(original)
        restored = model_to_file(model)
        assert restored.path == original.path
        assert restored.content == original.content
        assert restored.embedding == original.embedding
        assert restored.lines == original.lines
        assert restored.current_version == original.current_version

    def test_chunk_round_trip(self):
        original = FileChunk(
            path="/a.py#func",
            name="func",
            content="def func(): pass",
            embedding=[0.3, 0.4],
            line_start=5,
            line_end=10,
        )
        model = chunk_to_model(original, file_path="/a.py")
        restored = model_to_chunk(model)
        assert restored.path == original.path
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.embedding == original.embedding
        assert restored.line_start == original.line_start
        assert restored.line_end == original.line_end

    def test_connection_round_trip(self):
        original = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            weight=0.7,
        )
        model = connection_to_model(original)
        restored = model_to_connection(model)
        assert restored.source_path == original.source_path
        assert restored.target_path == original.target_path
        assert restored.type == original.type
        assert restored.weight == original.weight

    def test_file_none_embedding_round_trip(self):
        original = File(path="/a.py", embedding=None)
        model = file_to_model(original)
        restored = model_to_file(model)
        assert restored.embedding is None

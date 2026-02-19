"""Tests for FileChunk model, ChunkService, and SupportsFileChunks protocol."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, select

from grover.fs.chunks import ChunkService
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_fs import LocalFileSystem
from grover.fs.protocol import SupportsFileChunks
from grover.models.chunks import FileChunk, FileChunkBase

# ---------------------------------------------------------------------------
# Model tests (sync — same pattern as test_models.py)
# ---------------------------------------------------------------------------


class TestFileChunkModel:
    def test_table_exists(self, engine):
        """grover_file_chunks table is created by create_all."""
        assert "grover_file_chunks" in engine.dialect.get_table_names(engine.connect())

    def test_defaults(self, session: Session):
        chunk = FileChunk(
            file_path="/src/main.py",
            chunk_path="/.grover/chunks/src/main.py::MyClass",
            name="MyClass",
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)

        assert chunk.id  # UUID string
        assert chunk.file_path == "/src/main.py"
        assert chunk.chunk_path == "/.grover/chunks/src/main.py::MyClass"
        assert chunk.name == "MyClass"
        assert chunk.description == ""
        assert chunk.line_start == 0
        assert chunk.line_end == 0
        assert chunk.content == ""
        assert chunk.content_hash == ""
        assert chunk.user_id is None
        assert chunk.created_at is not None
        assert chunk.updated_at is not None

    def test_base_subclass_custom_table(self, engine):
        """Custom table name via subclassing FileChunkBase."""

        class CustomChunk(FileChunkBase, table=True):
            __tablename__ = "custom_chunks"

        SQLModel.metadata.create_all(engine)
        tables = engine.dialect.get_table_names(engine.connect())
        assert "custom_chunks" in tables

    def test_round_trip(self, session: Session):
        """Insert and query back a FileChunk."""
        chunk = FileChunk(
            file_path="/test.py",
            chunk_path="/.grover/chunks/test.py::foo",
            name="foo",
            line_start=10,
            line_end=20,
            content="def foo():\n    pass",
            content_hash="abc123",
        )
        session.add(chunk)
        session.commit()

        result = session.exec(select(FileChunk).where(FileChunk.file_path == "/test.py")).first()
        assert result is not None
        assert result.name == "foo"
        assert result.line_start == 10
        assert result.line_end == 20
        assert result.content == "def foo():\n    pass"

    def test_user_id(self, session: Session):
        chunk = FileChunk(
            file_path="/src/main.py",
            chunk_path="/.grover/chunks/src/main.py::fn",
            name="fn",
            user_id="alice",
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        assert chunk.user_id == "alice"

    def test_multiple_chunks_per_file(self, session: Session):
        for i in range(3):
            chunk = FileChunk(
                file_path="/src/main.py",
                chunk_path=f"/.grover/chunks/src/main.py::fn{i}",
                name=f"fn{i}",
                line_start=i * 10,
                line_end=i * 10 + 9,
            )
            session.add(chunk)
        session.commit()

        result = session.exec(select(FileChunk).where(FileChunk.file_path == "/src/main.py")).all()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# ChunkService tests (async)
# ---------------------------------------------------------------------------


class TestChunkService:
    @pytest.fixture
    def service(self):
        return ChunkService(FileChunk)

    async def test_replace_inserts(self, service, async_session):
        chunks = [
            {
                "chunk_path": "/.grover/chunks/a.py::foo",
                "name": "foo",
                "line_start": 1,
                "line_end": 5,
                "content": "def foo(): pass",
                "content_hash": "h1",
            },
            {
                "chunk_path": "/.grover/chunks/a.py::bar",
                "name": "bar",
                "line_start": 7,
                "line_end": 12,
                "content": "def bar(): pass",
                "content_hash": "h2",
            },
        ]
        count = await service.replace_file_chunks(async_session, "/a.py", chunks)
        assert count == 2

        rows = await service.list_file_chunks(async_session, "/a.py")
        assert len(rows) == 2
        assert rows[0].name == "foo"
        assert rows[1].name == "bar"

    async def test_replace_replaces(self, service, async_session):
        """Second replace deletes old chunks and inserts new."""
        old = [{"chunk_path": "old", "name": "old", "content": "old"}]
        await service.replace_file_chunks(async_session, "/a.py", old)

        new = [{"chunk_path": "new", "name": "new", "content": "new"}]
        count = await service.replace_file_chunks(async_session, "/a.py", new)
        assert count == 1

        rows = await service.list_file_chunks(async_session, "/a.py")
        assert len(rows) == 1
        assert rows[0].name == "new"

    async def test_replace_returns_count(self, service, async_session):
        chunks = [{"chunk_path": f"c{i}", "name": f"c{i}"} for i in range(5)]
        count = await service.replace_file_chunks(async_session, "/a.py", chunks)
        assert count == 5

    async def test_delete(self, service, async_session):
        chunks = [{"chunk_path": "c", "name": "c"}]
        await service.replace_file_chunks(async_session, "/a.py", chunks)

        deleted = await service.delete_file_chunks(async_session, "/a.py")
        assert deleted == 1

        rows = await service.list_file_chunks(async_session, "/a.py")
        assert len(rows) == 0

    async def test_delete_nonexistent(self, service, async_session):
        deleted = await service.delete_file_chunks(async_session, "/none.py")
        assert deleted == 0

    async def test_list_ordered_by_line_start(self, service, async_session):
        chunks = [
            {"chunk_path": "c3", "name": "c3", "line_start": 30},
            {"chunk_path": "c1", "name": "c1", "line_start": 10},
            {"chunk_path": "c2", "name": "c2", "line_start": 20},
        ]
        await service.replace_file_chunks(async_session, "/a.py", chunks)

        rows = await service.list_file_chunks(async_session, "/a.py")
        assert [r.line_start for r in rows] == [10, 20, 30]

    async def test_list_empty(self, service, async_session):
        rows = await service.list_file_chunks(async_session, "/missing.py")
        assert rows == []

    async def test_replace_with_user_id(self, service, async_session):
        chunks = [{"chunk_path": "c", "name": "c"}]
        await service.replace_file_chunks(async_session, "/a.py", chunks, user_id="alice")
        rows = await service.list_file_chunks(async_session, "/a.py")
        assert rows[0].user_id == "alice"

    async def test_replace_isolates_files(self, service, async_session):
        """Replacing chunks for file A does not affect file B."""
        await service.replace_file_chunks(
            async_session, "/a.py", [{"chunk_path": "a", "name": "a"}]
        )
        await service.replace_file_chunks(
            async_session, "/b.py", [{"chunk_path": "b", "name": "b"}]
        )

        # Replace a.py chunks
        await service.replace_file_chunks(
            async_session, "/a.py", [{"chunk_path": "a2", "name": "a2"}]
        )

        a_rows = await service.list_file_chunks(async_session, "/a.py")
        b_rows = await service.list_file_chunks(async_session, "/b.py")
        assert len(a_rows) == 1
        assert a_rows[0].name == "a2"
        assert len(b_rows) == 1
        assert b_rows[0].name == "b"


# ---------------------------------------------------------------------------
# Protocol satisfaction tests
# ---------------------------------------------------------------------------


class TestSupportsFileChunksProtocol:
    def test_database_fs_satisfies_protocol(self):
        dbfs = DatabaseFileSystem()
        assert isinstance(dbfs, SupportsFileChunks)

    def test_local_fs_satisfies_protocol(self, tmp_path):
        lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=tmp_path / ".grover")
        assert isinstance(lfs, SupportsFileChunks)


# ---------------------------------------------------------------------------
# Backend end-to-end tests (async, through backend methods)
# ---------------------------------------------------------------------------


class TestDatabaseFSChunks:
    async def test_replace_and_list(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [
            {
                "chunk_path": "/.grover/chunks/a.py::foo",
                "name": "foo",
                "line_start": 1,
                "line_end": 5,
                "content": "def foo(): pass",
            },
        ]
        count = await dbfs.replace_file_chunks("/a.py", chunks, session=async_session)
        assert count == 1

        rows = await dbfs.list_file_chunks("/a.py", session=async_session)
        assert len(rows) == 1
        assert rows[0].name == "foo"

    async def test_delete_through_backend(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [{"chunk_path": "c", "name": "c"}]
        await dbfs.replace_file_chunks("/a.py", chunks, session=async_session)

        deleted = await dbfs.delete_file_chunks("/a.py", session=async_session)
        assert deleted == 1

        rows = await dbfs.list_file_chunks("/a.py", session=async_session)
        assert rows == []

    async def test_replace_and_list_multiple(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [{"chunk_path": f"c{i}", "name": f"c{i}", "line_start": i * 10} for i in range(3)]
        count = await dbfs.replace_file_chunks("/multi.py", chunks, session=async_session)
        assert count == 3

        rows = await dbfs.list_file_chunks("/multi.py", session=async_session)
        assert len(rows) == 3
        # Ordered by line_start
        assert [r.line_start for r in rows] == [0, 10, 20]

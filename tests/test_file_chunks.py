"""Tests for FileChunk model and DefaultChunkProvider."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, select

from grover.backends.database import DatabaseFileSystem
from grover.backends.local import LocalFileSystem
from grover.backends.protocol import GroverFileSystem
from grover.models.chunk import FileChunk, FileChunkBase
from grover.providers.chunks import DefaultChunkProvider

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
            path="/src/main.py#MyClass",
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)

        assert chunk.id  # UUID string
        assert chunk.file_path == "/src/main.py"
        assert chunk.path == "/src/main.py#MyClass"
        assert chunk.line_start == 0
        assert chunk.line_end == 0
        assert chunk.content == ""
        assert chunk.content_hash == ""
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
            path="/test.py#foo",
            line_start=10,
            line_end=20,
            content="def foo():\n    pass",
            content_hash="abc123",
        )
        session.add(chunk)
        session.commit()

        result = session.exec(select(FileChunk).where(FileChunk.file_path == "/test.py")).first()
        assert result is not None
        assert result.line_start == 10
        assert result.line_end == 20
        assert result.content == "def foo():\n    pass"

    def test_multiple_chunks_per_file(self, session: Session):
        for i in range(3):
            chunk = FileChunk(
                file_path="/src/main.py",
                path=f"/src/main.py#fn{i}",
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


class TestDefaultChunkProvider:
    @pytest.fixture
    def service(self):
        return DefaultChunkProvider(FileChunk)

    async def test_replace_inserts(self, service, async_session):
        chunks = [
            {
                "path": "/a.py#foo",
                "line_start": 1,
                "line_end": 5,
                "content": "def foo(): pass",
                "content_hash": "h1",
            },
            {
                "path": "/a.py#bar",
                "line_start": 7,
                "line_end": 12,
                "content": "def bar(): pass",
                "content_hash": "h2",
            },
        ]
        result = await service.replace_file_chunks(async_session, "/a.py", chunks)
        assert result.count == 2

        result = await service.list_file_chunks(async_session, "/a.py")
        assert len(result.chunks) == 2
        assert result.chunks[0].path == "/a.py#foo"
        assert result.chunks[1].path == "/a.py#bar"

    async def test_replace_replaces(self, service, async_session):
        """Second replace deletes old chunks and inserts new."""
        old = [{"path": "old", "content": "old"}]
        await service.replace_file_chunks(async_session, "/a.py", old)

        new = [{"path": "new", "content": "new"}]
        result = await service.replace_file_chunks(async_session, "/a.py", new)
        assert result.count == 1

        result = await service.list_file_chunks(async_session, "/a.py")
        assert len(result.chunks) == 1
        assert result.chunks[0].path == "new"

    async def test_replace_returns_count(self, service, async_session):
        chunks = [{"path": f"c{i}"} for i in range(5)]
        result = await service.replace_file_chunks(async_session, "/a.py", chunks)
        assert result.count == 5

    async def test_delete(self, service, async_session):
        chunks = [{"path": "c"}]
        await service.replace_file_chunks(async_session, "/a.py", chunks)

        result = await service.delete_file_chunks(async_session, "/a.py")
        assert result.count == 1

        result = await service.list_file_chunks(async_session, "/a.py")
        assert len(result.chunks) == 0

    async def test_delete_nonexistent(self, service, async_session):
        result = await service.delete_file_chunks(async_session, "/none.py")
        assert result.count == 0

    async def test_list_ordered_by_line_start(self, service, async_session):
        chunks = [
            {"path": "c3", "line_start": 30},
            {"path": "c1", "line_start": 10},
            {"path": "c2", "line_start": 20},
        ]
        await service.replace_file_chunks(async_session, "/a.py", chunks)

        result = await service.list_file_chunks(async_session, "/a.py")
        assert [r.line_start for r in result.chunks] == [10, 20, 30]

    async def test_list_empty(self, service, async_session):
        result = await service.list_file_chunks(async_session, "/missing.py")
        assert result.chunks == []

    async def test_replace_isolates_files(self, service, async_session):
        """Replacing chunks for file A does not affect file B."""
        await service.replace_file_chunks(async_session, "/a.py", [{"path": "a"}])
        await service.replace_file_chunks(async_session, "/b.py", [{"path": "b"}])

        # Replace a.py chunks
        await service.replace_file_chunks(async_session, "/a.py", [{"path": "a2"}])

        a_result = await service.list_file_chunks(async_session, "/a.py")
        b_result = await service.list_file_chunks(async_session, "/b.py")
        assert len(a_result.chunks) == 1
        assert a_result.chunks[0].path == "a2"
        assert len(b_result.chunks) == 1
        assert b_result.chunks[0].path == "b"


# ---------------------------------------------------------------------------
# Protocol satisfaction tests
# ---------------------------------------------------------------------------


class TestGroverFileSystemProtocol:
    def test_database_fs_satisfies_protocol(self):
        dbfs = DatabaseFileSystem()
        assert isinstance(dbfs, GroverFileSystem)

    def test_local_fs_satisfies_protocol(self, tmp_path):
        lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=tmp_path / ".grover")
        assert isinstance(lfs, GroverFileSystem)


# ---------------------------------------------------------------------------
# Backend end-to-end tests (async, through backend methods)
# ---------------------------------------------------------------------------


class TestDatabaseFSChunks:
    async def test_replace_and_list(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [
            {
                "path": "/a.py#foo",
                "line_start": 1,
                "line_end": 5,
                "content": "def foo(): pass",
            },
        ]
        result = await dbfs.replace_file_chunks("/a.py", chunks, session=async_session)
        assert result.count == 1

        result = await dbfs.list_file_chunks("/a.py", session=async_session)
        assert len(result.chunks) == 1
        assert result.chunks[0].path == "/a.py#foo"

    async def test_delete_through_backend(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [{"path": "c"}]
        await dbfs.replace_file_chunks("/a.py", chunks, session=async_session)

        result = await dbfs.delete_file_chunks("/a.py", session=async_session)
        assert result.count == 1

        result = await dbfs.list_file_chunks("/a.py", session=async_session)
        assert result.chunks == []

    async def test_replace_and_list_multiple(self, async_session):
        dbfs = DatabaseFileSystem()
        chunks = [{"path": f"c{i}", "line_start": i * 10} for i in range(3)]
        result = await dbfs.replace_file_chunks("/multi.py", chunks, session=async_session)
        assert result.count == 3

        result = await dbfs.list_file_chunks("/multi.py", session=async_session)
        assert len(result.chunks) == 3
        # Ordered by line_start
        assert [r.line_start for r in result.chunks] == [0, 10, 20]

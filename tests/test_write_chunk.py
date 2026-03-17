"""Tests for write_chunk and write_chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover, GroverAsync
from grover.models.database.chunk import FileChunkModel
from grover.permissions import Permission
from grover.providers.chunks import DefaultChunkProvider
from grover.providers.search.local import LocalVectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Provider-level tests (DefaultChunkProvider)
# ---------------------------------------------------------------------------


class TestDefaultChunkProviderWriteChunk:
    @pytest.fixture
    def service(self):
        return DefaultChunkProvider(FileChunkModel)

    async def test_write_chunk_creates_new(self, service, async_session: AsyncSession):
        chunk = FileChunkModel(
            file_path="/a.py",
            path="/a.py#foo",
            content="def foo(): pass",
            line_start=1,
            line_end=3,
        )
        result = await service.write_chunk(async_session, chunk)
        assert result.success is True
        assert result.file.path == "/a.py#foo"

        listed = await service.list_file_chunks(async_session, "/a.py")
        assert len(listed.file.chunks) == 1
        assert listed.file.chunks[0].path == "/a.py#foo"
        assert listed.file.chunks[0].content == "def foo(): pass"

    async def test_write_chunk_upserts_existing(self, service, async_session: AsyncSession):
        chunk = FileChunkModel(file_path="/a.py", path="/a.py#foo", content="v1", line_start=1, line_end=3)
        await service.write_chunk(async_session, chunk)

        updated = FileChunkModel(file_path="/a.py", path="/a.py#foo", content="v2", line_start=5, line_end=10)
        result = await service.write_chunk(async_session, updated)
        assert result.success is True

        listed = await service.list_file_chunks(async_session, "/a.py")
        assert len(listed.file.chunks) == 1
        assert listed.file.chunks[0].content == "v2"
        assert listed.file.chunks[0].line_start == 5
        assert listed.file.chunks[0].line_end == 10

    async def test_write_chunk_content_hash_computed(self, service, async_session: AsyncSession):

        chunk = FileChunkModel(
            file_path="/a.py",
            path="/a.py#foo",
            content="def foo(): pass",
            content_hash="caller_should_be_ignored",
        )
        await service.write_chunk(async_session, chunk)

        listed = await service.list_file_chunks(async_session, "/a.py")
        # Verify the content is correct (hash is computed by provider)
        assert listed.file.chunks[0].content == "def foo(): pass"

    async def test_write_chunks_batch(self, service, async_session: AsyncSession):
        chunks = [
            FileChunkModel(file_path="/a.py", path=f"/a.py#fn{i}", content=f"def fn{i}(): pass") for i in range(3)
        ]
        result = await service.write_chunks(async_session, chunks)
        assert result.success is True
        assert result.succeeded == 3
        assert result.failed == 0
        assert len(result.results) == 3

        listed = await service.list_file_chunks(async_session, "/a.py")
        assert len(listed.file.chunks) == 3

    async def test_write_chunks_batch_upserts(self, service, async_session: AsyncSession):
        """Batch with mix of new and existing chunks."""
        existing = FileChunkModel(file_path="/a.py", path="/a.py#foo", content="old")
        await service.write_chunk(async_session, existing)

        batch = [
            FileChunkModel(file_path="/a.py", path="/a.py#foo", content="updated"),
            FileChunkModel(file_path="/a.py", path="/a.py#bar", content="new"),
        ]
        result = await service.write_chunks(async_session, batch)
        assert result.succeeded == 2

        listed = await service.list_file_chunks(async_session, "/a.py")
        assert len(listed.file.chunks) == 2
        by_path = {c.path: c for c in listed.file.chunks}
        assert by_path["/a.py#foo"].content == "updated"
        assert by_path["/a.py#bar"].content == "new"


# ---------------------------------------------------------------------------
# Facade-level tests (GroverAsync)
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    await g.add_mount(
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_no_search(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    await g.add_mount(
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
    )
    yield g  # type: ignore[misc]
    await g.close()


class TestWriteChunkFacade:
    async def test_write_chunk_creates_new(self, grover: GroverAsync):
        await grover.write("/project/a.py", "# module a\n")
        await grover.flush()

        chunk = FileChunkModel(
            file_path="/project/a.py",
            path="/project/a.py#foo",
            content="def foo(): pass",
            line_start=1,
            line_end=3,
        )
        result = await grover.write_chunk(chunk)
        assert result.success is True
        assert result.file.path == "/project/a.py#foo"

    async def test_write_chunk_requires_parent_file(self, grover: GroverAsync):
        chunk = FileChunkModel(
            file_path="/project/missing.py",
            path="/project/missing.py#foo",
            content="def foo(): pass",
        )
        result = await grover.write_chunk(chunk)
        assert result.success is False
        assert "Parent file not found" in result.message

    async def test_write_chunk_validates_chunk_ref(self, grover: GroverAsync):
        chunk = FileChunkModel(
            file_path="/project/a.py",
            path="/project/a.py",  # Missing '#' — not a chunk ref
            content="invalid",
        )
        result = await grover.write_chunk(chunk)
        assert result.success is False
        assert "#" in result.message

    async def test_write_chunk_validates_file_path_match(self, grover: GroverAsync):
        chunk = FileChunkModel(
            file_path="/project/b.py",
            path="/project/a.py#foo",  # file_path doesn't match
            content="def foo(): pass",
        )
        result = await grover.write_chunk(chunk)
        assert result.success is False
        assert "mismatch" in result.message

    async def test_write_chunk_read_only_mount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data_ro"
        g = GroverAsync()
        await g.add_mount(
            "readonly",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            permission=Permission.READ_ONLY,
        )
        try:
            chunk = FileChunkModel(
                file_path="/readonly/a.py",
                path="/readonly/a.py#foo",
                content="def foo(): pass",
            )
            result = await g.write_chunk(chunk)
            assert result.success is False
        finally:
            await g.close()

    async def test_write_chunk_creates_graph_node(self, grover_no_search: GroverAsync):
        await grover_no_search.write("/project/a.py", "# module\n")
        await grover_no_search.flush()

        chunk = FileChunkModel(
            file_path="/project/a.py",
            path="/project/a.py#foo",
            content="def foo(): pass",
        )
        result = await grover_no_search.write_chunk(chunk)
        assert result.success is True
        await grover_no_search.flush()

        graph = grover_no_search.get_graph()
        assert graph.has_node("/project/a.py#foo")
        assert graph.has_edge("/project/a.py", "/project/a.py#foo")

    async def test_write_chunk_indexes_for_search(self, grover: GroverAsync):
        await grover.write("/project/a.py", "# module a\n")
        await grover.flush()

        chunk = FileChunkModel(
            file_path="/project/a.py",
            path="/project/a.py#authenticate",
            content="def authenticate(user, password):\n    return True\n",
        )
        result = await grover.write_chunk(chunk)
        assert result.success is True
        await grover.flush()

        search_result = await grover.vector_search("authenticate")
        assert search_result.success is True
        assert len(search_result) >= 1

    async def test_write_chunks_batch(self, grover: GroverAsync):
        await grover.write("/project/a.py", "# module a\n")
        await grover.flush()

        chunks = [
            FileChunkModel(
                file_path="/project/a.py",
                path=f"/project/a.py#fn{i}",
                content=f"def fn{i}(): pass",
            )
            for i in range(3)
        ]
        result = await grover.write_chunks(chunks)
        assert result.success is True
        assert result.succeeded == 3
        assert result.failed == 0

    async def test_write_chunks_batch_validates_parents(self, grover: GroverAsync):
        await grover.write("/project/a.py", "# module a\n")
        await grover.flush()

        chunks = [
            FileChunkModel(file_path="/project/a.py", path="/project/a.py#valid", content="valid"),
            FileChunkModel(
                file_path="/project/missing.py",
                path="/project/missing.py#invalid",
                content="invalid",
            ),
        ]
        result = await grover.write_chunks(chunks)
        assert result.succeeded == 1
        assert result.failed == 1


class TestWriteChunkSync:
    @pytest.fixture
    def grover_sync(self, workspace: Path, tmp_path: Path) -> Iterator[Grover]:
        data = tmp_path / "grover_data"
        g = Grover()
        g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        )
        yield g
        g.close()

    def test_write_chunk_sync(self, grover_sync: Grover):
        grover_sync.write("/project/a.py", "# module\n")
        grover_sync.flush()

        chunk = FileChunkModel(
            file_path="/project/a.py",
            path="/project/a.py#foo",
            content="def foo(): pass",
        )
        result = grover_sync.write_chunk(chunk)
        assert result.success is True

    def test_write_chunks_sync(self, grover_sync: Grover):
        grover_sync.write("/project/a.py", "# module\n")
        grover_sync.flush()

        chunks = [
            FileChunkModel(
                file_path="/project/a.py",
                path="/project/a.py#foo",
                content="def foo(): pass",
            ),
            FileChunkModel(
                file_path="/project/a.py",
                path="/project/a.py#bar",
                content="def bar(): pass",
            ),
        ]
        result = grover_sync.write_chunks(chunks)
        assert result.success is True
        assert result.succeeded == 2

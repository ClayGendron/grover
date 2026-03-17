"""Tests for write_file and write_files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover, GroverAsync
from grover.models.database.file import FileModel
from grover.permissions import Permission
from grover.providers.search.local import LocalVectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# write_file tests
# ---------------------------------------------------------------------------


class TestWriteFile:
    async def test_write_file_model_creates_new(self, grover: GroverAsync):
        f = FileModel(path="/project/a.py", content="print('hi')\n")
        result = await grover.write_file(f)
        assert result.success is True
        assert "Created" in result.message
        assert result.file.current_version == 1

        read = await grover.read("/project/a.py")
        assert read.success is True
        assert "print('hi')" in read.file.content

    async def test_write_file_model_updates_existing(self, grover: GroverAsync):
        await grover.write("/project/a.py", "v1\n")

        f = FileModel(path="/project/a.py", content="v2\n")
        result = await grover.write_file(f)
        assert result.success is True
        assert "Created" not in result.message
        assert result.file.current_version == 2

        read = await grover.read("/project/a.py")
        assert "v2" in read.file.content

    async def test_write_file_model_ignores_caller_metadata(self, grover: GroverAsync):
        """System computes hash, size, version — ignores caller values."""
        f = FileModel(
            path="/project/a.py",
            content="hello\n",
            content_hash="should_be_ignored",
            size_bytes=99999,
        )
        result = await grover.write_file(f)
        assert result.success is True

        # System managed the write correctly
        read = await grover.read("/project/a.py")
        assert "hello" in read.file.content

    async def test_write_file_non_text_rejected(self, grover: GroverAsync):
        f = FileModel(path="/project/image.png", content="data")
        result = await grover.write_file(f)
        assert result.success is False

    async def test_write_file_read_only_mount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data_ro"
        g = GroverAsync()
        await g.add_mount(
            "readonly",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            permission=Permission.READ_ONLY,
        )
        try:
            f = FileModel(path="/readonly/a.py", content="print('hi')\n")
            result = await g.write_file(f)
            assert result.success is False
        finally:
            await g.close()


# ---------------------------------------------------------------------------
# write_files tests
# ---------------------------------------------------------------------------


class TestWriteFiles:
    async def test_write_files_batch_creates_multiple(self, grover: GroverAsync):
        files = [FileModel(path=f"/project/f{i}.py", content=f"# file {i}\n") for i in range(3)]
        result = await grover.write_files(files)
        assert result.succeeded == 3
        assert result.failed == 0

        for i in range(3):
            read = await grover.read(f"/project/f{i}.py")
            assert read.success is True
            assert f"file {i}" in read.file.content

    async def test_write_files_batch_mix_create_update(self, grover: GroverAsync):
        await grover.write("/project/existing.py", "old\n")

        files = [
            FileModel(path="/project/new1.py", content="new1\n"),
            FileModel(path="/project/existing.py", content="updated\n"),
            FileModel(path="/project/new2.py", content="new2\n"),
        ]
        result = await grover.write_files(files)
        assert result.succeeded == 3
        assert "Created" in result.results[0].message
        assert "Created" not in result.results[1].message  # Updated existing
        assert "Created" in result.results[2].message

    async def test_write_files_batch_versions_each(self, grover: GroverAsync):
        """Each file in batch gets its own version record."""
        files = [FileModel(path=f"/project/v{i}.py", content=f"content {i}\n") for i in range(3)]
        result = await grover.write_files(files)
        assert all(r.file.current_version == 1 for r in result.results)

        # Update one of them
        files2 = [FileModel(path="/project/v1.py", content="updated\n")]
        result2 = await grover.write_files(files2)
        assert result2.results[0].file.current_version == 2

    async def test_write_files_batch_large(self, grover: GroverAsync):
        """Large batches are auto-chunked internally — no user-facing limit."""
        files = [FileModel(path=f"/project/f{i}.py", content="x\n") for i in range(150)]
        result = await grover.write_files(files)
        assert result.success is True
        assert result.succeeded == 150
        assert len(result.results) == 150

    async def test_write_files_batch_read_only(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data_ro"
        g = GroverAsync()
        await g.add_mount(
            "readonly",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            permission=Permission.READ_ONLY,
        )
        try:
            files = [FileModel(path="/readonly/a.py", content="x\n")]
            result = await g.write_files(files)
            assert result.succeeded == 0
            assert result.failed == 1
        finally:
            await g.close()

    async def test_write_files_batch_parent_dirs_created(self, grover: GroverAsync):
        """New files in nested paths get parent dirs."""
        files = [
            FileModel(path="/project/deep/nested/a.py", content="x\n"),
            FileModel(path="/project/deep/nested/b.py", content="y\n"),
        ]
        result = await grover.write_files(files)
        assert result.succeeded == 2

        # Parent dir should exist
        exists = await grover.exists("/project/deep/nested")
        assert exists.message == "exists"

    async def test_write_files_batch_triggers_indexing(self, grover_no_search: GroverAsync):
        files = [
            FileModel(path="/project/a.py", content="def foo():\n    pass\n"),
        ]
        result = await grover_no_search.write_files(files)
        assert result.succeeded == 1
        await grover_no_search.flush()

        graph = grover_no_search.get_graph()
        assert graph.has_node("/project/a.py")

    async def test_write_files_batch_partial_failure(self, grover: GroverAsync):
        """Binary extension fails, text succeeds."""
        files = [
            FileModel(path="/project/good.py", content="# good\n"),
            FileModel(path="/project/bad.png", content="binary"),
        ]
        result = await grover.write_files(files)
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.results[0].success is True
        assert result.results[1].success is False


# ---------------------------------------------------------------------------
# Sync wrapper tests
# ---------------------------------------------------------------------------


class TestWriteFilesSync:
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

    def test_write_file_sync(self, grover_sync: Grover):
        f = FileModel(path="/project/a.py", content="print('hi')\n")
        result = grover_sync.write_file(f)
        assert result.success is True
        assert "Created" in result.message

    def test_write_files_sync(self, grover_sync: Grover):
        files = [
            FileModel(path="/project/a.py", content="# a\n"),
            FileModel(path="/project/b.py", content="# b\n"),
        ]
        result = grover_sync.write_files(files)
        assert result.succeeded == 2


# ---------------------------------------------------------------------------
# FileModel.create() factory tests
# ---------------------------------------------------------------------------


class TestFileModelCreate:
    def test_create_factory_populates_all_fields(self):
        f = FileModel.create("a.py", "x = 1\n", mount="project")
        assert f.path == "/project/a.py"
        assert f.parent_path == "/project"
        assert f.is_directory is False
        assert f.content == "x = 1\n"
        assert f.content_hash is not None
        assert f.mime_type == "text/x-python"
        assert f.lines == 1
        assert f.size_bytes == len(b"x = 1\n")
        assert f.created_at is not None
        assert f.updated_at is not None

    def test_create_factory_with_embedding_and_tokens(self):
        f = FileModel.create("a.py", "code\n", mount="proj", embedding=[0.1, 0.2], tokens=42)
        assert f.embedding is not None
        assert list(f.embedding) == [0.1, 0.2]
        assert f.tokens == 42

    def test_create_factory_no_mount(self):
        f = FileModel.create("/src/a.py", "hello\n")
        assert f.path == "/src/a.py"
        assert f.parent_path == "/src"

    def test_create_factory_mount_strips_slashes(self):
        f = FileModel.create("a.py", "", mount="/project/")
        assert f.path == "/project/a.py"

    def test_create_factory_empty_content(self):
        f = FileModel.create("empty.py", "", mount="proj")
        assert f.content == ""
        assert f.size_bytes == 0
        assert f.lines == 0

    def test_create_factory_owner_id(self):
        f = FileModel.create("a.py", "x\n", mount="proj", owner_id="alice")
        assert f.owner_id == "alice"


# ---------------------------------------------------------------------------
# Embedding / tokens flow-through tests
# ---------------------------------------------------------------------------


class TestWriteFilesModelFlowThrough:
    async def test_write_files_preserves_embedding(self, grover_no_search: GroverAsync):
        """Embedding set on model is persisted to DB."""
        f = FileModel.create("embed.py", "x = 1\n", mount="project", embedding=[0.1, 0.2, 0.3])
        result = await grover_no_search.write_files([f])
        assert result.succeeded == 1

        info = await grover_no_search.get_info("/project/embed.py")
        assert info.success is True

    async def test_write_files_preserves_tokens(self, grover_no_search: GroverAsync):
        """Tokens set on model is persisted to DB."""
        f = FileModel.create("tok.py", "x = 1\n", mount="project", tokens=150)
        result = await grover_no_search.write_files([f])
        assert result.succeeded == 1

    async def test_write_files_timestamps_set_by_backend(self, grover_no_search: GroverAsync):
        """Models without timestamps get them from the backend."""
        f = FileModel(path="/project/ts.py", content="x\n")
        assert f.created_at is None  # None before write
        result = await grover_no_search.write_files([f])
        assert result.succeeded == 1

    async def test_write_files_embedding_update_existing(self, grover_no_search: GroverAsync):
        """Updating an existing file with embedding merges it."""
        await grover_no_search.write("/project/up.py", "v1\n")

        f = FileModel.create("up.py", "v2\n", mount="project", embedding=[0.5, 0.6])
        result = await grover_no_search.write_files([f])
        assert result.succeeded == 1
        assert result.results[0].file.current_version == 2

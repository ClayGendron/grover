"""Tests for the GroverAsync class."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover._grover_async import GroverAsync
from grover.fs.local_fs import LocalFileSystem
from grover.graph._graph import Graph
from grover.search._index import SearchResult

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider (deterministic, fast)
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing."""

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def workspace2(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace2"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_no_search(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data))
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Lifecycle
# ==================================================================


class TestGroverAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_construction_no_args(self):
        g = GroverAsync()
        assert g._meta_fs is None  # No mounts yet
        assert g._in_transaction is False

    @pytest.mark.asyncio
    async def test_mount_creates_meta_fs(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        assert g._meta_fs is not None
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        await g.write("/app/test.txt", "hello")
        assert await g.exists("/app/test.txt")
        await g.unmount("/app")
        # Mount should be gone
        assert not g._registry.has_mount("/app")
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount_grover_raises(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        with pytest.raises(ValueError, match="Cannot unmount /.grover"):
            await g.unmount("/.grover")
        await g.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        await g.close()
        await g.close()  # Should not raise


# ==================================================================
# Direct Access Mode
# ==================================================================


class TestGroverAsyncDirectAccess:
    @pytest.mark.asyncio
    async def test_write_and_read(self, grover: GroverAsync):
        assert await grover.write("/project/hello.txt", "hello world")
        result = await grover.read("/project/hello.txt")
        assert result.success
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_edit(self, grover: GroverAsync):
        await grover.write("/project/doc.txt", "old text here")
        assert await grover.edit("/project/doc.txt", "old", "new")
        result = await grover.read("/project/doc.txt")
        assert result.content == "new text here"

    @pytest.mark.asyncio
    async def test_delete(self, grover: GroverAsync):
        await grover.write("/project/tmp.txt", "temporary")
        assert await grover.delete("/project/tmp.txt")
        result = await grover.read("/project/tmp.txt")
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_dir(self, grover: GroverAsync):
        await grover.write("/project/a.txt", "a")
        await grover.write("/project/b.txt", "b")
        entries = await grover.list_dir("/project")
        names = {e["name"] for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names

    @pytest.mark.asyncio
    async def test_exists(self, grover: GroverAsync):
        assert not await grover.exists("/project/nope.txt")
        await grover.write("/project/yes.txt", "yes")
        assert await grover.exists("/project/yes.txt")


# ==================================================================
# Transaction Mode
# ==================================================================


class TestGroverAsyncTransaction:
    @pytest.mark.asyncio
    async def test_context_manager_commits(self, grover: GroverAsync):
        async with grover:
            await grover.write("/project/a.txt", "a")
            await grover.write("/project/b.txt", "b")

        # Both should exist after context exit
        assert (await grover.read("/project/a.txt")).content == "a"
        assert (await grover.read("/project/b.txt")).content == "b"

    @pytest.mark.asyncio
    async def test_operations_after_transaction(self, grover: GroverAsync):
        async with grover:
            await grover.write("/project/in_tx.txt", "in transaction")

        # Should work after transaction ends
        await grover.write("/project/after_tx.txt", "after transaction")
        assert (await grover.read("/project/after_tx.txt")).content == "after transaction"

    @pytest.mark.asyncio
    async def test_transaction_sets_backend_flag(self, workspace: Path, tmp_path: Path):
        """``async with g:`` sets ``in_transaction`` on backends."""
        from grover.fs.database_fs import DatabaseFileSystem

        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        local = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local")
        await g.mount("/app", local)

        assert local.in_transaction is False
        async with g:
            assert local.in_transaction is True
        assert local.in_transaction is False
        await g.close()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, tmp_path: Path):
        """Writes inside ``async with g:`` are rolled back on exception."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel, select

        from grover.fs.database_fs import DatabaseFileSystem
        from grover.models.files import File

        db_path = tmp_path / "rollback_test.db"
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", echo=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False,
        )

        db = DatabaseFileSystem(session_factory=factory, dialect="sqlite")
        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.mount("/app", db)

        # Write inside a transaction that raises — should be rolled back
        with pytest.raises(RuntimeError, match="boom"):
            async with g:
                await g.write("/app/doomed.txt", "this should vanish")
                raise RuntimeError("boom")

        # The file should NOT exist after rollback
        async with factory() as session:
            result = await session.execute(
                select(File).where(File.path == "/doomed.txt")
            )
            assert result.scalar_one_or_none() is None, (
                "Data should be rolled back after exception"
            )

        await g.close()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_reads_visible_during_transaction(self, grover: GroverAsync):
        """Files written inside ``async with g:`` are readable before commit."""
        async with grover:
            await grover.write("/project/visible.txt", "hello")
            assert (await grover.read("/project/visible.txt")).content == "hello"
            assert await grover.exists("/project/visible.txt")

    @pytest.mark.asyncio
    async def test_rollback_multi_mount(self, tmp_path: Path):
        """Writes to multiple DatabaseFileSystem mounts roll back together."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel, select

        from grover.fs.database_fs import DatabaseFileSystem
        from grover.models.files import File

        # Set up two separate SQLite databases
        db_path_a = tmp_path / "mount_a.db"
        db_path_b = tmp_path / "mount_b.db"

        engine_a = create_async_engine(f"sqlite+aiosqlite:///{db_path_a}")
        engine_b = create_async_engine(f"sqlite+aiosqlite:///{db_path_b}")

        async with engine_a.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with engine_b.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory_a = async_sessionmaker(
            engine_a, class_=AsyncSession, expire_on_commit=False,
        )
        factory_b = async_sessionmaker(
            engine_b, class_=AsyncSession, expire_on_commit=False,
        )

        db_a = DatabaseFileSystem(session_factory=factory_a, dialect="sqlite")
        db_b = DatabaseFileSystem(session_factory=factory_b, dialect="sqlite")

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.mount("/a", db_a)
        await g.mount("/b", db_b)

        # Write to both mounts inside a transaction, then raise
        with pytest.raises(RuntimeError, match="cross-mount boom"):
            async with g:
                await g.write("/a/file.txt", "content a")
                await g.write("/b/file.txt", "content b")
                # Both readable during txn
                assert (await g.read("/a/file.txt")).content == "content a"
                assert (await g.read("/b/file.txt")).content == "content b"
                raise RuntimeError("cross-mount boom")

        # Both should be rolled back
        async with factory_a() as session:
            result = await session.execute(
                select(File).where(File.path == "/file.txt")
            )
            assert result.scalar_one_or_none() is None, (
                "Mount /a should be rolled back"
            )

        async with factory_b() as session:
            result = await session.execute(
                select(File).where(File.path == "/file.txt")
            )
            assert result.scalar_one_or_none() is None, (
                "Mount /b should be rolled back"
            )

        await g.close()
        await engine_a.dispose()
        await engine_b.dispose()


# ==================================================================
# Multi-Mount CRUD
# ==================================================================


class TestGroverAsyncMultiMount:
    @pytest.mark.asyncio
    async def test_two_mounts(self, workspace: Path, workspace2: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_app"))
        await g.mount("/data", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_data"))

        await g.write("/app/code.txt", "code content")
        await g.write("/data/doc.txt", "doc content")

        assert (await g.read("/app/code.txt")).content == "code content"
        assert (await g.read("/data/doc.txt")).content == "doc content"

        # List root should show both mounts (but not .grover)
        entries = await g.list_dir("/")
        names = {e["name"] for e in entries}
        assert "app" in names
        assert "data" in names
        assert ".grover" not in names

        await g.close()

    @pytest.mark.asyncio
    async def test_isolation_between_mounts(self, workspace: Path, workspace2: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/a", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a"))
        await g.mount("/b", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b"))

        await g.write("/a/file.txt", "in mount a")
        assert await g.exists("/a/file.txt")
        assert not await g.exists("/b/file.txt")

        await g.close()


# ==================================================================
# Graph
# ==================================================================


class TestGroverAsyncGraph:
    @pytest.mark.asyncio
    async def test_graph_property(self, grover: GroverAsync):
        assert isinstance(grover.graph, Graph)

    @pytest.mark.asyncio
    async def test_write_updates_graph(self, grover: GroverAsync):
        await grover.write("/project/mod.py", 'def work():\n    pass\n')
        assert grover.graph.has_node("/project/mod.py")

    @pytest.mark.asyncio
    async def test_contains_returns_chunks(self, grover: GroverAsync):
        code = 'def foo():\n    pass\n\ndef bar():\n    pass\n'
        await grover.write("/project/funcs.py", code)
        chunks = grover.contains("/project/funcs.py")
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_delete_removes_from_graph(self, grover: GroverAsync):
        await grover.write("/project/gone.py", 'def gone():\n    pass\n')
        assert grover.graph.has_node("/project/gone.py")
        await grover.delete("/project/gone.py")
        assert not grover.graph.has_node("/project/gone.py")


# ==================================================================
# Search
# ==================================================================


class TestGroverAsyncSearch:
    @pytest.mark.asyncio
    async def test_search_after_write(self, grover: GroverAsync):
        code = 'def authenticate_user():\n    """Verify user credentials."""\n    pass\n'
        await grover.write("/project/auth.py", code)
        results = await grover.search("authenticate")
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self, grover: GroverAsync):
        await grover.write("/project/data.txt", "important data content")
        results = await grover.search("data")
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty(self, grover: GroverAsync):
        results = await grover.search("nonexistent query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_raises_without_provider(self, grover_no_search: GroverAsync):
        if grover_no_search._search_index is not None:
            pytest.skip("sentence-transformers is installed; search available")
        with pytest.raises(RuntimeError, match="Search is not available"):
            await grover_no_search.search("anything")


# ==================================================================
# Index
# ==================================================================


class TestGroverAsyncIndex:
    @pytest.mark.asyncio
    async def test_index_scans_files(self, grover: GroverAsync, workspace: Path):
        (workspace / "one.py").write_text('def one():\n    return 1\n')
        (workspace / "two.py").write_text('def two():\n    return 2\n')
        stats = await grover.index()
        assert stats["files_scanned"] >= 2

    @pytest.mark.asyncio
    async def test_index_specific_mount(
        self, workspace: Path, workspace2: Path, tmp_path: Path,
    ):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/a", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a"))
        await g.mount("/b", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b"))

        (workspace / "a.py").write_text('def a():\n    pass\n')
        (workspace2 / "b.py").write_text('def b():\n    pass\n')

        stats = await g.index("/a")
        # Should only index mount /a
        assert stats["files_scanned"] >= 1
        assert g.graph.has_node("/a/a.py")

        await g.close()


# ==================================================================
# Persistence
# ==================================================================


class TestGroverAsyncPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, workspace: Path, tmp_path: Path):
        data_dir = tmp_path / "data"

        g1 = GroverAsync(
            data_dir=str(data_dir), embedding_provider=FakeProvider()
        )
        await g1.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local"))
        await g1.write("/project/keep.py", 'def keep():\n    pass\n')
        await g1.save()
        await g1.close()

        g2 = GroverAsync(
            data_dir=str(data_dir), embedding_provider=FakeProvider()
        )
        await g2.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local"))
        assert g2.graph.has_node("/project/keep.py")
        results = await g2.search("keep")
        assert len(results) >= 1
        await g2.close()


# ==================================================================
# Properties
# ==================================================================


class TestGroverAsyncProperties:
    @pytest.mark.asyncio
    async def test_fs_property(self, grover: GroverAsync):
        assert grover.fs is grover._ufs

    @pytest.mark.asyncio
    async def test_graph_property(self, grover: GroverAsync):
        assert grover.graph is grover._graph

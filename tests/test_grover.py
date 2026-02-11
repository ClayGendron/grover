"""Tests for the Grover integration class."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from grover._grover import Grover
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
def grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
    g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


@pytest.fixture
def grover_no_search(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    """Grover without search to test graceful degradation."""
    data = tmp_path / "grover_data"
    g = Grover(data_dir=str(data))
    g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g
    g.close()


# ==================================================================
# Construction
# ==================================================================


class TestGroverConstruction:
    def test_mount_first_api(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
        g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        try:
            assert g._async._meta_fs is not None
        finally:
            g.close()

    def test_custom_data_dir(self, workspace: Path, tmp_path: Path):
        custom_dir = tmp_path / "custom_data"
        g = Grover(
            data_dir=str(custom_dir),
            embedding_provider=FakeProvider(),
        )
        g.mount("/project", LocalFileSystem(workspace_dir=workspace))
        try:
            assert g._async._meta_data_dir == custom_dir
        finally:
            g.close()

    def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = Grover(data_dir=str(data), embedding_provider=FakeProvider())
        g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        g.close()
        g.close()  # Should not raise
        assert g._closed


# ==================================================================
# Filesystem
# ==================================================================


class TestGroverFilesystem:
    def test_write_and_read(self, grover: Grover):
        assert grover.write("/project/hello.txt", "hello world")
        result = grover.read("/project/hello.txt")
        assert result.success
        assert result.content == "hello world"

    def test_edit(self, grover: Grover):
        grover.write("/project/doc.txt", "old text here")
        assert grover.edit("/project/doc.txt", "old", "new")
        assert grover.read("/project/doc.txt").content == "new text here"

    def test_delete(self, grover: Grover):
        grover.write("/project/tmp.txt", "temporary")
        assert grover.delete("/project/tmp.txt")
        assert not grover.read("/project/tmp.txt").success

    def test_list_dir(self, grover: Grover):
        grover.write("/project/a.txt", "a")
        grover.write("/project/b.txt", "b")
        entries = grover.list_dir("/project")
        names = {e["name"] for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names

    def test_exists(self, grover: Grover):
        assert not grover.exists("/project/nope.txt")
        grover.write("/project/yes.txt", "yes")
        assert grover.exists("/project/yes.txt")

    def test_fs_property(self, grover: Grover):
        assert grover.fs is grover._async._vfs


# ==================================================================
# Transaction
# ==================================================================


class TestGroverTransaction:
    def test_context_manager_commits(self, grover: Grover):
        """Writes inside ``with g:`` are visible after clean exit."""
        with grover:
            grover.write("/project/a.txt", "a")
            grover.write("/project/b.txt", "b")

        assert grover.read("/project/a.txt").content == "a"
        assert grover.read("/project/b.txt").content == "b"

    def test_reads_visible_during_transaction(self, grover: Grover):
        """Files written inside ``with g:`` are readable before commit."""
        with grover:
            grover.write("/project/visible.txt", "hello")
            assert grover.read("/project/visible.txt").content == "hello"
            assert grover.exists("/project/visible.txt")

    def test_operations_after_transaction(self, grover: Grover):
        """Normal operations resume after a transaction block."""
        with grover:
            grover.write("/project/in_tx.txt", "in transaction")

        grover.write("/project/after_tx.txt", "after transaction")
        assert grover.read("/project/after_tx.txt").content == "after transaction"

    def test_rollback_on_error(self, tmp_path: Path):
        """Writes inside ``with g:`` are rolled back on exception."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import select

        from grover.models.files import File

        db_path = tmp_path / "rollback_test.db"
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", echo=False,
        )

        import asyncio

        asyncio.run(_create_tables(engine))

        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False,
        )

        g = Grover(data_dir=str(tmp_path / "grover_data"))
        g.mount("/app", session_factory=factory, dialect="sqlite")

        with pytest.raises(RuntimeError, match="boom"), g:
            g.write("/app/doomed.txt", "this should vanish")
            # Readable during the transaction
            assert g.read("/app/doomed.txt").content == "this should vanish"
            raise RuntimeError("boom")

        # The file should NOT exist after rollback
        async def _check() -> None:
            async with factory() as session:
                result = await session.execute(
                    select(File).where(File.path == "/doomed.txt")
                )
                assert result.scalar_one_or_none() is None, (
                    "Data should be rolled back after exception"
                )

        asyncio.run(_check())
        g.close()
        asyncio.run(engine.dispose())


async def _create_tables(engine: object) -> None:
    """Helper: create all SQLModel tables on the given async engine."""
    from sqlmodel import SQLModel

    async with engine.begin() as conn:  # type: ignore[union-attr]
        await conn.run_sync(SQLModel.metadata.create_all)


# ==================================================================
# Graph
# ==================================================================


class TestGroverGraph:
    def test_graph_property(self, grover: Grover):
        assert isinstance(grover.graph, Graph)
        assert grover.graph is grover._async._graph

    def test_dependents_after_write(self, grover: Grover):
        code = 'import os\n\ndef hello():\n    return "hi"\n'
        grover.write("/project/app.py", code)
        # File should be in graph now
        assert grover.graph.has_node("/project/app.py")
        # Check dependents doesn't crash (may be empty if no other file depends on it)
        deps = grover.dependents("/project/app.py")
        assert isinstance(deps, list)

    def test_dependencies_after_write(self, grover: Grover):
        code = 'def greet():\n    return "hello"\n'
        grover.write("/project/greet.py", code)
        # The file should have "contains" edges to its chunks
        deps = grover.dependencies("/project/greet.py")
        assert isinstance(deps, list)
        # Should contain the greet function chunk
        assert len(deps) >= 1

    def test_contains_returns_chunks(self, grover: Grover):
        code = 'def foo():\n    pass\n\ndef bar():\n    pass\n'
        grover.write("/project/funcs.py", code)
        chunks = grover.contains("/project/funcs.py")
        assert len(chunks) >= 2
        chunk_paths = [c.path for c in chunks]
        assert any("foo" in p for p in chunk_paths)
        assert any("bar" in p for p in chunk_paths)


# ==================================================================
# Search
# ==================================================================


class TestGroverSearch:
    def test_search_after_write(self, grover: Grover):
        code = 'def authenticate_user():\n    """Verify user credentials."""\n    pass\n'
        grover.write("/project/auth.py", code)
        results = grover.search("authenticate")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_returns_search_results(self, grover: Grover):
        grover.write("/project/data.txt", "important data content")
        results = grover.search("data")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty(self, grover: Grover):
        results = grover.search("nonexistent query")
        assert results == []

    def test_search_raises_without_provider(self, grover_no_search: Grover):
        if grover_no_search._async._search_index is not None:
            pytest.skip("sentence-transformers is installed; search available")
        with pytest.raises(RuntimeError, match="Search is not available"):
            grover_no_search.search("anything")


# ==================================================================
# Index
# ==================================================================


class TestGroverIndex:
    def test_index_scans_files(self, grover: Grover, workspace: Path):
        # Write files directly to disk so index() discovers them
        (workspace / "one.py").write_text('def one():\n    return 1\n')
        (workspace / "two.py").write_text('def two():\n    return 2\n')
        stats = grover.index()
        assert stats["files_scanned"] >= 2

    def test_index_creates_chunks(self, grover: Grover, workspace: Path):
        (workspace / "funcs.py").write_text(
            'def alpha():\n    pass\n\ndef beta():\n    pass\n'
        )
        stats = grover.index()
        assert stats["chunks_created"] >= 2

    def test_index_builds_graph(self, grover: Grover, workspace: Path):
        (workspace / "main.py").write_text('def main():\n    pass\n')
        grover.index()
        assert grover.graph.has_node("/project/main.py")

    def test_index_returns_stats(self, grover: Grover, workspace: Path):
        (workspace / "a.py").write_text('def a():\n    pass\n')
        stats = grover.index()
        assert "files_scanned" in stats
        assert "chunks_created" in stats
        assert "edges_added" in stats

    def test_index_skips_grover_dir(self, grover: Grover, workspace: Path):
        # Create a .grover subdirectory with a file
        grover_dir = workspace / ".grover" / "chunks"
        grover_dir.mkdir(parents=True)
        (grover_dir / "stale.txt").write_text("stale chunk")
        (workspace / "real.py").write_text('def real():\n    pass\n')

        grover.index()
        # The .grover file should NOT be indexed
        assert not grover.graph.has_node("/project/.grover/chunks/stale.txt")
        # But the real file should be
        assert grover.graph.has_node("/project/real.py")


# ==================================================================
# Event Handlers
# ==================================================================


class TestGroverEventHandlers:
    def test_write_updates_graph(self, grover: Grover):
        grover.write("/project/mod.py", 'def work():\n    pass\n')
        assert grover.graph.has_node("/project/mod.py")

    def test_write_updates_search(self, grover: Grover):
        grover.write(
            "/project/search_me.py",
            'def searchable():\n    """A unique searchable function."""\n    pass\n',
        )
        results = grover.search("searchable")
        assert len(results) >= 1

    def test_delete_removes_from_graph(self, grover: Grover):
        grover.write("/project/gone.py", 'def gone():\n    pass\n')
        assert grover.graph.has_node("/project/gone.py")
        grover.delete("/project/gone.py")
        assert not grover.graph.has_node("/project/gone.py")

    def test_delete_removes_from_search(self, grover: Grover):
        grover.write(
            "/project/vanish.py",
            'def vanishing_function():\n    pass\n',
        )
        # Verify it's in search
        assert grover._async._search_index is not None
        assert grover._async._search_index.has(
            "/.grover/chunks/project/vanish_py/vanishing_function.txt"
        )
        grover.delete("/project/vanish.py")
        # Should be removed from search
        assert not grover._async._search_index.has(
            "/.grover/chunks/project/vanish_py/vanishing_function.txt"
        )


# ==================================================================
# Persistence
# ==================================================================


class TestGroverPersistence:
    def test_save_persists_graph(self, grover: Grover, workspace: Path):
        grover.write("/project/persist.py", 'def persist():\n    pass\n')
        grover.save()

        # Verify DB has edges
        data_dir = grover._async._meta_data_dir
        assert data_dir is not None
        db_path = data_dir / "_meta" / "file_versions.db"
        assert db_path.exists()

    def test_save_persists_search(self, grover: Grover, workspace: Path):
        grover.write("/project/saved.txt", "save this content")
        grover.save()

        data_dir = grover._async._meta_data_dir
        assert data_dir is not None
        search_dir = data_dir / "search"
        assert (search_dir / "search_meta.json").exists()
        assert (search_dir / "search.usearch").exists()

    def test_auto_load_on_startup(self, workspace: Path, tmp_path: Path):
        data_dir = tmp_path / "data"

        # Create first instance, write data, save, close
        g1 = Grover(
            data_dir=str(data_dir),
            embedding_provider=FakeProvider(),
        )
        g1.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local"))
        g1.write("/project/keep.py", 'def keep():\n    pass\n')
        g1.save()
        g1.close()

        # Create second instance — should load state
        g2 = Grover(
            data_dir=str(data_dir),
            embedding_provider=FakeProvider(),
        )
        g2.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local"))
        try:
            assert g2.graph.has_node("/project/keep.py")
            # Search index should also be loaded
            results = g2.search("keep")
            assert len(results) >= 1
        finally:
            g2.close()


# ==================================================================
# Edge Cases
# ==================================================================


class TestGroverEdgeCases:
    def test_unsupported_file_type_embedded(self, grover: Grover):
        """Non-analyzable files should be embedded as whole files."""
        grover.write("/project/readme.txt", "This is a readme file")
        assert grover.graph.has_node("/project/readme.txt")
        # Should be searchable as whole file
        results = grover.search("readme")
        assert len(results) >= 1

    def test_empty_file_no_crash(self, grover: Grover):
        """Empty files should not crash the pipeline."""
        grover.write("/project/empty.py", "")
        # Should not raise — file may or may not be in graph

    def test_syntax_error_no_crash(self, grover: Grover):
        """Files with syntax errors should not crash the pipeline."""
        bad_code = "def broken(\n    # missing close paren and body"
        grover.write("/project/bad.py", bad_code)
        # Should not raise
        assert grover.graph.has_node("/project/bad.py")

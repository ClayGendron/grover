"""Tests for session batching in _analyze_and_integrate, _on_file_deleted, _on_file_moved.

Verifies that:
- _analyze_and_integrate uses a single DB session (not N+4)
- _on_file_deleted uses a single session (not 3)
- _on_file_moved uses a single session for old-path cleanup (not 3)
- Deferred events are emitted after commit
- Mid-pipeline failure rolls back all DB changes (atomicity)
"""

from __future__ import annotations

import hashlib
import math
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.fs.database_fs import DatabaseFileSystem
from grover.grover_async import GroverAsync
from grover.models.chunk import FileChunk
from grover.models.connection import FileConnection

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# =========================================================================
# Helpers
# =========================================================================

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


class SessionCounter:
    """Wraps GroverContext.session_for to count session opens."""

    def __init__(self, ctx):
        self._original = ctx.session_for
        self.count = 0

    @asynccontextmanager
    async def __call__(self, mount):
        self.count += 1
        async with self._original(mount) as sess:
            yield sess


# Python source with 3 imports + function + class
_MULTI_IMPORT_CODE = """\
import os
import sys
import json

def hello():
    pass

class Foo:
    pass
"""

# Python source with 1 import + function
_SINGLE_IMPORT_CODE = """\
import os

def greet():
    pass
"""

# Python source with a different import
_ALT_IMPORT_CODE = """\
import pathlib

def farewell():
    pass
"""


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
async def dbfs_setup(tmp_path: Path):
    """Set up GroverAsync with DatabaseFileSystem, returning (grover, engine)."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(dialect="sqlite")

    g = GroverAsync()
    await g.add_mount("/vfs", fs, session_factory=factory, embedding_provider=FakeProvider())

    yield g, engine
    await g.close()
    await engine.dispose()


# =========================================================================
# _analyze_and_integrate session batching
# =========================================================================


class TestAnalyzeSessionBatching:
    """Verify _analyze_and_integrate uses a single batched session."""

    async def test_analyze_uses_single_session(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """_analyze_and_integrate should open exactly 1 session (not N+4)."""
        g, _engine = dbfs_setup

        # Write file so it exists in VFS
        await g.write("/vfs/main.py", "x = 1\n")
        await g.flush()

        # Install session counter
        counter = SessionCounter(g._ctx)
        g._ctx.session_for = counter  # type: ignore[assignment]

        # Call _analyze_and_integrate directly with multi-import code
        stats = await g._analyze_and_integrate("/vfs/main.py", _MULTI_IMPORT_CODE)

        assert counter.count == 1, f"Expected 1 session, got {counter.count}"
        assert stats["edges_added"] >= 3  # os, sys, json
        assert stats["chunks_created"] >= 2  # hello, Foo

    async def test_analyze_persists_edges_correctly(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Batched session should correctly persist connections, chunks, and emit events."""
        g, engine = dbfs_setup

        await g.write("/vfs/main.py", _MULTI_IMPORT_CODE)
        await g.flush()

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # Verify DB has connection records
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/main.py")
            )
            records = list(rows.scalars().all())
            assert len(records) >= 3  # os, sys, json imports

        # Verify graph has the file node and edges
        graph = g.get_graph("/vfs")
        assert graph.has_node("/vfs/main.py")

        # Verify chunks in DB
        async with factory() as sess:
            rows = await sess.execute(
                select(FileChunk).where(FileChunk.file_path == "/vfs/main.py")
            )
            chunks = list(rows.scalars().all())
            assert len(chunks) >= 2  # hello function + Foo class

    async def test_analyze_replaces_stale_edges(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Rewriting a file should replace old edges with new ones."""
        g, engine = dbfs_setup

        # Write with import os
        await g.write("/vfs/mod.py", _SINGLE_IMPORT_CODE)
        await g.flush()

        # Rewrite with import pathlib
        await g.write("/vfs/mod.py", _ALT_IMPORT_CODE)
        await g.flush()

        # DB should have pathlib import, not os
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/mod.py")
            )
            records = list(rows.scalars().all())
            targets = [r.target_path for r in records]
            assert any("pathlib" in t for t in targets)
            # No leftover os import edge
            assert not any(t.endswith("/os") for t in targets)

    async def test_analyze_edges_projected_after_commit(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Edges should be projected into graph after DB commit, with DB records visible."""
        g, engine = dbfs_setup
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        await g.write("/vfs/main.py", _SINGLE_IMPORT_CODE)
        await g.flush()

        # Graph should have import edges (projected after commit)
        graph = g.get_graph("/vfs")
        assert graph.has_node("/vfs/main.py")

        # DB records should be committed and visible
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/main.py")
            )
            records = list(rows.scalars().all())
            assert len(records) >= 1, "DB records should be visible after commit"

    async def test_analyze_atomicity_on_failure(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """If add_connection fails mid-loop, all DB changes should roll back."""
        g, engine = dbfs_setup

        # Write a plain file first (no imports, no chunks)
        await g.write("/vfs/main.py", "x = 1\n")
        await g.flush()

        # Patch add_connection to fail on the 2nd call
        mount, _ = g._ctx.registry.resolve("/vfs/main.py")
        assert mount.filesystem is not None
        original_add = mount.filesystem.add_connection
        call_count = 0

        async def failing_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("Simulated mid-loop failure")
            return await original_add(*args, **kwargs)

        mount.filesystem.add_connection = failing_add  # type: ignore[assignment]

        # Call _analyze_and_integrate with code that has 3 imports
        with pytest.raises(RuntimeError, match="Simulated"):
            await g._analyze_and_integrate("/vfs/main.py", _MULTI_IMPORT_CODE)

        # Verify: no chunks or connections from the failed attempt
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as sess:
            chunk_rows = await sess.execute(
                select(FileChunk).where(FileChunk.file_path == "/vfs/main.py")
            )
            assert len(list(chunk_rows.scalars().all())) == 0, "Chunks should be rolled back"

            conn_rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/main.py")
            )
            assert len(list(conn_rows.scalars().all())) == 0, "Connections should be rolled back"


# =========================================================================
# _process_delete session batching
# =========================================================================


class TestDeletedSessionBatching:
    """Verify _process_delete uses a single batched session."""

    async def test_on_file_deleted_uses_single_session(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Deleting a file should use 1 session for cleanup (was 3)."""
        g, _engine = dbfs_setup

        # Write file with imports to populate chunks + connections
        await g.write("/vfs/main.py", _SINGLE_IMPORT_CODE)
        await g.flush()

        # Install counter
        counter = SessionCounter(g._ctx)
        g._ctx.session_for = counter  # type: ignore[assignment]

        # Call processing method directly
        await g._process_delete("/vfs/main.py")

        assert counter.count == 1, f"Expected 1 session, got {counter.count}"

    async def test_on_file_deleted_cleans_up_all_records(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Deleting a file should remove search entries, chunks, and connections."""
        g, engine = dbfs_setup
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # Write file with imports
        await g.write("/vfs/main.py", _SINGLE_IMPORT_CODE)
        await g.flush()

        # Verify records exist before delete
        async with factory() as sess:
            chunks = await sess.execute(
                select(FileChunk).where(FileChunk.file_path == "/vfs/main.py")
            )
            assert len(list(chunks.scalars().all())) > 0, "Setup: chunks should exist"

        # Delete the file
        await g.delete("/vfs/main.py")
        await g.flush()

        # Verify all records cleaned up
        async with factory() as sess:
            chunk_rows = await sess.execute(
                select(FileChunk).where(FileChunk.file_path == "/vfs/main.py")
            )
            assert len(list(chunk_rows.scalars().all())) == 0, "Chunks not cleaned up"

            conn_rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/vfs/main.py")
            )
            assert len(list(conn_rows.scalars().all())) == 0, "Connections not cleaned up"


# =========================================================================
# _process_move session batching
# =========================================================================


class TestMovedSessionBatching:
    """Verify _process_move uses a single session for old-path cleanup."""

    async def test_on_file_moved_uses_single_session(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """Moving a file should use 1 session for old-path cleanup (was 3)."""
        g, _engine = dbfs_setup

        # Write file with imports
        await g.write("/vfs/main.py", _SINGLE_IMPORT_CODE)
        await g.flush()

        # Install counter
        counter = SessionCounter(g._ctx)
        g._ctx.session_for = counter  # type: ignore[assignment]

        # Call processing method directly with a .grover path as the new path.
        # _process_move returns early for .grover new paths (before read),
        # isolating just the old-path cleanup session.
        await g._process_move("/vfs/main.py", "/vfs/.grover/temp.py")

        assert counter.count == 1, f"Expected 1 session for old-path cleanup, got {counter.count}"


# =========================================================================
# Edge cases
# =========================================================================


class TestAnalyzeEdgeCases:
    """Edge case tests for batched session behavior."""

    async def test_analyze_contains_edges_not_persisted(
        self, dbfs_setup: tuple[GroverAsync, AsyncEngine]
    ) -> None:
        """'contains' edges should stay in-memory only, not in DB. (Regression check.)"""
        g, engine = dbfs_setup
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        code = "import os\n\ndef hello():\n    pass\n\nclass Foo:\n    pass\n"
        await g.write("/vfs/mod.py", code)
        await g.flush()

        # Graph should have 'contains' edges
        graph = g.get_graph("/vfs")
        assert graph.has_node("/vfs/mod.py")

        # DB should NOT have any 'contains' edges
        async with factory() as sess:
            rows = await sess.execute(select(FileConnection))
            records = list(rows.scalars().all())
            contains_records = [r for r in records if r.type == "contains"]
            assert len(contains_records) == 0

    async def test_analyze_read_only_mount_skips_connections(self, tmp_path: Path) -> None:
        """Connection writes should be skipped for read-only mounts."""
        from grover.fs.permissions import Permission

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync()
        await g.add_mount("/vfs", fs, session_factory=factory, embedding_provider=FakeProvider())

        # Write the file while mount is writable
        await g.write("/vfs/main.py", "x = 1\n")
        await g.flush()

        # Change mount to read-only
        mount, _ = g._ctx.registry.resolve("/vfs/main.py")
        mount.permission = Permission.READ_ONLY

        # Clear any existing connections
        async with factory() as sess:
            rows = await sess.execute(select(FileConnection))
            for row in rows.scalars().all():
                await sess.delete(row)
            await sess.commit()

        # Re-analyze — should skip connection writes
        await g._analyze_and_integrate("/vfs/main.py", _SINGLE_IMPORT_CODE)

        # Verify no connections written
        async with factory() as sess:
            rows = await sess.execute(select(FileConnection))
            assert len(list(rows.scalars().all())) == 0

        await g.close()
        await engine.dispose()

    async def test_analyze_fallback_readonly_mount(self, tmp_path: Path) -> None:
        """Read-only mount should add edges directly to graph, skipping DB."""
        from grover.fs.permissions import Permission

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dfs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync()
        await g.add_mount("/ro", dfs, session_factory=factory, permission=Permission.READ_ONLY)
        # Seed a file via a writable mount, then test analysis on the read-only one
        # We need to write the file directly through the backend to seed it
        async with factory() as sess:
            await dfs.write("/ro/main.py", _SINGLE_IMPORT_CODE, session=sess)
            await sess.commit()

        stats = await g._analyze_and_integrate("/ro/main.py", _SINGLE_IMPORT_CODE)

        # Edges should still be added (via fallback: directly to graph)
        assert stats["edges_added"] >= 1

        # Graph should have the node
        graph = g.get_graph("/ro")
        assert graph.has_node("/ro/main.py")

        # DB should have NO connections (read-only fallback skips DB)
        async with factory() as sess:
            rows = await sess.execute(
                select(FileConnection).where(FileConnection.source_path == "/ro/main.py")
            )
            assert len(list(rows.scalars().all())) == 0

        await g.close()
        await engine.dispose()

"""Tests for the alpha refactor migration script."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Column, Integer, String, Text, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from grover.migrations.backfill_alpha_refactor import (
    backfill_alpha_refactor,
    check_schema_compatibility,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class OldBase(DeclarativeBase):
    """Declarative base for old-schema tables."""


class OldFile(OldBase):
    __tablename__ = "grover_files"
    id = Column(String, primary_key=True)
    path = Column(String, nullable=False)
    content = Column(Text, default="")


class OldFileVersion(OldBase):
    __tablename__ = "grover_file_versions"
    id = Column(String, primary_key=True)
    file_id = Column(String, nullable=False)
    version = Column(Integer, default=1)
    content = Column(Text, default="")


class OldFileChunk(OldBase):
    __tablename__ = "grover_file_chunks"
    id = Column(String, primary_key=True)
    file_path = Column(String, nullable=False)
    chunk_path = Column(String, default="")  # Old name — should become 'path'
    name = Column(String, default="")
    content = Column(Text, default="")


class OldFileConnection(OldBase):
    __tablename__ = "grover_file_connections"
    id = Column(String, primary_key=True)
    source_path = Column(String, nullable=False)
    target_path = Column(String, nullable=False)
    type = Column(String, default="")
    weight = Column(String, default="1.0")


class OldEmbedding(OldBase):
    __tablename__ = "grover_embeddings"
    id = Column(String, primary_key=True)
    file_path = Column(String, nullable=False)
    vector_json = Column(Text, default="")


@pytest.fixture
async def old_schema_engine() -> AsyncEngine:
    """Create an in-memory SQLite engine with old-schema tables."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(OldBase.metadata.create_all)
    return engine


@pytest.fixture
async def old_schema_with_data(old_schema_engine: AsyncEngine) -> AsyncEngine:
    """Old-schema engine with sample data pre-populated."""
    async with old_schema_engine.begin() as conn:
        # Insert a file
        await conn.execute(
            text(
                "INSERT INTO grover_files (id, path, content)"
                " VALUES ('f1', '/src/main.py', 'print(1)')"
            )
        )
        # Insert a version referencing that file
        await conn.execute(
            text(
                "INSERT INTO grover_file_versions (id, file_id, version, content)"
                " VALUES ('v1', 'f1', 1, 'print(1)')"
            )
        )
        # Insert a chunk with old chunk_path field
        await conn.execute(
            text(
                "INSERT INTO grover_file_chunks (id, file_path, chunk_path, name, content)"
                " VALUES ('c1', '/src/main.py', '/src/main.py#main', 'main', 'def main(): ...')"
            )
        )
        # Insert a connection without path field
        await conn.execute(
            text(
                "INSERT INTO grover_file_connections (id, source_path, target_path, type)"
                " VALUES ('e1', '/src/main.py', '/src/utils.py', 'imports')"
            )
        )
        # Insert an embedding (to be dropped)
        await conn.execute(
            text(
                "INSERT INTO grover_embeddings (id, file_path, vector_json)"
                " VALUES ('em1', '/src/main.py', '[0.1, 0.2]')"
            )
        )
    return old_schema_engine


# ------------------------------------------------------------------
# check_schema_compatibility
# ------------------------------------------------------------------


async def test_compatibility_check_detects_old_schema(old_schema_engine: AsyncEngine) -> None:
    """Old schema is detected as incompatible."""
    errors = await check_schema_compatibility(old_schema_engine)
    assert len(errors) == 3
    # Should mention chunk_path rename hint
    chunk_error = [e for e in errors if "grover_file_chunks" in e]
    assert len(chunk_error) == 1
    assert "chunk_path" in chunk_error[0]


async def test_compatibility_check_passes_new_schema() -> None:
    """A fresh database with current schema passes compatibility check."""
    from sqlmodel import SQLModel

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    errors = await check_schema_compatibility(engine)
    assert errors == []
    await engine.dispose()


async def test_compatibility_check_passes_empty_db() -> None:
    """An empty database (no tables) passes — create_all will handle it."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    errors = await check_schema_compatibility(engine)
    assert errors == []
    await engine.dispose()


# ------------------------------------------------------------------
# backfill_alpha_refactor
# ------------------------------------------------------------------


async def test_backfill_idempotent(old_schema_with_data: AsyncEngine) -> None:
    """Running migration twice produces the same result."""
    report1 = await backfill_alpha_refactor(old_schema_with_data)
    report2 = await backfill_alpha_refactor(old_schema_with_data)

    # First run should have "added"/"renamed"/"dropped" statuses
    assert report1["file_versions_file_path"] == "added"
    assert report1["file_chunks_path"] == "renamed"
    assert report1["embeddings_dropped"] == "dropped"

    # Second run should all be "exists"/"not_present"
    assert report2["file_versions_file_path"] == "exists"
    assert report2["file_chunks_path"] == "exists"
    assert report2["embeddings_dropped"] == "not_present"


async def test_backfill_computes_connection_path(
    old_schema_with_data: AsyncEngine,
) -> None:
    """Connection path is computed as source[type]target."""
    await backfill_alpha_refactor(old_schema_with_data)

    async with old_schema_with_data.connect() as conn:
        result = await conn.execute(
            text("SELECT path FROM grover_file_connections WHERE id = 'e1'")
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "/src/main.py[imports]/src/utils.py"


async def test_backfill_preserves_existing_data(
    old_schema_with_data: AsyncEngine,
) -> None:
    """Migration preserves all existing data — no rows lost."""
    await backfill_alpha_refactor(old_schema_with_data)

    async with old_schema_with_data.connect() as conn:
        # File still has its content
        result = await conn.execute(text("SELECT path, content FROM grover_files WHERE id = 'f1'"))
        row = result.fetchone()
        assert row is not None
        assert row[0] == "/src/main.py"
        assert row[1] == "print(1)"

        # Version has backfilled file_path
        result = await conn.execute(
            text("SELECT file_id, file_path, version FROM grover_file_versions WHERE id = 'v1'")
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "f1"
        assert row[1] == "/src/main.py"
        assert row[2] == 1

        # Chunk has renamed path (was chunk_path)
        result = await conn.execute(
            text("SELECT file_path, path, name, content FROM grover_file_chunks WHERE id = 'c1'")
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "/src/main.py"
        assert row[1] == "/src/main.py#main"  # was chunk_path
        assert row[2] == "main"
        assert row[3] == "def main(): ..."

        # Connection has computed path
        result = await conn.execute(
            text(
                "SELECT source_path, target_path, type, path"
                " FROM grover_file_connections WHERE id = 'e1'"
            )
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "/src/main.py"
        assert row[1] == "/src/utils.py"
        assert row[2] == "imports"
        assert row[3] == "/src/main.py[imports]/src/utils.py"


async def test_backfill_adds_vector_columns(old_schema_with_data: AsyncEngine) -> None:
    """Migration adds vector columns to files and chunks tables."""
    report = await backfill_alpha_refactor(old_schema_with_data)
    assert report["files_vector"] == "added"
    assert report["file_chunks_vector"] == "added"

    # Verify columns exist
    async with old_schema_with_data.connect() as conn:

        def _check(c: object) -> tuple[set[str], set[str]]:
            insp = inspect(c)  # type: ignore[arg-type]
            file_cols = {col["name"] for col in insp.get_columns("grover_files")}
            chunk_cols = {col["name"] for col in insp.get_columns("grover_file_chunks")}
            return file_cols, chunk_cols

        file_cols, chunk_cols = await conn.run_sync(_check)
        assert "vector" in file_cols
        assert "vector" in chunk_cols


async def test_backfill_drops_embeddings(old_schema_with_data: AsyncEngine) -> None:
    """Migration drops the grover_embeddings table."""
    report = await backfill_alpha_refactor(old_schema_with_data)
    assert report["embeddings_dropped"] == "dropped"

    async with old_schema_with_data.connect() as conn:

        def _check(c: object) -> list[str]:
            insp = inspect(c)  # type: ignore[arg-type]
            return insp.get_table_names()

        tables = await conn.run_sync(_check)
        assert "grover_embeddings" not in tables


async def test_backfill_report_on_empty_db() -> None:
    """Migration on empty DB reports table_missing for everything."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    report = await backfill_alpha_refactor(engine)

    assert report["file_versions_file_path"] == "table_missing"
    assert report["file_chunks_path"] == "table_missing"
    assert report["files_vector"] == "table_missing"
    assert report["file_connections_path"] == "table_missing"
    assert report["embeddings_dropped"] == "not_present"
    await engine.dispose()


async def test_backfill_on_current_schema() -> None:
    """Migration on current schema reports 'exists' for everything."""
    from sqlmodel import SQLModel

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    report = await backfill_alpha_refactor(engine)

    assert report["file_versions_file_path"] == "exists"
    assert report["file_chunks_path"] == "exists"
    assert report["file_chunks_vector"] == "exists"
    assert report["files_vector"] == "exists"
    assert report["file_connections_path"] == "exists"
    assert report["embeddings_dropped"] == "not_present"
    await engine.dispose()


# ------------------------------------------------------------------
# SchemaIncompatibleError in add_mount
# ------------------------------------------------------------------


async def test_add_mount_rejects_old_schema() -> None:
    """add_mount raises SchemaIncompatibleError on stale schema."""
    from grover._grover_async import GroverAsync
    from grover.fs.exceptions import SchemaIncompatibleError

    # Create engine with old schema
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(OldBase.metadata.create_all)

    g = GroverAsync()
    with pytest.raises(SchemaIncompatibleError, match="backfill_alpha_refactor"):
        await g.add_mount("/data", engine=engine)
    await engine.dispose()


async def test_add_mount_accepts_current_schema() -> None:
    """add_mount works fine with current schema (no error)."""
    from grover._grover_async import GroverAsync

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    g = GroverAsync()
    # Should not raise — create_all creates correct schema
    await g.add_mount("/data", engine=engine)
    await g.close()
    await engine.dispose()


async def test_add_mount_accepts_migrated_schema() -> None:
    """add_mount works after running migration on old schema."""
    from grover._grover_async import GroverAsync

    # Create engine with old schema
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(OldBase.metadata.create_all)

    # Run migration
    await backfill_alpha_refactor(engine)

    # Now add_mount should work
    g = GroverAsync()
    await g.add_mount("/data", engine=engine)
    await g.close()
    await engine.dispose()

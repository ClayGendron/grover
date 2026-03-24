"""Tests for external edit detection in versioned filesystems."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.backends.database import DatabaseFileSystem
from grover.models.database.file import FileModel
from grover.util.content import compute_content_hash
from grover.util.operations import check_external_edit


async def _make_fs():
    """Create a stateless DatabaseFileSystem + session factory + engine."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem()
    return fs, factory, engine


async def _simulate_external_edit(
    session: AsyncSession,
    path: str,
    new_content: str,
) -> None:
    """Simulate an external edit by directly modifying FileModel.content in the DB.

    This mimics what happens when a tool outside Grover (IDE, git, shell, etc.)
    modifies file content. The content changes but content_hash does NOT, which
    is exactly how the mismatch is detected.
    """
    result = await session.execute(select(FileModel).where(FileModel.path == path))
    file = result.scalar_one()
    file.content = new_content
    await session.flush()


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


class TestExternalEditDetection:
    async def test_edit_after_external_edit_creates_synthetic_version(self):
        """External edit between write and edit inserts a synthetic version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "hello world\n", session=session)
            await _simulate_external_edit(session, "/app.py", "hello world!!!\n")
            # Edit must match the ACTUAL (external) content
            result = await fs.edit("/app.py", "!!!", "???", session=session)

            assert result.success is True
            # v1=grover, v2=external, v3=grover edit
            assert result.file.current_version == 3
        await engine.dispose()

    async def test_no_external_edit_no_synthetic_version(self):
        """Normal write without external modification creates no extra version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "v1\n", session=session)
            result = await fs.write("/app.py", "v2\n", session=session)

            assert result.success is True
            assert result.file.current_version == 2
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestExternalEditEdgeCases:
    async def test_no_content_hash_skips_detection(self):
        """Files with no content_hash skip external edit detection."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            # Create a file, then clear its content_hash to simulate
            # a file record that never had a hash tracked
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(select(FileModel).where(FileModel.path == "/app.py"))
            file = file_result.scalar_one()
            file.content_hash = None
            await session.flush()

            # Simulate external edit
            await _simulate_external_edit(session, "/app.py", "external\n")

            # Write again — should NOT create synthetic version
            result = await fs.write("/app.py", "final\n", session=session)
            # Version goes from 1 → 2 (no synthetic v2 inserted)
            assert result.file.current_version == 2
        await engine.dispose()


# ---------------------------------------------------------------------------
# Unit test for check_external_edit() directly
# ---------------------------------------------------------------------------


class TestCheckExternalEditUnit:
    async def test_matching_hash_returns_false(self):
        """No divergence — function returns False without modifying file."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(select(FileModel).where(FileModel.path == "/app.py"))
            file = file_result.scalar_one()
            original_version = file.current_version

            detected = await check_external_edit(
                file,
                "content\n",
                session,
                versioning=fs.version_provider,
            )
            assert detected is False
            assert file.current_version == original_version
        await engine.dispose()

    async def test_mismatching_hash_returns_true(self):
        """Divergence detected — function returns True and increments version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(select(FileModel).where(FileModel.path == "/app.py"))
            file = file_result.scalar_one()
            original_version = file.current_version

            detected = await check_external_edit(
                file,
                "different content\n",
                session,
                versioning=fs.version_provider,
            )
            assert detected is True
            assert file.current_version == original_version + 1

            # Verify hash was updated to match the external content
            expected_hash = compute_content_hash("different content\n")[0]
            assert file.content_hash == expected_hash
        await engine.dispose()

    async def test_none_content_hash_returns_false(self):
        """File with no content_hash — returns False."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(select(FileModel).where(FileModel.path == "/app.py"))
            file = file_result.scalar_one()
            file.content_hash = None

            detected = await check_external_edit(
                file,
                "anything\n",
                session,
                versioning=fs.version_provider,
            )
            assert detected is False
        await engine.dispose()

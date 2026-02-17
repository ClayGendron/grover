"""Tests for external edit detection in versioned filesystems."""

from __future__ import annotations

import hashlib

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.metadata import compute_content_hash
from grover.fs.operations import check_external_edit
from grover.models.files import File, FileVersion


async def _make_fs():
    """Create a stateless DatabaseFileSystem + session factory + engine."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(dialect="sqlite")
    return fs, factory, engine


async def _simulate_external_edit(
    session: AsyncSession,
    path: str,
    new_content: str,
) -> None:
    """Simulate an external edit by directly modifying File.content in the DB.

    This mimics what happens when a tool outside Grover (IDE, git, shell, etc.)
    modifies file content. The content changes but content_hash does NOT, which
    is exactly how the mismatch is detected.
    """
    result = await session.execute(select(File).where(File.path == path))
    file = result.scalar_one()
    file.content = new_content
    await session.flush()


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


class TestExternalEditDetection:
    async def test_write_after_external_edit_creates_synthetic_version(self):
        """External edit between writes inserts a synthetic version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "hello\n", session=session)
            await _simulate_external_edit(session, "/app.py", "hello world\n")
            result = await fs.write("/app.py", "goodbye\n", session=session)

            assert result.success is True
            # v1=grover, v2=external, v3=grover
            assert result.version == 3

            versions = await fs.list_versions("/app.py", session=session)
            assert len(versions.versions) == 3
        await engine.dispose()

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
            assert result.version == 3
        await engine.dispose()

    async def test_no_external_edit_no_synthetic_version(self):
        """Normal write without external modification creates no extra version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "v1\n", session=session)
            result = await fs.write("/app.py", "v2\n", session=session)

            assert result.success is True
            assert result.version == 2

            versions = await fs.list_versions("/app.py", session=session)
            assert len(versions.versions) == 2
        await engine.dispose()


# ---------------------------------------------------------------------------
# Version reconstruction
# ---------------------------------------------------------------------------


class TestVersionReconstruction:
    async def test_all_versions_reconstruct_after_external_edit(self):
        """Every version reconstructs correctly when external edit is present."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "original\n", session=session)
            await _simulate_external_edit(session, "/app.py", "externally modified\n")
            await fs.write("/app.py", "final content\n", session=session)

            # v1 = original
            vc1 = await fs.get_version_content("/app.py", 1, session=session)
            assert vc1.success is True
            assert vc1.content == "original\n"

            # v2 = external
            vc2 = await fs.get_version_content("/app.py", 2, session=session)
            assert vc2.success is True
            assert vc2.content == "externally modified\n"

            # v3 = final
            vc3 = await fs.get_version_content("/app.py", 3, session=session)
            assert vc3.success is True
            assert vc3.content == "final content\n"
        await engine.dispose()

    async def test_hash_verification_passes_full_chain(self):
        """5+ versions with external edit — all reconstruct without ConsistencyError."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "v1\n", session=session)
            await fs.edit("/app.py", "v1", "v2", session=session)
            await _simulate_external_edit(session, "/app.py", "external content\n")
            await fs.edit("/app.py", "external content", "v4", session=session)
            await fs.edit("/app.py", "v4", "v5", session=session)

            # All 5 versions should reconstruct without ConsistencyError
            for v in range(1, 6):
                vc = await fs.get_version_content("/app.py", v, session=session)
                assert vc.success is True, f"Version {v} failed: {vc.message}"
        await engine.dispose()

    async def test_multiple_grover_ops_after_external_edit(self):
        """External edit followed by multiple Grover writes all chain correctly."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "start\n", session=session)
            await _simulate_external_edit(session, "/app.py", "external\n")
            await fs.write("/app.py", "write1\n", session=session)
            await fs.edit("/app.py", "write1", "edit1", session=session)
            await fs.write("/app.py", "write2\n", session=session)

            # v1=start, v2=external, v3=write1, v4=edit1, v5=write2
            for v in range(1, 6):
                vc = await fs.get_version_content("/app.py", v, session=session)
                assert vc.success is True, f"Version {v} failed: {vc.message}"

            vc5 = await fs.get_version_content("/app.py", 5, session=session)
            assert vc5.content == "write2\n"
        await engine.dispose()


# ---------------------------------------------------------------------------
# Synthetic version properties
# ---------------------------------------------------------------------------


class TestSyntheticVersionProperties:
    async def test_external_version_is_snapshot(self):
        """Synthetic external version is stored as a full snapshot."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "original\n", session=session)
            await _simulate_external_edit(session, "/app.py", "external\n")
            await fs.write("/app.py", "final\n", session=session)

            # Query the version 2 record directly
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            ver_result = await session.execute(
                select(FileVersion).where(
                    FileVersion.file_id == file.id,
                    FileVersion.version == 2,
                )
            )
            v2 = ver_result.scalar_one()
            assert v2.is_snapshot is True
        await engine.dispose()

    async def test_external_version_created_by_marker(self):
        """Synthetic version has created_by='external'."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "original\n", session=session)
            await _simulate_external_edit(session, "/app.py", "external\n")
            await fs.write("/app.py", "final\n", session=session)

            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            ver_result = await session.execute(
                select(FileVersion).where(
                    FileVersion.file_id == file.id,
                    FileVersion.version == 2,
                )
            )
            v2 = ver_result.scalar_one()
            assert v2.created_by == "external"
        await engine.dispose()

    async def test_external_edit_hash_matches_external_content(self):
        """Synthetic version's content_hash matches the external content."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            external_content = "externally modified\n"
            await fs.write("/app.py", "original\n", session=session)
            await _simulate_external_edit(session, "/app.py", external_content)
            await fs.write("/app.py", "final\n", session=session)

            vc2 = await fs.get_version_content("/app.py", 2, session=session)
            assert vc2.content == external_content

            expected_hash = hashlib.sha256(external_content.encode()).hexdigest()
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            ver_result = await session.execute(
                select(FileVersion).where(
                    FileVersion.file_id == file.id,
                    FileVersion.version == 2,
                )
            )
            v2 = ver_result.scalar_one()
            assert v2.content_hash == expected_hash
        await engine.dispose()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestExternalEditEdgeCases:
    async def test_external_edit_of_empty_content(self):
        """External edit from empty string to non-empty is detected."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "\n", session=session)
            await _simulate_external_edit(session, "/app.py", "hello\n")
            result = await fs.write("/app.py", "final\n", session=session)

            assert result.version == 3
            vc2 = await fs.get_version_content("/app.py", 2, session=session)
            assert vc2.content == "hello\n"
        await engine.dispose()

    async def test_no_content_hash_skips_detection(self):
        """Files with no content_hash skip external edit detection."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            # Create a file, then clear its content_hash to simulate
            # a file record that never had a hash tracked
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            file.content_hash = None
            await session.flush()

            # Simulate external edit
            await _simulate_external_edit(session, "/app.py", "external\n")

            # Write again — should NOT create synthetic version
            result = await fs.write("/app.py", "final\n", session=session)
            # Version goes from 1 → 2 (no synthetic v2 inserted)
            assert result.version == 2
        await engine.dispose()

    async def test_edit_applies_to_actual_content_after_detection(self):
        """After external edit detection, edit applies to the actual (external) content."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "hello world\n", session=session)
            await _simulate_external_edit(session, "/app.py", "hello world!!!\n")

            # Edit must target the external content, not the old Grover content
            result = await fs.edit("/app.py", "!!!", "???", session=session)
            assert result.success is True

            read = await fs.read("/app.py", session=session)
            assert read.content == "hello world???\n"

            # Version chain: v1=grover, v2=external, v3=edit
            vc1 = await fs.get_version_content("/app.py", 1, session=session)
            assert vc1.content == "hello world\n"

            vc2 = await fs.get_version_content("/app.py", 2, session=session)
            assert vc2.content == "hello world!!!\n"

            vc3 = await fs.get_version_content("/app.py", 3, session=session)
            assert vc3.content == "hello world???\n"
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
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            original_version = file.current_version

            detected = await check_external_edit(
                file, "content\n", session, versioning=fs.versioning,
            )
            assert detected is False
            assert file.current_version == original_version
        await engine.dispose()

    async def test_mismatching_hash_returns_true(self):
        """Divergence detected — function returns True and increments version."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/app.py", "content\n", session=session)
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            original_version = file.current_version

            detected = await check_external_edit(
                file, "different content\n", session, versioning=fs.versioning,
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
            file_result = await session.execute(
                select(File).where(File.path == "/app.py")
            )
            file = file_result.scalar_one()
            file.content_hash = None

            detected = await check_external_edit(
                file, "anything\n", session, versioning=fs.versioning,
            )
            assert detected is False
        await engine.dispose()

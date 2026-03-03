"""Tests for version chain integrity verification."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.providers.versioning import SNAPSHOT_INTERVAL
from grover.models.version import FileVersion
from grover.types.operations import VerifyVersionResult, VersionChainError


async def _make_fs():
    """Create a stateless DatabaseFileSystem + session factory + engine."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(dialect="sqlite")
    return fs, factory, engine


class TestVerifyChainHealthy:
    """Healthy version chains should pass verification."""

    async def test_verify_chain_healthy(self):
        """Write file + 5 edits (6 versions). All should verify."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "version 1\n", session=session)
            for i in range(2, 7):
                await fs.write("/f.py", f"version {i}\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is True
            assert result.versions_checked == 6
            assert result.versions_passed == 6
            assert result.versions_failed == 0
            assert result.errors == []
        await engine.dispose()

    async def test_verify_chain_single_version(self):
        """Single write (1 snapshot). Should verify."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "only version\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is True
            assert result.versions_checked == 1
            assert result.versions_passed == 1
            assert result.errors == []
        await engine.dispose()

    async def test_verify_chain_no_versions(self):
        """File record with no version records should succeed with 0 checked."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            # Create a file, then delete all its version records manually
            await fs.write("/f.py", "content\n", session=session)
            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            # Delete all version records
            versions = (
                (
                    await session.execute(
                        select(FileVersion).where(FileVersion.file_id == file_rec.id)
                    )
                )
                .scalars()
                .all()
            )
            for v in versions:
                await session.delete(v)
            await session.flush()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is True
            assert result.versions_checked == 0
            assert result.message == "No versions to verify"
        await engine.dispose()

    async def test_verify_chain_across_snapshot_interval(self):
        """Write SNAPSHOT_INTERVAL + 1 edits. Second snapshot should also pass."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            for i in range(2, SNAPSHOT_INTERVAL + 2):
                await fs.write("/f.py", f"v{i}\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            total_versions = SNAPSHOT_INTERVAL + 1
            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is True
            assert result.versions_checked == total_versions
            assert result.versions_passed == total_versions
            assert result.errors == []
        await engine.dispose()


class TestVerifyChainCorrupted:
    """Corrupted version chains should be detected."""

    async def test_verify_chain_corrupted_diff(self):
        """Corrupt version 3's content. Version 3+ should fail."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "line 1\n", session=session)
            await fs.write("/f.py", "line 1\nline 2\n", session=session)
            await fs.write("/f.py", "line 1\nline 2\nline 3\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            # Corrupt version 2's diff content (it's a diff, not snapshot)
            v2_rec = (
                await session.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 2,
                    )
                )
            ).scalar_one()
            v2_rec.content = "garbage diff data\n"
            session.add(v2_rec)
            await session.flush()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is False
            assert result.versions_failed > 0
            # Version 1 (snapshot) should still pass
            failed_versions = {e.version for e in result.errors}
            assert 1 not in failed_versions
            assert 2 in failed_versions or 3 in failed_versions
        await engine.dispose()

    async def test_verify_chain_corrupted_snapshot(self):
        """Corrupt version 1's snapshot content. Dependent versions should fail.

        Use a multi-line file with line-count-changing corruption so the diff
        application also fails (hunk out of bounds) for downstream versions.
        """
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "line1\nline2\nline3\n", session=session)
            await fs.write("/f.py", "line1\nline2\nline3\nline4\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            # Corrupt snapshot to a single-line value — diff hunk ranges
            # will be out of bounds since the diff expects 3 source lines.
            v1_rec = (
                await session.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 1,
                    )
                )
            ).scalar_one()
            v1_rec.content = "X\n"
            session.add(v1_rec)
            await session.flush()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is False
            failed_versions = {e.version for e in result.errors}
            # Version 1 fails (hash mismatch) and version 2 fails
            # (reconstruction error due to hunk out of bounds)
            assert 1 in failed_versions
            assert 2 in failed_versions
        await engine.dispose()

    async def test_verify_chain_corrupted_hash(self):
        """Corrupt a version's content_hash (content is fine). Should detect."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "good content\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            # Corrupt the stored hash (content is correct)
            v1_rec = (
                await session.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 1,
                    )
                )
            ).scalar_one()
            v1_rec.content_hash = "0000000000000000000000000000000000000000000000000000000000000000"
            session.add(v1_rec)
            await session.flush()

            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is False
            assert result.versions_failed == 1
            assert result.errors[0].version == 1
            assert result.errors[0].error == "Content hash mismatch"
            assert result.errors[0].expected_hash == "0" * 64
            # actual_hash should be the real hash of "good content\n"
            expected_real = hashlib.sha256(b"good content\n").hexdigest()
            assert result.errors[0].actual_hash == expected_real
        await engine.dispose()

    async def test_verify_chain_reconstruction_error(self):
        """Delete snapshot record leaving only diffs. Error captured, not raised."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            await fs.write("/f.py", "v1\nv2\n", session=session)

            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/f.py"))
            ).scalar_one()

            # Delete version 1 (the snapshot), leaving only the diff
            v1_rec = (
                await session.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 1,
                    )
                )
            ).scalar_one()
            await session.delete(v1_rec)
            await session.flush()

            # Should NOT raise — errors are captured in the result
            result = await fs.version_provider.verify_chain(session, file_rec)

            assert result.success is False
            assert result.versions_checked == 1  # only v2 remains
            assert result.versions_failed == 1
            # The error should explain missing snapshot
            assert result.errors[0].version == 2
            err_msg = result.errors[0].error
            assert "No snapshot found" in err_msg or "Reconstruction failed" in err_msg
        await engine.dispose()


class TestBackendVerifyVersions:
    """Backend-level verify_versions / verify_all_versions tests."""

    async def test_backend_verify_versions_healthy(self):
        """DatabaseFileSystem.verify_versions on a healthy file."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/f.py", "v1\n", session=session)
            await fs.write("/f.py", "v2\n", session=session)

            result = await fs.verify_versions("/f.py", session=session)
            assert result.success is True
            assert result.versions_checked == 2
            assert result.versions_passed == 2
            assert result.path == "/f.py"
        await engine.dispose()

    async def test_backend_verify_versions_not_found(self):
        """verify_versions on a nonexistent path returns success=False."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            result = await fs.verify_versions("/nope.py", session=session)
            assert result.success is False
            assert "not found" in result.message.lower()
        await engine.dispose()

    async def test_backend_verify_all_versions_multiple_files(self):
        """verify_all_versions checks all non-deleted, non-directory files."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/a.py", "a\n", session=session)
            await fs.write("/b.py", "b\n", session=session)
            await fs.write("/c.py", "c\n", session=session)

            results = await fs.verify_all_versions(session=session)
            assert len(results) == 3
            assert all(r.success for r in results)
        await engine.dispose()

    async def test_backend_verify_all_versions_skips_directories(self):
        """verify_all_versions only checks files, not directories."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.mkdir("/mydir", session=session)
            await fs.write("/mydir/f.py", "content\n", session=session)

            results = await fs.verify_all_versions(session=session)
            # Only the file, not the directory
            assert len(results) == 1
            assert results[0].path == "/mydir/f.py"
        await engine.dispose()

    async def test_backend_verify_all_versions_with_corruption(self):
        """verify_all_versions reports mixed results when one file is corrupted."""
        fs, factory, engine = await _make_fs()
        async with factory() as session:
            await fs.write("/good.py", "good\n", session=session)
            await fs.write("/bad.py", "bad\n", session=session)

            # Corrupt /bad.py's hash
            file_rec = (
                await session.execute(select(fs.file_model).where(fs.file_model.path == "/bad.py"))
            ).scalar_one()
            v1_rec = (
                await session.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 1,
                    )
                )
            ).scalar_one()
            v1_rec.content_hash = "0" * 64
            session.add(v1_rec)
            await session.flush()

            results = await fs.verify_all_versions(session=session)
            assert len(results) == 2
            paths = {r.path: r for r in results}
            assert paths["/good.py"].success is True
            assert paths["/bad.py"].success is False
        await engine.dispose()


class TestVerifyVersionResultFields:
    """Test VerifyVersionResult and VersionChainError types."""

    def test_verify_version_result_fields(self):
        """VerifyVersionResult has expected fields with correct defaults."""
        result = VerifyVersionResult()
        assert result.path == ""
        assert result.success is True
        assert result.versions_checked == 0
        assert result.versions_passed == 0
        assert result.versions_failed == 0
        assert result.errors == []
        assert result.message == ""

    def test_version_chain_error_is_frozen(self):
        """VersionChainError should be immutable."""
        err = VersionChainError(
            version=3,
            expected_hash="abc",
            actual_hash="def",
            error="mismatch",
        )
        assert err.version == 3
        assert err.expected_hash == "abc"
        assert err.actual_hash == "def"
        assert err.error == "mismatch"

        with pytest.raises(AttributeError):
            err.version = 5  # type: ignore[misc]

    def test_verify_version_result_is_mutable(self):
        """VerifyVersionResult should be mutable (facade may mutate path)."""
        result = VerifyVersionResult(path="/original")
        result.path = "/prefixed/original"
        assert result.path == "/prefixed/original"

    def test_verify_version_result_inherits_file_operation_result(self):
        """VerifyVersionResult is a FileOperationResult subclass."""
        from grover.types.operations import FileOperationResult

        result = VerifyVersionResult()
        assert isinstance(result, FileOperationResult)

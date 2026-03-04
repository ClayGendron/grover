"""Integration tests for version chain verification through the facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from _helpers import FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import Grover, GroverAsync
from grover.models.version import FileVersion
from grover.results.operations import VerifyVersionResult
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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
    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_two_mounts(workspace: Path, workspace2: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
    await g.add_mount(
        "/mount1",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local1"),
        embedding_provider=FakeProvider(),
    )
    await g.add_mount(
        "/mount2",
        LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2"),
        embedding_provider=FakeProvider(),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
def sync_grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover(indexing_mode=IndexingMode.MANUAL)
    g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g
    g.close()


# ------------------------------------------------------------------
# Facade: verify_versions
# ------------------------------------------------------------------


class TestFacadeVerifyVersions:
    async def test_facade_verify_versions(self, grover: GroverAsync):
        """verify_versions through the facade works on a healthy file."""
        await grover.write("/project/f.py", "v1\n")
        await grover.write("/project/f.py", "v2\n")

        result = await grover.verify_versions("/project/f.py")
        assert result.success is True
        assert result.versions_checked == 2
        assert result.versions_passed == 2

    async def test_facade_verify_versions_path_prefixed(self, grover: GroverAsync):
        """Result path should have the mount prefix."""
        await grover.write("/project/f.py", "content\n")

        result = await grover.verify_versions("/project/f.py")
        assert result.path == "/project/f.py"

    async def test_facade_verify_versions_not_found(self, grover: GroverAsync):
        """verify_versions on missing file returns failure."""
        result = await grover.verify_versions("/project/nope.py")
        assert result.success is False


# ------------------------------------------------------------------
# Facade: verify_all_versions
# ------------------------------------------------------------------


class TestFacadeVerifyAllVersions:
    async def test_facade_verify_all_versions(self, grover: GroverAsync):
        """verify_all_versions checks all files across mount."""
        await grover.write("/project/a.py", "a\n")
        await grover.write("/project/b.py", "b\n")

        results = await grover.verify_all_versions()
        assert len(results) == 2
        assert all(r.success for r in results)

    async def test_facade_verify_all_versions_mount_filter(self, grover_two_mounts: GroverAsync):
        """verify_all_versions filters to a specific mount."""
        g = grover_two_mounts
        await g.write("/mount1/a.py", "a\n")
        await g.write("/mount2/b.py", "b\n")

        # Only mount1
        results = await g.verify_all_versions("/mount1")
        assert len(results) == 1
        assert results[0].path == "/mount1/a.py"

        # Only mount2
        results = await g.verify_all_versions("/mount2")
        assert len(results) == 1
        assert results[0].path == "/mount2/b.py"

    async def test_facade_verify_all_versions_paths_prefixed(self, grover: GroverAsync):
        """All result paths should have mount prefix."""
        await grover.write("/project/f.py", "content\n")

        results = await grover.verify_all_versions()
        assert len(results) == 1
        assert results[0].path == "/project/f.py"


# ------------------------------------------------------------------
# Reconcile integration
# ------------------------------------------------------------------


class TestReconcileChainErrors:
    async def test_reconcile_clean_zero_chain_errors(self, grover: GroverAsync):
        """reconcile reports chain_errors=0 when no corruption."""
        await grover.write("/project/f.py", "content\n")
        stats = await grover.reconcile("/project")
        assert stats.chain_errors == 0

    async def test_reconcile_includes_chain_errors(self, workspace: Path, tmp_path: Path):
        """reconcile detects chain_errors when a version is corrupted."""
        data = tmp_path / "grover_data"
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        lfs = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local")
        await g.add_mount("/project", lfs, embedding_provider=FakeProvider())

        await g.write("/project/f.py", "v1\n")
        await g.write("/project/f.py", "v2\n")

        # Corrupt a version hash directly in DB
        async with lfs._session_factory() as sess:
            file_rec = (
                await sess.execute(select(lfs.file_model).where(lfs.file_model.path == "/f.py"))
            ).scalar_one()
            v1_rec = (
                await sess.execute(
                    select(FileVersion).where(
                        FileVersion.file_id == file_rec.id,
                        FileVersion.version == 1,
                    )
                )
            ).scalar_one()
            v1_rec.content_hash = "0" * 64
            sess.add(v1_rec)
            await sess.commit()

        stats = await g.reconcile("/project")
        assert stats.chain_errors > 0
        await g.close()


# ------------------------------------------------------------------
# Sync Grover wrappers
# ------------------------------------------------------------------


class TestSyncGroverVerify:
    def test_sync_grover_verify_versions(self, sync_grover: Grover):
        """Sync Grover.verify_versions works end-to-end."""
        sync_grover.write("/project/f.py", "content\n")
        result = sync_grover.verify_versions("/project/f.py")
        assert isinstance(result, VerifyVersionResult)
        assert result.success is True

    def test_sync_grover_verify_all_versions(self, sync_grover: Grover):
        """Sync Grover.verify_all_versions works end-to-end."""
        sync_grover.write("/project/a.py", "a\n")
        sync_grover.write("/project/b.py", "b\n")
        results = sync_grover.verify_all_versions()
        assert len(results) == 2
        assert all(isinstance(r, VerifyVersionResult) for r in results)
        assert all(r.success for r in results)

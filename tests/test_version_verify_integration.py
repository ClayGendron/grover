"""Integration tests for version chain verification and reconciliation through the facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from _helpers import FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.version import FileVersion
from grover.worker import IndexingMode

if TYPE_CHECKING:
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

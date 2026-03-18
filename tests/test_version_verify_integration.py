"""Integration tests for version chain verification through the facade."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from _helpers import FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.database.version import FileVersionModel
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_verify_message(message: str) -> tuple[int, int, int]:
    """Parse 'Verified: N checked, N passed, N failed' → (checked, passed, failed)."""
    m = re.search(r"(\d+) checked, (\d+) passed, (\d+) failed", message)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 0, 0, 0


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
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ------------------------------------------------------------------
# Reconcile integration
# ------------------------------------------------------------------


class TestReconcileChainErrors:
    async def test_reconcile_clean_no_errors(self, grover: GroverAsync):
        """reconcile succeeds when no corruption."""
        await grover.write("/project/f.py", "content\n")
        stats = await grover.reconcile("/project")
        assert stats.success is True

    async def test_verify_versions_detects_chain_errors(self, workspace: Path, tmp_path: Path):
        """verify_versions detects corruption when a version hash is corrupted."""
        data = tmp_path / "grover_data"
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        lfs = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local")
        await g.add_mount("project", filesystem=lfs, embedding_provider=FakeProvider())

        await g.write("/project/f.py", "v1\n")
        await g.write("/project/f.py", "v2\n")

        # Corrupt a version hash directly in DB
        async with lfs._session_factory() as sess:
            file_rec = (await sess.execute(select(lfs.file_model).where(lfs.file_model.path == "/f.py"))).scalar_one()
            v1_rec = (
                await sess.execute(
                    select(FileVersionModel).where(
                        FileVersionModel.file_id == file_rec.id,
                        FileVersionModel.version == 1,
                    )
                )
            ).scalar_one()
            v1_rec.content_hash = "0" * 64
            sess.add(v1_rec)
            await sess.commit()

        # verify_versions through the backend should detect the corruption
        mount = next(m for m in g._ctx.registry.list_mounts() if m.path == "/project")
        assert mount.filesystem is not None
        async with lfs._session_factory() as sess2:
            result = await mount.filesystem.verify_versions("/f.py", session=sess2)
        assert result.success is False
        _, _, failed = _parse_verify_message(result.message)
        assert failed > 0
        await g.close()

"""Integration tests for version chain verification through the facade."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest
from sqlmodel import select

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.local_fs import LocalFileSystem
from grover.models.files import FileVersion
from grover.types.operations import VerifyVersionResult
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
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
    g = GroverAsync(
        data_dir=str(data),
        embedding_provider=FakeProvider(),
        indexing_mode=IndexingMode.MANUAL,
    )
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_two_mounts(workspace: Path, workspace2: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(
        data_dir=str(data),
        embedding_provider=FakeProvider(),
        indexing_mode=IndexingMode.MANUAL,
    )
    await g.add_mount(
        "/mount1",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local1"),
    )
    await g.add_mount(
        "/mount2",
        LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local2"),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
def sync_grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover(
        data_dir=str(data),
        embedding_provider=FakeProvider(),
        indexing_mode=IndexingMode.MANUAL,
    )
    g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
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
        assert stats["chain_errors"] == 0

    async def test_reconcile_includes_chain_errors(self, workspace: Path, tmp_path: Path):
        """reconcile detects chain_errors when a version is corrupted."""
        data = tmp_path / "grover_data"
        g = GroverAsync(
            data_dir=str(data),
            embedding_provider=FakeProvider(),
            indexing_mode=IndexingMode.MANUAL,
        )
        lfs = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local")
        await g.add_mount("/project", lfs)

        await g.write("/project/f.py", "v1\n")
        await g.write("/project/f.py", "v2\n")

        # Corrupt a version hash directly in DB
        async with lfs._session_factory() as sess:
            file_rec = (
                await sess.execute(select(lfs._file_model).where(lfs._file_model.path == "/f.py"))
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
        assert stats["chain_errors"] > 0
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


# ------------------------------------------------------------------
# Unsupported capability
# ------------------------------------------------------------------


class TestVerifyUnsupported:
    async def test_facade_verify_versions_unsupported(self, tmp_path: Path):
        """verify_versions returns failure when mount doesn't support versions."""
        data = tmp_path / "grover_data"
        g = GroverAsync(
            data_dir=str(data),
            indexing_mode=IndexingMode.MANUAL,
        )

        # Create a mock backend that doesn't satisfy SupportsVersions
        class MinimalBackend:
            async def open(self):
                pass

            async def close(self):
                pass

            async def read(self, path, offset=0, limit=2000, *, session=None, user_id=None):
                from grover.types import ReadResult

                return ReadResult(success=False)

            async def list_dir(self, path="/", *, session=None, user_id=None):
                from grover.types import ListDirResult

                return ListDirResult()

            async def exists(self, path, *, session=None, user_id=None):
                return False

            async def get_info(self, path, *, session=None, user_id=None):
                return None

            async def write(
                self,
                path,
                content,
                created_by="agent",
                *,
                overwrite=True,
                session=None,
                owner_id=None,
                user_id=None,
            ):
                from grover.types import WriteResult

                return WriteResult()

            async def edit(
                self,
                path,
                old,
                new,
                replace_all=False,
                created_by="agent",
                *,
                session=None,
                user_id=None,
            ):
                from grover.types import EditResult

                return EditResult()

            async def delete(self, path, permanent=False, *, session=None, user_id=None):
                from grover.types import DeleteResult

                return DeleteResult()

            async def mkdir(self, path, parents=True, *, session=None, user_id=None):
                from grover.types import MkdirResult

                return MkdirResult()

            async def move(
                self,
                src,
                dest,
                *,
                session=None,
                follow=False,
                sharing=None,
                user_id=None,
            ):
                from grover.types import MoveResult

                return MoveResult()

            async def copy(self, src, dest, *, session=None, user_id=None):
                from grover.types import WriteResult

                return WriteResult()

            async def glob(self, pattern, path="/", *, session=None, user_id=None):
                from grover.types import GlobResult

                return GlobResult()

            async def grep(
                self,
                pattern,
                path="/",
                *,
                glob_filter=None,
                case_sensitive=True,
                fixed_string=False,
                invert=False,
                word_match=False,
                context_lines=0,
                max_results=1000,
                max_results_per_file=0,
                count_only=False,
                files_only=False,
                session=None,
                user_id=None,
            ):
                from grover.types import GrepResult

                return GrepResult()

            async def tree(self, path="/", *, max_depth=None, session=None, user_id=None):
                from grover.types import TreeResult

                return TreeResult()

        await g.add_mount("/minimal", MinimalBackend())  # type: ignore[arg-type]
        result = await g.verify_versions("/minimal/test.py")
        assert result.success is False
        assert "does not support versioning" in result.message
        await g.close()

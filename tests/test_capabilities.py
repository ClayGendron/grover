"""Tests for capability protocols, GroverAsync capability gating, and session handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover._grover_async import GroverAsync
from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import GroverError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.protocol import (
    StorageBackend,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.types import (
    DeleteResult,
    EditResult,
    FileInfoResult,
    FileSearchCandidate,
    GlobResult,
    GrepResult,
    ListDirEvidence,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    TreeResult,
    WriteResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# MinimalBackend — implements only StorageBackend, no capabilities
# =========================================================================


class MinimalBackend:
    """A backend with no versioning, trash, or reconciliation."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ReadResult:
        content = self._files.get(path)
        if content is None:
            return ReadResult(success=False, message=f"Not found: {path}")
        return ReadResult(success=True, message="OK", content=content, path=path)

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        self._files[path] = content
        return WriteResult(success=True, message="OK", path=path)

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> EditResult:
        content = self._files.get(path)
        if content is None:
            return EditResult(success=False, message=f"Not found: {path}")
        self._files[path] = content.replace(old_string, new_string, 1)
        return EditResult(success=True, message="OK", path=path)

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        if path in self._files:
            del self._files[path]
            return DeleteResult(success=True, message="OK", permanent=permanent)
        return DeleteResult(success=False, message=f"Not found: {path}")

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        return MkdirResult(success=True, message="OK", path=path)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MoveResult:
        content = self._files.pop(src, None)
        if content is None:
            return MoveResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return MoveResult(success=True, message="OK", old_path=src, new_path=dest)

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        content = self._files.get(src)
        if content is None:
            return WriteResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return WriteResult(success=True, message="OK", path=dest)

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GlobResult:
        return GlobResult(success=True, message="OK")

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GrepResult:
        return GrepResult(success=True, message="OK")

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> TreeResult:
        return TreeResult(success=True, message="OK")

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListDirResult:
        candidates = []
        prefix = path.rstrip("/") + "/"
        seen = set()
        for p in self._files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                name = rest.split("/")[0]
                if name not in seen:
                    seen.add(name)
                    full_path = prefix + name
                    candidates.append(
                        FileSearchCandidate(
                            path=full_path,
                            evidence=[
                                ListDirEvidence(
                                    strategy="list_dir",
                                    path=full_path,
                                    is_directory="/" in rest,
                                )
                            ],
                        )
                    )
        return ListDirResult(success=True, message="OK", candidates=candidates)

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> bool:
        return path in self._files

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfoResult | None:
        if path not in self._files:
            return None
        return FileInfoResult(path=path, is_directory=False)


# =========================================================================
# Protocol isinstance checks
# =========================================================================


class TestProtocolChecks:
    """Verify isinstance-based capability detection."""

    def test_minimal_is_storage_backend(self) -> None:
        backend = MinimalBackend()
        assert isinstance(backend, StorageBackend)

    def test_minimal_is_not_versions(self) -> None:
        backend = MinimalBackend()
        assert not isinstance(backend, SupportsVersions)

    def test_minimal_is_not_trash(self) -> None:
        backend = MinimalBackend()
        assert not isinstance(backend, SupportsTrash)

    def test_minimal_is_not_reconcile(self) -> None:
        backend = MinimalBackend()
        assert not isinstance(backend, SupportsReconcile)

    def test_local_supports_all(self, tmp_path: Path) -> None:
        lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=tmp_path / ".g")
        assert isinstance(lfs, StorageBackend)
        assert isinstance(lfs, SupportsVersions)
        assert isinstance(lfs, SupportsTrash)
        assert isinstance(lfs, SupportsReconcile)

    def test_database_supports_versions_and_trash(self) -> None:
        dfs = DatabaseFileSystem(dialect="sqlite")
        assert isinstance(dfs, StorageBackend)
        assert isinstance(dfs, SupportsVersions)
        assert isinstance(dfs, SupportsTrash)
        assert not isinstance(dfs, SupportsReconcile)


# =========================================================================
# GroverAsync with MinimalBackend — capability gating
# =========================================================================


@pytest.fixture
async def minimal_grover(tmp_path: Path):
    """GroverAsync with a single MinimalBackend at /mem (no session_factory)."""
    g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
    await g.add_mount("/mem", MinimalBackend())
    yield g
    await g.close()


class TestCapabilityGating:
    """GroverAsync returns failure results for unsupported capabilities."""

    async def test_core_ops_work(self, minimal_grover: GroverAsync) -> None:
        """MinimalBackend handles basic CRUD through GroverAsync."""
        result = await minimal_grover.write("/mem/hello.txt", "hi")
        assert result.success

        read = await minimal_grover.read("/mem/hello.txt")
        assert read.success
        assert read.content == "hi"

        assert await minimal_grover.exists("/mem/hello.txt")

    async def test_list_versions_returns_failure(self, minimal_grover: GroverAsync) -> None:
        result = await minimal_grover.list_versions("/mem/hello.txt")
        assert result.success is False
        assert "does not support versioning" in result.message

    async def test_get_version_content_returns_failure(self, minimal_grover: GroverAsync) -> None:
        result = await minimal_grover.get_version_content("/mem/hello.txt", 1)
        assert result.success is False
        assert "does not support versioning" in result.message

    async def test_restore_version_returns_failure(self, minimal_grover: GroverAsync) -> None:
        result = await minimal_grover.restore_version("/mem/hello.txt", 1)
        assert result.success is False
        assert "does not support versioning" in result.message

    async def test_restore_from_trash_returns_failure(self, minimal_grover: GroverAsync) -> None:
        result = await minimal_grover.restore_from_trash("/mem/hello.txt")
        assert result.success is False
        assert "does not support trash" in result.message

    async def test_delete_without_trash_rejects_soft_delete(
        self, minimal_grover: GroverAsync
    ) -> None:
        """delete(permanent=False) on non-trash backend returns failure, not raise."""
        await minimal_grover.write("/mem/hello.txt", "hi")
        result = await minimal_grover.delete("/mem/hello.txt", permanent=False)
        assert not result.success
        assert "Trash not supported" in result.message

    async def test_delete_permanent_works(self, minimal_grover: GroverAsync) -> None:
        """delete(permanent=True) bypasses trash check."""
        await minimal_grover.write("/mem/hello.txt", "hi")
        result = await minimal_grover.delete("/mem/hello.txt", permanent=True)
        assert result.success

    async def test_list_trash_skips_unsupported(self, minimal_grover: GroverAsync) -> None:
        """Aggregation endpoint skips unsupported mounts, returns empty."""
        result = await minimal_grover.list_trash()
        assert result.success
        assert len(result) == 0

    async def test_empty_trash_skips_unsupported(self, minimal_grover: GroverAsync) -> None:
        """Aggregation endpoint skips unsupported mounts, returns success."""
        result = await minimal_grover.empty_trash()
        assert result.success
        assert result.total_deleted == 0

    async def test_reconcile_skips_unsupported(self, minimal_grover: GroverAsync) -> None:
        """Reconcile skips non-reconcilable backends."""
        stats = await minimal_grover.reconcile()
        assert stats == {"created": 0, "updated": 0, "deleted": 0, "chain_errors": 0}


# =========================================================================
# Mixed mounts — SQL + minimal
# =========================================================================


class TestMixedMounts:
    """GroverAsync with both a SQL backend and a MinimalBackend."""

    @pytest.fixture
    async def mixed_grover(self, tmp_path: Path):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/db", DatabaseFileSystem(dialect="sqlite"), session_factory=factory)
        await g.add_mount("/mem", MinimalBackend())
        yield g
        await g.close()
        await engine.dispose()

    async def test_list_trash_aggregates_across_mounts(self, mixed_grover: GroverAsync) -> None:
        """list_trash aggregates trash-capable mounts, skips others."""
        await mixed_grover.write("/db/a.txt", "content")
        await mixed_grover.delete("/db/a.txt")  # soft-delete

        result = await mixed_grover.list_trash()
        assert result.success
        # Should have the DFS trashed file, MinimalBackend skipped
        assert len(result) == 1

    async def test_versioning_works_on_sql_mount(self, mixed_grover: GroverAsync) -> None:
        await mixed_grover.write("/db/a.txt", "v1")
        await mixed_grover.write("/db/a.txt", "v2")
        result = await mixed_grover.list_versions("/db/a.txt")
        assert result.success
        assert len(result) == 2

    async def test_versioning_fails_on_minimal_mount(self, mixed_grover: GroverAsync) -> None:
        await mixed_grover.write("/mem/a.txt", "v1")
        result = await mixed_grover.list_versions("/mem/a.txt")
        assert result.success is False
        assert "does not support versioning" in result.message


# =========================================================================
# GroverAsync session rollback
# =========================================================================


class _FailingBackend(MinimalBackend):
    """Backend that raises on write to test rollback."""

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        raise RuntimeError("Simulated backend failure")


class TestSessionRollback:
    """Test that GroverContext.session_for rolls back on backend exception."""

    @pytest.fixture
    async def rollback_grover(self, tmp_path: Path):
        """GroverAsync with a DFS mount where we can verify rollback."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/db", DatabaseFileSystem(dialect="sqlite"), session_factory=factory)
        yield g, factory
        await g.close()
        await engine.dispose()

    async def test_backend_exception_triggers_rollback(
        self,
        rollback_grover: tuple[GroverAsync, async_sessionmaker],
    ) -> None:
        """Write succeeds, then a forced failure rolls back -- original intact."""
        grover, _factory = rollback_grover

        # Successful write -- committed
        result = await grover.write("/db/test.txt", "original")
        assert result.success

        # Resolve the mount and monkey-patch the backend to raise on write
        mount, _ = grover._ctx.registry.resolve("/db/test.txt")
        original_write = mount.filesystem.write

        async def _exploding_write(*args, **kwargs):
            # Partially mutate session state, then blow up
            raise RuntimeError("Simulated mid-write failure")

        mount.filesystem.write = _exploding_write  # type: ignore[assignment]

        try:
            # This should return failure (GroverAsync.write catches exceptions)
            result = await grover.write("/db/test.txt", "corrupted")
            assert not result.success

            # Original content must still be intact (session was rolled back)
            read = await grover.read("/db/test.txt")
            assert read.success
            assert read.content == "original"
        finally:
            mount.filesystem.write = original_write  # type: ignore[assignment]

    async def test_failing_backend_returns_failure(self, tmp_path: Path) -> None:
        """GroverAsync returns failure result for backend exceptions."""
        g = GroverAsync(data_dir=str(tmp_path / "grover_data"))
        await g.add_mount("/fail", _FailingBackend())

        result = await g.write("/fail/test.txt", "content")
        assert not result.success
        await g.close()


# =========================================================================
# Session=None failure tests for LocalFileSystem
# =========================================================================


class TestLocalFileSystemRequiresSession:
    """LFS methods fail fast when session is None."""

    @pytest.fixture
    async def lfs(self, tmp_path: Path) -> LocalFileSystem:
        lfs = LocalFileSystem(
            workspace_dir=tmp_path,
            data_dir=tmp_path / ".grover_test",
        )
        await lfs.open()
        return lfs

    async def test_write_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.write("/test.txt", "content", session=None)

    async def test_read_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.read("/test.txt", session=None)

    async def test_edit_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.edit("/test.txt", "old", "new", session=None)

    async def test_delete_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.delete("/test.txt", session=None)

    async def test_list_dir_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.list_dir("/", session=None)

    async def test_list_versions_without_session_raises(self, lfs: LocalFileSystem) -> None:
        with pytest.raises(GroverError, match="requires a session"):
            await lfs.list_versions("/test.txt", session=None)

"""Tests for capability protocols, VFS capability gating, and session handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.exceptions import CapabilityNotSupportedError, GroverError
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.protocol import (
    StorageBackend,
    SupportsReconcile,
    SupportsTrash,
    SupportsVersions,
)
from grover.fs.types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepResult,
    ListResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    TreeResult,
    WriteResult,
)
from grover.fs.vfs import VFS

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
    ) -> ReadResult:
        content = self._files.get(path)
        if content is None:
            return ReadResult(success=False, message=f"Not found: {path}")
        return ReadResult(success=True, message="OK", content=content, file_path=path)

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
    ) -> WriteResult:
        self._files[path] = content
        return WriteResult(success=True, message="OK", file_path=path)

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession | None = None,
    ) -> EditResult:
        content = self._files.get(path)
        if content is None:
            return EditResult(success=False, message=f"Not found: {path}")
        self._files[path] = content.replace(old_string, new_string, 1)
        return EditResult(success=True, message="OK", file_path=path)

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
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
    ) -> MkdirResult:
        return MkdirResult(success=True, message="OK", path=path)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
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
    ) -> WriteResult:
        content = self._files.get(src)
        if content is None:
            return WriteResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return WriteResult(success=True, message="OK", file_path=dest)

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
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
    ) -> GrepResult:
        return GrepResult(success=True, message="OK")

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
    ) -> TreeResult:
        return TreeResult(success=True, message="OK")

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
    ) -> ListResult:
        entries = []
        prefix = path.rstrip("/") + "/"
        seen = set()
        for p in self._files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                name = rest.split("/")[0]
                if name not in seen:
                    seen.add(name)
                    entries.append(
                        FileInfo(
                            path=prefix + name,
                            name=name,
                            is_directory="/" in rest,
                        )
                    )
        return ListResult(success=True, message="OK", entries=entries, path=path)

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> bool:
        return path in self._files

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileInfo | None:
        if path not in self._files:
            return None
        name = path.rsplit("/", 1)[-1]
        return FileInfo(path=path, name=name, is_directory=False)


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
# VFS with MinimalBackend — capability gating
# =========================================================================


@pytest.fixture
def minimal_vfs():
    """VFS with a single MinimalBackend at /mem (no session_factory)."""
    backend = MinimalBackend()
    registry = MountRegistry()
    registry.add_mount(
        MountConfig(
            mount_path="/mem",
            backend=backend,
            session_factory=None,
            mount_type="memory",
        )
    )
    return VFS(registry)


class TestCapabilityGating:
    """VFS raises CapabilityNotSupportedError for unsupported capabilities."""

    async def test_core_ops_work(self, minimal_vfs: VFS) -> None:
        """MinimalBackend handles basic CRUD through VFS."""
        result = await minimal_vfs.write("/mem/hello.txt", "hi")
        assert result.success

        read = await minimal_vfs.read("/mem/hello.txt")
        assert read.success
        assert read.content == "hi"

        assert await minimal_vfs.exists("/mem/hello.txt")

    async def test_list_versions_raises(self, minimal_vfs: VFS) -> None:
        with pytest.raises(CapabilityNotSupportedError, match="does not support versioning"):
            await minimal_vfs.list_versions("/mem/hello.txt")

    async def test_get_version_content_raises(self, minimal_vfs: VFS) -> None:
        with pytest.raises(CapabilityNotSupportedError, match="does not support versioning"):
            await minimal_vfs.get_version_content("/mem/hello.txt", 1)

    async def test_restore_version_raises(self, minimal_vfs: VFS) -> None:
        with pytest.raises(CapabilityNotSupportedError, match="does not support versioning"):
            await minimal_vfs.restore_version("/mem/hello.txt", 1)

    async def test_restore_from_trash_raises(self, minimal_vfs: VFS) -> None:
        with pytest.raises(CapabilityNotSupportedError, match="does not support trash"):
            await minimal_vfs.restore_from_trash("/mem/hello.txt")

    async def test_delete_without_trash_rejects_soft_delete(self, minimal_vfs: VFS) -> None:
        """delete(permanent=False) on non-trash backend returns failure, not raise."""
        await minimal_vfs.write("/mem/hello.txt", "hi")
        result = await minimal_vfs.delete("/mem/hello.txt", permanent=False)
        assert not result.success
        assert "Trash not supported" in result.message

    async def test_delete_permanent_works(self, minimal_vfs: VFS) -> None:
        """delete(permanent=True) bypasses trash check."""
        await minimal_vfs.write("/mem/hello.txt", "hi")
        result = await minimal_vfs.delete("/mem/hello.txt", permanent=True)
        assert result.success

    async def test_list_trash_skips_unsupported(self, minimal_vfs: VFS) -> None:
        """Aggregation endpoint skips unsupported mounts, returns empty."""
        result = await minimal_vfs.list_trash()
        assert result.success
        assert len(result.entries) == 0

    async def test_empty_trash_skips_unsupported(self, minimal_vfs: VFS) -> None:
        """Aggregation endpoint skips unsupported mounts, returns success."""
        result = await minimal_vfs.empty_trash()
        assert result.success
        assert result.total_deleted == 0

    async def test_reconcile_skips_unsupported(self, minimal_vfs: VFS) -> None:
        """Reconcile skips non-reconcilable backends."""
        stats = await minimal_vfs.reconcile()
        assert stats == {"created": 0, "updated": 0, "deleted": 0}


# =========================================================================
# Mixed mounts — SQL + minimal
# =========================================================================


class TestMixedMounts:
    """VFS with both a SQL backend and a MinimalBackend."""

    @pytest.fixture
    async def mixed_vfs(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dfs = DatabaseFileSystem(dialect="sqlite")

        minimal = MinimalBackend()

        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/db",
                backend=dfs,
                session_factory=factory,
                mount_type="vfs",
            )
        )
        registry.add_mount(
            MountConfig(
                mount_path="/mem",
                backend=minimal,
                session_factory=None,
                mount_type="memory",
            )
        )

        vfs = VFS(registry)
        yield vfs
        await vfs.close()
        await engine.dispose()

    async def test_list_trash_aggregates_across_mounts(self, mixed_vfs: VFS) -> None:
        """list_trash aggregates trash-capable mounts, skips others."""
        await mixed_vfs.write("/db/a.txt", "content")
        await mixed_vfs.delete("/db/a.txt")  # soft-delete

        result = await mixed_vfs.list_trash()
        assert result.success
        # Should have the DFS trashed file, MinimalBackend skipped
        assert len(result.entries) == 1

    async def test_versioning_works_on_sql_mount(self, mixed_vfs: VFS) -> None:
        await mixed_vfs.write("/db/a.txt", "v1")
        await mixed_vfs.write("/db/a.txt", "v2")
        result = await mixed_vfs.list_versions("/db/a.txt")
        assert result.success
        assert len(result.versions) == 2

    async def test_versioning_fails_on_minimal_mount(self, mixed_vfs: VFS) -> None:
        await mixed_vfs.write("/mem/a.txt", "v1")
        with pytest.raises(CapabilityNotSupportedError):
            await mixed_vfs.list_versions("/mem/a.txt")


# =========================================================================
# VFS session rollback
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
    ) -> WriteResult:
        raise RuntimeError("Simulated backend failure")


class TestVFSSessionRollback:
    """Test that VFS _session_for rolls back on backend exception."""

    @pytest.fixture
    async def rollback_vfs(self):
        """VFS with a DFS mount where we can verify rollback."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dfs = DatabaseFileSystem(dialect="sqlite")

        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/db",
                backend=dfs,
                session_factory=factory,
                mount_type="vfs",
            )
        )
        vfs = VFS(registry)
        yield vfs, factory
        await vfs.close()
        await engine.dispose()

    async def test_backend_exception_triggers_rollback(
        self,
        rollback_vfs: tuple[VFS, async_sessionmaker],
    ) -> None:
        """Write succeeds, then a forced failure rolls back — original intact."""
        vfs, _factory = rollback_vfs

        # Successful write — committed
        result = await vfs.write("/db/test.txt", "original")
        assert result.success

        # Resolve the mount and monkey-patch the backend to raise on write
        mount, _ = vfs._registry.resolve("/db/test.txt")
        original_write = mount.backend.write

        async def _exploding_write(*args, **kwargs):
            # Partially mutate session state, then blow up
            raise RuntimeError("Simulated mid-write failure")

        mount.backend.write = _exploding_write  # type: ignore[assignment]

        try:
            # This should raise, and _session_for should rollback
            with pytest.raises(RuntimeError, match="Simulated mid-write failure"):
                await vfs.write("/db/test.txt", "corrupted")

            # Original content must still be intact (session was rolled back)
            read = await vfs.read("/db/test.txt")
            assert read.success
            assert read.content == "original"
        finally:
            mount.backend.write = original_write  # type: ignore[assignment]

    async def test_failing_backend_propagates_exception(self) -> None:
        """VFS propagates backend exceptions (not swallowed)."""
        backend = _FailingBackend()
        registry = MountRegistry()
        registry.add_mount(
            MountConfig(
                mount_path="/fail",
                backend=backend,
                session_factory=None,
                mount_type="memory",
            )
        )
        vfs = VFS(registry)

        with pytest.raises(RuntimeError, match="Simulated backend failure"):
            await vfs.write("/fail/test.txt", "content")


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
        await lfs._ensure_db()
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

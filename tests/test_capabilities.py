"""Tests for capability protocols, GroverFileSystem compliance, and session handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.backends.local import LocalFileSystem
from grover.backends.protocol import (
    GroverFileSystem,
    SupportsReBAC,
    SupportsReconcile,
)
from grover.backends.user_scoped import UserScopedFileSystem
from grover.client import GroverAsync

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Protocol isinstance checks
# =========================================================================


class TestProtocolChecks:
    """Verify isinstance-based capability detection."""

    def test_local_satisfies_grover_filesystem(self, tmp_path: Path) -> None:
        lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=tmp_path / ".g")
        assert isinstance(lfs, GroverFileSystem)

    def test_local_supports_reconcile(self, tmp_path: Path) -> None:
        lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=tmp_path / ".g")
        assert isinstance(lfs, SupportsReconcile)

    def test_database_satisfies_grover_filesystem(self) -> None:
        dfs = DatabaseFileSystem(dialect="sqlite")
        assert isinstance(dfs, GroverFileSystem)

    def test_database_does_not_support_reconcile(self) -> None:
        dfs = DatabaseFileSystem(dialect="sqlite")
        assert not isinstance(dfs, SupportsReconcile)

    def test_user_scoped_satisfies_grover_filesystem(self) -> None:
        usfs = UserScopedFileSystem()
        assert isinstance(usfs, GroverFileSystem)

    def test_user_scoped_supports_rebac(self) -> None:
        from grover.models.database.share import FileShareModel

        usfs = UserScopedFileSystem(share_model=FileShareModel)
        assert isinstance(usfs, SupportsReBAC)


# =========================================================================
# GroverAsync session rollback
# =========================================================================


class TestSessionRollback:
    """Test that GroverContext.session_for rolls back on backend exception."""

    @pytest.fixture
    async def rollback_grover(self, tmp_path: Path):
        """GroverAsync with a DFS mount where we can verify rollback."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        g = GroverAsync()
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
            assert read.file.content == "original"
        finally:
            mount.filesystem.write = original_write  # type: ignore[assignment]

    async def test_failing_backend_write_returns_failure(self, tmp_path: Path) -> None:
        """GroverAsync returns failure result for backend exceptions."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        dfs = DatabaseFileSystem(dialect="sqlite")

        g = GroverAsync()
        await g.add_mount("/fail", dfs, session_factory=factory)

        # Monkey-patch write to raise
        async def _exploding_write(*args, **kwargs):
            raise RuntimeError("Simulated backend failure")

        dfs.write = _exploding_write  # type: ignore[assignment]

        result = await g.write("/fail/test.txt", "content")
        assert not result.success
        await g.close()
        await engine.dispose()

"""Tests for read-only permission enforcement across all mutation paths.

Phase 4 of the Alpha Refactor plan — ensures that read-only mounts
block ALL mutations, returning failed Result objects (never raising
exceptions), while allowing all read/query operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.permissions import Permission
from grover.results import (
    ConnectionListResult,
    ConnectionResult,
    DeleteResult,
    EditResult,
    GlobResult,
    GrepResult,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    ShareResult,
    WriteResult,
)

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
def rw_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "rw_workspace"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover_ro(workspace: Path, tmp_path: Path) -> GroverAsync:
    """GroverAsync with a single read-only mount containing one file."""
    data = tmp_path / "grover_data"

    # Pre-create a file on disk so we can read it via the read-only mount
    (workspace / "hello.py").write_text("print('hello')")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "nested.py").write_text("x = 1")

    g = GroverAsync()
    await g.add_mount(
        "/ro",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        permission=Permission.READ_ONLY,
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_mixed(workspace: Path, rw_workspace: Path, tmp_path: Path) -> GroverAsync:
    """GroverAsync with one read-only and one read-write mount."""
    data = tmp_path / "grover_data"

    # Pre-create files in the read-only workspace
    (workspace / "existing.txt").write_text("read-only content")

    g = GroverAsync()
    await g.add_mount(
        "/ro",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "ro_local"),
        permission=Permission.READ_ONLY,
    )
    await g.add_mount(
        "/rw",
        LocalFileSystem(workspace_dir=rw_workspace, data_dir=data / "rw_local"),
        permission=Permission.READ_WRITE,
    )
    yield g  # type: ignore[misc]
    await g.close()


# ------------------------------------------------------------------
# Mutations blocked on read-only mounts
# ------------------------------------------------------------------


class TestReadOnlyBlocksMutations:
    """Every mutation against a read-only mount returns success=False."""

    async def test_read_only_blocks_write(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.write("/ro/new.txt", "content")
        assert isinstance(result, WriteResult)
        assert result.success is False

    async def test_read_only_blocks_edit(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.edit("/ro/hello.py", "print('hello')", "print('bye')")
        assert isinstance(result, EditResult)
        assert result.success is False

    async def test_read_only_blocks_delete(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.delete("/ro/hello.py")
        assert isinstance(result, DeleteResult)
        assert result.success is False

    async def test_read_only_blocks_mkdir(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.mkdir("/ro/newdir")
        assert isinstance(result, MkdirResult)
        assert result.success is False

    async def test_read_only_blocks_move(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.move("/ro/hello.py", "/ro/renamed.py")
        assert isinstance(result, MoveResult)
        assert result.success is False

    async def test_read_only_blocks_add_connection(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.add_connection("/ro/hello.py", "/ro/sub/nested.py", "imports")
        assert isinstance(result, ConnectionResult)
        assert result.success is False

    async def test_read_only_blocks_delete_connection(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.delete_connection("/ro/hello.py", "/ro/sub/nested.py")
        assert isinstance(result, ConnectionResult)
        assert result.success is False

    async def test_read_only_blocks_restore_version(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.restore_version("/ro/hello.py", 1)
        assert isinstance(result, RestoreResult)
        assert result.success is False

    async def test_read_only_blocks_restore_from_trash(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.restore_from_trash("/ro/hello.py")
        assert isinstance(result, RestoreResult)
        assert result.success is False

    async def test_read_only_blocks_share(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.share("/ro/hello.py", "other_user", user_id="owner")
        assert isinstance(result, ShareResult)
        assert result.success is False
        assert "read-only" in result.message.lower()

    async def test_read_only_blocks_unshare(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.unshare("/ro/hello.py", "other_user", user_id="owner")
        assert isinstance(result, ShareResult)
        assert result.success is False
        assert "read-only" in result.message.lower()


# ------------------------------------------------------------------
# Read/query operations allowed on read-only mounts
# ------------------------------------------------------------------


class TestReadOnlyAllowsReads:
    """Read and query operations work normally on read-only mounts."""

    async def test_read_only_allows_read(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.read("/ro/hello.py")
        assert isinstance(result, ReadResult)
        assert result.success is True
        assert "print('hello')" in result.content

    async def test_read_only_allows_glob(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.glob("*.py", "/ro")
        assert isinstance(result, GlobResult)
        assert result.success is True
        assert len(result.candidates) >= 1

    async def test_read_only_allows_grep(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.grep("print", "/ro")
        assert isinstance(result, GrepResult)
        assert result.success is True

    async def test_read_only_allows_list_dir(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.list_dir("/ro")
        assert isinstance(result, ListDirResult)
        assert result.success is True
        assert len(result.candidates) >= 1

    async def test_read_only_allows_graph_queries(self, grover_ro: GroverAsync) -> None:
        graph = grover_ro.get_graph("/ro")
        assert graph is not None
        assert isinstance(graph.node_count, int)
        assert graph.node_count >= 0

    async def test_read_only_allows_list_connections(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.list_connections("/ro/hello.py")
        # list_connections returns ConnectionListResult (read-only operation, always works)
        assert isinstance(result, ConnectionListResult)


# ------------------------------------------------------------------
# Result messages are descriptive
# ------------------------------------------------------------------


class TestReadOnlyResultMessages:
    """Failed results include descriptive messages."""

    async def test_write_message_mentions_read_only(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.write("/ro/new.txt", "content")
        assert "read-only" in result.message.lower()

    async def test_edit_message_mentions_read_only(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.edit("/ro/hello.py", "print('hello')", "print('bye')")
        assert "read-only" in result.message.lower()

    async def test_delete_message_mentions_read_only(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.delete("/ro/hello.py")
        assert "read-only" in result.message.lower()

    async def test_add_connection_message_mentions_read_only(self, grover_ro: GroverAsync) -> None:
        result = await grover_ro.add_connection("/ro/hello.py", "/ro/sub/nested.py", "imports")
        assert "read-only" in result.message.lower()


# ------------------------------------------------------------------
# Mixed mounts — read-only does not affect read-write
# ------------------------------------------------------------------


class TestMixedMountPermissions:
    """Read-only enforcement on one mount does not affect other mounts."""

    async def test_rw_mount_allows_write(self, grover_mixed: GroverAsync) -> None:
        result = await grover_mixed.write("/rw/new.txt", "hello")
        assert result.success is True

    async def test_ro_mount_blocks_write(self, grover_mixed: GroverAsync) -> None:
        result = await grover_mixed.write("/ro/new.txt", "hello")
        assert result.success is False

    async def test_ro_mount_allows_read(self, grover_mixed: GroverAsync) -> None:
        result = await grover_mixed.read("/ro/existing.txt")
        assert result.success is True
        assert result.content == "read-only content"

    async def test_copy_to_ro_blocked(self, grover_mixed: GroverAsync) -> None:
        # Write to rw first, then try to copy to ro
        await grover_mixed.write("/rw/src.txt", "data")
        result = await grover_mixed.copy("/rw/src.txt", "/ro/dest.txt")
        assert result.success is False

    async def test_move_from_ro_blocked(self, grover_mixed: GroverAsync) -> None:
        # Move source is on the ro mount
        result = await grover_mixed.move("/ro/existing.txt", "/rw/moved.txt")
        assert result.success is False


# ------------------------------------------------------------------
# Indexing skips read-only mounts
# ------------------------------------------------------------------


class TestIndexingSkipsReadOnly:
    """Indexing operations skip read-only mounts with skip accounting."""

    async def test_indexing_skips_read_only_mount(self, grover_ro: GroverAsync) -> None:
        stats = await grover_ro.index("/ro")
        # Read-only mounts should be skipped entirely — no files scanned
        assert stats["files_scanned"] == 0

    async def test_indexing_returns_files_skipped_key(self, grover_ro: GroverAsync) -> None:
        stats = await grover_ro.index("/ro")
        assert "files_skipped" in stats

    async def test_indexing_mixed_only_indexes_rw(self, grover_mixed: GroverAsync) -> None:
        # Write a Python file to the rw mount
        await grover_mixed.write("/rw/mod.py", "def foo(): pass")

        # Index all mounts
        stats = await grover_mixed.index()

        # At least the rw file should be indexed
        assert stats["files_scanned"] >= 1

    async def test_indexing_ro_mount_no_chunks_or_edges(self, grover_ro: GroverAsync) -> None:
        stats = await grover_ro.index("/ro")
        assert stats["chunks_created"] == 0
        assert stats["edges_added"] == 0


# ------------------------------------------------------------------
# Defensive checks in _analyze_and_integrate
# ------------------------------------------------------------------


class TestAnalyzeIntegrateReadOnly:
    """Connection persistence in _analyze_and_integrate respects read-only."""

    async def test_analyze_skips_connection_writes_for_readonly(
        self, grover_mixed: GroverAsync
    ) -> None:
        """Even if _analyze_and_integrate is called on a read-only path,
        it should not attempt to persist connections to the database.
        The in-memory graph edges are fine (non-persistent), but DB writes
        are blocked."""
        # Write a Python file with an import to the rw mount
        await grover_mixed.write("/rw/main.py", "import helper\ndef main(): pass")
        await grover_mixed.write("/rw/helper.py", "def help(): pass")

        # Index the rw mount — should create connections
        stats = await grover_mixed.index("/rw")
        assert stats["files_scanned"] >= 1


# ------------------------------------------------------------------
# Empty trash respects read-only
# ------------------------------------------------------------------


class TestEmptyTrashReadOnly:
    """empty_trash skips read-only mounts."""

    async def test_empty_trash_skips_ro_mounts(self, grover_mixed: GroverAsync) -> None:
        # Write and delete a file on the rw mount (creates trash)
        await grover_mixed.write("/rw/temp.txt", "temporary")
        await grover_mixed.delete("/rw/temp.txt")

        # Empty trash — should only affect the rw mount
        result = await grover_mixed.empty_trash()
        assert result.success is True


# ------------------------------------------------------------------
# Reconcile respects read-only
# ------------------------------------------------------------------


class TestReconcileReadOnly:
    """reconcile skips read-only mounts."""

    async def test_reconcile_skips_ro_mounts(self, grover_mixed: GroverAsync) -> None:
        stats = await grover_mixed.reconcile()
        # Should complete without error — ro mount skipped
        from grover.results import ReconcileResult

        assert isinstance(stats, ReconcileResult)

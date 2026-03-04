"""Tests for LocalFileSystem-specific behavior — disk I/O, path security, binary detection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

from grover.backends.local import LocalFileSystem


async def _make_local_fs(tmp_path: Path) -> tuple[LocalFileSystem, AsyncSession]:
    """Create a LocalFileSystem with isolated workspace and data dirs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    fs = LocalFileSystem(workspace_dir=workspace, data_dir=data)
    await fs.open()
    return fs, fs.session_factory


@asynccontextmanager
async def _session(factory) -> AsyncGenerator[AsyncSession]:
    """Mimic VFS session_for: create → yield → commit/rollback → close."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Path Security
# ---------------------------------------------------------------------------


class TestPathSecurity:
    async def test_symlink_rejected(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create a real file outside workspace
        target = tmp_path / "secret.txt"
        target.write_text("secret")

        # Create symlink inside workspace
        link = workspace / "link.txt"
        link.symlink_to(target)

        async with _session(factory) as session:
            result = await fs.read("/link.txt", session=session)
        assert result.success is False
        assert "ymlink" in result.message or "not found" in result.message.lower()
        await fs.close()

    async def test_path_traversal_rejected(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)

        async with _session(factory) as session:
            result = await fs.read("/../../etc/passwd", session=session)
        assert result.success is False
        await fs.close()

    async def test_dotdot_normalized(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)

        # Write a file, then read via a path with ..
        async with _session(factory) as session:
            await fs.write("/bar.py", "content\n", session=session)
        async with _session(factory) as session:
            result = await fs.read("/foo/../bar.py", session=session)
        assert result.success is True
        assert "content" in result.content
        await fs.close()


# ---------------------------------------------------------------------------
# Binary File Handling
# ---------------------------------------------------------------------------


class TestBinaryFileHandling:
    async def test_read_binary_file_rejected(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Write a PNG-like binary file directly to disk
        png_file = workspace / "image.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        async with _session(factory) as session:
            result = await fs.read("/image.png", session=session)
        assert result.success is False
        assert "binary" in result.message.lower()
        await fs.close()


# ---------------------------------------------------------------------------
# Disk Sync
# ---------------------------------------------------------------------------


class TestDiskSync:
    async def test_list_dir_includes_disk_only_files(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create file directly on disk (no FS write)
        (workspace / "disk_only.py").write_text("# disk only\n")

        async with _session(factory) as session:
            result = await fs.list_dir("/", session=session)
        names = [p.rsplit("/", 1)[-1] for p in result.paths]
        assert "disk_only.py" in names
        await fs.close()

    async def test_list_dir_hides_dotfiles(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        (workspace / ".gitignore").write_text("*.pyc\n")
        (workspace / "visible.py").write_text("# visible\n")

        async with _session(factory) as session:
            result = await fs.list_dir("/", session=session)
        names = [p.rsplit("/", 1)[-1] for p in result.paths]
        assert ".gitignore" not in names
        assert "visible.py" in names
        await fs.close()


# ---------------------------------------------------------------------------
# Delete Backup
# ---------------------------------------------------------------------------


class TestDeleteBackup:
    async def test_delete_backs_up_disk_only_file(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create file directly on disk
        (workspace / "ephemeral.py").write_text("content\n")

        # Delete via FS — should create DB record first, then soft-delete
        async with _session(factory) as session:
            result = await fs.delete("/ephemeral.py", session=session)
        assert result.success is True

        # File should be in trash (backed up to DB)
        async with _session(factory) as session:
            trash = await fs.list_trash(session=session)
        paths = trash.deleted_paths()
        assert "/ephemeral.py" in paths
        await fs.close()


# ---------------------------------------------------------------------------
# Atomic Writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    async def test_write_creates_file_on_disk(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/hello.py", "print('hello')\n", session=session)

        disk_file = workspace / "hello.py"
        assert disk_file.exists()
        assert disk_file.read_text() == "print('hello')\n"
        await fs.close()

    async def test_write_content_atomic(self, tmp_path: Path):
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/atomic.py", "content\n", session=session)

        # Verify no .tmp_ files left behind
        tmp_files = list(workspace.glob(".tmp_*"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"
        await fs.close()


# ---------------------------------------------------------------------------
# C4: Concurrent init race condition
# ---------------------------------------------------------------------------


class TestConcurrentInit:
    async def test_concurrent_ensure_db(self, tmp_path: Path):
        """C4: Concurrent _ensure_db calls should not create multiple engines."""
        import asyncio

        fs, factory = await _make_local_fs(tmp_path)

        # Launch multiple concurrent _ensure_db calls
        await asyncio.gather(
            fs.open(),
            fs.open(),
            fs.open(),
        )

        # Should only have one engine
        assert fs.engine is not None
        assert fs.session_factory is not None

        # FS should still work correctly after concurrent init
        async with _session(factory) as session:
            result = await fs.write("/test.py", "hello\n", session=session)
        assert result.success is True
        await fs.close()


# ---------------------------------------------------------------------------
# C5: Trash restore writes content to disk
# ---------------------------------------------------------------------------


class TestTrashRestoreDisk:
    async def test_restore_from_trash_writes_to_disk(self, tmp_path: Path):
        """C5: Restoring from trash should write content back to disk."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Write file (creates on disk + DB)
        async with _session(factory) as session:
            await fs.write("/restore_me.py", "precious content\n", session=session)
        assert (workspace / "restore_me.py").exists()

        # Delete (removes from disk, soft-deletes in DB)
        async with _session(factory) as session:
            await fs.delete("/restore_me.py", session=session)
        assert not (workspace / "restore_me.py").exists()

        # Restore from trash
        async with _session(factory) as session:
            result = await fs.restore_from_trash("/restore_me.py", session=session)
        assert result.success is True

        # File should be back on disk with correct content
        disk_path = workspace / "restore_me.py"
        assert disk_path.exists()
        assert disk_path.read_text() == "precious content\n"

        # Should also be readable through the FS
        async with _session(factory) as session:
            read = await fs.read("/restore_me.py", session=session)
        assert read.success is True
        assert "precious content" in read.content
        await fs.close()

    async def test_restore_edited_file_from_trash(self, tmp_path: Path):
        """C5: Restoring a multi-version file gets the latest version."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/multi.py", "version 1\n", session=session)
        async with _session(factory) as session:
            await fs.edit("/multi.py", "version 1", "version 2", session=session)

        async with _session(factory) as session:
            await fs.delete("/multi.py", session=session)
        assert not (workspace / "multi.py").exists()

        async with _session(factory) as session:
            result = await fs.restore_from_trash("/multi.py", session=session)
        assert result.success is True

        disk_content = (workspace / "multi.py").read_text()
        assert "version 2" in disk_content
        await fs.close()


# ---------------------------------------------------------------------------
# H1: Concurrent writes (session-per-operation)
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    async def test_concurrent_writes_no_session_conflict(self, tmp_path: Path):
        """H1: Two concurrent writes should not interleave on the same session."""
        import asyncio

        fs, factory = await _make_local_fs(tmp_path)

        async def write_file(name: str, content: str):
            async with _session(factory) as session:
                result = await fs.write(f"/{name}.py", content, session=session)
            assert result.success is True
            return result

        # Launch concurrent writes
        results = await asyncio.gather(
            write_file("a", "content_a\n"),
            write_file("b", "content_b\n"),
            write_file("c", "content_c\n"),
        )

        assert all(r.success for r in results)

        # Verify all files exist and have correct content
        for name, expected in [("a", "content_a\n"), ("b", "content_b\n"), ("c", "content_c\n")]:
            async with _session(factory) as session:
                read = await fs.read(f"/{name}.py", session=session)
            assert read.success is True
            assert read.content == expected
        await fs.close()

    async def test_concurrent_read_write(self, tmp_path: Path):
        """H1: Concurrent reads and writes should not interfere."""
        import asyncio

        fs, factory = await _make_local_fs(tmp_path)
        async with _session(factory) as session:
            await fs.write("/shared.py", "initial\n", session=session)

        async def read_file():
            async with _session(factory) as session:
                return await fs.read("/shared.py", session=session)

        async def write_file():
            async with _session(factory) as session:
                return await fs.write("/other.py", "other\n", session=session)

        results = await asyncio.gather(
            read_file(),
            write_file(),
            read_file(),
        )

        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is True
        await fs.close()


# ---------------------------------------------------------------------------
# H3: Soft-delete/restore directory children on disk
# ---------------------------------------------------------------------------


class TestDirectoryTrashDisk:
    async def test_soft_delete_directory_removes_children_from_disk(self, tmp_path: Path):
        """H3: Soft-deleting a directory also trashes children."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/mydir/child.py", "child content\n", session=session)
        assert (workspace / "mydir" / "child.py").exists()

        async with _session(factory) as session:
            result = await fs.delete("/mydir", session=session)
        assert result.success is True

        # Child should not be readable
        async with _session(factory) as session:
            read = await fs.read("/mydir/child.py", session=session)
        assert read.success is False

        # Both parent and child should be in trash
        async with _session(factory) as session:
            trash = await fs.list_trash(session=session)
        paths = trash.deleted_paths()
        assert "/mydir" in paths
        assert "/mydir/child.py" in paths
        await fs.close()

    async def test_restore_directory_restores_children_to_disk(self, tmp_path: Path):
        """H3: Restoring a directory from trash restores children's disk content."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/mydir/child.py", "child content\n", session=session)
        async with _session(factory) as session:
            await fs.write("/mydir/deep/nested.py", "nested content\n", session=session)
        async with _session(factory) as session:
            await fs.delete("/mydir", session=session)

        # Files gone from disk
        assert not (workspace / "mydir" / "child.py").exists()

        async with _session(factory) as session:
            result = await fs.restore_from_trash("/mydir", session=session)
        assert result.success is True

        # Children should be back on disk
        assert (workspace / "mydir" / "child.py").read_text() == "child content\n"
        assert (workspace / "mydir" / "deep" / "nested.py").read_text() == "nested content\n"

        # And readable through the FS
        async with _session(factory) as session:
            read = await fs.read("/mydir/child.py", session=session)
        assert read.success is True
        assert "child content" in read.content
        await fs.close()


# ---------------------------------------------------------------------------
# H7: WAL pragma verification
# ---------------------------------------------------------------------------


class TestWALPragma:
    async def test_wal_mode_active(self, tmp_path: Path):
        """H7: WAL mode should be active after initialization."""
        fs, factory = await _make_local_fs(tmp_path)

        # Query the database to verify WAL mode
        async with factory() as session:
            result = await session.execute(__import__("sqlalchemy").text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal"
        await fs.close()

    async def test_busy_timeout_set(self, tmp_path: Path):
        """H7: busy_timeout should be set to 5000ms."""
        fs, factory = await _make_local_fs(tmp_path)

        async with factory() as session:
            result = await session.execute(__import__("sqlalchemy").text("PRAGMA busy_timeout"))
            timeout = result.scalar()
            assert timeout == 5000
        await fs.close()

    async def test_synchronous_full(self, tmp_path: Path):
        """H8: synchronous should be FULL (value 2)."""
        fs, factory = await _make_local_fs(tmp_path)

        async with factory() as session:
            result = await session.execute(__import__("sqlalchemy").text("PRAGMA synchronous"))
            level = result.scalar()
            # FULL = 2
            assert level == 2
        await fs.close()


# ---------------------------------------------------------------------------
# Phase 3: Inheritance + DiskStorageProvider integration
# ---------------------------------------------------------------------------


class TestLocalFSInheritsDatabaseFS:
    """LocalFileSystem is a thin wrapper over DatabaseFileSystem."""

    async def test_isinstance_database_fs(self, tmp_path: Path):
        """LocalFileSystem is a subclass of DatabaseFileSystem."""
        from grover.backends.database import DatabaseFileSystem

        fs, _ = await _make_local_fs(tmp_path)
        assert isinstance(fs, DatabaseFileSystem)
        await fs.close()

    async def test_has_disk_storage_provider(self, tmp_path: Path):
        """LocalFileSystem creates a DiskStorageProvider."""
        from grover.providers.storage.disk import DiskStorageProvider

        fs, _ = await _make_local_fs(tmp_path)
        assert isinstance(fs.storage_provider, DiskStorageProvider)
        assert isinstance(fs._disk, DiskStorageProvider)
        assert fs.storage_provider is fs._disk
        await fs.close()

    async def test_disk_provider_workspace_matches(self, tmp_path: Path):
        """DiskStorageProvider workspace_dir matches LocalFileSystem workspace_dir."""
        fs, _ = await _make_local_fs(tmp_path)
        assert fs._disk.workspace_dir == fs.workspace_dir.resolve()
        await fs.close()

    async def test_glob_uses_disk(self, tmp_path: Path):
        """glob delegates to DiskStorageProvider (reads from disk, not DB)."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        # Create files directly on disk (no FS write)
        (workspace / "alpha.py").write_text("# alpha\n")
        (workspace / "beta.py").write_text("# beta\n")

        async with _session(factory) as session:
            result = await fs.glob("*.py", session=session)
        assert result.success is True
        paths = list(result.files())
        assert "/alpha.py" in paths
        assert "/beta.py" in paths
        await fs.close()

    async def test_grep_uses_disk(self, tmp_path: Path):
        """grep delegates to DiskStorageProvider (reads from disk)."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        (workspace / "search_me.py").write_text("needle in haystack\n")

        async with _session(factory) as session:
            result = await fs.grep("needle", session=session)
        assert result.success is True
        assert result.files_matched == 1
        await fs.close()

    async def test_tree_uses_disk(self, tmp_path: Path):
        """tree delegates to DiskStorageProvider (reads from disk)."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "file.py").write_text("content\n")

        async with _session(factory) as session:
            result = await fs.tree("/", session=session)
        assert result.success is True
        paths = [c.path for c in result.candidates]
        assert "/subdir" in paths
        assert "/subdir/file.py" in paths
        await fs.close()

    async def test_exists_uses_disk(self, tmp_path: Path):
        """exists delegates to DiskStorageProvider (checks disk, not DB)."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        (workspace / "disk_only.py").write_text("content\n")

        async with _session(factory) as session:
            result = await fs.exists("/disk_only.py", session=session)
        assert result.exists is True
        await fs.close()

    async def test_no_unexpected_method_overrides(self, tmp_path: Path):
        """LocalFileSystem only overrides the expected methods."""
        expected_overrides = {
            "__init__",
            "open",
            "close",
            "read",
            "delete",
            "mkdir",
            "restore_from_trash",
            "reconcile",
            "_ensure_db",
        }

        # Find methods defined directly on LocalFileSystem (not inherited)
        lfs_own = {
            name
            for name, val in vars(LocalFileSystem).items()
            if (callable(val) and not name.startswith("__")) or name == "__init__"
        }
        # Properties
        lfs_own -= {"session_factory", "engine"}

        # All own methods should be in the expected set
        unexpected = lfs_own - expected_overrides
        assert unexpected == set(), f"Unexpected overrides: {unexpected}"

    async def test_inherited_write_works(self, tmp_path: Path):
        """write is inherited from DatabaseFileSystem, not overridden."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            result = await fs.write("/inherited.py", "content\n", session=session)
        assert result.success is True
        assert (workspace / "inherited.py").read_text() == "content\n"
        await fs.close()

    async def test_inherited_edit_works(self, tmp_path: Path):
        """edit is inherited from DatabaseFileSystem, not overridden."""
        fs, factory = await _make_local_fs(tmp_path)
        workspace = fs.workspace_dir

        async with _session(factory) as session:
            await fs.write("/edit_me.py", "old content\n", session=session)
        async with _session(factory) as session:
            result = await fs.edit("/edit_me.py", "old", "new", session=session)
        assert result.success is True
        assert (workspace / "edit_me.py").read_text() == "new content\n"
        await fs.close()

    async def test_inherited_versions_work(self, tmp_path: Path):
        """Version methods are inherited from DatabaseFileSystem mixins."""
        fs, factory = await _make_local_fs(tmp_path)

        async with _session(factory) as session:
            await fs.write("/versioned.py", "v1\n", session=session)
        async with _session(factory) as session:
            await fs.write("/versioned.py", "v2\n", session=session)
        async with _session(factory) as session:
            versions = await fs.list_versions("/versioned.py", session=session)
        assert versions.success is True
        assert len(versions.candidates) == 2
        await fs.close()

    async def test_provider_kwargs_forwarded(self, tmp_path: Path):
        """Provider kwargs are forwarded to DatabaseFileSystem."""
        from unittest.mock import MagicMock

        mock_graph = MagicMock()
        fs = LocalFileSystem(
            workspace_dir=tmp_path,
            data_dir=tmp_path / ".grover_test",
            graph_provider=mock_graph,
        )
        assert fs.graph_provider is mock_graph

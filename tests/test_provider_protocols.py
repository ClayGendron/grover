"""Tests for provider protocols, default implementations, and DiskStorageProvider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.providers.chunks.protocol import ChunkProvider
from grover.providers.graph import RustworkxGraph
from grover.providers.graph.protocol import GraphProvider
from grover.providers.storage.disk import DiskStorageProvider
from grover.providers.storage.protocol import StorageProvider
from grover.providers.versioning.protocol import VersionProvider

if TYPE_CHECKING:
    from pathlib import Path

# ======================================================================
# Protocol runtime checkability
# ======================================================================


class TestStorageProviderProtocol:
    def test_runtime_checkable(self) -> None:
        assert not isinstance(object(), StorageProvider)

    def test_disk_storage_satisfies(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        assert isinstance(dsp, StorageProvider)


class TestStorageProviderQueryMethods:
    def test_disk_storage_has_query_methods(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        assert hasattr(dsp, "storage_glob")
        assert hasattr(dsp, "storage_grep")
        assert hasattr(dsp, "storage_tree")
        assert hasattr(dsp, "storage_list_dir")

    def test_disk_storage_has_reconcile(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        assert hasattr(dsp, "reconcile")


class TestGraphProviderProtocol:
    def test_runtime_checkable(self) -> None:
        assert not isinstance(object(), GraphProvider)

    def test_rustworkx_satisfies(self) -> None:
        g = RustworkxGraph()
        assert isinstance(g, GraphProvider)


class TestVersionProviderProtocol:
    def test_runtime_checkable(self) -> None:
        assert not isinstance(object(), VersionProvider)


class TestChunkProviderProtocol:
    def test_runtime_checkable(self) -> None:
        assert not isinstance(object(), ChunkProvider)


# ======================================================================
# DiskStorageProvider integration tests
# ======================================================================


class TestDiskStorageReadWriteDelete:
    async def test_write_and_read(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/hello.txt", "world")
        content = await dsp.read_content("/hello.txt")
        assert content == "world"

    async def test_read_nonexistent(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.read_content("/missing.txt")
        assert result is None

    async def test_delete(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/to_delete.txt", "bye")
        assert await dsp.exists("/to_delete.txt")
        await dsp.delete_content("/to_delete.txt")
        assert not await dsp.exists("/to_delete.txt")

    async def test_delete_nonexistent(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        # Should not raise
        await dsp.delete_content("/nope.txt")

    async def test_overwrite(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/file.txt", "v1")
        await dsp.write_content("/file.txt", "v2")
        content = await dsp.read_content("/file.txt")
        assert content == "v2"

    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/deep/nested/dir/file.txt", "content")
        content = await dsp.read_content("/deep/nested/dir/file.txt")
        assert content == "content"


class TestDiskStorageMoveCopy:
    async def test_move(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/src.txt", "data")
        await dsp.move_content("/src.txt", "/dest.txt")
        assert not await dsp.exists("/src.txt")
        assert await dsp.read_content("/dest.txt") == "data"

    async def test_copy(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/src.txt", "data")
        await dsp.copy_content("/src.txt", "/copy.txt")
        assert await dsp.exists("/src.txt")
        assert await dsp.read_content("/copy.txt") == "data"

    async def test_move_creates_parent_dirs(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/a.txt", "hello")
        await dsp.move_content("/a.txt", "/sub/dir/b.txt")
        assert await dsp.read_content("/sub/dir/b.txt") == "hello"


class TestDiskStorageExists:
    async def test_exists_true(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/file.txt", "hi")
        assert await dsp.exists("/file.txt")

    async def test_exists_false(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        assert not await dsp.exists("/nope.txt")

    async def test_exists_directory(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.mkdir("/mydir")
        assert await dsp.exists("/mydir")


class TestDiskStorageMkdir:
    async def test_mkdir(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.mkdir("/new_dir")
        assert (tmp_path / "new_dir").is_dir()

    async def test_mkdir_parents(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.mkdir("/a/b/c")
        assert (tmp_path / "a" / "b" / "c").is_dir()

    async def test_mkdir_idempotent(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.mkdir("/existing")
        await dsp.mkdir("/existing")  # Should not raise
        assert (tmp_path / "existing").is_dir()


class TestDiskStorageGetInfo:
    async def test_get_info_file(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/info.txt", "some content")
        info = await dsp.get_info("/info.txt")
        assert info.success
        assert info.path == "/info.txt"
        assert not info.is_directory
        assert info.size_bytes > 0

    async def test_get_info_directory(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.mkdir("/mydir")
        info = await dsp.get_info("/mydir")
        assert info.success
        assert info.is_directory

    async def test_get_info_missing(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        info = await dsp.get_info("/missing.txt")
        assert not info.success


class TestDiskStoragePathTraversal:
    async def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.read_content("/../../../etc/passwd")
        assert result is None

    async def test_symlink_blocked(self, tmp_path: Path) -> None:
        target = tmp_path / "target.txt"
        target.write_text("secret")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.read_content("/link.txt")
        # Should fail because symlinks are blocked
        assert result is None


# ======================================================================
# DiskStorageProvider query tests
# ======================================================================


class TestDiskStorageGlob:
    async def test_glob_finds_files(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/a.py", "print('a')")
        await dsp.write_content("/b.py", "print('b')")
        await dsp.write_content("/readme.md", "# readme")

        result = await dsp.storage_glob("*.py")
        assert result.success
        paths = list(result.files())
        assert "/a.py" in paths
        assert "/b.py" in paths
        assert "/readme.md" not in paths

    async def test_glob_nested(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/src/main.py", "main")
        await dsp.write_content("/src/lib/utils.py", "utils")

        result = await dsp.storage_glob("**/*.py")
        assert result.success
        paths = list(result.files())
        assert "/src/main.py" in paths
        assert "/src/lib/utils.py" in paths

    async def test_glob_empty_pattern(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.storage_glob("")
        assert not result.success


class TestDiskStorageGrep:
    async def test_grep_finds_matches(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/a.py", "import os\nimport sys\n")
        await dsp.write_content("/b.py", "print('hello')\n")

        result = await dsp.storage_grep("import")
        assert result.success
        paths = [c.path for c in result.candidates]
        assert "/a.py" in paths
        assert "/b.py" not in paths

    async def test_grep_case_insensitive(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/file.txt", "Hello World\n")

        result = await dsp.storage_grep("hello", case_sensitive=False)
        assert result.success
        assert len(result.candidates) == 1

    async def test_grep_invalid_regex(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.storage_grep("[invalid")
        assert not result.success


class TestDiskStorageTree:
    async def test_tree(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/src/main.py", "main")
        await dsp.mkdir("/src/lib")
        await dsp.write_content("/src/lib/utils.py", "utils")

        result = await dsp.storage_tree()
        assert result.success
        paths = [c.path for c in result.candidates]
        assert "/src" in paths
        assert "/src/main.py" in paths
        assert "/src/lib" in paths
        assert "/src/lib/utils.py" in paths

    async def test_tree_max_depth(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/a/b/c/deep.txt", "deep")

        result = await dsp.storage_tree(max_depth=1)
        assert result.success
        paths = [c.path for c in result.candidates]
        assert "/a" in paths
        # Depth 2+ should be excluded
        assert "/a/b/c/deep.txt" not in paths


class TestDiskStorageListDir:
    async def test_list_dir(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/file1.txt", "one")
        await dsp.write_content("/file2.txt", "two")
        await dsp.mkdir("/subdir")

        result = await dsp.storage_list_dir("/")
        assert result.success
        paths = [c.path for c in result.candidates]
        assert "/file1.txt" in paths
        assert "/file2.txt" in paths
        assert "/subdir" in paths

    async def test_list_dir_missing(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        result = await dsp.storage_list_dir("/nonexistent")
        assert not result.success

    async def test_list_dir_hides_dotfiles(self, tmp_path: Path) -> None:
        dsp = DiskStorageProvider(tmp_path)
        await dsp.write_content("/visible.txt", "yes")
        # Create a dotfile directly
        (tmp_path / ".hidden").write_text("no")

        result = await dsp.storage_list_dir("/")
        assert result.success
        paths = [c.path for c in result.candidates]
        assert "/visible.txt" in paths
        assert "/.hidden" not in paths


# ======================================================================
# Default provider re-exports
# ======================================================================


class TestDefaultProviderExports:
    def test_default_version_provider_is_accessible(self) -> None:
        from grover.providers import versioning

        assert versioning.DefaultVersionProvider is not None

    def test_default_chunk_provider_is_accessible(self) -> None:
        from grover.providers import chunks

        assert chunks.DefaultChunkProvider is not None

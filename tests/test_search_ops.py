"""Tests for glob, grep, and tree operations on DatabaseFS, LocalFS, and VFS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.local_fs import LocalFileSystem
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.utils import glob_to_sql_like, match_glob
from grover.fs.vfs import VFS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# =========================================================================
# Unit tests: glob utility functions
# =========================================================================


class TestGlobToSqlLike:
    """Tests for glob_to_sql_like()."""

    def test_star(self) -> None:
        assert glob_to_sql_like("*.py", "/") == "/%.py"

    def test_double_star(self) -> None:
        result = glob_to_sql_like("**/*.py", "/")
        assert result is not None
        assert "%" in result

    def test_question_mark(self) -> None:
        assert glob_to_sql_like("?.txt", "/src") == "/src/_.txt"

    def test_bracket_returns_none(self) -> None:
        assert glob_to_sql_like("[abc].txt", "/") is None

    def test_base_path(self) -> None:
        result = glob_to_sql_like("*.py", "/src")
        assert result is not None
        assert result.startswith("/src/")

    def test_absolute_pattern(self) -> None:
        result = glob_to_sql_like("/src/*.py", "/")
        assert result == "/src/%.py"

    def test_escapes_percent(self) -> None:
        result = glob_to_sql_like("100%.txt", "/")
        assert result is not None
        assert "\\%" in result

    def test_escapes_underscore(self) -> None:
        result = glob_to_sql_like("my_file.txt", "/")
        assert result is not None
        assert "\\_" in result


class TestMatchGlob:
    """Tests for match_glob()."""

    def test_star(self) -> None:
        assert match_glob("/src/main.py", "*.py", "/src")
        assert not match_glob("/src/main.txt", "*.py", "/src")

    def test_star_no_cross_directory(self) -> None:
        # Single * should not match across directories
        assert not match_glob("/src/sub/main.py", "*.py", "/src")

    def test_double_star(self) -> None:
        assert match_glob("/src/sub/deep/main.py", "**/*.py", "/src")
        assert match_glob("/src/main.py", "**/*.py", "/src")

    def test_question_mark(self) -> None:
        assert match_glob("/a.txt", "?.txt", "/")
        assert not match_glob("/ab.txt", "?.txt", "/")

    def test_bracket(self) -> None:
        assert match_glob("/a.txt", "[abc].txt", "/")
        assert not match_glob("/d.txt", "[abc].txt", "/")

    def test_root_base(self) -> None:
        assert match_glob("/file.py", "*.py", "/")

    def test_absolute_pattern(self) -> None:
        assert match_glob("/src/main.py", "/src/*.py")

    def test_negated_bracket(self) -> None:
        assert not match_glob("/a.txt", "[!abc].txt", "/")
        assert match_glob("/d.txt", "[!abc].txt", "/")

    def test_unclosed_bracket(self) -> None:
        # Should not crash, just return False
        assert not match_glob("/a.txt", "[abc.txt", "/")


# =========================================================================
# Fixtures: DatabaseFileSystem
# =========================================================================


@pytest.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session


@pytest.fixture
def dfs() -> DatabaseFileSystem:
    return DatabaseFileSystem(dialect="sqlite")


async def _seed_db(dfs: DatabaseFileSystem, session: AsyncSession) -> None:
    """Create a directory tree with files for testing."""
    await dfs.mkdir("/src", session=session)
    await dfs.mkdir("/src/sub", session=session)
    await dfs.mkdir("/docs", session=session)

    await dfs.write("/src/main.py", "def main():\n    print('hello')\n", session=session)
    await dfs.write("/src/utils.py", "def helper():\n    return 42\n", session=session)
    await dfs.write("/src/sub/deep.py", "# deep module\nclass Deep:\n    pass\n", session=session)
    await dfs.write("/docs/readme.md", "# README\nThis is a test project.\n", session=session)
    await dfs.write("/docs/guide.txt", "Step 1: install\nStep 2: run\n", session=session)
    await session.commit()


# =========================================================================
# DatabaseFileSystem: glob
# =========================================================================


class TestDatabaseGlob:
    async def test_star_pattern(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.glob("*.py", "/src", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        # Should not include deep subdirectory files
        assert "/src/sub/deep.py" not in paths

    async def test_double_star_pattern(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.glob("**/*.py", "/src", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub/deep.py" in paths

    async def test_question_mark(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        # guide.txt has 5 chars before extension; readme.md doesn't match
        result = await dfs.glob("*.md", "/docs", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/docs/readme.md" in paths

    async def test_empty_result(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.glob("*.rs", "/src", session=db_session)
        assert result.success
        assert len(result.entries) == 0

    async def test_nonexistent_directory(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.glob("*.py", "/nonexistent", session=db_session)
        assert not result.success
        assert "not found" in result.message.lower()

    async def test_root_glob(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.glob("**/*.py", "/", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub/deep.py" in paths

    async def test_bracket_pattern(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        # [mu]*.py should match main.py and utils.py
        result = await dfs.glob("[mu]*.py", "/src", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths


# =========================================================================
# DatabaseFileSystem: grep
# =========================================================================


class TestDatabaseGrep:
    async def test_basic_regex(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def ", "/src", session=db_session)
        assert result.success
        assert result.files_matched >= 2
        assert len(result.matches) >= 2

    async def test_fixed_string(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("print('hello')", "/src", fixed_string=True, session=db_session)
        assert result.success
        assert result.files_matched == 1
        assert result.matches[0].file_path == "/src/main.py"

    async def test_case_insensitive(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("readme", "/docs", case_sensitive=False, session=db_session)
        assert result.success
        assert result.files_matched >= 1

    async def test_invert(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        # Invert: lines that don't match "def" in a single file
        result = await dfs.grep("def", "/src/main.py", invert=True, session=db_session)
        assert result.success
        assert len(result.matches) >= 1  # Should have non-def lines
        for m in result.matches:
            assert "def" not in m.line_content

    async def test_grep_single_file(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def main", "/src/main.py", session=db_session)
        assert result.success
        assert result.files_matched == 1
        assert len(result.matches) == 1
        assert result.matches[0].file_path == "/src/main.py"

    async def test_word_match(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("main", "/src", word_match=True, session=db_session)
        assert result.success
        assert result.files_matched >= 1

    async def test_context_lines(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        # "print" is on line 2 of main.py ("def main():\n    print('hello')\n")
        result = await dfs.grep("print", "/src/main.py", context_lines=1, session=db_session)
        assert result.success
        assert len(result.matches) == 1
        m = result.matches[0]
        assert "print" in m.line_content
        # Line 2 has 1 line before it (line 1: "def main():")
        assert len(m.context_before) == 1
        assert "def main" in m.context_before[0]

    async def test_max_results(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep(".", "/", max_results=2, session=db_session)
        assert result.success
        assert len(result.matches) <= 2

    async def test_count_only(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def", "/src", count_only=True, session=db_session)
        assert result.success
        assert "Count:" in result.message
        assert len(result.matches) == 0  # count_only returns no matches

    async def test_files_only(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def", "/src", files_only=True, session=db_session)
        assert result.success
        # Each file should appear only once
        file_paths = [m.file_path for m in result.matches]
        assert len(file_paths) == len(set(file_paths))

    async def test_glob_filter(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("Step", "/docs", glob_filter="*.txt", session=db_session)
        assert result.success
        assert result.files_matched == 1
        assert result.matches[0].file_path == "/docs/guide.txt"

    async def test_invalid_regex(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("[invalid", "/src", session=db_session)
        assert not result.success
        assert "invalid regex" in result.message.lower() or "Invalid" in result.message

    async def test_max_results_per_file(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep(".", "/src/main.py", max_results_per_file=1, session=db_session)
        assert result.success
        # Should have at most 1 match for main.py
        main_matches = [m for m in result.matches if m.file_path == "/src/main.py"]
        assert len(main_matches) <= 1

    async def test_line_numbers_are_1_indexed(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def main", "/src", session=db_session)
        assert result.success
        assert len(result.matches) >= 1
        assert result.matches[0].line_number == 1

    async def test_nonexistent_directory(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.grep("def", "/nonexistent", session=db_session)
        assert not result.success
        assert "not found" in result.message.lower()


# =========================================================================
# DatabaseFileSystem: tree
# =========================================================================


class TestDatabaseTree:
    async def test_full_tree(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.tree("/", session=db_session)
        assert result.success
        assert result.total_files >= 5
        assert result.total_dirs >= 3

    async def test_subtree(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.tree("/src", session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub" in paths
        assert "/src/sub/deep.py" in paths
        # Should not include docs
        assert "/docs/readme.md" not in paths

    async def test_max_depth(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.tree("/src", max_depth=1, session=db_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub" in paths
        # deep.py is at depth 2 relative to /src, should be excluded
        assert "/src/sub/deep.py" not in paths

    async def test_nonexistent_directory(
        self, dfs: DatabaseFileSystem, db_session: AsyncSession
    ) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.tree("/nonexistent", session=db_session)
        assert not result.success

    async def test_empty_directory(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        await dfs.mkdir("/empty", session=db_session)
        await db_session.commit()
        result = await dfs.tree("/empty", session=db_session)
        assert result.success
        assert result.total_files == 0
        assert result.total_dirs == 0

    async def test_sorted_by_path(self, dfs: DatabaseFileSystem, db_session: AsyncSession) -> None:
        await _seed_db(dfs, db_session)
        result = await dfs.tree("/", session=db_session)
        assert result.success
        paths = [e.path for e in result.entries]
        assert paths == sorted(paths)


# =========================================================================
# LocalFileSystem fixtures
# =========================================================================


@pytest.fixture
async def local_fs(tmp_path: Path) -> AsyncIterator[LocalFileSystem]:
    data_dir = tmp_path / ".grover_test"
    lfs = LocalFileSystem(workspace_dir=tmp_path, data_dir=data_dir)
    await lfs.open()
    yield lfs
    await lfs.close()


@pytest.fixture
async def local_session(local_fs: LocalFileSystem) -> AsyncIterator[AsyncSession]:
    assert local_fs.session_factory is not None
    async with local_fs.session_factory() as session:
        yield session


async def _seed_local(lfs: LocalFileSystem, session: AsyncSession) -> None:
    """Create files on disk + DB for local FS testing."""
    await lfs.mkdir("/src", session=session)
    await lfs.mkdir("/src/sub", session=session)
    await lfs.mkdir("/docs", session=session)
    await session.commit()

    await lfs.write("/src/main.py", "def main():\n    print('hello')\n", session=session)
    await lfs.write("/src/utils.py", "def helper():\n    return 42\n", session=session)
    await lfs.write("/src/sub/deep.py", "# deep module\nclass Deep:\n    pass\n", session=session)
    await lfs.write("/docs/readme.md", "# README\nThis is a test project.\n", session=session)
    await lfs.write("/docs/guide.txt", "Step 1: install\nStep 2: run\n", session=session)
    await session.commit()


# =========================================================================
# LocalFileSystem: glob
# =========================================================================


class TestLocalGlob:
    async def test_star_pattern(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.glob("*.py", "/src", session=local_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        assert "/src/sub/deep.py" not in paths

    async def test_double_star(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.glob("**/*.py", "/src", session=local_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub/deep.py" in paths

    async def test_empty_result(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.glob("*.rs", "/src", session=local_session)
        assert result.success
        assert len(result.entries) == 0

    async def test_nonexistent_directory(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        result = await local_fs.glob("*.py", "/nonexistent", session=local_session)
        assert not result.success


# =========================================================================
# LocalFileSystem: grep
# =========================================================================


class TestLocalGrep:
    async def test_basic_regex(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("def ", "/src", session=local_session)
        assert result.success
        assert result.files_matched >= 2

    async def test_fixed_string(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep(
            "print('hello')", "/src", fixed_string=True, session=local_session
        )
        assert result.success
        assert result.files_matched == 1

    async def test_case_insensitive(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("readme", "/docs", case_sensitive=False, session=local_session)
        assert result.success
        assert result.files_matched >= 1

    async def test_invalid_regex(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("[invalid", "/src", session=local_session)
        assert not result.success

    async def test_glob_filter(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("Step", "/docs", glob_filter="*.txt", session=local_session)
        assert result.success
        assert result.files_matched == 1

    async def test_nonexistent_directory(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        result = await local_fs.grep("def", "/nonexistent", session=local_session)
        assert not result.success
        assert "not found" in result.message.lower()

    async def test_invert(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("def", "/src/main.py", invert=True, session=local_session)
        assert result.success
        assert len(result.matches) >= 1
        for m in result.matches:
            assert "def" not in m.line_content

    async def test_context_lines(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep(
            "print", "/src/main.py", context_lines=1, session=local_session
        )
        assert result.success
        assert len(result.matches) == 1
        m = result.matches[0]
        assert len(m.context_before) == 1
        assert "def main" in m.context_before[0]

    async def test_word_match(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("main", "/src", word_match=True, session=local_session)
        assert result.success
        assert result.files_matched >= 1

    async def test_count_only(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("def", "/src", count_only=True, session=local_session)
        assert result.success
        assert "Count:" in result.message
        assert len(result.matches) == 0

    async def test_files_only(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("def", "/src", files_only=True, session=local_session)
        assert result.success
        file_paths = [m.file_path for m in result.matches]
        assert len(file_paths) == len(set(file_paths))

    async def test_max_results(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep(".", "/", max_results=2, session=local_session)
        assert result.success
        assert len(result.matches) <= 2

    async def test_max_results_per_file(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep(
            ".", "/src/main.py", max_results_per_file=1, session=local_session
        )
        assert result.success
        main_matches = [m for m in result.matches if m.file_path == "/src/main.py"]
        assert len(main_matches) <= 1

    async def test_grep_single_file(
        self, local_fs: LocalFileSystem, local_session: AsyncSession
    ) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.grep("def main", "/src/main.py", session=local_session)
        assert result.success
        assert result.files_matched == 1
        assert len(result.matches) == 1


# =========================================================================
# LocalFileSystem: tree
# =========================================================================


class TestLocalTree:
    async def test_full_tree(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.tree("/", session=local_session)
        assert result.success
        assert result.total_files >= 5
        assert result.total_dirs >= 3

    async def test_max_depth(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.tree("/src", max_depth=1, session=local_session)
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/src/main.py" in paths
        assert "/src/sub" in paths
        assert "/src/sub/deep.py" not in paths

    async def test_sorted(self, local_fs: LocalFileSystem, local_session: AsyncSession) -> None:
        await _seed_local(local_fs, local_session)
        result = await local_fs.tree("/", session=local_session)
        assert result.success
        paths = [e.path for e in result.entries]
        assert paths == sorted(paths)


# =========================================================================
# VFS cross-mount aggregation
# =========================================================================


@pytest.fixture
async def vfs_setup(tmp_path: Path) -> AsyncIterator[tuple[VFS, AsyncEngine]]:
    """VFS with a DatabaseFS at /db and a LocalFS at /local."""
    # DB backend
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    db_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dfs = DatabaseFileSystem(dialect="sqlite")

    # Local backend
    local_dir = tmp_path / "workspace"
    local_dir.mkdir()
    data_dir = tmp_path / ".grover_vfs"
    lfs = LocalFileSystem(workspace_dir=local_dir, data_dir=data_dir)
    await lfs.open()

    registry = MountRegistry()
    registry.add_mount(
        MountConfig(
            mount_path="/db",
            backend=dfs,
            session_factory=db_factory,
            mount_type="database",
        )
    )
    registry.add_mount(
        MountConfig(
            mount_path="/local",
            backend=lfs,
            session_factory=lfs.session_factory,
            mount_type="local",
        )
    )

    vfs = VFS(registry)

    # Seed both mounts
    await vfs.write("/db/hello.py", "print('hello from db')\n")
    await vfs.write("/db/data.txt", "some data\n")
    await vfs.write("/local/world.py", "print('hello from local')\n")
    await vfs.write("/local/notes.txt", "notes here\n")

    yield vfs, engine

    await lfs.close()
    await engine.dispose()


class TestVfsGlob:
    async def test_root_glob_aggregates(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.glob("**/*.py", "/")
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/db/hello.py" in paths
        assert "/local/world.py" in paths

    async def test_mount_specific_glob(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.glob("*.py", "/db")
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/db/hello.py" in paths
        assert "/local/world.py" not in paths


class TestVfsGrep:
    async def test_root_grep_aggregates(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.grep("hello")
        assert result.success
        file_paths = {m.file_path for m in result.matches}
        assert "/db/hello.py" in file_paths
        assert "/local/world.py" in file_paths

    async def test_mount_specific_grep(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.grep("hello", "/db")
        assert result.success
        file_paths = {m.file_path for m in result.matches}
        assert "/db/hello.py" in file_paths
        assert "/local/world.py" not in file_paths

    async def test_max_results_across_mounts(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.grep(".", "/", max_results=1)
        assert result.success
        assert len(result.matches) <= 1

    async def test_count_only_at_root(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.grep("hello", "/", count_only=True)
        assert result.success
        assert "Count:" in result.message
        assert len(result.matches) == 0
        # Count should reflect actual matches across mounts
        count = int(result.message.split(":")[1].strip())
        assert count >= 2  # "hello" appears in both mounts
        assert result.files_matched >= 2


class TestVfsTree:
    async def test_root_tree_includes_mounts(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.tree("/")
        assert result.success
        paths = {e.path for e in result.entries}
        # Mount roots should be present
        assert "/db" in paths
        assert "/local" in paths
        # Files within mounts should also be present
        assert "/db/hello.py" in paths
        assert "/local/world.py" in paths

    async def test_mount_specific_tree(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.tree("/db")
        assert result.success
        paths = {e.path for e in result.entries}
        assert "/db/hello.py" in paths
        assert "/local/world.py" not in paths

    async def test_root_tree_max_depth_0(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.tree("/", max_depth=0)
        assert result.success
        # With depth 0 at root, should only show mount roots (no children)
        paths = {e.path for e in result.entries}
        # Mount roots are depth 0
        assert "/db" in paths
        assert "/local" in paths
        # No files inside mounts
        assert "/db/hello.py" not in paths

    async def test_root_tree_max_depth_1(self, vfs_setup: tuple[VFS, AsyncEngine]) -> None:
        vfs, _ = vfs_setup
        result = await vfs.tree("/", max_depth=1)
        assert result.success
        paths = {e.path for e in result.entries}
        # Mount roots (depth 0) and their children (depth 1)
        assert "/db" in paths
        assert "/local" in paths
        assert "/db/hello.py" in paths
        assert "/local/world.py" in paths

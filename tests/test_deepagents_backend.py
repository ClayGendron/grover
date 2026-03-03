"""Tests for GroverBackend — deepagents BackendProtocol implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from _helpers import FakeProvider

da = pytest.importorskip("deepagents")

from grover.fs.local_fs import LocalFileSystem  # noqa: E402
from grover.grover import Grover  # noqa: E402
from grover.grover_async import GroverAsync  # noqa: E402
from grover.integrations.deepagents._backend import GroverBackend  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
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
def grover(workspace: Path, tmp_path: Path) -> Iterator[Grover]:
    data = tmp_path / "grover_data"
    g = Grover()
    g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g
    g.close()


@pytest.fixture
def backend(grover: Grover) -> GroverBackend:
    return GroverBackend(grover)


@pytest.fixture
async def grover_async(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data_async"
    g = GroverAsync()
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def backend_async(grover_async: GroverAsync) -> GroverBackend:
    return GroverBackend(grover_async)


# ==================================================================
# ls_info
# ==================================================================


class TestLsInfo:
    def test_ls_info_returns_file_info_dicts(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/a.txt", "a")
        grover.write("/project/b.txt", "b")
        infos = backend.ls_info("/project")
        assert isinstance(infos, list)
        assert len(infos) >= 2
        paths = {fi["path"] for fi in infos}
        assert "/project/a.txt" in paths
        assert "/project/b.txt" in paths
        # Check TypedDict shape
        for fi in infos:
            assert "path" in fi

    def test_ls_info_empty_dir(self, backend: GroverBackend):
        infos = backend.ls_info("/project")
        assert isinstance(infos, list)
        assert len(infos) == 0


# ==================================================================
# read
# ==================================================================


class TestRead:
    def test_read_returns_numbered_lines(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/hello.txt", "line one\nline two\nline three")
        result = backend.read("/project/hello.txt")
        assert isinstance(result, str)
        # Should be in cat -n format with line numbers
        assert "1" in result
        assert "line one" in result
        assert "line two" in result
        assert "line three" in result
        # Verify tab-separated format
        assert "\t" in result

    def test_read_with_offset_and_limit(self, backend: GroverBackend, grover: Grover):
        lines = "\n".join(f"line {i}" for i in range(20))
        grover.write("/project/lines.txt", lines)
        result = backend.read("/project/lines.txt", offset=5, limit=3)
        assert "line 5" in result
        assert "line 7" in result
        # line 8 should not be included (offset=5, limit=3 → lines 5,6,7)
        assert "line 8" not in result

    def test_read_missing_file_returns_error_string(self, backend: GroverBackend):
        result = backend.read("/project/nonexistent.txt")
        assert isinstance(result, str)
        assert "Error" in result or "error" in result

    def test_read_truncates_long_lines(self, backend: GroverBackend, grover: Grover):
        # deepagents MAX_LINE_LENGTH is 5000; lines longer get chunked
        long_line = "x" * 6000
        grover.write("/project/long.txt", long_line)
        result = backend.read("/project/long.txt")
        # Should contain continuation markers (e.g., "1.1")
        assert "1" in result
        assert "x" in result


# ==================================================================
# write (create-only)
# ==================================================================


class TestWrite:
    def test_write_creates_new_file(self, backend: GroverBackend, grover: Grover):
        result = backend.write("/project/new.txt", "hello")
        assert result.error is None
        assert result.path == "/project/new.txt"
        # Verify file was actually created
        assert grover.read("/project/new.txt").content == "hello"

    def test_write_existing_file_returns_error(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/exists.txt", "original")
        result = backend.write("/project/exists.txt", "new content")
        assert result.error is not None
        # Original content unchanged
        assert grover.read("/project/exists.txt").content == "original"

    def test_write_returns_files_update_none(self, backend: GroverBackend):
        result = backend.write("/project/test.txt", "content")
        assert result.files_update is None


# ==================================================================
# edit
# ==================================================================


class TestEdit:
    def test_edit_replaces_string(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/doc.txt", "hello world")
        result = backend.edit("/project/doc.txt", "hello", "goodbye")
        assert result.error is None
        assert result.path == "/project/doc.txt"
        assert result.files_update is None
        assert grover.read("/project/doc.txt").content == "goodbye world"

    def test_edit_replace_all(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/multi.txt", "foo bar foo baz foo")
        result = backend.edit("/project/multi.txt", "foo", "qux", replace_all=True)
        assert result.error is None
        assert grover.read("/project/multi.txt").content == "qux bar qux baz qux"

    def test_edit_missing_file_returns_error(self, backend: GroverBackend):
        result = backend.edit("/project/nope.txt", "old", "new")
        assert result.error is not None

    def test_edit_string_not_found_returns_error(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/doc.txt", "hello world")
        result = backend.edit("/project/doc.txt", "xyz_not_here", "replaced")
        assert result.error is not None

    def test_edit_non_unique_without_replace_all_returns_error(
        self, backend: GroverBackend, grover: Grover
    ):
        grover.write("/project/dup.txt", "foo bar foo")
        result = backend.edit("/project/dup.txt", "foo", "baz")
        assert result.error is not None


# ==================================================================
# grep_raw
# ==================================================================


class TestGrepRaw:
    def test_grep_raw_literal_search(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/code.py", "def hello():\n    return 42\n")
        result = backend.grep_raw("hello", "/project")
        assert isinstance(result, list)
        assert len(result) >= 1
        match = result[0]
        assert match["path"] == "/project/code.py"
        assert match["line"] == 1
        assert "hello" in match["text"]

    def test_grep_raw_with_glob_filter(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/code.py", "hello world\n")
        grover.write("/project/readme.txt", "hello docs\n")
        result = backend.grep_raw("hello", "/project", glob="*.py")
        assert isinstance(result, list)
        paths = {m["path"] for m in result}
        assert "/project/code.py" in paths
        assert "/project/readme.txt" not in paths

    def test_grep_raw_no_matches(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/code.py", "def hello():\n    pass\n")
        result = backend.grep_raw("nonexistent_string", "/project")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_grep_raw_error_returns_string(self, backend: GroverBackend):
        result = backend.grep_raw("pattern", "/project/../etc/passwd")
        assert isinstance(result, str)
        assert "Error" in result or "error" in result


# ==================================================================
# glob_info
# ==================================================================


class TestGlobInfo:
    def test_glob_info_matches_pattern(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/main.py", "code")
        grover.write("/project/test.py", "test")
        grover.write("/project/readme.txt", "text")
        result = backend.glob_info("*.py", "/project")
        assert isinstance(result, list)
        paths = {fi["path"] for fi in result}
        assert "/project/main.py" in paths
        assert "/project/test.py" in paths
        assert "/project/readme.txt" not in paths

    def test_glob_info_no_matches(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/code.py", "code")
        result = backend.glob_info("*.rs", "/project")
        assert isinstance(result, list)
        assert len(result) == 0


# ==================================================================
# upload_files / download_files
# ==================================================================


class TestUploadDownload:
    def test_upload_files_creates_files(self, backend: GroverBackend, grover: Grover):
        files = [
            ("/project/up1.txt", b"content one"),
            ("/project/up2.txt", b"content two"),
        ]
        responses = backend.upload_files(files)
        assert len(responses) == 2
        for resp in responses:
            assert resp.error is None
        assert grover.read("/project/up1.txt").content == "content one"
        assert grover.read("/project/up2.txt").content == "content two"

    def test_upload_files_existing_returns_error(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/exists.txt", "original")
        responses = backend.upload_files([("/project/exists.txt", b"new")])
        assert len(responses) == 1
        assert responses[0].error is not None

    def test_download_files_returns_bytes(self, backend: GroverBackend, grover: Grover):
        grover.write("/project/dl.txt", "download me")
        responses = backend.download_files(["/project/dl.txt"])
        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"download me"

    def test_download_files_missing_returns_error(self, backend: GroverBackend):
        responses = backend.download_files(["/project/nope.txt"])
        assert len(responses) == 1
        assert responses[0].error is not None


# ==================================================================
# Path validation
# ==================================================================


class TestPathValidation:
    def test_path_validation_rejects_traversal(self, backend: GroverBackend):
        # write
        result = backend.write("/../etc/passwd", "bad")
        assert result.error is not None
        # read
        result_str = backend.read("/../etc/passwd")
        assert "Error" in result_str
        # edit
        edit_result = backend.edit("/../etc/passwd", "old", "new")
        assert edit_result.error is not None

    def test_path_validation_rejects_tilde(self, backend: GroverBackend):
        result = backend.write("~/bad.txt", "bad")
        assert result.error is not None

    def test_path_validation_rejects_no_leading_slash(self, backend: GroverBackend):
        result = backend.write("relative/path.txt", "bad")
        assert result.error is not None


# ==================================================================
# Factories
# ==================================================================


class TestFactories:
    def test_from_local_factory(self, tmp_path: Path):
        ws = tmp_path / "factory_ws"
        ws.mkdir()
        backend = GroverBackend.from_local(str(ws), data_dir=str(tmp_path / "data"))
        try:
            result = backend.write("/test.txt", "hello")
            assert result.error is None
            content = backend.read("/test.txt")
            assert "hello" in content
        finally:
            backend.grover.close()

    def test_from_database_factory(self, tmp_path: Path):
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        backend = GroverBackend.from_database(engine)
        try:
            result = backend.write("/test.txt", "hello")
            assert result.error is None
            content = backend.read("/test.txt")
            assert "hello" in content
        finally:
            backend.grover.close()


# ==================================================================
# is_async flag
# ==================================================================


class TestIsAsyncFlag:
    def test_is_async_false_with_grover(self, backend: GroverBackend):
        assert backend._is_async is False

    async def test_is_async_true_with_grover_async(self, backend_async: GroverBackend):
        assert backend_async._is_async is True


# ==================================================================
# Async native methods (GroverAsync)
# ==================================================================


class TestAsyncNative:
    async def test_als_info(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/a.txt", "a")
        infos = await backend_async.als_info("/project")
        assert isinstance(infos, list)
        paths = {fi["path"] for fi in infos}
        assert "/project/a.txt" in paths

    async def test_aread(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/hello.txt", "line one\nline two")
        result = await backend_async.aread("/project/hello.txt")
        assert isinstance(result, str)
        assert "line one" in result
        assert "line two" in result

    async def test_awrite(self, backend_async: GroverBackend, grover_async: GroverAsync):
        result = await backend_async.awrite("/project/new.txt", "async hello")
        assert result.error is None
        assert result.path == "/project/new.txt"

    async def test_aedit(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/doc.txt", "hello world")
        result = await backend_async.aedit("/project/doc.txt", "hello", "goodbye")
        assert result.error is None
        read = await grover_async.read("/project/doc.txt")
        assert read.content == "goodbye world"

    async def test_agrep_raw(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/code.py", "def hello():\n    return 42\n")
        result = await backend_async.agrep_raw("hello", "/project")
        assert isinstance(result, list)
        assert len(result) >= 1

    async def test_aglob_info(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/main.py", "code")
        await grover_async.write("/project/readme.txt", "text")
        result = await backend_async.aglob_info("*.py", "/project")
        assert isinstance(result, list)
        paths = {fi["path"] for fi in result}
        assert "/project/main.py" in paths
        assert "/project/readme.txt" not in paths

    async def test_aupload_files(self, backend_async: GroverBackend, grover_async: GroverAsync):
        responses = await backend_async.aupload_files([("/project/up.txt", b"async content")])
        assert len(responses) == 1
        assert responses[0].error is None

    async def test_adownload_files(self, backend_async: GroverBackend, grover_async: GroverAsync):
        await grover_async.write("/project/dl.txt", "download me")
        responses = await backend_async.adownload_files(["/project/dl.txt"])
        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"download me"


# ==================================================================
# TypeError when calling async methods with sync Grover
# ==================================================================


class TestAsyncTypeError:
    async def test_als_info_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.als_info("/project")

    async def test_aread_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.aread("/project/file.txt")

    async def test_awrite_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.awrite("/project/file.txt", "content")

    async def test_aedit_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.aedit("/project/file.txt", "old", "new")

    async def test_agrep_raw_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.agrep_raw("pattern", "/project")

    async def test_aglob_info_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.aglob_info("*.py", "/project")

    async def test_aupload_files_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.aupload_files([("/project/file.txt", b"content")])

    async def test_adownload_files_raises_type_error(self, backend: GroverBackend):
        with pytest.raises(TypeError, match="Async methods require GroverAsync"):
            await backend.adownload_files(["/project/file.txt"])


# ==================================================================
# Sync wrapper tests (GroverAsync backend, sync methods via asyncio.run)
# ==================================================================


def _make_sync_backend(tmp_path: Path) -> tuple[GroverBackend, GroverAsync]:
    """Create a GroverAsync-backed GroverBackend outside an event loop."""
    data = tmp_path / "grover_data_sync_wrapper"
    ws = tmp_path / "workspace_sync_wrapper"
    ws.mkdir(exist_ok=True)

    async def _setup() -> GroverAsync:
        g = GroverAsync()
        await g.add_mount(
            "/project",
            LocalFileSystem(workspace_dir=ws, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        return g

    ga = asyncio.run(_setup())
    return GroverBackend(ga), ga


class TestSyncWrapper:
    def test_ls_info_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/a.txt", "a"))
            infos = backend.ls_info("/project")
            assert isinstance(infos, list)
            paths = {fi["path"] for fi in infos}
            assert "/project/a.txt" in paths
        finally:
            asyncio.run(ga.close())

    def test_read_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/hello.txt", "hello world"))
            result = backend.read("/project/hello.txt")
            assert "hello world" in result
        finally:
            asyncio.run(ga.close())

    def test_write_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            result = backend.write("/project/new.txt", "sync via async")
            assert result.error is None
        finally:
            asyncio.run(ga.close())

    def test_edit_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/doc.txt", "hello world"))
            result = backend.edit("/project/doc.txt", "hello", "goodbye")
            assert result.error is None
        finally:
            asyncio.run(ga.close())

    def test_grep_raw_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/code.py", "def hello():\n    pass\n"))
            result = backend.grep_raw("hello", "/project")
            assert isinstance(result, list)
            assert len(result) >= 1
        finally:
            asyncio.run(ga.close())

    def test_glob_info_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/main.py", "code"))
            result = backend.glob_info("*.py", "/project")
            assert isinstance(result, list)
            paths = {fi["path"] for fi in result}
            assert "/project/main.py" in paths
        finally:
            asyncio.run(ga.close())

    def test_upload_files_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            responses = backend.upload_files([("/project/up.txt", b"content")])
            assert len(responses) == 1
            assert responses[0].error is None
        finally:
            asyncio.run(ga.close())

    def test_download_files_sync_wrapper(self, tmp_path: Path):
        backend, ga = _make_sync_backend(tmp_path)
        try:
            asyncio.run(ga.write("/project/dl.txt", "download"))
            responses = backend.download_files(["/project/dl.txt"])
            assert len(responses) == 1
            assert responses[0].content == b"download"
        finally:
            asyncio.run(ga.close())


# ==================================================================
# Async factory tests
# ==================================================================


class TestAsyncFactories:
    async def test_from_local_async_factory(self, tmp_path: Path):
        ws = tmp_path / "factory_async_ws"
        ws.mkdir()
        backend = await GroverBackend.from_local_async(str(ws), data_dir=str(tmp_path / "data"))
        try:
            result = await backend.awrite("/test.txt", "async hello")
            assert result.error is None
            content = await backend.aread("/test.txt")
            assert "async hello" in content
        finally:
            await backend.grover.close()

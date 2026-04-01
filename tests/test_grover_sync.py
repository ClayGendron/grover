"""Tests for Grover sync wrapper and raise_on_error integration."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.client import Grover
from grover.exceptions import (
    GraphError,
    GroverError,
    MountError,
    NotFoundError,
    ValidationError,
    WriteConflictError,
    _classify_error,
)
from grover.results import GroverResult

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _sqlite_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


@pytest.fixture
def g():
    grover = Grover()
    grover.add_mount("data", engine_url="sqlite+aiosqlite://")
    yield grover
    grover.close()


# ==================================================================
# Exception hierarchy
# ==================================================================


class TestExceptionHierarchy:
    def test_not_found_is_grover_error(self):
        assert issubclass(NotFoundError, GroverError)

    def test_mount_error_is_grover_error(self):
        assert issubclass(MountError, GroverError)

    def test_write_conflict_is_grover_error(self):
        assert issubclass(WriteConflictError, GroverError)

    def test_validation_is_grover_error(self):
        assert issubclass(ValidationError, GroverError)

    def test_graph_error_is_grover_error(self):
        assert issubclass(GraphError, GroverError)

    def test_catch_base_catches_subclass(self):
        with pytest.raises(GroverError):
            raise NotFoundError("gone")

    def test_error_has_result_attribute(self):
        result = GroverResult(success=False, errors=["oops"])
        e = GroverError("oops", result)
        assert e.result is result

    def test_error_result_defaults_to_none(self):
        e = GroverError("oops")
        assert e.result is None


# ==================================================================
# _classify_error
# ==================================================================


class TestClassifyError:
    def _result(self, *errors: str) -> GroverResult:
        return GroverResult(success=False, errors=list(errors))

    def test_not_found(self):
        r = self._result("Not found: /x.txt")
        assert isinstance(_classify_error(r.error_message, r.errors, r), NotFoundError)

    def test_not_a_directory(self):
        r = self._result("Not a directory: /x")
        assert isinstance(_classify_error(r.error_message, r.errors, r), NotFoundError)

    def test_no_mount(self):
        r = self._result("No mount found for path: /x")
        assert isinstance(_classify_error(r.error_message, r.errors, r), MountError)

    def test_already_exists(self):
        r = self._result("Already exists (overwrite=False): /x")
        assert isinstance(_classify_error(r.error_message, r.errors, r), WriteConflictError)

    def test_cannot_write(self):
        r = self._result("Cannot write to root path")
        assert isinstance(_classify_error(r.error_message, r.errors, r), WriteConflictError)

    def test_cannot_delete(self):
        r = self._result("Cannot delete root path")
        assert isinstance(_classify_error(r.error_message, r.errors, r), WriteConflictError)

    def test_invalid_pattern(self):
        r = self._result("Invalid glob pattern: [")
        assert isinstance(_classify_error(r.error_message, r.errors, r), ValidationError)

    def test_requires_missing(self):
        r = self._result("edit requires old and new strings")
        assert isinstance(_classify_error(r.error_message, r.errors, r), ValidationError)

    def test_graph_failed(self):
        r = self._result("predecessors failed: KeyError('x')")
        assert isinstance(_classify_error(r.error_message, r.errors, r), GraphError)

    def test_unknown_falls_back_to_base(self):
        r = self._result("something unexpected")
        e = _classify_error(r.error_message, r.errors, r)
        assert type(e) is GroverError

    def test_result_attached_to_exception(self):
        r = self._result("Not found: /x")
        e = _classify_error(r.error_message, r.errors, r)
        assert e.result is r


# ==================================================================
# raise_on_error on GroverFileSystem directly
# ==================================================================


class TestRaiseOnErrorFlag:
    async def test_error_returns_result_when_false(self):
        engine = await _sqlite_engine()
        try:
            fs = DatabaseFileSystem(engine=engine)
            assert fs._raise_on_error is False
            r = fs._error("test error")
            assert isinstance(r, GroverResult)
            assert not r.success
        finally:
            await engine.dispose()

    async def test_error_raises_when_true(self):
        engine = await _sqlite_engine()
        try:
            fs = DatabaseFileSystem(engine=engine)
            fs._raise_on_error = True
            with pytest.raises(GroverError, match="test error"):
                fs._error("test error")
        finally:
            await engine.dispose()

    async def test_mount_propagates_raise_on_error(self):
        from grover.base import GroverFileSystem

        router = GroverFileSystem(storage=False, raise_on_error=True)
        engine = await _sqlite_engine()
        try:
            child = DatabaseFileSystem(engine=engine)
            assert child._raise_on_error is False
            await router.add_mount("/data", child)
            assert child._raise_on_error is True
        finally:
            await engine.dispose()


# ==================================================================
# Grover sync wrapper — construction and lifecycle
# ==================================================================


class TestGroverConstruction:
    def test_creates_with_defaults(self):
        g = Grover()
        assert g._async._raise_on_error is True
        assert g._thread.is_alive()
        g.close()

    def test_close_is_idempotent(self):
        g = Grover()
        g.close()
        g.close()  # should not raise

    def test_close_joins_thread(self):
        g = Grover()
        thread = g._thread
        g.close()
        assert not thread.is_alive()


# ==================================================================
# Grover sync wrapper — mount management
# ==================================================================


class TestGroverMount:
    def test_add_mount_with_engine_url(self):
        g = Grover()
        g.add_mount("data", engine_url="sqlite+aiosqlite://")
        assert "/data" in g._async._mounts
        g.close()

    def test_add_mount_no_args_raises(self):
        g = Grover()
        try:
            with pytest.raises(ValueError, match="requires one of"):
                g.add_mount("data")
        finally:
            g.close()

    def test_remove_mount(self):
        g = Grover()
        g.add_mount("data", engine_url="sqlite+aiosqlite://")
        g.remove_mount("data")
        assert "/data" not in g._async._mounts
        g.close()


# ==================================================================
# Grover sync wrapper — CRUD return types and error raising
# ==================================================================


class TestGroverCRUD:
    def test_write_and_read_roundtrip(self, g: Grover):
        c = g.write("/data/hello.txt", "hello world")
        assert c.path == "/data/hello.txt"

        c = g.read("/data/hello.txt")
        assert c.content == "hello world"

    def test_read_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        g.write("/data/test.txt", "x")
        c = g.read("/data/test.txt")
        assert isinstance(c, Candidate)

    def test_write_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        c = g.write("/data/test.txt", "x")
        assert isinstance(c, Candidate)

    def test_read_not_found_raises(self, g: Grover):
        with pytest.raises(NotFoundError, match="Not found"):
            g.read("/data/nonexistent.txt")

    def test_read_unmounted_path_raises(self, g: Grover):
        with pytest.raises(MountError, match="No mount found"):
            g.read("/unknown/file.txt")

    def test_write_overwrite_false_raises(self, g: Grover):
        g.write("/data/exists.txt", "first")
        with pytest.raises(WriteConflictError, match="Already exists"):
            g.write("/data/exists.txt", "second", overwrite=False)

    def test_edit_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        g.write("/data/test.txt", "old text")
        c = g.edit("/data/test.txt", "old", "new")
        assert isinstance(c, Candidate)

    def test_delete_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        g.write("/data/test.txt", "x")
        c = g.delete("/data/test.txt")
        assert isinstance(c, Candidate)

    def test_mkdir_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        c = g.mkdir("/data/subdir")
        assert isinstance(c, Candidate)

    def test_stat_returns_candidate(self, g: Grover):
        from grover.results import Candidate

        g.write("/data/test.txt", "x")
        c = g.stat("/data/test.txt")
        assert isinstance(c, Candidate)


# ==================================================================
# Grover sync wrapper — search and listing
# ==================================================================


class TestGroverSearchAndListing:
    def test_glob_returns_grover_result(self, g: Grover):
        g.write("/data/a.py", "a")
        g.write("/data/b.py", "b")
        result = g.glob("**/*.py")
        assert isinstance(result, GroverResult)
        assert len(result.candidates) == 2

    def test_grep_returns_grover_result(self, g: Grover):
        g.write("/data/test.txt", "needle in haystack")
        result = g.grep("needle")
        assert isinstance(result, GroverResult)
        assert len(result.candidates) >= 1

    def test_set_algebra_works(self, g: Grover):
        g.write("/data/a.py", "import os")
        g.write("/data/b.py", "hello world")
        g.write("/data/c.txt", "import sys")

        py_files = g.glob("**/*.py")
        importers = g.grep("import")
        intersection = py_files & importers
        assert any("a.py" in c.path for c in intersection.candidates)
        assert not any("b.py" in c.path for c in intersection.candidates)

    def test_ls_returns_grover_result(self, g: Grover):
        g.mkdir("/data/subdir")
        result = g.ls("/data")
        assert isinstance(result, GroverResult)

    def test_tree_returns_grover_result(self, g: Grover):
        g.write("/data/a.txt", "a")
        result = g.tree("/data")
        assert isinstance(result, GroverResult)


# ==================================================================
# Grover sync wrapper — query engine
# ==================================================================


class TestGroverQuery:
    def test_run_query_returns_grover_result(self, g: Grover):
        g.write("/data/hello.py", "print('hi')")
        result = g.run_query('glob "**/*.py"')
        assert isinstance(result, GroverResult)
        assert any("hello.py" in c.path for c in result.candidates)

    def test_cli_returns_str(self, g: Grover):
        g.write("/data/hello.py", "print('hi')")
        output = g.cli('glob "**/*.py"')
        assert isinstance(output, str)
        assert "hello.py" in output

    def test_parse_query(self, g: Grover):
        plan = g.parse_query('glob "**/*.py"')
        assert plan is not None

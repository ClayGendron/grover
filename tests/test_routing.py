"""Tests for GroverFileSystem mount management and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from grover.base import GroverFileSystem
from grover.results import GroverResult
from tests.conftest import candidate as _candidate, dummy_session_factory, make_fs as _make_fs


class _RoutingFS(GroverFileSystem):
    def __init__(self, name: str = "test") -> None:
        super().__init__(session_factory=dummy_session_factory())
        self._name = name
        self.read_mock = AsyncMock(return_value=GroverResult())
        self.write_mock = AsyncMock(return_value=GroverResult())
        self.delete_mock = AsyncMock(return_value=GroverResult())
        self.glob_mock = AsyncMock(return_value=GroverResult())

    async def _read_impl(self, path=None, candidates=None, *, session):
        return await self.read_mock(path=path, candidates=candidates, session=session)

    async def _write_impl(self, path="", content="", overwrite=True, *, session):
        return await self.write_mock(path=path, content=content, overwrite=overwrite, session=session)

    async def _delete_impl(self, path=None, candidates=None, permanent=False, *, session):
        return await self.delete_mock(
            path=path,
            candidates=candidates,
            permanent=permanent,
            session=session,
        )

    async def _glob_impl(self, pattern="", candidates=None, *, session):
        return await self.glob_mock(pattern=pattern, candidates=candidates, session=session)


# =========================================================================
# add_mount
# =========================================================================


class TestAddMount:
    async def test_basic_mount(self):
        root = _make_fs("root")
        child = _make_fs("child")
        await root.add_mount("/data", child)
        assert "/data" in root._mounts
        assert root._mounts["/data"] is child

    async def test_root_mount_forbidden(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="owns its own root"):
            await root.add_mount("/", child)

    async def test_unnormalized_path_rejected(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="must be normalized"):
            await root.add_mount("/data/", child)

    async def test_unnormalized_path_hint(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="did you mean '/data'"):
            await root.add_mount("/data/", child)

    async def test_double_slash_rejected(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="must be normalized"):
            await root.add_mount("/data//deep", child)

    async def test_relative_path_rejected(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="must be normalized"):
            await root.add_mount("data", child)

    async def test_dot_segments_rejected(self):
        root = _make_fs()
        child = _make_fs()
        with pytest.raises(ValueError, match="must be normalized"):
            await root.add_mount("/data/../other", child)

    async def test_duplicate_mount_forbidden(self):
        root = _make_fs()
        await root.add_mount("/data", _make_fs())
        with pytest.raises(ValueError, match="already exists"):
            await root.add_mount("/data", _make_fs())

    async def test_nested_mounts_allowed(self):
        root = _make_fs()
        await root.add_mount("/data", _make_fs("shallow"))
        await root.add_mount("/data/archive", _make_fs("deep"))
        assert "/data" in root._mounts
        assert "/data/archive" in root._mounts


# =========================================================================
# remove_mount
# =========================================================================


class TestRemoveMount:
    async def test_basic_remove(self):
        root = _make_fs()
        await root.add_mount("/data", _make_fs())
        await root.remove_mount("/data")
        assert "/data" not in root._mounts

    async def test_remove_nonexistent_raises(self):
        root = _make_fs()
        with pytest.raises(ValueError, match="No mount at"):
            await root.remove_mount("/data")

    async def test_remove_unnormalized_no_false_hint(self):
        """Hint should NOT appear when the normalized form also doesn't exist."""
        root = _make_fs()
        with pytest.raises(ValueError, match="No mount at") as exc_info:
            await root.remove_mount("/data/")
        assert "did you mean" not in str(exc_info.value)

    async def test_remove_unnormalized_with_hint(self):
        """Hint SHOULD appear when the normalized form exists as a mount."""
        root = _make_fs()
        await root.add_mount("/data", _make_fs())
        with pytest.raises(ValueError, match="did you mean '/data'"):
            await root.remove_mount("/data/")


# =========================================================================
# _match_mount
# =========================================================================


class TestMatchMount:
    async def test_file_under_mount(self):
        root = _make_fs()
        child = _make_fs()
        await root.add_mount("/web", child)
        result = root._match_mount("/web/page.html")
        assert result == ("/web", child)

    async def test_exact_mount_path(self):
        root = _make_fs()
        child = _make_fs()
        await root.add_mount("/web", child)
        result = root._match_mount("/web")
        assert result == ("/web", child)

    async def test_no_match_returns_none(self):
        root = _make_fs()
        await root.add_mount("/web", _make_fs())
        assert root._match_mount("/src/a.py") is None

    async def test_root_returns_none(self):
        root = _make_fs()
        await root.add_mount("/web", _make_fs())
        assert root._match_mount("/") is None

    async def test_longest_prefix_wins(self):
        root = _make_fs()
        shallow = _make_fs("shallow")
        deep = _make_fs("deep")
        await root.add_mount("/data", shallow)
        await root.add_mount("/data/deep", deep)
        result = root._match_mount("/data/deep/file.txt")
        assert result == ("/data/deep", deep)

    async def test_shorter_prefix_for_non_deep_path(self):
        root = _make_fs()
        shallow = _make_fs("shallow")
        await root.add_mount("/data", shallow)
        await root.add_mount("/data/deep", _make_fs())
        result = root._match_mount("/data/other.txt")
        assert result == ("/data", shallow)

    async def test_prefix_boundary_not_substring(self):
        """``/webinar`` must NOT match mount ``/web``."""
        root = _make_fs()
        await root.add_mount("/web", _make_fs())
        assert root._match_mount("/webinar/page.html") is None

    async def test_prefix_boundary_exact_suffix(self):
        """``/web2`` must NOT match mount ``/web``."""
        root = _make_fs()
        await root.add_mount("/web", _make_fs())
        assert root._match_mount("/web2") is None

    def test_no_mounts_returns_none(self):
        root = _make_fs()
        assert root._match_mount("/anything") is None


# =========================================================================
# _resolve_terminal
# =========================================================================


class TestResolveTerminal:
    async def test_single_mount(self):
        root = _make_fs("root")
        db = _make_fs("db")
        await root.add_mount("/web", db)
        fs, rel, prefix = root._resolve_terminal("/web/page.html")
        assert fs is db
        assert rel == "/page.html"
        assert prefix == "/web"

    async def test_nested_mount(self):
        root = _make_fs("root")
        db = _make_fs("db")
        arc = _make_fs("archive")
        await root.add_mount("/data", db)
        await db.add_mount("/archive", arc)
        fs, rel, prefix = root._resolve_terminal("/data/archive/old.txt")
        assert fs is arc
        assert rel == "/old.txt"
        assert prefix == "/data/archive"

    async def test_three_level_nesting(self):
        root = _make_fs("root")
        a = _make_fs("a")
        b = _make_fs("b")
        c = _make_fs("c")
        await root.add_mount("/l1", a)
        await a.add_mount("/l2", b)
        await b.add_mount("/l3", c)
        fs, rel, prefix = root._resolve_terminal("/l1/l2/l3/file.txt")
        assert fs is c
        assert rel == "/file.txt"
        assert prefix == "/l1/l2/l3"

    async def test_unmounted_path_stays_on_root(self):
        root = _make_fs("root")
        await root.add_mount("/jira", _make_fs())
        fs, rel, prefix = root._resolve_terminal("/src/a.py")
        assert fs is root
        assert rel == "/src/a.py"
        assert prefix == ""

    async def test_root_path_stays_on_root(self):
        root = _make_fs("root")
        await root.add_mount("/web", _make_fs())
        fs, rel, prefix = root._resolve_terminal("/")
        assert fs is root
        assert rel == "/"
        assert prefix == ""

    async def test_exact_mount_path_resolves_to_root_of_child(self):
        root = _make_fs("root")
        db = _make_fs("db")
        await root.add_mount("/web", db)
        fs, rel, prefix = root._resolve_terminal("/web")
        assert fs is db
        assert rel == "/"
        assert prefix == "/web"

    async def test_input_normalized(self):
        """Unnormalized input is normalized before routing."""
        root = _make_fs("root")
        db = _make_fs("db")
        await root.add_mount("/web", db)
        fs, rel, prefix = root._resolve_terminal("web/page.html")
        assert fs is db
        assert rel == "/page.html"
        assert prefix == "/web"

    async def test_double_slash_normalized(self):
        root = _make_fs("root")
        db = _make_fs("db")
        await root.add_mount("/web", db)
        fs, rel, prefix = root._resolve_terminal("/web//page.html")
        assert fs is db
        assert rel == "/page.html"
        assert prefix == "/web"

    def test_no_mounts(self):
        root = _make_fs("root")
        fs, rel, prefix = root._resolve_terminal("/any/path")
        assert fs is root
        assert rel == "/any/path"
        assert prefix == ""

    async def test_prefix_boundary_not_confused(self):
        """``/webinar`` must not resolve through ``/web`` mount."""
        root = _make_fs("root")
        await root.add_mount("/web", _make_fs())
        fs, rel, prefix = root._resolve_terminal("/webinar")
        assert fs is root
        assert rel == "/webinar"
        assert prefix == ""


# =========================================================================
# Candidate routing regressions
# =========================================================================


class TestCandidateRouting:
    async def test_empty_candidate_read_short_circuits_dispatch(self):
        root = _RoutingFS("root")
        child = _RoutingFS("child")
        await root.add_mount("/data", child)

        result = await root.read(candidates=GroverResult())

        assert result.success is True
        assert result.candidates == []
        root.read_mock.assert_not_awaited()
        child.read_mock.assert_not_awaited()

    async def test_empty_candidate_glob_short_circuits_fanout(self):
        root = _RoutingFS("root")
        child = _RoutingFS("child")
        await root.add_mount("/data", child)

        result = await root.glob("*.py", candidates=GroverResult())

        assert result.success is True
        assert result.candidates == []
        root.glob_mock.assert_not_awaited()
        child.glob_mock.assert_not_awaited()


# =========================================================================
# Cross-mount transfer regressions
# =========================================================================


class TestCrossMountTransfers:
    async def test_cross_mount_copy_rebases_source_read_failures_to_source_prefix(self):
        root = _RoutingFS("root")
        src = _RoutingFS("src")
        dst = _RoutingFS("dst")
        await root.add_mount("/src", src)
        await root.add_mount("/dst", dst)

        src.read_mock.return_value = GroverResult(
            success=False,
            errors=["read failed"],
            candidates=[_candidate("/file.txt")],
        )

        result = await root.copy("/src/file.txt", "/dst/file.txt")

        assert result.success is False
        assert result.paths == ("/src/file.txt",)
        dst.write_mock.assert_not_awaited()

    async def test_cross_mount_move_rebases_delete_failures_to_source_prefix(self):
        root = _RoutingFS("root")
        src = _RoutingFS("src")
        dst = _RoutingFS("dst")
        await root.add_mount("/src", src)
        await root.add_mount("/dst", dst)

        src.read_mock.return_value = GroverResult(
            candidates=[_candidate("/file.txt", content="hello")],
        )
        dst.write_mock.return_value = GroverResult(
            candidates=[_candidate("/file.txt", content="hello")],
        )
        src.delete_mock.return_value = GroverResult(
            success=False,
            errors=["delete failed"],
            candidates=[_candidate("/file.txt")],
        )

        result = await root.move("/src/file.txt", "/dst/file.txt")

        assert result.success is False
        assert result.paths == ("/src/file.txt",)

    async def test_cross_mount_copy_keeps_destination_prefix_for_write_failures(self):
        root = _RoutingFS("root")
        src = _RoutingFS("src")
        dst = _RoutingFS("dst")
        await root.add_mount("/src", src)
        await root.add_mount("/dst", dst)

        src.read_mock.return_value = GroverResult(
            candidates=[_candidate("/file.txt", content="hello")],
        )
        dst.write_mock.return_value = GroverResult(
            success=False,
            errors=["write failed"],
            candidates=[_candidate("/file.txt")],
        )

        result = await root.copy("/src/file.txt", "/dst/file.txt")

        assert result.success is False
        assert result.paths == ("/dst/file.txt",)

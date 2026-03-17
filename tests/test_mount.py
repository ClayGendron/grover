"""Tests for the Mount class — minimal routing dataclass."""

from __future__ import annotations

from typing import Any

import pytest

from grover.mount import Mount
from grover.permissions import Permission

# ------------------------------------------------------------------
# Fake components for testing
# ------------------------------------------------------------------


class FakeFilesystem:
    """Minimal fake filesystem."""

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def read(self, path: str, **kw: Any) -> Any: ...

    async def write(self, path: str, content: str, **kw: Any) -> Any: ...


# ==================================================================
# Construction
# ==================================================================


class TestMountConstruction:
    def test_basic_construction(self):
        fs = FakeFilesystem()
        m = Mount(name="project", filesystem=fs)
        assert m.path == "/project"
        assert m.filesystem is fs

    def test_name_with_slash_rejected(self):
        with pytest.raises(ValueError, match="must not contain"):
            Mount(name="project/src", filesystem=FakeFilesystem())

    def test_trailing_slash_stripped(self):
        m = Mount(name="project/", filesystem=FakeFilesystem())
        assert m.name == "project"
        assert m.path == "/project"

    def test_default_label_from_path(self):
        m = Mount(name="my-project", filesystem=FakeFilesystem())
        assert m.label == "my-project"

    def test_custom_label(self):
        m = Mount(name="project", filesystem=FakeFilesystem(), label="My App")
        assert m.label == "My App"

    def test_default_permission(self):
        m = Mount(name="project", filesystem=FakeFilesystem())
        assert m.permission == Permission.READ_WRITE

    def test_custom_permission(self):
        m = Mount(
            name="project",
            filesystem=FakeFilesystem(),
            permission=Permission.READ_ONLY,
        )
        assert m.permission == Permission.READ_ONLY

    def test_hidden_default(self):
        m = Mount(name="project", filesystem=FakeFilesystem())
        assert m.hidden is False

    def test_hidden_true(self):
        m = Mount(name="project", filesystem=FakeFilesystem(), hidden=True)
        assert m.hidden is True

    def test_read_only_paths_default(self):
        m = Mount(name="project", filesystem=FakeFilesystem())
        assert m.read_only_paths == set()

    def test_read_only_paths_custom(self):
        m = Mount(
            name="project",
            filesystem=FakeFilesystem(),
            read_only_paths={"/project/locked"},
        )
        assert m.read_only_paths == {"/project/locked"}

    def test_mount_type_default(self):
        m = Mount(name="project", filesystem=FakeFilesystem())
        assert m.mount_type == "vfs"

    def test_no_graph_or_search_attributes(self):
        """Mount no longer has graph or search attributes."""
        m = Mount(name="project", filesystem=FakeFilesystem())
        assert not hasattr(m, "graph")
        assert not hasattr(m, "search")


# ==================================================================
# Repr
# ==================================================================


class TestMountRepr:
    def test_repr_basic(self):
        m = Mount(name="project", filesystem=FakeFilesystem())
        r = repr(m)
        assert "Mount(" in r
        assert "name='project'" in r
        assert "FakeFilesystem" in r

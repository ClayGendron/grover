"""Tests for MountRegistry and Mount."""

from __future__ import annotations

import pytest

from grover.exceptions import MountNotFoundError
from grover.mount import Mount, MountRegistry
from grover.permissions import Permission


class FakeBackend:
    """Minimal mock backend for mount tests."""

    pass


# ---------------------------------------------------------------------------
# MountRegistry — Basic Operations
# ---------------------------------------------------------------------------


class TestMountRegistry:
    def test_add_and_list(self):
        reg = MountRegistry()
        reg.add_mount(Mount(name="a", filesystem=FakeBackend()))
        reg.add_mount(Mount(name="b", filesystem=FakeBackend()))
        mounts = reg.list_mounts()
        assert len(mounts) == 2
        assert mounts[0].path == "/a"
        assert mounts[1].path == "/b"

    def test_remove_mount(self):
        reg = MountRegistry()
        reg.add_mount(Mount(name="a", filesystem=FakeBackend()))
        reg.remove_mount("/a")
        assert reg.list_mounts() == []

    def test_has_mount(self):
        reg = MountRegistry()
        reg.add_mount(Mount(name="data", filesystem=FakeBackend()))
        assert reg.has_mount("/data") is True
        assert reg.has_mount("/nope") is False


# ---------------------------------------------------------------------------
# MountRegistry — Resolution
# ---------------------------------------------------------------------------


class TestMountResolution:
    def test_basic_resolve(self):
        backend = FakeBackend()
        reg = MountRegistry()
        reg.add_mount(Mount(name="data", filesystem=backend))

        mount, rel = reg.resolve("/data/hello.txt")
        assert mount.filesystem is backend
        assert rel == "/hello.txt"

    def test_resolve_mount_root(self):
        reg = MountRegistry()
        reg.add_mount(Mount(name="data", filesystem=FakeBackend()))

        _mount, rel = reg.resolve("/data")
        assert rel == "/"

    def test_nested_mount_name_rejected(self):
        """Mount names cannot contain '/'."""
        with pytest.raises(ValueError, match="must not contain"):
            Mount(name="data/deep", filesystem=FakeBackend())

    def test_resolve_no_mount(self):
        reg = MountRegistry()
        with pytest.raises(MountNotFoundError, match="No mount"):
            reg.resolve("/unknown/path")

    def test_resolve_partial_name_no_match(self):
        """'/datafile' should NOT match mount at '/data'."""
        reg = MountRegistry()
        reg.add_mount(Mount(name="data", filesystem=FakeBackend()))

        with pytest.raises(MountNotFoundError):
            reg.resolve("/datafile")


# ---------------------------------------------------------------------------
# MountRegistry — Permissions
# ---------------------------------------------------------------------------


class TestMountPermissions:
    def test_default_permission(self):
        reg = MountRegistry()
        reg.add_mount(Mount(name="data", filesystem=FakeBackend()))
        assert reg.get_permission("/data/file.txt") == Permission.READ_WRITE

    def test_read_only_mount(self):
        reg = MountRegistry()
        reg.add_mount(
            Mount(
                name="data",
                filesystem=FakeBackend(),
                permission=Permission.READ_ONLY,
            )
        )
        assert reg.get_permission("/data/file.txt") == Permission.READ_ONLY

    def test_read_only_path_override(self):
        reg = MountRegistry()
        reg.add_mount(
            Mount(
                name="data",
                filesystem=FakeBackend(),
                read_only_paths={"/config"},
            )
        )

        # Normal paths are read-write
        assert reg.get_permission("/data/other.txt") == Permission.READ_WRITE
        # Config dir is read-only
        assert reg.get_permission("/data/config") == Permission.READ_ONLY
        # Children of config dir are also read-only
        assert reg.get_permission("/data/config/settings.json") == Permission.READ_ONLY

    def test_read_only_path_override_root_file(self):
        reg = MountRegistry()
        reg.add_mount(
            Mount(
                name="data",
                filesystem=FakeBackend(),
                read_only_paths={"/important.txt"},
            )
        )
        assert reg.get_permission("/data/important.txt") == Permission.READ_ONLY
        assert reg.get_permission("/data/other.txt") == Permission.READ_WRITE

"""Tests for MountRegistry and Mount."""

from __future__ import annotations

import pytest

from grover.fs.exceptions import MountNotFoundError
from grover.fs.mounts import MountRegistry
from grover.fs.permissions import Permission
from grover.mount import Mount


class FakeBackend:
    """Minimal mock backend for mount tests."""

    pass


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------


class TestMount:
    def test_normalize_path(self):
        cfg = Mount(path="/web/", filesystem=FakeBackend())
        assert cfg.path == "/web"

    def test_default_label(self):
        cfg = Mount(path="/data", filesystem=FakeBackend())
        assert cfg.label == "data"

    def test_custom_label(self):
        cfg = Mount(path="/data", filesystem=FakeBackend(), label="My Data")
        assert cfg.label == "My Data"

    def test_default_permission(self):
        cfg = Mount(path="/x", filesystem=FakeBackend())
        assert cfg.permission == Permission.READ_WRITE

    def test_read_only_mount(self):
        cfg = Mount(path="/x", filesystem=FakeBackend(), permission=Permission.READ_ONLY)
        assert cfg.permission == Permission.READ_ONLY

    def test_default_mount_type(self):
        cfg = Mount(path="/x", filesystem=FakeBackend())
        assert cfg.mount_type == "vfs"


# ---------------------------------------------------------------------------
# MountRegistry — Basic Operations
# ---------------------------------------------------------------------------


class TestMountRegistry:
    def test_add_and_list(self):
        reg = MountRegistry()
        reg.add_mount(Mount(path="/a", filesystem=FakeBackend()))
        reg.add_mount(Mount(path="/b", filesystem=FakeBackend()))
        mounts = reg.list_mounts()
        assert len(mounts) == 2
        assert mounts[0].path == "/a"
        assert mounts[1].path == "/b"

    def test_remove_mount(self):
        reg = MountRegistry()
        reg.add_mount(Mount(path="/a", filesystem=FakeBackend()))
        reg.remove_mount("/a")
        assert reg.list_mounts() == []

    def test_has_mount(self):
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=FakeBackend()))
        assert reg.has_mount("/data") is True
        assert reg.has_mount("/nope") is False


# ---------------------------------------------------------------------------
# MountRegistry — Resolution
# ---------------------------------------------------------------------------


class TestMountResolution:
    def test_basic_resolve(self):
        backend = FakeBackend()
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=backend))

        mount, rel = reg.resolve("/data/hello.txt")
        assert mount.filesystem is backend
        assert rel == "/hello.txt"

    def test_resolve_mount_root(self):
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=FakeBackend()))

        _mount, rel = reg.resolve("/data")
        assert rel == "/"

    def test_longest_prefix_match(self):
        backend_a = FakeBackend()
        backend_b = FakeBackend()
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=backend_a))
        reg.add_mount(Mount(path="/data/deep", filesystem=backend_b))

        mount, rel = reg.resolve("/data/deep/file.txt")
        assert mount.filesystem is backend_b
        assert rel == "/file.txt"

    def test_resolve_no_mount(self):
        reg = MountRegistry()
        with pytest.raises(MountNotFoundError, match="No mount"):
            reg.resolve("/unknown/path")

    def test_resolve_partial_name_no_match(self):
        """'/datafile' should NOT match mount at '/data'."""
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=FakeBackend()))

        with pytest.raises(MountNotFoundError):
            reg.resolve("/datafile")


# ---------------------------------------------------------------------------
# MountRegistry — Permissions
# ---------------------------------------------------------------------------


class TestMountPermissions:
    def test_default_permission(self):
        reg = MountRegistry()
        reg.add_mount(Mount(path="/data", filesystem=FakeBackend()))
        assert reg.get_permission("/data/file.txt") == Permission.READ_WRITE

    def test_read_only_mount(self):
        reg = MountRegistry()
        reg.add_mount(
            Mount(
                path="/data",
                filesystem=FakeBackend(),
                permission=Permission.READ_ONLY,
            )
        )
        assert reg.get_permission("/data/file.txt") == Permission.READ_ONLY

    def test_read_only_path_override(self):
        reg = MountRegistry()
        reg.add_mount(
            Mount(
                path="/data",
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
                path="/data",
                filesystem=FakeBackend(),
                read_only_paths={"/important.txt"},
            )
        )
        assert reg.get_permission("/data/important.txt") == Permission.READ_ONLY
        assert reg.get_permission("/data/other.txt") == Permission.READ_WRITE

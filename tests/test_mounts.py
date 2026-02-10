"""Tests for MountRegistry and MountConfig."""

from __future__ import annotations

import pytest

from grover.fs.exceptions import MountNotFoundError
from grover.fs.mounts import MountConfig, MountRegistry
from grover.fs.permissions import Permission


class FakeBackend:
    """Minimal mock backend for mount tests."""

    pass


# ---------------------------------------------------------------------------
# MountConfig
# ---------------------------------------------------------------------------


class TestMountConfig:
    def test_normalize_path(self):
        cfg = MountConfig(mount_path="/web/", backend=FakeBackend())
        assert cfg.mount_path == "/web"

    def test_default_label(self):
        cfg = MountConfig(mount_path="/data", backend=FakeBackend())
        assert cfg.label == "data"

    def test_custom_label(self):
        cfg = MountConfig(mount_path="/data", backend=FakeBackend(), label="My Data")
        assert cfg.label == "My Data"

    def test_default_permission(self):
        cfg = MountConfig(mount_path="/x", backend=FakeBackend())
        assert cfg.permission == Permission.READ_WRITE

    def test_read_only_mount(self):
        cfg = MountConfig(
            mount_path="/x", backend=FakeBackend(), permission=Permission.READ_ONLY
        )
        assert cfg.permission == Permission.READ_ONLY

    def test_default_mount_type(self):
        cfg = MountConfig(mount_path="/x", backend=FakeBackend())
        assert cfg.mount_type == "vfs"


# ---------------------------------------------------------------------------
# MountRegistry — Basic Operations
# ---------------------------------------------------------------------------


class TestMountRegistry:
    def test_add_and_list(self):
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/a", backend=FakeBackend()))
        reg.add_mount(MountConfig(mount_path="/b", backend=FakeBackend()))
        mounts = reg.list_mounts()
        assert len(mounts) == 2
        assert mounts[0].mount_path == "/a"
        assert mounts[1].mount_path == "/b"

    def test_remove_mount(self):
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/a", backend=FakeBackend()))
        reg.remove_mount("/a")
        assert reg.list_mounts() == []

    def test_has_mount(self):
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=FakeBackend()))
        assert reg.has_mount("/data") is True
        assert reg.has_mount("/nope") is False


# ---------------------------------------------------------------------------
# MountRegistry — Resolution
# ---------------------------------------------------------------------------


class TestMountResolution:
    def test_basic_resolve(self):
        backend = FakeBackend()
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=backend))

        mount, rel = reg.resolve("/data/hello.txt")
        assert mount.backend is backend
        assert rel == "/hello.txt"

    def test_resolve_mount_root(self):
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=FakeBackend()))

        _mount, rel = reg.resolve("/data")
        assert rel == "/"

    def test_longest_prefix_match(self):
        backend_a = FakeBackend()
        backend_b = FakeBackend()
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=backend_a))
        reg.add_mount(MountConfig(mount_path="/data/deep", backend=backend_b))

        mount, rel = reg.resolve("/data/deep/file.txt")
        assert mount.backend is backend_b
        assert rel == "/file.txt"

    def test_resolve_no_mount(self):
        reg = MountRegistry()
        with pytest.raises(MountNotFoundError, match="No mount"):
            reg.resolve("/unknown/path")

    def test_resolve_partial_name_no_match(self):
        """'/datafile' should NOT match mount at '/data'."""
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=FakeBackend()))

        with pytest.raises(MountNotFoundError):
            reg.resolve("/datafile")


# ---------------------------------------------------------------------------
# MountRegistry — Permissions
# ---------------------------------------------------------------------------


class TestMountPermissions:
    def test_default_permission(self):
        reg = MountRegistry()
        reg.add_mount(MountConfig(mount_path="/data", backend=FakeBackend()))
        assert reg.get_permission("/data/file.txt") == Permission.READ_WRITE

    def test_read_only_mount(self):
        reg = MountRegistry()
        reg.add_mount(
            MountConfig(
                mount_path="/data",
                backend=FakeBackend(),
                permission=Permission.READ_ONLY,
            )
        )
        assert reg.get_permission("/data/file.txt") == Permission.READ_ONLY

    def test_read_only_path_override(self):
        reg = MountRegistry()
        reg.add_mount(
            MountConfig(
                mount_path="/data",
                backend=FakeBackend(),
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
            MountConfig(
                mount_path="/data",
                backend=FakeBackend(),
                read_only_paths={"/important.txt"},
            )
        )
        assert reg.get_permission("/data/important.txt") == Permission.READ_ONLY
        assert reg.get_permission("/data/other.txt") == Permission.READ_WRITE

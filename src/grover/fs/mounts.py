"""MountRegistry and Mount integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .exceptions import MountNotFoundError
from .permissions import Permission
from .utils import normalize_path

if TYPE_CHECKING:
    from grover.mount import Mount


class MountRegistry:
    """Registry of active mount points.

    Resolves virtual paths to ``(Mount, relative_path)`` tuples
    and determines effective permissions for any path.
    """

    def __init__(self) -> None:
        self._mounts: dict[str, Mount] = {}

    def add_mount(self, config: Mount) -> None:
        """Add or replace a mount point."""
        self._mounts[config.path] = config

    def remove_mount(self, mount_path: str) -> None:
        """Remove a mount point."""
        mount_path = normalize_path(mount_path).rstrip("/")
        self._mounts.pop(mount_path, None)

    def resolve(self, virtual_path: str) -> tuple[Mount, str]:
        """Resolve a virtual path to its mount and relative path.

        Finds the longest matching mount prefix and strips it.
        """
        virtual_path = normalize_path(virtual_path)

        best_match: Mount | None = None
        best_len = -1

        for mount_path, config in self._mounts.items():
            if (virtual_path == mount_path or virtual_path.startswith(mount_path + "/")) and len(
                mount_path
            ) > best_len:
                best_match = config
                best_len = len(mount_path)

        if best_match is None:
            raise MountNotFoundError(f"No mount found for path: {virtual_path}")

        relative = virtual_path[best_len:]
        if not relative:
            relative = "/"
        elif not relative.startswith("/"):
            relative = "/" + relative

        return best_match, relative

    def list_mounts(self) -> list[Mount]:
        """List all registered mounts, sorted by path."""
        return sorted(self._mounts.values(), key=lambda m: m.path)

    def list_visible_mounts(self) -> list[Mount]:
        """List non-hidden mounts, sorted by path."""
        return [m for m in self.list_mounts() if not m.hidden]

    def get_permission(self, virtual_path: str) -> Permission:
        """Get the effective permission for a virtual path."""
        mount, relative = self.resolve(virtual_path)

        if mount.permission == Permission.READ_ONLY:
            return Permission.READ_ONLY

        rel_normalized = normalize_path(relative)
        current = rel_normalized
        while True:
            if current in mount.read_only_paths:
                return Permission.READ_ONLY
            if current == "/":
                break
            parent = current.rsplit("/", 1)[0] or "/"
            current = parent

        return mount.permission

    def get_mount(self, mount_path: str) -> Mount | None:
        """Return the Mount at *mount_path*, or ``None``."""
        mount_path = normalize_path(mount_path).rstrip("/")
        return self._mounts.get(mount_path)

    def has_mount(self, mount_path: str) -> bool:
        """Check if a mount exists at the given path."""
        return self.get_mount(mount_path) is not None

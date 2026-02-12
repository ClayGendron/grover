"""MountRegistry and MountConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .exceptions import MountNotFoundError
from .permissions import Permission
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from .protocol import StorageBackend


@dataclass
class MountConfig:
    """Configuration for a single mount point."""

    mount_path: str
    """Virtual path prefix, e.g. "/web", "/my_project"."""

    backend: StorageBackend
    """Storage backend implementing the StorageBackend protocol."""

    # Session factory — set by GroverAsync.mount() for both DB and local mounts
    session_factory: Callable[..., AsyncSession] | None = None
    """Async session factory for UFS-managed sessions.  ``None`` for standalone use."""

    permission: Permission = Permission.READ_WRITE
    """Default permission for this mount."""

    label: str = ""
    """Display name for the mount."""

    mount_type: str = "vfs"
    """Type identifier: "vfs" or "local"."""

    hidden: bool = False
    """If True, this mount is excluded from ``list_visible_mounts()``."""

    authenticated: bool = False
    """If True, operations require ``user_id`` and paths are auto-namespaced per user."""

    read_only_paths: set[str] = field(default_factory=set)
    """Paths within this mount that are forced read-only."""

    @property
    def has_session_factory(self) -> bool:
        """True when this mount has a UFS-managed session factory."""
        return self.session_factory is not None

    def __post_init__(self) -> None:
        self.mount_path = normalize_path(self.mount_path).rstrip("/")
        if not self.label:
            self.label = self.mount_path.lstrip("/") or "root"


class MountRegistry:
    """Registry of active mount points.

    Resolves virtual paths to (MountConfig, relative_path) tuples
    and determines effective permissions for any path.
    """

    def __init__(self) -> None:
        self._mounts: dict[str, MountConfig] = {}

    def add_mount(self, config: MountConfig) -> None:
        """Add or replace a mount point."""
        self._mounts[config.mount_path] = config

    def remove_mount(self, mount_path: str) -> None:
        """Remove a mount point."""
        mount_path = normalize_path(mount_path).rstrip("/")
        self._mounts.pop(mount_path, None)

    def resolve(self, virtual_path: str) -> tuple[MountConfig, str]:
        """Resolve a virtual path to its mount and relative path.

        Finds the longest matching mount prefix and strips it.
        """
        virtual_path = normalize_path(virtual_path)

        best_match: MountConfig | None = None
        best_len = 0

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

    def list_mounts(self) -> list[MountConfig]:
        """List all registered mounts, sorted by mount_path."""
        return sorted(self._mounts.values(), key=lambda m: m.mount_path)

    def list_visible_mounts(self) -> list[MountConfig]:
        """List non-hidden mounts, sorted by mount_path."""
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

    def has_mount(self, mount_path: str) -> bool:
        """Check if a mount exists at the given path."""
        mount_path = normalize_path(mount_path).rstrip("/")
        return mount_path in self._mounts

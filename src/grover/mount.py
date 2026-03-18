"""Mount — minimal routing dataclass and registry for filesystem composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.backends import DatabaseFileSystem
from grover.exceptions import MountNotFoundError
from grover.permissions import Permission
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from grover.backends.protocol import GroverFileSystem


class Mount:
    """A mount point binding a name to a filesystem.

    Each mount has:
    - ``name`` — simple identifier (no ``/``), e.g. ``"project"``
    - ``filesystem`` — the storage backend
    - ``session_factory`` — optional async session factory for DB-backed filesystems
    - ``permission`` — read-write or read-only

    Internally, ``path`` is derived as ``f"/{name}"`` for routing.

    Graph, search, and other providers live on the filesystem itself
    (via ``filesystem.graph_provider``, ``filesystem.search_provider``, etc.).
    """

    def __init__(
        self,
        name: str = "",
        filesystem: GroverFileSystem | None = None,
        permission: Permission = Permission.READ_WRITE,
        *,
        session_factory: Callable[..., AsyncSession] | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        name = name.strip("/")
        if "/" in name:
            raise ValueError(f"Mount name must not contain '/': {name!r}")
        self.name: str = name
        self.path: str = f"/{name}" if name else ""
        if filesystem is not None:
            self.filesystem: GroverFileSystem = filesystem
        else:
            self.filesystem = DatabaseFileSystem()
        self.session_factory: Callable[..., AsyncSession] | None = session_factory
        self.engine: AsyncEngine | None = engine
        self.permission: Permission = permission

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"filesystem={type(self.filesystem).__name__}"]
        return f"Mount({', '.join(parts)})"


class MountRegistry:
    """Registry of active mount points.

    Resolves virtual paths to ``(Mount, relative_path)`` tuples
    and determines effective permissions for any path.
    """

    def __init__(self) -> None:
        self.mounts: dict[str, Mount] = {}

    def add_mount(self, config: Mount) -> None:
        """Add or replace a mount point."""
        self.mounts[config.path] = config

    def remove_mount(self, mount_path: str) -> None:
        """Remove a mount point."""
        mount_path = normalize_path(mount_path).rstrip("/")
        self.mounts.pop(mount_path, None)

    def resolve(self, virtual_path: str) -> tuple[Mount, str]:
        """Resolve a virtual path to its mount and relative path.

        Finds the longest matching mount prefix and strips it.
        """
        virtual_path = normalize_path(virtual_path)

        best_match: Mount | None = None
        best_len = -1

        for mount_path, config in self.mounts.items():
            if (virtual_path == mount_path or virtual_path.startswith(mount_path + "/")) and len(mount_path) > best_len:
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
        return sorted(self.mounts.values(), key=lambda m: m.path)

    def get_permission(self, virtual_path: str) -> Permission:
        """Get the effective permission for a virtual path."""
        mount, _relative = self.resolve(virtual_path)
        return mount.permission

    def get_mount(self, mount_path: str) -> Mount | None:
        """Return the Mount at *mount_path*, or ``None``."""
        mount_path = normalize_path(mount_path).rstrip("/")
        return self.mounts.get(mount_path)

    def has_mount(self, mount_path: str) -> bool:
        """Check if a mount exists at the given path."""
        return self.get_mount(mount_path) is not None

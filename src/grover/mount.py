"""Mount — minimal routing dataclass and registry for filesystem composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    - ``filesystem`` — the storage backend (required)
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
        *,
        session_factory: Callable[..., AsyncSession] | None = None,
        engine: AsyncEngine | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        mount_type: str = "vfs",
        hidden: bool = False,
        read_only_paths: set[str] | None = None,
    ) -> None:
        name = name.strip("/")
        if "/" in name:
            raise ValueError(f"Mount name must not contain '/': {name!r}")
        self.name: str = name
        self.path: str = f"/{name}" if name else ""
        self.filesystem: GroverFileSystem | None = filesystem
        self.session_factory: Callable[..., AsyncSession] | None = session_factory
        self.engine: AsyncEngine | None = engine
        self.permission: Permission = permission
        self.label: str = label or self.name or "root"
        self.mount_type: str = mount_type
        self.hidden: bool = hidden
        self.read_only_paths: set[str] = read_only_paths if read_only_paths is not None else set()

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.filesystem is not None:
            parts.append(f"filesystem={type(self.filesystem).__name__}")
        return f"Mount({', '.join(parts)})"


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

"""GroverContext — shared state and helpers for GroverAsync mixins."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from grover.fs.exceptions import MountNotFoundError
from grover.fs.permissions import Permission
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.analyzers import AnalyzerRegistry
    from grover.fs.providers.protocols import GraphProvider
    from grover.mount import Mount
    from grover.mount.mounts import MountRegistry
    from grover.types import FileInfoResult
    from grover.worker import BackgroundWorker

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GroverContext:
    """Shared state for GroverAsync operations."""

    worker: BackgroundWorker
    registry: MountRegistry
    analyzer_registry: AnalyzerRegistry
    indexing_mode: IndexingMode = IndexingMode.BACKGROUND
    initialized: bool = False
    closed: bool = False

    # ------------------------------------------------------------------
    # Session management & helpers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session_for(self, mount: Mount) -> AsyncGenerator[AsyncSession | None]:
        """Yield a session for the given mount, or ``None`` for non-SQL backends."""
        if mount.session_factory is None:
            yield None
            return

        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def drain(self) -> None:
        """Drain all pending background work."""
        await self.worker.drain()

    def check_writable(self, virtual_path: str) -> str | None:
        """Return an error message if *virtual_path* is read-only, else ``None``.

        Replaces the previous raise-based pattern to avoid unnecessary
        exception overhead in the common (writable) case.
        """
        try:
            perm = self.registry.get_permission(virtual_path)
        except MountNotFoundError as e:
            return str(e)
        if perm == Permission.READ_ONLY:
            return f"Cannot write to read-only path: {virtual_path}"
        return None

    @staticmethod
    def get_capability(backend: object, protocol: type[T]) -> T | None:
        """Return *backend* if it satisfies *protocol*, else ``None``."""
        if isinstance(backend, protocol):
            return backend
        return None

    def prefix_path(self, path: str | None, mount_path: str) -> str | None:
        if path is None:
            return None
        if path == "/":
            return mount_path
        return mount_path + path

    def prefix_file_info(self, info: FileInfoResult, mount: Mount) -> FileInfoResult:
        prefixed_path = self.prefix_path(info.path, mount.path) or info.path
        info.path = prefixed_path
        info.mount_type = mount.mount_type
        info.permission = self.registry.get_permission(prefixed_path).value
        return info

    # ------------------------------------------------------------------
    # Per-mount graph resolution
    # ------------------------------------------------------------------

    def resolve_graph(self, path: str) -> GraphProvider:
        """Return the graph provider for the mount owning *path*."""
        try:
            mount, _rel = self.registry.resolve(path)
        except MountNotFoundError:
            msg = f"No mount found for path: {path!r}"
            raise RuntimeError(msg) from None
        gp = getattr(mount.filesystem, "graph_provider", None)
        if gp is None:
            msg = f"No graph on mount at {mount.path}"
            raise RuntimeError(msg)
        return gp

    def resolve_graph_any(self, path: str | None = None) -> GraphProvider:
        """Get graph for a specific path, or first available mount's graph."""
        if path is not None:
            return self.resolve_graph(path)
        for mount in self.registry.list_visible_mounts():
            gp = getattr(mount.filesystem, "graph_provider", None)
            if gp is not None:
                return gp
        msg = "No graph available on any mount"
        raise RuntimeError(msg)

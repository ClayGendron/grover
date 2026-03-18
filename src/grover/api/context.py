"""GroverContext — shared state and helpers for GroverAsync mixins."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from grover.exceptions import MountNotFoundError
from grover.models.internal.results import GroverResult
from grover.permissions import Permission
from grover.util.paths import normalize_path
from grover.worker import IndexingMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.analyzers import AnalyzerRegistry
    from grover.models.internal.ref import Directory, File, FileConnection
    from grover.mount import Mount, MountRegistry
    from grover.providers.graph.protocol import GraphProvider
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
    async def session_for(self, mount: Mount) -> AsyncGenerator[AsyncSession]:
        """Yield a session for the given mount."""
        if mount.session_factory is None:
            raise RuntimeError(f"No session factory on mount {mount.path}")

        session = mount.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def drain(self, *, timeout: float | None = None) -> None:
        """Drain all pending background work."""
        await self.worker.drain(timeout=timeout)

    def check_writable(self, virtual_path: str) -> GroverResult | None:
        """Return a failed ``GroverResult`` if *virtual_path* is read-only, else ``None``."""
        virtual_path = normalize_path(virtual_path)
        try:
            perm = self.registry.get_permission(virtual_path)
        except MountNotFoundError as e:
            return GroverResult(success=False, message=str(e))
        if perm == Permission.READ_ONLY:
            return GroverResult(success=False, message=f"Cannot write to read-only path: {virtual_path}")
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

    # ------------------------------------------------------------------
    # Per-mount graph resolution
    # ------------------------------------------------------------------

    def resolve_graph_with_mount(self, path: str) -> tuple[GraphProvider, Mount]:
        """Return ``(graph_provider, mount)`` for the mount owning *path*."""
        try:
            mount, _rel = self.registry.resolve(path)
        except MountNotFoundError:
            msg = f"No mount found for path: {path!r}"
            raise RuntimeError(msg) from None
        gp = getattr(mount.filesystem, "graph_provider", None)
        if gp is None:
            msg = f"No graph on mount at {mount.path}"
            raise RuntimeError(msg)
        return gp, mount

    def resolve_graph(self, path: str) -> GraphProvider:
        """Return the graph provider for the mount owning *path*."""
        gp, _mount = self.resolve_graph_with_mount(path)
        return gp

    def resolve_graph_any_with_mount(self, path: str | None = None) -> tuple[GraphProvider, Mount]:
        """Get ``(graph_provider, mount)`` for a path, or first available."""
        if path is not None:
            return self.resolve_graph_with_mount(path)
        for mount in self.registry.list_mounts():
            gp = getattr(mount.filesystem, "graph_provider", None)
            if gp is not None:
                return gp, mount
        msg = "No graph available on any mount"
        raise RuntimeError(msg)

    def resolve_graph_any(self, path: str | None = None) -> GraphProvider:
        """Get graph for a specific path, or first available mount's graph."""
        gp, _mount = self.resolve_graph_any_with_mount(path)
        return gp

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def group_by_mount(
        self,
        items: list[T],
        path_fn: Callable[[T], str],
    ) -> dict[str, list[T]]:
        """Group *items* by mount path.

        *path_fn* extracts the path used to resolve each item's mount.
        """
        from collections import defaultdict

        groups: dict[str, list[T]] = defaultdict(list)
        for item in items:
            mount, _rel = self.registry.resolve(path_fn(item))
            groups[mount.path].append(item)
        return dict(groups)

    def group_by_mount_writable(
        self,
        items: list[T],
        path_fn: Callable[[T], str],
    ) -> tuple[dict[str, list[T]], GroverResult | None]:
        """Group *items* by mount and verify all mounts are writable.

        *path_fn* extracts the path used to resolve each item's mount.

        Returns ``(groups, None)`` on success, or ``({}, GroverResult)``
        if any resolved mount is read-only.
        """
        groups = self.group_by_mount(items, path_fn)

        for mount_path in groups:
            if err := self.check_writable(mount_path):
                return {}, err

        return groups, None

    @asynccontextmanager
    async def mount_session(self, path: str) -> AsyncGenerator[tuple[Mount, str, AsyncSession]]:
        """Resolve path to mount and yield ``(mount, rel_path, session)``.

        Eliminates the three-line dispatch boilerplate
        (resolve → assert → session) common to single-path operations.
        """
        mount, rel_path = self.registry.resolve(path)
        assert mount.filesystem is not None
        async with self.session_for(mount) as session:
            yield mount, rel_path, session

    async def dispatch_to_mounts(
        self,
        groups: dict[str, list],
        handler: Callable[..., Awaitable[GroverResult]],
    ) -> GroverResult:
        """Dispatch grouped items to mounts in parallel, merge results.

        *groups* maps mount paths to lists of items.  *handler* is called
        as ``handler(mount, items, session)`` for each group.

        Results are concatenated (not union-merged by path).  ``success``
        is ``False`` if any mount result has ``success=False``.  ``message``
        is left empty — the caller sets it based on method-specific semantics.
        """

        async def _run(mount_path: str, items: list) -> GroverResult:
            mount = self.registry.mounts[mount_path]
            async with self.session_for(mount) as session:
                return await handler(mount, items, session)

        results = await asyncio.gather(*(_run(mp, items) for mp, items in groups.items()))

        all_files: list[File] = []
        all_dirs: list[Directory] = []
        all_conns: list[FileConnection] = []
        all_success = True
        for r in results:
            all_files.extend(r.files)
            all_dirs.extend(r.directories)
            all_conns.extend(r.connections)
            if not r.success:
                all_success = False

        return GroverResult(
            files=all_files,
            directories=all_dirs,
            connections=all_conns,
            success=all_success,
        )

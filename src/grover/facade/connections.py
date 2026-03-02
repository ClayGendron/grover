"""ConnectionMixin — connection operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.fs.exceptions import MountNotFoundError
from grover.fs.protocol import SupportsConnections
from grover.fs.utils import normalize_path
from grover.types import ConnectionResult

if TYPE_CHECKING:
    from grover.facade.context import GroverContext


class ConnectionMixin:
    """Connection operations extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Connection operations (persist through FS, graph updated via worker)
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
    ) -> ConnectionResult:
        """Add a connection between two files, persisted through the filesystem.

        The graph is updated via the worker after the DB transaction commits.
        """
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        if err := self._ctx.check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
            )

        async with self._ctx.session_for(mount) as sess:
            result = await backend.add_connection(
                source_path,
                target_path,
                connection_type,
                weight=weight,
                session=sess,
            )

        # Update graph AFTER session commits (post-commit ordering)
        if result.success:
            self._ctx.worker.schedule_immediate(
                self._process_connection_added(source_path, target_path, connection_type, weight)  # type: ignore[attr-defined]
            )

        return result

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
    ) -> ConnectionResult:
        """Delete a connection between two files.

        The graph is updated via the worker after the DB transaction commits.
        """
        source_path = normalize_path(source_path)
        target_path = normalize_path(target_path)

        if err := self._ctx.check_writable(source_path):
            return ConnectionResult(
                success=False,
                message=err,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        try:
            mount, _rel = self._ctx.registry.resolve(source_path)
        except MountNotFoundError:
            return ConnectionResult(
                success=False,
                message=f"No mount found for path: {source_path}",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return ConnectionResult(
                success=False,
                message="Backend does not support connections",
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type or "",
            )

        async with self._ctx.session_for(mount) as sess:
            result = await backend.delete_connection(
                source_path,
                target_path,
                connection_type=connection_type,
                session=sess,
            )

        # Update graph AFTER session commits (post-commit ordering)
        if result.success:
            self._ctx.worker.schedule_immediate(
                self._process_connection_deleted(source_path, target_path)  # type: ignore[attr-defined]
            )

        return result

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
    ) -> list[object]:
        """List connections for a path."""
        path = normalize_path(path)

        try:
            mount, _rel = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return []

        backend = self._ctx.get_capability(mount.filesystem, SupportsConnections)
        if backend is None:
            return []

        async with self._ctx.session_for(mount) as sess:
            return await backend.list_connections(
                path,
                direction=direction,
                connection_type=connection_type,
                session=sess,
            )

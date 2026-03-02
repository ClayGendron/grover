"""ConnectionService — stateless connection CRUD for DB-backed edge storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import select

from grover.types.operations import ConnectionResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.connections import FileConnectionBase


class ConnectionService:
    """Stateless helpers for file connection record CRUD.

    Receives the concrete connection model at construction so callers can
    use custom SQLModel subclasses.  Never creates, commits, or closes
    sessions — callers are responsible for session lifecycle.
    """

    def __init__(self, connection_model: type[FileConnectionBase]) -> None:
        self._model = connection_model

    async def add_connection(
        self,
        session: AsyncSession,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
    ) -> ConnectionResult:
        """Create or update a connection. Returns ConnectionResult."""
        model = self._model
        path = f"{source_path}[{connection_type}]{target_path}"

        # Check for existing connection by path
        result = await session.execute(select(model).where(model.path == path))
        existing = result.scalar_one_or_none()

        if existing is not None:
            # Update existing
            existing.weight = weight
            await session.flush()
            return ConnectionResult(
                path=path,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
                message="Connection updated",
            )

        # Create new
        record = model(
            path=path,
            source_path=source_path,
            target_path=target_path,
            type=connection_type,
            weight=weight,
        )
        session.add(record)
        await session.flush()
        return ConnectionResult(
            path=path,
            source_path=source_path,
            target_path=target_path,
            connection_type=connection_type,
            message="Connection created",
        )

    async def delete_connection(
        self,
        session: AsyncSession,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
    ) -> ConnectionResult:
        """Delete a connection. If connection_type is None, delete all between source and target."""
        model = self._model

        if connection_type is not None:
            path = f"{source_path}[{connection_type}]{target_path}"
            result = await session.execute(select(model).where(model.path == path))
            row = result.scalar_one_or_none()
            if row is None:
                return ConnectionResult(
                    success=False,
                    path=path,
                    source_path=source_path,
                    target_path=target_path,
                    connection_type=connection_type,
                    message=f"Connection not found: {path}",
                )
            await session.delete(row)
            await session.flush()
            return ConnectionResult(
                path=path,
                source_path=source_path,
                target_path=target_path,
                connection_type=connection_type,
                message="Connection deleted",
            )

        # Delete all connections between source and target
        result = await session.execute(
            select(model).where(
                model.source_path == source_path,
                model.target_path == target_path,
            )
        )
        rows = list(result.scalars().all())
        if not rows:
            return ConnectionResult(
                success=False,
                source_path=source_path,
                target_path=target_path,
                message=f"No connections found from {source_path} to {target_path}",
            )
        for row in rows:
            await session.delete(row)
        await session.flush()
        return ConnectionResult(
            source_path=source_path,
            target_path=target_path,
            message=f"Deleted {len(rows)} connection(s)",
        )

    async def delete_connections_for_path(
        self,
        session: AsyncSession,
        path: str,
    ) -> int:
        """Delete all connections where path is source or target. Returns count deleted."""
        model = self._model
        result = await session.execute(
            select(model).where((model.source_path == path) | (model.target_path == path))
        )
        rows = list(result.scalars().all())
        for row in rows:
            await session.delete(row)
        if rows:
            await session.flush()
        return len(rows)

    async def delete_outgoing_connections(
        self,
        session: AsyncSession,
        path: str,
    ) -> int:
        """Delete connections where path is the source only. Returns count deleted.

        Used by ``_analyze_and_integrate`` to clear stale outgoing edges before
        re-adding them.  Incoming edges (from other files) are preserved.
        """
        model = self._model
        result = await session.execute(select(model).where(model.source_path == path))
        rows = list(result.scalars().all())
        for row in rows:
            await session.delete(row)
        if rows:
            await session.flush()
        return len(rows)

    async def list_connections(
        self,
        session: AsyncSession,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
    ) -> list[FileConnectionBase]:
        """List connections for a path. direction: 'out', 'in', or 'both'."""
        model = self._model
        conditions = []

        if direction == "out":
            conditions.append(model.source_path == path)
        elif direction == "in":
            conditions.append(model.target_path == path)
        else:
            conditions.append((model.source_path == path) | (model.target_path == path))

        if connection_type is not None:
            conditions.append(model.type == connection_type)

        stmt = select(model).where(*conditions)
        result = await session.execute(stmt)
        return list(result.scalars().all())

"""ChunkService — stateless chunk CRUD for DB-backed chunk storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunks import FileChunkBase


class ChunkService:
    """Stateless helpers for file chunk record CRUD.

    Receives the concrete chunk model at construction so callers can
    use custom SQLModel subclasses.  Never creates, commits, or closes
    sessions — callers are responsible for session lifecycle.
    """

    def __init__(self, chunk_model: type[FileChunkBase]) -> None:
        self._chunk_model = chunk_model

    async def replace_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
        chunks: list[dict],
        *,
        user_id: str | None = None,
    ) -> int:
        """Delete all chunks for *file_path*, insert new ones. Returns count inserted."""
        await self.delete_file_chunks(session, file_path)

        model = self._chunk_model
        count = 0
        for chunk_data in chunks:
            record = model(
                file_path=file_path,
                chunk_path=chunk_data.get("chunk_path", ""),
                name=chunk_data.get("name", ""),
                description=chunk_data.get("description", ""),
                line_start=chunk_data.get("line_start", 0),
                line_end=chunk_data.get("line_end", 0),
                content=chunk_data.get("content", ""),
                content_hash=chunk_data.get("content_hash", ""),
                user_id=user_id,
            )
            session.add(record)
            count += 1

        await session.flush()
        return count

    async def delete_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> int:
        """Delete all chunks for *file_path*. Returns count deleted."""
        model = self._chunk_model
        result = await session.execute(select(model).where(model.file_path == file_path))
        rows = list(result.scalars().all())
        count = len(rows)
        for row in rows:
            await session.delete(row)
        if count > 0:
            await session.flush()
        return count

    async def list_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> list[FileChunkBase]:
        """List all chunks for *file_path*, ordered by line_start."""
        model = self._chunk_model
        result = await session.execute(
            select(model).where(model.file_path == file_path).order_by(model.line_start)  # type: ignore[arg-type]
        )
        return list(result.scalars().all())

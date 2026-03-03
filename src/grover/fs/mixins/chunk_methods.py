"""ChunkMethodsMixin — chunk delegates for DatabaseFileSystem."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.types.operations import ChunkListResult, ChunkResult


class ChunkMethodsMixin:
    """Delegates chunk operations to ``self.chunk_provider``."""

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        return await self.chunk_provider.replace_file_chunks(sess, file_path, chunks)  # type: ignore[attr-defined]

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        return await self.chunk_provider.delete_file_chunks(sess, file_path)  # type: ignore[attr-defined]

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> ChunkListResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        return await self.chunk_provider.list_file_chunks(sess, file_path)  # type: ignore[attr-defined]

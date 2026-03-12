"""DefaultChunkProvider — stateless chunk CRUD for DB-backed chunk storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from grover.results.operations import BatchChunkResult, ChunkListResult, ChunkResult
from grover.util.content import compute_content_hash

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunk import FileChunkBase


class DefaultChunkProvider:
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
    ) -> ChunkResult:
        """Delete all chunks for *file_path*, insert new ones. Returns count inserted."""
        await self.delete_file_chunks(session, file_path)

        model = self._chunk_model
        count = 0
        for chunk_data in chunks:
            record = model(
                file_path=file_path,
                path=chunk_data.get("path", ""),
                line_start=chunk_data.get("line_start", 0),
                line_end=chunk_data.get("line_end", 0),
                content=chunk_data.get("content", ""),
                content_hash=chunk_data.get("content_hash", ""),
            )
            session.add(record)
            count += 1

        await session.flush()
        return ChunkResult(count=count, path=file_path)

    async def delete_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> ChunkResult:
        """Delete all chunks for *file_path*. Returns count deleted."""
        model = self._chunk_model
        result = await session.execute(select(model).where(model.file_path == file_path))
        rows = list(result.scalars().all())
        count = len(rows)
        for row in rows:
            await session.delete(row)
        if count > 0:
            await session.flush()
        return ChunkResult(count=count, path=file_path)

    async def list_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> ChunkListResult:
        """List all chunks for *file_path*, ordered by line_start."""
        model = self._chunk_model
        result = await session.execute(
            select(model).where(model.file_path == file_path).order_by(model.line_start)  # type: ignore[arg-type]
        )
        chunks = list(result.scalars().all())
        return ChunkListResult(chunks=chunks, path=file_path)

    async def write_chunk(
        self,
        session: AsyncSession,
        chunk: FileChunkBase,
    ) -> ChunkResult:
        """Upsert a single chunk. Delegates to write_chunks."""
        result = await self.write_chunks(session, [chunk])
        if result.results:
            return result.results[0]
        return ChunkResult(count=0, path=chunk.path, success=False, message="Write failed")

    async def write_chunks(
        self,
        session: AsyncSession,
        chunks: list[FileChunkBase],
    ) -> BatchChunkResult:
        """Batch upsert chunks. System manages content_hash and timestamps."""
        model = self._chunk_model
        now = datetime.now(UTC)

        # Batch lookup: one query for all chunk paths
        chunk_paths = [c.path for c in chunks]
        existing_result = await session.execute(
            select(model).where(model.path.in_(chunk_paths))  # type: ignore[arg-type]
        )
        existing_map: dict[str, FileChunkBase] = {
            row.path: row for row in existing_result.scalars().all()
        }

        results: list[ChunkResult] = []
        for chunk in chunks:
            content_hash, _ = compute_content_hash(chunk.content)

            existing = existing_map.get(chunk.path)
            if existing is not None:
                # Update existing record
                existing.content = chunk.content
                existing.content_hash = content_hash
                existing.line_start = chunk.line_start
                existing.line_end = chunk.line_end
                existing.updated_at = now
                results.append(ChunkResult(count=1, path=chunk.path))
            else:
                # Insert new record using the configured model class
                record = model(
                    file_path=chunk.file_path,
                    path=chunk.path,
                    content=chunk.content,
                    content_hash=content_hash,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                results.append(ChunkResult(count=1, path=chunk.path))

        await session.flush()

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        return BatchChunkResult(
            success=failed == 0,
            message=f"Wrote {succeeded} chunk(s)" + (f", {failed} failed" if failed else ""),
            results=results,
            succeeded=succeeded,
            failed=failed,
        )

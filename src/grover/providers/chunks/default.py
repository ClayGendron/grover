"""DefaultChunkProvider — stateless chunk CRUD for DB-backed chunk storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from grover.models.internal.detail import WriteDetail
from grover.models.internal.evidence import Evidence
from grover.models.internal.ref import File, FileChunk
from grover.models.internal.results import GroverResult
from grover.util.content import compute_content_hash

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.chunk import FileChunkModelBase


class DefaultChunkProvider:
    """Stateless helpers for file chunk record CRUD.

    Receives the concrete chunk model at construction so callers can
    use custom SQLModel subclasses.  Never creates, commits, or closes
    sessions — callers are responsible for session lifecycle.
    """

    def __init__(self, chunk_model: type[FileChunkModelBase]) -> None:
        self._chunk_model = chunk_model

    async def replace_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
        chunks: list[dict],
    ) -> GroverResult:
        """Delete all chunks for *file_path*, insert new ones. Returns count inserted."""
        await self.delete_file_chunks(session, file_path)

        model = self._chunk_model
        internal_chunks: list[FileChunk] = []
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
            internal_chunks.append(
                FileChunk(
                    path=record.path,
                    name=record.path.split("#")[-1] if "#" in record.path else record.path,
                    content=record.content,
                    line_start=record.line_start,
                    line_end=record.line_end,
                )
            )

        await session.flush()
        return GroverResult(
            success=True,
            message=f"{len(internal_chunks)} chunks replaced",
            files=[File(path=file_path, chunks=internal_chunks)],
        )

    async def delete_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> GroverResult:
        """Delete all chunks for *file_path*. Returns count deleted."""
        model = self._chunk_model
        result = await session.execute(select(model).where(model.file_path == file_path))
        rows = list(result.scalars().all())
        deleted_chunks = [
            FileChunk(
                path=row.path,
                name=row.path.split("#")[-1] if "#" in row.path else row.path,
                content=row.content,
                line_start=row.line_start,
                line_end=row.line_end,
            )
            for row in rows
        ]
        for row in rows:
            await session.delete(row)
        if deleted_chunks:
            await session.flush()
        return GroverResult(
            success=True,
            message=f"{len(deleted_chunks)} chunks deleted",
            files=[File(path=file_path, chunks=deleted_chunks)],
        )

    async def list_file_chunks(
        self,
        session: AsyncSession,
        file_path: str,
    ) -> GroverResult:
        """List all chunks for *file_path*, ordered by line_start."""
        model = self._chunk_model
        result = await session.execute(
            select(model).where(model.file_path == file_path).order_by(model.line_start)  # type: ignore[arg-type]
        )
        db_chunks = list(result.scalars().all())
        internal_chunks = [
            FileChunk(
                path=c.path,
                name=c.path.split("#")[-1] if "#" in c.path else c.path,
                content=c.content,
                tokens=0,
                line_start=c.line_start,
                line_end=c.line_end,
            )
            for c in db_chunks
        ]
        return GroverResult(
            success=True,
            message=f"{len(db_chunks)} chunks found",
            files=[File(path=file_path, chunks=internal_chunks)],
        )

    async def write_chunks(
        self,
        session: AsyncSession,
        chunks: list[FileChunkModelBase],
    ) -> GroverResult:
        """Batch upsert chunks. System manages content_hash and timestamps."""
        model = self._chunk_model
        now = datetime.now(UTC)

        # Batch lookup: one query for all chunk paths
        chunk_paths = [c.path for c in chunks]
        existing_result = await session.execute(
            select(model).where(model.path.in_(chunk_paths))  # type: ignore[arg-type]
        )
        existing_map: dict[str, FileChunkModelBase] = {row.path: row for row in existing_result.scalars().all()}

        # Group results by parent file_path
        file_map: dict[str, tuple[list[FileChunk], list[Evidence]]] = {}
        total = 0
        for chunk in chunks:
            content_hash, _ = compute_content_hash(chunk.content)

            chunk_ref = FileChunk(
                path=chunk.path,
                name=chunk.path.split("#")[-1] if "#" in chunk.path else chunk.path,
                content=chunk.content,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )

            existing = existing_map.get(chunk.path)
            if existing is not None:
                existing.content = chunk.content
                existing.content_hash = content_hash
                existing.line_start = chunk.line_start
                existing.line_end = chunk.line_end
                existing.updated_at = now
                detail = WriteDetail(operation="write_chunk", success=True, message=f"Updated chunk: {chunk.path}")
            else:
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
                detail = WriteDetail(operation="write_chunk", success=True, message=f"Created chunk: {chunk.path}")

            if chunk.file_path not in file_map:
                file_map[chunk.file_path] = ([], [])
            file_map[chunk.file_path][0].append(chunk_ref)
            file_map[chunk.file_path][1].append(detail)
            total += 1

        await session.flush()

        result_files = [
            File(path=fp, chunks=chunk_refs, evidence=details) for fp, (chunk_refs, details) in file_map.items()
        ]
        return GroverResult(
            success=True,
            message=f"{total} chunk(s) written",
            files=result_files,
        )

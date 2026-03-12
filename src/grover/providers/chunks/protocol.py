"""Chunk provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.models.chunk import FileChunkBase
    from grover.results.operations import BatchChunkResult, ChunkListResult, ChunkResult


@runtime_checkable
class ChunkProvider(Protocol):
    """Chunk storage — file chunk CRUD."""

    async def replace_file_chunks(
        self, session: Any, file_path: str, chunks: list[dict]
    ) -> ChunkResult: ...

    async def delete_file_chunks(self, session: Any, file_path: str) -> ChunkResult: ...

    async def list_file_chunks(self, session: Any, file_path: str) -> ChunkListResult: ...

    async def write_chunk(
        self,
        session: Any,
        chunk: FileChunkBase,
    ) -> ChunkResult: ...

    async def write_chunks(
        self,
        session: Any,
        chunks: list[FileChunkBase],
    ) -> BatchChunkResult: ...

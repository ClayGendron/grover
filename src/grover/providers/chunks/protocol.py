"""Chunk provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.internal.results import GroverResult


@runtime_checkable
class ChunkProvider(Protocol):
    """Chunk storage — file chunk CRUD."""

    async def replace_file_chunks(self, session: Any, file_path: str, chunks: list[dict]) -> GroverResult: ...

    async def delete_file_chunks(self, session: Any, file_path: str) -> GroverResult: ...

    async def list_file_chunks(self, session: Any, file_path: str) -> GroverResult: ...

    async def write_chunks(
        self,
        session: Any,
        chunks: list[FileChunkModelBase],
    ) -> GroverResult: ...

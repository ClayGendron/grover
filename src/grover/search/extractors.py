"""Text extraction — bridge between analyzers and the search index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grover.graph.analyzers._base import ChunkFile


@dataclass(frozen=True, slots=True)
class EmbeddableChunk:
    """A chunk of text ready for embedding.

    Attributes:
        path: Chunk path (or file path for whole-file embedding).
        content: Text to embed.
        parent_path: Parent file path if this is a sub-file chunk.
    """

    path: str
    content: str
    parent_path: str | None = None


def extract_from_chunks(chunks: list[ChunkFile]) -> list[EmbeddableChunk]:
    """Convert analyzer ``ChunkFile`` output to embeddable entries.

    Filters out chunks with empty content.
    """
    return [
        EmbeddableChunk(
            path=chunk.chunk_path,
            content=chunk.content,
            parent_path=chunk.parent_path,
        )
        for chunk in chunks
        if chunk.content.strip()
    ]


def extract_from_file(path: str, content: str) -> list[EmbeddableChunk]:
    """Create a single embeddable entry for a whole file.

    Returns an empty list if the content is empty or whitespace-only.
    """
    if not content or not content.strip():
        return []
    return [EmbeddableChunk(path=path, content=content)]

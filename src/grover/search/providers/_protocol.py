"""EmbeddingProvider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations convert text into fixed-dimension float vectors
    suitable for similarity search.
    """

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...

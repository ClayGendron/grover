"""SentenceTransformerEmbedding — local embedding provider (all-MiniLM-L6-v2)."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformerEmbedding:
    """Embedding provider backed by ``sentence-transformers``.

    The model is loaded lazily on the first call to :meth:`embed` or
    :meth:`embed_batch`.  Async methods run the underlying CPU-bound model
    inference in a thread pool via :func:`asyncio.to_thread`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            msg = (
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install grover[search]"
            )
            raise ImportError(msg)
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    def embed_sync(self, text: str) -> list[float]:
        """Embed a single text string (synchronous)."""
        model = self._load_model()
        result: Any = model.encode([text])
        return result[0].tolist()

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (synchronous)."""
        model = self._load_model()
        result: Any = model.encode(texts)
        return [row.tolist() for row in result]

    # ------------------------------------------------------------------
    # Async methods (EmbeddingProvider protocol)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string in a thread pool."""
        return await asyncio.to_thread(self.embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a thread pool."""
        return await asyncio.to_thread(self.embed_batch_sync, texts)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        model = self._load_model()
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            msg = f"Model {self._model_name!r} did not report embedding dimensions"
            raise RuntimeError(msg)
        return dim

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name


# Backward-compatible alias
SentenceTransformerProvider = SentenceTransformerEmbedding

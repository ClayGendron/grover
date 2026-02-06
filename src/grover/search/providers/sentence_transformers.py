"""SentenceTransformerProvider — default embedding provider (all-MiniLM-L6-v2)."""

from __future__ import annotations

from typing import Any

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformerProvider:
    """Embedding provider backed by ``sentence-transformers``.

    The model is loaded lazily on the first call to :meth:`embed` or
    :meth:`embed_batch`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            msg = (
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install it with: pip install grover[search]"
            )
            raise ImportError(msg)
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        model = self._load_model()
        result: Any = model.encode([text])
        return result[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        model = self._load_model()
        result: Any = model.encode(texts)
        return [row.tolist() for row in result]

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

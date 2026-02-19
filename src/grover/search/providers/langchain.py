"""LangChainEmbedding — adapter wrapping any LangChain Embeddings instance."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _HAS_LANGCHAIN = False


class LangChainEmbedding:
    """Adapter that wraps any ``langchain_core.embeddings.Embeddings`` instance.

    Delegates to ``aembed_query`` / ``aembed_documents`` for async I/O.
    LangChain's default implementations use a thread pool, but many
    providers (e.g. ``OpenAIEmbeddings``) override with native async.

    *dimensions* is discovered lazily by embedding a sample text if not
    provided explicitly at construction.

    Requires ``langchain-core``::

        pip install grover[langchain]
    """

    def __init__(
        self,
        embeddings: Any,
        *,
        dimensions: int | None = None,
        model_name: str | None = None,
    ) -> None:
        if not _HAS_LANGCHAIN:
            msg = (
                "langchain-core is required for LangChainEmbedding. "
                "Install it with: pip install grover[langchain]"
            )
            raise ImportError(msg)

        if not isinstance(embeddings, _LCEmbeddings):
            msg = (
                "embeddings must be a langchain_core.embeddings.Embeddings instance, "
                f"got {type(embeddings).__name__}"
            )
            raise TypeError(msg)

        self._embeddings = embeddings
        self._dimensions = dimensions
        self._model_name = model_name or _discover_model_name(embeddings)
        self._probed_dimensions: int | None = None

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Embed a single query text via the wrapped LangChain provider."""
        return await self._embeddings.aembed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts via the wrapped LangChain provider."""
        if not texts:
            return []
        return await self._embeddings.aembed_documents(texts)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality.

        If not provided at construction, probes the underlying provider
        by embedding a sample text on first access.
        """
        if self._dimensions is not None:
            return self._dimensions
        if self._probed_dimensions is not None:
            return self._probed_dimensions
        # Synchronous probe — use embed_query to discover dimensions
        vec = self._embeddings.embed_query("dimension probe")
        self._probed_dimensions = len(vec)
        return self._probed_dimensions

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name


def _discover_model_name(embeddings: Any) -> str:
    """Try to discover the model name from a LangChain Embeddings instance."""
    # Most LangChain providers use 'model' or 'model_name'
    for attr in ("model", "model_name"):
        name = getattr(embeddings, attr, None)
        if isinstance(name, str) and name:
            return name
    return type(embeddings).__name__

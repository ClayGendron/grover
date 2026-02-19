"""OpenAIEmbedding — async embedding provider backed by OpenAI's API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

try:
    from openai import AsyncOpenAI

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False

if TYPE_CHECKING:
    from openai import AsyncOpenAI as AsyncOpenAIType

# Default dimensions per model when the user does not specify.
_MODEL_DEFAULTS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding:
    """Async embedding provider backed by the OpenAI Embeddings API.

    Uses ``AsyncOpenAI`` for native async I/O.  Large batches are
    automatically chunked at *batch_size* texts per API call to stay
    within the 300K token limit.

    Requires the ``openai`` package::

        pip install grover[openai]
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        batch_size: int = 512,
    ) -> None:
        if not _HAS_OPENAI:
            msg = (
                "openai is required for OpenAIEmbedding. "
                "Install it with: pip install grover[openai]"
            )
            raise ImportError(msg)

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            msg = (
                "No OpenAI API key provided. Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )
            raise ValueError(msg)

        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client: AsyncOpenAIType = AsyncOpenAI(
            api_key=resolved_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via the OpenAI API."""
        result = await self._call_api([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, chunking at *batch_size* per API call."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            vectors = await self._call_api(chunk)
            all_vectors.extend(vectors)
        return all_vectors

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        if self._dimensions is not None:
            return self._dimensions
        default = _MODEL_DEFAULTS.get(self._model)
        if default is not None:
            return default
        msg = f"Unknown default dimensions for model {self._model!r}. Pass dimensions= explicitly."
        raise ValueError(msg)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings endpoint and return ordered vectors."""
        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda e: e.index)
        return [item.embedding for item in sorted_data]

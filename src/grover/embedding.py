"""Embedding providers — protocol and LangChain adapter.

``EmbeddingProvider`` is the async-first protocol for text-to-vector embedding.
All methods return ``Vector`` instances with dimension and model-name tracking,
so embeddings carry provenance from creation through database storage.

``LangChainEmbeddingProvider`` wraps any ``langchain_core.embeddings.Embeddings``
instance, adapting its ``list[float]`` results into properly-typed ``Vector``
instances.  Requires ``langchain-core``::

    pip install grover[langchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from grover.vector import Vector

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _LCEmbeddings = None  # ty: ignore[invalid-assignment]
    _HAS_LANGCHAIN = False


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async-first protocol for text-to-vector embedding.

    Implementations convert text into fixed-dimension ``Vector`` instances
    suitable for similarity search.  The returned vectors carry dimension
    and model-name metadata.
    """

    async def embed(self, text: str) -> Vector:
        """Embed a single text string into a vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
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


class LangChainEmbeddingProvider:
    """Adapter wrapping any ``langchain_core.embeddings.Embeddings`` instance.

    Delegates to ``aembed_query`` / ``aembed_documents`` for async I/O and
    wraps results in ``Vector[dim, model]`` instances.

    Dimensions are discovered lazily by embedding a sample text on first
    use if not provided explicitly at construction.

    Requires ``langchain-core``::

        pip install grover[langchain]
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        dimensions: int | None = None,
        model_name: str | None = None,
    ) -> None:
        if not _HAS_LANGCHAIN:
            msg = (
                "langchain-core is required for LangChainEmbeddingProvider."
                " Install it with: pip install grover[langchain]"
            )
            raise ImportError(msg)

        if not isinstance(embeddings, _LCEmbeddings):
            msg = f"embeddings must be a langchain_core.embeddings.Embeddings instance, got {type(embeddings).__name__}"
            raise TypeError(msg)

        self._embeddings = embeddings
        self._dimensions = dimensions
        self._model_name = model_name or _discover_model_name(embeddings)
        self._vector_cls: type[Vector] | None = None

        if dimensions is not None:
            self._vector_cls = Vector[dimensions, self._model_name]

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> Vector:
        """Embed a single query text via the wrapped LangChain provider."""
        cls = await self._ensure_vector_cls()
        raw = await self._embeddings.aembed_query(text)
        return cls(raw)

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Embed multiple document texts via the wrapped LangChain provider."""
        if not texts:
            return []
        cls = await self._ensure_vector_cls()
        raw_vectors = await self._embeddings.aembed_documents(texts)
        return [cls(raw) for raw in raw_vectors]

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality.

        Raises ``RuntimeError`` if dimensions were not provided at
        construction and no embedding call has been made yet to probe them.
        """
        if self._dimensions is not None:
            return self._dimensions
        msg = "Dimensions not yet known — call embed() first or pass dimensions= at construction"
        raise RuntimeError(msg)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_vector_cls(self) -> type[Vector]:
        """Build and cache the ``Vector[dim, model]`` subclass.

        If dimensions were not provided at construction, probes the
        underlying LangChain provider by embedding a sample text.
        """
        if self._vector_cls is not None:
            return self._vector_cls
        raw = await self._embeddings.aembed_query("dimension probe")
        self._dimensions = len(raw)
        self._vector_cls = Vector[self._dimensions, self._model_name]
        return self._vector_cls


def _discover_model_name(embeddings: object) -> str:
    """Try to discover the model name from a LangChain Embeddings instance."""
    for attr in ("model", "model_name"):
        name = getattr(embeddings, attr, None)
        if isinstance(name, str) and name:
            return name
    return type(embeddings).__name__

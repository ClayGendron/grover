"""Tests for EmbeddingProvider protocol and LangChainEmbeddingProvider."""

from __future__ import annotations

from typing import Any, cast

import pytest

from grover.embedding import EmbeddingProvider, LangChainEmbeddingProvider
from grover.vector import Vector

# ---------------------------------------------------------------------------
# Mock LangChain Embeddings — avoids requiring langchain-core in test deps
# ---------------------------------------------------------------------------

MOCK_DIM = 4
MOCK_MODEL = "mock-embed-v1"


class MockEmbeddings:
    """Minimal stand-in for ``langchain_core.embeddings.Embeddings``.

    Registered as a virtual subclass of the real ABC so isinstance() passes
    in the LangChainEmbeddingProvider constructor.
    """

    model = MOCK_MODEL

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1 * (i + 1) for i in range(MOCK_DIM)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1) for i in range(MOCK_DIM)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1 * (i + 1) for i in range(MOCK_DIM)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1) for i in range(MOCK_DIM)] for _ in texts]


# Register MockEmbeddings as a virtual subclass of the real LangChain ABC
# so that isinstance() checks pass without inheriting from it.
try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings

    _LCEmbeddings.register(MockEmbeddings)
except ImportError:
    pytest.skip("langchain-core not installed", allow_module_level=True)


def _provider(
    embeddings: MockEmbeddings,
    *,
    dimensions: int | None = None,
    model_name: str | None = None,
) -> LangChainEmbeddingProvider:
    return LangChainEmbeddingProvider(
        cast("Any", embeddings),
        dimensions=dimensions,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    assert isinstance(provider, EmbeddingProvider)


# ---------------------------------------------------------------------------
# embed() — single text
# ---------------------------------------------------------------------------


async def test_embed_returns_vector():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    result = await provider.embed("hello world")

    assert isinstance(result, Vector)
    assert len(result) == MOCK_DIM
    assert result.dimension == MOCK_DIM
    assert result.model_name == MOCK_MODEL


async def test_embed_vector_values():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    result = await provider.embed("test")

    expected = [0.1 * (i + 1) for i in range(MOCK_DIM)]
    assert list(result) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# embed_batch() — multiple texts
# ---------------------------------------------------------------------------


async def test_embed_batch_returns_vectors():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    results = await provider.embed_batch(["one", "two", "three"])

    assert len(results) == 3
    for vec in results:
        assert isinstance(vec, Vector)
        assert len(vec) == MOCK_DIM
        assert vec.dimension == MOCK_DIM
        assert vec.model_name == MOCK_MODEL


async def test_embed_batch_empty():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    results = await provider.embed_batch([])

    assert results == []


# ---------------------------------------------------------------------------
# Lazy dimension probe
# ---------------------------------------------------------------------------


async def test_lazy_dimension_probe():
    """Without explicit dimensions, embed() probes and discovers them."""
    mock = MockEmbeddings()
    provider = _provider(mock)

    # Before any embed call, dimensions are unknown
    with pytest.raises(RuntimeError, match="Dimensions not yet known"):
        _ = provider.dimensions

    # First embed call triggers probe
    result = await provider.embed("probe me")

    assert provider.dimensions == MOCK_DIM
    assert isinstance(result, Vector)
    assert result.dimension == MOCK_DIM


async def test_explicit_dimensions_skip_probe():
    """With explicit dimensions, no probe is needed."""
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)

    assert provider.dimensions == MOCK_DIM


# ---------------------------------------------------------------------------
# Model name discovery
# ---------------------------------------------------------------------------


def test_model_name_from_model_attr():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    assert provider.model_name == MOCK_MODEL


def test_model_name_from_model_name_attr():

    class EmbeddingsWithModelName(MockEmbeddings):
        model = None
        model_name = "custom-model"

    mock = EmbeddingsWithModelName()
    provider = _provider(mock, dimensions=MOCK_DIM)
    assert provider.model_name == "custom-model"


def test_model_name_fallback_to_class_name():

    class CustomEmbeddings(MockEmbeddings):
        model = None

    mock = CustomEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM)
    assert provider.model_name == "CustomEmbeddings"


def test_explicit_model_name():
    mock = MockEmbeddings()
    provider = _provider(mock, dimensions=MOCK_DIM, model_name="override")
    assert provider.model_name == "override"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_rejects_non_embeddings_instance():
    with pytest.raises(TypeError, match="must be a langchain_core"):
        LangChainEmbeddingProvider("not an embeddings")  # type: ignore[arg-type]


def test_import_error_when_langchain_missing(monkeypatch: pytest.MonkeyPatch):
    """LangChainEmbeddingProvider raises ImportError when langchain-core is absent."""
    import grover.embedding as mod

    monkeypatch.setattr(mod, "_HAS_LANGCHAIN", False)
    with pytest.raises(ImportError, match="langchain-core is required"):
        LangChainEmbeddingProvider(MockEmbeddings())  # type: ignore[arg-type]

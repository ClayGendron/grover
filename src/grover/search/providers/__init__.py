"""Embedding providers — protocol and implementations."""

from grover.search.protocols import EmbeddingProvider
from grover.search.providers.sentence_transformers import (
    SentenceTransformerEmbedding,
    SentenceTransformerProvider,
)

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbedding",
    "SentenceTransformerProvider",
]

# Optional providers — import-guarded, available only when deps are installed.
try:
    from grover.search.providers.openai import OpenAIEmbedding

    __all__.append("OpenAIEmbedding")
except ImportError:  # pragma: no cover
    pass

try:
    from grover.search.providers.langchain import LangChainEmbedding

    __all__.append("LangChainEmbedding")
except ImportError:  # pragma: no cover
    pass

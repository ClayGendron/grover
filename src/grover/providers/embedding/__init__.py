"""Embedding providers — protocol and implementations."""

from .protocol import EmbeddingProvider

__all__ = [
    "EmbeddingProvider",
]

# Optional providers — import-guarded, available only when deps are installed.
try:
    from grover.providers.embedding.openai import OpenAIEmbedding

    __all__.append("OpenAIEmbedding")
except ImportError:  # pragma: no cover
    pass

try:
    from grover.providers.embedding.langchain import LangChainEmbedding

    __all__.append("LangChainEmbedding")
except ImportError:  # pragma: no cover
    pass

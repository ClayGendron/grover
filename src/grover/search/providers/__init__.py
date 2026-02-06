"""Embedding providers — protocol and implementations."""

from grover.search.providers._protocol import EmbeddingProvider
from grover.search.providers.sentence_transformers import SentenceTransformerProvider

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
]

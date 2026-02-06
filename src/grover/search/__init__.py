"""Vector search layer — HNSW index, text extraction, embedding providers."""

from grover.search._index import SearchIndex, SearchResult
from grover.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.search.providers import EmbeddingProvider, SentenceTransformerProvider

__all__ = [
    "EmbeddableChunk",
    "EmbeddingProvider",
    "SearchIndex",
    "SearchResult",
    "SentenceTransformerProvider",
    "extract_from_chunks",
    "extract_from_file",
]

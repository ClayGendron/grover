"""Vector search layer — engine, stores, text extraction, embedding providers."""

from grover.search._engine import SearchEngine
from grover.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.search.protocols import EmbeddingProvider, VectorStore
from grover.search.stores.local import LocalVectorStore
from grover.search.types import SearchResult

__all__ = [
    "EmbeddableChunk",
    "EmbeddingProvider",
    "LocalVectorStore",
    "SearchEngine",
    "SearchResult",
    "VectorStore",
    "extract_from_chunks",
    "extract_from_file",
]

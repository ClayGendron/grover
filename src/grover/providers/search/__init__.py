"""Vector search layer — stores, text extraction, embedding providers."""

from grover.providers.embedding.protocol import EmbeddingProvider
from grover.providers.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.providers.search.local import LocalVectorStore
from grover.providers.search.types import SearchResult

__all__ = [
    "EmbeddableChunk",
    "EmbeddingProvider",
    "LocalVectorStore",
    "SearchResult",
    "extract_from_chunks",
    "extract_from_file",
]

# Optional stores — import-guarded, available only when deps are installed.
try:
    from grover.providers.search.pinecone import PineconeVectorStore

    __all__.append("PineconeVectorStore")
except ImportError:  # pragma: no cover
    pass

try:
    from grover.providers.search.databricks import DatabricksVectorStore

    __all__.append("DatabricksVectorStore")
except ImportError:  # pragma: no cover
    pass

"""Vector search layer — stores, text extraction, embedding providers."""

from grover.fs.providers.protocols import EmbeddingProvider
from grover.fs.providers.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.fs.providers.search.local import LocalVectorStore
from grover.fs.providers.search.protocols import VectorStore
from grover.fs.providers.search.types import SearchResult

__all__ = [
    "EmbeddableChunk",
    "EmbeddingProvider",
    "LocalVectorStore",
    "SearchResult",
    "VectorStore",
    "extract_from_chunks",
    "extract_from_file",
]

# Optional stores — import-guarded, available only when deps are installed.
try:
    from grover.fs.providers.search.pinecone import PineconeVectorStore

    __all__.append("PineconeVectorStore")
except ImportError:  # pragma: no cover
    pass

try:
    from grover.fs.providers.search.databricks import DatabricksVectorStore

    __all__.append("DatabricksVectorStore")
except ImportError:  # pragma: no cover
    pass

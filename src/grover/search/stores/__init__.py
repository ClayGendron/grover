"""Vector stores — VectorStore protocol implementations."""

from grover.search.stores.local import LocalVectorStore

__all__ = [
    "LocalVectorStore",
]

# Optional stores — import-guarded, available only when deps are installed.
try:
    from grover.search.stores.pinecone import PineconeVectorStore

    __all__.append("PineconeVectorStore")
except ImportError:  # pragma: no cover
    pass

try:
    from grover.search.stores.databricks import DatabricksVectorStore

    __all__.append("DatabricksVectorStore")
except ImportError:  # pragma: no cover
    pass

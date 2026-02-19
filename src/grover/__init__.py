"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.2"

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.types import ListSharesResult, ShareInfo, ShareResult
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.ref import Ref, file_ref
from grover.search._engine import SearchEngine
from grover.search.filters import (
    FilterExpression,
    and_,
    eq,
    exists,
    gt,
    gte,
    in_,
    lt,
    lte,
    ne,
    not_in,
    or_,
)
from grover.search.protocols import (
    EmbeddingProvider,
    SupportsHybridSearch,
    SupportsIndexLifecycle,
    SupportsMetadataFilter,
    SupportsNamespaces,
    SupportsReranking,
    SupportsTextIngest,
    SupportsTextSearch,
    VectorStore,
)
from grover.search.types import (
    DeleteResult as SearchDeleteResult,
)
from grover.search.types import (
    IndexConfig,
    IndexInfo,
    SearchResult,
    UpsertResult,
    VectorEntry,
    VectorSearchResult,
)

__all__ = [
    "EmbeddingProvider",
    "FilterExpression",
    "Grover",
    "GroverAsync",
    "IndexConfig",
    "IndexInfo",
    "ListSharesResult",
    "Ref",
    "SearchDeleteResult",
    "SearchEngine",
    "SearchResult",
    "ShareInfo",
    "ShareResult",
    "SupportsHybridSearch",
    "SupportsIndexLifecycle",
    "SupportsMetadataFilter",
    "SupportsNamespaces",
    "SupportsReranking",
    "SupportsTextIngest",
    "SupportsTextSearch",
    "UpsertResult",
    "UserScopedFileSystem",
    "VectorEntry",
    "VectorSearchResult",
    "VectorStore",
    "__version__",
    "and_",
    "eq",
    "exists",
    "file_ref",
    "gt",
    "gte",
    "in_",
    "lt",
    "lte",
    "ne",
    "not_in",
    "or_",
]

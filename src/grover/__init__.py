"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.3"

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.query_types import (
    ChunkMatch,
    GlobHit,
    GlobQueryResult,
    GrepHit,
    GrepQueryResult,
    LineMatch,
    SearchHit,
    SearchQueryResult,
)
from grover.fs.types import ListSharesResult, ShareInfo, ShareResult
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.graph.protocols import GraphStore
from grover.graph.types import SubgraphResult
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
    "ChunkMatch",
    "EmbeddingProvider",
    "FilterExpression",
    "GlobHit",
    "GlobQueryResult",
    "GraphStore",
    "GrepHit",
    "GrepQueryResult",
    "Grover",
    "GroverAsync",
    "IndexConfig",
    "IndexInfo",
    "LineMatch",
    "ListSharesResult",
    "Ref",
    "SearchDeleteResult",
    "SearchEngine",
    "SearchHit",
    "SearchQueryResult",
    "SearchResult",
    "ShareInfo",
    "ShareResult",
    "SubgraphResult",
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

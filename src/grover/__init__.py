"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.3"

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.types import ListSharesResult, ShareInfo, ShareResult
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.graph.protocols import GraphStore
from grover.graph.types import SubgraphResult
from grover.mount import Mount, ProtocolConflictError, ProtocolNotAvailableError
from grover.ref import Ref, file_ref
from grover.results import Evidence, FileOperationResult, FileSearchResult
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
from grover.search.results import (
    GlobEvidence,
    GlobResult,
    GraphEvidence,
    GraphResult,
    GrepEvidence,
    GrepResult,
    HybridSearchResult,
    LexicalSearchResult,
    LineMatch,
    ListDirResult,
    TrashResult,
    TreeResult,
    VectorEvidence,
)
from grover.search.results import (
    VectorSearchResult as VectorSearchResult,
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
)

__all__ = [
    "EmbeddingProvider",
    "Evidence",
    "FileOperationResult",
    "FileSearchResult",
    "FilterExpression",
    "GlobEvidence",
    "GlobResult",
    "GraphEvidence",
    "GraphResult",
    "GraphStore",
    "GrepEvidence",
    "GrepResult",
    "Grover",
    "GroverAsync",
    "HybridSearchResult",
    "IndexConfig",
    "IndexInfo",
    "LexicalSearchResult",
    "LineMatch",
    "ListDirResult",
    "ListSharesResult",
    "Mount",
    "ProtocolConflictError",
    "ProtocolNotAvailableError",
    "Ref",
    "SearchDeleteResult",
    "SearchEngine",
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
    "TrashResult",
    "TreeResult",
    "UpsertResult",
    "UserScopedFileSystem",
    "VectorEntry",
    "VectorEvidence",
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

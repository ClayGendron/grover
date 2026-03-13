"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.4"

from grover.backends.user_scoped import UserScopedFileSystem
from grover.client import Grover, GroverAsync
from grover.models.internal.evidence import (
    Evidence,
    GlobEvidence,
    GraphEvidence,
    GrepEvidence,
    HybridEvidence,
    LexicalEvidence,
    LineMatch,
    ListDirEvidence,
    ShareEvidence,
    TrashEvidence,
    TreeEvidence,
    VectorEvidence,
    VersionEvidence,
)
from grover.models.internal.ref import File, FileChunk, FileConnection, FileVersion
from grover.models.internal.results import (
    FileOperationResult,
    FileSearchResult,
    FileSearchSet,
)
from grover.mount import Mount
from grover.providers import (
    ChunkProvider,
    DefaultChunkProvider,
    DefaultVersionProvider,
    DiskStorageProvider,
    EmbeddingProvider,
    GraphProvider,
    SearchProvider,
    StorageProvider,
    VersionProvider,
)
from grover.providers.graph.types import SubgraphResult
from grover.providers.search.filters import (
    FilterExpression,
    FilterValue,
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
from grover.providers.search.protocol import (
    SupportsHybridSearch,
    SupportsIndexLifecycle,
    SupportsMetadataFilter,
    SupportsNamespaces,
    SupportsReranking,
)
from grover.providers.search.types import (
    DeleteResult as SearchDeleteResult,
)
from grover.providers.search.types import (
    IndexConfig,
    IndexInfo,
    SearchResult,
    UpsertResult,
    VectorEntry,
)
from grover.ref import Ref
from grover.worker import IndexingMode

__all__ = [
    "ChunkProvider",
    "DefaultChunkProvider",
    "DefaultVersionProvider",
    "DiskStorageProvider",
    "EmbeddingProvider",
    "Evidence",
    "File",
    "FileChunk",
    "FileConnection",
    "FileOperationResult",
    "FileSearchResult",
    "FileSearchSet",
    "FileVersion",
    "FilterExpression",
    "FilterValue",
    "GlobEvidence",
    "GraphEvidence",
    "GraphProvider",
    "GrepEvidence",
    "Grover",
    "GroverAsync",
    "HybridEvidence",
    "IndexConfig",
    "IndexInfo",
    "IndexingMode",
    "LexicalEvidence",
    "LineMatch",
    "ListDirEvidence",
    "Mount",
    "Ref",
    "SearchDeleteResult",
    "SearchProvider",
    "SearchResult",
    "ShareEvidence",
    "StorageProvider",
    "SubgraphResult",
    "SupportsHybridSearch",
    "SupportsIndexLifecycle",
    "SupportsMetadataFilter",
    "SupportsNamespaces",
    "SupportsReranking",
    "TrashEvidence",
    "TreeEvidence",
    "UpsertResult",
    "UserScopedFileSystem",
    "VectorEntry",
    "VectorEvidence",
    "VersionEvidence",
    "VersionProvider",
    "__version__",
    "and_",
    "eq",
    "exists",
    "gt",
    "gte",
    "in_",
    "lt",
    "lte",
    "ne",
    "not_in",
    "or_",
]

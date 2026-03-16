"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.7"

from grover.backends.user_scoped import UserScopedFileSystem
from grover.client import Grover, GroverAsync
from grover.models.config import EngineConfig, SessionConfig, create_async_engine_factory
from grover.models.internal.evidence import (
    Evidence,
    GlobEvidence,
    GraphCentralityEvidence,
    GraphRelationshipEvidence,
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
    BatchResult,
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
from grover.providers.search.protocol import IndexConfig
from grover.ref import Ref
from grover.worker import IndexingMode

__all__ = [
    "BatchResult",
    "ChunkProvider",
    "DefaultChunkProvider",
    "DefaultVersionProvider",
    "DiskStorageProvider",
    "EmbeddingProvider",
    "EngineConfig",
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
    "GraphCentralityEvidence",
    "GraphProvider",
    "GraphRelationshipEvidence",
    "GrepEvidence",
    "Grover",
    "GroverAsync",
    "HybridEvidence",
    "IndexConfig",
    "IndexingMode",
    "LexicalEvidence",
    "LineMatch",
    "ListDirEvidence",
    "Mount",
    "Ref",
    "SearchProvider",
    "SessionConfig",
    "ShareEvidence",
    "StorageProvider",
    "SubgraphResult",
    "TrashEvidence",
    "TreeEvidence",
    "UserScopedFileSystem",
    "VectorEvidence",
    "VersionEvidence",
    "VersionProvider",
    "__version__",
    "and_",
    "create_async_engine_factory",
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

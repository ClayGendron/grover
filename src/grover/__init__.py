"""Grover: The agentic filesystem.

Safe file operations, knowledge graphs, and semantic search — unified for AI agents.
"""

__version__ = "0.0.2"

from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.types import ListSharesResult, ShareInfo, ShareResult
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.ref import Ref, file_ref
from grover.search._index import SearchResult

__all__ = [
    "Grover",
    "GroverAsync",
    "ListSharesResult",
    "Ref",
    "SearchResult",
    "ShareInfo",
    "ShareResult",
    "UserScopedFileSystem",
    "__version__",
    "file_ref",
]

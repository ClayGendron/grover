"""Dispatch protocols for mount-level method routing.

These protocols determine which mount component (filesystem, graph, search)
handles each search/query method.  Mount checks all components at construction
time and builds a dispatch map.

* If 2+ components implement the same protocol → ``ProtocolConflictError``
* If 0 components implement it → ``ProtocolNotAvailableError`` when called
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Filesystem dispatch protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsGlob(Protocol):
    """Component can perform glob pattern matching."""

    async def glob(self, pattern: str, path: str = "/", **kwargs: object) -> object: ...


@runtime_checkable
class SupportsGrep(Protocol):
    """Component can perform content search via regex."""

    async def grep(self, pattern: str, path: str = "/", **kwargs: object) -> object: ...


@runtime_checkable
class SupportsTree(Protocol):
    """Component can produce a directory tree listing."""

    async def tree(self, path: str = "/", **kwargs: object) -> object: ...


@runtime_checkable
class SupportsListDir(Protocol):
    """Component can list directory contents."""

    async def list_dir(self, path: str = "/", **kwargs: object) -> object: ...


# ---------------------------------------------------------------------------
# Search dispatch protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsVectorSearch(Protocol):
    """Component can perform vector/semantic search."""

    async def vector_search(self, query: str, **kwargs: object) -> object: ...


@runtime_checkable
class SupportsLexicalSearch(Protocol):
    """Component can perform lexical/keyword (BM25) search."""

    async def lexical_search(self, query: str, **kwargs: object) -> object: ...


@runtime_checkable
class SupportsHybridSearch(Protocol):
    """Component can perform hybrid (vector + lexical) search."""

    async def hybrid_search(self, query: str, **kwargs: object) -> object: ...


@runtime_checkable
class SupportsEmbedding(Protocol):
    """Component can produce text embeddings."""

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Protocol registry
# ---------------------------------------------------------------------------

DISPATCH_PROTOCOLS: list[type] = [
    SupportsGlob,
    SupportsGrep,
    SupportsTree,
    SupportsListDir,
    SupportsVectorSearch,
    SupportsLexicalSearch,
    SupportsHybridSearch,
    SupportsEmbedding,
]

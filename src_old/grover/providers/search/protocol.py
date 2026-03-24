"""Search layer protocols — vector storage and capability interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.models.internal.ref import File
    from grover.models.internal.results import BatchResult, FileSearchResult, FileSearchSet


# ---------------------------------------------------------------------------
# IndexConfig — inline in protocol (types.py deleted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Configuration for creating a vector index."""

    name: str
    dimension: int
    metric: str = "cosine"
    cloud_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def parent_path_from_id(entry_id: str) -> str:
    """Extract the parent file path from a chunk ID.

    ``/a.py#login`` → ``/a.py``.  If no ``#`` is present, returns *entry_id*
    unchanged.
    """
    return entry_id.split("#")[0]


# ---------------------------------------------------------------------------
# Core search protocol (filesystem-level)
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchProvider(Protocol):
    """Search interface — vector storage + index management.

    Existing stores (``LocalVectorStore``, ``PineconeVectorStore``,
    ``DatabricksVectorStore``) implement this directly.
    """

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def create_index(self, config: IndexConfig) -> None: ...

    async def upsert(self, *, files: list[File]) -> BatchResult: ...

    async def delete(self, *, files: list[str]) -> BatchResult: ...

    async def vector_search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        candidates: FileSearchSet | None = None,
    ) -> FileSearchResult: ...

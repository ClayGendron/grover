"""FullTextStore protocol — interface for BM25/keyword search backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.search.fulltext.types import FullTextResult


@runtime_checkable
class FullTextStore(Protocol):
    """Protocol for full-text search stores.

    Implementations use native DB features:
    - SQLite FTS5 (``MATCH``, ``bm25()``, ``snippet()``)
    - PostgreSQL full-text (``to_tsvector``, ``ts_rank_cd``, GIN index)
    - MSSQL fulltext (``FREETEXTTABLE``)
    """

    async def index(
        self,
        path: str,
        content: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Index *content* for the given *path*."""
        ...

    async def remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single path from the index."""
        ...

    async def remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove *path* and all entries whose path starts with *path#*."""
        ...

    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[FullTextResult]:
        """Search the index and return ranked results."""
        ...

"""SQLite FTS5 full-text search store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from grover.search.fulltext.types import FullTextResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class SQLiteFullTextStore:
    """Full-text search backed by SQLite FTS5.

    Uses a virtual table ``grover_fts`` with columns ``path`` and ``content``.
    Ranking uses FTS5's built-in ``bm25()`` function; snippets use ``snippet()``.

    Parameters
    ----------
    engine:
        AsyncEngine used to create the FTS virtual table on first call to
        ``ensure_table()``.  If None, the table must already exist.
    table_name:
        Name of the FTS virtual table.  Defaults to ``"grover_fts"``.
    """

    def __init__(
        self,
        engine: AsyncEngine | None = None,
        *,
        table_name: str = "grover_fts",
    ) -> None:
        self._engine = engine
        self._table = table_name
        self._table_created = False

    async def ensure_table(self, engine: AsyncEngine | None = None) -> None:
        """Create the FTS5 virtual table if it doesn't exist."""
        eng = engine or self._engine
        if eng is None:
            return
        self._engine = eng
        if self._table_created:
            return
        async with eng.begin() as conn:
            await conn.execute(
                text(f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} USING fts5(path, content)")
            )
        self._table_created = True

    # ------------------------------------------------------------------
    # FullTextStore protocol
    # ------------------------------------------------------------------

    async def index(
        self,
        path: str,
        content: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Index *content* at *path*.  Replaces existing entry if present."""
        if session is None:
            return
        # Delete existing entry first (FTS5 doesn't support ON CONFLICT)
        await session.execute(
            text(f"DELETE FROM {self._table} WHERE path = :path"),
            {"path": path},
        )
        await session.execute(
            text(f"INSERT INTO {self._table} (path, content) VALUES (:path, :content)"),
            {"path": path, "content": content},
        )
        await session.flush()

    async def remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single path from the FTS index."""
        if session is None:
            return
        await session.execute(
            text(f"DELETE FROM {self._table} WHERE path = :path"),
            {"path": path},
        )
        await session.flush()

    async def remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove *path* and all chunk entries (``path#chunk``)."""
        if session is None:
            return
        escaped = path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        await session.execute(
            text(f"DELETE FROM {self._table} WHERE path = :path OR path LIKE :prefix ESCAPE '\\'"),
            {"path": path, "prefix": f"{escaped}#%"},
        )
        await session.flush()

    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[FullTextResult]:
        """Search the FTS index using BM25 ranking."""
        if session is None:
            return []
        if not query.strip():
            return []
        # Escape double-quotes in the query to prevent FTS5 syntax errors
        safe_query = query.replace('"', '""')
        result = await session.execute(
            text(
                f"SELECT path, "
                f"snippet({self._table}, 1, '>>>', '<<<', '...', 32) AS snippet, "
                f"bm25({self._table}) AS rank "
                f"FROM {self._table} WHERE {self._table} MATCH :query "
                f"ORDER BY rank "
                f"LIMIT :k"
            ),
            {"query": f'"{safe_query}"', "k": k},
        )
        rows = result.fetchall()
        return [
            FullTextResult(
                path=row.path,
                snippet=row.snippet,
                rank=float(row.rank),
            )
            for row in rows
        ]

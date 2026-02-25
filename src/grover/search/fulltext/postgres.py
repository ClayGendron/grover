"""PostgreSQL full-text search store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from grover.search.fulltext.types import FullTextResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class PostgresFullTextStore:
    """Full-text search backed by PostgreSQL ``tsvector`` / ``tsquery``.

    Uses a table ``grover_fts`` with columns ``path``, ``content``, and
    ``tsv`` (generated ``tsvector``).  Ranking uses ``ts_rank_cd``; snippets
    use ``ts_headline``.

    Parameters
    ----------
    engine:
        AsyncEngine used to create the table and GIN index.
    table_name:
        Name of the FTS table.  Defaults to ``"grover_fts"``.
    config:
        PostgreSQL text search configuration.  Defaults to ``"english"``.
    """

    def __init__(
        self,
        engine: AsyncEngine | None = None,
        *,
        table_name: str = "grover_fts",
        config: str = "english",
    ) -> None:
        self._engine = engine
        self._table = table_name
        self._config = config
        self._table_created = False

    async def ensure_table(self, engine: AsyncEngine | None = None) -> None:
        """Create the FTS table and GIN index if they don't exist."""
        eng = engine or self._engine
        if eng is None:
            return
        self._engine = eng
        if self._table_created:
            return
        async with eng.begin() as conn:
            await conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ("
                    f"  path TEXT PRIMARY KEY,"
                    f"  content TEXT NOT NULL DEFAULT '',"
                    f"  tsv tsvector GENERATED ALWAYS AS ("
                    f"    to_tsvector('{self._config}', content)"
                    f"  ) STORED"
                    f")"
                )
            )
            await conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_tsv "
                    f"ON {self._table} USING gin(tsv)"
                )
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
        """Index *content* at *path* using upsert."""
        if session is None:
            return
        await session.execute(
            text(
                f"INSERT INTO {self._table} (path, content) "
                f"VALUES (:path, :content) "
                f"ON CONFLICT (path) DO UPDATE SET content = EXCLUDED.content"
            ),
            {"path": path, "content": content},
        )
        await session.flush()

    async def remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single path from the index."""
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
        """Remove *path* and all chunk entries."""
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
        """Search using ``ts_rank_cd`` for ranking."""
        if session is None:
            return []
        if not query.strip():
            return []
        result = await session.execute(
            text(
                f"SELECT path, "
                f"ts_headline('{self._config}', content, "
                f"  plainto_tsquery('{self._config}', :query), "
                f"  'MaxFragments=2,MaxWords=32') AS snippet, "
                f"ts_rank_cd(tsv, plainto_tsquery('{self._config}', :query)) AS rank "
                f"FROM {self._table} "
                f"WHERE tsv @@ plainto_tsquery('{self._config}', :query) "
                f"ORDER BY rank DESC "
                f"LIMIT :k"
            ),
            {"query": query, "k": k},
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

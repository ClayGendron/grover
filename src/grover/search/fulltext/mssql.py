"""MSSQL full-text search store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from grover.search.fulltext.types import FullTextResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class MSSQLFullTextStore:
    """Full-text search backed by MSSQL ``FREETEXTTABLE``.

    Uses a table ``grover_fts`` with columns ``id`` (identity PK),
    ``path``, and ``content``.  A full-text catalog and index are created
    on the ``content`` column.

    Parameters
    ----------
    engine:
        AsyncEngine used to create the table and full-text index.
    table_name:
        Name of the FTS table.  Defaults to ``"grover_fts"``.
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
        """Create the FTS table and full-text index if they don't exist."""
        eng = engine or self._engine
        if eng is None:
            return
        self._engine = eng
        if self._table_created:
            return
        async with eng.begin() as conn:
            # Create base table with identity PK (required for MSSQL fulltext)
            await conn.execute(
                text(
                    f"IF OBJECT_ID('{self._table}', 'U') IS NULL "
                    f"CREATE TABLE {self._table} ("
                    f"  id INT IDENTITY(1,1) PRIMARY KEY,"
                    f"  path NVARCHAR(900) NOT NULL UNIQUE,"
                    f"  content NVARCHAR(MAX) NOT NULL DEFAULT ''"
                    f")"
                )
            )
            # Create full-text catalog and index
            await conn.execute(
                text(
                    "IF NOT EXISTS ("
                    "  SELECT 1 FROM sys.fulltext_catalogs "
                    "  WHERE name = 'grover_fts_catalog'"
                    ") CREATE FULLTEXT CATALOG grover_fts_catalog"
                )
            )
            await conn.execute(
                text(
                    f"IF NOT EXISTS ("
                    f"  SELECT 1 FROM sys.fulltext_indexes "
                    f"  WHERE object_id = OBJECT_ID('{self._table}')"
                    f") CREATE FULLTEXT INDEX ON {self._table}(content) "
                    f"KEY INDEX PK_{self._table}_id "
                    f"ON grover_fts_catalog"
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
        """Index *content* at *path* using MERGE."""
        if session is None:
            return
        await session.execute(
            text(
                f"MERGE INTO {self._table} WITH (HOLDLOCK) AS target "
                f"USING (SELECT :path AS path) AS source "
                f"ON target.path = source.path "
                f"WHEN MATCHED THEN UPDATE SET content = :content "
                f"WHEN NOT MATCHED THEN INSERT (path, content) "
                f"VALUES (:path, :content);"
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
        """Search using ``FREETEXTTABLE`` for ranking."""
        if session is None:
            return []
        if not query.strip():
            return []
        result = await session.execute(
            text(
                f"SELECT TOP (:k) t.path, t.content AS snippet, ft.[RANK] AS rank "
                f"FROM {self._table} t "
                f"INNER JOIN FREETEXTTABLE({self._table}, content, :query) ft "
                f"ON t.id = ft.[KEY] "
                f"ORDER BY ft.[RANK] DESC"
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

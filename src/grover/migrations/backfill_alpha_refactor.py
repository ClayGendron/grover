"""Alpha refactor migration — backfill new columns from existing data.

Handles the schema changes introduced by the alpha refactor:

1. ``grover_file_versions``: Add ``file_path`` column, backfill from
   joined ``grover_files.path``.
2. ``grover_file_chunks``: Rename ``chunk_path`` → ``path``.
3. ``grover_file_chunks``: Add ``vector`` column (nullable JSON text).
4. ``grover_files``: Add ``vector`` column (nullable JSON text).
5. ``grover_file_connections``: Add ``path`` column, compute from
   ``source_path || '[' || type || ']' || target_path``.
6. Drop ``grover_embeddings`` table (vectors now stored on models).

Idempotent — safe to run multiple times. Skips steps that are already done.

Usage::

    from sqlalchemy.ext.asyncio import create_async_engine
    from grover.migrations import backfill_alpha_refactor

    engine = create_async_engine("sqlite+aiosqlite:///grover.db")
    report = await backfill_alpha_refactor(engine)
    print(report)  # {"file_versions_file_path": "added", ...}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import inspect, text

if TYPE_CHECKING:
    from sqlalchemy import Inspector
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


def _get_column_names(inspector: Inspector, table: str) -> set[str]:
    """Return the set of column names for *table*, or empty set if table missing."""
    try:
        return {c["name"] for c in inspector.get_columns(table)}
    except Exception:
        return set()


def _table_exists(inspector: Inspector, table: str) -> bool:
    """Check whether *table* exists in the database."""
    return table in inspector.get_table_names()


def _get_dialect(conn: Connection) -> str:
    """Return normalized dialect name."""
    name = conn.engine.dialect.name
    if name in ("postgresql", "postgres"):
        return "postgresql"
    if name in ("mssql", "pyodbc"):
        return "mssql"
    return name


def _add_column_sql(dialect: str, table: str, column: str, col_type: str) -> str:
    """Return ADD COLUMN SQL for the given dialect."""
    if dialect == "mssql":
        return f"ALTER TABLE {table} ADD {column} {col_type}"
    # SQLite and PostgreSQL
    return f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"


def _rename_column_sql(dialect: str, table: str, old_name: str, new_name: str) -> str:
    """Return RENAME COLUMN SQL for the given dialect."""
    if dialect == "mssql":
        return f"EXEC sp_rename '{table}.{old_name}', '{new_name}', 'COLUMN'"
    # SQLite 3.25+ and PostgreSQL
    return f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}"


def _run_sync(conn: Connection) -> dict[str, str]:
    """Run all migration steps synchronously (called via run_sync)."""
    report: dict[str, str] = {}
    inspector = inspect(conn)
    dialect = _get_dialect(conn)

    # ------------------------------------------------------------------
    # 1. grover_file_versions: add file_path column
    # ------------------------------------------------------------------
    if _table_exists(inspector, "grover_file_versions"):
        cols = _get_column_names(inspector, "grover_file_versions")
        if "file_path" not in cols:
            sql = _add_column_sql(dialect, "grover_file_versions", "file_path", "TEXT DEFAULT ''")
            conn.execute(text(sql))
            # Backfill from joined grover_files.path
            if _table_exists(inspector, "grover_files"):
                conn.execute(
                    text(
                        "UPDATE grover_file_versions SET file_path = COALESCE(("
                        "  SELECT path FROM grover_files"
                        "  WHERE grover_files.id = grover_file_versions.file_id"
                        "), '')"
                    )
                )
            report["file_versions_file_path"] = "added"
            logger.info("Added file_path column to grover_file_versions and backfilled")
        else:
            report["file_versions_file_path"] = "exists"
    else:
        report["file_versions_file_path"] = "table_missing"

    # ------------------------------------------------------------------
    # 2. grover_file_chunks: rename chunk_path → path
    # ------------------------------------------------------------------
    if _table_exists(inspector, "grover_file_chunks"):
        cols = _get_column_names(inspector, "grover_file_chunks")
        if "chunk_path" in cols and "path" not in cols:
            conn.execute(
                text(_rename_column_sql(dialect, "grover_file_chunks", "chunk_path", "path"))
            )
            report["file_chunks_path"] = "renamed"
            logger.info("Renamed chunk_path → path on grover_file_chunks")
        elif "path" not in cols:
            conn.execute(
                text(_add_column_sql(dialect, "grover_file_chunks", "path", "TEXT DEFAULT ''"))
            )
            report["file_chunks_path"] = "added"
            logger.info("Added path column to grover_file_chunks")
        else:
            report["file_chunks_path"] = "exists"

        # 3. grover_file_chunks: add vector column
        cols = _get_column_names(inspector, "grover_file_chunks")
        if "vector" not in cols:
            conn.execute(text(_add_column_sql(dialect, "grover_file_chunks", "vector", "TEXT")))
            report["file_chunks_vector"] = "added"
            logger.info("Added vector column to grover_file_chunks")
        else:
            report["file_chunks_vector"] = "exists"
    else:
        report["file_chunks_path"] = "table_missing"
        report["file_chunks_vector"] = "table_missing"

    # ------------------------------------------------------------------
    # 4. grover_files: add vector column
    # ------------------------------------------------------------------
    if _table_exists(inspector, "grover_files"):
        cols = _get_column_names(inspector, "grover_files")
        if "vector" not in cols:
            conn.execute(text(_add_column_sql(dialect, "grover_files", "vector", "TEXT")))
            report["files_vector"] = "added"
            logger.info("Added vector column to grover_files")
        else:
            report["files_vector"] = "exists"
    else:
        report["files_vector"] = "table_missing"

    # ------------------------------------------------------------------
    # 5. grover_file_connections: add path column
    # ------------------------------------------------------------------
    if _table_exists(inspector, "grover_file_connections"):
        cols = _get_column_names(inspector, "grover_file_connections")
        if "path" not in cols:
            conn.execute(
                text(_add_column_sql(dialect, "grover_file_connections", "path", "TEXT DEFAULT ''"))
            )
            # Backfill: source_path[type]target_path
            if dialect == "mssql":
                concat_sql = (
                    "UPDATE grover_file_connections"
                    " SET path = source_path + '[' + COALESCE(type, '') + ']' + target_path"
                )
            else:
                concat_sql = (
                    "UPDATE grover_file_connections"
                    " SET path = source_path || '[' || COALESCE(type, '') || ']' || target_path"
                )
            conn.execute(text(concat_sql))
            report["file_connections_path"] = "added"
            logger.info("Added path column to grover_file_connections and backfilled")
        else:
            report["file_connections_path"] = "exists"
    else:
        report["file_connections_path"] = "table_missing"

    # ------------------------------------------------------------------
    # 6. Drop grover_embeddings
    # ------------------------------------------------------------------
    if _table_exists(inspector, "grover_embeddings"):
        conn.execute(text("DROP TABLE grover_embeddings"))
        report["embeddings_dropped"] = "dropped"
        logger.info("Dropped grover_embeddings table")
    else:
        report["embeddings_dropped"] = "not_present"

    return report


async def backfill_alpha_refactor(engine: AsyncEngine) -> dict[str, str]:
    """Run the alpha refactor migration. Idempotent.

    Parameters
    ----------
    engine:
        An async SQLAlchemy engine connected to the target database.

    Returns
    -------
    dict[str, str]
        A report mapping each migration step to its outcome
        (``"added"``, ``"renamed"``, ``"exists"``, ``"table_missing"``,
        ``"dropped"``, ``"not_present"``).
    """
    async with engine.begin() as conn:
        report: dict[str, str] = await conn.run_sync(_run_sync)
    return report


async def check_schema_compatibility(
    engine: AsyncEngine,
    *,
    file_chunks_table: str = "grover_file_chunks",
    file_connections_table: str = "grover_file_connections",
    file_versions_table: str = "grover_file_versions",
) -> list[str]:
    """Check whether the database schema is compatible with the current code.

    Returns a list of error messages. An empty list means the schema is
    compatible. Only checks tables that already exist — new/empty databases
    are always compatible (``create_all`` will create the correct schema).
    """

    def _check(conn: Connection) -> list[str]:
        inspector = inspect(conn)
        errors: list[str] = []

        # Check grover_file_chunks has 'path' (not old 'chunk_path')
        if _table_exists(inspector, file_chunks_table):
            cols = _get_column_names(inspector, file_chunks_table)
            if "path" not in cols:
                hint = (
                    " (found 'chunk_path' — run migration to rename)"
                    if "chunk_path" in cols
                    else ""
                )
                errors.append(
                    f"Table '{file_chunks_table}' is missing required column 'path'{hint}"
                )

        # Check grover_file_connections has 'path'
        if _table_exists(inspector, file_connections_table):
            cols = _get_column_names(inspector, file_connections_table)
            if "path" not in cols:
                errors.append(f"Table '{file_connections_table}' is missing required column 'path'")

        # Check grover_file_versions has 'file_path'
        if _table_exists(inspector, file_versions_table):
            cols = _get_column_names(inspector, file_versions_table)
            if "file_path" not in cols:
                errors.append(
                    f"Table '{file_versions_table}' is missing required column 'file_path'"
                )

        return errors

    async with engine.connect() as conn:
        return await conn.run_sync(_check)

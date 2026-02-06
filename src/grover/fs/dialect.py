"""Dialect-aware SQL helpers — upsert, merge, date functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


def get_dialect(engine: Engine | AsyncEngine) -> str:
    """Return 'sqlite', 'postgresql', or 'mssql'."""
    # AsyncEngine wraps a sync engine
    sync_engine = getattr(engine, "sync_engine", engine)
    name = sync_engine.dialect.name
    if name == "sqlite":
        return "sqlite"
    if name in ("postgresql", "postgres"):
        return "postgresql"
    if name in ("mssql", "pyodbc"):
        return "mssql"
    return name


async def upsert_file(
    session: AsyncSession,
    dialect: str,
    values: dict[str, Any],
    conflict_keys: list[str],
) -> int:
    """Dialect-aware upsert into grover_files. Returns rowcount.

    - SQLite/PostgreSQL: INSERT ... ON CONFLICT DO UPDATE
    - MSSQL: MERGE INTO ... WITH (HOLDLOCK)
    """
    if dialect == "mssql":
        return await _upsert_mssql(session, values, conflict_keys)
    return await _upsert_sqlite_pg(session, dialect, values, conflict_keys)


async def _upsert_sqlite_pg(
    session: AsyncSession,
    dialect: str,
    values: dict[str, Any],
    conflict_keys: list[str],
) -> int:
    """SQLite / PostgreSQL upsert using INSERT ... ON CONFLICT DO UPDATE."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    from grover.models.files import GroverFile

    dialect_module = sqlite_dialect
    if dialect == "postgresql":
        from sqlalchemy.dialects import postgresql as pg_dialect

        dialect_module = pg_dialect

    stmt = dialect_module.insert(GroverFile).values(**values)

    # Columns to update on conflict (exclude conflict keys)
    update_cols = {k: v for k, v in values.items() if k not in conflict_keys}

    if update_cols:
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_keys,
            set_=update_cols,
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_keys)

    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


async def _upsert_mssql(
    session: AsyncSession,
    values: dict[str, Any],
    conflict_keys: list[str],
) -> int:
    """MSSQL upsert using MERGE INTO ... WITH (HOLDLOCK)."""
    on_clause = " AND ".join(f"target.{k} = :{k}" for k in conflict_keys)
    insert_cols = ", ".join(values.keys())
    insert_vals = ", ".join(f":{k}" for k in values)
    update_set = ", ".join(
        f"target.{k} = :{k}" for k in values if k not in conflict_keys
    )

    merge_sql = f"""
        MERGE INTO grover_files WITH (HOLDLOCK) AS target
        USING (SELECT {', '.join(f':{k} AS {k}' for k in conflict_keys)}) AS source
        ON {on_clause}
        WHEN NOT MATCHED THEN
            INSERT ({insert_cols})
            VALUES ({insert_vals})
    """
    if update_set:
        merge_sql += f"""
        WHEN MATCHED THEN
            UPDATE SET {update_set}
        """
    merge_sql += ";"

    result = await session.execute(text(merge_sql), values)
    return result.rowcount  # type: ignore[return-value]


def now_expression(dialect: str) -> Any:
    """Return a dialect-appropriate 'now' expression for SQL.

    - SQLite: func.datetime('now')
    - PostgreSQL: func.now()
    - MSSQL: func.sysdatetimeoffset()
    """
    from sqlalchemy import func

    if dialect == "sqlite":
        return func.datetime("now")
    if dialect == "mssql":
        return func.sysdatetimeoffset()
    return func.now()

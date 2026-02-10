"""Store repo files across two database backends (SQLite + PostgreSQL).

Demonstrates the mount-first Grover API with multiple DatabaseFileSystem
backends mounted at different paths on a single Grover instance.

- /code   → SQLite   (source code)
- /docs   → PostgreSQL (docs, config, markdown)

Usage:
    uv run python scripts/store_repo.py
"""

from __future__ import annotations

import asyncio
import mimetypes
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover import Grover
from grover.fs.database_fs import DatabaseFileSystem

REPO_ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = REPO_ROOT / "grover_repo.db"
PG_DB = "grover_test"
PG_URL = f"postgresql+asyncpg://localhost/{PG_DB}"

# File classification
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".h", ".java"}
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}

# Directories and patterns to skip
SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".idea", ".claude", "docs",
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
    ".ttf", ".eot", ".DS_Store", ".ipynb",
}
MAX_FILE_SIZE = 512 * 1024  # 512 KB


def should_skip(path: Path) -> bool:
    """Return True if the file should be skipped."""
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    if path.name.startswith("."):
        return True
    if path.stat().st_size > MAX_FILE_SIZE:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    if mime and not mime.startswith("text") and mime != "application/json":
        return True
    return False


def classify_file(path: Path) -> str | None:
    """Return 'code', 'docs', or None (skip)."""
    ext = path.suffix.lower()
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in DOC_EXTENSIONS:
        return "docs"
    # Default: put anything with a recognisable text extension in docs
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("text"):
        return "docs"
    return None


def collect_files(root: Path) -> dict[str, list[Path]]:
    """Walk the repo and classify files into code vs docs."""
    buckets: dict[str, list[Path]] = {"code": [], "docs": []}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            if should_skip(p):
                continue
        except OSError:
            continue
        kind = classify_file(p)
        if kind:
            buckets[kind].append(p)
    return buckets


def _create_pg_database() -> None:
    """Create the PostgreSQL database if it doesn't exist (sync, via psycopg2 or psql)."""
    import subprocess

    result = subprocess.run(
        ["psql", "-d", "postgres", "-tc",
         f"SELECT 1 FROM pg_database WHERE datname = '{PG_DB}'"],
        capture_output=True, text=True,
    )
    if "1" not in result.stdout:
        print(f"  Creating PostgreSQL database '{PG_DB}'...")
        subprocess.run(["createdb", PG_DB], check=True)
    else:
        print(f"  PostgreSQL database '{PG_DB}' already exists")


def _drop_pg_database() -> None:
    """Drop the PostgreSQL database."""
    import subprocess

    subprocess.run(["dropdb", "--if-exists", PG_DB], check=True)


def main() -> None:
    print(f"Repo root:    {REPO_ROOT}")
    print(f"SQLite path:  {SQLITE_PATH}")
    print(f"PostgreSQL:   {PG_URL}")
    print()

    # ------------------------------------------------------------------
    # Collect and classify files
    # ------------------------------------------------------------------
    buckets = collect_files(REPO_ROOT)
    print(f"Found {len(buckets['code'])} code files, {len(buckets['docs'])} doc files\n")

    # ------------------------------------------------------------------
    # Set up backends
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SETUP: Creating databases and backends")
    print("=" * 60)

    # Create tables using throwaway engines (avoids event-loop mismatch
    # with Grover's internal daemon-thread loop).
    _create_pg_database()

    async def _create_tables() -> None:
        for url in [f"sqlite+aiosqlite:///{SQLITE_PATH}", PG_URL]:
            eng = create_async_engine(url, echo=False)
            async with eng.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            await eng.dispose()

    asyncio.run(_create_tables())

    # Create the real engines (will be used on Grover's event loop)
    sqlite_engine = create_async_engine(
        f"sqlite+aiosqlite:///{SQLITE_PATH}", echo=False,
    )
    pg_engine = create_async_engine(PG_URL, echo=False)

    sqlite_factory = async_sessionmaker(
        sqlite_engine, class_=AsyncSession, expire_on_commit=False,
    )
    pg_factory = async_sessionmaker(
        pg_engine, class_=AsyncSession, expire_on_commit=False,
    )

    sqlite_db = DatabaseFileSystem(
        session_factory=sqlite_factory, dialect="sqlite",
    )
    pg_db = DatabaseFileSystem(
        session_factory=pg_factory, dialect="postgresql",
    )

    # ------------------------------------------------------------------
    # Mount both backends on a single Grover instance
    # ------------------------------------------------------------------
    g = Grover()
    g.mount("/code", sqlite_db)
    g.mount("/docs", pg_db)
    print(f"  Mounted /code → SQLite ({SQLITE_PATH.name})")
    print(f"  Mounted /docs → PostgreSQL ({PG_DB})\n")

    try:
        # --------------------------------------------------------------
        # Phase 1: Batch-write files to both backends
        # --------------------------------------------------------------
        print("=" * 60)
        print("PHASE 1: Writing files (batch/transaction mode)")
        print("=" * 60)

        stats = {"code_written": 0, "docs_written": 0, "failed": 0}

        with g:
            # Write code files to /code
            for path in buckets["code"]:
                rel = path.relative_to(REPO_ROOT)
                virtual_path = f"/code/{rel}".replace("\\", "/")
                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as e:
                    print(f"  SKIP {virtual_path} ({e.__class__.__name__})")
                    stats["failed"] += 1
                    continue
                if g.write(virtual_path, content):
                    stats["code_written"] += 1
                else:
                    stats["failed"] += 1

            # Write doc files to /docs
            for path in buckets["docs"]:
                rel = path.relative_to(REPO_ROOT)
                virtual_path = f"/docs/{rel}".replace("\\", "/")
                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as e:
                    print(f"  SKIP {virtual_path} ({e.__class__.__name__})")
                    stats["failed"] += 1
                    continue
                if g.write(virtual_path, content):
                    stats["docs_written"] += 1
                else:
                    stats["failed"] += 1
            # All writes committed together on clean exit

        print(f"\n  Code files written (SQLite):      {stats['code_written']}")
        print(f"  Doc files written (PostgreSQL):    {stats['docs_written']}")
        print(f"  Failed:                           {stats['failed']}")
        if SQLITE_PATH.exists():
            print(f"  SQLite DB size: {SQLITE_PATH.stat().st_size / 1024:.1f} KB")

        # --------------------------------------------------------------
        # Phase 2: Read back and verify from both backends
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 2: Verifying round-trip (read back + compare)")
        print("=" * 60)

        for label, mount, file_list in [
            ("Code (SQLite)", "/code", buckets["code"]),
            ("Docs (PostgreSQL)", "/docs", buckets["docs"]),
        ]:
            verified = 0
            mismatches = 0
            for path in file_list:
                rel = path.relative_to(REPO_ROOT)
                virtual_path = f"{mount}/{rel}".replace("\\", "/")
                try:
                    original = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                content = g.read(virtual_path)
                if content is None:
                    mismatches += 1
                elif content == original:
                    verified += 1
                else:
                    mismatches += 1
            print(f"\n  {label}: {verified} verified, {mismatches} mismatches")

        # --------------------------------------------------------------
        # Phase 3: Explore the virtual filesystem
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 3: Exploring the virtual filesystem")
        print("=" * 60)

        # List root — should show both mounts
        root_entries = g.list_dir("/")
        print(f"\n  / (root) — {len(root_entries)} mount(s):")
        for e in root_entries:
            print(f"    {'DIR ' if e['is_directory'] else 'FILE'} {e['name']}")

        # List each mount root
        for mount in ["/code", "/docs"]:
            entries = g.list_dir(mount)
            print(f"\n  {mount}/ — {len(entries)} entries:")
            for e in sorted(entries, key=lambda x: x["name"])[:15]:
                kind = "DIR " if e["is_directory"] else "FILE"
                print(f"    {kind} {e['name']}")
            if len(entries) > 15:
                print(f"    ... ({len(entries) - 15} more)")

        # Cross-mount exists checks
        print(f"\n  exists('/code/src/grover/_grover.py'): "
              f"{g.exists('/code/src/grover/_grover.py')}")
        print(f"  exists('/docs/README.md'): "
              f"{g.exists('/docs/README.md')}")
        print(f"  exists('/code/README.md'): "
              f"{g.exists('/code/README.md')}  (should be False — README is in /docs)")

        # --------------------------------------------------------------
        # Phase 4: Cross-mount graph + search
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 4: Graph info (spans both mounts)")
        print("=" * 60)

        graph = g.graph
        print(f"\n  Graph nodes: {graph.node_count}")
        print(f"  Graph edges: {graph.edge_count}")

        code_nodes = [n for n in graph.nodes() if n.startswith("/code/")]
        docs_nodes = [n for n in graph.nodes() if n.startswith("/docs/")]
        print(f"  Nodes in /code: {len(code_nodes)}")
        print(f"  Nodes in /docs: {len(docs_nodes)}")

        if code_nodes:
            print(f"  Sample /code nodes: {code_nodes[:3]}")
        if docs_nodes:
            print(f"  Sample /docs nodes: {docs_nodes[:3]}")

        # --------------------------------------------------------------
        # Phase 5: Edit + versioning on each backend
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 5: Edit + versioning (both backends)")
        print("=" * 60)

        ufs = g.fs
        for mount, label in [("/code", "SQLite"), ("/docs", "PostgreSQL")]:
            test_path = f"{mount}/test_versioning.txt"
            g.write(test_path, "version 1\n")
            g.edit(test_path, "version 1", "version 2")
            g.edit(test_path, "version 2", "version 3")

            current = g.read(test_path)
            versions = g._run(ufs.list_versions(test_path))
            print(f"\n  [{label}] {test_path}")
            print(f"    Current: {current!r}")
            print(f"    Versions: {len(versions)}")
            for v in versions:
                vc = g._run(ufs.get_version_content(test_path, v.version))
                print(f"      v{v.version}: {vc!r}")

            # Restore to v1
            restore = g._run(ufs.restore_version(test_path, 1))
            restored = g.read(test_path)
            print(f"    Restored to v1: {restore.success}, content={restored!r}")

            # Clean up
            g.delete(test_path)

        # Save state
        g.save()
        print("\n  State saved.")

    finally:
        g.close()

    # Dispose engines to release all connections before dropping databases
    async def _dispose_engines() -> None:
        await sqlite_engine.dispose()
        await pg_engine.dispose()

    asyncio.run(_dispose_engines())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANUP: Deleting databases")
    print("=" * 60)
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        print(f"  Deleted {SQLITE_PATH}")
    _drop_pg_database()
    print(f"  Dropped PostgreSQL database '{PG_DB}'")
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

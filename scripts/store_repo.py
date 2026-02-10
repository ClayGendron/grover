"""Store all text files in this repo into a SQLite-backed Grover instance.

Uses the mount-first Grover API with a DatabaseFileSystem backend.
Files are mounted at /repo and accessed through Grover's sync API.

Usage:
    uv run python scripts/store_repo.py
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover import Grover
from grover.fs.database_fs import DatabaseFileSystem

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "grover_repo.db"
MOUNT = "/project"

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


def collect_files(root: Path) -> list[Path]:
    """Walk the repo and collect text files."""
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            if should_skip(p):
                continue
            files.append(p)
        except OSError:
            continue
    return files


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"DB path:   {DB_PATH}")
    print()

    # Collect files from disk
    files = collect_files(REPO_ROOT)
    print(f"Found {len(files)} text files to store\n")

    # Create async engine + session factory for a SQLite database backend
    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)

    # Create tables synchronously before starting
    import asyncio

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    asyncio.run(_create_tables())

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False,
    )

    # Create Grover with mount-first API
    db = DatabaseFileSystem(session_factory=session_factory, dialect="sqlite")
    g = Grover()
    g.mount(MOUNT, db)

    try:
        # --------------------------------------------------------------
        # Phase 1: Batch-write all repo files (transaction mode)
        # --------------------------------------------------------------
        print("=" * 60)
        print("PHASE 1: Writing files via Grover (batch/transaction mode)")
        print("=" * 60)

        written = 0
        failed = 0
        with g:
            for path in files:
                rel = path.relative_to(REPO_ROOT)
                virtual_path = f"{MOUNT}/{rel}".replace("\\", "/")

                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as e:
                    print(f"  SKIP {virtual_path} ({e.__class__.__name__})")
                    failed += 1
                    continue

                ok = g.write(virtual_path, content)
                if ok:
                    written += 1
                else:
                    print(f"  FAIL {virtual_path}")
                    failed += 1
            # All writes committed together on clean exit from `with g:`

        print(f"\n  Written: {written}")
        print(f"  Failed:  {failed}")
        print(f"  DB size: {DB_PATH.stat().st_size / 1024:.1f} KB\n")

        # --------------------------------------------------------------
        # Phase 2: Read back and verify
        # --------------------------------------------------------------
        print("=" * 60)
        print("PHASE 2: Verifying round-trip (read back + compare)")
        print("=" * 60)

        verified = 0
        mismatches = 0
        for path in files:
            rel = path.relative_to(REPO_ROOT)
            virtual_path = f"{MOUNT}/{rel}".replace("\\", "/")

            try:
                original = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            content = g.read(virtual_path)
            if content is None:
                mismatches += 1
                print(f"  MISS {virtual_path}")
                continue

            if content == original:
                verified += 1
            else:
                mismatches += 1
                print(f"  DIFF {virtual_path}: "
                      f"expected {len(original)} chars, got {len(content)}")

        print(f"\n  Verified:   {verified}")
        print(f"  Mismatches: {mismatches}\n")

        # --------------------------------------------------------------
        # Phase 3: Explore the virtual filesystem
        # --------------------------------------------------------------
        print("=" * 60)
        print("PHASE 3: Exploring the virtual filesystem")
        print("=" * 60)

        # List root (shows mount points)
        root_entries = g.list_dir("/")
        print(f"\n  / (root) — {len(root_entries)} mount(s):")
        for e in root_entries:
            print(f"    {'DIR ' if e['is_directory'] else 'FILE'} {e['name']}")

        # List mount root
        mount_entries = g.list_dir(MOUNT)
        print(f"\n  {MOUNT}/ — {len(mount_entries)} entries:")
        for e in sorted(mount_entries, key=lambda x: x["name"]):
            kind = "DIR " if e["is_directory"] else "FILE"
            print(f"    {kind} {e['name']}")

        # List src/grover
        grover_entries = g.list_dir(f"{MOUNT}/src/grover")
        print(f"\n  {MOUNT}/src/grover/ — {len(grover_entries)} entries:")
        for e in sorted(grover_entries, key=lambda x: x["name"]):
            kind = "DIR " if e["is_directory"] else "FILE"
            print(f"    {kind} {e['name']}")

        # Exists checks
        print(f"\n  exists('{MOUNT}/pyproject.toml'): "
              f"{g.exists(f'{MOUNT}/pyproject.toml')}")
        print(f"  exists('{MOUNT}/nonexistent.txt'): "
              f"{g.exists(f'{MOUNT}/nonexistent.txt')}")

        # Read a file
        readme = g.read(f"{MOUNT}/README.md")
        if readme:
            lines = readme.split("\n")
            print(f"\n  {MOUNT}/README.md ({len(lines)} lines):")
            for line in lines[:5]:
                print(f"    {line}")
            print("    ...")

        # --------------------------------------------------------------
        # Phase 4: Edit + versioning via underlying filesystem
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 4: Edit + versioning")
        print("=" * 60)

        test_path = f"{MOUNT}/test_versioning.txt"
        g.write(test_path, "version 1\n")
        g.edit(test_path, "version 1", "version 2")
        g.edit(test_path, "version 2", "version 3")

        current = g.read(test_path)
        print(f"\n  Current content: {current!r}")

        # Access version history via the underlying UnifiedFileSystem
        ufs = g.fs
        versions = g._run(ufs.list_versions(test_path))
        print(f"  Version records: {len(versions)}")
        for v in versions:
            vc = g._run(ufs.get_version_content(test_path, v.version))
            print(f"    v{v.version}: {vc!r}")

        # Restore to v1
        restore = g._run(ufs.restore_version(test_path, 1))
        restored = g.read(test_path)
        print(f"\n  Restored to v1: success={restore.success}")
        print(f"  Content after restore: {restored!r}")

        # --------------------------------------------------------------
        # Phase 5: Graph check
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 5: Graph info")
        print("=" * 60)

        graph = g.graph
        print(f"\n  Graph nodes: {graph.node_count}")
        print(f"  Graph edges: {graph.edge_count}")
        if graph.node_count > 0:
            sample = graph.nodes()[:5]
            print(f"  Sample nodes: {sample}")

        # --------------------------------------------------------------
        # Phase 6: Delete test
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("PHASE 6: Delete test")
        print("=" * 60)

        ok = g.delete(test_path)
        print(f"\n  Deleted {test_path}: {ok}")
        print(f"  exists after delete: {g.exists(test_path)}")

        # Save state
        g.save()
        print(f"\n  Saved. DB size: {DB_PATH.stat().st_size / 1024:.1f} KB")

    finally:
        g.close()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANUP: Deleting database")
    print("=" * 60)
    DB_PATH.unlink()
    print(f"  Deleted {DB_PATH}")
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

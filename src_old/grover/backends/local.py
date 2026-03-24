"""LocalFileSystem — thin wrapper over DatabaseFileSystem with disk storage."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from grover.backends.database import DatabaseFileSystem
from grover.models.internal.detail import ReadDetail, ReconcileDetail
from grover.models.internal.ref import File
from grover.models.internal.results import GroverResult
from grover.providers.storage.disk import DiskStorageProvider
from grover.util.content import get_similar_files, is_binary_file
from grover.util.operations import write_file
from grover.util.paths import normalize_path, to_trash_path, validate_path

logger = logging.getLogger(__name__)


def _workspace_slug(workspace_dir: Path) -> str:
    """Derive a directory-safe slug from a workspace path."""
    try:
        relative = workspace_dir.resolve().relative_to(Path.home())
    except ValueError:
        relative = Path(str(workspace_dir.resolve()).lstrip("/"))
    return str(relative).replace("/", "_").strip("_")


def _default_data_dir(workspace_dir: Path) -> Path:
    """Return the global data directory for a given workspace."""
    return Path.home() / ".grover" / _workspace_slug(workspace_dir)


class LocalFileSystem(DatabaseFileSystem):
    """Local disk storage with SQLite versioning.

    Thin wrapper over :class:`DatabaseFileSystem`:

    1. Creates a :class:`DiskStorageProvider` for content I/O and disk queries
    2. Manages SQLite engine lifecycle (``open`` / ``close``)
    3. Overrides a few methods for disk-specific behavior (binary detection,
       disk-only file backup on delete, disk mkdir, restore-to-disk, reconcile)

    Everything else — write, edit, move, copy, exists, get_info, glob, grep,
    tree, list_dir, versions, chunks, trash, connections — is inherited from
    ``DatabaseFileSystem`` and delegates to the ``DiskStorageProvider``.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        **provider_kwargs: object,
    ) -> None:
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir(self.workspace_dir)

        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._init_lock = asyncio.Lock()

        # Typed reference for disk-specific operations in overrides
        self._disk = DiskStorageProvider(self.workspace_dir)

        super().__init__(
            storage_provider=self._disk,
            **provider_kwargs,  # type: ignore[arg-type]
        )

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession] | None:
        """The async session factory, available after ``open()``."""
        return self._session_factory

    @property
    def engine(self) -> AsyncEngine | None:
        """The async engine, available after ``open()``."""
        return self._engine

    # ------------------------------------------------------------------
    # Database Management
    # ------------------------------------------------------------------

    async def _ensure_db(self) -> None:
        """Initialize database if needed."""
        if self._session_factory is not None:
            return
        async with self._init_lock:
            if self._session_factory is not None:
                return

            self.data_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.data_dir / "file_versions.db"

            from sqlalchemy.ext.asyncio import create_async_engine

            self._engine = create_async_engine(
                f"sqlite+aiosqlite:///{db_path}",
                echo=False,
            )

            @event.listens_for(self._engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
                cursor.execute("PRAGMA journal_mode=WAL")
                result = cursor.fetchone()
                if result[0].lower() != "wal":
                    logging.getLogger(__name__).warning(
                        "WAL mode not active, got: %s",
                        result[0],
                    )
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA synchronous=FULL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            async with self._engine.begin() as conn:
                fm_table = self.file_model.__table__  # type: ignore[unresolved-attribute]
                fv_table = self.file_version_model.__table__  # type: ignore[unresolved-attribute]
                fc_table = self.file_chunk_model.__table__  # type: ignore[unresolved-attribute]
                edge_table = self.file_connection_model.__table__  # type: ignore[unresolved-attribute]
                await conn.run_sync(lambda c: fm_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: fv_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: fc_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: edge_table.create(c, checkfirst=True))

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Initialize the SQLite database."""
        await self._ensure_db()

    async def close(self) -> None:
        """Close database engine and release resources."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    # ------------------------------------------------------------------
    # Override: read — binary detection + similar file suggestions
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        """Read file with binary detection and similar file suggestions."""

        valid, error = validate_path(path)
        if not valid:
            return GroverResult(success=False, message=error)

        path = normalize_path(path)

        try:
            actual_path = await self._disk._resolve_path(path)
        except PermissionError as e:
            return GroverResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            parent_exists = await asyncio.to_thread(actual_path.parent.exists)
            if parent_exists:
                suggestions = await asyncio.to_thread(get_similar_files, actual_path.parent, actual_path.name)
                if suggestions:
                    suggestion_text = "\n".join(f"  {s}" for s in suggestions)
                    return GroverResult(
                        success=False,
                        message=f"File not found: {path}\n\nDid you mean?\n{suggestion_text}",
                    )
            return GroverResult(success=False, message=f"File not found: {path}")

        if await asyncio.to_thread(is_binary_file, actual_path):
            return GroverResult(success=False, message=f"Cannot read binary file: {path}")

        if await asyncio.to_thread(actual_path.is_dir):
            return GroverResult(success=False, message=f"Path is a directory, not a file: {path}")

        content = await self._read_content(path, session)

        if content is None:
            return GroverResult(success=False, message=f"Could not read file: {path}")

        from grover.models.internal.ref import File

        lines = content.split("\n")
        total_lines = len(lines)
        if total_lines == 0 or (total_lines == 1 and lines[0] == ""):
            msg = f"File is empty: {path}"
            file = File(path=path, content="", lines=0)
        else:
            msg = f"Read {total_lines} lines from {path}"
            file = File(path=path, content=content, lines=total_lines)

        file.evidence = [
            ReadDetail(
                operation="read",
                success=True,
                message=msg,
            )
        ]
        return GroverResult(success=True, message=msg, files=[file])

    # ------------------------------------------------------------------
    # Override: delete — backup disk-only files before delete
    # ------------------------------------------------------------------

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        """Delete file, backing up content to the database first.

        Content-before-commit: DB soft-delete -> flush -> unlink disk -> return.
        VFS commits after.
        """
        norm = normalize_path(path)

        # Read content from disk before anything else
        content = await self._read_content(norm, session)

        # Ensure a DB record exists so delete can soft-delete it
        if content is not None:
            file = await self._get_file_record(session, norm)
            if file is None:
                # Disk-only file: create a DB record + version 1 snapshot
                await write_file(
                    norm,
                    content,
                    "backup",
                    True,
                    session,
                    get_file_record=self._get_file_record,
                    versioning=self.version_provider,
                    ensure_parent_dirs=self._ensure_parent_dirs,
                    file_model=self.file_model,
                    read_content=self._read_content,
                    write_content=self._write_content,
                )

        result = await super().delete(norm, permanent, session=session, user_id=user_id)

        # Remove from disk regardless of soft/permanent
        if result.success:
            await self._delete_content(norm, session)

        return result

    # ------------------------------------------------------------------
    # Override: mkdir — also create disk directory
    # ------------------------------------------------------------------

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        """Create directory in database and on disk."""
        result = await super().mkdir(path, parents, session=session, user_id=user_id)

        if result.success:
            actual_path = await self._disk._resolve_path(path)
            await asyncio.to_thread(actual_path.mkdir, parents=True, exist_ok=True)

        return result

    # ------------------------------------------------------------------
    # Override: reconcile — disk-to-DB sync
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Walk disk, compare with DB, create/update/soft-delete as needed."""
        created_files: list[File] = []
        deleted_files: list[File] = []

        # Walk workspace files
        disk_paths: set[str] = set()

        def _walk() -> list[str]:
            import os

            items = []
            for root, dirs, files in os.walk(self.workspace_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    if name.startswith("."):
                        continue
                    full = Path(root) / name
                    try:
                        vpath = self._disk._to_virtual_path(full)
                        items.append(vpath)
                    except (ValueError, PermissionError):
                        continue
            return items

        async def _noop_write(
            _path: str,
            _content: str,
            _session: AsyncSession,
        ) -> None:
            pass

        items = await asyncio.to_thread(_walk)
        for vpath in items:
            disk_paths.add(vpath)

            file = await self._get_file_record(session, vpath)
            if file is None:
                # File on disk but not in DB — create DB record only
                content = await self._read_content(vpath, session)
                if content is not None:
                    await write_file(
                        vpath,
                        content,
                        "reconcile",
                        True,
                        session,
                        get_file_record=self._get_file_record,
                        versioning=self.version_provider,
                        ensure_parent_dirs=self._ensure_parent_dirs,
                        file_model=self.file_model,
                        read_content=self._read_content,
                        write_content=_noop_write,
                    )
                    created_files.append(
                        File(
                            path=vpath,
                            evidence=[ReconcileDetail(operation="reconcile", action="created")],
                        )
                    )

        # Check DB records against disk
        from sqlmodel import select

        model = self.file_model
        result = await session.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        for file in result.scalars().all():
            if file.path not in disk_paths:
                exists = await self._disk.exists(file.path)
                if not exists:
                    original_path = file.path
                    file.original_path = file.path
                    file.path = to_trash_path(file.path, file.id)
                    file.deleted_at = datetime.now(UTC)
                    deleted_files.append(
                        File(
                            path=original_path,
                            evidence=[ReconcileDetail(operation="reconcile", action="deleted")],
                        )
                    )

        await session.flush()

        all_files = created_files + deleted_files
        return GroverResult(
            success=True,
            message=f"Reconcile complete: {len(created_files)} created, {len(deleted_files)} deleted",
            files=all_files,
        )

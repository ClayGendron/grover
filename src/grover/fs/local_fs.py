"""LocalFileSystem — thin wrapper over DatabaseFileSystem with disk storage."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from grover.types.operations import (
    DeleteResult,
    MkdirResult,
    ReadResult,
    ReconcileResult,
    RestoreResult,
)

from .database_fs import DatabaseFileSystem
from .operations import paginate_content, write_file
from .providers.storage.disk import DiskStorageProvider
from .utils import (
    get_similar_files,
    is_binary_file,
    normalize_path,
    to_trash_path,
    validate_path,
)

if TYPE_CHECKING:
    from grover.models.chunks import FileChunkBase
    from grover.models.connections import FileConnectionBase
    from grover.models.files import FileBase, FileVersionBase

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
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
        file_connection_model: type[FileConnectionBase] | None = None,
        schema: str | None = None,
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
            dialect="sqlite",
            file_model=file_model,
            file_version_model=file_version_model,
            file_chunk_model=file_chunk_model,
            file_connection_model=file_connection_model,
            schema=schema,
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
                fm_table = self._file_model.__table__  # type: ignore[unresolved-attribute]
                fv_table = self._file_version_model.__table__  # type: ignore[unresolved-attribute]
                fc_table = self._file_chunk_model.__table__  # type: ignore[unresolved-attribute]
                edge_table = self._file_connection_model.__table__  # type: ignore[unresolved-attribute]
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
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ReadResult:
        """Read file with binary detection and similar file suggestions."""
        sess = self._require_session(session)

        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        try:
            actual_path = await self._disk._resolve_path(path)
        except PermissionError as e:
            return ReadResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            parent_exists = await asyncio.to_thread(actual_path.parent.exists)
            if parent_exists:
                suggestions = await asyncio.to_thread(
                    get_similar_files, actual_path.parent, actual_path.name
                )
                if suggestions:
                    suggestion_text = "\n".join(f"  {s}" for s in suggestions)
                    return ReadResult(
                        success=False,
                        message=f"File not found: {path}\n\nDid you mean?\n{suggestion_text}",
                    )
            return ReadResult(success=False, message=f"File not found: {path}")

        if await asyncio.to_thread(is_binary_file, actual_path):
            return ReadResult(success=False, message=f"Cannot read binary file: {path}")

        if await asyncio.to_thread(actual_path.is_dir):
            return ReadResult(success=False, message=f"Path is a directory, not a file: {path}")

        content = await self._read_content(path, sess)

        if content is None:
            return ReadResult(success=False, message=f"Could not read file: {path}")

        return paginate_content(content, path, offset, limit)

    # ------------------------------------------------------------------
    # Override: delete — backup disk-only files before delete
    # ------------------------------------------------------------------

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        """Delete file, backing up content to the database first.

        Content-before-commit: DB soft-delete -> flush -> unlink disk -> return.
        VFS commits after.
        """
        sess = self._require_session(session)
        norm = normalize_path(path)

        # Read content from disk before anything else
        content = await self._read_content(norm, sess)

        # Ensure a DB record exists so delete can soft-delete it
        if content is not None:
            file = await self._get_file_record(sess, norm)
            if file is None:
                # Disk-only file: create a DB record + version 1 snapshot
                await write_file(
                    norm,
                    content,
                    "backup",
                    True,
                    sess,
                    get_file_record=self._get_file_record,
                    versioning=self.version_provider,
                    directories=self.directories,
                    file_model=self._file_model,
                    read_content=self._read_content,
                    write_content=self._write_content,
                )

        result = await super().delete(norm, permanent, session=sess, user_id=user_id)

        # Remove from disk regardless of soft/permanent
        if result.success:
            await self._delete_content(norm, sess)

        return result

    # ------------------------------------------------------------------
    # Override: mkdir — also create disk directory
    # ------------------------------------------------------------------

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        """Create directory in database and on disk."""
        result = await super().mkdir(path, parents, session=session, user_id=user_id)

        if result.success:
            actual_path = await self._disk._resolve_path(path)
            await asyncio.to_thread(actual_path.mkdir, parents=True, exist_ok=True)

        return result

    # ------------------------------------------------------------------
    # Override: restore_from_trash — write content back to disk
    # ------------------------------------------------------------------

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> RestoreResult:
        """Restore a file from trash, writing content back to disk."""
        sess = self._require_session(session)
        result = await self.trash.restore_from_trash(
            sess, path, self._get_file_record, owner_id=owner_id
        )
        if not result.success:
            return result

        restored_path = result.path or path
        file = await self._get_file_record(sess, restored_path)
        if file:
            if file.is_directory:
                from sqlmodel import select

                model = self._file_model
                children_result = await sess.execute(
                    select(model).where(
                        model.path.startswith(restored_path + "/"),
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
                for child in children_result.scalars().all():
                    if not child.is_directory:
                        vc = await self.get_version_content(
                            child.path,
                            child.current_version,
                            session=sess,
                        )
                        if vc.success and vc.content is not None:
                            await self._write_content(child.path, vc.content, sess)
            else:
                vc = await self.get_version_content(
                    restored_path,
                    file.current_version,
                    session=sess,
                )
                if vc.success and vc.content is not None:
                    await self._write_content(restored_path, vc.content, sess)

        return result

    # ------------------------------------------------------------------
    # Override: reconcile — disk-to-DB sync
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> ReconcileResult:
        """Walk disk, compare with DB, create/update/soft-delete as needed."""
        sess = self._require_session(session)
        stats = ReconcileResult()

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

            file = await self._get_file_record(sess, vpath)
            if file is None:
                # File on disk but not in DB — create DB record only
                content = await self._read_content(vpath, sess)
                if content is not None:
                    await write_file(
                        vpath,
                        content,
                        "reconcile",
                        True,
                        sess,
                        get_file_record=self._get_file_record,
                        versioning=self.version_provider,
                        directories=self.directories,
                        file_model=self._file_model,
                        read_content=self._read_content,
                        write_content=_noop_write,
                    )
                    stats.created += 1

        # Check DB records against disk
        from sqlmodel import select

        model = self._file_model
        result = await sess.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        for file in result.scalars().all():
            if file.path not in disk_paths:
                exists = await self._disk.exists(file.path)
                if not exists:
                    file.original_path = file.path
                    file.path = to_trash_path(file.path, file.id)
                    file.deleted_at = datetime.now(UTC)
                    stats.deleted += 1

        await sess.flush()

        # Verify version chain integrity
        verification_results = await self.verify_all_versions(session=sess)
        stats.chain_errors = sum(r.versions_failed for r in verification_results)

        return stats

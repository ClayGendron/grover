"""LocalFileSystem — disk + SQLite versioning."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generic

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .base import FV, BaseFileSystem, F
from .types import DeleteResult, FileInfo, ListResult, MkdirResult, ReadResult, RestoreResult
from .utils import get_similar_files, is_binary_file, normalize_path, validate_path


def _workspace_slug(workspace_dir: Path) -> str:
    """Derive a directory-safe slug from a workspace path.

    Converts to a path relative to the user's home directory, then replaces
    path separators with underscores.  For paths outside of home, uses the
    full absolute path minus the leading ``/``.

    Examples::

        ~/Git/Repos/grover  →  Git_Repos_grover
        /opt/projects/app   →  opt_projects_app
    """
    try:
        relative = workspace_dir.resolve().relative_to(Path.home())
    except ValueError:
        relative = Path(str(workspace_dir.resolve()).lstrip("/"))
    return str(relative).replace("/", "_").strip("_")


def _default_data_dir(workspace_dir: Path) -> Path:
    """Return the global data directory for a given workspace.

    Stored under ``~/.grover/{slug}/`` so project directories stay clean.
    """
    return Path.home() / ".grover" / _workspace_slug(workspace_dir)


class LocalFileSystem(BaseFileSystem[F, FV], Generic[F, FV]):
    """Local file system with disk storage and SQLite versioning.

    - Files stored on disk at ``{workspace_dir}/{path}``
    - Metadata and versions in SQLite at ``~/.grover/{slug}/file_versions.db``
    - IDE, git, and other tools can see/edit files directly
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        file_model: type[F] | None = None,
        file_version_model: type[FV] | None = None,
        schema: str | None = None,
    ) -> None:
        super().__init__(
            dialect="sqlite",
            file_model=file_model,
            file_version_model=file_version_model,
            schema=schema,
        )

        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir(self.workspace_dir)

        self._engine = None
        self._session_factory = None
        self._init_lock = asyncio.Lock()

    # =========================================================================
    # Database Management
    # =========================================================================

    async def _ensure_db(self) -> None:
        """Initialize database if needed."""
        if self._session_factory is not None:
            return
        async with self._init_lock:
            if self._session_factory is not None:
                return

            self.data_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.data_dir / "file_versions.db"

            self._engine = create_async_engine(
                f"sqlite+aiosqlite:///{db_path}",
                echo=False,
            )

            # Scope the pragma listener to this engine only
            @event.listens_for(self._engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
                cursor.execute("PRAGMA journal_mode=WAL")
                result = cursor.fetchone()
                if result[0].lower() != "wal":
                    import logging
                    logging.getLogger(__name__).warning(
                        "WAL mode not active, got: %s", result[0],
                    )
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA synchronous=FULL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            async with self._engine.begin() as conn:
                fm_table = self._file_model.__table__  # type: ignore[unresolved-attribute]
                fv_table = self._file_version_model.__table__  # type: ignore[unresolved-attribute]
                await conn.run_sync(lambda c: fm_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: fv_table.create(c, checkfirst=True))

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    async def close(self) -> None:
        """Close database engine and release resources."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def __aenter__(self) -> LocalFileSystem[F, FV]:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        await self.close()

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _resolve_path_sync(self, virtual_path: str) -> Path:
        """Convert virtual path to an actual disk path within the workspace.

        Walks each path component to reject symlinks (prevents TOCTOU attacks)
        and verifies the resolved path stays within ``workspace_dir``.

        Raises:
            PermissionError: On symlinks or path traversal.
        """
        virtual_path = normalize_path(virtual_path)
        rel = virtual_path.lstrip("/")
        if not rel:
            return self.workspace_dir

        candidate = self.workspace_dir / rel

        # Walk each component checking for symlinks
        current = self.workspace_dir
        for part in Path(rel).parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError(
                    f"Symlinks not allowed: {virtual_path} contains symlink at "
                    f"{current.relative_to(self.workspace_dir)}"
                )

        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_dir.resolve())
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: {virtual_path} resolves outside workspace"
            ) from None

        return resolved

    async def _resolve_path(self, virtual_path: str) -> Path:
        """Async wrapper around _resolve_path_sync to avoid blocking the event loop."""
        return await asyncio.to_thread(self._resolve_path_sync, virtual_path)

    def _to_virtual_path(self, physical_path: Path) -> str:
        """Convert a physical path back to a virtual path."""
        rel = physical_path.resolve().relative_to(self.workspace_dir.resolve())
        vpath = "/" + str(rel).replace("\\", "/")
        return vpath if vpath != "/." else "/"

    async def _read_content(self, path: str, session: AsyncSession) -> str | None:
        try:
            actual_path = await self._resolve_path(path)
        except (PermissionError, ValueError):
            return None

        def _do_read() -> str | None:
            if not actual_path.exists() or actual_path.is_dir():
                return None
            return actual_path.read_text(encoding="utf-8")

        try:
            return await asyncio.to_thread(_do_read)
        except (UnicodeDecodeError, PermissionError, OSError):
            return None

    async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
        actual_path = await self._resolve_path(path)

        def _do_write() -> None:
            actual_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=actual_path.parent,
                prefix=".tmp_",
                suffix=actual_path.suffix,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp_path).replace(actual_path)
            except Exception:
                tmp = Path(tmp_path)
                if tmp.exists():
                    tmp.unlink()
                raise

        await asyncio.to_thread(_do_write)

    async def _delete_content(self, path: str, session: AsyncSession) -> None:
        try:
            actual_path = await self._resolve_path(path)
        except (PermissionError, ValueError):
            return

        def _do_delete() -> None:
            try:
                if actual_path.is_dir():
                    shutil.rmtree(actual_path)
                else:
                    actual_path.unlink()
            except FileNotFoundError:
                pass

        await asyncio.to_thread(_do_delete)

    async def _content_exists(self, path: str, session: AsyncSession) -> bool:
        try:
            actual_path = await self._resolve_path(path)
            return await asyncio.to_thread(actual_path.exists)
        except (PermissionError, ValueError):
            return False

    # =========================================================================
    # Override read() to add binary check and suggestions
    # =========================================================================

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession,
    ) -> ReadResult:
        """Read file with binary check and similar file suggestions."""
        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
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

        content = await self._read_content(path, session)

        if content is None:
            return ReadResult(success=False, message=f"Could not read file: {path}")

        return self._paginate_content(content, path, offset, limit)

    # =========================================================================
    # Override delete to back up content to DB before removing from disk
    # =========================================================================

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession,
    ) -> DeleteResult:
        """Delete file, backing up content to the database first.

        For files that exist on disk but have no DB record, a record and
        version snapshot are created before soft-deleting, so the content
        is always recoverable from the database.
        """
        norm = normalize_path(path)

        # Read content from disk before anything else
        content = await self._read_content(norm, session)

        # Ensure a DB record exists so super().delete() can soft-delete it
        if content is not None:
            file = await self._get_file(session, norm)
            if file is None:
                # Disk-only file: create a DB record + version 1 snapshot
                await super().write(norm, content, created_by="backup", session=session)

        result = await super().delete(norm, permanent, session=session)

        # Remove from disk regardless of soft/permanent
        if result.success:
            await self._delete_content(norm, session)

        return result

    # =========================================================================
    # Override restore_from_trash to write content back to disk
    # =========================================================================

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession,
    ) -> RestoreResult:
        """Restore a file from trash, writing content back to disk."""
        result = await super().restore_from_trash(path, session=session)
        if not result.success:
            return result

        restored_path = result.file_path or path
        file = await self._get_file(session, restored_path)
        if file:
            if file.is_directory:
                # Restore children's disk content
                model = self._file_model
                from sqlmodel import select
                children_result = await session.execute(
                    select(model).where(
                        model.path.startswith(restored_path + "/"),  # type: ignore[union-attr]
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
                for child in children_result.scalars().all():
                    if not child.is_directory:
                        child_content = await self.get_version_content(
                            child.path, child.current_version, session=session,
                        )
                        if child_content is not None:
                            await self._write_content(child.path, child_content, session)
            else:
                content = await self.get_version_content(
                    restored_path, file.current_version, session=session,
                )
                if content is not None:
                    await self._write_content(restored_path, content, session)

        return result

    # =========================================================================
    # Override mkdir to create on disk too
    # =========================================================================

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession,
    ) -> MkdirResult:
        """Create directory in database and on disk."""
        result = await super().mkdir(path, parents, session=session)

        if result.success:
            actual_path = await self._resolve_path(path)
            await asyncio.to_thread(actual_path.mkdir, parents=True, exist_ok=True)

        return result

    # =========================================================================
    # Override list_dir to sync with disk
    # =========================================================================

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession,
    ) -> ListResult:
        """List directory, including files only on disk."""
        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return ListResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return ListResult(success=False, message=f"Directory not found: {path}")

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return ListResult(success=False, message=f"Not a directory: {path}")

        entries: list[FileInfo] = []
        disk_items = await asyncio.to_thread(lambda: list(actual_path.iterdir()))

        for item in disk_items:
            if item.name.startswith("."):
                continue

            item_path = f"{path}/{item.name}" if path != "/" else f"/{item.name}"
            item_path = normalize_path(item_path)

            file = await self._get_file(session, item_path)

            entries.append(
                FileInfo(
                    path=item_path,
                    name=item.name,
                    is_directory=item.is_dir(),
                    size_bytes=(
                        file.size_bytes
                        if file
                        else (item.stat().st_size if item.is_file() else None)
                    ),
                    mime_type=file.mime_type if file else None,
                    version=file.current_version if file else 1,
                    created_at=file.created_at if file else None,
                    updated_at=file.updated_at if file else None,
                )
            )

        return ListResult(
            success=True,
            message=f"Listed {len(entries)} items in {path}",
            entries=entries,
            path=path,
        )

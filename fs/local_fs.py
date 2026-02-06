"""
Local File System - files on disk, versioning in SQLite.

Files are stored on the actual disk (readable by IDE, git, etc.),
while metadata and version history are tracked in a SQLite database.

Database location: ~/.datum/datum.db (configurable)
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..db_models import FileVersion, WorkspaceFile
from .base import BaseFileSystem
from .types import FileInfo, ListResult, ReadResult
from .utils import get_similar_files, is_binary_file, normalize_path, validate_path

# Default locations
DEFAULT_DB_DIR = Path.home() / ".datum"
DEFAULT_WORKSPACE = Path.cwd() / "workspace"


# Disable foreign key checks for SQLite (we don't have a users table locally)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    # Only run PRAGMA for SQLite connections (not SQL Server or other DBs)
    connection_module = type(dbapi_connection).__module__
    if "sqlite" in connection_module or "aiosqlite" in connection_module:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.close()


class LocalFileSystem(BaseFileSystem):
    """
    Local file system with disk storage and SQLite versioning.

    - Files stored on disk at {workspace_dir}/{path}
    - Metadata and versions in SQLite at {db_path}
    - IDE, git, and other tools can see/edit files directly

    USAGE:
        fs = LocalFileSystem(
            user_id="user123",
            workspace_dir="./workspace",
        )
        await fs.write("/main.py", "print('hello')")
        result = await fs.read("/main.py")
    """

    def __init__(
        self,
        user_id: str = "local",
        workspace_dir: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        """
        Initialize the local file system.

        Args:
            user_id: User ID for file ownership (default "local")
            workspace_dir: Directory for files on disk. Defaults to ./workspace
            db_path: Path to SQLite DB. Defaults to ~/.datum/datum.db
        """
        super().__init__(user_id)

        self.workspace_dir = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_DIR / "datum.db"

        self._engine = None
        self._session_factory = None
        self._session: AsyncSession | None = None

    # =========================================================================
    # Database Management
    # =========================================================================

    async def _ensure_db(self) -> None:
        """Initialize database if needed."""
        if self._engine is not None:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: WorkspaceFile.__table__.create(sync_conn, checkfirst=True)
            )
            await conn.run_sync(
                lambda sync_conn: FileVersion.__table__.create(sync_conn, checkfirst=True)
            )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def close(self):
        """Close database connection."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def __aenter__(self) -> "LocalFileSystem":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager - commit or rollback then close."""
        if self._session is not None:
            try:
                if exc_type is None:
                    await self._session.commit()
                else:
                    await self._session.rollback()
            finally:
                await self.close()

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    async def _get_session(self) -> AsyncSession:
        """Get or create database session."""
        await self._ensure_db()
        if self._session is None:
            self._session = self._session_factory()  # type: ignore
        return self._session

    async def _commit(self, session: AsyncSession) -> None:
        """Commit changes to database."""
        await session.commit()

    def _resolve_path(self, virtual_path: str) -> Path:
        """Convert virtual path to actual disk path."""
        virtual_path = normalize_path(virtual_path)
        relative = virtual_path.lstrip("/")
        actual = self.workspace_dir / relative

        # Security: prevent path traversal
        resolved = actual.resolve()
        workspace_resolved = self.workspace_dir.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            raise ValueError(f"Path traversal attempt: {virtual_path}")

        return actual

    async def _read_content(self, path: str) -> str | None:
        """Read file content from disk."""
        try:
            actual_path = self._resolve_path(path)
        except ValueError:
            return None

        if not actual_path.exists() or actual_path.is_dir():
            return None

        return await asyncio.to_thread(actual_path.read_text, encoding="utf-8")

    async def _write_content(self, path: str, content: str) -> None:
        """Write content to disk atomically."""
        actual_path = self._resolve_path(path)

        def _do_write():
            actual_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=actual_path.parent,
                prefix=".tmp_",
                suffix=actual_path.suffix
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, actual_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        await asyncio.to_thread(_do_write)

    async def _delete_content(self, path: str) -> None:
        """Delete content from disk."""
        try:
            actual_path = self._resolve_path(path)
        except ValueError:
            return

        if actual_path.exists():
            if actual_path.is_dir():
                await asyncio.to_thread(shutil.rmtree, actual_path)
            else:
                await asyncio.to_thread(os.unlink, actual_path)

    async def _content_exists(self, path: str) -> bool:
        """Check if content exists on disk."""
        try:
            actual_path = self._resolve_path(path)
            return actual_path.exists()
        except ValueError:
            return False

    # =========================================================================
    # Override read() to add binary check and suggestions
    # =========================================================================

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file with binary check and similar file suggestions."""
        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        try:
            actual_path = self._resolve_path(path)
        except ValueError as e:
            return ReadResult(success=False, message=str(e))

        # Check if file exists on disk
        if not actual_path.exists():
            if actual_path.parent.exists():
                suggestions = get_similar_files(actual_path.parent, actual_path.name)
                if suggestions:
                    suggestion_text = "\n".join(f"  {s}" for s in suggestions)
                    return ReadResult(
                        success=False,
                        message=f"File not found: {path}\n\nDid you mean?\n{suggestion_text}"
                    )
            return ReadResult(success=False, message=f"File not found: {path}")

        # Check for binary file
        if is_binary_file(actual_path):
            return ReadResult(success=False, message=f"Cannot read binary file: {path}")

        # Use base class implementation
        return await super().read(path, offset, limit)

    # =========================================================================
    # Override mkdir to create on disk too
    # =========================================================================

    async def mkdir(self, path: str, parents: bool = True):
        """Create directory in database and on disk."""
        result = await super().mkdir(path, parents)

        if result.success:
            actual_path = self._resolve_path(path)
            await asyncio.to_thread(actual_path.mkdir, parents=True, exist_ok=True)

        return result

    # =========================================================================
    # Override list_dir to sync with disk
    # =========================================================================

    async def list_dir(self, path: str = "/") -> ListResult:
        """List directory, including files only on disk."""
        path = normalize_path(path)

        try:
            actual_path = self._resolve_path(path)
        except ValueError as e:
            return ListResult(success=False, message=str(e))

        if not actual_path.exists():
            return ListResult(success=False, message=f"Directory not found: {path}")

        if not actual_path.is_dir():
            return ListResult(success=False, message=f"Not a directory: {path}")

        # List from disk
        entries = []
        disk_items = await asyncio.to_thread(lambda: list(actual_path.iterdir()))

        session = await self._get_session()
        try:
            for item in disk_items:
                if item.name.startswith("."):
                    continue  # Skip hidden files

                item_path = f"{path}/{item.name}" if path != "/" else f"/{item.name}"
                item_path = normalize_path(item_path)

                # Get metadata from DB if available
                file = await self._get_file(session, item_path)

                entries.append(FileInfo(
                    path=item_path,
                    name=item.name,
                    is_directory=item.is_dir(),
                    size_bytes=file.size_bytes if file else (item.stat().st_size if item.is_file() else None),
                    mime_type=file.mime_type if file else None,
                    version=file.version if file else 1,
                    created_at=file.created_at if file else None,
                    updated_at=file.updated_at if file else None,
                ))

            return ListResult(
                success=True,
                message=f"Listed {len(entries)} items in {path}",
                entries=entries,
                path=path
            )
        finally:
            await self._commit(session)
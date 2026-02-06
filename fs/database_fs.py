"""
Database File System - everything stored in the database.

All file content and metadata is stored in the database.
Works with any SQLAlchemy-supported database via the provided session.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlmodel import select

from ..db_models import WorkspaceFile
from .base import BaseFileSystem
from .utils import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class DatabaseFileSystem(BaseFileSystem):
    """
    Database-backed file system using SQLAlchemy ORM.

    All content is stored in the database - portable and consistent
    across deployments. Works with Azure SQL, SQLite, etc.

    Uses a session factory to create per-operation sessions, avoiding
    concurrent session issues in async streaming contexts.

    USAGE:
        fs = DatabaseFileSystem(user_id="user123", session_factory=get_session_factory())
        await fs.write("/main.py", "print('hello')")
        result = await fs.read("/main.py")
    """

    def __init__(self, user_id: str, session_factory: Callable):
        """
        Initialize the database file system.

        Args:
            user_id: User ID for file isolation
            session_factory: Factory function that returns an async session context manager
        """
        super().__init__(user_id)
        self.session_factory = session_factory
        self._session: AsyncSession | None = None
        self._session_cm = None  # Context manager for cleanup

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    async def _get_session(self) -> "AsyncSession":
        """Get or create a database session for this operation."""
        if self._session is None:
            # Create session from factory and enter context
            self._session_cm = self.session_factory()
            self._session = await self._session_cm.__aenter__()
        return self._session

    async def _commit(self, session: "AsyncSession") -> None:
        """Commit changes to the database."""
        await session.commit()

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        # Clear the context manager reference (session already closed above)
        self._session_cm = None

    async def __aenter__(self) -> "DatabaseFileSystem":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager - commit and close session."""
        if self._session is not None:
            try:
                if exc_type is None:
                    # No exception - commit changes
                    await self._session.commit()
                else:
                    # Exception occurred - rollback
                    await self._session.rollback()
            finally:
                # Always close the session
                await self._session.close()
                self._session = None
        if self._session_cm is not None:
            self._session_cm = None

    async def _read_content(self, path: str) -> str | None:
        """Read content from database."""
        path = normalize_path(path)
        session = await self._get_session()
        result = await session.execute(
            select(WorkspaceFile.content).where(
                WorkspaceFile.user_id == self.user_id,
                WorkspaceFile.path == path,
                WorkspaceFile.deleted_at.is_(None),
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _write_content(self, path: str, content: str) -> None:
        """Write content to database (handled by base class via file record)."""
        # Content is stored in WorkspaceFile.content field
        # The base class write() handles updating the file record
        path = normalize_path(path)
        session = await self._get_session()
        result = await session.execute(
            select(WorkspaceFile).where(
                WorkspaceFile.user_id == self.user_id,
                WorkspaceFile.path == path,
            )
        )
        file = result.scalar_one_or_none()
        if file:
            file.content = content

    async def _delete_content(self, path: str) -> None:
        """Content is deleted when file record is deleted."""
        pass  # No separate content storage

    async def _content_exists(self, path: str) -> bool:
        """Check if content exists in database."""
        content = await self._read_content(path)
        return content is not None

    # =========================================================================
    # Conversation Folder Helper
    # =========================================================================

    async def ensure_conversation_folder(self, conversation_id: str) -> str:
        """Create /conversations/{id}/ folder if needed."""
        folder_path = f"/conversations/{conversation_id}"
        session = await self._get_session()
        file = await self._get_file(session, folder_path)
        if not file:
            await self.mkdir(folder_path, parents=True)
        return folder_path
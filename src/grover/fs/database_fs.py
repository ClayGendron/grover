"""DatabaseFileSystem — pure SQL storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import select

from grover.models.files import GroverFile

from .base import BaseFileSystem
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession


class DatabaseFileSystem(BaseFileSystem):
    """Database-backed file system using SQLAlchemy ORM.

    All content is stored in the database — portable and consistent
    across deployments. Works with SQLite, PostgreSQL, MSSQL, etc.

    Uses a session factory to create per-operation sessions, avoiding
    concurrent session issues in async streaming contexts.
    """

    def __init__(
        self,
        user_id: str,
        session_factory: Callable[..., AsyncSession],
        dialect: str = "sqlite",
    ) -> None:
        super().__init__(user_id, dialect=dialect)
        self.session_factory = session_factory
        self._session: AsyncSession | None = None
        self._session_cm: object = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    async def _get_session(self) -> AsyncSession:
        if self._session is None:
            self._session_cm = self.session_factory()
            self._session = await self._session_cm.__aenter__()
        return self._session

    async def _commit(self, session: AsyncSession) -> None:
        await session.commit()

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._session_cm = None

    async def __aenter__(self) -> DatabaseFileSystem:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        if self._session is not None:
            try:
                if exc_type is None:
                    await self._session.commit()
                else:
                    await self._session.rollback()
            finally:
                await self._session.close()
                self._session = None
        self._session_cm = None

    async def _read_content(self, path: str) -> str | None:
        path = normalize_path(path)
        session = await self._get_session()
        result = await session.execute(
            select(GroverFile.content).where(
                GroverFile.user_id == self.user_id,
                GroverFile.path == path,
                GroverFile.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _write_content(self, path: str, content: str) -> None:
        path = normalize_path(path)
        session = await self._get_session()
        result = await session.execute(
            select(GroverFile).where(
                GroverFile.user_id == self.user_id,
                GroverFile.path == path,
            )
        )
        file = result.scalar_one_or_none()
        if file:
            file.content = content

    async def _delete_content(self, path: str) -> None:
        pass  # Content lives in the file record

    async def _content_exists(self, path: str) -> bool:
        content = await self._read_content(path)
        return content is not None

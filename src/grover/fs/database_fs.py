"""DatabaseFileSystem — pure SQL storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic

from sqlmodel import select

from .base import FV, BaseFileSystem, F
from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DatabaseFileSystem(BaseFileSystem[F, FV], Generic[F, FV]):
    """Database-backed file system using SQLAlchemy ORM.

    All content is stored in the database — portable and consistent
    across deployments. Works with SQLite, PostgreSQL, MSSQL, etc.

    Uses a session factory to create per-operation sessions, avoiding
    concurrent session issues in async streaming contexts.
    """

    def __init__(
        self,
        session_factory: Callable[..., AsyncSession],
        dialect: str = "sqlite",
        file_model: type[F] | None = None,
        file_version_model: type[FV] | None = None,
        schema: str | None = None,
    ) -> None:
        super().__init__(
            dialect=dialect,
            file_model=file_model,
            file_version_model=file_version_model,
            schema=schema,
        )
        self.session_factory = session_factory
        self._session: AsyncSession | None = None
        self._session_cm: AbstractAsyncContextManager[AsyncSession] | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    async def _get_session(self) -> AsyncSession:
        if self._session is None:
            self._session_cm = self.session_factory()
            self._session = await self._session_cm.__aenter__()
        return self._session

    async def _commit(self, session: AsyncSession) -> None:
        if self.in_transaction:
            await session.flush()
        else:
            await session.commit()

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("Session CM exit failed", exc_info=True)
            self._session_cm = None

    async def __aenter__(self) -> DatabaseFileSystem[F, FV]:
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
        model = self._file_model
        result = await session.execute(
            select(model.content).where(  # type: ignore[arg-type]
                model.path == path,  # type: ignore[arg-type]
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _write_content(self, path: str, content: str) -> None:
        path = normalize_path(path)
        session = await self._get_session()
        model = self._file_model
        result = await session.execute(
            select(model).where(
                model.path == path,  # type: ignore[arg-type]
            )
        )
        file = result.scalar_one_or_none()
        if file:
            file.content = content

    async def _delete_content(self, path: str) -> None:
        pass  # Content lives in the file record

    async def _content_exists(self, path: str) -> bool:
        path = normalize_path(path)
        session = await self._get_session()
        model = self._file_model
        result = await session.execute(
            select(model.id).where(  # type: ignore[arg-type]
                model.path == path,  # type: ignore[arg-type]
                model.content.isnot(None),  # type: ignore[union-attr]
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        return result.first() is not None

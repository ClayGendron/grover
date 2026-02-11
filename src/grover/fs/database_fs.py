"""DatabaseFileSystem — pure SQL storage, stateless."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic

from sqlmodel import select

from .base import FV, BaseFileSystem, F
from .utils import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DatabaseFileSystem(BaseFileSystem[F, FV], Generic[F, FV]):
    """Database-backed file system — stateless, sessions provided per-operation.

    All content is stored in the database — portable and consistent
    across deployments. Works with SQLite, PostgreSQL, MSSQL, etc.

    This class holds only configuration (dialect, models, schema).
    It has no session factory, no mutable state, and is safe for
    concurrent use from multiple requests.  Sessions are injected
    via the ``session`` kwarg on every public method.
    """

    def __init__(
        self,
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

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    async def _get_session(self) -> AsyncSession:
        raise RuntimeError(
            "DatabaseFileSystem requires session= on each operation"
        )

    async def _commit(self, session: AsyncSession) -> None:
        await session.flush()

    async def close(self) -> None:
        """No-op — DFS has no resources to release."""

    async def __aenter__(self) -> DatabaseFileSystem[F, FV]:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        pass

    async def _read_content(self, path: str, session: AsyncSession) -> str | None:
        path = normalize_path(path)
        model = self._file_model
        result = await session.execute(
            select(model.content).where(  # type: ignore[arg-type]
                model.path == path,  # type: ignore[arg-type]
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
        path = normalize_path(path)
        model = self._file_model
        result = await session.execute(
            select(model).where(
                model.path == path,  # type: ignore[arg-type]
            )
        )
        file = result.scalar_one_or_none()
        if file:
            file.content = content

    async def _delete_content(self, path: str, session: AsyncSession) -> None:
        pass  # Content lives in the file record

    async def _content_exists(self, path: str, session: AsyncSession) -> bool:
        path = normalize_path(path)
        model = self._file_model
        result = await session.execute(
            select(model.id).where(  # type: ignore[arg-type]
                model.path == path,  # type: ignore[arg-type]
                model.content.isnot(None),  # type: ignore[union-attr]
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        return result.first() is not None

"""EngineConfig / SessionConfig — mount configuration for database-backed filesystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from grover.models.database.chunk import FileChunkModel
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.database.version import FileVersionModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy import URL
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.connection import FileConnectionModelBase
    from grover.models.database.file import FileModelBase
    from grover.models.database.version import FileVersionModelBase


@dataclass(frozen=True)
class EngineConfig:
    """Grover creates and owns the engine. Disposed on unmount/close.

    Provide either ``url`` (simple) or ``engine_factory`` (advanced).
    Use :func:`create_async_engine_factory` to build a factory with
    custom engine kwargs (pool size, connect_args, etc.).

    Usage::

        # Simple
        EngineConfig(url="postgresql+asyncpg://user:pw@host/db")

        # Advanced
        factory = create_async_engine_factory(
            "mssql+aioodbc://...",
            connect_args={"TrustServerCertificate": "yes"},
        )
        EngineConfig(engine_factory=factory)
    """

    url: str | None = None
    engine_factory: Callable[[], AsyncEngine] | None = None
    create_tables: bool = True
    file_model: type[FileModelBase] = field(default=FileModel)
    file_version_model: type[FileVersionModelBase] = field(default=FileVersionModel)
    file_chunk_model: type[FileChunkModelBase] = field(default=FileChunkModel)
    file_connection_model: type[FileConnectionModelBase] = field(default=FileConnectionModel)

    def __post_init__(self) -> None:
        if self.url is None and self.engine_factory is None:
            raise ValueError("EngineConfig requires either url or engine_factory")
        if self.url is not None and self.engine_factory is not None:
            raise ValueError("Provide url or engine_factory, not both")

    def create_engine(self) -> AsyncEngine:
        """Create an AsyncEngine from the configured url or factory."""
        if self.engine_factory is not None:
            return self.engine_factory()
        from sqlalchemy.ext.asyncio import create_async_engine

        assert self.url is not None
        return create_async_engine(self.url)


def create_async_engine_factory(url: str | URL, **kw: Any) -> Callable[[], AsyncEngine]:
    """Deferred engine factory — same signature as ``create_async_engine``.

    Captures *url* and *kw*; returns a zero-arg callable that creates
    an ``AsyncEngine`` when called.  Used with :class:`EngineConfig` for
    advanced cases (custom pool config, connect_args, etc.).

    Usage::

        factory = create_async_engine_factory(
            "mssql+aioodbc://...",
            connect_args={"TrustServerCertificate": "yes"},
        )
        config = EngineConfig(engine_factory=factory)
    """

    def factory() -> AsyncEngine:
        from sqlalchemy.ext.asyncio import create_async_engine

        return create_async_engine(url, **kw)

    return factory


@dataclass(frozen=True)
class SessionConfig:
    """App owns the session factory and engine. Grover does NOT dispose.

    Use when your application already manages an engine and session factory
    (e.g. FastAPI dependency injection).

    Usage::

        SessionConfig(session_factory=app_session_factory)
        SessionConfig(session_factory=app_session_factory, dialect="postgresql")
    """

    session_factory: Callable[[], AsyncSession]
    dialect: str | None = None
    file_model: type[FileModelBase] = field(default=FileModel)
    file_version_model: type[FileVersionModelBase] = field(default=FileVersionModel)
    file_chunk_model: type[FileChunkModelBase] = field(default=FileChunkModel)
    file_connection_model: type[FileConnectionModelBase] = field(default=FileConnectionModel)

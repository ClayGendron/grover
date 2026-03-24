"""Shared fixtures for Grover tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import Session, SQLModel, create_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest.fixture
def engine() -> Engine:
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """SQLModel session bound to the in-memory engine, rolled back after each test."""
    with Session(engine) as s:
        s.begin()
        yield s
        s.rollback()


@pytest.fixture
async def async_engine() -> AsyncIterator[AsyncEngine]:
    """Async in-memory SQLite engine with all tables created."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def async_session(async_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Async SQLModel session, rolled back after each test."""
    factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session

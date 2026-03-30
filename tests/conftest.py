"""Shared test helpers for Grover v2 tests."""

from __future__ import annotations

import subprocess
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.base import GroverFileSystem
from grover.results import Candidate

# ------------------------------------------------------------------
# --postgres CLI flag
# ------------------------------------------------------------------

_PG_DB = "grover_test"
_PG_URL = f"postgresql+asyncpg://localhost/{_PG_DB}"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--postgres",
        action="store_true",
        default=False,
        help="Run database tests against a local PostgreSQL instance instead of SQLite.",
    )
    parser.addoption(
        "--scale",
        type=int,
        default=1_000,
        help="Row count for batch/pressure tests (default: 1000).",
    )


# ------------------------------------------------------------------
# Database fixtures (shared by test_database.py, test_write_pressure.py)
# ------------------------------------------------------------------


@pytest.fixture
async def engine(request: pytest.FixtureRequest):
    use_pg = request.config.getoption("--postgres")
    if use_pg:
        subprocess.run(["createdb", _PG_DB], check=False)
        eng = create_async_engine(_PG_URL)
    else:
        eng = create_async_engine("sqlite+aiosqlite://")

    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await eng.dispose()

    if use_pg:
        subprocess.run(["dropdb", _PG_DB], check=False)


@pytest.fixture
async def db(engine):
    return DatabaseFileSystem(engine=engine)


@pytest.fixture
def scale(request: pytest.FixtureRequest) -> int:
    """Row count for batch/pressure tests. Override with ``--scale 100000``."""
    return request.config.getoption("--scale")


class DummySession:
    """Minimal session stub that tracks commit/rollback calls."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


def dummy_session_factory():
    """Return an async-context-manager that yields a fresh DummySession."""
    @asynccontextmanager
    async def _factory():
        yield DummySession()

    return _factory


def tracking_session_factory():
    """Return (factory, sessions_list) where sessions_list collects every session created."""
    sessions: list[DummySession] = []

    @asynccontextmanager
    async def _factory():
        s = DummySession()
        sessions.append(s)
        yield s

    return _factory, sessions


def candidate(path: str, *, content: str | None = None) -> Candidate:
    """Create a minimal Candidate for test assertions."""
    return Candidate(path=path, content=content)


def make_fs(name: str = "test") -> GroverFileSystem:
    """Create a GroverFileSystem with a dummy session factory for routing tests."""
    from unittest.mock import AsyncMock

    fs = GroverFileSystem(session_factory=AsyncMock())
    fs._name = name
    return fs

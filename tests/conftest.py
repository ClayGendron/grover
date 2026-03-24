"""Shared test helpers for Grover v2 tests."""

from __future__ import annotations

from contextlib import asynccontextmanager

from grover.base import GroverFileSystem
from grover.results import Candidate, GroverResult


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

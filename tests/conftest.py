"""Shared fixtures for Grover tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import Session, SQLModel, create_engine

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine


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

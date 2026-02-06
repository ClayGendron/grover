"""Shared fixtures for Grover tests."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def engine() -> Engine:
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Session:
    """SQLModel session bound to the in-memory engine."""
    with Session(engine) as s:
        yield s

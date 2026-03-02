"""FileConnection model — directed edges in the knowledge graph.

Provides ``FileConnectionBase`` (non-table base) and ``FileConnection``
(concrete table).  Subclass ``FileConnectionBase`` with ``table=True`` and
a custom ``__tablename__`` to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


class FileConnectionBase(SQLModel):
    """Base fields for a graph edge. Subclass with ``table=True`` for a concrete table.

    The ``path`` field is the canonical edge identity in ``source[type]target``
    format — unique and indexed. ``source_path`` and ``target_path`` are
    persisted separately for efficient queries.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(default="", unique=True, index=True)
    source_path: str = Field(index=True)
    target_path: str = Field(index=True)
    type: str = Field(default="")
    weight: float = Field(default=1.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )


class FileConnection(FileConnectionBase, table=True):
    """Default graph edge table — ``grover_file_connections``."""

    __tablename__ = "grover_file_connections"

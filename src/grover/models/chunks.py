"""FileChunk model — DB-backed chunk storage.

Provides ``FileChunkBase`` (non-table base) and ``FileChunk`` (concrete table).
Subclass ``FileChunkBase`` with ``table=True`` and a custom ``__tablename__``
to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel

from grover.models.vector import Vector, VectorType


class FileChunkBase(SQLModel):
    """Base fields for a file chunk. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(default="", index=True)
    file_path: str = Field(index=True)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    line_start: int = Field(default=0)
    line_end: int = Field(default=0)
    vector: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )


class FileChunk(FileChunkBase, table=True):
    """Default file chunk table — ``grover_file_chunks``."""

    __tablename__ = "grover_file_chunks"

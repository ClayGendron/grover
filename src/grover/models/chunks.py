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


class FileChunkBase(SQLModel):
    """Base fields for a file chunk. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    file_path: str = Field(index=True)
    chunk_path: str = Field(index=True)
    name: str = Field(default="")
    description: str = Field(default="")
    line_start: int = Field(default=0)
    line_end: int = Field(default=0)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    user_id: str | None = Field(default=None, index=True)
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

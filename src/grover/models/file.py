"""File model

Provides ``FileBase`` non-table base class.
Subclass with ``table=True`` and a custom ``__tablename__`` to use a
different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel

from grover.models.vector import Vector, VectorType


class FileBase(SQLModel):
    """Base fields for a tracked file. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(index=True, unique=True)
    parent_path: str = Field(default="", index=True)
    is_directory: bool = Field(default=False)
    content: str | None = Field(default=None)
    content_hash: str | None = Field(default=None)
    mime_type: str = Field(default="text/plain")
    lines: int = Field(default=0)
    size_bytes: int = Field(default=0)
    current_version: int = Field(default=1)
    original_path: str | None = Field(default=None)
    owner_id: str | None = Field(default=None, index=True)
    vector: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )


class File(FileBase, table=True):
    """Default file table — ``grover_files``."""

    __tablename__ = "grover_files"

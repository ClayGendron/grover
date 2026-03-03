"""FileVersion model

Provides ``FileBaseVersion`` non-table base class.
Subclass with ``table=True`` and a custom ``__tablename__`` to use a
different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, UniqueConstraint
from sqlmodel import Field, SQLModel

from grover.models.vector import Vector, VectorType


class FileVersionBase(SQLModel):
    """Base fields for a file version record. Subclass with ``table=True`` for a concrete table."""

    __table_args__ = (UniqueConstraint("file_id", "version"),)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    file_id: str = Field(index=True)
    file_path: str = Field(default="", index=True)
    version: int = Field(default=1)
    is_snapshot: bool = Field(default=False)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    size_bytes: int = Field(default=0)
    created_by: str | None = Field(default=None)
    vector: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )

    @property
    def path(self) -> str:
        """Canonical version path: ``file_path@version``."""
        return f"{self.file_path}@{self.version}"


class FileVersion(FileVersionBase, table=True):
    """Default file version table — ``grover_file_versions``."""

    __tablename__ = "grover_file_versions"

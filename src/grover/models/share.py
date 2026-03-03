"""FileShare model — tracks path-based shares between users.

Provides ``FileShareBase`` (non-table) and ``FileShare`` (concrete table).
Subclass ``FileShareBase`` with ``table=True`` and a custom ``__tablename__``
to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


class FileShareBase(SQLModel):
    """Base fields for a file share record. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(index=True)
    grantee_id: str = Field(index=True)
    permission: str = Field(default="read")
    granted_by: str = Field(default="")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    expires_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )


class FileShare(FileShareBase, table=True):
    """Default file share table — ``grover_file_shares``."""

    __tablename__ = "grover_file_shares"

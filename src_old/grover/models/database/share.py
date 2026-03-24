"""FileShareModel — tracks path-based shares between users.

Provides ``FileShareModelBase`` (non-table) and ``FileShareModel`` (concrete table).
Subclass ``FileShareModelBase`` with ``table=True`` and a custom ``__tablename__``
to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import model_validator
from sqlalchemy import DateTime
from sqlmodel import Field

from grover.util.paths import normalize_path

from .base import ValidatedSQLModel


class FileShareModelBase(ValidatedSQLModel):
    """Base fields for a file share record. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=1024, primary_key=True)
    path: str = Field(max_length=1024, index=True)
    grantee_id: str = Field(max_length=1024, index=True)
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_paths(cls, data: dict[str, object]) -> dict[str, object]:
        path = data.get("path")
        if isinstance(path, str):
            data["path"] = normalize_path(path)
        return data


class FileShareModel(FileShareModelBase, table=True):
    """Default file share table — ``grover_file_shares``."""

    __tablename__ = "grover_file_shares"

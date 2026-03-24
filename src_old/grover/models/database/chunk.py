"""FileChunkModel — DB-backed chunk storage.

Provides ``FileChunkModelBase`` (non-table base) and ``FileChunkModel`` (concrete table).
Subclass ``FileChunkModelBase`` with ``table=True`` and a custom ``__tablename__``
to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import model_validator
from sqlalchemy import DateTime
from sqlmodel import Field

from grover.util.content import compute_content_hash
from grover.util.paths import normalize_path

from .base import ValidatedSQLModel
from .vector import Vector, VectorType


class FileChunkModelBase(ValidatedSQLModel):
    """Base fields for a file chunk. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=1024, primary_key=True)
    path: str = Field(default="", max_length=1024, unique=True, index=True)
    file_path: str = Field(max_length=1024, index=True)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    line_start: int = Field(default=0)
    line_end: int = Field(default=0)
    tokens: int = Field(default=0)
    embedding: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_paths(cls, data: dict[str, object]) -> dict[str, object]:
        fp = data.get("file_path")
        if isinstance(fp, str):
            data["file_path"] = normalize_path(fp)
        path = data.get("path")
        if isinstance(path, str) and "#" in path:
            base, sep, symbol = path.rpartition("#")
            data["path"] = f"{normalize_path(base)}{sep}{symbol}"
        return data

    @classmethod
    def create(
        cls,
        file_path: str,
        name: str | int,
        content: str = "",
        *,
        mount: str | None = None,
        line_start: int = 0,
        line_end: int = 0,
        embedding: list[float] | None = None,
        tokens: int = 0,
    ) -> FileChunkModelBase:
        """Factory for building a fully-populated chunk model.

        ``name`` is the symbol identifier that appears after ``#`` in the
        chunk path (e.g. ``"login"`` → ``/src/auth.py#login``).
        """
        name = str(name)
        if mount:
            mount = mount.strip("/")
            file_path = f"/{mount}/{file_path.lstrip('/')}"
        content_hash, _ = compute_content_hash(content)
        now = datetime.now(UTC)
        return cls(
            file_path=file_path,
            path=f"{file_path}#{name}",
            content=content,
            content_hash=content_hash,
            line_start=line_start,
            line_end=line_end,
            tokens=tokens,
            embedding=Vector(embedding) if embedding is not None else None,
            created_at=now,
            updated_at=now,
        )


class FileChunkModel(FileChunkModelBase, table=True):
    """Default file chunk table — ``grover_file_chunks``."""

    __tablename__ = "grover_file_chunks"

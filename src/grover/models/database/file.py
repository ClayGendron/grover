"""FileModel — DB model for tracked files.

Provides ``FileModelBase`` non-table base class.
Subclass with ``table=True`` and a custom ``__tablename__`` to use a
different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import model_validator
from sqlalchemy import DateTime
from sqlmodel import Field

from grover.util.content import compute_content_hash, guess_mime_type, is_text_file
from grover.util.paths import normalize_path, split_path

from .base import ValidatedSQLModel
from .vector import Vector, VectorType


class FileModelBase(ValidatedSQLModel):
    """Base fields for a tracked file. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=1024, primary_key=True)
    path: str = Field(max_length=1024, index=True, unique=True)
    parent_path: str = Field(default="", max_length=1024, index=True)
    is_directory: bool = Field(default=False)
    content: str | None = Field(default=None)
    content_hash: str | None = Field(default=None)
    mime_type: str = Field(default="text/plain")
    lines: int = Field(default=0)
    size_bytes: int = Field(default=0)
    tokens: int = Field(default=0)
    current_version: int = Field(default=1)
    original_path: str | None = Field(default=None)
    owner_id: str | None = Field(default=None, max_length=1024, index=True)
    embedding: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate(cls, data: dict[str, object]) -> dict[str, object]:
        raw_path = data.get("path")
        if not isinstance(raw_path, str):
            return data

        norm = normalize_path(raw_path)
        data["path"] = norm
        parent, name = split_path(norm)
        data["parent_path"] = parent

        if not data.get("is_directory", False):
            if name and not is_text_file(name):
                raise ValueError(f"Cannot create non-text file: {name}")
            mime = data.get("mime_type")
            if name and (not mime or mime == "text/plain"):
                data["mime_type"] = guess_mime_type(name)

        content = data.get("content")
        if isinstance(content, str):
            content_hash, size_bytes = compute_content_hash(content)
            data["content_hash"] = content_hash
            data["size_bytes"] = size_bytes
            data["lines"] = content.count("\n")

        if not data.get("created_at"):
            data["created_at"] = datetime.now(UTC)
        if not data.get("updated_at"):
            data["updated_at"] = datetime.now(UTC)
        return data

    @classmethod
    def create(
        cls,
        path: str,
        content: str = "",
        *,
        mount: str | None = None,
        embedding: list[float] | None = None,
        tokens: int = 0,
        owner_id: str | None = None,
    ) -> FileModelBase:
        """Factory for building a fully-populated file model.

        Computes content_hash, size_bytes, mime_type, lines, and timestamps
        so the caller doesn't have to.
        """
        if mount:
            mount = mount.strip("/")
            path = f"/{mount}/{path.lstrip('/')}"
        content_hash, size_bytes = compute_content_hash(content)
        _, name = split_path(path)
        now = datetime.now(UTC)
        return cls(
            path=path,
            parent_path=split_path(path)[0],
            is_directory=False,
            content=content,
            content_hash=content_hash,
            mime_type=guess_mime_type(name),
            lines=content.count("\n"),
            size_bytes=size_bytes,
            tokens=tokens,
            embedding=Vector(embedding) if embedding is not None else None,
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
        )


class FileModel(FileModelBase, table=True):
    """Default file table — ``grover_files``."""

    __tablename__ = "grover_files"

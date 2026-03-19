"""FileVersionModel — version records for tracked files.

Provides ``FileVersionModelBase`` non-table base class.
Subclass with ``table=True`` and a custom ``__tablename__`` to use a
different table name per backend.
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


class FileVersionModelBase(ValidatedSQLModel):
    """Base fields for a file version record. Subclass with ``table=True`` for a concrete table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=1024, primary_key=True)
    path: str = Field(default="", max_length=1024, unique=True, index=True)
    file_path: str = Field(default="", max_length=1024, index=True)
    version: int = Field(default=1)
    is_snapshot: bool = Field(default=False)
    content: str = Field(default="")
    content_hash: str = Field(default="")
    size_bytes: int = Field(default=0)
    created_by: str | None = Field(default=None)
    embedding: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[invalid-argument-type]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),  # type: ignore[invalid-argument-type]
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_paths(cls, data: dict[str, object]) -> dict[str, object]:
        fp = data.get("file_path")
        if isinstance(fp, str):
            fp = normalize_path(fp)
            data["file_path"] = fp
            # Derive path from file_path@version if not explicitly set
            ver = data.get("version")
            if not data.get("path") and ver is not None:
                data["path"] = f"{fp}@{ver}"
        return data

    @classmethod
    def create(
        cls,
        file_path: str,
        version: int,
        content: str = "",
        *,
        is_snapshot: bool = False,
        created_by: str | None = None,
        embedding: list[float] | None = None,
    ) -> FileVersionModelBase:
        """Factory for building a fully-populated version model.

        Computes content_hash and size_bytes from content.
        Path is derived as ``file_path@version``.
        """
        content_hash, size_bytes = compute_content_hash(content)
        return cls(
            file_path=file_path,
            version=version,
            is_snapshot=is_snapshot,
            content=content,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_by=created_by,
            embedding=Vector(embedding) if embedding is not None else None,
        )


class FileVersionModel(FileVersionModelBase, table=True):
    """Default file version table — ``grover_file_versions``."""

    __tablename__ = "grover_file_versions"

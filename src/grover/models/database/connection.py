"""FileConnectionModel — directed edges in the knowledge graph.

Provides ``FileConnectionModelBase`` (non-table base) and ``FileConnectionModel``
(concrete table).  Subclass ``FileConnectionModelBase`` with ``table=True`` and
a custom ``__tablename__`` to use a different table name per backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import model_validator
from sqlalchemy import DateTime
from sqlmodel import Field

from grover.util.paths import normalize_path

from .base import ValidatedSQLModel


class FileConnectionModelBase(ValidatedSQLModel):
    """Base fields for a graph edge. Subclass with ``table=True`` for a concrete table.

    The ``path`` field is the canonical edge identity in ``source[type]target``
    format — unique and indexed. ``source_path`` and ``target_path`` are
    persisted separately for efficient queries.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=1024, primary_key=True)
    path: str = Field(default="", max_length=1024, unique=True, index=True)
    source_path: str = Field(max_length=1024, index=True)
    target_path: str = Field(max_length=1024, index=True)
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_paths(cls, data: dict[str, object]) -> dict[str, object]:
        src = data.get("source_path")
        if isinstance(src, str):
            data["source_path"] = normalize_path(src)
        tgt = data.get("target_path")
        if isinstance(tgt, str):
            data["target_path"] = normalize_path(tgt)
        if not data.get("path") and data.get("source_path") and data.get("target_path"):
            data["path"] = f"{data['source_path']}[{data.get('type', '')}]{data['target_path']}"
        return data

    @classmethod
    def create(
        cls,
        source_path: str,
        target_path: str,
        connection_type: str = "",
        *,
        weight: float = 1.0,
    ) -> FileConnectionModelBase:
        """Factory for building a connection model.

        Path is derived as ``source_path[connection_type]target_path``.
        """
        now = datetime.now(UTC)
        return cls(
            source_path=source_path,
            target_path=target_path,
            type=connection_type,
            weight=weight,
            created_at=now,
            updated_at=now,
        )


class FileConnectionModel(FileConnectionModelBase, table=True):
    """Default graph edge table — ``grover_file_connections``."""

    __tablename__ = "grover_file_connections"

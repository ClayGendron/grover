"""Embedding metadata model for change detection."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class Embedding(SQLModel, table=True):
    """Tracks which file versions have been embedded.

    This table records metadata for change detection — the actual vectors
    are stored externally (e.g. usearch index).
    """

    __tablename__ = "grover_embeddings"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    file_id: str = Field(index=True)
    file_version: int = Field(default=1)
    content_hash: str = Field(default="")
    model_name: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

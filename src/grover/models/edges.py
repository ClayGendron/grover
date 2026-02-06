"""GroverEdge model — single table for all graph edges."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class GroverEdge(SQLModel, table=True):
    """A directed edge in the knowledge graph.

    Edge types are free-form strings, e.g. ``"imports"``, ``"contains"``,
    ``"references"``, ``"inherits"``, ``"depends_on"``.
    """

    __tablename__ = "grover_edges"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    source_path: str = Field(index=True)
    target_path: str = Field(index=True)
    type: str = Field(default="")
    weight: float = Field(default=1.0)
    metadata_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

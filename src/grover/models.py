"""GroverObject — unified kinded object model for the ``grover_objects`` table.

All entities in the Grover namespace (files, directories, chunks, versions,
connections, api nodes) live in a single table.  The ``kind`` column determines
which nullable fields are relevant and how operations dispatch.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from pydantic import model_validator
from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel
from sqlmodel._compat import finish_init

from grover.paths import (
    decompose_connection,
    normalize_path,
    parse_kind,
    split_path,
    validate_path,
)
from grover.paths import (
    parent_path as compute_parent_path,
)
from grover.vector import Vector, VectorType

# ---------------------------------------------------------------------------
# Base class — adds Pydantic validation back to SQLModel table models
# ---------------------------------------------------------------------------


class ValidatedSQLModel(SQLModel):
    """SQLModel base that runs Pydantic validation on explicit ``__init__``.

    SQLModel ``table=True`` models normally skip validation.  This override
    restores it while preserving no-validation for ORM loads and
    ``model_validate()`` calls.
    """

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if not self.__class__.model_config.get("table", False):
            return
        if not finish_init.get():
            return
        sa_state = self.__dict__.get("_sa_instance_state")
        field_values = {}
        for field_name in self.__class__.model_fields:
            if hasattr(self, field_name):
                field_values[field_name] = getattr(self, field_name)
        self.__pydantic_validator__.validate_python(field_values, self_instance=self)
        if sa_state is not None:
            self.__dict__["_sa_instance_state"] = sa_state


# ---------------------------------------------------------------------------
# The unified object model
# ---------------------------------------------------------------------------


class GroverObjectBase(ValidatedSQLModel):
    """Base fields for a Grover namespace entity.

    Every entity — file, directory, chunk, version, connection, api node —
    shares these fields.  The ``kind`` column determines which nullable
    fields are relevant and how operations dispatch.

    Subclass with ``table=True`` and a ``__tablename__`` to create a
    concrete table model.
    """

    # --- Identity -----------------------------------------------------------

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        max_length=36,
        primary_key=True,
    )
    path: str = Field(max_length=4096, unique=True, index=True)
    name: str = Field(default="", max_length=255)
    parent_path: str = Field(default="", max_length=4096, index=True)
    kind: str = Field(default="", max_length=32, index=True)

    # --- Content ------------------------------------------------------------

    content: str | None = Field(default=None)
    content_hash: str | None = Field(default=None, max_length=64)
    mime_type: str | None = Field(default=None, max_length=255)

    # --- Metrics ------------------------------------------------------------

    lines: int = Field(default=0)
    size_bytes: int = Field(default=0)
    tokens: int = Field(default=0)

    # --- Chunk-specific -----------------------------------------------------

    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)

    # --- Version-specific ---------------------------------------------------

    is_snapshot: bool | None = Field(default=None)
    created_by: str | None = Field(default=None, max_length=255)

    # --- Connection-specific ------------------------------------------------

    source_path: str | None = Field(default=None, max_length=4096, index=True)
    target_path: str | None = Field(default=None, max_length=4096, index=True)
    connection_type: str | None = Field(default=None, max_length=255)
    connection_weight: float | None = Field(default=None)
    connection_distance: float | None = Field(default=None)

    # --- Embedding ----------------------------------------------------------

    embedding: Vector | None = Field(default=None, sa_type=VectorType())  # type: ignore[call-overload]

    # --- Ownership ----------------------------------------------------------

    owner_id: str | None = Field(default=None, max_length=255, index=True)
    original_path: str | None = Field(default=None, max_length=4096)

    # --- Timestamps ---------------------------------------------------------

    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[call-overload]
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[call-overload]
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[call-overload]
    )

    # --- Validator ----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_derive(cls, data: dict[str, object]) -> dict[str, object]:
        """Normalize path, derive parent_path, infer kind, compute metrics."""
        raw_path = data.get("path")
        if not isinstance(raw_path, str):
            return data

        # Validate and normalize path
        valid, err = validate_path(raw_path)
        if not valid:
            msg = f"Invalid path {raw_path!r}: {err}"
            raise ValueError(msg)

        path = normalize_path(raw_path)
        data["path"] = path

        # Derive name and parent_path from path
        if not data.get("name"):
            data["name"] = split_path(path)[1]
        if not data.get("parent_path"):
            data["parent_path"] = compute_parent_path(path)

        # Infer kind from path markers if not explicitly set
        if not data.get("kind"):
            data["kind"] = parse_kind(path)

        # For connections, extract source/target/type from path
        if data.get("kind") == "connection":
            parts = decompose_connection(path)
            if parts:
                if not data.get("source_path"):
                    data["source_path"] = parts.source
                if not data.get("connection_type"):
                    data["connection_type"] = parts.connection_type
                if not data.get("target_path"):
                    data["target_path"] = parts.target

        # Compute content metrics (empty string is valid content, distinct from None)
        content = data.get("content")
        if isinstance(content, str):
            encoded = content.encode()
            data["content_hash"] = hashlib.sha256(encoded).hexdigest()
            data["size_bytes"] = len(encoded)
            data["lines"] = content.count("\n") + 1 if content else 0

        # Ensure timestamps
        now = datetime.now(UTC)
        if not data.get("created_at"):
            data["created_at"] = now
        if not data.get("updated_at"):
            data["updated_at"] = now

        return data


class GroverObject(GroverObjectBase, table=True):
    """Default concrete table — ``grover_objects``."""

    __tablename__ = "grover_objects"

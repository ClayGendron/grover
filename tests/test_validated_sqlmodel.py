"""Tests for ValidatedSQLModel — validation on direct construction of table=True models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, model_validator
from sqlmodel import Field, Session, SQLModel, create_engine, select

from grover.models.database.base import ValidatedSQLModel
from grover.models.database.file import FileModel

# ---------------------------------------------------------------------------
# Test model definitions
# ---------------------------------------------------------------------------


class StrictBase(ValidatedSQLModel):
    """Non-table base with a validator that rejects negative values."""

    id: int = Field(default=1, primary_key=True)
    value: int = Field(default=0)
    computed: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, data: dict[str, object]) -> dict[str, object]:
        val = data.get("value")
        if isinstance(val, int) and val < 0:
            raise ValueError("value must be non-negative")
        if isinstance(val, int):
            data["computed"] = f"v{val}"
        return data


class StrictModel(StrictBase, table=True):
    __tablename__ = "test_strict"


# ---------------------------------------------------------------------------
# Validation on construction
# ---------------------------------------------------------------------------


class TestValidationOnConstruction:
    def test_valid_construction(self):
        m = StrictModel(value=42)
        assert m.value == 42
        assert m.computed == "v42"

    def test_invalid_construction_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            StrictModel(value=-1)

    def test_non_table_base_validates_normally(self):
        """Non-table models already validate via Pydantic's normal path."""
        m = StrictBase(value=10)
        assert m.computed == "v10"

        with pytest.raises(ValidationError, match="non-negative"):
            StrictBase(value=-1)

    def test_model_validate_still_works(self):
        m = StrictModel.model_validate({"value": 5})
        assert m.computed == "v5"

        with pytest.raises(ValidationError, match="non-negative"):
            StrictModel.model_validate({"value": -1})


# ---------------------------------------------------------------------------
# ORM round-trip — validates on insert, no validation on load
# ---------------------------------------------------------------------------


class TestOrmRoundTrip:
    @pytest.fixture
    def engine(self):
        eng = create_engine("sqlite://")
        SQLModel.metadata.create_all(eng)
        return eng

    def test_orm_round_trip(self, engine):
        """Insert a validated model and load it back via ORM."""
        with Session(engine) as session:
            m = StrictModel(value=7)
            assert m.computed == "v7"
            session.add(m)
            session.commit()

        with Session(engine) as session:
            loaded = session.exec(select(StrictModel)).first()
            assert loaded is not None
            assert loaded.value == 7
            assert loaded.computed == "v7"

    def test_sa_instance_state_preserved(self):
        """Construction preserves _sa_instance_state for ORM operations."""
        m = StrictModel(value=1)
        assert "_sa_instance_state" in m.__dict__


# ---------------------------------------------------------------------------
# FileModel-specific validation
# ---------------------------------------------------------------------------


class TestFileModelValidation:
    def test_binary_extension_rejected_at_construction(self):
        with pytest.raises(ValidationError, match="non-text"):
            FileModel(path="/project/image.png", content="data")

    def test_text_file_computes_fields(self):
        f = FileModel(path="/project/a.py", content="hello\n")
        assert f.content_hash is not None
        assert f.created_at is not None
        assert f.updated_at is not None
        assert f.mime_type == "text/x-python"
        assert f.size_bytes == 6
        assert f.lines == 1
        assert f.parent_path == "/project"

    def test_content_hash_recomputed(self):
        """Caller-provided content_hash is overwritten by the validator."""
        f = FileModel(path="/a.py", content="x\n", content_hash="fake")
        assert f.content_hash != "fake"

    def test_directory_skips_text_check(self):
        d = FileModel(path="/project", is_directory=True)
        assert d.is_directory is True

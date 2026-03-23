"""Tests for grover.vector — Vector type and VectorType SQLAlchemy decorator."""

from __future__ import annotations

import json

import pytest

from grover.vector import Vector, VectorType


# =========================================================================
# Vector — construction and subscript forms
# =========================================================================


class TestVectorConstruction:
    def test_basic(self):
        v = Vector([1.0, 2.0, 3.0])
        assert list(v) == [1.0, 2.0, 3.0]

    def test_empty(self):
        v = Vector()
        assert list(v) == []

    def test_none_gives_empty(self):
        v = Vector(None)
        assert list(v) == []

    def test_is_list_subclass(self):
        v = Vector([1.0, 2.0])
        assert isinstance(v, list)
        assert len(v) == 2


class TestVectorSubscript:
    def test_dimension_only(self):
        V3 = Vector[3]
        v = V3([1.0, 2.0, 3.0])
        assert v.dimension == 3
        assert v.model_name is None

    def test_dimension_enforced(self):
        V3 = Vector[3]
        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            V3([1.0, 2.0])

    def test_model_name_only(self):
        V = Vector["text-embedding-3-large"]
        v = V([0.1, 0.2])
        assert v.dimension is None
        assert v.model_name == "text-embedding-3-large"

    def test_dimension_and_model(self):
        V = Vector[1024, "text-embedding-3-large"]
        v = V([0.1] * 1024)
        assert v.dimension == 1024
        assert v.model_name == "text-embedding-3-large"

    def test_dimension_and_model_enforced(self):
        V = Vector[3, "my-model"]
        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            V([1.0, 2.0])

    def test_bad_tuple_length(self):
        with pytest.raises(TypeError, match="tuple must be"):
            Vector[1, "a", "b"]  # type: ignore[misc]

    def test_bad_tuple_types(self):
        with pytest.raises(TypeError, match="tuple must be"):
            Vector["a", "b"]  # type: ignore[misc]

    def test_bad_param_type(self):
        with pytest.raises(TypeError, match="requires int, str, or"):
            Vector[3.14]  # type: ignore[misc]


class TestVectorProperties:
    def test_unconstrained_dimension(self):
        v = Vector([1.0, 2.0])
        assert v.dimension is None

    def test_unconstrained_model_name(self):
        v = Vector([1.0, 2.0])
        assert v.model_name is None


# =========================================================================
# Vector — Pydantic integration
# =========================================================================


class TestVectorPydantic:
    def test_validate_none(self):
        assert Vector._pydantic_validate(None) is None

    def test_validate_vector(self):
        v = Vector([1.0, 2.0])
        assert Vector._pydantic_validate(v) is v

    def test_validate_list(self):
        result = Vector._pydantic_validate([1.0, 2.0])
        assert isinstance(result, Vector)
        assert list(result) == [1.0, 2.0]

    def test_validate_bad_type(self):
        with pytest.raises(ValueError, match="Expected list or Vector"):
            Vector._pydantic_validate("not a vector")


# =========================================================================
# VectorType — SQLAlchemy TypeDecorator
# =========================================================================


class TestVectorTypeBind:
    def test_none(self):
        vt = VectorType()
        assert vt.process_bind_param(None, None) is None  # type: ignore[arg-type]

    def test_serializes_to_json(self):
        vt = VectorType()
        result = vt.process_bind_param([1.0, 2.0, 3.0], None)  # type: ignore[arg-type]
        assert result == "[1.0, 2.0, 3.0]"
        assert json.loads(result) == [1.0, 2.0, 3.0]

    def test_dimension_enforced(self):
        vt = VectorType(dimension=3)
        with pytest.raises(ValueError, match="expected 3 dims"):
            vt.process_bind_param([1.0, 2.0], None)  # type: ignore[arg-type]

    def test_dimension_passes(self):
        vt = VectorType(dimension=3)
        result = vt.process_bind_param([1.0, 2.0, 3.0], None)  # type: ignore[arg-type]
        assert result is not None

    def test_model_name_mismatch(self):
        vt = VectorType(model_name="expected-model")
        v = Vector["wrong-model"]([1.0, 2.0])
        with pytest.raises(ValueError, match="model name mismatch"):
            vt.process_bind_param(v, None)  # type: ignore[arg-type]

    def test_model_name_matches(self):
        vt = VectorType(model_name="my-model")
        v = Vector["my-model"]([1.0, 2.0])
        result = vt.process_bind_param(v, None)  # type: ignore[arg-type]
        assert result is not None

    def test_model_name_skipped_for_plain_vector(self):
        vt = VectorType(model_name="my-model")
        v = Vector([1.0, 2.0])  # no model name on vector
        result = vt.process_bind_param(v, None)  # type: ignore[arg-type]
        assert result is not None


class TestVectorTypeResult:
    def test_none(self):
        vt = VectorType()
        assert vt.process_result_value(None, None) is None  # type: ignore[arg-type]

    def test_deserializes_from_json(self):
        vt = VectorType()
        result = vt.process_result_value("[1.0, 2.0, 3.0]", None)  # type: ignore[arg-type]
        assert isinstance(result, Vector)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_dimension_enforced_on_read(self):
        vt = VectorType(dimension=3)
        with pytest.raises(ValueError, match="expected 3 dims"):
            vt.process_result_value("[1.0, 2.0]", None)  # type: ignore[arg-type]

    def test_dimension_preserved_on_read(self):
        vt = VectorType(dimension=3)
        result = vt.process_result_value("[1.0, 2.0, 3.0]", None)  # type: ignore[arg-type]
        assert result is not None
        assert result.dimension == 3

    def test_model_name_preserved_on_read(self):
        vt = VectorType(model_name="my-model")
        result = vt.process_result_value("[1.0, 2.0]", None)  # type: ignore[arg-type]
        assert result is not None
        assert result.model_name == "my-model"

    def test_dimension_and_model_preserved(self):
        vt = VectorType(dimension=2, model_name="my-model")
        result = vt.process_result_value("[1.0, 2.0]", None)  # type: ignore[arg-type]
        assert result is not None
        assert result.dimension == 2
        assert result.model_name == "my-model"


class TestVectorTypeRoundTrip:
    def test_unconstrained(self):
        vt = VectorType()
        data = [0.1, 0.2, 0.3]
        serialized = vt.process_bind_param(data, None)  # type: ignore[arg-type]
        deserialized = vt.process_result_value(serialized, None)  # type: ignore[arg-type]
        assert list(deserialized) == data  # type: ignore[arg-type]

    def test_with_dimension(self):
        vt = VectorType(dimension=3)
        v = Vector[3]([1.0, 2.0, 3.0])
        serialized = vt.process_bind_param(v, None)  # type: ignore[arg-type]
        result = vt.process_result_value(serialized, None)  # type: ignore[arg-type]
        assert list(result) == [1.0, 2.0, 3.0]  # type: ignore[arg-type]
        assert result.dimension == 3  # type: ignore[union-attr]

    def test_with_model_name(self):
        vt = VectorType(model_name="my-model")
        v = Vector["my-model"]([0.5, 0.6])
        serialized = vt.process_bind_param(v, None)  # type: ignore[arg-type]
        result = vt.process_result_value(serialized, None)  # type: ignore[arg-type]
        assert result.model_name == "my-model"  # type: ignore[union-attr]

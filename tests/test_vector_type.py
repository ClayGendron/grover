"""Tests for Vector type and VectorType SQLAlchemy decorator."""

from __future__ import annotations

import json

import pytest

from grover.models.database.vector import Vector, VectorType


class TestVectorClassGetitem:
    def test_creates_subclass(self):
        vec3 = Vector[3]
        assert vec3._dimension == 3
        assert vec3.__name__ == "Vector[3]"

    def test_different_dimensions(self):
        vec3 = Vector[3]
        vec1024 = Vector[1024]
        assert vec3._dimension == 3
        assert vec1024._dimension == 1024
        assert vec3 is not vec1024

    def test_base_has_no_dimension(self):
        assert Vector._dimension is None


class TestVectorRuntimeValidation:
    def test_correct_dimension_accepted(self):
        v = Vector[3]([1.0, 2.0, 3.0])
        assert len(v) == 3

    def test_wrong_dimension_rejected(self):
        with pytest.raises(ValueError, match="Expected 3 dimensions, got 2"):
            Vector[3]([1.0, 2.0])

    def test_empty_vector_wrong_dimension(self):
        with pytest.raises(ValueError, match="Expected 3 dimensions, got 0"):
            Vector[3]([])


class TestVectorNoDimension:
    def test_any_length_accepted(self):
        v = Vector([1.0, 2.0])
        assert len(v) == 2

    def test_empty_accepted(self):
        v = Vector([])
        assert len(v) == 0

    def test_none_becomes_empty(self):
        v = Vector()
        assert len(v) == 0


class TestVectorIsList:
    def test_isinstance(self):
        assert isinstance(Vector([1, 2, 3]), list)

    def test_subscripted_isinstance(self):
        assert isinstance(Vector[3]([1, 2, 3]), list)

    def test_indexing(self):
        v = Vector([1.0, 2.0, 3.0])
        assert v[0] == 1.0
        assert v[2] == 3.0

    def test_iteration(self):
        v = Vector([1.0, 2.0, 3.0])
        assert list(v) == [1.0, 2.0, 3.0]


# ==================================================================
# Pydantic validation (_pydantic_validate)
# ==================================================================


class TestVectorPydanticValidate:
    def test_none_returns_none(self):
        result = Vector._pydantic_validate(None)
        assert result is None

    def test_vector_passthrough(self):
        v = Vector([1.0, 2.0, 3.0])
        result = Vector._pydantic_validate(v)
        assert result is v

    def test_list_converted(self):
        result = Vector._pydantic_validate([1.0, 2.0])
        assert isinstance(result, Vector)
        assert list(result) == [1.0, 2.0]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Expected list or Vector, got"):
            Vector._pydantic_validate("not a vector")

    def test_invalid_type_int_raises(self):
        with pytest.raises(ValueError, match="Expected list or Vector, got"):
            Vector._pydantic_validate(42)


class TestVectorTypeJsonRoundtrip:
    def test_bind_and_result(self):
        vt = VectorType()
        data = [1.0, 2.0, 3.0]
        bound = vt.process_bind_param(data, None)
        assert bound is not None
        assert json.loads(bound) == data

        result = vt.process_result_value(bound, None)
        assert isinstance(result, Vector)
        assert list(result) == data

    def test_vector_instance_roundtrip(self):
        vt = VectorType()
        v = Vector([1.0, 2.0, 3.0])
        bound = vt.process_bind_param(v, None)
        result = vt.process_result_value(bound, None)
        assert list(result) == [1.0, 2.0, 3.0]


class TestVectorTypeNoneHandling:
    def test_bind_none(self):
        vt = VectorType()
        assert vt.process_bind_param(None, None) is None

    def test_result_none(self):
        vt = VectorType()
        assert vt.process_result_value(None, None) is None

    def test_dimensioned_bind_none(self):
        vt = VectorType(dimension=3)
        assert vt.process_bind_param(None, None) is None

    def test_dimensioned_result_none(self):
        vt = VectorType(dimension=3)
        assert vt.process_result_value(None, None) is None


class TestVectorTypeEnforcesOnBind:
    def test_wrong_dimension_rejected(self):
        vt = VectorType(dimension=3)
        with pytest.raises(ValueError, match="Vector bind: expected 3 dims, got 2"):
            vt.process_bind_param([1.0, 2.0], None)

    def test_correct_dimension_accepted(self):
        vt = VectorType(dimension=3)
        result = vt.process_bind_param([1.0, 2.0, 3.0], None)
        assert result is not None

    def test_no_dimension_accepts_any(self):
        vt = VectorType()
        result = vt.process_bind_param([1.0, 2.0], None)
        assert result is not None


class TestVectorTypeEnforcesOnRead:
    def test_wrong_dimension_rejected(self):
        vt = VectorType(dimension=3)
        bad_json = json.dumps([1.0, 2.0])
        with pytest.raises(ValueError, match="Vector read: expected 3 dims, got 2"):
            vt.process_result_value(bad_json, None)

    def test_correct_dimension_accepted(self):
        vt = VectorType(dimension=3)
        good_json = json.dumps([1.0, 2.0, 3.0])
        result = vt.process_result_value(good_json, None)
        assert list(result) == [1.0, 2.0, 3.0]


class TestVectorTypePreservesDimensionOnRead:
    def test_returns_dimensioned_subclass(self):
        vt = VectorType(dimension=3)
        good_json = json.dumps([1.0, 2.0, 3.0])
        result = vt.process_result_value(good_json, None)
        assert isinstance(result, Vector)
        assert result._dimension == 3

    def test_no_dimension_returns_base(self):
        vt = VectorType()
        good_json = json.dumps([1.0, 2.0, 3.0])
        result = vt.process_result_value(good_json, None)
        assert isinstance(result, Vector)
        assert result._dimension is None

    def test_cache_ok(self):
        vt = VectorType(dimension=3)
        assert vt.cache_ok is True


# ==================================================================
# Model name subscript forms
# ==================================================================


class TestVectorModelNameSubscript:
    def test_model_name_only(self):
        vec = Vector["text-embedding-3-large"]
        assert vec._model_name == "text-embedding-3-large"
        assert vec._dimension is None
        assert vec.__name__ == "Vector['text-embedding-3-large']"

    def test_dimension_and_model_name(self):
        vec = Vector[1536, "text-embedding-3-large"]
        assert vec._dimension == 1536
        assert vec._model_name == "text-embedding-3-large"
        assert vec.__name__ == "Vector[1536, 'text-embedding-3-large']"

    def test_model_name_only_accepts_any_length(self):
        v = Vector["some-model"]([1.0, 2.0])
        assert len(v) == 2
        v2 = Vector["some-model"]([1.0] * 100)
        assert len(v2) == 100

    def test_dimension_and_model_validates_dimension(self):
        with pytest.raises(ValueError, match="Expected 3 dimensions, got 2"):
            Vector[3, "model"]([1.0, 2.0])

    def test_dimension_and_model_accepts_correct(self):
        v = Vector[3, "model"]([1.0, 2.0, 3.0])
        assert len(v) == 3
        assert v._model_name == "model"

    def test_backward_compat_int_subscript(self):
        vec = Vector[1536]
        assert vec._dimension == 1536
        assert vec._model_name is None
        assert vec.__name__ == "Vector[1536]"

    def test_invalid_subscript_type(self):
        with pytest.raises(TypeError, match="requires int, str, or"):
            Vector[3.5]  # type: ignore[type-var]

    def test_tuple_wrong_length(self):
        with pytest.raises(TypeError, match="tuple must be"):
            Vector[1, "a", "b"]  # type: ignore[type-var]

    def test_tuple_wrong_types(self):
        with pytest.raises(TypeError, match="tuple must be"):
            Vector["model", 1536]  # type: ignore[type-var]

    def test_base_has_no_model_name(self):
        assert Vector._model_name is None


# ==================================================================
# Vector properties
# ==================================================================


class TestVectorProperties:
    def test_dimension_property(self):
        v = Vector[3]([1.0, 2.0, 3.0])
        assert v.dimension == 3

    def test_model_name_property(self):
        v = Vector["fake"]([1.0, 2.0])
        assert v.model_name == "fake"

    def test_base_properties_none(self):
        v = Vector([1.0, 2.0])
        assert v.dimension is None
        assert v.model_name is None

    def test_both_properties(self):
        v = Vector[3, "model"]([1.0, 2.0, 3.0])
        assert v.dimension == 3
        assert v.model_name == "model"


# ==================================================================
# Vector.for_provider()
# ==================================================================


class _FakeProvider:
    """Minimal provider for testing for_provider()."""

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "fake"

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 3

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]


class TestVectorForProvider:
    def test_creates_typed_subclass(self):
        provider = _FakeProvider()
        vec_type = Vector.for_provider(provider)
        assert vec_type._dimension == 3
        assert vec_type._model_name == "fake"
        assert vec_type.__name__ == "Vector[3, 'fake']"

    def test_validates_dimension(self):
        provider = _FakeProvider()
        vec_type = Vector.for_provider(provider)
        with pytest.raises(ValueError, match="Expected 3 dimensions, got 2"):
            vec_type([1.0, 2.0])

    def test_carries_model_name(self):
        provider = _FakeProvider()
        vec_type = Vector.for_provider(provider)
        v = vec_type([1.0, 2.0, 3.0])
        assert v.model_name == "fake"
        assert v.dimension == 3


# ==================================================================
# VectorType model_name
# ==================================================================


class TestVectorTypeModelName:
    def test_constructor_stores_model_name(self):
        vt = VectorType(model_name="text-embedding-3-large")
        assert vt.model_name == "text-embedding-3-large"

    def test_constructor_default_none(self):
        vt = VectorType()
        assert vt.model_name is None

    def test_bind_model_name_mismatch(self):
        vt = VectorType(model_name="model-a")
        v = Vector["model-b"]([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="model name mismatch"):
            vt.process_bind_param(v, None)

    def test_bind_model_name_match(self):
        vt = VectorType(model_name="model-a")
        v = Vector["model-a"]([1.0, 2.0, 3.0])
        result = vt.process_bind_param(v, None)
        assert result is not None

    def test_bind_plain_list_ignores_model_check(self):
        vt = VectorType(model_name="model-a")
        result = vt.process_bind_param([1.0, 2.0, 3.0], None)
        assert result is not None

    def test_bind_vector_without_model_ignores_check(self):
        vt = VectorType(model_name="model-a")
        v = Vector([1.0, 2.0, 3.0])  # no model_name
        result = vt.process_bind_param(v, None)
        assert result is not None

    def test_result_with_model_returns_typed(self):
        vt = VectorType(model_name="model-a")
        data_json = json.dumps([1.0, 2.0, 3.0])
        result = vt.process_result_value(data_json, None)
        assert isinstance(result, Vector)
        assert result._model_name == "model-a"
        assert result._dimension is None

    def test_result_with_dimension_and_model(self):
        vt = VectorType(dimension=3, model_name="model-a")
        data_json = json.dumps([1.0, 2.0, 3.0])
        result = vt.process_result_value(data_json, None)
        assert isinstance(result, Vector)
        assert result._dimension == 3
        assert result._model_name == "model-a"

    def test_result_none_with_model(self):
        vt = VectorType(model_name="model-a")
        assert vt.process_result_value(None, None) is None

    def test_bind_none_with_model(self):
        vt = VectorType(model_name="model-a")
        assert vt.process_bind_param(None, None) is None


# ==================================================================
# VectorType.from_provider()
# ==================================================================


class TestVectorTypeFromProvider:
    def test_creates_with_dimension_and_model(self):
        provider = _FakeProvider()
        vt = VectorType.from_provider(provider)
        assert vt.dimension == 3
        assert vt.model_name == "fake"

        # Round-trip: bind and read back
        v = Vector[3, "fake"]([1.0, 2.0, 3.0])
        bound = vt.process_bind_param(v, None)
        result = vt.process_result_value(bound, None)
        assert isinstance(result, Vector)
        assert result._dimension == 3
        assert result._model_name == "fake"
        assert list(result) == [1.0, 2.0, 3.0]

"""Vector type — fixed-dimension vector with SQLAlchemy storage.

``Vector[1024]`` creates a dimension-specific subclass that validates length
on construction. ``VectorType`` is a SQLAlchemy ``TypeDecorator`` that
serializes vectors as JSON text and enforces dimensions on both read and write.

Usage::

    from grover.vector import Vector, VectorType

    # As a model field (any dimension):
    vector: Vector | None = Field(default=None, sa_type=VectorType())

    # As a runtime validator:
    v = Vector[3]([1.0, 2.0, 3.0])

    # With dimension enforcement in the DB layer:
    vector: Vector | None = Field(default=None, sa_type=VectorType(dimension=1024))

    # With model name tracking:
    v = Vector[1536, "text-embedding-3-large"]([0.1] * 1536)
    v = Vector["text-embedding-3-large"]([0.1] * 1536)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from pydantic_core import core_schema
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema
    from sqlalchemy.engine.interfaces import Dialect


class Vector(list[float]):
    """Fixed-dimension vector with optional model name tracking.

    Subscript forms:

    - ``Vector[1024]`` — dimension only
    - ``Vector["text-embedding-3-large"]`` — model name only
    - ``Vector[1024, "text-embedding-3-large"]`` — both

    Unsubscripted ``Vector()`` accepts any length with no model tracking.
    """

    _dimension: int | None = None
    _model_name: str | None = None

    def __class_getitem__(cls, params: int | str | tuple[int, str]) -> type[Vector]:  # type: ignore[override]
        """Create a dimension/model-specific Vector subclass."""
        if isinstance(params, int):
            name, attrs = f"Vector[{params}]", {"_dimension": params, "_model_name": None}
        elif isinstance(params, str):
            name, attrs = f"Vector['{params}']", {"_dimension": None, "_model_name": params}
        elif isinstance(params, tuple):
            if len(params) != 2:
                msg = f"Vector[...] tuple must be (int, str), got {len(params)} elements"
                raise TypeError(msg)
            dim, model = params
            if not isinstance(dim, int) or not isinstance(model, str):
                msg = f"Vector[...] tuple must be (int, str), got ({type(dim).__name__}, {type(model).__name__})"
                raise TypeError(msg)
            name, attrs = f"Vector[{dim}, '{model}']", {"_dimension": dim, "_model_name": model}
        else:
            msg = f"Vector[...] requires int, str, or (int, str), got {type(params).__name__}"
            raise TypeError(msg)
        return cast("type[Vector]", type(name, (cls,), attrs))

    def __init__(self, data: list[float] | None = None) -> None:
        super().__init__(data or [])
        if self._dimension is not None and len(self) != self._dimension:
            msg = f"Expected {self._dimension} dimensions, got {len(self)}"
            raise ValueError(msg)

    @property
    def dimension(self) -> int | None:
        """Number of dimensions, or None if unconstrained."""
        return self._dimension

    @property
    def model_name(self) -> str | None:
        """Embedding model name, or None if unset."""
        return self._model_name

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: list(v) if v is not None else None,
                info_arg=False,
            ),
        )

    @classmethod
    def _pydantic_validate(cls, value: object) -> Vector | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Vector):
            return cls(list(value))
        if isinstance(value, list):
            return cls(value)  # type: ignore[arg-type]
        msg = f"Expected list or Vector, got {type(value)}"
        raise ValueError(msg)


class VectorType(TypeDecorator[Vector]):
    """SQLAlchemy type: stores Vector as JSON text.

    Enforces dimension on BOTH read and write when ``dimension`` is set.
    Validates model name on write when ``model_name`` is set.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dimension: int | None = None, model_name: str | None = None) -> None:
        super().__init__()
        self.dimension = dimension
        self.model_name = model_name

    def process_bind_param(self, value: list[float] | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if self.dimension is not None and len(value) != self.dimension:
            msg = f"Vector bind: expected {self.dimension} dims, got {len(value)}"
            raise ValueError(msg)
        if (
            self.model_name is not None
            and isinstance(value, Vector)
            and value._model_name is not None
            and value._model_name != self.model_name
        ):
            msg = f"Vector bind: model name mismatch — column expects '{self.model_name}', got '{value._model_name}'"
            raise ValueError(msg)
        return json.dumps(list(value))

    def process_result_value(self, value: str | None, dialect: Dialect) -> Vector | None:
        if value is None:
            return None
        data = json.loads(value)
        if self.dimension is not None and len(data) != self.dimension:
            msg = f"Vector read: expected {self.dimension} dims, got {len(data)}"
            raise ValueError(msg)
        if self.dimension is not None and self.model_name is not None:
            return Vector[self.dimension, self.model_name](data)
        if self.dimension is not None:
            return Vector[self.dimension](data)
        if self.model_name is not None:
            return Vector[self.model_name](data)
        return Vector(data)

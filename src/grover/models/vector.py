"""Vector type — fixed-dimension vector with SQLAlchemy storage.

``Vector[1024]`` creates a dimension-specific subclass that validates length
on construction. ``VectorType`` is a SQLAlchemy ``TypeDecorator`` that
serializes vectors as JSON text and enforces dimensions on both read and write.

Usage::

    from grover.models.vector import Vector, VectorType

    # As a model field (any dimension):
    vector: Vector | None = Field(default=None, sa_type=VectorType())

    # As a runtime validator:
    v = Vector[3]([1.0, 2.0, 3.0])

    # With dimension enforcement in the DB layer:
    vector: Vector | None = Field(default=None, sa_type=VectorType(dimension=1024))
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic_core import core_schema
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema
    from sqlalchemy.engine.interfaces import Dialect


class Vector(list[float]):
    """Fixed-dimension vector. Use ``Vector[1024]`` for 1024-dim enforcement.

    - As model field type: ``vector: Vector | None = Field(default=None, sa_type=VectorType())``
    - As runtime validator: ``v = Vector[1024]([0.1, 0.2, ...])``
    - Unsubscripted ``Vector()`` accepts any length.
    """

    _dimension: int | None = None

    def __class_getitem__(cls, dimension: int) -> type[Vector]:  # type: ignore[override]
        """Create a dimension-specific Vector subclass."""
        return type(f"Vector[{dimension}]", (cls,), {"_dimension": dimension})

    def __init__(self, data: list[float] | None = None) -> None:
        super().__init__(data or [])
        if self._dimension is not None and len(self) != self._dimension:
            msg = f"Expected {self._dimension} dimensions, got {len(self)}"
            raise ValueError(msg)

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
        if isinstance(value, Vector):
            return value
        if isinstance(value, list):
            return cls(value)  # type: ignore[arg-type]
        msg = f"Expected list or Vector, got {type(value)}"
        raise ValueError(msg)


class VectorType(TypeDecorator[Vector]):
    """SQLAlchemy type: stores Vector as JSON text.

    Enforces dimension on BOTH read and write when ``dimension`` is set.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dimension: int | None = None) -> None:
        super().__init__()
        self.dimension = dimension

    def process_bind_param(self, value: list[float] | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if self.dimension is not None and len(value) != self.dimension:
            msg = f"Vector bind: expected {self.dimension} dims, got {len(value)}"
            raise ValueError(msg)
        return json.dumps(list(value))

    def process_result_value(self, value: str | None, dialect: Dialect) -> Vector | None:
        if value is None:
            return None
        data = json.loads(value)
        if self.dimension is not None and len(data) != self.dimension:
            msg = f"Vector read: expected {self.dimension} dims, got {len(data)}"
            raise ValueError(msg)
        if self.dimension is not None:
            return Vector[self.dimension](data)
        return Vector(data)

"""Filter AST — provider-agnostic filter expressions with compilers for each store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class FilterOp(Enum):
    """Comparison operators for metadata filtering."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class LogicalOp(Enum):
    """Logical combinators for grouping filter expressions."""

    AND = "and"
    OR = "or"


# ------------------------------------------------------------------
# AST nodes
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    """A single field comparison (e.g. ``field == value``).

    Attributes:
        field: Metadata field name.
        op: Comparison operator.
        value: Value to compare against.  For ``EXISTS``, this is a bool.
    """

    field: str
    op: FilterOp
    value: Any


@dataclass(frozen=True, slots=True)
class LogicalGroup:
    """A logical combination of filter expressions.

    Attributes:
        op: Logical operator (AND / OR).
        expressions: Child expressions to combine.
    """

    op: LogicalOp
    expressions: list[FilterExpression]


FilterExpression = Comparison | LogicalGroup
"""Union type for the filter AST: either a leaf :class:`Comparison` or a
:class:`LogicalGroup` combining sub-expressions."""


# ------------------------------------------------------------------
# Builder helpers — concise filter construction
# ------------------------------------------------------------------


def eq(field: str, value: Any) -> Comparison:
    """``field == value``."""
    return Comparison(field=field, op=FilterOp.EQ, value=value)


def ne(field: str, value: Any) -> Comparison:
    """``field != value``."""
    return Comparison(field=field, op=FilterOp.NE, value=value)


def gt(field: str, value: Any) -> Comparison:
    """``field > value``."""
    return Comparison(field=field, op=FilterOp.GT, value=value)


def gte(field: str, value: Any) -> Comparison:
    """``field >= value``."""
    return Comparison(field=field, op=FilterOp.GTE, value=value)


def lt(field: str, value: Any) -> Comparison:
    """``field < value``."""
    return Comparison(field=field, op=FilterOp.LT, value=value)


def lte(field: str, value: Any) -> Comparison:
    """``field <= value``."""
    return Comparison(field=field, op=FilterOp.LTE, value=value)


def in_(field: str, values: list[Any]) -> Comparison:
    """``field IN values``."""
    return Comparison(field=field, op=FilterOp.IN, value=values)


def not_in(field: str, values: list[Any]) -> Comparison:
    """``field NOT IN values``."""
    return Comparison(field=field, op=FilterOp.NOT_IN, value=values)


def exists(field: str, *, exists: bool = True) -> Comparison:
    """``field EXISTS`` (or ``NOT EXISTS`` if ``exists=False``)."""
    return Comparison(field=field, op=FilterOp.EXISTS, value=exists)


def and_(*exprs: FilterExpression) -> LogicalGroup:
    """Combine expressions with AND."""
    return LogicalGroup(op=LogicalOp.AND, expressions=list(exprs))


def or_(*exprs: FilterExpression) -> LogicalGroup:
    """Combine expressions with OR."""
    return LogicalGroup(op=LogicalOp.OR, expressions=list(exprs))


# ------------------------------------------------------------------
# Compilers — convert the AST to provider-native formats
# ------------------------------------------------------------------

_PINECONE_OPS: dict[FilterOp, str] = {
    FilterOp.EQ: "$eq",
    FilterOp.NE: "$ne",
    FilterOp.GT: "$gt",
    FilterOp.GTE: "$gte",
    FilterOp.LT: "$lt",
    FilterOp.LTE: "$lte",
    FilterOp.IN: "$in",
    FilterOp.NOT_IN: "$nin",
    FilterOp.EXISTS: "$exists",
}


def compile_pinecone(expr: FilterExpression) -> dict[str, Any]:
    """Compile a ``FilterExpression`` to Pinecone's MongoDB-style filter dict.

    Examples::

        compile_pinecone(eq("genre", "comedy"))
        # {"genre": {"$eq": "comedy"}}

        compile_pinecone(and_(eq("genre", "comedy"), gt("year", 2000)))
        # {"$and": [{"genre": {"$eq": "comedy"}}, {"year": {"$gt": 2000}}]}
    """
    if isinstance(expr, Comparison):
        op_str = _PINECONE_OPS[expr.op]
        return {expr.field: {op_str: expr.value}}

    # LogicalGroup
    logical_key = "$and" if expr.op == LogicalOp.AND else "$or"
    return {logical_key: [compile_pinecone(child) for child in expr.expressions]}


_DATABRICKS_OPS: dict[FilterOp, str] = {
    FilterOp.EQ: "=",
    FilterOp.NE: "!=",
    FilterOp.GT: ">",
    FilterOp.GTE: ">=",
    FilterOp.LT: "<",
    FilterOp.LTE: "<=",
}


def _databricks_quote(value: Any) -> str:
    """Quote a value for use in a Databricks filter string."""
    if isinstance(value, str):
        # Escape single quotes within the string
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def compile_databricks(expr: FilterExpression) -> str:
    """Compile a ``FilterExpression`` to a Databricks SQL-like filter string.

    Examples::

        compile_databricks(eq("genre", "comedy"))
        # "genre = 'comedy'"

        compile_databricks(and_(eq("genre", "comedy"), gt("year", 2000)))
        # "(genre = 'comedy' AND year > 2000)"
    """
    if isinstance(expr, Comparison):
        if expr.op == FilterOp.IN:
            items = ", ".join(_databricks_quote(v) for v in expr.value)
            return f"{expr.field} IN ({items})"
        if expr.op == FilterOp.NOT_IN:
            items = ", ".join(_databricks_quote(v) for v in expr.value)
            return f"{expr.field} NOT IN ({items})"
        if expr.op == FilterOp.EXISTS:
            if expr.value:
                return f"{expr.field} IS NOT NULL"
            return f"{expr.field} IS NULL"

        op_str = _DATABRICKS_OPS[expr.op]
        return f"{expr.field} {op_str} {_databricks_quote(expr.value)}"

    # LogicalGroup
    joiner = " AND " if expr.op == LogicalOp.AND else " OR "
    parts = [compile_databricks(child) for child in expr.expressions]
    return f"({joiner.join(parts)})"


def compile_dict(expr: FilterExpression) -> dict[str, Any]:
    """Compile a ``FilterExpression`` to a simple dict for local stores.

    Only supports equality comparisons (``EQ``).  Raises ``ValueError`` for
    unsupported operators.  Logical ``AND`` merges field constraints; ``OR``
    is not supported.

    Examples::

        compile_dict(eq("genre", "comedy"))
        # {"genre": "comedy"}

        compile_dict(and_(eq("genre", "comedy"), eq("year", 2000)))
        # {"genre": "comedy", "year": 2000}
    """
    if isinstance(expr, Comparison):
        if expr.op != FilterOp.EQ:
            msg = f"compile_dict only supports EQ comparisons, got {expr.op.value}"
            raise ValueError(msg)
        return {expr.field: expr.value}

    # LogicalGroup
    if expr.op != LogicalOp.AND:
        msg = "compile_dict only supports AND logical groups"
        raise ValueError(msg)

    merged: dict[str, Any] = {}
    for child in expr.expressions:
        merged.update(compile_dict(child))
    return merged

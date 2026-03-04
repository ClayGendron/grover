"""Graph result types — immutable data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SubgraphResult:
    """A subgraph extracted from the main graph.

    Deeply immutable: ``nodes`` and ``edges`` are tuples,
    ``scores`` is a ``MappingProxyType``.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, dict[str, Any]], ...]
    scores: MappingProxyType[str, float] = field(default_factory=lambda: MappingProxyType({}))


def subgraph_result(
    nodes: list[str],
    edges: list[tuple[str, str, dict[str, Any]]],
    scores: dict[str, float] | None = None,
) -> SubgraphResult:
    """Convenience factory — converts mutable inputs to immutable result."""
    return SubgraphResult(
        nodes=tuple(nodes),
        edges=tuple(edges),
        scores=MappingProxyType(scores or {}),
    )

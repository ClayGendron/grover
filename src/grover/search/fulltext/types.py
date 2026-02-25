"""Full-text search result type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullTextResult:
    """A single full-text search hit."""

    path: str
    snippet: str = ""
    rank: float = 0.0

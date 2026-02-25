"""Full-text search stores — BM25 keyword search via native DB features."""

from __future__ import annotations

from grover.search.fulltext.protocol import FullTextStore
from grover.search.fulltext.types import FullTextResult

__all__ = [
    "FullTextResult",
    "FullTextStore",
]

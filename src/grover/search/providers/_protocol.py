"""EmbeddingProvider protocol — deprecated, use grover.search.protocols instead.

This module re-exports the ``EmbeddingProvider`` protocol from
:mod:`grover.search.protocols` for backward compatibility.  New code
should import directly from ``grover.search.protocols``.
"""

from grover.search.protocols import EmbeddingProvider

__all__ = ["EmbeddingProvider"]

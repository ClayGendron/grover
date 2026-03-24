"""Shared test helpers for Grover tests."""

from __future__ import annotations

import hashlib
import math

FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing.

    Uses SHA-256 to hash text into a normalized vector of dimension FAKE_DIM.
    Shared across test files to avoid copy-paste duplication.
    """

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

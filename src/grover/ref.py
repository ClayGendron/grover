"""Ref — file path identity for Grover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Ref:
    """Immutable reference to a file (or chunk) at a specific version.

    Attributes:
        path: Normalized virtual filesystem path.
        version: Optional version identifier (int or str).
        line_start: Optional start line for chunk references.
        line_end: Optional end line for chunk references.
        metadata: Arbitrary metadata (excluded from hash/equality).
    """

    path: str
    version: int | str | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __repr__(self) -> str:
        parts = [f"path={self.path!r}"]
        if self.version is not None:
            parts.append(f"version={self.version!r}")
        if self.line_start is not None:
            parts.append(f"line_start={self.line_start!r}")
        if self.line_end is not None:
            parts.append(f"line_end={self.line_end!r}")
        if self.metadata:
            parts.append(f"metadata={self.metadata!r}")
        return f"Ref({', '.join(parts)})"


def file_ref(path: str, version: int | str | None = None) -> Ref:
    """Create a Ref for a whole file, normalizing the path."""
    from grover.fs.utils import normalize_path

    return Ref(path=normalize_path(path), version=version)

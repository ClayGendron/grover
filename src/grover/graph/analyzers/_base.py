"""Analyzer protocol and shared types."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ChunkFile:
    """A structural chunk extracted from a source file."""

    chunk_path: str
    parent_path: str
    content: str
    line_start: int
    line_end: int
    name: str


@dataclass(frozen=True, slots=True)
class EdgeData:
    """A directed edge extracted from source analysis."""

    source: str
    target: str
    edge_type: str
    metadata: dict[str, object] = field(default_factory=dict)


type AnalysisResult = tuple[list[ChunkFile], list[EdgeData]]


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for language-specific AST analyzers.

    Analyzers are pure functions: they receive a file path and its content,
    and return extracted chunks and edges without mutating any state.
    """

    @property
    def extensions(self) -> frozenset[str]:
        """File extensions this analyzer handles (e.g. ``{".py"}``)."""
        ...

    def analyze_file(self, path: str, content: str) -> AnalysisResult:
        """Analyze *content* of the file at *path*.

        Returns ``(chunks, edges)`` — never raises on malformed input.
        """
        ...


def build_chunk_path(parent_path: str, symbol_name: str) -> str:
    """Build the canonical chunk path for a symbol.

    >>> build_chunk_path("/src/auth.py", "login")
    '/.grover/chunks/src/auth_py/login.txt'
    >>> build_chunk_path("/src/auth.py", "Client.connect")
    '/.grover/chunks/src/auth_py/Client.connect.txt'
    """
    parent_dir = posixpath.dirname(parent_path)
    stem = posixpath.basename(parent_path).replace(".", "_")
    return posixpath.join("/.grover/chunks", parent_dir.lstrip("/"), stem, f"{symbol_name}.txt")


def extract_lines(content: str, line_start: int, line_end: int) -> str:
    """Extract lines from *content* (1-indexed, inclusive).

    Clamps bounds to actual content length.
    """
    lines = content.splitlines(keepends=True)
    start = max(line_start - 1, 0)
    end = min(line_end, len(lines))
    return "".join(lines[start:end])

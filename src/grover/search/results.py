"""FileSearchResult subclasses and Evidence types for search/query operations.

Each subclass wraps a specific kind of search result with typed evidence
and convenience accessors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from grover.results import Evidence, FileSearchResult

if TYPE_CHECKING:
    from datetime import datetime


# =====================================================================
# Evidence types (frozen dataclasses)
# =====================================================================


@dataclass(frozen=True)
class LineMatch:
    """A single line match within a file."""

    line_number: int
    line_content: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlobEvidence(Evidence):
    """Evidence from a glob match."""

    is_directory: bool = False
    size_bytes: int | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class GrepEvidence(Evidence):
    """Evidence from a grep match."""

    line_matches: tuple[LineMatch, ...] = ()


@dataclass(frozen=True)
class TreeEvidence(Evidence):
    """Evidence from a tree listing."""

    depth: int = 0
    is_directory: bool = False


@dataclass(frozen=True)
class ListDirEvidence(Evidence):
    """Evidence from a directory listing."""

    is_directory: bool = False
    size_bytes: int | None = None


@dataclass(frozen=True)
class TrashEvidence(Evidence):
    """Evidence from a trash listing."""

    deleted_at: datetime | None = None
    original_path: str = ""


@dataclass(frozen=True)
class VectorEvidence(Evidence):
    """Evidence from a vector (semantic) search."""

    snippet: str = ""


@dataclass(frozen=True)
class LexicalEvidence(Evidence):
    """Evidence from a lexical (BM25/full-text) search."""

    snippet: str = ""


@dataclass(frozen=True)
class HybridEvidence(Evidence):
    """Evidence from a hybrid search."""

    snippet: str = ""


@dataclass(frozen=True)
class GraphEvidence(Evidence):
    """Evidence from a graph query."""

    algorithm: str = ""
    relationship: str = ""


# =====================================================================
# FileSearchResult subclasses
# =====================================================================


@dataclass
class GlobResult(FileSearchResult):
    """Result of a glob operation — file pattern matching."""

    def directories(self) -> tuple[str, ...]:
        """Return paths that are directories."""
        return tuple(
            p
            for p, evs in self._entries.items()
            if any(isinstance(e, GlobEvidence) and e.is_directory for e in evs)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files (not directories)."""
        return tuple(
            p
            for p, evs in self._entries.items()
            if any(isinstance(e, GlobEvidence) and not e.is_directory for e in evs)
        )

    def file_info(self, path: str) -> GlobEvidence | None:
        """Return the GlobEvidence for *path*, or ``None``."""
        for e in self._entries.get(path, []):
            if isinstance(e, GlobEvidence):
                return e
        return None


@dataclass
class GrepResult(FileSearchResult):
    """Result of a grep operation — pattern matching within files."""

    def line_matches(self, path: str) -> tuple[LineMatch, ...]:
        """Return all line matches for *path*."""
        for e in self._entries.get(path, []):
            if isinstance(e, GrepEvidence):
                return e.line_matches
        return ()

    def all_matches(self) -> list[tuple[str, LineMatch]]:
        """Return all (path, line_match) pairs across all files."""
        result: list[tuple[str, LineMatch]] = []
        for path, evs in self._entries.items():
            for e in evs:
                if isinstance(e, GrepEvidence):
                    result.extend((path, lm) for lm in e.line_matches)
        return result


@dataclass
class TreeResult(FileSearchResult):
    """Result of a tree operation — recursive directory listing."""

    @property
    def total_files(self) -> int:
        """Count of files in the tree."""
        return sum(
            1
            for evs in self._entries.values()
            if any(isinstance(e, TreeEvidence) and not e.is_directory for e in evs)
        )

    @property
    def total_dirs(self) -> int:
        """Count of directories in the tree."""
        return sum(
            1
            for evs in self._entries.values()
            if any(isinstance(e, TreeEvidence) and e.is_directory for e in evs)
        )


@dataclass
class ListDirResult(FileSearchResult):
    """Result of a list_dir operation."""

    def directories(self) -> tuple[str, ...]:
        """Return paths that are directories."""
        return tuple(
            p
            for p, evs in self._entries.items()
            if any(isinstance(e, ListDirEvidence) and e.is_directory for e in evs)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files."""
        return tuple(
            p
            for p, evs in self._entries.items()
            if any(isinstance(e, ListDirEvidence) and not e.is_directory for e in evs)
        )


@dataclass
class TrashResult(FileSearchResult):
    """Result of a list_trash operation."""

    def deleted_paths(self) -> tuple[str, ...]:
        """Return all original paths of deleted items."""
        return tuple(
            e.original_path
            for evs in self._entries.values()
            for e in evs
            if isinstance(e, TrashEvidence) and e.original_path
        )


@dataclass
class VectorSearchResult(FileSearchResult):
    """Result of a vector (semantic) search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        return tuple(
            e.snippet
            for e in self._entries.get(path, [])
            if isinstance(e, VectorEvidence) and e.snippet
        )


@dataclass
class LexicalSearchResult(FileSearchResult):
    """Result of a lexical (BM25/full-text) search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        return tuple(
            e.snippet
            for e in self._entries.get(path, [])
            if isinstance(e, LexicalEvidence) and e.snippet
        )


@dataclass
class HybridSearchResult(FileSearchResult):
    """Result of a hybrid search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        return tuple(
            e.snippet
            for e in self._entries.get(path, [])
            if isinstance(e, HybridEvidence) and e.snippet
        )


@dataclass
class GraphResult(FileSearchResult):
    """Result of a graph query."""

    @property
    def algorithm(self) -> str:
        """Return the algorithm used, from the first GraphEvidence found."""
        for evs in self._entries.values():
            for e in evs:
                if isinstance(e, GraphEvidence):
                    return e.algorithm
        return ""

    def relationships(self, path: str) -> tuple[str, ...]:
        """Return relationship types for *path*."""
        return tuple(
            e.relationship
            for e in self._entries.get(path, [])
            if isinstance(e, GraphEvidence) and e.relationship
        )

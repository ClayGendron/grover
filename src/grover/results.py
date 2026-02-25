"""Base result types: FileOperationResult (content ops) and FileSearchResult (reference ops).

Every Grover method returns a typed subclass of one of these two bases:

- **FileOperationResult** — non-chainable, for content operations (read, write, edit, ...)
- **FileSearchResult** — chainable via set algebra (``&``, ``|``, ``-``, ``>>``),
  for reference/query operations (glob, grep, search, graph queries, ...)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from grover.ref import Ref


# =====================================================================
# FileOperationResult — non-chainable base for content operations
# =====================================================================


@dataclass
class FileOperationResult:
    """Base for content operations. Non-chainable.

    Subclasses: ``ReadResult``, ``WriteResult``, ``EditResult``,
    ``DeleteResult``, ``MoveResult``, ``MkdirResult``, ``RestoreResult``,
    ``ListVersionsResult``, ``GetVersionContentResult``, ``ShareResult``,
    ``ListSharesResult``.
    """

    success: bool
    message: str


# =====================================================================
# Evidence — why a path appeared in a search result
# =====================================================================


@dataclass(frozen=True)
class Evidence:
    """Base evidence — why a path appeared in a search result."""

    strategy: str
    path: str


# =====================================================================
# FileSearchResult — chainable base for reference operations
# =====================================================================


@dataclass
class FileSearchResult:
    """Base for reference operations. Returns file paths, not content.

    Supports set algebra:

    - ``&`` (intersection) — paths in both, merges evidence
    - ``|`` (union) — paths from either, merges evidence
    - ``-`` (difference) — paths in LHS not in RHS
    - ``>>`` (pipeline) — passes LHS paths as candidates to RHS
    """

    success: bool
    message: str
    _entries: dict[str, list[Evidence]] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file paths in this result."""
        return tuple(self._entries.keys())

    def explain(self, path: str) -> list[Evidence]:
        """Return the evidence chain for *path*."""
        return list(self._entries.get(path, []))

    def to_refs(self) -> list[Ref]:
        """Convert to a list of ``Ref`` objects."""
        from grover.ref import Ref

        return [Ref(path=p) for p in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return self.success and len(self._entries) > 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __contains__(self, path: object) -> bool:
        return path in self._entries

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> Self:
        """Return a new result with all paths prefixed by *prefix*.

        Preserves subclass type and any extra fields via shallow copy.
        Evidence objects are reconstructed with updated paths.
        """
        result = copy.copy(self)
        new_entries: dict[str, list[Evidence]] = {}
        for path, evs in self._entries.items():
            new_path = prefix + path if path != "/" else prefix
            new_evs = [dc_replace(e, path=new_path) for e in evs]
            new_entries[new_path] = new_evs
        result._entries = new_entries
        return result

    def remap_paths(self, fn: Callable[[str], str]) -> Self:
        """Return a new result with all paths transformed by *fn*.

        Preserves subclass type and any extra fields via shallow copy.
        """
        result = copy.copy(self)
        new_entries: dict[str, list[Evidence]] = {}
        for path, evs in self._entries.items():
            new_path = fn(path)
            new_evs = [dc_replace(e, path=new_path) for e in evs]
            if new_path in new_entries:
                new_entries[new_path].extend(new_evs)
            else:
                new_entries[new_path] = new_evs
        result._entries = new_entries
        return result

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str], *, strategy: str = "unknown") -> FileSearchResult:
        """Create a result from a list of paths with default evidence."""
        entries: dict[str, list[Evidence]] = {}
        for p in paths:
            entries[p] = [Evidence(strategy=strategy, path=p)]
        return cls(success=True, message=f"{len(paths)} paths", _entries=entries)

    @classmethod
    def from_refs(cls, refs: list[Ref], *, strategy: str = "unknown") -> FileSearchResult:
        """Create a result from a list of ``Ref`` objects."""
        entries: dict[str, list[Evidence]] = {}
        for ref in refs:
            entries[ref.path] = [Evidence(strategy=strategy, path=ref.path)]
        return cls(success=True, message=f"{len(refs)} refs", _entries=entries)

    # -----------------------------------------------------------------
    # Set algebra
    # -----------------------------------------------------------------

    def _merge_entries(self, other: FileSearchResult, paths: set[str]) -> dict[str, list[Evidence]]:
        """Merge evidence from both sides for the given *paths*."""
        merged: dict[str, list[Evidence]] = {}
        for p in paths:
            evidence: list[Evidence] = []
            evidence.extend(self._entries.get(p, []))
            evidence.extend(other._entries.get(p, []))
            merged[p] = evidence
        return merged

    def _result_class(self, other: FileSearchResult) -> type[FileSearchResult]:
        """Return the subclass to use for the result of a set operation."""
        if type(self) is type(other) and type(self) is not FileSearchResult:
            return type(self)
        return FileSearchResult

    def __and__(self, other: Any) -> FileSearchResult:
        """Intersection — paths in both, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        common = set(self._entries) & set(other._entries)
        merged = self._merge_entries(other, common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(success=success, message=f"{len(merged)} paths", _entries=merged)

    def __or__(self, other: Any) -> FileSearchResult:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        all_paths = set(self._entries) | set(other._entries)
        merged = self._merge_entries(other, all_paths)
        cls = self._result_class(other)
        success = self.success or other.success
        return cls(success=success, message=f"{len(merged)} paths", _entries=merged)

    def __sub__(self, other: Any) -> FileSearchResult:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        diff = set(self._entries) - set(other._entries)
        entries = {p: list(self._entries[p]) for p in diff}
        cls = self._result_class(other)
        return cls(success=self.success, message=f"{len(entries)} paths", _entries=entries)

    def __rshift__(self, other: Any) -> FileSearchResult:
        """Pipeline — passes LHS paths as candidates to RHS (intersection semantics)."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        # Same as intersection, but evidence comes from both
        common = set(self._entries) & set(other._entries)
        merged = self._merge_entries(other, common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(success=success, message=f"{len(merged)} paths", _entries=merged)

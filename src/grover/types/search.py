"""Search result types — FileSearchResult, Evidence, and all subclasses.

Every reference/query method in Grover returns a typed subclass of
``FileSearchResult``.  These are chainable via set algebra
(``&``, ``|``, ``-``, ``>>``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

    from grover.ref import Ref


# =====================================================================
# Evidence — why a path appeared in a search result
# =====================================================================


@dataclass(frozen=True)
class Evidence:
    """Base evidence — why a path appeared in a search result."""

    strategy: str
    path: str


# =====================================================================
# FileSearchCandidate — a single path with its evidence chain
# =====================================================================


@dataclass(frozen=True)
class FileSearchCandidate:
    """A path and its associated evidence chain.

    Replaces the old ``_entries: dict[str, list[Evidence]]`` internal dict.
    """

    path: str
    evidence: list[Evidence]


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

    success: bool = True
    message: str = ""
    candidates: list[FileSearchCandidate] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Internal dict conversion (for efficient set algebra)
    # -----------------------------------------------------------------

    def _as_dict(self) -> dict[str, list[Evidence]]:
        """Convert candidates to dict for set algebra."""
        return {c.path: list(c.evidence) for c in self.candidates}

    @staticmethod
    def _dict_to_candidates(entries: dict[str, list[Evidence]]) -> list[FileSearchCandidate]:
        """Convert dict back to candidates list."""
        return [FileSearchCandidate(path=p, evidence=evs) for p, evs in entries.items()]

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file paths in this result."""
        return tuple(c.path for c in self.candidates)

    def explain(self, path: str) -> list[Evidence]:
        """Return the evidence chain for *path*."""
        for c in self.candidates:
            if c.path == path:
                return list(c.evidence)
        return []

    def to_refs(self) -> list[Ref]:
        """Convert to a list of ``Ref`` objects."""
        from grover.ref import Ref

        return [Ref(path=c.path) for c in self.candidates]

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return self.success and len(self.candidates) > 0

    def __iter__(self) -> Iterator[str]:
        return iter(c.path for c in self.candidates)

    def __contains__(self, path: object) -> bool:
        return any(c.path == path for c in self.candidates)

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> Self:
        """Return a new result with all paths prefixed by *prefix*.

        Preserves subclass type and any extra fields via shallow copy.
        Evidence objects are reconstructed with updated paths.
        """
        result = copy.copy(self)
        new_candidates: list[FileSearchCandidate] = []
        for c in self.candidates:
            new_path = prefix + c.path if c.path != "/" else prefix
            new_evs = [dc_replace(e, path=new_path) for e in c.evidence]
            new_candidates.append(FileSearchCandidate(path=new_path, evidence=new_evs))
        result.candidates = new_candidates
        return result

    def remap_paths(self, fn: Callable[[str], str]) -> Self:
        """Return a new result with all paths transformed by *fn*.

        Preserves subclass type and any extra fields via shallow copy.
        """
        result = copy.copy(self)
        # Use dict to merge evidence for paths that map to the same new path
        merged: dict[str, list[Evidence]] = {}
        for c in self.candidates:
            new_path = fn(c.path)
            new_evs = [dc_replace(e, path=new_path) for e in c.evidence]
            if new_path in merged:
                merged[new_path].extend(new_evs)
            else:
                merged[new_path] = new_evs
        result.candidates = self._dict_to_candidates(merged)
        return result

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str], *, strategy: str = "unknown") -> Self:
        """Create a result from a list of paths with default evidence."""
        candidates = [
            FileSearchCandidate(path=p, evidence=[Evidence(strategy=strategy, path=p)])
            for p in paths
        ]
        return cls(success=True, message=f"{len(paths)} paths", candidates=candidates)

    @classmethod
    def from_refs(cls, refs: list[Ref], *, strategy: str = "unknown") -> Self:
        """Create a result from a list of ``Ref`` objects."""
        candidates = [
            FileSearchCandidate(
                path=ref.path, evidence=[Evidence(strategy=strategy, path=ref.path)]
            )
            for ref in refs
        ]
        return cls(success=True, message=f"{len(refs)} refs", candidates=candidates)

    # -----------------------------------------------------------------
    # Set algebra
    # -----------------------------------------------------------------

    def _merge_dicts(self, other: FileSearchResult, paths: set[str]) -> dict[str, list[Evidence]]:
        """Merge evidence from both sides for the given *paths*."""
        d1 = self._as_dict()
        d2 = other._as_dict()
        merged: dict[str, list[Evidence]] = {}
        for p in paths:
            evidence: list[Evidence] = []
            evidence.extend(d1.get(p, []))
            evidence.extend(d2.get(p, []))
            merged[p] = evidence
        return merged

    def _result_class(self, other: FileSearchResult) -> type[Self]:
        """Return the subclass to use for the result of a set operation."""
        if type(self) is type(other) and type(self) is not FileSearchResult:
            return type(self)
        return FileSearchResult  # type: ignore[return-value]

    def __and__(self, other: object) -> Self:
        """Intersection — paths in both, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        common = set(d1) & set(d2)
        merged = self._merge_dicts(other, common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            candidates=self._dict_to_candidates(merged),
        )

    def __or__(self, other: object) -> Self:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        all_paths = set(d1) | set(d2)
        merged = self._merge_dicts(other, all_paths)
        cls = self._result_class(other)
        success = self.success or other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            candidates=self._dict_to_candidates(merged),
        )

    def __sub__(self, other: object) -> Self:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        diff = set(d1) - set(d2)
        entries = {p: list(d1[p]) for p in diff}
        cls = self._result_class(other)
        return cls(
            success=self.success,
            message=f"{len(entries)} paths",
            candidates=self._dict_to_candidates(entries),
        )

    def __rshift__(self, other: object) -> Self:
        """Pipeline — passes LHS paths as candidates to RHS (intersection semantics)."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        common = set(d1) & set(d2)
        merged = self._merge_dicts(other, common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            candidates=self._dict_to_candidates(merged),
        )


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


@dataclass(frozen=True)
class VersionEvidence(Evidence):
    """Evidence from a version listing."""

    version: int = 0
    content_hash: str = ""
    size_bytes: int = 0
    created_at: datetime | None = None
    created_by: str | None = None


@dataclass(frozen=True)
class ShareEvidence(Evidence):
    """Evidence from a share listing."""

    grantee_id: str = ""
    permission: str = ""
    granted_by: str = ""
    expires_at: datetime | None = None


# =====================================================================
# FileSearchResult subclasses
# =====================================================================


@dataclass
class GlobResult(FileSearchResult):
    """Result of a glob operation — file pattern matching."""

    pattern: str = ""

    def directories(self) -> tuple[str, ...]:
        """Return paths that are directories."""
        return tuple(
            c.path
            for c in self.candidates
            if any(isinstance(e, GlobEvidence) and e.is_directory for e in c.evidence)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files (not directories)."""
        return tuple(
            c.path
            for c in self.candidates
            if any(isinstance(e, GlobEvidence) and not e.is_directory for e in c.evidence)
        )

    def file_info(self, path: str) -> GlobEvidence | None:
        """Return the GlobEvidence for *path*, or ``None``."""
        for c in self.candidates:
            if c.path == path:
                for e in c.evidence:
                    if isinstance(e, GlobEvidence):
                        return e
        return None


@dataclass
class GrepResult(FileSearchResult):
    """Result of a grep operation — pattern matching within files."""

    pattern: str = ""
    files_searched: int = 0
    files_matched: int = 0
    truncated: bool = False

    def line_matches(self, path: str) -> tuple[LineMatch, ...]:
        """Return all line matches for *path*."""
        for c in self.candidates:
            if c.path == path:
                for e in c.evidence:
                    if isinstance(e, GrepEvidence):
                        return e.line_matches
        return ()

    def all_matches(self) -> list[tuple[str, LineMatch]]:
        """Return all (path, line_match) pairs across all files."""
        result: list[tuple[str, LineMatch]] = []
        for c in self.candidates:
            for e in c.evidence:
                if isinstance(e, GrepEvidence):
                    result.extend((c.path, lm) for lm in e.line_matches)
        return result


@dataclass
class TreeResult(FileSearchResult):
    """Result of a tree operation — recursive directory listing."""

    @property
    def total_files(self) -> int:
        """Count of files in the tree."""
        return sum(
            1
            for c in self.candidates
            if any(isinstance(e, TreeEvidence) and not e.is_directory for e in c.evidence)
        )

    @property
    def total_dirs(self) -> int:
        """Count of directories in the tree."""
        return sum(
            1
            for c in self.candidates
            if any(isinstance(e, TreeEvidence) and e.is_directory for e in c.evidence)
        )


@dataclass
class ListDirResult(FileSearchResult):
    """Result of a list_dir operation."""

    def directories(self) -> tuple[str, ...]:
        """Return paths that are directories."""
        return tuple(
            c.path
            for c in self.candidates
            if any(isinstance(e, ListDirEvidence) and e.is_directory for e in c.evidence)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files."""
        return tuple(
            c.path
            for c in self.candidates
            if any(isinstance(e, ListDirEvidence) and not e.is_directory for e in c.evidence)
        )


@dataclass
class TrashResult(FileSearchResult):
    """Result of a list_trash operation."""

    def deleted_paths(self) -> tuple[str, ...]:
        """Return all original paths of deleted items."""
        return tuple(
            e.original_path
            for c in self.candidates
            for e in c.evidence
            if isinstance(e, TrashEvidence) and e.original_path
        )


@dataclass
class VectorSearchResult(FileSearchResult):
    """Result of a vector (semantic) search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        for c in self.candidates:
            if c.path == path:
                return tuple(
                    e.snippet for e in c.evidence if isinstance(e, VectorEvidence) and e.snippet
                )
        return ()


@dataclass
class LexicalSearchResult(FileSearchResult):
    """Result of a lexical (BM25/full-text) search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        for c in self.candidates:
            if c.path == path:
                return tuple(
                    e.snippet for e in c.evidence if isinstance(e, LexicalEvidence) and e.snippet
                )
        return ()


@dataclass
class HybridSearchResult(FileSearchResult):
    """Result of a hybrid search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        for c in self.candidates:
            if c.path == path:
                return tuple(
                    e.snippet for e in c.evidence if isinstance(e, HybridEvidence) and e.snippet
                )
        return ()


@dataclass
class GraphResult(FileSearchResult):
    """Result of a graph query."""

    @property
    def algorithm(self) -> str:
        """Return the algorithm used, from the first GraphEvidence found."""
        for c in self.candidates:
            for e in c.evidence:
                if isinstance(e, GraphEvidence):
                    return e.algorithm
        return ""

    def relationships(self, path: str) -> tuple[str, ...]:
        """Return relationship types for *path*."""
        for c in self.candidates:
            if c.path == path:
                return tuple(
                    e.relationship
                    for e in c.evidence
                    if isinstance(e, GraphEvidence) and e.relationship
                )
        return ()


@dataclass
class VersionResult(FileSearchResult):
    """Result of a list_versions operation.

    Each candidate's path is ``file_path@version`` and carries
    ``VersionEvidence`` with version metadata.
    """


@dataclass
class ShareSearchResult(FileSearchResult):
    """Result of a list_shares operation.

    Each candidate carries ``ShareEvidence`` with share metadata.
    """

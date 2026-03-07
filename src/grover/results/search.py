"""Search result types — FileSearchResult, Evidence, and all subclasses.

Every reference/query method in Grover returns a typed subclass of
``FileSearchResult``.  These are chainable via set algebra
(``&``, ``|``, ``-``, ``>>``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from datetime import datetime

    from grover.ref import Ref


# =====================================================================
# Evidence — why a path appeared in a search result
# =====================================================================


@dataclass(frozen=True)
class Evidence:
    """Base evidence — why a path appeared in a search result."""

    operation: str
    score: float = 0.0
    query_args: dict = field(default_factory=dict)


# =====================================================================
# FileCandidate — a single path with its evidence chain
# =====================================================================


@dataclass(frozen=True)
class FileCandidate:
    """A path and its associated evidence chain."""

    path: str
    evidence: list[Evidence]

    @property
    def scores(self) -> dict[str, float]:
        """Aggregate scores from evidence, keyed by operation."""
        return {e.operation: e.score for e in self.evidence if e.score > 0}


# =====================================================================
# ConnectionCandidate — a connection (edge) with its evidence chain
# =====================================================================


@dataclass(frozen=True)
class ConnectionCandidate:
    """A connection (edge) with its evidence chain."""

    source_path: str
    target_path: str
    connection_type: str
    weight: float = 1.0
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def path(self) -> str:
        """Ref-format: source[type]target."""
        return f"{self.source_path}[{self.connection_type}]{self.target_path}"

    @property
    def scores(self) -> dict[str, float]:
        """Aggregate scores from evidence, keyed by operation."""
        return {e.operation: e.score for e in self.evidence if e.score > 0}


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
    file_candidates: list[FileCandidate] = field(default_factory=list)
    connection_candidates: list[ConnectionCandidate] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Internal dict conversion (for efficient set algebra)
    # -----------------------------------------------------------------

    def _as_dict(self) -> dict[str, list[Evidence]]:
        """Convert file_candidates to dict for set algebra."""
        return {c.path: list(c.evidence) for c in self.file_candidates}

    @staticmethod
    def _dict_to_candidates(entries: Mapping[str, Sequence[Evidence]]) -> list[FileCandidate]:
        """Convert dict back to file_candidates list."""
        return [FileCandidate(path=p, evidence=list(evs)) for p, evs in entries.items()]

    # -----------------------------------------------------------------
    # Connection dict helpers
    # -----------------------------------------------------------------

    def _connections_as_dict(self) -> dict[str, ConnectionCandidate]:
        """Convert connection_candidates to dict keyed by ref-format path."""
        return {cc.path: cc for cc in self.connection_candidates}

    @staticmethod
    def _merge_connection_candidates(
        d1: dict[str, ConnectionCandidate],
        d2: dict[str, ConnectionCandidate],
        paths: set[str],
    ) -> list[ConnectionCandidate]:
        """Merge connection candidates from both sides for the given *paths*."""
        merged: list[ConnectionCandidate] = []
        for p in paths:
            c1, c2 = d1.get(p), d2.get(p)
            if c1 and c2:
                merged.append(
                    ConnectionCandidate(
                        source_path=c1.source_path,
                        target_path=c1.target_path,
                        connection_type=c1.connection_type,
                        weight=c1.weight,
                        evidence=list(c1.evidence) + list(c2.evidence),
                    )
                )
            elif c1:
                merged.append(c1)
            elif c2:
                merged.append(c2)
        return merged

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file paths in this result."""
        return tuple(c.path for c in self.file_candidates)

    @property
    def connection_paths(self) -> tuple[str, ...]:
        """All connection ref-format paths in this result."""
        return tuple(cc.path for cc in self.connection_candidates)

    def explain(self, path: str) -> list[Evidence]:
        """Return the evidence chain for *path*."""
        for c in self.file_candidates:
            if c.path == path:
                return list(c.evidence)
        return []

    def to_refs(self) -> list[Ref]:
        """Convert to a list of ``Ref`` objects."""
        from grover.ref import Ref

        return [Ref(path=c.path) for c in self.file_candidates]

    def __len__(self) -> int:
        return len(self.file_candidates)

    def __bool__(self) -> bool:
        return self.success and len(self.file_candidates) > 0

    def __iter__(self) -> Iterator[str]:
        return iter(c.path for c in self.file_candidates)

    def __contains__(self, path: object) -> bool:
        return any(c.path == path for c in self.file_candidates)

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> Self:
        """Return a new result with all paths prefixed by *prefix*.

        Preserves subclass type and any extra fields via shallow copy.
        """
        result = copy.copy(self)
        new_candidates: list[FileCandidate] = []
        for c in self.file_candidates:
            new_path = prefix + c.path if c.path != "/" else prefix
            new_candidates.append(FileCandidate(path=new_path, evidence=list(c.evidence)))
        result.file_candidates = new_candidates
        new_conns: list[ConnectionCandidate] = []
        for cc in self.connection_candidates:
            new_src = prefix + cc.source_path if cc.source_path != "/" else prefix
            new_tgt = prefix + cc.target_path if cc.target_path != "/" else prefix
            new_conns.append(
                ConnectionCandidate(
                    source_path=new_src,
                    target_path=new_tgt,
                    connection_type=cc.connection_type,
                    weight=cc.weight,
                    evidence=list(cc.evidence),
                )
            )
        result.connection_candidates = new_conns
        return result

    def remap_paths(self, fn: Callable[[str], str]) -> Self:
        """Return a new result with all paths transformed by *fn*.

        Preserves subclass type and any extra fields via shallow copy.
        """
        result = copy.copy(self)
        # Use dict to merge evidence for paths that map to the same new path
        merged: dict[str, list[Evidence]] = {}
        for c in self.file_candidates:
            new_path = fn(c.path)
            if new_path in merged:
                merged[new_path].extend(c.evidence)
            else:
                merged[new_path] = list(c.evidence)
        result.file_candidates = self._dict_to_candidates(merged)
        # Remap connection paths
        result.connection_candidates = [
            ConnectionCandidate(
                source_path=fn(cc.source_path),
                target_path=fn(cc.target_path),
                connection_type=cc.connection_type,
                weight=cc.weight,
                evidence=list(cc.evidence),
            )
            for cc in self.connection_candidates
        ]
        return result

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str], *, operation: str = "unknown") -> Self:
        """Create a result from a list of paths with default evidence."""
        candidates = [
            FileCandidate(path=p, evidence=[Evidence(operation=operation)]) for p in paths
        ]
        return cls(success=True, message=f"{len(paths)} paths", file_candidates=candidates)

    @classmethod
    def from_refs(cls, refs: list[Ref], *, operation: str = "unknown") -> Self:
        """Create a result from a list of ``Ref`` objects."""
        candidates = [
            FileCandidate(path=ref.path, evidence=[Evidence(operation=operation)]) for ref in refs
        ]
        return cls(success=True, message=f"{len(refs)} refs", file_candidates=candidates)

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
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_common = set(cd1) & set(cd2)
        conn_merged = self._merge_connection_candidates(cd1, cd2, conn_common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            file_candidates=self._dict_to_candidates(merged),
            connection_candidates=conn_merged,
        )

    def __or__(self, other: object) -> Self:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        all_paths = set(d1) | set(d2)
        merged = self._merge_dicts(other, all_paths)
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_all = set(cd1) | set(cd2)
        conn_merged = self._merge_connection_candidates(cd1, cd2, conn_all)
        cls = self._result_class(other)
        success = self.success or other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            file_candidates=self._dict_to_candidates(merged),
            connection_candidates=conn_merged,
        )

    def __sub__(self, other: object) -> Self:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        diff = set(d1) - set(d2)
        entries = {p: list(d1[p]) for p in diff}
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_diff = set(cd1) - set(cd2)
        conn_entries = [cd1[p] for p in conn_diff]
        cls = self._result_class(other)
        return cls(
            success=self.success,
            message=f"{len(entries)} paths",
            file_candidates=self._dict_to_candidates(entries),
            connection_candidates=conn_entries,
        )

    def __rshift__(self, other: object) -> Self:
        """Pipeline — passes LHS paths as candidates to RHS (intersection semantics)."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        common = set(d1) & set(d2)
        merged = self._merge_dicts(other, common)
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_common = set(cd1) & set(cd2)
        conn_merged = self._merge_connection_candidates(cd1, cd2, conn_common)
        cls = self._result_class(other)
        success = self.success and other.success
        return cls(
            success=success,
            message=f"{len(merged)} paths",
            file_candidates=self._dict_to_candidates(merged),
            connection_candidates=conn_merged,
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
            for c in self.file_candidates
            if any(isinstance(e, GlobEvidence) and e.is_directory for e in c.evidence)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files (not directories)."""
        return tuple(
            c.path
            for c in self.file_candidates
            if any(isinstance(e, GlobEvidence) and not e.is_directory for e in c.evidence)
        )

    def file_info(self, path: str) -> GlobEvidence | None:
        """Return the GlobEvidence for *path*, or ``None``."""
        for c in self.file_candidates:
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
        for c in self.file_candidates:
            if c.path == path:
                for e in c.evidence:
                    if isinstance(e, GrepEvidence):
                        return e.line_matches
        return ()

    def all_matches(self) -> list[tuple[str, LineMatch]]:
        """Return all (path, line_match) pairs across all files."""
        result: list[tuple[str, LineMatch]] = []
        for c in self.file_candidates:
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
            for c in self.file_candidates
            if any(isinstance(e, TreeEvidence) and not e.is_directory for e in c.evidence)
        )

    @property
    def total_dirs(self) -> int:
        """Count of directories in the tree."""
        return sum(
            1
            for c in self.file_candidates
            if any(isinstance(e, TreeEvidence) and e.is_directory for e in c.evidence)
        )


@dataclass
class ListDirResult(FileSearchResult):
    """Result of a list_dir operation."""

    def directories(self) -> tuple[str, ...]:
        """Return paths that are directories."""
        return tuple(
            c.path
            for c in self.file_candidates
            if any(isinstance(e, ListDirEvidence) and e.is_directory for e in c.evidence)
        )

    def files(self) -> tuple[str, ...]:
        """Return paths that are files."""
        return tuple(
            c.path
            for c in self.file_candidates
            if any(isinstance(e, ListDirEvidence) and not e.is_directory for e in c.evidence)
        )


@dataclass
class TrashResult(FileSearchResult):
    """Result of a list_trash operation."""

    def deleted_paths(self) -> tuple[str, ...]:
        """Return all original paths of deleted items."""
        return tuple(
            e.original_path
            for c in self.file_candidates
            for e in c.evidence
            if isinstance(e, TrashEvidence) and e.original_path
        )


@dataclass
class VectorSearchResult(FileSearchResult):
    """Result of a vector (semantic) search."""

    def snippets(self, path: str) -> tuple[str, ...]:
        """Return all snippets for *path*."""
        for c in self.file_candidates:
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
        for c in self.file_candidates:
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
        for c in self.file_candidates:
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
        for c in self.file_candidates:
            for e in c.evidence:
                if isinstance(e, GraphEvidence):
                    return e.algorithm
        return ""

    def relationships(self, path: str) -> tuple[str, ...]:
        """Return relationship types for *path*."""
        for c in self.file_candidates:
            if c.path == path:
                return tuple(
                    e.relationship
                    for e in c.evidence
                    if isinstance(e, GraphEvidence) and e.relationship
                )
        return ()


# =====================================================================
# Typed GraphResult subclasses
# =====================================================================

# --- Traversal results ---


@dataclass
class PredecessorsResult(GraphResult):
    """Result of a predecessors query."""


@dataclass
class SuccessorsResult(GraphResult):
    """Result of a successors query."""


@dataclass
class AncestorsResult(GraphResult):
    """Result of an ancestors query."""


@dataclass
class DescendantsResult(GraphResult):
    """Result of a descendants query."""


@dataclass
class ShortestPathResult(GraphResult):
    """Result of a shortest path query."""


@dataclass
class HasPathResult(GraphResult):
    """Result of a has_path check. bool(result) indicates connectivity."""


# --- Subgraph results ---


@dataclass
class SubgraphSearchResult(GraphResult):
    """Result of an induced subgraph extraction."""


@dataclass
class MeetingSubgraphResult(GraphResult):
    """Result of a minimum meeting subgraph extraction."""


@dataclass
class EgoGraphResult(GraphResult):
    """Result of an ego graph (neighborhood) extraction."""


# --- Centrality results ---


@dataclass
class PageRankResult(GraphResult):
    """Result of PageRank computation."""


@dataclass
class HitsResult(GraphResult):
    """Result of HITS computation. Each candidate has two evidence records:
    hits_authority and hits_hub."""

    def hub_score(self, path: str) -> float:
        """Return the hub score for *path*."""
        for c in self.file_candidates:
            if c.path == path:
                for e in c.evidence:
                    if isinstance(e, GraphEvidence) and e.operation == "hits_hub":
                        return e.score
        return 0.0

    def authority_score(self, path: str) -> float:
        """Return the authority score for *path*."""
        for c in self.file_candidates:
            if c.path == path:
                for e in c.evidence:
                    if isinstance(e, GraphEvidence) and e.operation == "hits_authority":
                        return e.score
        return 0.0


@dataclass
class BetweennessResult(GraphResult):
    """Result of betweenness centrality computation."""


@dataclass
class ClosenessResult(GraphResult):
    """Result of closeness centrality computation."""


@dataclass
class HarmonicResult(GraphResult):
    """Result of harmonic centrality computation."""


@dataclass
class KatzResult(GraphResult):
    """Result of Katz centrality computation."""


@dataclass
class DegreeResult(GraphResult):
    """Result of degree centrality computation (degree, in-degree, or out-degree)."""


# --- Other graph results ---


@dataclass
class CommonNeighborsResult(GraphResult):
    """Result of a common neighbors query."""


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

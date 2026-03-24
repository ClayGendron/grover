"""Composable result types for Grover.

Every Grover operation returns ``GroverResult``.  Results carry candidates,
provenance details, and an optional back-reference to the ``Grover`` instance
that produced them — enabling method chaining:

    g.semantic_search("auth", k=5).min_meeting_subgraph().pagerank().top(3)

Results serialize cleanly to JSON via Pydantic's ``model_dump(exclude_none=True)``
for use in REST APIs.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs this at runtime for field resolution
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from grover.paths import split_path

_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---------------------------------------------------------------------------
# Operation models — batch inputs for public methods
# ---------------------------------------------------------------------------


class EditOperation(BaseModel):
    """A single find-and-replace edit.

    Multiple ``EditOperation`` objects are applied sequentially — each
    sees the content left by the previous one.
    """

    model_config = ConfigDict(frozen=True)

    old: str
    new: str
    replace_all: bool = False


class TwoPathOperation(BaseModel):
    """A source/destination pair for move or copy."""

    model_config = ConfigDict(frozen=True)

    src: str
    dest: str


# ---------------------------------------------------------------------------
# Detail — provenance for a single chain step
# ---------------------------------------------------------------------------


class Detail(BaseModel):
    """Provenance record appended by each chain step.

    One flat model — optional fields cover all operation types.  Null fields
    are excluded from JSON via ``model_dump(exclude_none=True)``.
    """

    model_config = ConfigDict(frozen=True)

    operation: str
    success: bool = True
    message: str = ""
    score: float | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Candidate — one entity in a result set
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """Read-only projection of a ``GroverObject``.

    Scores live on ``Detail`` only — the ``score`` property derives the
    current score from the most recent detail.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str
    path: str
    kind: str

    # Content (populated by read)
    content: str | None = None

    # Metrics (populated by stat/read/write)
    lines: int = 0
    size_bytes: int = 0
    tokens: int = 0
    mime_type: str | None = None

    # Graph metrics
    weight: float | None = None
    distance: float | None = None

    # Provenance — accumulates through chain steps (tuple for true immutability)
    details: tuple[Detail, ...] = ()

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def score(self) -> float:
        """Score from the most recent detail, or 0.0."""
        if not self.details:
            return 0.0
        s = self.details[-1].score
        return s if s is not None else 0.0

    def score_for(self, operation: str) -> float:
        """Score for a specific operation, or 0.0.

        Searches details in reverse (most recent first).
        """
        for d in reversed(self.details):
            if d.operation == operation:
                return d.score if d.score is not None else 0.0
        return 0.0

    @property
    def name(self) -> str:
        """Last segment of the path."""
        return split_path(self.path)[1]


# ---------------------------------------------------------------------------
# GroverResult — the single composable result type
# ---------------------------------------------------------------------------


class GroverResult(BaseModel):
    """Unified result from every Grover operation.

    Supports:
    - **Data inspection:** ``.paths``, ``.content``, ``.file``, ``.explain()``
    - **Set algebra:** ``&`` (intersection), ``|`` (union), ``-`` (difference)
    - **Method chaining:** ``.semantic_search()``, ``.pagerank()``, ``.read()``, etc.
    - **Enrichment:** ``.sort()``, ``.top()``, ``.filter()``, ``.kinds()``
    - **Serialization:** ``.model_dump(exclude_none=True)`` for REST APIs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    errors: list[str] = []
    candidates: list[Candidate] = []

    # Back-reference for chaining — not serialized
    _grover: Any = PrivateAttr(default=None)

    # -------------------------------------------------------------------
    # Data access
    # -------------------------------------------------------------------

    @property
    def error_message(self) -> str:
        """All errors joined as a single string."""
        return "; ".join(self.errors)

    @property
    def paths(self) -> tuple[str, ...]:
        """All candidate paths as a tuple."""
        return tuple(c.path for c in self.candidates)

    @property
    def file(self) -> Candidate | None:
        """First candidate, or ``None`` if empty."""
        return self.candidates[0] if self.candidates else None

    @property
    def content(self) -> str | None:
        """Content of the first candidate, or ``None``."""
        return self.candidates[0].content if self.candidates else None

    def explain(self, path: str) -> list[Detail]:
        """Return the detail chain for a specific path."""
        for c in self.candidates:
            if c.path == path:
                return list(c.details)
        return []

    # -------------------------------------------------------------------
    # Iteration / truthiness
    # -------------------------------------------------------------------

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return self.success and len(self.candidates) > 0

    def __contains__(self, path: str) -> bool:
        return path in self.paths

    # -------------------------------------------------------------------
    # Set algebra
    # -------------------------------------------------------------------

    def _as_dict(self) -> dict[str, Candidate]:
        return {c.path: c for c in self.candidates}

    @staticmethod
    def _first_set(a: _T | None, b: _T | None, default: _T | None = None) -> _T | None:
        """Return *a* if not None, else *b*, else *default*."""
        return a if a is not None else (b if b is not None else default)

    @staticmethod
    def _merge_candidate(a: Candidate, b: Candidate) -> Candidate:
        """Merge two candidates for the same path — combine details.

        Left candidate (a) wins for all fields.  Falls back to right (b)
        only when left is None.  Uses explicit None checks — never falsy
        coalescing — so ``0``, ``""``, and ``0.0`` are preserved.
        """
        fs = GroverResult._first_set
        return Candidate(
            id=a.id,
            path=a.path,
            kind=a.kind,
            content=fs(a.content, b.content),
            lines=a.lines,
            size_bytes=a.size_bytes,
            tokens=a.tokens,
            mime_type=fs(a.mime_type, b.mime_type),
            weight=fs(a.weight, b.weight),
            distance=fs(a.distance, b.distance),
            details=a.details + b.details,
            created_at=fs(a.created_at, b.created_at),
            updated_at=fs(a.updated_at, b.updated_at),
        )

    def __and__(self, other: GroverResult) -> GroverResult:
        """Intersection — candidates in both, details merged."""
        left = self._as_dict()
        right = other._as_dict()
        merged = [self._merge_candidate(left[p], right[p]) for p in left if p in right]
        result = GroverResult(
            candidates=merged,
            success=self.success and other.success,
            errors=self.errors + other.errors,
        )
        result._grover = self._grover or other._grover
        return result

    def __or__(self, other: GroverResult) -> GroverResult:
        """Union — candidates from either, details merged where overlap."""
        left = self._as_dict()
        right = other._as_dict()
        merged: dict[str, Candidate] = {}
        for p, c in left.items():
            merged[p] = self._merge_candidate(c, right[p]) if p in right else c
        for p, c in right.items():
            if p not in merged:
                merged[p] = c
        result = GroverResult(
            candidates=list(merged.values()),
            success=self.success and other.success,
            errors=self.errors + other.errors,
        )
        result._grover = self._grover or other._grover
        return result

    def __sub__(self, other: GroverResult) -> GroverResult:
        """Difference — candidates in left not in right."""
        right_paths = set(other.paths)
        remaining = [c for c in self.candidates if c.path not in right_paths]
        result = GroverResult(
            candidates=remaining,
            success=self.success,
            errors=self.errors,
        )
        result._grover = self._grover
        return result

    # -------------------------------------------------------------------
    # Enrichment chains (local, no backend call)
    # -------------------------------------------------------------------

    def _with_candidates(self, candidates: list[Candidate]) -> GroverResult:
        """Return a new result with the given candidates, preserving _grover."""
        result = GroverResult(
            candidates=candidates,
            success=self.success,
            errors=self.errors,
        )
        result._grover = self._grover
        return result

    def sort(
        self,
        *,
        operation: str | None = None,
        key: Callable[[Candidate], Any] | None = None,
        reverse: bool = True,
    ) -> GroverResult:
        """Re-order candidates by score.

        Resolution order for sort key:
        1. *key* callable — custom sort function
        2. *operation* string — sort by that operation's score via ``score_for()``
        3. Default — ``candidate.score`` (most recent non-null detail score)
        """
        if key:
            sort_key = key
        elif operation:

            def sort_key(c: Candidate) -> float:
                return c.score_for(operation)  # type: ignore[arg-type]
        else:

            def sort_key(c: Candidate) -> float:
                return c.score

        return self._with_candidates(sorted(self.candidates, key=sort_key, reverse=reverse))

    def top(self, k: int, *, operation: str | None = None) -> GroverResult:
        """Top *k* candidates by score. *k* must be >= 1."""
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)
        sorted_result = self.sort(operation=operation)
        return sorted_result._with_candidates(sorted_result.candidates[:k])

    def filter(self, fn: Callable[[Candidate], bool]) -> GroverResult:
        """Keep candidates where *fn(candidate)* is truthy."""
        return self._with_candidates([c for c in self.candidates if fn(c)])

    def kinds(self, *kinds: str) -> GroverResult:
        """Filter candidates by kind."""
        kind_set = set(kinds)
        return self.filter(lambda c: c.kind in kind_set)

    # -------------------------------------------------------------------
    # Chain helpers
    # -------------------------------------------------------------------

    def _require_grover(self) -> Any:
        """Return the bound Grover instance or raise."""
        if self._grover is None:
            msg = "Chain methods require a bound Grover instance. This result was not returned by a Grover operation."
            raise RuntimeError(msg)
        return self._grover

    # -------------------------------------------------------------------
    # CRUD chain stubs
    # -------------------------------------------------------------------

    def read(self) -> GroverResult:
        """Chain: read content for all candidates (one batched query)."""
        return self._require_grover().read(candidates=self)

    def stat(self) -> GroverResult:
        """Chain: populate metadata on all candidates (one batched query)."""
        return self._require_grover().stat(candidates=self)

    def edit(
        self,
        old: str = "",
        new: str = "",
        replace_all: bool = False,
        edits: list[EditOperation] | None = None,
    ) -> GroverResult:
        """Chain: find-and-replace across all candidates."""
        return self._require_grover().edit(
            old=old,
            new=new,
            replace_all=replace_all,
            edits=edits,
            candidates=self,
        )

    def ls(self) -> GroverResult:
        """Chain: list children of each candidate directory.

        Named ``ls`` to avoid shadowing the ``list`` builtin, which Pydantic
        needs to resolve the ``candidates: list[Candidate]`` annotation.
        Calls the facade's ``ls()`` method under the hood.
        """
        return self._require_grover().ls(candidates=self)

    def delete(self) -> GroverResult:
        """Chain: delete all candidates (one batched query)."""
        return self._require_grover().delete(candidates=self)

    # -------------------------------------------------------------------
    # Query chain stubs
    # -------------------------------------------------------------------

    def glob(self, pattern: str) -> GroverResult:
        """Chain: glob within current candidates."""
        return self._require_grover().glob(pattern, candidates=self)

    def grep(self, pattern: str) -> GroverResult:
        """Chain: grep within current candidates."""
        return self._require_grover().grep(pattern, candidates=self)

    def semantic_search(self, query: str, *, k: int = 15) -> GroverResult:
        """Chain: semantic search filtered to current candidates."""
        return self._require_grover().semantic_search(query, k=k, candidates=self)

    def vector_search(self, vector: list[float], *, k: int = 15) -> GroverResult:
        """Chain: vector search filtered to current candidates."""
        return self._require_grover().vector_search(vector, k=k, candidates=self)

    def lexical_search(self, query: str, *, k: int = 15) -> GroverResult:
        """Chain: lexical search filtered to current candidates."""
        return self._require_grover().lexical_search(query, k=k, candidates=self)

    # -------------------------------------------------------------------
    # Graph chain stubs
    # -------------------------------------------------------------------

    def predecessors(self) -> GroverResult:
        """Chain: one-hop backward traversal."""
        return self._require_grover().predecessors(candidates=self)

    def successors(self) -> GroverResult:
        """Chain: one-hop forward traversal."""
        return self._require_grover().successors(candidates=self)

    def ancestors(self) -> GroverResult:
        """Chain: transitive backward traversal."""
        return self._require_grover().ancestors(candidates=self)

    def descendants(self) -> GroverResult:
        """Chain: transitive forward traversal."""
        return self._require_grover().descendants(candidates=self)

    def neighborhood(self, *, depth: int = 2) -> GroverResult:
        """Chain: bounded BFS around candidates."""
        return self._require_grover().neighborhood(candidates=self, depth=depth)

    def meeting_subgraph(self) -> GroverResult:
        """Chain: all paths between candidates."""
        return self._require_grover().meeting_subgraph(candidates=self)

    def min_meeting_subgraph(self) -> GroverResult:
        """Chain: minimum connecting subgraph."""
        return self._require_grover().min_meeting_subgraph(candidates=self)

    def pagerank(self) -> GroverResult:
        """Chain: rank candidates by PageRank."""
        return self._require_grover().pagerank(candidates=self)

    def betweenness_centrality(self) -> GroverResult:
        """Chain: rank candidates by betweenness centrality."""
        return self._require_grover().betweenness_centrality(candidates=self)

    def closeness_centrality(self) -> GroverResult:
        """Chain: rank candidates by closeness centrality."""
        return self._require_grover().closeness_centrality(candidates=self)

    def degree_centrality(self) -> GroverResult:
        """Chain: rank candidates by degree centrality."""
        return self._require_grover().degree_centrality(candidates=self)

    def in_degree_centrality(self) -> GroverResult:
        """Chain: rank candidates by in-degree centrality."""
        return self._require_grover().in_degree_centrality(candidates=self)

    def out_degree_centrality(self) -> GroverResult:
        """Chain: rank candidates by out-degree centrality."""
        return self._require_grover().out_degree_centrality(candidates=self)

    def hits(self) -> GroverResult:
        """Chain: rank candidates by HITS authority/hub scores."""
        return self._require_grover().hits(candidates=self)

"""New query response types for glob, grep, and search.

These are consistently-formatted, path-first result types that present
query results like search engine results: links with minimal previews,
not full content.

Evidence types (``LineMatch``, ``ChunkMatch``) carry the proof of why a
hit matched.  Hit types (``GlobHit``, ``GrepHit``, ``SearchHit``) are
path-first containers.  Result types (``GlobQueryResult``,
``GrepQueryResult``, ``SearchQueryResult``) wrap hits with metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

# =====================================================================
# Evidence types
# =====================================================================


@dataclass(frozen=True, slots=True)
class LineMatch:
    """A single line match within a file (grep evidence).

    Attributes:
        line_number: 1-indexed line number.
        line_content: The matched line text.
        context_before: Lines immediately before the match.
        context_after: Lines immediately after the match.
    """

    line_number: int
    line_content: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """A chunk match within a file (search evidence).

    Attributes:
        name: Symbol name (e.g. ``"MyClass.my_method"``).
        line_start: 1-indexed start line in the parent file.
        line_end: 1-indexed end line in the parent file.
        score: Similarity score for this chunk.
        snippet: First 200 characters of chunk content.
    """

    name: str
    line_start: int
    line_end: int
    score: float
    snippet: str = ""


# =====================================================================
# Hit types (all path-first)
# =====================================================================


@dataclass(frozen=True, slots=True)
class GlobHit:
    """A file or directory matched by glob.

    Attributes:
        path: Virtual path of the matched entry.
        is_directory: Whether this entry is a directory.
        size_bytes: File size in bytes (``None`` for directories).
        mime_type: MIME type (``None`` if unknown).
    """

    path: str
    is_directory: bool = False
    size_bytes: int | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class GrepHit:
    """A file containing grep matches.

    Attributes:
        path: Virtual path of the file.
        line_matches: Individual line matches within this file.
    """

    path: str
    line_matches: tuple[LineMatch, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A file matching a semantic query.

    Attributes:
        path: Virtual path of the file.
        score: Maximum chunk similarity score for this file.
        chunk_matches: Individual chunk matches within this file.
    """

    path: str
    score: float = 0.0
    chunk_matches: tuple[ChunkMatch, ...] = ()


# =====================================================================
# Result types (consistent shape)
# =====================================================================


@dataclass(frozen=True, slots=True)
class GlobQueryResult:
    """Result of a glob query.

    Attributes:
        success: Whether the query succeeded.
        message: Human-readable status message.
        hits: Matched files/directories.
        pattern: The glob pattern that was used.
        path: The search root path.
    """

    success: bool
    message: str
    hits: tuple[GlobHit, ...] = ()
    pattern: str = ""
    path: str = "/"


@dataclass(frozen=True, slots=True)
class GrepQueryResult:
    """Result of a grep query.

    Attributes:
        success: Whether the query succeeded.
        message: Human-readable status message.
        hits: Files containing matches, each with line evidence.
        pattern: The regex pattern that was used.
        path: The search root path.
        files_searched: Total number of files searched.
        files_matched: Number of files with at least one match.
        truncated: Whether results were truncated.
    """

    success: bool
    message: str
    hits: tuple[GrepHit, ...] = ()
    pattern: str = ""
    path: str = "/"
    files_searched: int = 0
    files_matched: int = 0
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class SearchQueryResult:
    """Result of a semantic search query.

    Attributes:
        success: Whether the query succeeded.
        message: Human-readable status message.
        hits: Files matching the query, each with chunk evidence.
        query: The natural-language query that was used.
        path: The search root path.
        files_matched: Number of files with at least one match.
        truncated: Whether results were truncated.
    """

    success: bool
    message: str
    hits: tuple[SearchHit, ...] = ()
    query: str = ""
    path: str = "/"
    files_matched: int = 0
    truncated: bool = False

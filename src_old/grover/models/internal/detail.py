"""Detail types — why a path appeared in a result and what happened to it.

``Detail`` extends the former ``Evidence`` base with ``success`` and
``message`` fields so that the same type can describe *search* hits
(``GlobDetail``, ``VectorDetail``, …) **and** *operation* outcomes
(``WriteDetail``, ``ReadDetail``, ``DeleteDetail``).

All detail types are frozen dataclasses.  They attach to ``File``,
``Directory``, ``FileChunk``, ``FileVersion``, and ``FileConnection``
objects.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Detail:
    """Base detail — why a file appeared in a result and what happened to it."""

    operation: str
    score: float = 0.0
    success: bool = True
    message: str = ""
    query_args: dict = field(default_factory=dict)


# =====================================================================
# Search details (renamed from *Evidence)
# =====================================================================


@dataclass(frozen=True, slots=True)
class LineMatch:
    """A single line match within a file."""

    line_number: int
    line_content: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GlobDetail(Detail):
    """Detail from a glob match."""

    is_directory: bool = False
    size_bytes: int | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class GrepDetail(Detail):
    """Detail from a grep match."""

    line_matches: tuple[LineMatch, ...] = ()


@dataclass(frozen=True, slots=True)
class TreeDetail(Detail):
    """Detail from a tree listing."""

    depth: int = 0
    is_directory: bool = False


@dataclass(frozen=True, slots=True)
class ListDirDetail(Detail):
    """Detail from a directory listing."""

    is_directory: bool = False
    size_bytes: int | None = None


# =====================================================================
# Trash / version / share details
# =====================================================================


@dataclass(frozen=True, slots=True)
class TrashDetail(Detail):
    """Detail from a trash listing."""

    deleted_at: datetime | None = None
    original_path: str = ""


@dataclass(frozen=True, slots=True)
class VersionDetail(Detail):
    """Detail from a version listing."""

    version: int = 0
    content_hash: str = ""
    size_bytes: int = 0
    created_at: datetime | None = None
    created_by: str | None = None


@dataclass(frozen=True, slots=True)
class ShareDetail(Detail):
    """Detail from a share listing."""

    grantee_id: str = ""
    permission: str = ""
    granted_by: str = ""
    expires_at: datetime | None = None


# =====================================================================
# Vector / lexical / hybrid details
# =====================================================================


@dataclass(frozen=True, slots=True)
class VectorDetail(Detail):
    """Detail from a vector (semantic) search."""

    snippet: str = ""


@dataclass(frozen=True, slots=True)
class LexicalDetail(Detail):
    """Detail from a lexical (BM25/full-text) search."""

    snippet: str = ""


@dataclass(frozen=True, slots=True)
class HybridDetail(Detail):
    """Detail from a hybrid search."""

    snippet: str = ""


# =====================================================================
# Graph details
# =====================================================================


@dataclass(frozen=True, slots=True)
class GraphRelationshipDetail(Detail):
    """Detail from a graph relationship query (predecessors, successors, ancestors, descendants).

    *paths* lists the candidate nodes this file is related to.
    """

    paths: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GraphCentralityDetail(Detail):
    """Detail from a graph centrality algorithm (pagerank, betweenness, etc.)."""

    scores: dict[str, float] = field(default_factory=dict)


# =====================================================================
# Operation details (new — carry per-file outcome info)
# =====================================================================


@dataclass(frozen=True, slots=True)
class WriteDetail(Detail):
    """Detail from a write/create/update operation."""

    version: int = 0


@dataclass(frozen=True, slots=True)
class ReadDetail(Detail):
    """Detail from a read operation."""


@dataclass(frozen=True, slots=True)
class MoveDetail(Detail):
    """Detail from a move operation."""

    source_path: str = ""
    version: int = 0


@dataclass(frozen=True, slots=True)
class CopyDetail(Detail):
    """Detail from a copy operation."""

    source_path: str = ""
    version: int = 0


@dataclass(frozen=True, slots=True)
class DeleteDetail(Detail):
    """Detail from a delete operation."""

    permanent: bool = False


@dataclass(frozen=True, slots=True)
class ReconcileDetail(Detail):
    """Detail from a reconcile operation."""

    action: str = ""  # "created" or "deleted"

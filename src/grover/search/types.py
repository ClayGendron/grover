"""Search layer data types — value objects for vectors, results, and index configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grover.ref import Ref


# ------------------------------------------------------------------
# Vector data
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorEntry:
    """A vector with its ID and metadata, ready for storage.

    Attributes:
        id: Unique identifier (file path in Grover).
        vector: Embedding vector.
        metadata: Arbitrary key-value metadata stored alongside the vector.
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextEntry:
    """A text entry for stores that embed on ingest (SupportsTextIngest).

    Attributes:
        id: Unique identifier.
        text: Raw text (the store/provider will embed it).
        metadata: Arbitrary key-value metadata.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Sparse vector representation for hybrid search.

    Attributes:
        indices: Non-zero dimension indices.
        values: Corresponding non-zero values.
    """

    indices: list[int]
    values: list[float]


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A single result from a VectorStore search.

    Attributes:
        id: Identifier of the matched entry.
        score: Similarity score (higher is more similar).
        metadata: Metadata stored with the vector.
        vector: The vector itself, if requested.
    """

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """Result of a vector upsert operation.

    Attributes:
        upserted_count: Number of vectors successfully upserted.
        errors: List of error messages for failed upserts.
    """

    upserted_count: int
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DeleteResult:
    """Result of a vector delete operation.

    Attributes:
        deleted_count: Number of vectors deleted.
    """

    deleted_count: int


# ------------------------------------------------------------------
# Index configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Configuration for creating a vector index.

    Attributes:
        name: Index name.
        dimension: Vector dimensionality.
        metric: Distance metric (cosine, euclidean, dotproduct).
        cloud_config: Provider-specific settings (e.g. serverless spec for Pinecone).
    """

    name: str
    dimension: int
    metric: str = "cosine"
    cloud_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexInfo:
    """Information about an existing vector index.

    Attributes:
        name: Index name.
        dimension: Vector dimensionality.
        metric: Distance metric.
        vector_count: Number of vectors in the index.
        metadata: Additional provider-specific info.
    """

    name: str
    dimension: int
    metric: str
    vector_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Grover user-facing search result
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result from Grover's search layer.

    This is the user-facing result type returned by ``Grover.search()`` and
    ``GroverAsync.search()``.  It wraps a :class:`~grover.ref.Ref` with the
    similarity score and matched content.

    Attributes:
        ref: Reference to the matched chunk/file.
        score: Cosine similarity (0-1, higher is more similar).
        content: The embedded text that matched.
        parent_path: Parent file path if the result is a chunk.
    """

    ref: Ref
    score: float
    content: str
    parent_path: str | None = None

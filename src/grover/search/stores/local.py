"""LocalVectorStore — in-process usearch HNSW vector store."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from usearch.index import Index

from grover.search.filters import FilterExpression, compile_dict
from grover.search.types import DeleteResult, UpsertResult, VectorEntry, VectorSearchResult

_INDEX_FILE = "search.usearch"
_META_FILE = "search_meta.json"


class LocalVectorStore:
    """In-process vector store backed by usearch HNSW index.

    Implements the ``VectorStore`` protocol for local development use.
    Does **not** support namespaces (raises ``ValueError`` if non-None is
    passed).  Supports simple equality-based metadata filtering.

    Thread-safe via :class:`threading.Lock`.
    """

    def __init__(self, *, dimension: int, metric: str = "cosine") -> None:
        usearch_metric = "cos" if metric == "cosine" else metric
        self._dimension = dimension
        self._metric = metric
        self._index_name = "local"

        self._index = Index(ndim=dimension, metric=usearch_metric, dtype="f32")
        self._lock = threading.Lock()
        self._next_key: int = 0

        # key → metadata (includes "id", "vector", plus any user metadata)
        self._key_to_meta: dict[int, dict[str, Any]] = {}
        # id → usearch key
        self._id_to_key: dict[str, int] = {}
        # parent_path → set of child IDs (for remove_file)
        self._parent_to_children: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def upsert(
        self,
        entries: list[VectorEntry],
        *,
        namespace: str | None = None,
    ) -> UpsertResult:
        """Insert or update vector entries."""
        self._reject_namespace(namespace)

        count = 0
        for entry in entries:
            # Remove old entry if it exists (deduplication)
            if entry.id in self._id_to_key:
                self._remove_by_id(entry.id)

            vector = np.array(entry.vector, dtype=np.float32)
            key = self._next_key
            self._next_key += 1

            with self._lock:
                self._index.add(key, vector)

            self._key_to_meta[key] = {
                "id": entry.id,
                "vector": entry.vector,
                **entry.metadata,
            }
            self._id_to_key[entry.id] = key

            # Track parent-child relationships
            parent = entry.metadata.get("parent_path")
            if parent is not None:
                self._parent_to_children.setdefault(parent, set()).add(entry.id)

            count += 1

        return UpsertResult(upserted_count=count)

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        namespace: str | None = None,
        filter: FilterExpression | None = None,  # noqa: A002
        include_metadata: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for the *k* nearest vectors."""
        self._reject_namespace(namespace)

        if len(self) == 0:
            return []

        query = np.array(vector, dtype=np.float32)

        # For filtered search, we may need to over-fetch to account for filtering
        effective_k = min(k * 3, len(self)) if filter is not None else min(k, len(self))

        with self._lock:
            matches = self._index.search(query, effective_k)

        # Compile filter to simple dict if provided
        filter_dict: dict[str, Any] | None = None
        if filter is not None:
            filter_dict = compile_dict(filter)

        results: list[VectorSearchResult] = []
        for match_key, distance in zip(
            matches.keys.tolist(), matches.distances.tolist(), strict=True
        ):
            meta = self._key_to_meta.get(int(match_key))
            if meta is None:
                continue

            # Apply metadata filter
            if filter_dict is not None and not all(
                meta.get(fk) == fv for fk, fv in filter_dict.items()
            ):
                continue

            score = 1.0 - distance

            if score_threshold is not None and score < score_threshold:
                continue

            result_meta = {mk: mv for mk, mv in meta.items() if mk not in ("id", "vector")}

            results.append(
                VectorSearchResult(
                    id=meta["id"],
                    score=score,
                    metadata=result_meta if include_metadata else {},
                    vector=meta.get("vector") if include_metadata else None,
                )
            )

            if len(results) >= k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> DeleteResult:
        """Delete vectors by their IDs."""
        self._reject_namespace(namespace)

        count = 0
        for entry_id in ids:
            if self._remove_by_id(entry_id):
                count += 1

        return DeleteResult(deleted_count=count)

    async def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> list[VectorEntry | None]:
        """Fetch vectors by their IDs."""
        self._reject_namespace(namespace)

        results: list[VectorEntry | None] = []
        for entry_id in ids:
            key = self._id_to_key.get(entry_id)
            if key is None:
                results.append(None)
                continue
            meta = self._key_to_meta.get(key)
            if meta is None:
                results.append(None)
                continue
            user_meta = {mk: mv for mk, mv in meta.items() if mk not in ("id", "vector")}
            results.append(
                VectorEntry(
                    id=meta["id"],
                    vector=meta.get("vector", []),
                    metadata=user_meta,
                )
            )
        return results

    async def connect(self) -> None:
        """No-op for local store."""

    async def close(self) -> None:
        """No-op for local store."""

    @property
    def index_name(self) -> str:
        """Return the index name."""
        return self._index_name

    # ------------------------------------------------------------------
    # Local-specific methods
    # ------------------------------------------------------------------

    def has(self, entry_id: str) -> bool:
        """Return whether *entry_id* is present in the store."""
        return entry_id in self._id_to_key

    def content_hash(self, entry_id: str) -> str | None:
        """Return the content hash for *entry_id*, or None if not stored."""
        key = self._id_to_key.get(entry_id)
        if key is None:
            return None
        meta = self._key_to_meta.get(key)
        if meta is None:
            return None
        return meta.get("content_hash")

    def remove_file(self, path: str) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches."""
        self._remove_by_id(path)
        children = self._parent_to_children.pop(path, set())
        for child_id in list(children):
            self._remove_by_id(child_id)

    def __len__(self) -> int:
        """Return the number of indexed entries."""
        return len(self._key_to_meta)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the index and metadata to *directory*."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        index_path = dir_path / _INDEX_FILE
        meta_path = dir_path / _META_FILE

        with self._lock:
            self._index.save(str(index_path))

        # Serialize metadata — strip vectors from sidecar (they're in usearch)
        serializable_meta: dict[str, dict[str, Any]] = {}
        for k, v in self._key_to_meta.items():
            entry = {mk: mv for mk, mv in v.items() if mk != "vector"}
            serializable_meta[str(k)] = entry

        sidecar: dict[str, Any] = {
            "next_key": self._next_key,
            "key_to_meta": serializable_meta,
            "path_to_keys": {eid: [key] for eid, key in self._id_to_key.items()},
        }
        with meta_path.open("w") as f:
            json.dump(sidecar, f)

    def load(self, directory: str) -> None:
        """Load a previously saved index from *directory*."""
        dir_path = Path(directory)
        index_path = dir_path / _INDEX_FILE
        meta_path = dir_path / _META_FILE

        with self._lock:
            self._index.load(str(index_path))

        with meta_path.open() as f:
            sidecar = json.load(f)

        self._next_key = sidecar["next_key"]
        self._key_to_meta = {}
        self._id_to_key = {}
        self._parent_to_children = {}

        raw_meta: dict[str, dict[str, Any]] = sidecar.get("key_to_meta", {})
        for k_str, meta in raw_meta.items():
            key = int(k_str)
            # Old format uses "path" as ID; new format uses "id"
            entry_id = meta.get("id", meta.get("path", ""))
            meta["id"] = entry_id
            self._key_to_meta[key] = meta
            self._id_to_key[entry_id] = key

            parent = meta.get("parent_path")
            if parent is not None:
                self._parent_to_children.setdefault(parent, set()).add(entry_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _remove_by_id(self, entry_id: str) -> bool:
        """Remove a single entry by ID. Returns True if found."""
        key = self._id_to_key.pop(entry_id, None)
        if key is None:
            return False
        meta = self._key_to_meta.pop(key, None)
        with self._lock:
            self._index.remove(key)

        # Clean up parent tracking
        if meta is not None:
            parent = meta.get("parent_path")
            if parent is not None and parent in self._parent_to_children:
                self._parent_to_children[parent].discard(entry_id)
                if not self._parent_to_children[parent]:
                    del self._parent_to_children[parent]
        return True

    @staticmethod
    def _reject_namespace(namespace: str | None) -> None:
        """Raise ValueError if a namespace is provided."""
        if namespace is not None:
            msg = "LocalVectorStore does not support namespaces"
            raise ValueError(msg)

"""LocalVectorStore — in-process usearch HNSW vector store."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from usearch.index import Index

from grover.models.internal.evidence import Evidence, VectorEvidence
from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileOperationResult, FileSearchResult, FileSearchSet
from grover.providers.search.filters import FilterExpression, compile_dict
from grover.providers.search.protocol import IndexConfig, parent_path_from_id

_INDEX_FILE = "search.usearch"
_META_FILE = "search_meta.json"


class LocalVectorStore:
    """In-process vector store backed by usearch HNSW index.

    Implements the ``SearchProvider`` protocol for local development use.
    Supports simple equality-based metadata filtering via extra kwargs
    on ``vector_search``.

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
    # SearchProvider protocol
    # ------------------------------------------------------------------

    async def create_index(self, config: IndexConfig) -> None:
        """No-op — local store creates the index in __init__."""

    async def upsert(self, *, files: list[File]) -> BatchResult:
        """Insert or update vector entries."""
        results: list[FileOperationResult] = []
        succeeded = 0

        for file in files:
            # Remove old entry if it exists (deduplication)
            if file.path in self._id_to_key:
                self._remove_by_id(file.path)

            vector = np.array(file.embedding, dtype=np.float32)
            key = self._next_key
            self._next_key += 1

            with self._lock:
                self._index.add(key, vector)

            self._key_to_meta[key] = {
                "id": file.path,
                "vector": file.embedding,
            }
            self._id_to_key[file.path] = key

            # Track parent-child relationships for chunk entries
            if "#" in file.path:
                parent = parent_path_from_id(file.path)
                self._parent_to_children.setdefault(parent, set()).add(file.path)

            results.append(FileOperationResult(file=File(path=file.path), success=True))
            succeeded += 1

        return BatchResult(
            results=results,
            succeeded=succeeded,
            failed=0,
            success=True,
            message=f"Upserted {succeeded} entries",
        )

    async def vector_search(
        self,
        vector: list[float],
        *,
        k: int = 10,
        candidates: FileSearchSet | None = None,
        filter: FilterExpression | None = None,  # noqa: A002
        score_threshold: float | None = None,
    ) -> FileSearchResult:
        """Search for the *k* nearest vectors, returning a ``FileSearchResult``."""
        if len(self) == 0:
            return FileSearchResult(success=True, message="No entries indexed")

        query = np.array(vector, dtype=np.float32)

        # For filtered search, we may need to over-fetch to account for filtering
        effective_k = min(k * 3, len(self)) if filter is not None else min(k, len(self))

        with self._lock:
            matches = self._index.search(query, effective_k)

        # Compile filter to simple dict if provided
        filter_dict: dict[str, Any] | None = None
        if filter is not None:
            filter_dict = compile_dict(filter)

        # Candidate set for post-filtering
        allowed: set[str] | None = None
        if candidates is not None:
            allowed = set(candidates.paths)

        hits: list[tuple[str, float]] = []
        for match_key, distance in zip(matches.keys.tolist(), matches.distances.tolist(), strict=True):
            meta = self._key_to_meta.get(int(match_key))
            if meta is None:
                continue

            # Apply metadata filter
            if filter_dict is not None and not all(meta.get(fk) == fv for fk, fv in filter_dict.items()):
                continue

            score = 1.0 - distance

            if score_threshold is not None and score < score_threshold:
                continue

            entry_id = meta["id"]
            fp = parent_path_from_id(entry_id)

            if allowed is not None and fp not in allowed:
                continue

            hits.append((fp, score))

            if len(hits) >= k:
                break

        hits.sort(key=lambda r: r[1], reverse=True)

        # Wrap into FileSearchResult
        entries: dict[str, list[Evidence]] = {}
        for fp, _score in hits:
            ev = VectorEvidence(operation="vector_search", snippet="")
            entries.setdefault(fp, []).append(ev)

        result = FileSearchResult(
            success=True,
            message=f"Found matches in {len(entries)} file(s)",
            files=[File(path=fp, evidence=evs) for fp, evs in entries.items()],
        )
        return result

    async def delete(self, *, files: list[str]) -> BatchResult:
        """Delete vectors by their IDs."""
        results: list[FileOperationResult] = []
        succeeded = 0

        for entry_id in files:
            if self._remove_by_id(entry_id):
                results.append(FileOperationResult(file=File(path=entry_id), success=True))
                succeeded += 1
            else:
                results.append(FileOperationResult(file=File(path=entry_id), success=False, message="Not found"))

        return BatchResult(
            results=results,
            succeeded=succeeded,
            failed=len(files) - succeeded,
            success=True,
            message=f"Deleted {succeeded} of {len(files)} entries",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """No-op for local store."""

    async def close(self) -> None:
        """No-op for local store."""

    @property
    def index_name(self) -> str:
        """Return the index name."""
        return self._index_name

    @property
    def dimension(self) -> int:
        """Return the vector dimension this store was created with."""
        return self._dimension

    # ------------------------------------------------------------------
    # Local-specific methods
    # ------------------------------------------------------------------

    def has(self, entry_id: str) -> bool:
        """Return whether *entry_id* is present in the store."""
        return entry_id in self._id_to_key

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

            # Rebuild parent tracking from ID format
            if "#" in entry_id:
                parent = parent_path_from_id(entry_id)
                self._parent_to_children.setdefault(parent, set()).add(entry_id)
            else:
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
        self._key_to_meta.pop(key, None)
        with self._lock:
            self._index.remove(key)

        # Clean up parent tracking
        if "#" in entry_id:
            parent = parent_path_from_id(entry_id)
            if parent in self._parent_to_children:
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

"""SearchIndex — usearch HNSW index with metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from usearch.index import Index

from grover.ref import Ref

if TYPE_CHECKING:
    from grover.search.extractors import EmbeddableChunk
    from grover.search.providers._protocol import EmbeddingProvider

_INDEX_FILE = "search.usearch"
_META_FILE = "search_meta.json"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result from the vector index.

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


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class SearchIndex:
    """In-memory HNSW vector index backed by usearch.

    Wraps a usearch ``Index`` with path-based metadata so callers can
    add, remove, and search by file/chunk paths.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._index = Index(ndim=provider.dimensions, metric="cos")
        self._next_key: int = 0
        # key -> metadata
        self._key_to_meta: dict[int, dict] = {}
        # path -> list of usearch keys
        self._path_to_keys: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        path: str,
        content: str,
        parent_path: str | None = None,
    ) -> int:
        """Embed *content* and insert it into the index.

        If *path* already exists it is removed first so the index stays
        deduplicated.
        """
        if path in self._path_to_keys:
            self.remove(path)

        vector = np.array(self._provider.embed(content), dtype=np.float32)
        key = self._next_key
        self._next_key += 1

        self._index.add(key, vector)
        self._key_to_meta[key] = {
            "path": path,
            "content": content,
            "parent_path": parent_path,
            "content_hash": _content_hash(content),
        }
        self._path_to_keys.setdefault(path, []).append(key)
        return key

    def add_batch(self, entries: list[EmbeddableChunk]) -> list[int]:
        """Embed and insert multiple entries at once."""
        if not entries:
            return []

        # Remove existing entries for paths that will be re-added
        for entry in entries:
            if entry.path in self._path_to_keys:
                self.remove(entry.path)

        texts = [e.content for e in entries]
        vectors = np.array(
            self._provider.embed_batch(texts), dtype=np.float32
        )

        keys: list[int] = []
        for i, entry in enumerate(entries):
            key = self._next_key
            self._next_key += 1
            self._index.add(key, vectors[i])
            self._key_to_meta[key] = {
                "path": entry.path,
                "content": entry.content,
                "parent_path": entry.parent_path,
                "content_hash": _content_hash(entry.content),
            }
            self._path_to_keys.setdefault(entry.path, []).append(key)
            keys.append(key)

        return keys

    def remove(self, path: str) -> None:
        """Remove all entries for *path* from the index."""
        keys = self._path_to_keys.pop(path, [])
        for key in keys:
            self._index.remove(key)
            self._key_to_meta.pop(key, None)

    def remove_file(self, path: str) -> None:
        """Remove *path* **and** all entries whose ``parent_path`` matches."""
        self.remove(path)
        children = [
            meta["path"]
            for meta in self._key_to_meta.values()
            if meta.get("parent_path") == path
        ]
        for child_path in children:
            self.remove(child_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to *k* results ranked by cosine similarity."""
        if len(self) == 0:
            return []

        vector = np.array(self._provider.embed(query), dtype=np.float32)
        effective_k = min(k, len(self))
        matches = self._index.search(vector, effective_k)

        results: list[SearchResult] = []
        for key, distance in zip(
            matches.keys.tolist(), matches.distances.tolist(), strict=True
        ):
            meta = self._key_to_meta.get(int(key))
            if meta is None:
                continue
            results.append(
                SearchResult(
                    ref=Ref(path=meta["path"]),
                    score=1.0 - distance,
                    content=meta["content"],
                    parent_path=meta.get("parent_path"),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def has(self, path: str) -> bool:
        """Return whether *path* is present in the index."""
        return path in self._path_to_keys

    def content_hash(self, path: str) -> str | None:
        """Return the content hash for *path*, or ``None`` if not indexed."""
        keys = self._path_to_keys.get(path)
        if not keys:
            return None
        meta = self._key_to_meta.get(keys[0])
        if meta is None:
            return None
        return meta["content_hash"]

    def __len__(self) -> int:
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

        self._index.save(str(index_path))

        sidecar = {
            "next_key": self._next_key,
            "key_to_meta": {str(k): v for k, v in self._key_to_meta.items()},
            "path_to_keys": self._path_to_keys,
        }
        with meta_path.open("w") as f:
            json.dump(sidecar, f)

    def load(self, directory: str) -> None:
        """Load a previously saved index from *directory*."""
        dir_path = Path(directory)
        index_path = dir_path / _INDEX_FILE
        meta_path = dir_path / _META_FILE

        self._index.load(str(index_path))

        with meta_path.open() as f:
            sidecar = json.load(f)

        self._next_key = sidecar["next_key"]
        self._key_to_meta = {
            int(k): v for k, v in sidecar["key_to_meta"].items()
        }
        self._path_to_keys = dict(sidecar["path_to_keys"])

    @classmethod
    def from_directory(
        cls, directory: str, provider: EmbeddingProvider
    ) -> SearchIndex:
        """Load an existing index from *directory*."""
        instance = cls(provider)
        instance.load(directory)
        return instance

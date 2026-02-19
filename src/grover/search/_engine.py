"""SearchEngine — orchestrator wiring EmbeddingProvider + VectorStore."""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import TYPE_CHECKING

from grover.ref import Ref
from grover.search.types import SearchResult, VectorEntry

if TYPE_CHECKING:
    from grover.search.extractors import EmbeddableChunk
    from grover.search.protocols import EmbeddingProvider, VectorStore
    from grover.search.stores.local import LocalVectorStore

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class SearchEngine:
    """Orchestrates :class:`EmbeddingProvider` and :class:`VectorStore`.

    The engine is the high-level interface that :class:`GroverAsync` uses
    for search.  It embeds text via the provider, stores vectors in the
    store, and converts between Grover-level types
    (``EmbeddableChunk`` / ``SearchResult``) and store-level types
    (``VectorEntry`` / ``VectorSearchResult``).
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._store = store
        self._embedding_provider = embedding_provider

    # ------------------------------------------------------------------
    # High-level operations (what GroverAsync calls)
    # ------------------------------------------------------------------

    async def add(
        self,
        path: str,
        content: str,
        *,
        parent_path: str | None = None,
    ) -> None:
        """Embed *content* and upsert to the store."""
        if self._embedding_provider is None:
            msg = "Cannot add: no embedding provider configured"
            raise RuntimeError(msg)

        vector = await self._embed(content)
        entry = VectorEntry(
            id=path,
            vector=vector,
            metadata={
                "content": content,
                "parent_path": parent_path,
                "content_hash": _content_hash(content),
            },
        )
        await self._store.upsert([entry])

    async def add_batch(self, entries: list[EmbeddableChunk]) -> None:
        """Embed a batch of entries and upsert to the store."""
        if not entries:
            return
        if self._embedding_provider is None:
            msg = "Cannot add_batch: no embedding provider configured"
            raise RuntimeError(msg)

        texts = [e.content for e in entries]
        vectors = await self._embed_batch(texts)

        vector_entries = [
            VectorEntry(
                id=entry.path,
                vector=vectors[i],
                metadata={
                    "content": entry.content,
                    "parent_path": entry.parent_path,
                    "content_hash": _content_hash(entry.content),
                },
            )
            for i, entry in enumerate(entries)
        ]
        await self._store.upsert(vector_entries)

    async def remove(self, path: str) -> None:
        """Remove a single entry by path."""
        await self._store.delete([path])

    async def remove_file(self, path: str) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches.

        For a :class:`LocalVectorStore`, uses the efficient ``remove_file``
        method.  For other stores, falls back to deleting just the path.
        """
        local_store = self._get_local_store()
        if local_store is not None:
            local_store.remove_file(path)
        else:
            # For non-local stores, delete the path itself.
            # Child cleanup requires metadata filter support, which
            # will be handled in Phase 4+ store implementations.
            await self._store.delete([path])

    async def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Embed *query*, search the store, and return Grover-level results."""
        if self._embedding_provider is None:
            msg = "Cannot search: no embedding provider configured"
            raise RuntimeError(msg)

        vector = await self._embed(query)
        vs_results = await self._store.search(vector, k=k)

        return [
            SearchResult(
                ref=Ref(path=vsr.id),
                score=vsr.score,
                content=vsr.metadata.get("content", ""),
                parent_path=vsr.metadata.get("parent_path"),
            )
            for vsr in vs_results
        ]

    # ------------------------------------------------------------------
    # Passthrough helpers
    # ------------------------------------------------------------------

    def has(self, path: str) -> bool:
        """Return whether *path* is present in the store."""
        local = self._get_local_store()
        if local is not None:
            return local.has(path)
        return False

    def content_hash(self, path: str) -> str | None:
        """Return the content hash for *path*, or None if not stored."""
        local = self._get_local_store()
        if local is not None:
            return local.content_hash(path)
        return None

    def __len__(self) -> int:
        """Return the number of indexed entries."""
        local = self._get_local_store()
        if local is not None:
            return len(local)
        return 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the store to *directory* (if the store supports it)."""
        save_fn = getattr(self._store, "save", None)
        if save_fn is not None:
            save_fn(directory)

    def load(self, directory: str) -> None:
        """Load the store from *directory* (if the store supports it)."""
        load_fn = getattr(self._store, "load", None)
        if load_fn is not None:
            load_fn(directory)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect the underlying store."""
        await self._store.connect()

    async def close(self) -> None:
        """Close the underlying store."""
        await self._store.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> VectorStore:
        """Return the underlying :class:`VectorStore`."""
        return self._store

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """Return the :class:`EmbeddingProvider`, if any."""
        return self._embedding_provider

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """Embed a single text, handling both sync and async providers."""
        assert self._embedding_provider is not None
        result = self._embedding_provider.embed(text)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling both sync and async providers."""
        assert self._embedding_provider is not None
        result = self._embedding_provider.embed_batch(texts)
        if inspect.isawaitable(result):
            return await result
        return result

    def _get_local_store(self) -> LocalVectorStore | None:
        """Return the store as a LocalVectorStore if it is one, else None."""
        from grover.search.stores.local import LocalVectorStore

        if isinstance(self._store, LocalVectorStore):
            return self._store
        return None

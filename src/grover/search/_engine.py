"""SearchEngine — orchestrator wiring EmbeddingProvider + VectorStore."""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import TYPE_CHECKING

from grover.ref import Ref
from grover.search.types import SearchResult, VectorEntry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.search.extractors import EmbeddableChunk
    from grover.search.fulltext.protocol import FullTextStore
    from grover.search.fulltext.types import FullTextResult
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
    (``VectorEntry`` / ``VectorHit``).
    """

    def __init__(
        self,
        *,
        vector: VectorStore | None = None,
        embedding: EmbeddingProvider | None = None,
        lexical: FullTextStore | None = None,
        hybrid: object | None = None,
    ) -> None:
        self._store: VectorStore | None = vector
        self._embedding_provider = embedding
        self._lexical = lexical
        self._hybrid = hybrid

    # ------------------------------------------------------------------
    # High-level operations (what GroverAsync calls)
    # ------------------------------------------------------------------

    async def add(
        self,
        path: str,
        content: str,
        *,
        parent_path: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed *content* and upsert to the store.  Also indexes in FTS."""
        if self._store is not None and self._embedding_provider is not None:
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

        if self._lexical is not None:
            await self._lexical.index(path, content, session=session)

    async def add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed a batch of entries and upsert to the store.  Also indexes in FTS."""
        if not entries:
            return

        if self._store is not None and self._embedding_provider is not None:
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
                        "chunk_name": entry.chunk_name,
                        "line_start": entry.line_start,
                        "line_end": entry.line_end,
                    },
                )
                for i, entry in enumerate(entries)
            ]
            await self._store.upsert(vector_entries)

        if self._lexical is not None:
            for entry in entries:
                await self._lexical.index(entry.path, entry.content, session=session)

    async def remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single entry by path."""
        if self._store is not None:
            await self._store.delete([path])
        if self._lexical is not None:
            await self._lexical.remove(path, session=session)

    async def remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches.

        For a :class:`LocalVectorStore`, uses the efficient ``remove_file``
        method.  For other stores, falls back to deleting just the path.
        Also removes entries from the FTS index.
        """
        if self._store is not None:
            local_store = self._get_local_store()
            if local_store is not None:
                local_store.remove_file(path)
            else:
                await self._store.delete([path])

        if self._lexical is not None:
            await self._lexical.remove_file(path, session=session)

    async def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Embed *query*, search the store, and return Grover-level results."""
        if self._embedding_provider is None:
            msg = "Cannot search: no embedding provider configured"
            raise RuntimeError(msg)

        vector = await self._embed(query)
        assert self._store is not None
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

    async def lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[FullTextResult]:
        """Search the FTS index using BM25 ranking.

        Raises
        ------
        RuntimeError
            If no lexical store is configured.
        """
        if self._lexical is None:
            msg = "Cannot lexical_search: no lexical store configured"
            raise RuntimeError(msg)
        return await self._lexical.search(query, k=k, session=session)

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
        if self._store is not None:
            await self._store.connect()

    async def close(self) -> None:
        """Close the underlying store."""
        if self._store is not None:
            await self._store.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vector(self) -> VectorStore | None:
        """Return the underlying :class:`VectorStore`."""
        return self._store

    @property
    def embedding(self) -> EmbeddingProvider | None:
        """Return the :class:`EmbeddingProvider`, if any."""
        return self._embedding_provider

    @property
    def lexical(self) -> FullTextStore | None:
        """Return the lexical (full-text) store, if any."""
        return self._lexical

    def supported_protocols(self) -> set[type]:
        """Return mount-level dispatch protocols based on configured components.

        Used by :class:`~grover.mount.Mount` to build the dispatch map.
        """
        from grover.mount.protocols import (
            SupportsEmbedding,
            SupportsHybridSearch,
            SupportsLexicalSearch,
            SupportsVectorSearch,
        )

        protos: set[type] = set()
        if self._store is not None and self._embedding_provider is not None:
            protos.add(SupportsVectorSearch)
        if self._lexical is not None:
            protos.add(SupportsLexicalSearch)
        if self._hybrid is not None:
            protos.add(SupportsHybridSearch)
        if self._embedding_provider is not None:
            protos.add(SupportsEmbedding)
        return protos

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

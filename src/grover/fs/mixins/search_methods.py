"""SearchMethodsMixin — search orchestration for DatabaseFileSystem.

Absorbs the logic from ``SearchEngine`` into filesystem-level methods.
No ``FullTextStore`` — lexical search queries the DB directly using
dialect-appropriate FTS when available, with LIKE fallback.
"""

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
    from grover.search.stores.local import LocalVectorStore

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class SearchMethodsMixin:
    """Search orchestration — embeds, upserts, queries.

    Requires ``self.search_provider`` (:class:`VectorStore` or ``None``)
    and ``self.embedding_provider`` (:class:`EmbeddingProvider` or ``None``)
    to be set on the owning class.

    Dimension validation runs at init time when both providers are set.
    """

    def _validate_search_dimensions(self) -> None:
        """Check that embedding dimensions match search store dimensions."""
        embedding = getattr(self, "embedding_provider", None)
        search = getattr(self, "search_provider", None)
        if embedding is not None and search is not None:
            store_dim = getattr(search, "dimension", None)
            if store_dim is not None and embedding.dimensions != store_dim:
                msg = (
                    f"Dimension mismatch: embedding provider "
                    f"'{embedding.model_name}' produces "
                    f"{embedding.dimensions}-dim vectors, but vector "
                    f"store expects {store_dim}-dim"
                )
                raise ValueError(msg)

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    async def search_add(
        self,
        path: str,
        content: str,
        *,
        parent_path: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed *content* and upsert to the vector store."""
        search = getattr(self, "search_provider", None)
        embedding = getattr(self, "embedding_provider", None)

        if search is not None and embedding is not None:
            vector = await self._search_embed(content)
            entry = VectorEntry(
                id=path,
                vector=vector,
                metadata={
                    "content": content,
                    "parent_path": parent_path,
                    "content_hash": _content_hash(content),
                },
            )
            await search.upsert([entry])

    async def search_add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed a batch of entries and upsert to the vector store."""
        if not entries:
            return

        search = getattr(self, "search_provider", None)
        embedding = getattr(self, "embedding_provider", None)

        if search is not None and embedding is not None:
            texts = [e.content for e in entries]
            vectors = await self._search_embed_batch(texts)

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
            await search.upsert(vector_entries)

    async def search_remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single entry by path from the vector store."""
        search = getattr(self, "search_provider", None)
        if search is not None:
            await search.delete([path])

    async def search_remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches."""
        search = getattr(self, "search_provider", None)
        if search is not None:
            local_store = self._search_get_local_store()
            if local_store is not None:
                local_store.remove_file(path)
            else:
                await search.delete([path])

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def search_query(self, query: str, k: int = 10) -> list[SearchResult]:
        """Embed *query*, search the vector store, return results."""
        embedding = getattr(self, "embedding_provider", None)
        search = getattr(self, "search_provider", None)

        if embedding is None:
            msg = "Cannot search: no embedding provider configured"
            raise RuntimeError(msg)
        if search is None:
            msg = "Cannot search: no search provider configured"
            raise RuntimeError(msg)

        vector = await self._search_embed(query)
        vs_results = await search.search(vector, k=k)

        return [
            SearchResult(
                ref=Ref(path=vsr.id),
                score=vsr.score,
                content=vsr.metadata.get("content", ""),
                parent_path=vsr.metadata.get("parent_path"),
            )
            for vsr in vs_results
        ]

    async def lexical_search_query(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SearchResult]:
        """Dialect-aware full-text search against DB content.

        Uses FTS5 (SQLite), tsvector (PostgreSQL), FREETEXT (MSSQL),
        or LIKE fallback for other dialects.
        """
        from sqlalchemy import text
        from sqlmodel import select

        sess = self._require_session(session)  # type: ignore[attr-defined]
        dialect = getattr(self, "dialect", "sqlite")
        model = self._file_model  # type: ignore[attr-defined]

        results: list[SearchResult] = []

        if dialect == "sqlite":
            # Try FTS5 virtual table
            try:
                table_name = getattr(model, "__tablename__", "grover_files")
                fts_table = f"{table_name}_fts"
                stmt = text(
                    f"SELECT path, content FROM {fts_table} WHERE {fts_table} MATCH :query LIMIT :k"
                )
                rows = await sess.execute(stmt, {"query": query, "k": k})
                results.extend(
                    SearchResult(
                        ref=Ref(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                # FTS table may not exist — fall through to LIKE
                logger.debug("FTS5 not available, falling back to LIKE")

        elif dialect == "postgresql":
            try:
                stmt = (
                    select(model.path, model.content)
                    .where(
                        text("content_tsv @@ plainto_tsquery('english', :query)"),
                        model.deleted_at.is_(None),
                    )
                    .limit(k)
                )
                rows = await sess.execute(stmt, {"query": query})
                results.extend(
                    SearchResult(
                        ref=Ref(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                logger.debug("PostgreSQL FTS not available, falling back to LIKE")

        elif dialect == "mssql":
            try:
                stmt = (
                    select(model.path, model.content)
                    .where(
                        text("FREETEXT(content, :query)"),
                        model.deleted_at.is_(None),
                    )
                    .limit(k)
                )
                rows = await sess.execute(stmt, {"query": query})
                results.extend(
                    SearchResult(
                        ref=Ref(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                logger.debug("MSSQL FTS not available, falling back to LIKE")

        # Fallback: LIKE search
        like_pattern = f"%{query}%"
        stmt = (
            select(model.path, model.content)
            .where(
                model.content.like(like_pattern),
                model.deleted_at.is_(None),
                model.is_directory.is_(False),
            )
            .limit(k)
        )
        rows = await sess.execute(stmt)
        results.extend(
            SearchResult(
                ref=Ref(path=row[0]),
                score=0.5,
                content=row[1] or "",
            )
            for row in rows
        )
        return results

    # ------------------------------------------------------------------
    # Passthrough helpers (LocalVectorStore)
    # ------------------------------------------------------------------

    def search_has(self, path: str) -> bool:
        """Return whether *path* is present in the local vector store."""
        local = self._search_get_local_store()
        if local is not None:
            return local.has(path)
        return False

    def search_content_hash(self, path: str) -> str | None:
        """Return the content hash for *path*, or None."""
        local = self._search_get_local_store()
        if local is not None:
            return local.content_hash(path)
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def search_save(self, directory: str) -> None:
        """Persist the vector store to *directory* (if supported)."""
        search = getattr(self, "search_provider", None)
        save_fn = getattr(search, "save", None)
        if save_fn is not None:
            save_fn(directory)

    def search_load(self, directory: str) -> None:
        """Load the vector store from *directory* (if supported)."""
        search = getattr(self, "search_provider", None)
        load_fn = getattr(search, "load", None)
        if load_fn is not None:
            load_fn(directory)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def search_connect(self) -> None:
        """Connect the underlying vector store."""
        search = getattr(self, "search_provider", None)
        if search is not None:
            await search.connect()

    async def search_close(self) -> None:
        """Close the underlying vector store."""
        search = getattr(self, "search_provider", None)
        if search is not None:
            await search.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _search_embed(self, text: str) -> list[float]:
        """Embed a single text, handling both sync and async providers."""
        embedding = self.embedding_provider  # type: ignore[attr-defined]
        assert embedding is not None
        result = embedding.embed(text)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _search_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling both sync and async providers."""
        embedding = self.embedding_provider  # type: ignore[attr-defined]
        assert embedding is not None
        result = embedding.embed_batch(texts)
        if inspect.isawaitable(result):
            return await result
        return result

    def _search_get_local_store(self) -> LocalVectorStore | None:
        """Return the store as a LocalVectorStore if it is one."""
        from grover.search.stores.local import LocalVectorStore

        search = getattr(self, "search_provider", None)
        if isinstance(search, LocalVectorStore):
            return search
        return None

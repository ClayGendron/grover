"""GroverRetriever — LangChain retriever backed by Grover semantic search."""

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from grover._grover import Grover


class GroverRetriever(BaseRetriever):
    """LangChain retriever backed by Grover's semantic search.

    Each :class:`~grover.search._index.SearchResult` is converted to a
    LangChain :class:`~langchain_core.documents.Document` with metadata
    containing the file path, similarity score, version, and line range.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverRetriever

        g = Grover(embedding_provider=provider)
        g.mount("/project", backend)
        g.index()

        retriever = GroverRetriever(grover=g, k=5)
        docs = retriever.invoke("authentication flow")

    The retriever works in any LangChain chain::

        from langchain_core.runnables import RunnablePassthrough

        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grover: Grover
    """The Grover instance to search against."""

    k: int = 10
    """Maximum number of results to return."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """Search Grover's vector index and return matching documents.

        Returns an empty list when the search index is not available
        (e.g. no embedding provider configured).
        """
        try:
            results = self.grover.search(query, k=self.k)
        except Exception:
            return []

        return [self._to_document(sr) for sr in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """Async variant — delegates to sync via thread executor."""
        return await asyncio.to_thread(
            self._get_relevant_documents, query, run_manager=run_manager
        )

    @staticmethod
    def _to_document(sr: Any) -> Document:
        """Convert a SearchResult to a LangChain Document."""
        metadata: dict[str, object] = {
            "path": sr.ref.path,
            "score": sr.score,
        }
        if sr.ref.version is not None:
            metadata["version"] = sr.ref.version
        if sr.parent_path is not None:
            metadata["parent_path"] = sr.parent_path
        if sr.ref.line_start is not None:
            metadata["line_start"] = sr.ref.line_start
        if sr.ref.line_end is not None:
            metadata["line_end"] = sr.ref.line_end

        return Document(
            page_content=sr.content,
            metadata=metadata,
            id=sr.ref.path,
        )

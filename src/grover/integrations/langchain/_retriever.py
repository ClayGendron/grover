"""GroverRetriever — LangChain retriever backed by Grover semantic search."""

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from grover._grover import Grover


class GroverRetriever(BaseRetriever):
    """LangChain retriever backed by Grover's semantic search.

    Each matched file path is converted to a LangChain
    :class:`~langchain_core.documents.Document` with metadata
    containing the file path and vector evidence snippets.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverRetriever

        g = Grover(embedding_provider=provider)
        g.add_mount("/project", backend)
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
            result = self.grover.vector_search(query, k=self.k)
        except Exception:
            return []

        if not result.success:
            return []

        return [self._path_to_document(path, result) for path in result.paths]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """Async variant — delegates to sync via thread executor."""
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)

    @staticmethod
    def _path_to_document(path: str, result: Any) -> Document:
        """Convert a path from VectorSearchResult to a LangChain Document."""
        metadata: dict[str, object] = {
            "path": path,
        }
        # Build page_content from vector evidence snippets
        snippets = list(result.snippets(path))
        if snippets:
            metadata["chunks"] = len(snippets)
        page_content = "\n\n".join(snippets) if snippets else path

        return Document(
            page_content=page_content,
            metadata=metadata,
            id=path,
        )

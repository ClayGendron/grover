"""GroverRetriever — LangChain retriever backed by Grover semantic search."""

import asyncio
from typing import TYPE_CHECKING, Union, cast

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from grover.client import Grover, GroverAsync

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )


class GroverRetriever(BaseRetriever):
    """LangChain retriever backed by Grover's semantic search.

    Accepts either a sync :class:`~grover.Grover` or async
    :class:`~grover.GroverAsync` instance:

    - **Grover:** ``_get_relevant_documents`` works directly;
      ``_aget_relevant_documents`` raises ``TypeError``.
    - **GroverAsync:** ``_aget_relevant_documents`` calls native async API;
      ``_get_relevant_documents`` wraps via ``asyncio.run()``.

    Usage::

        from grover import Grover, GroverAsync
        from grover.integrations.langchain import GroverRetriever

        # Sync
        g = Grover(embedding_provider=provider)
        g.add_mount("project", backend)
        g.index()
        retriever = GroverRetriever(grover=g, k=5)
        docs = retriever.invoke("authentication flow")

        # Async
        ga = GroverAsync(embedding_provider=provider)
        await ga.add_mount("project", backend)
        await ga.index()
        retriever = GroverRetriever(grover=ga, k=5)
        docs = await retriever.ainvoke("authentication flow")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grover: Union[Grover, GroverAsync]  # noqa: UP007
    """The Grover instance to search against."""

    k: int = 10
    """Maximum number of results to return."""

    @property
    def _is_async(self) -> bool:
        return isinstance(self.grover, GroverAsync)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun | None" = None,
    ) -> list[Document]:
        """Search Grover's vector index and return matching documents.

        Returns an empty list when the search index is not available
        (e.g. no embedding provider configured).
        """
        if self._is_async:
            return asyncio.run(self._aget_relevant_documents(query, run_manager=None))

        g = cast("Grover", self.grover)
        try:
            result = g.vector_search(query, k=self.k)
        except Exception:
            return []

        if not result.success:
            return []

        return [self._file_to_document(f) for f in result.files]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "AsyncCallbackManagerForRetrieverRun | None" = None,
    ) -> list[Document]:
        """Async variant — native async when GroverAsync, TypeError otherwise."""
        if not self._is_async:
            raise TypeError(
                "Async methods require GroverAsync. Pass a GroverAsync instance or use sync methods instead."
            )

        g = cast("GroverAsync", self.grover)
        try:
            result = await g.vector_search(query, k=self.k)
        except Exception:
            return []

        if not result.success:
            return []

        return [self._file_to_document(f) for f in result.files]

    @staticmethod
    def _file_to_document(f: object) -> Document:
        """Convert a File from FileSearchResult to a LangChain Document."""
        from grover.models.internal.evidence import VectorEvidence

        path: str = f.path  # type: ignore[union-attr]
        metadata: dict[str, object] = {"path": path}
        # Build page_content from vector evidence snippets
        snippets = [
            ev.snippet
            for ev in f.evidence  # type: ignore[union-attr]
            if isinstance(ev, VectorEvidence) and ev.snippet
        ]
        if snippets:
            metadata["chunks"] = len(snippets)
        page_content = "\n\n".join(snippets) if snippets else path

        return Document(
            page_content=page_content,
            metadata=metadata,
            id=path,
        )

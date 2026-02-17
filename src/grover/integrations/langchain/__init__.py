"""Grover integration with LangChain and LangGraph.

Provides ``GroverRetriever`` (LangChain retriever backed by semantic search),
``GroverLoader`` (document loader for RAG ingestion), and optionally
``GroverStore`` (LangGraph persistent memory store).

Usage::

    from grover import Grover
    from grover.integrations.langchain import GroverRetriever, GroverLoader

    g = Grover(embedding_provider=provider)
    g.mount("/project", backend)

    retriever = GroverRetriever(grover=g, k=5)
    docs = retriever.invoke("search query")

    loader = GroverLoader(grover=g, path="/project")
    docs = loader.load()
"""

try:
    from langchain_core.retrievers import BaseRetriever as _BaseRetriever
except ImportError as _exc:
    raise ImportError(
        "langchain-core is required for the Grover LangChain integration. "
        "Install it with: pip install 'grover[langchain]'"
    ) from _exc

from grover.integrations.langchain._loader import GroverLoader
from grover.integrations.langchain._retriever import GroverRetriever

__all__ = ["GroverLoader", "GroverRetriever"]

# GroverStore is conditional on langgraph
try:
    from grover.integrations.langchain._store import GroverStore

    __all__ = [*__all__, "GroverStore"]
except ImportError:
    pass

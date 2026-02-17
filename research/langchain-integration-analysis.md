# LangChain/LangGraph Integration Analysis

## Overview

This analysis evaluates integration points between Grover and the LangChain/LangGraph ecosystem. Grover's combination of versioned filesystem + knowledge graph + semantic search is unique — no other integration offers all three.

## Integration Points Evaluated

### Tier 1 — High Value, Clear Fit

| Integration | Interface | What It Does | Grover Advantage |
|-------------|-----------|-------------|------------------|
| **Retriever** | `langchain_core.retrievers.BaseRetriever` | Expose semantic search as LangChain retriever | Any RAG pipeline can use Grover's vector search |
| **Loader** | `langchain_core.document_loaders.BaseLoader` | Stream Grover files as Documents | Versioned filesystem + glob filtering for ingestion |
| **Store** | `langgraph.store.base.BaseStore` | Persistent memory store for LangGraph agents | Hierarchical namespaces map to file paths, versioned |
| **CheckpointSaver** | `langgraph.checkpoint.base.BaseCheckpointSaver` | Persist agent checkpoints | Versioned storage gives checkpoint history for free |

### Tier 2 — Moderate Value, Possible Later

| Integration | Interface | Notes |
|-------------|-----------|-------|
| **VectorStore** | `langchain_core.vectorstores.VectorStore` | Overlaps with Retriever; more complex (add/delete docs) |
| **ChatMessageHistory** | `langchain_core.chat_history.BaseChatMessageHistory` | Store conversation history as versioned files |
| **Tool** | `langchain_core.tools.BaseTool` | Wrap Grover ops as callable tools (read, write, search) |

### Tier 3 — Low Priority

| Integration | Notes |
|-------------|-------|
| **Embeddings** | Grover already has pluggable providers; wrapping LangChain Embeddings adds indirection |
| **OutputParser** | No natural fit |
| **Memory** | Deprecated in favor of Store |

## Selected for Implementation (v0.1)

1. **GroverRetriever** — BaseRetriever wrapping `grover.search()`
2. **GroverLoader** — BaseLoader wrapping `grover.tree()` / `grover.list_dir()` + `grover.read()`
3. **GroverStore** — LangGraph BaseStore wrapping filesystem ops with namespace-to-path mapping

CheckpointSaver deferred — requires deeper study of checkpoint serialization format.

## Key Interface Contracts

### BaseRetriever (`langchain_core.retrievers`)
- Inherits from `RunnableSerializable[str, list[Document]]` and `ABC`
- Abstract: `_get_relevant_documents(query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]`
- Optional: `_aget_relevant_documents(query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]`
- Pydantic model — needs `ConfigDict(arbitrary_types_allowed=True)` for Grover field
- Public interface: `.invoke(query)` / `.ainvoke(query)` from Runnable

### BaseLoader (`langchain_core.document_loaders`)
- Abstract: `lazy_load() -> Iterator[Document]` (generator-based)
- `load()` provided by base class: `list(self.lazy_load())`
- Optional: `alazy_load() -> AsyncIterator[Document]` (default runs lazy_load in executor)

### BaseStore (`langgraph.store.base`)
- Abstract: `batch(ops: Iterable[Op]) -> list[Result]` and `abatch(ops: Iterable[Op]) -> list[Result]`
- Op types: `GetOp(namespace, key)`, `PutOp(namespace, key, value)`, `SearchOp(namespace_prefix, ...)`, `ListNamespacesOp(...)`
- Result types: `Item(value, key, namespace, created_at, updated_at)`, `SearchItem(Item + score)`
- Namespaces are `tuple[str, ...]` — maps to directory paths under a prefix
- All concrete methods (`get`, `put`, `search`, `delete`, `list_namespaces`) delegate to `batch`/`abatch`

### Document (`langchain_core.documents`)
- `Document(page_content: str, metadata: dict = {}, id: str | None = None)`
- `metadata["source"]` is a convention — loaders set it to origin path

## Ecosystem Patterns

- **Import guards**: Optional deps use try/except ImportError with helpful message
- **Pydantic integration**: BaseRetriever is Pydantic — use `model_config = ConfigDict(arbitrary_types_allowed=True)`
- **Async pattern**: `asyncio.to_thread()` for sync-to-async bridging
- **Testing**: `pytest.importorskip("langchain_core")` to skip when dep not installed

## Architecture Decision

All three integrations live in `src/grover/integrations/langchain/`:
- `__init__.py` — import guard + exports
- `_retriever.py` — GroverRetriever
- `_loader.py` — GroverLoader
- `_store.py` — GroverStore (conditional on langgraph)

This mirrors the existing `src/grover/integrations/deepagents/` pattern.

Optional dependencies:
- `langchain` extra → `langchain-core>=0.3`
- `langgraph` extra → `langgraph>=0.2`
- `all` extra updated to include both

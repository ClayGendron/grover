[![PyPI version](https://img.shields.io/pypi/v/grover)](https://pypi.org/project/grover/)
[![Python](https://img.shields.io/pypi/pyversions/grover)](https://pypi.org/project/grover/)
[![License](https://img.shields.io/github/license/ClayGendron/grover)](https://github.com/ClayGendron/grover/blob/main/LICENSE)

# Grover

**The agentic filesystem.** Safe file operations, knowledge graphs, and semantic search — unified for AI agents.

> **Alpha** — Grover is under active development. The core API is functional and tested, but expect breaking changes before 1.0.

Grover gives AI agents a single toolkit for working with any knowledge base — documents, codebases, research, datasets, or any collection of files:

- **Versioned filesystem** — mount local directories or databases, write safely with automatic versioning, and recover mistakes with soft-delete trash and rollback.
- **Knowledge graph** — relationship, impact, and containment queries powered by [rustworkx](https://github.com/Qiskit/rustworkx). Add edges manually or let built-in analyzers extract structure automatically (Python via AST; JS/TS/Go via tree-sitter).
- **Semantic search** — pluggable vector stores (local [usearch](https://github.com/unum-cloud/usearch), [Pinecone](https://www.pinecone.io/), [Databricks](https://docs.databricks.com/en/generative-ai/vector-search.html)) with pluggable embedding providers (sentence-transformers, OpenAI, LangChain). Search by meaning, not just keywords.

All three layers stay in sync — write a file and the graph rebuilds and embeddings re-index automatically.

The name comes from **grove** (a connected cluster of trees) + **rover** (an agent that explores). Grover treats your data as a grove of interconnected files and lets agents navigate it safely.

## Installation

```bash
pip install grover
```

Optional extras:

```bash
pip install grover[search]       # sentence-transformers + usearch (local search)
pip install grover[openai]       # OpenAI embeddings
pip install grover[pinecone]     # Pinecone vector store
pip install grover[databricks]   # Databricks Vector Search
pip install grover[treesitter]   # JS/TS/Go code analyzers
pip install grover[postgres]     # PostgreSQL backend
pip install grover[mssql]        # MSSQL backend
pip install grover[deepagents]   # deepagents/LangGraph integration
pip install grover[langchain]    # LangChain retriever + document loader
pip install grover[langgraph]    # LangGraph persistent store
pip install grover[all]          # everything
```

Requires Python 3.12+.

## Quick start

```python
from grover import Grover
from grover.fs import LocalFileSystem

# Create a Grover instance (state is stored in .grover/)
g = Grover()

# Mount a local project directory
backend = LocalFileSystem(workspace_dir="/path/to/project")
g.mount("/project", backend)

# Write files — every write is automatically versioned
g.write("/project/hello.py", "def greet(name):\n    return f'Hello, {name}!'\n")
g.write("/project/main.py", "from hello import greet\nprint(greet('world'))\n")

# Read, edit, delete
content = g.read("/project/hello.py")
g.edit("/project/hello.py", "Hello", "Hi")
g.delete("/project/main.py")  # soft-delete — recoverable from trash

# Index the project (analyze code, build graph + search index)
stats = g.index()
# {"files_scanned": 42, "chunks_created": 187, "edges_added": 95}

# Knowledge graph queries
g.dependencies("/project/main.py")   # what does main.py depend on?
g.dependents("/project/hello.py")    # what depends on hello.py?
g.impacts("/project/hello.py")       # transitive impact analysis
g.contains("/project/hello.py")      # functions and classes inside

# Graph algorithms (centrality, traversal, subgraph extraction)
scores = g.pagerank()                                     # PageRank centrality
anc = g.ancestors("/project/main.py")                     # transitive predecessors
sub = g.meeting_subgraph(["/project/a.py", "/project/b.py"])  # connecting subgraph
nodes = g.find_nodes(lang="python")                       # filter by attributes

# Semantic search (requires the search extra)
result = g.search("greeting function", k=5)
for hit in result.hits:
    print(hit.path, hit.score)

# Persist and clean up
g.save()
g.close()
```

A full async API is also available:

```python
from grover import GroverAsync

g = GroverAsync()
await g.mount("/project", backend)
await g.write("/project/hello.py", "...")
await g.save()
await g.close()
```

## Architecture

Grover is composed of three layers that share a common identity model — every node in the graph and every entry in the search index is a file path.

```mermaid
graph TD
    A["Grover (sync) / GroverAsync"]
    A --> B["VFS — Virtual Filesystem"]
    A --> C["Graph — Knowledge Graph"]
    A --> D["SearchEngine"]
    A --> E["EventBus"]

    B --> F["LocalFileSystem<br/><i>disk + SQLite</i>"]
    B --> G["DatabaseFileSystem<br/><i>PostgreSQL · MSSQL · SQLite</i>"]

    C --> H["rustworkx DiGraph"]
    C --> I["Analyzers<br/><i>Python · JS/TS · Go</i>"]

    D --> J["VectorStore<br/><i>Local · Pinecone · Databricks</i>"]
    D --> K["EmbeddingProvider<br/><i>sentence-transformers · OpenAI · LangChain</i>"]

    E -.->|FILE_WRITTEN| C
    E -.->|FILE_WRITTEN| D
    E -.->|FILE_DELETED| C
    E -.->|FILE_DELETED| D
```

**VFS** routes operations to the right backend based on mount paths. Multiple backends can be mounted simultaneously.

**Graph** maintains an in-memory directed graph of file dependencies. Code analyzers automatically extract imports, function definitions, and class hierarchies. You can also add manual edges.

**SearchEngine** orchestrates embedding and vector storage. It wires together an `EmbeddingProvider` (text → vectors) and a `VectorStore` (store/search vectors). The default setup uses `all-MiniLM-L6-v2` embeddings + local usearch HNSW. For production, swap in Pinecone or Databricks with OpenAI embeddings.

**EventBus** keeps everything consistent — when a file is written or deleted, the graph and search engine update automatically.

## Backends

Grover supports two storage backends through a common protocol:

**LocalFileSystem** — for desktop development and code editing. Files live on disk where your IDE, git, and other tools can see them. Metadata and version history are stored in a local SQLite database. This is the default for local projects.

**DatabaseFileSystem** — for web applications and shared knowledge bases. All content lives in the database (PostgreSQL, MSSQL, or SQLite). There are no physical files. This is ideal for multi-tenant platforms, enterprise document stores, or any environment where state should be centralized.

Both backends support versioning and trash. You can mount them side by side:

```python
from grover.fs import LocalFileSystem, DatabaseFileSystem

g = Grover()

# Local code on disk
g.mount("/code", LocalFileSystem(workspace_dir="./my-project"))

# Shared docs in PostgreSQL
g.mount("/docs", DatabaseFileSystem(dialect="postgresql"))
```

### User-scoped mounts

For multi-tenant deployments, mount a `UserScopedFileSystem` to enable per-user namespacing:

```python
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.fs.sharing import SharingService
from grover.models.shares import FileShare

g = GroverAsync()
backend = UserScopedFileSystem(sharing=SharingService(FileShare))
await g.mount("/ws", backend, engine=engine)

# Each user has their own namespace
await g.write("/ws/notes.md", "hello", user_id="alice")
await g.write("/ws/notes.md", "world", user_id="bob")
r1 = await g.read("/ws/notes.md", user_id="alice")  # "hello"
r2 = await g.read("/ws/notes.md", user_id="bob")    # "world"

# Share files between users
await g.share("/ws/notes.md", "bob", user_id="alice")
r3 = await g.read("/ws/@shared/alice/notes.md", user_id="bob")  # "hello"
```

### deepagents integration

Use Grover as a storage backend for [deepagents](https://github.com/langchain-ai/deepagents) (LangGraph agent framework):

```python
from grover.integrations.deepagents import GroverBackend, GroverMiddleware

# GroverBackend implements deepagents BackendProtocol
backend = GroverBackend.from_local("/path/to/workspace")

# GroverMiddleware adds version, search, graph, and trash tools
middleware = [GroverMiddleware(backend.grover)]
```

Requires the `deepagents` extra: `pip install grover[deepagents]`

### LangChain / LangGraph integration

Use Grover as a LangChain retriever, document loader, or LangGraph persistent store:

```python
from grover.integrations.langchain import GroverRetriever, GroverLoader, GroverStore

# Retriever — semantic search as a LangChain retriever
retriever = GroverRetriever(grover=g, k=5)
docs = retriever.invoke("authentication logic")

# Loader — stream files as LangChain Documents
loader = GroverLoader(grover=g, path="/project", glob_pattern="*.py")
docs = loader.load()

# Store — LangGraph persistent memory backed by Grover
store = GroverStore(grover=g, prefix="/data/store")
store.put(("users", "alice"), "prefs", {"theme": "dark"})
item = store.get(("users", "alice"), "prefs")
```

Requires `pip install grover[langchain]` for retriever/loader, `pip install grover[langgraph]` for store.

## What's in `.grover/`

When you use Grover, a `.grover/` directory is created to store internal state:

| Path | Contents |
|------|----------|
| `grover.db` | SQLite database with file metadata, version history, graph edges, and extracted code chunks |
| `search.usearch` | The HNSW vector index for semantic search |
| `search_meta.json` | Metadata mapping for the search index |

This directory is excluded from indexing automatically. You'll typically want to add `.grover/` to your `.gitignore`.

## API overview

The full API reference is in [`docs/api.md`](docs/api.md). Here's a summary:

| Category | Methods |
|----------|---------|
| **Filesystem** | `read`, `write`, `edit`, `delete`, `list_dir`, `exists`, `move`, `copy` |
| **Versioning** | `list_versions`, `get_version_content`, `restore_version` |
| **Trash** | `list_trash`, `restore_from_trash`, `empty_trash` |
| **Sharing** | `share`, `unshare`, `list_shares`, `list_shared_with_me` |
| **Graph** | `dependencies`, `dependents`, `impacts`, `path_between`, `contains`, `pagerank`, `ancestors`, `descendants`, `meeting_subgraph`, `neighborhood`, `find_nodes` |
| **Search** | `search` |
| **Lifecycle** | `mount`, `unmount`, `index`, `save`, `close` |

Key types:

```python
from grover import Ref, file_ref, SearchQueryResult, SearchHit, ChunkMatch

# Ref — immutable reference to a file or chunk
Ref(path="/project/hello.py", version=2, line_start=1, line_end=5)

# SearchQueryResult — document-first search results
result = g.search("greeting function", k=5)
result.success       # bool
result.hits          # tuple[SearchHit, ...]
hit = result.hits[0]
hit.path             # str — file path
hit.score            # float — max chunk similarity (0–1)
hit.chunk_matches    # tuple[ChunkMatch, ...] — matching chunks within the file
```

## Error handling

All filesystem operations return **result objects** instead of raising exceptions. Every result has a `success: bool` field and a `message: str` field. Always check `success` before using other fields:

```python
result = g.write("/project/hello.py", "content")
if result.success:
    print(f"Created version {result.version}")
else:
    print(f"Write failed: {result.message}")
```

This design is intentional — agents running in loops should never crash on a failed file operation. The full set of result types (`ReadResult`, `WriteResult`, `EditResult`, etc.) is documented in [`docs/api.md`](docs/api.md#result-types).

## Roadmap

Grover is in its first release cycle. Here's what's coming:

- **MCP server** — expose Grover as a Model Context Protocol server for Claude Code, Cursor, and other MCP-compatible agents
- **CLI** — `grover init`, `grover status`, `grover search`, `grover rollback`
- **More framework integrations** — Aider plugin, fsspec adapter
- **More language analyzers** — Rust, Java, C#
- **More embedding providers** — Cohere, Voyage (OpenAI and LangChain adapters are already available)

See the [implementation plan](grover_implementation_plan.md) for the full roadmap.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, workflow, and guidelines.

## License

[Apache-2.0](LICENSE)

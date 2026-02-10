# grover

Safe files, knowledge graphs, and semantic search for AI agents.

## Features

- **Filesystem** — mount-based virtual filesystem with local and database backends, automatic versioning, soft-delete trash, and cross-mount operations
- **Knowledge graph** — dependency, impact, and containment queries powered by rustworkx
- **Semantic search** — HNSW vector index (usearch) with pluggable embedding providers
- **Code analysis** — built-in Python AST analyzer; JavaScript, TypeScript, and Go via tree-sitter
- **Async-first** — full async API (`GroverAsync`) with a sync wrapper (`Grover`)

## Installation

```bash
pip install grover
```

Optional extras:

```bash
pip install grover[search]       # sentence-transformers + usearch
pip install grover[treesitter]   # JS/TS/Go analyzers
pip install grover[postgres]     # PostgreSQL backend
pip install grover[mssql]        # MSSQL backend
pip install grover[all]          # search + treesitter + postgres
```

## Quick start

### Sync API

```python
from grover import Grover
from grover.fs import LocalFileSystem

g = Grover(data_dir="/tmp/grover-demo")

# Mount a local directory
backend = LocalFileSystem(workspace_dir="/path/to/project")
g.mount("/project", backend)

# Use a context manager for transaction mode (auto-commits on success)
with g:
    # Write files — automatically versioned
    g.write("/project/hello.py", "def greet(name):\n    return f'Hello, {name}!'\n")
    g.write("/project/main.py", "from hello import greet\nprint(greet('world'))\n")

# Read files
content = g.read("/project/hello.py")

# Edit files (find-and-replace)
g.edit("/project/hello.py", "Hello", "Hi")

# List directory contents
entries = g.list_dir("/project")

# Check existence
g.exists("/project/hello.py")  # → True

# Delete files (soft-delete to trash)
g.delete("/project/main.py")

# Knowledge graph queries (return list[Ref])
g.dependencies("/project/main.py")   # → nodes that main.py depends on
g.dependents("/project/hello.py")    # → nodes that depend on hello.py
g.impacts("/project/hello.py")       # → transitive reverse reachability
g.contains("/project/hello.py")      # → chunks (functions, classes)
g.path_between("/project/main.py", "/project/hello.py")  # → shortest path or None

# Bulk-index an existing project (analyze files, build graph + search index)
stats = g.index()
print(stats)  # {"files_scanned": 42, "chunks_created": 187, "edges_added": 95}

# Semantic search (requires search extra + embedding provider)
results = g.search("greeting function", k=5)
for r in results:
    print(r.ref.path, r.score, r.content)

# Persist graph and search index
g.save()

# Clean up
g.close()
```

### Async API

```python
from grover import GroverAsync
from grover.fs import LocalFileSystem

g = GroverAsync(data_dir="/tmp/grover-demo")

backend = LocalFileSystem(workspace_dir="/path/to/project")
await g.mount("/project", backend)

async with g:
    await g.write("/project/hello.py", "def greet(): ...")

content = await g.read("/project/hello.py")
stats = await g.index()
await g.save()
await g.close()
```

## API reference

### Grover / GroverAsync

| Method | Returns | Description |
|--------|---------|-------------|
| `mount(path, backend, *, mount_type, permission, label, hidden)` | `None` | Mount a storage backend at a virtual path |
| `unmount(path)` | `None` | Remove a mount |
| `read(path)` | `str \| None` | Read file content |
| `write(path, content)` | `bool` | Write file (creates or updates) |
| `edit(path, old, new)` | `bool` | Find-and-replace within a file |
| `delete(path)` | `bool` | Soft-delete a file |
| `list_dir(path="/")` | `list[dict]` | List directory entries |
| `exists(path)` | `bool` | Check if a path exists |
| `dependencies(path)` | `list[Ref]` | Nodes this file depends on |
| `dependents(path)` | `list[Ref]` | Nodes that depend on this file |
| `impacts(path, max_depth=3)` | `list[Ref]` | Transitive reverse reachability |
| `path_between(source, target)` | `list[Ref] \| None` | Shortest path between two nodes |
| `contains(path)` | `list[Ref]` | Chunks contained in this file |
| `search(query, k=10)` | `list[SearchResult]` | Semantic similarity search |
| `index(mount_path=None)` | `dict[str, int]` | Analyze files, build graph and search index |
| `save()` | `None` | Persist graph and search index to disk |
| `close()` | `None` | Shut down subsystems |

### Key types

```python
from grover import Ref, SearchResult, file_ref

# Ref — immutable reference to a file or chunk
Ref(path="/project/hello.py", version=2, line_start=1, line_end=5)

# file_ref — shorthand constructor
file_ref("/project/hello.py", version=2)

# SearchResult
result.ref      # Ref
result.score    # float (cosine similarity, 0–1)
result.content  # str (the matched text)
```

### Filesystem layer

```python
from grover.fs import (
    LocalFileSystem,       # Local disk + SQLite versioning
    DatabaseFileSystem,    # Pure-database storage
    UnifiedFileSystem,     # Mount-routing layer
    MountConfig,           # Mount configuration dataclass
    Permission,            # READ_WRITE | READ_ONLY
)
```

### Graph

```python
from grover.graph import Graph

graph = g.graph  # access via Grover instance
graph.add_node("/project/foo.py")
graph.add_edge("/project/main.py", "/project/foo.py", "imports")
graph.nodes()       # list[str]
graph.edges()       # list[tuple[str, str, dict]]
graph.node_count    # int
graph.edge_count    # int
graph.is_dag()      # bool
```

### Search

```python
from grover.search import (
    SearchIndex,
    EmbeddingProvider,              # protocol for custom providers
    SentenceTransformerProvider,    # default provider (requires search extra)
)
```

## License

Apache-2.0

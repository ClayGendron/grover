# grover

Versioned file systems, graph retrieval, and vector search for AI agents.

## Features

- **Filesystem** — local and database backends with automatic versioning, soft-delete trash, and a unified mount layer
- **Knowledge graph** — dependency, impact, and containment queries powered by rustworkx
- **Semantic search** — HNSW vector index (usearch) with pluggable embedding providers
- **Code analysis** — built-in Python AST analyzer; JavaScript, TypeScript, and Go via tree-sitter

## Installation

```bash
pip install grover
```

Optional extras:

```bash
pip install grover[search]       # sentence-transformers + usearch
pip install grover[treesitter]   # JS/TS/Go analyzers
pip install grover[postgres]     # PostgreSQL backend
pip install grover[all]          # everything above
```

## Quick start

```python
from grover import Grover

with Grover("/path/to/project") as g:
    # Write files — automatically versioned and indexed
    g.write("/project/hello.py", "def greet(name):\n    return f'Hello, {name}!'\n")
    g.write("/project/main.py", "from hello import greet\nprint(greet('world'))\n")

    # Semantic search
    results = g.search("greeting function")
    print(results[0].path, results[0].score)

    # Knowledge graph queries
    g.dependencies("/project/main.py")   # → ['/project/hello.py']
    g.impacts("/project/hello.py")       # → files affected by a change
    g.contains("/project/hello.py")      # → chunks (functions, classes)

    # Bulk-index an existing project
    stats = g.index()
    print(stats)  # {'files_scanned': 42, 'chunks_created': 187, 'edges_added': 95}

    # Persist graph and search index
    g.save()
```

## License

Apache-2.0

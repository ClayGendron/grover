# Grover v0.1 Implementation Plan

## Context

Grover is a Python library that gives AI agents three unified primitives over a shared file system: **safe versioned file operations**, **a knowledge graph**, and **semantic search**. The file system is the source of truth — **everything is a file or a directory**. The graph adds structural metadata on top. Search discovers content by meaning.

**Core invariant:** Every entity in Grover — whether it represents source code, a document, a person, or a concept — is a file or directory in the filesystem. There are no non-file entities. If a graph edge references something that doesn't exist, a file or directory is created for it. On the **local backend**, chunks of files (functions, document sections, code blocks) are stored as separate files in `.grover/chunks/`. On the **database backend**, chunks are rows in `grover_files` with `parent_path` set — no physical files needed.

It evolved from two prior projects:
- **Existing fs/ code** (~3,800 LOC): A mature virtual filesystem with `StorageBackend` protocol, local + database backends, mount composition, versioning, and structured result types. Needs adaptation (missing `db_models.py`, dialect-specific SQL needs to be made portable across SQLite/PostgreSQL/MSSQL).
- **QuiverDB**: A graph+vector+SQL system with a rustworkx-based `Graph` class, schema-driven models, change tracking, and embedding protocols. Provides patterns for the graph layer.
- **DeepAgents** (LangChain): Proved protocol-driven virtual filesystems for agents. Grover fills its gaps: versioning, code intelligence, semantic search.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Identity model | **Everything is a file or directory.** Paths are identity. Graph nodes are file paths. Chunks are files in `.grover/chunks/`. No URIs, no RefKind. |
| Chunk storage | **Local backend:** Chunks are real files stored in `.grover/chunks/`, organized by parent file path. **Database backend:** Chunks are rows in `grover_files` with content stored in the database. Both backends track chunks in `grover_files` with `parent_path` and `line_start`/`line_end` metadata. |
| Graph naming | `Graph` (not CodeGraph). |
| Graph scoping | One `Graph` per mounted filesystem (not per user). Query methods filter results by caller's permissions (`include_restricted=False` by default). |
| Graph node storage | **No `grover_nodes` table.** Graph nodes are file paths looked up from `grover_files`. The filesystem IS the node registry. |
| Graph edge storage | Single `grover_edges` table. Edge types are free-form strings with built-in conventions. |
| Permissions | Mount-level: access to a mount = ability to use graph/search/fs. Directory-level: restricts read/write of content AND filters graph/search results (by default). `include_restricted=True` opts out of filtering. Directories are the permission gates, not individual files. |
| Framework coupling | Framework-agnostic Python library. No LangGraph/LangChain dependency. |
| Interface | Python library only. No CLI or MCP server for v0.1. |
| Versioning | Diff-based (forward diffs + periodic snapshots). Text files only. |
| Graph engine | rustworkx (from QuiverDB). Potential custom Rust CSR graph later for async support. |
| Database | One SQLite database per local mount (`.grover/grover.db`). Location configurable by the app developer. Database filesystems already have their own database. |
| Multi-node files | Many graph nodes can reference the same parent file (via chunks). |
| Database backends | SQLite, PostgreSQL, MSSQL — deep SQLModel/SQLAlchemy integration. Dialect-aware SQL. |
| Search flow | `vector search → graph traversal → query from filesystem`. |
| Sync/async | **Async internals** (existing codebase is async throughout). The `Grover` class provides **sync wrappers** for the public API. `Graph.to_sql()`/`from_sql()` use `AsyncSession` to share transaction boundaries with the filesystem layer. |
| Tenancy | **Single-tenant for v0.1.** `user_id` defaults to a constant. Multi-tenant scoping across all tables is a v0.2 concern. |
| Tooling | `hatch` (hatchling build backend) + `uv` (package management) + `ty` (type checking) + `ruff` (linting/formatting). |
| Typing | Everything typed. `py.typed` marker. Full `ty` and IDE support. |
| Code style | Pure functions where possible. Testable. Protocols for interfaces. Composition over inheritance. |

---

## Package Structure

```
grover/
├── pyproject.toml
├── uv.lock
├── .python-version               # "3.11"
├── README.md
├── LICENSE
├── src/grover/
│   ├── __init__.py               # Public exports: Grover, Ref, SearchResult
│   ├── py.typed                  # PEP 561 typed marker
│   ├── _grover.py                # Main Grover class (lifecycle, sync wrappers)
│   │
│   ├── ref.py                    # Ref dataclass (file path identity)
│   ├── events.py                 # EventBus, EventType, event dataclasses
│   │
│   ├── models/                   # SQLModel database models
│   │   ├── __init__.py
│   │   ├── files.py              # GroverFile, FileVersion (diff-based)
│   │   ├── edges.py              # GroverEdge (single table)
│   │   └── embeddings.py         # Embedding metadata table
│   │
│   ├── fs/                       # Filesystem layer (adapted from existing fs/)
│   │   ├── __init__.py
│   │   ├── protocol.py           # StorageBackend protocol (runtime-checkable)
│   │   ├── types.py              # ReadResult, WriteResult, EditResult, etc.
│   │   ├── permissions.py        # Permission enum, directory-level gates
│   │   ├── base.py               # BaseFileSystem (shared SQL logic, dialect-aware)
│   │   ├── local_fs.py           # LocalFileSystem (disk + SQLite versioning)
│   │   ├── database_fs.py        # DatabaseFileSystem (pure SQL)
│   │   ├── local_disk.py         # LocalDiskBackend (no versioning)
│   │   ├── mounts.py             # MountRegistry, MountConfig
│   │   ├── unified.py            # UnifiedFileSystem (routing + permissions + events)
│   │   ├── utils.py              # Path utils, text replacement, binary detection
│   │   └── dialect.py            # Dialect-aware SQL helpers (upsert, merge)
│   │
│   ├── graph/                    # Knowledge graph layer
│   │   ├── __init__.py
│   │   ├── _graph.py             # Graph (rustworkx wrapper, file-path-based nodes)
│   │   └── analyzers/
│   │       ├── __init__.py       # AnalyzerRegistry, get_analyzer()
│   │       ├── _base.py          # Analyzer protocol
│   │       ├── python.py         # PythonAnalyzer (stdlib ast)
│   │       ├── javascript.py     # JavaScriptAnalyzer (tree-sitter)
│   │       └── go.py             # GoAnalyzer (tree-sitter)
│   │
│   └── search/                   # Vector search layer
│       ├── __init__.py
│       ├── _index.py             # SearchIndex (usearch HNSW + metadata)
│       ├── extractors.py         # Text extraction → chunk file creation
│       └── providers/
│           ├── __init__.py
│           ├── _protocol.py      # EmbeddingProvider protocol
│           └── sentence_transformers.py  # Default: all-MiniLM-L6-v2
│
└── tests/
    ├── conftest.py
    ├── test_ref.py
    ├── test_events.py
    ├── fs/ ...
    ├── graph/ ...
    └── search/ ...
```

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/grover"]

[project]
name = "grover"
version = "0.1.0"
description = "Safe files, knowledge graphs, and semantic search for AI agents"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "sqlmodel>=0.0.31",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "rustworkx>=0.17",
]

[project.optional-dependencies]
search = [
    "sentence-transformers>=3.0",
    "usearch>=2.0",
]
treesitter = [
    "tree-sitter>=0.22",
    "tree-sitter-javascript>=0.21",
    "tree-sitter-typescript>=0.21",
    "tree-sitter-go>=0.21",
]
postgres = ["asyncpg>=0.29"]
mssql = ["aioodbc>=0.5"]
all = ["grover[search,treesitter,postgres]"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.11",
    "ty>=0.1",
    "pre-commit>=4.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.format]
docstring-code-format = true
docstring-code-line-length = 80

[tool.ruff.lint]
select = [
    "F",      # pyflakes
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "ANN",    # flake8-annotations
    "B",      # flake8-bugbear
    "A",      # flake8-builtins
    "C4",     # flake8-comprehensions
    "DTZ",    # flake8-datetimez
    "T20",    # flake8-print
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "RUF",    # ruff-specific
    "PTH",    # flake8-use-pathlib
    "PERF",   # perflint
]
ignore = [
    "ANN101",  # missing type annotation for self
    "ANN102",  # missing type annotation for cls
    "ANN401",  # dynamically typed expressions (Any)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*.py" = ["ANN", "S101"]

[tool.ruff.lint.isort]
known-first-party = ["grover"]

[tool.ty]

[tool.ty.environment]
python-version = "3.11"

[tool.ty.src]
root = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]

[tool.coverage.run]
source = ["src/grover"]
omit = ["tests/*"]

[tool.coverage.report]
show_missing = true
skip_empty = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.",
    "@overload",
    "raise NotImplementedError",
    "\\.\\.\\.",
]
```

---

## Component Plans

### 1. Ref — File Path Identity (`src/grover/ref.py`)

Everything is a file or directory. Identity is a path. Refs are frozen dataclasses (value objects, not Pydantic — lightweight, hashable, passed around frequently).

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Ref:
    """Reference to a file in Grover. Immutable and hashable.

    The path is the identity. Line numbers are optional metadata
    for search results pointing to specific locations within a file.
    """

    path: str                            # /src/auth.py or .grover/chunks/src/auth_py/login.txt
    version: int | None = None           # version at time of reference
    line_start: int | None = None        # optional: where in the file (for search results)
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


def file_ref(path: str, version: int | None = None) -> Ref:
    """Create a Ref to a file."""
    return Ref(path=normalize_path(path), version=version)


def normalize_path(path: str) -> str:
    """Normalize a file path. Delegates to fs.utils.normalize_path().

    Uses posixpath.normpath() to resolve . and .. components,
    ensure leading /, collapse //, and remove trailing /.
    One normalize_path() used everywhere — ref.py imports from fs.utils.
    """
    from grover.fs.utils import normalize_path as _normalize
    return _normalize(path)
```

### 2. Database Models (`src/grover/models/`)

#### GroverFile and FileVersion (`models/files.py`)

Files and directories. Chunks are files with `parent_path` set.

```python
from datetime import datetime

from sqlmodel import Field, SQLModel


class GroverFile(SQLModel, table=True):
    """File or directory in the Grover filesystem.

    Chunks (functions, document sections, etc.) are files where
    parent_path is set. They live in .grover/chunks/ and carry
    line_start/line_end metadata pointing back to their parent file.
    """

    __tablename__ = "grover_files"

    file_id: str = Field(primary_key=True)
    user_id: str = Field(default="default", index=True)
    path: str = Field(index=True)
    name: str
    is_directory: bool = Field(default=False)
    content: str | None = None            # Current content (DB backend only)
    content_hash: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    version: int = Field(default=1)

    # Chunk metadata — set when this file is a chunk of another file
    parent_path: str | None = Field(default=None, index=True)
    line_start: int | None = None
    line_end: int | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: datetime | None = None    # Soft delete
    original_path: str | None = None      # Pre-trash path


class FileVersion(SQLModel, table=True):
    """Version entry: either a full snapshot or a forward diff."""

    __tablename__ = "grover_file_versions"

    id: str = Field(primary_key=True)
    file_id: str = Field(index=True)
    version: int
    is_snapshot: bool = Field(default=False)   # True = full content, False = diff
    content: str                                # Full content if snapshot, unified diff if not
    content_hash: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    change_summary: str | None = None
```

**Versioning mechanics:**
- Version 1 is always a snapshot (`is_snapshot=True`, `content` = full text).
- Subsequent versions store a unified diff (`is_snapshot=False`, `content` = `difflib.unified_diff`).
- Every N versions (configurable, default 20), a snapshot is stored.
- To reconstruct version K: find nearest snapshot at or before K, apply forward diffs.
- To get current: read from disk (local) or `GroverFile.content` (database).

```python
SNAPSHOT_INTERVAL: int = 20


def compute_diff(old: str, new: str) -> str:
    """Compute unified diff between two text contents."""

def apply_diff(base: str, diff: str) -> str:
    """Apply a unified diff to base content, producing the new version."""

def reconstruct_version(snapshots_and_diffs: list[FileVersion]) -> str:
    """Reconstruct content at a specific version from snapshot + forward diffs."""
```

#### GroverEdge (`models/edges.py`)

Single table for all graph edges. No typed edge subtables.

```python
class GroverEdge(SQLModel, table=True):
    """Edge in the knowledge graph. Connects two file paths."""

    __tablename__ = "grover_edges"

    id: str = Field(primary_key=True)
    source_path: str = Field(index=True)     # file path (must exist in grover_files)
    target_path: str = Field(index=True)     # file path (must exist in grover_files)
    type: str = Field(index=True)            # "imports", "contains", "references", etc.
    is_derived: bool = Field(default=False)  # True if auto-generated (AST analysis)
    stale: bool = Field(default=False)       # True if source file changed since edge was created
    weight: float = Field(default=1.0)
    metadata_json: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**Built-in edge type conventions** (free-form strings, not an enum):

| Type | Meaning | Example |
|------|---------|---------|
| `"imports"` | File A imports from file B | `auth.py → db/pool.py` |
| `"contains"` | File contains a chunk | `auth.py → .grover/chunks/auth_py/login.txt` |
| `"references"` | General reference between files | `README.md → src/auth.py` |
| `"inherits"` | Class in file A inherits from class in file B | Derived from AST |
| `"depends_on"` | Generic dependency | User-defined |

**Dangling edge behavior:** If an edge is created with a `target_path` that doesn't exist in `grover_files`, the system creates a blank file (or directory, if the path has no extension) at that path. This enforces the "everything is a file" invariant.

#### Embedding (`models/embeddings.py`)

```python
class Embedding(SQLModel, table=True):
    """Tracks what has been embedded for change detection."""

    __tablename__ = "grover_embeddings"

    id: str = Field(primary_key=True)
    file_path: str = Field(index=True)
    source_type: str                       # "docstring", "comment", "signature", "chunk"
    source_hash: str                       # Hash of source text — for change detection
    model_name: str
    dimensions: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### 3. Dialect-Aware SQL (`src/grover/fs/dialect.py`)

The existing `base.py` has MSSQL-specific SQL. Make it dialect-aware.

```python
def get_dialect(engine: Engine) -> str:
    """Extract dialect name: 'sqlite', 'postgresql', or 'mssql'."""

async def upsert_file(
    session: AsyncSession,
    dialect_name: str,
    table_name: str,
    values: dict[str, Any],
    conflict_keys: list[str],
) -> int:
    """Dialect-aware upsert. Returns rowcount.

    - SQLite:      INSERT ... ON CONFLICT DO UPDATE
    - PostgreSQL:  INSERT ... ON CONFLICT DO UPDATE
    - MSSQL:       MERGE INTO ... WITH (HOLDLOCK)
    """
```

### 4. Filesystem Layer (`src/grover/fs/`)

**Adapted from:** `/Users/claygendron/Git/Repos/grover/fs/`

| File | Changes |
|------|---------|
| `protocol.py` | Add `overwrite: bool = True` parameter to `StorageBackend.write()`. Default `True` is backwards-compatible; chunk pipelines pass `overwrite=False` to catch collisions. |
| `types.py` | Keep all result dataclasses. Add optional `ref: Ref \| None` field. |
| `permissions.py` | Keep Permission enum. Extend with directory-level gates. |
| `utils.py` | Keep as-is. Add `.gitignore` pattern matching for chunk creation. |
| `base.py` | Replace `from ..db_models` → `from grover.models.files`. Replace MSSQL-specific SQL with `dialect.upsert_file()`. Replace `SYSDATETIMEOFFSET()` with `func.now()`. Integrate diff-based versioning in `_save_version()`. |
| `local_fs.py` | Change imports. DB path configurable: default `.grover/grover.db`, or app developer specifies location via constructor. |
| `database_fs.py` | Change imports. Remove Datum-specific code. |
| `local_disk.py` | Minimal changes. |
| `mounts.py` | Keep as-is. Mounts are the permission boundary for graph access. |
| `unified.py` | Add `EventBus` integration. Emit events after successful mutations. |
| `dialect.py` | **NEW** — dialect-aware upsert, merge, date functions. |

**Chunk management:**

When an analyzer or extractor identifies a chunk within a file, the chunk is stored as a `grover_files` row with `parent_path` set. On the local backend, a corresponding file is also written to disk.

```python
# Analyzing /src/auth.py finds function "login" at lines 10-25

# Chunk path uses symbol name only — NO line numbers in path (they drift on edits)
chunk_path = "/.grover/chunks/src/auth_py/login.txt"
chunk_content = extract_lines(parent_content, line_start=10, line_end=25)

# Local backend: creates a real file on disk
# write(overwrite=False) is the default — catches collisions in the pipeline
g.fs.write(chunk_path, chunk_content)

# Database backend: content stored in grover_files.content column
# No physical file created — the row IS the chunk

# Both backends: registered in grover_files with chunk metadata:
# parent_path="/src/auth.py", line_start=10, line_end=25
# (line_start/line_end are updatable metadata, refreshed on re-analysis)

# Graph edge created:
# /src/auth.py --contains--> /.grover/chunks/src/auth_py/login.txt
```

**Chunk path stability:** Chunk paths use **symbol names only**, never line numbers. Line numbers are unstable — editing a file shifts them. When a parent file changes, re-analysis updates `line_start`/`line_end` metadata in `grover_files` and rewrites chunk content, but the chunk path stays the same as long as the symbol name hasn't changed. If a symbol is removed, the chunk file and its edges are deleted.

**Write behavior:** `write(overwrite=False)` is the default. If a pipeline attempts to create a chunk that already exists, the write fails with an error, allowing the developer to detect and handle the collision.

**Local backend** chunk files:
- Live in `/.grover/chunks/` within the mount, mirroring the parent file's directory structure
- Are root-directory scoped (persist across sessions — no need to recreate when reopening a repo)
- Honor `.gitignore` (files matched by `.gitignore` are not analyzed/chunked)
- Contain the actual text for the chunk (extracted from the parent file)

**Database backend** chunks:
- Stored as rows in `grover_files` with `content` populated and `parent_path` set
- No physical files — the database is the storage layer
- Same chunk paths used as identifiers for consistency across backends

**Configurable `.grover/` location:**

```python
# Default: .grover/ in the project root
g = Grover("/path/to/project")
# → database at /path/to/project/.grover/grover.db

# Custom: app developer specifies data directory
g = Grover("/path/to/project", data_dir="/path/to/my_app/data")
# → database at /path/to/my_app/data/grover.db
# → chunks at /path/to/my_app/data/chunks/

# Database backend: already has its own database
g = Grover("postgresql://localhost/mydb")
# → no .grover/ directory needed
```

### 5. Knowledge Graph (`src/grover/graph/`)

One `Graph` per mounted filesystem. Nodes are file paths from `grover_files`. No separate node table.

#### Graph (`graph/_graph.py`)

Wraps `rustworkx.PyDiGraph`. Simplified from QuiverDB's `Graph` — no schema validation, no typed models, no change tracker (for v0.1).

```python
from __future__ import annotations

from typing import Any, Hashable

import rustworkx as rx

from grover.ref import Ref


class Graph:
    """In-memory directed graph over file paths.

    Nodes are file paths (from grover_files). Edges are typed
    relationships between files. The graph is metadata — structural
    information about the relationships between files.

    Loaded from SQL on startup. Persisted back to SQL on save.
    """

    def __init__(self) -> None:
        self._graph: rx.PyDiGraph = rx.PyDiGraph()
        self._path_to_idx: dict[str, int] = {}
        self._idx_to_path: dict[int, str] = {}
        self._node_data: dict[str, dict[str, Any]] = {}

    # --- Node operations ---

    def add_node(self, path: str, **attrs: Any) -> None:
        """Add a file path as a node. Merges attrs if already exists."""

    def remove_node(self, path: str) -> None:
        """Remove a node and all incident edges."""

    def has_node(self, path: str) -> bool:
        """Check if a path is in the graph."""

    def get_node(self, path: str) -> dict[str, Any]:
        """Get node attributes."""

    # --- Edge operations ---

    def add_edge(
        self, source: str, target: str, type: str, **attrs: Any
    ) -> None:
        """Add a typed edge between two file paths.

        If source or target does not exist as a node, it is created.
        """

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge."""

    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists."""

    # --- Query methods ---
    # Note: Graph itself is permission-unaware. The Grover class wraps these
    # methods and filters results via include_restricted flag + permission layer.

    def dependents(self, path: str) -> list[Ref]:
        """What depends on this file? (incoming edges)"""

    def dependencies(self, path: str) -> list[Ref]:
        """What does this file depend on? (outgoing edges)"""

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        """Transitive BFS: what is affected if this file changes?"""

    def path_between(self, source: str, target: str) -> list[str] | None:
        """Shortest path via rustworkx Dijkstra."""

    def contains(self, path: str) -> list[Ref]:
        """Chunks/entities defined inside this file (via 'contains' edges)."""

    def by_parent(self, parent_path: str) -> list[Ref]:
        """All chunk nodes referencing a given parent file."""

    def remove_file_subgraph(self, path: str) -> list[str]:
        """Remove a file node and all its chunk nodes (for incremental rebuild)."""

    # --- Graph-level queries ---

    def nodes(self) -> list[str]:
        """All node paths."""

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """All edges as (source, target, attrs) triples."""

    @property
    def node_count(self) -> int:
        """Number of nodes."""

    @property
    def edge_count(self) -> int:
        """Number of edges."""

    def is_dag(self) -> bool:
        """Check if graph is acyclic."""

    # --- Persistence ---

    async def to_sql(self, session: AsyncSession) -> None:
        """Persist graph edges to grover_edges table.

        Nodes are not persisted here — they live in grover_files.
        Only edges need to be saved/loaded.
        Uses AsyncSession to share transaction boundaries with the FS layer.
        """

    async def from_sql(self, session: AsyncSession, files: list[GroverFile]) -> None:
        """Load graph from grover_files (nodes) + grover_edges (edges).

        1. Create a node for each file in grover_files
        2. Load all edges from grover_edges
        3. Wire up the rustworkx graph
        """
```

**Key principle:** The graph is metadata. It describes relationships between files. One graph per mount (not per user). The `Graph` class is permission-unaware — the `Grover` class wraps query methods and filters results through the permission layer before returning them. By default (`include_restricted=False`), nodes in restricted directories are excluded from results. Callers can opt in to unfiltered results with `include_restricted=True`.

#### AST Analyzers (`graph/analyzers/`)

Analyzers auto-populate the graph from source code. They:
1. Parse source files to extract structure
2. Create chunk files in `.grover/chunks/` for functions, classes, etc.
3. Return edges to be added to the graph

Each analyzer is a pure function that returns chunk file data and edges:

```python
from dataclasses import dataclass


@dataclass
class ChunkFile:
    """A chunk to be written as a file in .grover/chunks/.

    Chunk paths include enclosing scope to avoid collisions:
      /.grover/chunks/src/auth_py/Client.connect.txt
      /.grover/chunks/src/auth_py/Server.connect.txt
      /.grover/chunks/src/auth_py/login.locals.validate.txt
    """
    chunk_path: str         # e.g., /.grover/chunks/src/auth_py/Client.connect.txt
    parent_path: str        # e.g., /src/auth.py
    content: str            # The extracted text
    line_start: int
    line_end: int
    name: str               # e.g., "Client.connect" (scoped symbol name)


@dataclass
class EdgeData:
    """An edge to be added to the graph."""
    source: str
    target: str
    type: str
    metadata: dict[str, Any] | None = None


def analyze_file(
    path: str,
    content: str,
) -> tuple[list[ChunkFile], list[EdgeData]]:
    """Parse a source file and return chunks + edges.

    Returns data only — does not mutate the filesystem or graph.
    The caller writes chunk files and adds edges.
    """
```

**Analyzers:**
- **PythonAnalyzer** (`analyzers/python.py`): stdlib `ast`. Extracts imports, functions, classes, inheritance.
- **JavaScriptAnalyzer** (`analyzers/javascript.py`): tree-sitter. Same for JS/TS.
- **GoAnalyzer** (`analyzers/go.py`): tree-sitter. Same for Go.

Analyzers honor `.gitignore` — files matching ignore patterns are skipped.

### 6. Vector Search (`src/grover/search/`)

**Search flow:** `vector search → graph traversal → query from filesystem`

1. User searches: `g.search("database connection retry logic")`
2. Search index returns matching chunk files (or whole files) with scores
3. Graph traversal finds parent files and related files
4. Agent reads file content from the filesystem

**EmbeddingProvider protocol** (from QuiverDB):

```python
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
```

Default: `SentenceTransformerProvider` using `all-MiniLM-L6-v2` (~80MB, CPU, 384 dims, no API key).

**Text extraction** (`extractors.py`): Extracts embeddable text from files. Creates chunk files in `.grover/chunks/` for each meaningful segment (docstrings, function signatures, document sections). Does NOT embed raw code bodies.

**SearchIndex** (`_index.py`): HNSW index via `usearch`.
- `add_file(path, content)` — extract chunks, embed, index
- `remove_file(path)` — remove all embeddings for a file and its chunks
- `search(query, k=10) -> list[SearchResult]`
- `save(path)` / `load(path)` — persist to `.grover/` directory

```python
@dataclass
class SearchResult:
    """A search result with file reference and score."""
    ref: Ref                     # Points to chunk file or whole file
    score: float                 # Similarity score
    source_type: str             # "docstring", "signature", "chunk", etc.
    source_text: str             # The text that matched
    parent_path: str | None      # Parent file path (if result is a chunk)
```

### 7. Consistency Layer (`src/grover/events.py`)

Synchronous event bus. Pure handler functions.

```python
from enum import Enum
from dataclasses import dataclass


class EventType(Enum):
    FILE_WRITTEN = "file_written"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    FILE_RESTORED = "file_restored"


@dataclass
class FileEvent:
    type: EventType
    path: str
    content: str | None = None
    old_path: str | None = None   # For moves


class EventBus:
    """Register handlers for file events. Errors logged, not propagated."""

    def register(self, event_type: EventType, handler: Callable) -> None: ...
    def emit(self, event: FileEvent) -> None: ...
```

**Wiring:**
- `FILE_WRITTEN` → re-analyze file (create/update chunks, rebuild graph subgraph, re-embed)
- `FILE_DELETED` → remove file node + chunk nodes from graph, remove from search index
- `FILE_MOVED` → update paths in graph and search index

### 8. Top-Level Grover Class (`src/grover/_grover.py`)

```python
# Local backend — disk + .grover/ SQLite
g = Grover("/path/to/project")

# Local backend with custom data directory
g = Grover("/path/to/project", data_dir="/app/data")

# Database backends
g = Grover("sqlite:///workspace.db")
g = Grover("postgresql://localhost/mydb")
g = Grover("mssql+aioodbc://...")

# --- Filesystem (sync) ---
g.fs.read("/src/auth.py")                   # ReadResult
g.fs.write("/src/auth.py", content)          # WriteResult (auto-versions)
g.fs.edit("/src/auth.py", old, new)          # EditResult
g.fs.rollback("/src/auth.py")               # RestoreResult
g.fs.versions("/src/auth.py")               # list[VersionInfo]

# --- Graph (sync wrappers, one graph per mount) ---
g.graph.dependents("/src/auth.py")           # list[Ref] — filtered by permissions
g.graph.impacts("/src/models/user.py")       # list[Ref] — transitive impact
g.graph.contains("/src/auth.py")             # list[Ref] — chunks within this file
g.graph.path_between("/src/a.py", "/src/b.py")  # list[str] | None

# Opt-in: include nodes in restricted directories
g.graph.dependents("/src/auth.py", include_restricted=True)

# --- Search (vector search → graph → filesystem) ---
g.search("database connection retry logic")  # list[SearchResult] — filtered
g.search("retry logic", include_restricted=True)  # unfiltered

# --- Lifecycle ---
g.index()                                    # Full scan: analyze all files, build graph, embed
g.save()                                     # Persist graph edges + search index
g.close()                                    # Save + cleanup
```

**Constructor:**
1. Detect backend type (local path vs database URL)
2. Set up filesystem (`UnifiedFileSystem` with appropriate backend)
3. Create `Graph` instance (one per mount)
4. Set up search index (load from `.grover/` if exists)
5. Wire event bus → graph + search
6. Load existing graph from SQL if available

---

## Permissions Model (v0.1)

Simple, two-level model with query-time filtering.

**Level 1 — Mount access:** If a user/agent has access to a mounted filesystem, they can:
- Traverse the graph (one graph per mount, shared by all users on that mount)
- List files and directories
- Search across indexed content

**Level 2 — Directory gates:** Within a mount, directories control data access:
- Directories are the permission boundary, not individual files
- Files inherit the permission of their containing directory
- Walking up the directory tree finds the nearest explicit permission
- Users/teams/departments can own directories

**Level 3 — Query-time filtering:** Graph and search queries filter results by the caller's readable paths. One graph is constructed per mount (not per user), but query methods accept `include_restricted: bool = False`:

```python
# Default: only returns nodes/results the caller can read
refs = g.graph.dependents("/src/auth.py")

# Opt-in: include nodes in restricted directories (metadata-only access)
refs = g.graph.dependents("/src/auth.py", include_restricted=True)

# Search respects the same flag
results = g.search("retry logic")                              # filtered
results = g.search("retry logic", include_restricted=True)     # unfiltered
```

When `include_restricted=False` (the default), the `Grover` class wraps `Graph` query methods and checks each returned node's path against the permission layer before including it in results. This avoids constructing separate graphs per user while preventing metadata leakage about files in restricted directories.

**Explicit design choice:** Mount access grants the ability to traverse the graph structure, but `read()` and `write()` are gated by directory permissions. With the default `include_restricted=False`, graph queries and search results are also filtered — a user won't learn that `/secret/project/auth.py` exists unless they have read access to `/secret/project/`.

```python
from grover.fs.permissions import Permission

# Mount-level access
mount = MountConfig(
    virtual_prefix="/engineering",
    backend=local_fs,
    permission=Permission.READ_WRITE,
)

# Directory-level restrictions within the mount
mount.read_only_paths = {"/engineering/archived"}

# The existing MountRegistry.get_permission() handles inheritance
```

**For v0.2+:** Extend to Zanzibar-style ReBAC with relationship tuples, recursive group expansion, and richer graph query filtering. The `scope` field and `NodeScope` table can be added when non-file entities are introduced.

---

## Implementation Order

### Step 1: Package Scaffolding
- Directory structure, `pyproject.toml` (hatch + uv + ty + ruff), all `__init__.py`
- `py.typed` marker
- pytest + pytest-asyncio setup
- `uv sync --group dev` to verify toolchain works
- **Files:** `pyproject.toml`, `.python-version`, all `__init__.py`, `py.typed`

### Step 2: Ref
- `src/grover/ref.py` — Ref frozen dataclass, `file_ref()`, `normalize_path()`
- `tests/test_ref.py`

### Step 3: Database Models
- `src/grover/models/files.py` — GroverFile (with chunk fields), FileVersion
- `src/grover/models/edges.py` — GroverEdge
- `src/grover/models/embeddings.py` — Embedding
- Diff utility functions: `compute_diff()`, `apply_diff()`, `reconstruct_version()`
- Test: table creation on SQLite, diff round-trip

### Step 4: Dialect-Aware SQL
- `src/grover/fs/dialect.py` — `upsert_file()`, `get_dialect()`, portable date functions
- Test: verify upsert works on SQLite

### Step 5: Filesystem Layer
- Copy and adapt all fs/ files from existing code
- Replace imports, make SQL dialect-aware, integrate diff versioning
- Add `.gitignore` pattern matching to `utils.py`
- Fix paths in `local_fs.py` (configurable data dir, default `.grover/grover.db`)
- `tests/fs/` — test both backends, versioning with diffs

### Step 6: Consistency Layer
- `src/grover/events.py` — EventBus, event types, handler signatures
- Add event emission to `unified.py`
- `tests/test_events.py`

### Step 7: Graph
- `src/grover/graph/_graph.py` — Graph (rustworkx wrapper, file-path-based nodes)
- Query methods: dependents, dependencies, impacts, path_between, contains, by_parent
- SQL persistence: `to_sql()` / `from_sql()` (edges table + files as nodes)
- Dangling edge handling (auto-create blank files)
- `tests/graph/test_graph.py`

### Step 8: AST Analyzers
- `src/grover/graph/analyzers/python.py` — PythonAnalyzer (stdlib ast)
- `src/grover/graph/analyzers/javascript.py` — tree-sitter
- `src/grover/graph/analyzers/go.py` — tree-sitter
- Chunk file creation in `.grover/chunks/`
- Analyzer registry
- Tests with sample source files

### Step 9: Vector Search
- `src/grover/search/providers/_protocol.py` — EmbeddingProvider protocol
- `src/grover/search/providers/sentence_transformers.py` — default provider
- `src/grover/search/extractors.py` — text extraction → chunk file creation
- `src/grover/search/_index.py` — SearchIndex with usearch
- `tests/search/`

### Step 10: Grover Class (integration)
- `src/grover/_grover.py` — main class
- Backend detection, lifecycle, configurable data_dir
- Wire event bus → graph + search
- `index()` full project scan
- End-to-end tests

### Step 11: Polish
- `src/grover/__init__.py` public exports
- Docstrings on all public APIs
- Final `ruff` + `ty` clean pass

---

## Verification

1. **Unit tests per step.** Run `uv run pytest tests/` at each step.
2. **Type checking per step.** Run `uv run ty check` at each step.
3. **Linting per step.** Run `uv run ruff check .` at each step.
4. **Diff versioning round-trip:** Write file → edit 5 times → reconstruct each version from diffs → verify content matches.
5. **Multi-backend filesystem:** Run fs tests against SQLite. PostgreSQL/MSSQL tested via optional CI.
6. **Graph from AST:** Parse the grover repo itself → verify node/edge counts → query `dependents()` and `impacts()`.
7. **Chunk creation:** Analyze a Python file → verify chunk files created in `.grover/chunks/` → verify graph edges connect parent to chunks.
8. **Graph from SQL:** Load graph from SQL tables → verify traversal works → save and reload.
9. **Search quality:** Index the grover repo → search "file versioning" → verify relevant files rank high.
10. **Search flow:** `search()` → graph traversal for related files → `fs.read()` for content.
11. **Cross-layer consistency:** Write a file via `g.fs.write()` → verify graph and search index updated automatically.
12. **End-to-end:** `Grover(".")` → `index()` → `search()` → `graph.dependents()` → `fs.read()`.
13. **.gitignore respect:** Files matching `.gitignore` patterns are not analyzed or chunked.
14. **Dangling edges:** Create edge to non-existent path → verify blank file auto-created.

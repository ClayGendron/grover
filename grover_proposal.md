# Grover: Safe Files, Code Intelligence, and Semantic Search for AI Agents

**Proposal Document — February 2026 — v3.1 (dual-backend parity)**

---

## What Is Grover?

Grover is a Python toolkit that makes codebases safe and navigable for AI agents. Point it at a local directory or connect it to a database, and it provides three capabilities immediately: **safe versioned file operations**, **automatic dependency graphs** parsed from source code, and **semantic search** over code and documentation — with zero configuration regardless of backend.

Under the hood, Grover is three retrieval primitives that share a common reference model: a **virtual filesystem**, an **in-memory graph**, and a **vector search index**. Each is a separate tool with its own API. What connects them is a shared identity layer — every entity in Grover has a canonical URI, and tools exchange typed references that carry identity, permissions, and version metadata.

The name comes from *grove* (a connected cluster of trees) and *rover* (an agent that explores). An agent roves through a grove of code.

```
pip install grover
```

---

## Why These Three?

Retrieval for coding agents has three fundamental modes, and no existing tool combines them cleanly.

**Filesystem** — Agents already know how to navigate files. Every agent framework ships with `ls`, `read`, `write`, `glob`, `grep`. But raw disk writes are dangerous — one bad edit and your code is gone. A versioned filesystem wraps the operations agents already use with automatic safety, without requiring a new interface.

**Graph** — Relationships between code entities matter. "What depends on this file?" "What modules does this function touch?" "What breaks if I change this?" Graph traversal answers questions that flat file listings and grep cannot. Crucially, source code already has this structure — imports, class hierarchies, function calls — you just need to parse it.

**Vector Search** — Similarity search finds what's semantically close when you don't know the exact path or name. "Find code related to authentication." "Where's the retry logic?" This is discovery — the entry point when the agent doesn't know where to start. With a local backend, search runs entirely on-device with no API key; with a database backend, embeddings persist across sessions and can be shared.

The power is in composition. An agent searches by similarity, then traverses the graph to find what's structurally connected, then reads the file content. Three calls, three tools, one shared reference model.

```python
# Search by similarity — returns refs with canonical IDs
results = g.search("database connection pooling")
# → grover://file/src/db/pool.py (score: 0.92)

# Traverse the graph for dependencies — accepts any canonical ID
deps = g.graph.dependents("grover://file/src/db/pool.py")
# → grover://file/src/api/routes.py, grover://file/src/workers/sync.py

# Read the file — path alias works as shorthand
content = g.fs.read("/src/api/routes.py")
```

The reference model also supports non-file entities for enterprise use cases — `grover://table/users.sessions`, `grover://ticket/AUTH-1234` — but the coding agent runtime works entirely with `grover://file/...` references.

---

## Architecture

### The Reference Model

The central architectural decision in Grover is promoting the "shared namespace" concept into a first-class **reference model**. Every entity in Grover — whether it's a file, a database table, a person, or an abstract concept — has a canonical identity. All three tools speak in terms of these references.

**Canonical URI scheme.** Every entity gets a typed URI:

```
grover://file/src/db/pool.py        # a file in the VFS
grover://table/analytics.sessions   # a database table
grover://person/jane.smith          # a person
grover://ticket/AUTH-1234           # an external ticket
grover://concept/connection-pooling # an abstract concept
```

The type prefix (`file/`, `table/`, `person/`) prevents collisions between file paths and non-file entities. File paths like `/src/db/pool.py` are convenience aliases that resolve to `grover://file/src/db/pool.py`. Agents can use either form — the system normalizes internally.

**Typed references.** Every tool input and output includes the canonical ID, optional path alias, and metadata. A result from any tool carries enough context to be passed directly to another tool:

```python
@dataclass
class Ref:
    uri: str                    # grover://file/src/db/pool.py
    kind: str                   # "file", "table", "person", etc.
    path: str | None = None     # /src/db/pool.py (alias, files only)
    version: int | None = None  # current version at time of reference
    permissions: str | None = None  # effective permission for caller
```

**Single permission evaluation.** Permissions are evaluated once at the reference model layer, not independently per tool. When an agent resolves a reference, the permission check happens at that point. Graph traversal and vector search filter results through the same policy — preventing the case where `filesystem.read()` denies access but `graph.dependents()` leaks the relationship.

```
┌─────────────────────────────────────────┐
│              Agent Tools                │
├──────────────┬────────────┬─────────────┤
│  filesystem  │   graph    │   search    │
│  read/write  │  traverse  │  similarity │
│  ls/edit     │  depends   │  discover   │
├──────────────┴────────────┴─────────────┤
│       Reference Model (Ref + URI)       │
│   identity · permissions · versioning   │
├─────────────────────────────────────────┤
│           Consistency Layer             │
│        events · index rebuild           │
└─────────────────────────────────────────┘
```

### Consistency Model

When a file changes, the graph and vector index may need to update. Grover defines clear semantics for this through an internal event log.

**Events.** Write operations on any tool emit typed events: `file_written`, `file_deleted`, `node_updated`, `edge_added`. These are internal — not a public message queue — but they provide hooks for cross-layer consistency.

**Rebuild strategy.** The vector search index always reflects the latest version of an entity (not historical versions). When a file is written, a `file_written` event triggers re-embedding. Graph edges that are derived from file content (e.g., Python import analysis) are flagged stale and rebuilt on next access or via an explicit `grover rebuild` command. Manually-added graph edges are not affected by file changes.

**Initial implementation.** For v0.1, event processing is synchronous and local. A `file_written` event immediately re-indexes the embedding and marks derived edges stale. This is simple, correct, and avoids the complexity of background workers or message queues. Async/background processing is a later optimization.

### The Graph Is Not the Filesystem

An earlier iteration of this project explored presenting graph traversal *as* directory navigation — where `ls` would traverse edges and directories would represent graph structure. The current design rejects that approach.

The filesystem is the filesystem. The graph is a separate knowledge layer *about* entities in the reference model. They share identity through canonical URIs but have distinct APIs. This is cleaner because:

- The graph models things that aren't files — `grover://person/jane`, `grover://ticket/AUTH-1234`, `grover://table/users.sessions` — without forcing them into a directory metaphor.
- The filesystem stays simple enough that any existing agent framework can use it without learning anything new.
- Agents compose the workflow themselves — Grover provides primitives, not prescribed patterns.

---

## Primary Use Case: Coding Agent Runtime

The most immediate and compelling application of Grover is as a **drop-in runtime for coding agents**. Rather than requiring developers to wire together three tools, Grover delivers all three capabilities automatically — whether backed by a local directory or a database.

### The Problem

Every coding agent today — Claude Code, Aider, Cursor, Copilot Workspace — writes directly to disk. There is no standard safety net. If the agent makes a bad edit, your options are `git stash`, `Cmd+Z` if your editor caught it, or manually reconstructing the file. Agents also navigate codebases poorly: they grep, they guess, they stuff entire files into context hoping to find what they need.

### Two Backends, Same Interface

Grover ships with two storage backends. Both expose the same `StorageBackend` protocol. Agents don't know or care which one is underneath.

**Local backend** — Files live on disk, visible to your IDE, git, and other tools. A `.grover/` directory holds version snapshots and metadata in SQLite. Best for: solo development, working with existing repos, keeping files in your normal workflow. Versioning works without git — any directory becomes safe for agent writes.

**Database backend** — Files and metadata stored in SQL (SQLite, PostgreSQL, Azure SQL). Content lives entirely in the database. Best for: multi-tenant applications, shared agent workspaces, environments where files don't need to exist on a physical disk, web-based tools, and sandboxed execution. Versioning is native to the storage layer.

```python
from grover import Grover

# Local backend — point at a directory
g = Grover("/path/to/my/project")

# Database backend — connect to SQL
g = Grover("postgresql://localhost/myproject")
# or
g = Grover("sqlite:///workspace.db")
```

### What Grover Provides

**Safe versioned file operations.** Every write automatically snapshots the previous version. Rollback is instant. Agents get `write`, `edit`, `rollback`, and `diff` out of the box, and developers get peace of mind — regardless of whether the backend is local disk or a database.

**Automatic dependency graphs from AST parsing.** Grover parses source files to build the import/dependency graph automatically. No configuration, no LLM extraction. The agent asks "what depends on this file?" or "what breaks if I change this function?" and gets a structural answer derived from the actual code.

Initial language support targets Python, JavaScript/TypeScript, and Go — covering the vast majority of agent-assisted development. Each language analyzer extracts imports, function definitions, class hierarchies, and module boundaries from the AST. The graph is rebuilt incrementally as files change (via the consistency layer's `file_written` events).

**Semantic search over code and documentation.** Grover embeds docstrings, comments, function signatures, and class descriptions using `all-MiniLM-L6-v2` by default — an 80MB model that runs on CPU with no API key. For database backends or teams needing higher quality, the embedding provider is pluggable (OpenAI, Cohere, Voyage). Search works the same either way.

### The Developer Experience

```python
from grover import Grover

# Either backend — same API
g = Grover("/path/to/my/project")          # local
g = Grover("postgresql://localhost/mydb")   # database

# Safe versioned writes
g.fs.write("/src/auth.py", new_content)   # auto-snapshots previous version
g.fs.rollback("/src/auth.py")             # instant undo
g.fs.versions("/src/auth.py")             # full edit history

# Auto-generated dependency graph (from AST, not LLM)
g.graph.dependents("/src/db/pool.py")     # what imports this?
g.graph.impacts("/src/models/user.py")    # what breaks if I change this?
g.graph.path("/src/auth.py", "/src/db/pool.py")  # how are these connected?

# Semantic search
g.search("database connection retry logic")
# → /src/db/pool.py (0.89), /src/db/retry.py (0.84), /src/config/db.py (0.71)
```

### Who This Is For

The primary audience is **developers using coding agents who want safety and intelligence**. Anyone running Claude Code, Aider, or similar tools on their codebase is a potential user. The local backend serves solo developers working on existing repos. The database backend serves teams building agent-powered applications, multi-tenant platforms, or sandboxed environments.

The secondary audience is agent developers who want to embed these capabilities into their own tools — they use Grover as a library rather than a CLI.

---

## The Filesystem Layer

### An Uncommon Gap in the Python Ecosystem

A survey of the Python ecosystem found no established library that provides safe, versioned file operations for agents — whether backed by a local directory or a database:

| Library | What It Does | What It Doesn't Do |
|---|---|---|
| **fsspec** | Unified interface to real storage (S3, GCS, HDFS, local) | No database-backed filesystem, no versioning, no agent-native operations |
| **PyFilesystem2** | Similar to fsspec, older | Same gaps — no SQL backend |
| **Deep Agents BackendProtocol** | Agent VFS for LangChain | Stores files as JSON line arrays, tightly coupled to LangChain |
| **Django Storage** | Blob storage for media uploads | Not a hierarchical filesystem |
| **FUSE/fusepy** | OS-level virtual mount | Linux kernel interface, far too heavy |

This combination — agent-native file operations with automatic versioning, structured result types, and mount composition — is uncommon in the Python ecosystem. No off-the-shelf package was found that provides it, whether backed by local disk or a database. The gap is especially notable for agent use cases: there is no standard way for a coding agent to write files safely with built-in undo.

### Existing Implementation

A working implementation already exists from a prior project. The core components are:

**`StorageBackend` protocol** — A runtime-checkable Python protocol defining the full interface: `read`, `write`, `edit`, `delete`, `list_dir`, `mkdir`, `move`, `copy`, plus versioning (`list_versions`, `restore_version`) and trash management. Returns structured result types (`ReadResult`, `WriteResult`, `EditResult`) with agent-friendly metadata like `total_lines`, `truncated`, and `lines_read` for LLM context window management.

**`LocalFileSystem`** — Files stored on disk (visible to IDE, git, other tools), with SQLite tracking metadata and version history. Atomic writes via temp files, binary file detection, fuzzy filename suggestions for near-misses. Ideal for working with existing repos where files need to stay on the real filesystem.

**`DatabaseFileSystem`** — All content and metadata stored in SQL via SQLAlchemy/SQLModel. Works with any supported database (SQLite, PostgreSQL, Azure SQL). Async session management with proper commit/rollback lifecycle. Ideal for multi-tenant applications, sandboxed environments, and cases where files don't need to exist on disk.

**`MountRegistry`** — Composite layer that maps virtual path prefixes to storage backends. Longest-prefix matching for path resolution. Permission inheritance walking up the directory tree. Supports mixing backends (e.g., `/project` → local disk, `/sandbox` → database).

**`Permission` system** — Mount-level and directory-level read-write / read-only controls with inheritance.

### Why Not fsspec?

fsspec is designed for bytes-in, bytes-out file I/O between storage backends. It is the right tool for "read a CSV from S3." It is the wrong foundation for Grover because:

- **fsspec has no concept of agent-native operations.** `edit()` with string replacement, `list_versions()`, structured result types with pagination metadata — these don't exist in fsspec and don't fit its design philosophy.
- **fsspec returns bytes or throws exceptions.** Grover's protocol returns structured results that agents can reason about. A `ReadResult` with `success`, `message`, `total_lines`, and `truncated` is fundamentally different from a file-like object.
- **fsspec would fight the graph integration.** When `list_dir` is backed by graph traversal and `search` falls through to vector similarity, the fsspec abstraction adds friction rather than removing it.

**The plan:** Build on the existing `StorageBackend` protocol as the core. Ship an optional **fsspec adapter** later — a thin `GroverFileSystem(AbstractFileSystem)` wrapper that translates fsspec's `cat`/`put`/`ls` into `StorageBackend` calls. This gives `pd.read_csv("grover://...")` interop for free without compromising the core design. Same approach for a Deep Agents adapter if that ecosystem matures.

```
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──┐  ┌──────▼─────┐  ┌──▼──────────┐
    │ StorageBack-│  │  fsspec    │  │ DeepAgents  │
    │ end (core) │  │ (adapter)  │  │ (adapter)   │
    └─────────┬──┘  └──────┬─────┘  └──┬──────────┘
              └────────────┼────────────┘
                           │
                  ┌────────▼────────┐
                  │   Grover Core   │
                  └─────────────────┘
```

---

## The Graph Layer

The graph is a separate API and tool. It models relationships between entities — files, modules, functions, classes, tables, concepts — as a directed graph. Entities are identified by canonical URIs from the reference model. Agents query the graph to understand structure, dependencies, and impact.

The key architectural decision: the graph is built from **existing structure**, not LLM-extracted entities. For the coding agent use case, this means **AST parsing** — Grover reads the actual source code and extracts the real dependency graph.

### Automatic Graph Construction from Source Code

For the local coding agent runtime, the graph is built automatically. No configuration. Grover ships with language analyzers that parse source files and extract:

- **Import/dependency edges.** `from foo import bar` → edge from current file to `foo`. `require('./utils')` → edge to `utils.js`. `use crate::db` → edge to `db` module.
- **Function and class definitions.** Nodes for each callable, with edges to their containing module.
- **Class hierarchies.** Inheritance edges between classes.
- **Module boundaries.** Which functions/classes belong to which files.

Initial language support:

| Language | Parser | Extracts |
|---|---|---|
| **Python** | `ast` (stdlib) | imports, functions, classes, decorators, `__all__` |
| **JavaScript/TypeScript** | Tree-sitter | imports/require, exports, class/function declarations |
| **Go** | Tree-sitter | package imports, function/struct/interface declarations |

Tree-sitter is the right choice for JS/TS/Go: it handles syntax variations cleanly, is fast, and has mature grammars for most languages. Python's stdlib `ast` is sufficient and avoids a native dependency for the most common case. Additional languages (Rust, Java, C#) can be added later by writing new analyzers against the same graph schema.

The graph rebuilds incrementally. When a file changes (`file_written` event from the consistency layer), only that file's AST is re-parsed and its edges are updated. Full repo indexing happens once on `Grover("/path/to/repo")` initialization.

### Beyond Code: Enterprise and Custom Graphs

The AST-generated graph is the zero-config default for code repos. But the graph API is general-purpose. Enterprise users can also populate the graph from foreign keys, API schemas, organizational hierarchies, or any other structured data — using the same typed-node, typed-edge API. The coding agent runtime is the launch product; the general-purpose graph is the platform.

Technical foundation from prior work includes NetworkX-compatible API with Pydantic-validated schemas, typed nodes and edges, and CSR (Compressed Sparse Row) format for memory-efficient traversal.

### Graph Scalability

In-memory graph storage is fast but has real constraints. Enterprise-scale graphs can exceed memory budgets, and CSR format — while compact (~12 bytes/edge) — doesn't support efficient incremental updates. Grover addresses this through backend flexibility:

- **v0.1: Pure in-memory (NetworkX/CSR).** Simple, fast, correct. Suitable for project-scale graphs (thousands to low hundreds of thousands of nodes). Persistence via snapshot serialization to disk or database.
- **v0.2: SQLite/PostgreSQL backend.** Same API, backed by SQL. Trades query speed for persistence, incremental updates, and larger-than-memory graphs. The `StorageBackend` pattern from the filesystem layer applies here — the graph API is a protocol, not an implementation.
- **Later: Hybrid cached.** Hot subgraphs in memory, cold storage in the database. LRU eviction policy. This is the production architecture for large-scale deployments.

The API stays stable across all backends. This is a non-negotiable constraint — agents should not need to know or care where the graph is stored.

---

## The Vector Search Layer

Vector search provides similarity-based discovery over code and documentation. This is the "I don't know the exact path" entry point — the agent searches by meaning, then uses the graph and filesystem to explore what it finds.

### What Gets Embedded

- **Docstrings and comments.** The richest source of semantic meaning in code. A function's docstring describes what it does in natural language — exactly what an agent needs to search against.
- **Function and class signatures.** `def retry_with_backoff(fn, max_retries=3, delay=1.0)` carries semantic information even without a docstring.
- **Module-level descriptions.** File-level docstrings, README content, and header comments.
- **Commit messages** (optional, if git history is available). Recent commit messages often describe *why* code changed.

What doesn't get embedded (at least initially): raw code bodies. Embedding `for i in range(10)` produces poor search results. The docstrings and signatures are where the meaning lives.

### Embedding Strategy

The default embedding model is **`all-MiniLM-L6-v2`** — roughly 80MB, runs on CPU, no API key required, produces 384-dimensional embeddings. This makes the zero-config experience work: `pip install grover` and search works immediately.

The embedding layer is pluggable. Users who want higher-quality embeddings or are already using an API provider can configure OpenAI, Cohere, or Voyage via a simple interface. The HNSW index format (via `usearch`) is the same regardless of which model produced the embeddings.

For a typical repo of 500–5,000 files, index construction takes seconds and search is sub-millisecond.

---

## Brand Identity

### Name

**Grover** — *grove* (connected trees, data structure) + *rover* (agent that explores). Two syllables, memorable, approachable. The Sesame Street association is a feature for a DX-first open-source library, not a bug.

### Visual Identity

A full brand lookbook has been developed with the following system:

**Fonts**
- DM Serif Display — wordmark and headings (craftsmanship, warmth)
- Instrument Sans — body and UI (clean, contemporary, legible)
- JetBrains Mono — code and technical content

**Color System**
- Core palette: Forest gradient from Canopy (#1A2F23) through Daylight (#F0FAF4). Dark-mode-first, inverts cleanly to light.
- Pillar accents: Fern green for graph, Violet (#8B6CC1) for vectors, Cyan (#4ABCE8) for filesystem, Ember orange (#E8734A) for primary actions.

**Voice**
- Speaks like a senior engineer explaining something to a peer.
- Direct, opinionated, respectful of the reader's intelligence.
- No hype, no hand-waving, no "leverage" or "revolutionize."

---

## Prioritized Roadmap

### Phase 1: Coding Agent Runtime (launch)

| Priority | Component | Description |
|---|---|---|
| **1** | Reference Model | Canonical URI scheme, `Ref` type, permission evaluation contract |
| **2** | `StorageBackend` with both backends | Local (disk + `.grover/` SQLite metadata) and Database (SQLAlchemy/SQLModel). Auto-versioning, rollback, diff on both. |
| **3** | Python AST analyzer | Parse imports, functions, classes; build dependency graph automatically |
| **4** | Consistency layer | `file_written` events trigger incremental graph rebuild and re-embedding |
| **5** | Vector search | `all-MiniLM-L6-v2` default embedding of docstrings/comments/signatures, HNSW index via usearch |
| **6** | JS/TS + Go analyzers | Tree-sitter-based AST parsing for second and third language support |
| **7** | CLI | `grover init`, `grover status`, `grover rollback`, `grover search` |

### Phase 2: Ecosystem Expansion

| Priority | Component | Description |
|---|---|---|
| **8** | Additional language analyzers | Rust, Java, C# |
| **9** | Pluggable embedding providers | OpenAI, Cohere, Voyage as alternatives to default local model |
| **10** | fsspec adapter | Optional wrapper for pandas/Dask/PyArrow interop |
| **11** | Agent framework integrations | LangGraph tool definitions, Claude Code MCP server, Aider plugin |

---

## Distribution

- **Product name:** Grover
- **Package name:** `grover` is the target, but collision risk is real ("Grover" is a common name). Fallback options in preference order: `grover-ai`, `grover-kit`, `grover-dev`. The namespace should be reserved on PyPI early regardless of launch timeline.
- **Interfaces:** Both a Python library (`from grover import Grover`) and a CLI (`grover init`, `grover search`, `grover rollback`)
- **License:** MIT
- **Primary launch user:** Developers using coding agents (Claude Code, Aider, Cursor) who want safe file operations and better code navigation — on local repos or database-backed workspaces
- **Secondary launch user:** Agent developers embedding Grover as a library in their own tools
- **Dependencies at launch:** `sentence-transformers` (for all-MiniLM-L6-v2), `usearch` (HNSW index), `tree-sitter` + language grammars (JS/TS/Go parsing), `pydantic`, `sqlmodel`. Python AST parsing uses stdlib only.
- **Monetization (later, not key objective):** Hosted multi-tenant Grover instances with PostgreSQL storage and API access. Mirrors the DuckDB → MotherDuck trajectory.

---

## Summary

Grover is three retrieval primitives — filesystem, graph, vector search — unified by a shared reference model. The launch product is concrete: **point Grover at a local directory or connect it to a database, and get safe versioned writes, automatic dependency graphs from AST parsing, and semantic search over your code. Zero config, same API either way.**

The filesystem layer is already built with both backends. The graph layer auto-generates from source code. The vector layer ships with a default local embedding model and supports pluggable providers. Every entity gets a canonical URI. Every tool respects the same identity, permissions, and versioning contract.

The audience is every developer running a coding agent on their codebase who wants a safety net and better code intelligence — whether that's a solo developer working on a local repo or a team building agent-powered applications on a shared database. The platform underneath — general-purpose reference model, pluggable backends, framework integrations — is what makes it extensible.

*Rove through your code.*

---

## Appendix A: Changes from Peer Review (v2)

This document was updated based on detailed architectural feedback. The following changes were made:

| Feedback | Resolution |
|---|---|
| Path as universal key is brittle for non-file entities | Introduced canonical URI scheme (`grover://file/...`, `grover://table/...`, `grover://person/...`) with file paths as convenience aliases |
| Cross-layer permissions need single source of truth | Permissions evaluated once at the reference model layer; all tools filter through the same policy |
| Update/consistency mechanics undefined | Added Consistency Model section with internal event log, rebuild hooks, and clear "latest version only" semantics for search index |
| In-memory graph scalability constraint | Added graph backend flexibility roadmap (in-memory → SQLite/Postgres → hybrid cached) with stable API guarantee |
| "Novel VFS" claim risky as marketing headline | Softened to "uncommon in the Python ecosystem" with note to validate further before marketing use |
| Package name collision risk | Added fallback naming strategy and recommendation to reserve namespace early |

## Appendix B: Coding Agent Runtime Pivot (v3/v3.1)

Repositioned Grover from "three retrieval primitives for agent developers" to "coding agent runtime with zero-config code intelligence and dual backends." Key changes:

| Change | Rationale |
|---|---|
| Added "Primary Use Case: Coding Agent Runtime" section | This is the launch product — specific, concrete, immediately useful |
| Local and Database as equal peer backends | Both ship at launch with same API. Local for existing repos; database for multi-tenant, sandboxed, or shared workspaces |
| Automatic AST-based graph construction | Graph layer becomes zero-config for code repos (Python, JS/TS, Go via AST parsing) |
| `all-MiniLM-L6-v2` as default embedding model | 80MB, CPU-only, no API key — zero-friction default. Pluggable providers for teams wanting more. |
| Embedding targets: docstrings, comments, signatures | Not raw code bodies — search quality matters more than coverage |
| Audience shift to developers *using* coding agents | Larger market than developers *building* agents |
| Two-phase roadmap | Phase 1 ships both backends + code intelligence; Phase 2 expands ecosystem |
| CLI interface added alongside Python library | `grover init`, `grover search`, `grover rollback` for direct developer use |
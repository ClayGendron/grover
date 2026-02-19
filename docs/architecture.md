# Architecture Guide

This document describes the key design patterns and principles behind Grover's codebase. It's intended for contributors who want to understand *why* the code is structured the way it is before making changes.

For implementation details about the filesystem layer specifically, see [internals/fs.md](internals/fs.md).

---

## Core principle: everything is a file

Grover's identity model is built on one rule: **every entity is a file or directory**. Graph nodes are file paths. Search index entries are file paths. Chunks (functions, classes extracted from source code) are files stored in `.grover/chunks/`.

This means:

- There is no separate `grover_nodes` table. The `grover_files` table *is* the node registry.
- A `Ref` is just a path with optional version and line range metadata.
- Graph edges connect paths to paths. If you can see the file, you can see the node.

This simplification keeps the three layers (filesystem, graph, search) naturally aligned. A file write creates a node, generates edges, and indexes embeddings — all keyed by the same path.

## Composition over inheritance

Grover uses composition and protocols instead of class hierarchies. There is no `BaseFileSystem` abstract class.

**Backends** (`LocalFileSystem`, `DatabaseFileSystem`) are independent classes that both implement the `StorageBackend` protocol. They compose shared services internally:

```
LocalFileSystem
├── MetadataService    (file lookup, hashing)
├── VersioningService  (diff storage, reconstruction)
├── DirectoryService   (hierarchy operations)
└── TrashService       (soft-delete, restore)
```

**Orchestration functions** in `operations.py` are pure functions that take services and callbacks as parameters. Both backends call the same functions — no duplication, no inheritance.

**Why?** Inheritance creates coupling. When `LocalFileSystem` needs to write to disk and `DatabaseFileSystem` needs to write to a DB column, a shared base class either forces awkward abstractions or leaves half the logic in the subclass anyway. Composition lets each backend wire up exactly the behavior it needs.

## Capability protocols

Not every backend supports every feature. Rather than checking flags or catching `NotImplementedError`, Grover uses runtime-checkable protocols:

```python
class StorageBackend(Protocol):       # Core: read, write, edit, delete, ...
class SupportsVersions(Protocol):     # list_versions, restore_version, ...
class SupportsTrash(Protocol):        # list_trash, restore_from_trash, ...
class SupportsReconcile(Protocol):    # reconcile (sync disk ↔ DB)
class SupportsSearch(Protocol):       # search (per-mount semantic search)
class SupportsFileChunks(Protocol):   # replace/delete/list file chunks
```

VFS checks capabilities with `isinstance(backend, SupportsVersions)` at runtime. If a backend doesn't support a capability:

- **Targeted operations** (e.g., `list_versions("/path")`) raise `CapabilityNotSupportedError`.
- **Aggregate operations** (e.g., `list_trash()` across all mounts) silently skip unsupported backends.
- **GroverAsync** catches capability errors and returns `Result(success=False, message=...)` so the caller always gets a clean result, never an unhandled exception.

This makes it straightforward to write a minimal custom backend — just implement `StorageBackend` and skip the optional protocols.

### Graph capability protocols

The graph layer uses the same pattern. `GraphStore` is the core protocol (node/edge CRUD + basic queries). Capability protocols add algorithms:

```python
class GraphStore(Protocol):              # Core: add/remove/query nodes and edges
class SupportsCentrality(Protocol):      # PageRank, betweenness, closeness, katz, degree
class SupportsConnectivity(Protocol):    # Connected components, is_weakly_connected
class SupportsTraversal(Protocol):       # Ancestors, descendants, topological sort, shortest paths
class SupportsSubgraph(Protocol):        # Subgraph extraction, neighborhood, meeting subgraph
class SupportsFiltering(Protocol):       # Attribute-based node/edge filtering
class SupportsNodeSimilarity(Protocol):  # Jaccard structural similarity
class SupportsPersistence(Protocol):     # SQL persistence (to_sql / from_sql)
```

`RustworkxGraph` implements all protocols. To write a custom graph backend, implement `GraphStore` plus whichever capabilities you need. `GroverAsync` checks capabilities at runtime with `isinstance()` and raises `CapabilityNotSupportedError` for unsupported operations.

## Content-before-commit write ordering

All mutating operations follow the same sequence:

```
1. Save version record to session     (not yet committed)
2. Write content to storage            (disk or DB column)
3. Flush the session                   (session.flush)
4. VFS commits on context manager exit (session.commit)
```

If step 2 fails, the session rolls back and the version record is discarded. Clean state.

If step 4 fails after step 2, the content exists but the DB has no record of it. This is an **orphan file** — invisible to the system and harmless.

The opposite ordering (commit-first) would create **phantom metadata**: the DB says a file exists, but the content is missing. That breaks reads, fails hash verification, and can't be rolled back. Orphan files are strictly better than phantom metadata.

**This ordering is intentional and should not be changed.** See [internals/fs.md](internals/fs.md) for the full rationale and history.

## Per-mount graph and search

Each mounted backend holds its own `RustworkxGraph` and `SearchEngine` instance, injected at mount time. There is no global graph or search engine on `GroverAsync`.

```
Mount "/project" → LocalFileSystem
    ├── _graph: RustworkxGraph     (own graph instance)
    └── _search_engine: SearchEngine (own search index)

Mount "/data" → DatabaseFileSystem
    ├── _graph: RustworkxGraph     (own graph instance)
    └── _search_engine: SearchEngine (own search index)
```

**Graph resolution**: operations like `dependents(path)` resolve the mount from the path, then delegate to that mount's graph. `get_graph(path)` is the public method (replaces the removed `.graph` property).

**Search routing**: `search()` routes through VFS to per-mount backends via the `SupportsSearch` protocol. Root-level searches aggregate results across all mounts; path-scoped searches target a single mount.

**Persistence**: each mount's graph saves to its own database (via `to_sql`/`from_sql`). Search indices save per-mount under `data_dir/search/{mount_slug}/`.

**Embedding provider**: shared across mounts (stateless). Each mount gets its own `LocalVectorStore` by default.

Hidden mounts (like `/.grover`) do not receive graph or search engine injection.

## Event-driven consistency

The three layers stay in sync through an `EventBus`. When VFS completes a file operation, it emits an event:

| Event | Triggers |
|-------|----------|
| `FILE_WRITTEN` | Re-analyze file, update graph edges, re-index embeddings |
| `FILE_DELETED` | Remove file and children from graph and search index |
| `FILE_MOVED` | Remove old path, re-analyze at new path |
| `FILE_RESTORED` | Re-analyze restored file |

Event handlers resolve the mount from the event path, then use that mount's graph and search engine. Exceptions in handlers are logged but never propagated — a failed re-index should not cause a file write to fail.

This design means the graph and search index are always *eventually* consistent with the filesystem. For v0.1, handlers run synchronously within the same operation. Background workers are a future optimization.

## Session management

Sessions are owned by VFS, never by backends.

VFS creates a session per operation via `_session_for(mount)`. The session is injected into backend methods as a keyword argument. Backends only call `session.flush()` to make changes visible within the transaction. VFS handles commit (on success) and rollback (on exception).

**LocalFileSystem** manages its own SQLite engine internally (lazy init with `asyncio.Lock`), but session creation is still driven by VFS.

**DatabaseFileSystem** is fully stateless — it holds no engine, no session factory, and no mutable state. It receives everything it needs through the injected session. This makes it safe for concurrent use in web servers.

## Mount-first architecture

All file operations go through mount paths. There is no global filesystem — you mount backends at virtual paths and interact through those paths:

```python
g.mount("/code", LocalFileSystem(workspace_dir="."))
g.mount("/docs", DatabaseFileSystem(dialect="postgresql"))

g.read("/code/src/main.py")   # → routes to LocalFileSystem
g.read("/docs/guide.md")      # → routes to DatabaseFileSystem
```

`MountRegistry` resolves paths using longest-prefix matching. Mounts are permission boundaries — a read-only mount rejects all writes regardless of the file path.

Grover also creates a hidden metadata mount at `/.grover` for internal state (graph edges, search metadata). This mount is excluded from indexing and listing.

## Versioning strategy

Versions use a **snapshot + forward diff** approach:

- Version 1 is always a full snapshot.
- Subsequent versions store unified diffs from the previous content.
- Every 20 versions, a fresh snapshot is taken.

To reconstruct version N: find the nearest snapshot at or before N, then apply forward diffs in order. The `content_hash` (SHA-256) on each version record enables integrity verification after reconstruction.

This balances storage efficiency (diffs are small) with reconstruction speed (at most 19 diffs to replay).

**External edit detection:** If a file is modified outside Grover (by an IDE, git, etc.), the diff chain would silently break. At `write()` and `edit()` time, `check_external_edit()` compares the storage content's hash against the last Grover-written hash. On mismatch, a synthetic snapshot version is inserted with `created_by="external"` to keep the chain intact. See [internals/fs.md](internals/fs.md#external-edit-detection) for details.

## Graph model

The knowledge graph is an in-memory `rustworkx.PyDiGraph` with string-path-keyed nodes. Edges have a free-form `type` string — there's no enum or schema for edge types.

Built-in conventions:

| Edge type | Meaning |
|-----------|---------|
| `"imports"` | File imports another file |
| `"contains"` | File contains a chunk (function, class) |
| `"references"` | File references a symbol in another file |
| `"inherits"` | Class inherits from another class |

Code analyzers produce edges automatically. You can also add manual edges with any type string you like.

The graph persists to the `grover_edges` table on `save()` and loads from it on startup. The persistence strategy is full-sync: upsert all in-memory edges, delete any DB edges that are no longer in memory.

## Analyzer architecture

Analyzers implement a simple protocol: given a file path and its content, return a list of `ChunkFile` records and `EdgeData` records.

```python
class Analyzer(Protocol):
    def analyze_file(self, path: str, content: str) -> AnalysisResult: ...
```

The `AnalyzerRegistry` maps file extensions to analyzer implementations. Built-in analyzers:

- **Python** — uses stdlib `ast` module (no external dependencies)
- **JavaScript/TypeScript** — uses tree-sitter (requires `treesitter` extra)
- **Go** — uses tree-sitter (requires `treesitter` extra)

Chunks are stored as files in `.grover/chunks/` with stable paths based on symbol names (not line numbers, which drift on edits). This means chunk paths survive refactoring as long as the symbol name doesn't change.

## Adding a new analyzer

1. Create `src/grover/graph/analyzers/your_language.py`.
2. Implement the `Analyzer` protocol — `analyze_file(path, content) -> AnalysisResult`.
3. Register it in `src/grover/graph/analyzers/__init__.py` with the appropriate file extensions.
4. Add tests in `tests/test_analyzers.py`.

Analyzers should be pure functions of `(path, content)`. They should never raise on malformed input — return an empty result instead.

## User-scoped file systems

Grover supports **user-scoped mounts** where every operation requires a `user_id`. This enables multi-tenant deployments where multiple users share the same database but operate in isolated namespaces.

### UserScopedFileSystem

User scoping is implemented in `UserScopedFileSystem`, a subclass of `DatabaseFileSystem`. VFS is a pure mount router — it passes `user_id` through to the backend, and the backend handles all path rewriting, share checks, and trash scoping.

To create a user-scoped mount, pass a `UserScopedFileSystem` as the backend:

```python
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.fs.sharing import SharingService
from grover.models.shares import FileShare

backend = UserScopedFileSystem(sharing=SharingService(FileShare))
await g.mount("/ws", backend, engine=engine)
```

When `user_id` is provided:

1. **Write:** `g.write("/ws/notes.md", "hello", user_id="alice")` → backend stores at `/alice/notes.md`
2. **Read:** `g.read("/ws/notes.md", user_id="alice")` → backend reads `/alice/notes.md`
3. **Results:** Backend returns paths with `/{user_id}/` stripped, user sees `/ws/notes.md`

This design keeps path rewriting localized to the backend and prevents AI agents from escaping their namespace.

### `@shared/` virtual namespace

Files shared between users are browseable via `@shared/{owner}/`:

```
/ws/                    ← user's own files
/ws/@shared/            ← virtual directory listing shared owners
/ws/@shared/alice/      ← alice's files shared with the current user
/ws/@shared/alice/doc.md ← resolves to /alice/doc.md in the backend
```

Access to `@shared/` paths is permission-checked via `SharingService`. Directory shares grant access to all descendants (prefix matching). Write access requires an explicit `"write"` share.

### SupportsReBAC capability protocol

Share dispatch in `GroverAsync` uses the `SupportsReBAC` runtime-checkable protocol. Any backend implementing `share()`, `unshare()`, `list_shares_on_path()`, and `list_shared_with_me()` can participate in share operations. `UserScopedFileSystem` implements this protocol. Plain `DatabaseFileSystem` and `LocalFileSystem` do not — calling `g.share()` on those mounts returns a failure result.

### Move semantics (path is identity)

Following the git model, **path is identity**. The default `move()` creates a clean break — a new file record at the destination with no version history. Use `follow=True` for rename semantics where the file record, versions, and share paths follow the move. See [internals/fs.md](internals/fs.md#move-semantics) for details.

### Trash scoping

On user-scoped mounts, trash operations are scoped by `owner_id`. Each user can only list, restore, and empty their own trashed files. Regular mounts are unaffected.

## Search layer architecture

The search layer follows the same two-protocol pattern as the filesystem:

- **EmbeddingProvider** — converts text to vectors. Async-first, with `embed()` and `embed_batch()` methods. Built-in providers: sentence-transformers (local), OpenAI (API), LangChain adapter (any LangChain `Embeddings` instance).
- **VectorStore** — stores and searches vectors. Async-first. Built-in stores: LocalVectorStore (in-process usearch HNSW), PineconeVectorStore (Pinecone cloud), DatabricksVectorStore (Databricks Vector Search).

**SearchEngine** orchestrates them: embed text via provider → store vectors via store → search by embedding queries. `GroverAsync` creates a `SearchEngine` internally and wires it to the `EventBus`.

### Capability protocols (search)

Like filesystem backends, vector stores can advertise capabilities via runtime-checkable protocols:

| Protocol | What it enables | Stores |
|----------|----------------|--------|
| `SupportsNamespaces` | Namespace partitioning | Pinecone |
| `SupportsMetadataFilter` | Filter expressions on metadata | Pinecone, Databricks |
| `SupportsIndexLifecycle` | Create/delete/list indexes | Pinecone, Databricks |
| `SupportsHybridSearch` | Dense + sparse/keyword search | Pinecone, Databricks |
| `SupportsReranking` | Server-side reranking | Pinecone |
| `SupportsTextSearch` | Text query without external embeddings | (custom) |
| `SupportsTextIngest` | Text upsert without external embeddings | (custom) |

### Filter expression AST

Metadata filters are expressed as a provider-agnostic AST (`Comparison` and `LogicalGroup` nodes) and compiled to provider-native formats by store-specific compilers:

- `compile_pinecone()` → MongoDB-style dicts (`{"field": {"$eq": value}}`)
- `compile_databricks()` → SQL-like strings (`"field = 'value'"`)
- `compile_dict()` → simple dicts for local store (`{"field": value}`)

## Adding a new embedding provider

1. Create `src/grover/search/providers/your_provider.py`.
2. Implement the `EmbeddingProvider` protocol: async `embed(text)`, async `embed_batch(texts)`, plus `dimensions` and `model_name` properties.
3. Import-guard any optional dependencies.
4. Add the provider to `src/grover/search/providers/__init__.py`.
5. Add tests in `tests/test_embedding_providers.py`.

The provider is passed to `Grover(embedding_provider=...)` at construction time.

## Adding a new vector store

1. Create `src/grover/search/stores/your_store.py`.
2. Implement the `VectorStore` protocol: `upsert()`, `search()`, `delete()`, `fetch()`, `connect()`, `close()`, and `index_name` property.
3. Add any applicable capability protocols (e.g., `SupportsMetadataFilter`).
4. Import-guard any optional dependencies.
5. Add the store to `src/grover/search/stores/__init__.py`.
6. Add tests in `tests/test_your_store.py`.

The store is passed to `Grover(vector_store=...)` at construction time.

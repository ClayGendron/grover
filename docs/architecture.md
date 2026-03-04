# Architecture Guide

This document describes the key design patterns and principles behind Grover's codebase. It's intended for contributors who want to understand *why* the code is structured the way it is before making changes.

For implementation details about the filesystem layer specifically, see [internals/fs.md](internals/fs.md).

---

## Core principle: everything is a file

Grover's identity model is built on one rule: **every entity is a file or directory**. Graph nodes are file paths. Search index entries are file paths. Chunks (functions, classes extracted from source code) are stored as database rows in `grover_file_chunks` but are represented in the graph as nodes with synthetic path identifiers.

This means:

- There is no separate `grover_nodes` table. The `grover_files` table *is* the node registry.
- A `Ref` is a thin wrapper around a path string that can represent any entity: files (`/src/auth.py`), chunks (`/src/auth.py#login`), versions (`/src/auth.py@3`), or connections (`/src/auth.py[imports]/src/utils.py`). Properties lazily decompose the path.
- Graph edges connect paths to paths. If you can see the file, you can see the node.

This simplification keeps the three layers (filesystem, graph, search) naturally aligned. A file write creates a node, generates edges, and indexes embeddings — all keyed by the same path.

## Filesystem-centric architecture

Grover uses a **filesystem-centric** design where `DatabaseFileSystem` is the base class that owns all providers. `LocalFileSystem` is a thin subclass (~330 lines) that adds disk-specific overrides.

**Provider composition**: Instead of separate graph and search objects on the mount, all capabilities are pluggable **providers** on the filesystem itself:

```
DatabaseFileSystem
├── storage_provider: StorageProvider | None    (disk I/O — only LocalFileSystem)
├── graph_provider: GraphProvider | None        (in-memory knowledge graph)
├── search_provider: SearchProvider | None      (vector/lexical search)
├── embedding_provider: EmbeddingProvider | None (text → vector embedding)
├── version_provider: VersionProvider           (diff-based versioning)
├── chunk_provider: ChunkProvider               (code chunk storage)
├── ConnectionService                           (edge persistence)
├── DirectoryService                            (hierarchy operations)
└── TrashService                                (soft-delete, restore)
```

**LocalFileSystem** inherits from `DatabaseFileSystem` and passes a `DiskStorageProvider` to its parent. It only overrides lifecycle (`open`/`close`) and disk-specific operations (`read`, `delete`, `mkdir`, `restore`, `reconcile`). All other behavior — versioning, chunking, graph, search — comes from `DatabaseFileSystem` and its mixins.

**Orchestration functions** in `operations.py` are pure functions that take services and callbacks as parameters. Both the base class and its subclass call the same functions.

**Internal mixins** on `DatabaseFileSystem` organize provider-dependent methods:

| Mixin | Responsibility |
|-------|---------------|
| `GraphMethodsMixin` | Delegates to `graph_provider` — node/edge CRUD |
| `SearchMethodsMixin` | Orchestrates `embedding_provider` + `search_provider` |
| `VersionMethodsMixin` | Delegates to `version_provider` — diff storage, reconstruction |
| `ChunkMethodsMixin` | Delegates to `chunk_provider` — file chunk CRUD |

## GroverAsync facade structure

`GroverAsync` is split into mixin classes, each in its own file under `src/grover/api/`. The main class inherits all mixins and defines only `__init__`:

```
GroverAsync(MountMixin, FileOpsMixin, SearchOpsMixin, GraphOpsMixin,
            ShareMixin, ConnectionMixin, IndexMixin)
```

Shared state lives in a `GroverContext` dataclass stored as `self._ctx`. Every mixin declares `_ctx: GroverContext` as a class-level annotation so type checkers resolve attribute access.

| Mixin | File | Responsibility |
|-------|------|---------------|
| `MountMixin` | `api/mounting.py` | Mount lifecycle: add, unmount, init meta FS |
| `FileOpsMixin` | `api/file_ops.py` | File CRUD, tree, versions, trash, reconciliation |
| `SearchOpsMixin` | `api/search_ops.py` | Queries: glob, grep, vector/lexical/hybrid search |
| `GraphOpsMixin` | `api/graph_ops.py` | Graph queries: predecessors, successors, pagerank |
| `ShareMixin` | `api/sharing.py` | Share/unshare between users |
| `ConnectionMixin` | `api/connections.py` | Manual edge CRUD (persisted through FS) |
| `IndexMixin` | `api/indexing.py` | Event handlers, analysis pipeline, indexing, save, close |

`GroverContext` (`api/context.py`) holds the worker, mount registry, analyzer registry, indexing mode, and helper methods used across all mixins (session management, permission checks, path prefixing, graph resolution via `resolve_graph()`).

**Why mixins?** Each method exists once — no forwarding stubs or delegation boilerplate. The public API is unchanged: `from grover import GroverAsync` works exactly as before.

## Capability protocols

Grover uses a two-tier protocol structure:

```python
class GroverFileSystem(Protocol):     # Full interface: CRUD, queries, versioning,
                                      # trash, search, connections, file chunks
class SupportsReBAC(Protocol):        # Opt-in: relationship-based access control
class SupportsReconcile(Protocol):    # Opt-in: disk ↔ DB reconciliation
```

Every backend must implement the full `GroverFileSystem` protocol. The facade calls `mount.filesystem` methods directly without capability checks.

Two opt-in protocols remain for features that only some backends provide:

- **`SupportsReBAC`** — user-scoped paths and sharing. Only `UserScopedFileSystem` implements this.
- **`SupportsReconcile`** — disk ↔ DB reconciliation. Only `LocalFileSystem` implements this.

These are checked at runtime with `isinstance(backend, SupportsReconcile)`. The facade skips backends that don't implement opt-in protocols (e.g., `reconcile()` silently skips non-reconcilable mounts).

### Provider protocols

Provider protocols are co-located with their provider implementations. Each subdirectory under `providers/` has a `protocol.py` defining its protocol:

```python
class StorageProvider(Protocol):    # Disk I/O: read/write/delete content
class GraphProvider(Protocol):      # Node/edge CRUD + graph queries
class SearchProvider(Protocol):     # Vector + lexical search
class EmbeddingProvider(Protocol):  # Text → vector embedding
class VersionProvider(Protocol):    # Diff-based version storage
class ChunkProvider(Protocol):      # Code chunk CRUD
```

`GraphProvider` is the core graph protocol — `RustworkxGraph` implements it plus the algorithm capability protocols. `SearchProvider` unifies vector and lexical search — `LocalVectorStore`, `PineconeVectorStore`, and `DatabricksVectorStore` implement it directly. `EmbeddingProvider` stays separate from `SearchProvider` because embedding is stateless and shared across operations.

### Graph capability protocols

The graph layer uses capability protocols for optional algorithms:

```python
class GraphProvider(Protocol):           # Core: add/remove/query nodes and edges
class SupportsCentrality(Protocol):      # PageRank, betweenness, closeness, katz, degree
class SupportsConnectivity(Protocol):    # Connected components, is_weakly_connected
class SupportsTraversal(Protocol):       # Ancestors, descendants, topological sort, shortest paths
class SupportsSubgraph(Protocol):        # Subgraph extraction, neighborhood, meeting subgraph
class SupportsFiltering(Protocol):       # Attribute-based node/edge filtering
class SupportsNodeSimilarity(Protocol):  # Jaccard structural similarity
class SupportsPersistence(Protocol):     # SQL persistence (to_sql / from_sql)
```

`RustworkxGraph` implements all protocols. To write a custom graph backend, implement `GraphProvider` plus whichever capabilities you need. `GroverAsync` checks capabilities at runtime with `isinstance()` and raises `CapabilityNotSupportedError` for unsupported operations.

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

## Mount as minimal routing dataclass

`Mount` is a minimal dataclass that binds a virtual path to a filesystem. All capabilities (graph, search, embedding) live on the filesystem itself as **providers**:

```
Mount "/project"
    ├── path: str                      (mount point)
    ├── filesystem: GroverFileSystem   (required — the backend with providers)
    ├── session_factory: ...           (optional — for DB-backed filesystems)
    ├── permission: Permission         (read-write or read-only)
    └── hidden: bool                   (excluded from listing/indexing)
```

Graph, search, and embedding providers are accessed through the filesystem:
- `mount.filesystem.graph_provider` — the per-mount `RustworkxGraph`
- `mount.filesystem.search_provider` — `LocalVectorStore`, `PineconeVectorStore`, etc.
- `mount.filesystem.embedding_provider` — `OpenAIEmbedding`, `LangChainEmbedding`, etc.

There is no `mount.graph`, `mount.search`, or protocol dispatch on Mount. Mount does not check component compatibility — it simply holds the filesystem.

**Provider injection**: `add_mount()` auto-creates a `RustworkxGraph` as `graph_provider` for non-hidden mounts. `embedding_provider` and `search_provider` must be passed explicitly — no auto-creation. If no search providers are configured, search methods return `success=False` with a descriptive message.

```python
# No search — graph and filesystem only
g.add_mount("/project", LocalFileSystem(workspace_dir="."))

# With search — pass providers to add_mount
g.add_mount("/data", engine=engine,
            embedding_provider=OpenAIEmbedding(model="text-embedding-3-small"),
            search_provider=LocalVectorStore(dimension=1536))
```

**Graph resolution**: operations like `predecessors(path)` resolve the mount from the path, then delegate to `mount.filesystem.graph_provider`. `get_graph(path)` is the public method.

**Search routing**: `vector_search()`, `lexical_search()`, and `hybrid_search()` check each mount's filesystem for `search_provider` and `embedding_provider`. Root-level searches aggregate results across all mounts; path-scoped searches target a single mount.

**Lexical search**: `DatabaseFileSystem` provides DB-backed lexical search via SQL LIKE queries. Stores that implement the `SearchProvider` protocol can optionally support `lexical_search()` as well.

Hidden mounts (like `/.grover`) do not receive provider injection.

## Background indexing and consistency

The three layers stay in sync through a `BackgroundWorker`. When a facade method completes a file operation, it schedules a processing function via the worker:

| Operation | Processing method | What it does |
|-----------|------------------|--------------|
| `write()`, `edit()`, `copy()`, `restore_*()` | `_process_write(path, content, user_id)` | Re-analyze file, write chunk DB rows, update graph edges, re-index embeddings |
| `delete()` | `_process_delete(path, user_id)` | Remove file and children from graph, search index, and chunk DB rows |
| `move()` | `_process_move(old, new, user_id)` | `_process_delete(old)` + `_process_write(new)` |
| `add_connection()` | `_process_connection_added(src, tgt, type, weight)` | Add edge to in-memory graph |
| `delete_connection()` | `_process_connection_deleted(src, tgt)` | Remove edge from in-memory graph |

Processing methods accept direct parameters (not event objects). They resolve the mount from the path, then access graph and search providers through the filesystem (`mount.filesystem.graph_provider`, `mount.filesystem.search_provider`). Exceptions are logged but never propagated — a failed re-index should not cause a file write to fail.

### Indexing modes

The `BackgroundWorker` supports two modes, controlled by the `IndexingMode` enum passed to the `Grover`/`GroverAsync` constructor:

**`BACKGROUND` mode (default):** Write/edit/copy/restore operations use `worker.schedule(key, factory)` which dispatches to background `asyncio.Task` instances with **per-path debouncing** — multiple rapid writes to the same file within the `debounce_delay` window (default 0.1s) are coalesced into a single analysis pass using the latest content. This significantly reduces redundant work during burst writes.

Delete and move operations use `worker.cancel(key)` + `worker.schedule_immediate(coro)` — they cancel any pending debounced work for the affected path, then run cleanup immediately. Connection operations also use `schedule_immediate()` since they are lightweight graph updates.

**`MANUAL` mode:** All scheduling is suppressed — `schedule()` and `schedule_immediate()` are no-ops. The caller is responsible for calling `index()` explicitly to populate the graph and search engine. This is useful for batch import scenarios where you write many files first, then index once at the end.

### Post-commit graph projection

`_analyze_and_integrate` collects dependency edges in an `edges_to_project` list during the DB session. After the session commits, edges are projected directly into the in-memory graph via `graph.add_edge()`. This ensures graph edges are only added after the DB commit succeeds (post-commit ordering). "contains" edges (file → chunk) are structural and remain in-memory only — they are not persisted to the database.

### flush() and drain lifecycle

`flush()` (public API) calls `drain()` on the `BackgroundWorker`, which:

1. Fires all pending debounce timers immediately (cancelling the timers and dispatching the work)
2. Awaits all active background tasks
3. Loops until settled — tasks may schedule new work during drain, which is processed in subsequent iterations

`close()` and `save()` automatically call `drain()` before persisting state, so pending work is always processed before shutdown.

`index()` bypasses the worker entirely — it calls `_analyze_and_integrate` directly for each file. `index()` completes all analysis inline regardless of indexing mode.

## Session management

Sessions are owned by VFS, never by backends.

VFS creates a session per operation via `_session_for(mount)`. The session is injected into backend methods as a keyword argument. Backends only call `session.flush()` to make changes visible within the transaction. VFS handles commit (on success) and rollback (on exception).

**LocalFileSystem** manages its own SQLite engine internally (lazy init with `asyncio.Lock`), but session creation is still driven by VFS.

**DatabaseFileSystem** is fully stateless — it holds no engine, no session factory, and no mutable state. It receives everything it needs through the injected session. This makes it safe for concurrent use in web servers.

## Mount-first architecture

All file operations go through mount paths. There is no global filesystem — you mount backends at virtual paths and interact through those paths:

```python
g.add_mount("/code", LocalFileSystem(workspace_dir="."))
g.add_mount("/docs", DatabaseFileSystem(dialect="postgresql"))

g.read("/code/src/main.py")   # → routes to LocalFileSystem
g.read("/docs/guide.md")      # → routes to DatabaseFileSystem
```

`MountRegistry` resolves paths using longest-prefix matching. Mounts are permission boundaries — a read-only mount rejects all writes regardless of the file path. Sub-mount paths can also be marked read-only via `read_only_paths` on the mount config.

**Permission enforcement model**: The facade layer (`GroverAsync`) checks writability via `_check_writable()` before dispatching to backends. This method returns an error message string for read-only paths (or `None` for writable paths) — it never raises exceptions. Every mutation facade method checks the return value and immediately returns a failed Result (`success=False`) with a descriptive message. Multi-mount operations (`empty_trash`, `reconcile`, `index`) silently skip read-only mounts. Read/query operations are never gated. Additional defensive checks exist in `_analyze_and_integrate` (skips connection DB writes for read-only content) and `_walk_and_index` (skips entire read-only mounts during indexing).

Grover also creates a hidden metadata mount at `/.grover` for internal state (graph edges, search metadata). This mount is excluded from indexing and listing.

## Versioning strategy

Versions use a **snapshot + forward diff** approach:

- Version 1 is always a full snapshot.
- Subsequent versions store unified diffs from the previous content.
- Every 20 versions, a fresh snapshot is taken.

To reconstruct version N: find the nearest snapshot at or before N, then apply forward diffs in order. The `content_hash` (SHA-256) on each version record enables integrity verification after reconstruction.

This balances storage efficiency (diffs are small) with reconstruction speed (at most 19 diffs to replay).

**External edit detection:** If a file is modified outside Grover (by an IDE, git, etc.), the diff chain would silently break. At `write()` and `edit()` time, `check_external_edit()` compares the storage content's hash against the last Grover-written hash. On mismatch, a synthetic snapshot version is inserted with `created_by="external"` to keep the chain intact. See [internals/fs.md](internals/fs.md#external-edit-detection) for details.

**Proactive chain verification:** `verify_versions(path)` and `verify_all_versions()` reconstruct every version in a file's chain and verify each SHA-256 hash. Corruption is reported per-version in the `VerifyVersionResult.errors` list rather than raising exceptions — this enables full chain audits that report all problems at once. `reconcile()` calls `verify_all_versions()` automatically and includes a `chain_errors` count in its return dict.

## Graph model

The knowledge graph is an in-memory `rustworkx.PyDiGraph` with string-path-keyed nodes. Edges have a free-form `type` string — there's no enum or schema for edge types. **The graph is a pure in-memory projection** — it is populated from the database on mount init and updated via events at runtime.

Built-in conventions:

| Edge type | Meaning | Persisted? |
|-----------|---------|------------|
| `"imports"` | File imports another file | Yes (through `ConnectionService`) |
| `"contains"` | File contains a chunk (function, class) | No (in-memory only, rebuilt on analysis) |
| `"references"` | File references a symbol in another file | Yes |
| `"inherits"` | Class inherits from another class | Yes |

Code analyzers produce edges automatically. You can also add manual edges via `add_connection()`.

### Filesystem owns connection data

All persistent edges (user-created and analyzer-discovered) are persisted through the filesystem layer via `ConnectionService`, stored in the `grover_file_connections` table. The graph is updated via background worker processing after the DB transaction commits:

1. Caller invokes `add_connection(source, target, type)` on `GroverAsync`
2. `GroverAsync` delegates to the backend's `add_connection()` (part of `GroverFileSystem`)
3. `ConnectionService` writes the record to `grover_file_connections`
4. After the session commits, `_process_connection_added` updates the in-memory graph (`graph.add_edge()`)

On mount init, `from_sql()` loads file nodes from `grover_files` and edges from `grover_file_connections` to populate the graph projection.

**Structural "contains" edges** (file-to-chunk membership) are NOT persisted — they are in-memory only and rebuilt every time a file is analyzed. This keeps the DB clean and avoids conflicts between structural and dependency edges.

**Single-session batching in `_analyze_and_integrate`**: All DB operations within the analysis pipeline (search entry removal, chunk replacement, connection deletion/creation, search indexing) are wrapped in a single `session_for` block. This provides atomicity — if any operation fails, the entire transaction rolls back with no partial state. Edges to project into the in-memory graph are collected in an `edges_to_project` list and projected only after the session commits, ensuring DB records are visible before the graph is updated. The same single-session pattern is used in `_process_delete` and `_process_move` for their cleanup operations.

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

Chunks are stored as database rows in `grover_file_chunks` (via `GroverFileSystem` chunk methods). Their graph node paths use stable synthetic identifiers based on symbol names (not line numbers, which drift on edits). This means chunk paths survive refactoring as long as the symbol name doesn't change.

## Adding a new analyzer

1. Create `src/grover/analyzers/your_language.py`.
2. Implement the `Analyzer` protocol — `analyze_file(path, content) -> AnalysisResult`.
3. Register it in `src/grover/analyzers/__init__.py` with the appropriate file extensions.
4. Add tests in `tests/test_analyzers.py`.

Analyzers should be pure functions of `(path, content)`. They should never raise on malformed input — return an empty result instead.

## User-scoped file systems

Grover supports **user-scoped mounts** where every operation requires a `user_id`. This enables multi-tenant deployments where multiple users share the same database but operate in isolated namespaces.

### UserScopedFileSystem

User scoping is implemented in `UserScopedFileSystem`, a subclass of `DatabaseFileSystem`. VFS is a pure mount router — it passes `user_id` through to the backend, and the backend handles all path rewriting, share checks, and trash scoping.

To create a user-scoped mount, pass a `UserScopedFileSystem` as the backend:

```python
from grover.backends.user_scoped import UserScopedFileSystem

backend = UserScopedFileSystem()
await g.add_mount("/ws", backend, engine=engine)
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

The search layer uses two provider protocols on the filesystem:

- **EmbeddingProvider** — converts text to vectors. Async-first, with `embed()` and `embed_batch()` methods. Built-in providers: OpenAI (API), LangChain adapter (any LangChain `Embeddings` instance).
- **SearchProvider** — stores and searches vectors, with optional lexical search. Async-first. Built-in stores: `LocalVectorStore` (in-process usearch HNSW), `PineconeVectorStore` (Pinecone cloud), `DatabricksVectorStore` (Databricks Vector Search).

Both providers live directly on the filesystem. `DatabaseFileSystem.SearchMethodsMixin` orchestrates them: embed text via `embedding_provider` → store/search vectors via `search_provider.vector_search()`. There is no `SearchEngine` intermediary — the filesystem itself coordinates embedding and search.

**No auto-creation**: Unlike the `graph_provider` (which is auto-injected as a `RustworkxGraph`), search providers are never auto-created. Users must explicitly pass both `search_provider` and `embedding_provider` to `add_mount()`. If either is missing, search operations return `success=False` with a descriptive message.

**Construction-time validation**: When both `embedding_provider` and `search_provider` are set on a filesystem, `_validate_search_dimensions()` checks that the store's dimension (if exposed) matches the provider's `dimensions`. This catches model swaps and dimension mismatches before data is indexed.

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

1. Create `src/grover/providers/embedding/your_provider.py`.
2. Implement the `EmbeddingProvider` protocol: async `embed(text)`, async `embed_batch(texts)`, plus `dimensions` and `model_name` properties.
3. Import-guard any optional dependencies.
4. Add the provider to `src/grover/providers/embedding/__init__.py`.
5. Add tests in `tests/test_embedding_providers.py`.

The provider is passed to `add_mount(..., embedding_provider=...)` at mount time.

## Integration async patterns

All five integration classes (`GroverBackend`, `GroverMiddleware`, `GroverRetriever`, `GroverLoader`, `GroverStore`) accept either `Grover` (sync) or `GroverAsync` (native async). The constructor detects which was passed via `isinstance(grover, GroverAsync)` and sets an `_is_async` flag.

**When `GroverAsync` is passed:**
- Async methods (`aread`, `abatch`, `_aget_relevant_documents`, `alazy_load`) call `GroverAsync` natively with `await`.
- Sync methods (`read`, `batch`, `_get_relevant_documents`, `lazy_load`) wrap async via `asyncio.run()` for convenience. This raises `RuntimeError` if called from within an existing event loop — use async methods instead.

**When `Grover` is passed:**
- Sync methods work directly (unchanged from before).
- Async methods raise `TypeError` directing the user to pass `GroverAsync` or use sync methods.

**Graph operations** (`successors`, `predecessors`) are sync on both `Grover` and `GroverAsync` because they operate on an in-memory `rustworkx` graph. `GroverMiddleware` does NOT pass `coroutine` to `StructuredTool.from_function` for graph tools — they are always sync.

**Formatting helpers** are extracted as module-level functions shared between sync and async code paths. This prevents logic drift and ensures identical output format regardless of which path is used.

**GroverRetriever** is a Pydantic model — it uses `Union[Grover, GroverAsync]` (not `X | Y` with `from __future__ import annotations`) because Pydantic needs the runtime `Union` for validation. The `_is_async` flag is a `@property` rather than an instance variable since Pydantic models manage their own `__init__`.

## Adding a new vector store

1. Create `src/grover/providers/search/your_store.py`.
2. Implement the `SearchProvider` protocol: `upsert()`, `vector_search()`, `delete()`, `fetch()`, `lexical_search()`, `connect()`, `close()`.
3. Add any applicable capability protocols (e.g., `SupportsMetadataFilter`).
4. Import-guard any optional dependencies.
5. Add the store to `src/grover/providers/search/__init__.py`.
6. Add tests in `tests/test_your_store.py`.

The store is passed to `add_mount(..., search_provider=...)` at mount time. Stores that don't support lexical search should return an empty `LexicalSearchResult` from `lexical_search()`.

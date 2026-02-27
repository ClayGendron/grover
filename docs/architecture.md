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

## GroverAsync facade structure

`GroverAsync` is split into mixin classes, each in its own file under `src/grover/facade/`. The main class inherits all mixins and defines only `__init__`:

```
GroverAsync(MountMixin, FileOpsMixin, SearchOpsMixin, GraphOpsMixin,
            VersionTrashMixin, ShareMixin, ConnectionMixin, IndexMixin)
```

Shared state lives in a `GroverContext` dataclass stored as `self._ctx`. Every mixin declares `_ctx: GroverContext` as a class-level annotation so type checkers resolve attribute access.

| Mixin | File | Responsibility |
|-------|------|---------------|
| `MountMixin` | `facade/mounting.py` | Mount lifecycle: add, unmount, init meta FS |
| `FileOpsMixin` | `facade/file_ops.py` | File CRUD: read, write, edit, delete, mkdir, list_dir, move, copy |
| `SearchOpsMixin` | `facade/search_ops.py` | Queries: glob, grep, tree, vector/lexical/hybrid search |
| `GraphOpsMixin` | `facade/graph_ops.py` | Graph queries: dependents, dependencies, impacts, pagerank |
| `VersionTrashMixin` | `facade/version_trash.py` | Versions, trash, reconciliation |
| `ShareMixin` | `facade/sharing.py` | Share/unshare between users |
| `ConnectionMixin` | `facade/connections.py` | Manual edge CRUD (persisted through FS) |
| `IndexMixin` | `facade/indexing.py` | Event handlers, analysis pipeline, indexing, save, close |

`GroverContext` (`facade/context.py`) holds the event bus, mount registry, analyzer registry, embedding/vector config, and helper methods used across all mixins (session management, permission checks, path prefixing, graph/search resolution).

**Why mixins?** Each method exists once — no forwarding stubs or delegation boilerplate. The public API is unchanged: `from grover import GroverAsync` works exactly as before.

## Capability protocols

Not every backend supports every feature. Rather than checking flags or catching `NotImplementedError`, Grover uses runtime-checkable protocols:

```python
class StorageBackend(Protocol):       # Core: read, write, edit, delete, ...
class SupportsVersions(Protocol):     # list_versions, restore_version, ...
class SupportsTrash(Protocol):        # list_trash, restore_from_trash, ...
class SupportsReconcile(Protocol):    # reconcile (sync disk ↔ DB)
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

## Mount as first-class composition unit

`Mount` is the central composition class. Each mount composes three components as **public attributes**:

```
Mount "/project"
    ├── filesystem: LocalFileSystem    (required — storage backend)
    ├── graph: RustworkxGraph          (optional — in-memory knowledge graph)
    └── search: SearchEngine           (optional — vector/lexical search)
```

Graph and search are no longer injected as private attributes on the backend. They live on the `Mount` itself, accessible as `mount.graph` and `mount.search`.

**Protocol dispatch**: Mount checks all three components against dispatch protocols at construction time and builds a dispatch map. If two components implement the same protocol, `ProtocolConflictError` is raised. If no component implements a requested protocol, `ProtocolNotAvailableError` is raised when the method is called.

```python
# Dispatch protocols (mount-level routing)
SupportsGlob          # glob pattern matching → filesystem
SupportsGrep          # regex content search → filesystem
SupportsTree          # directory tree listing → filesystem
SupportsListDir       # directory listing → filesystem
SupportsVectorSearch  # semantic search → search engine
SupportsLexicalSearch # keyword/BM25 search → search engine
SupportsHybridSearch  # combined search → search engine
SupportsEmbedding     # text embedding → search engine
```

**SearchEngine composition**: SearchEngine accepts pluggable components via keyword args:

```python
SearchEngine(
    vector=LocalVectorStore(...),     # → satisfies SupportsVectorSearch
    embedding=OpenAIEmbedding(...),   # → satisfies SupportsEmbedding
    lexical=SQLiteFullText(...),      # → satisfies SupportsLexicalSearch
    hybrid=HybridProvider(...),       # → satisfies SupportsHybridSearch
)
```

SearchEngine exposes `supported_protocols()` which Mount uses instead of `isinstance()` to determine what dispatch protocols the search component satisfies.

**Full-text (BM25) search**: the `lexical` component of SearchEngine provides keyword search via native DB features. Grover auto-detects the dialect and creates the appropriate store:
- SQLite → `SQLiteFullTextStore` (FTS5 virtual table, `bm25()` ranking, `snippet()`)
- PostgreSQL → `PostgresFullTextStore` (`to_tsvector`/`tsquery`, `ts_rank_cd`, GIN index)
- MSSQL → `MSSQLFullTextStore` (`FREETEXTTABLE`, full-text catalog)

FTS stays in sync with content changes: `add()`, `add_batch()`, `remove()`, and `remove_file()` on SearchEngine propagate to both vector and lexical stores. Event handlers pass DB sessions for FTS operations.

**Graph resolution**: operations like `dependents(path)` resolve the mount from the path, then delegate to that mount's graph. `get_graph(path)` is the public method (replaces the removed `.graph` property).

**Search routing**: `search()` routes through VFS, checking `mount.search` on the resolved mount. Root-level searches aggregate results across all mounts; path-scoped searches target a single mount.

**Persistence**: each mount's graph saves to its own database (via `to_sql`/`from_sql`). Search indices save per-mount under `data_dir/search/{mount_slug}/`.

**Embedding provider**: shared across mounts (stateless). Each mount gets its own `LocalVectorStore` by default.

Hidden mounts (like `/.grover`) do not receive graph or search engine injection.

## Event-driven consistency

The three layers stay in sync through an `EventBus`. When VFS completes a file operation, it emits an event:

| Event | Triggers |
|-------|----------|
| `FILE_WRITTEN` | Re-analyze file, write chunk DB rows, update graph edges, re-index embeddings |
| `FILE_DELETED` | Remove file and children from graph, search index, and chunk DB rows |
| `FILE_MOVED` | Remove old path (graph, search, chunks), re-analyze at new path |
| `FILE_RESTORED` | Re-analyze restored file |
| `CONNECTION_ADDED` | Add edge to in-memory graph |
| `CONNECTION_DELETED` | Remove edge from in-memory graph |

Events carry an optional `user_id` field so chunk records and other downstream operations are tagged with the correct owner in user-scoped environments.

Event handlers resolve the mount from the event path, then use that mount's graph and search engine. Exceptions in handlers are logged but never propagated — a failed re-index should not cause a file write to fail.

### Event dispatch and indexing modes

The `EventBus` supports two modes, controlled by the `IndexingMode` enum passed to the `Grover`/`GroverAsync` constructor:

**`BACKGROUND` mode (default):** Events are dispatched to background `asyncio.Task` instances so that `write()`, `edit()`, and other mutating operations return immediately. File mutation events (`FILE_WRITTEN`, `FILE_RESTORED`) are **debounced per-path** — multiple rapid writes to the same file within the `debounce_delay` window (default 0.1s) are coalesced into a single analysis pass using the latest content. This significantly reduces redundant work during burst writes.

`FILE_DELETED` and `FILE_MOVED` events fire immediately (no debounce) and cancel any pending debounced event for the affected path — there's no point analyzing a file that was just deleted or moved. `CONNECTION_ADDED` and `CONNECTION_DELETED` events also fire immediately (they are lightweight graph operations).

**`MANUAL` mode:** All event dispatch is suppressed. `emit()` is a no-op. The caller is responsible for calling `index()` explicitly to populate the graph and search engine. This is useful for batch import scenarios where you write many files first, then index once at the end.

### Nested emit handling

The `_analyze_and_integrate` handler (triggered by `FILE_WRITTEN`) itself emits `CONNECTION_ADDED` events as it discovers import edges. To preserve ordering, a `_dispatching` flag tracks whether we're currently inside a handler. When `emit()` is called from within a running handler (nested emit), the event is dispatched **inline** rather than scheduled as a new background task. This ensures that by the time a `FILE_WRITTEN` handler completes, all its emitted connection events have also been processed.

### flush() and drain lifecycle

`flush()` (public API) calls `drain()` on the EventBus, which:

1. Fires all pending debounce timers immediately (cancelling the timers and dispatching the events)
2. Awaits all active background tasks
3. Loops until settled — handlers may emit new events during drain, which are processed in subsequent iterations

`close()` and `save()` automatically call `drain()` before persisting state, so pending events are always processed before shutdown.

`index()` bypasses the event bus for file events — it calls `_analyze_and_integrate` directly rather than emitting `FILE_WRITTEN` events. However, `_analyze_and_integrate` still emits `CONNECTION_ADDED` events through the event bus as it discovers edges. `index()` completes all analysis inline regardless of indexing mode.

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

All persistent edges (user-created and analyzer-discovered) are persisted through the filesystem layer via `ConnectionService`, stored in the `grover_file_connections` table. The graph is updated via `CONNECTION_ADDED` / `CONNECTION_DELETED` events after the DB transaction commits:

1. Caller invokes `add_connection(source, target, type)` on `GroverAsync`
2. `GroverAsync` delegates to the backend's `add_connection()` (through `SupportsConnections`)
3. `ConnectionService` writes the record to `grover_file_connections`
4. After the session commits, a `CONNECTION_ADDED` event is emitted
5. The event handler updates the in-memory graph (`graph.add_edge()`)

On mount init, `from_sql()` loads file nodes from `grover_files` and edges from `grover_file_connections` to populate the graph projection.

**Structural "contains" edges** (file-to-chunk membership) are NOT persisted — they are in-memory only and rebuilt every time a file is analyzed. This keeps the DB clean and avoids conflicts between structural and dependency edges.

**Single-session batching in `_analyze_and_integrate`**: All DB operations within the analysis pipeline (search entry removal, chunk replacement, connection deletion/creation, search indexing) are wrapped in a single `session_for` block. This provides atomicity — if any operation fails, the entire transaction rolls back with no partial state. `CONNECTION_ADDED` events are collected in a deferred list and emitted only after the session commits, ensuring DB records are visible when event handlers query them. The same single-session pattern is used in `_on_file_deleted` and `_on_file_moved` for their cleanup operations.

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

Chunks are stored as database rows in `grover_file_chunks` (via `SupportsFileChunks` protocol). Their graph node paths use stable synthetic identifiers based on symbol names (not line numbers, which drift on edits). This means chunk paths survive refactoring as long as the symbol name doesn't change.

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

The search layer follows the same two-protocol pattern as the filesystem:

- **EmbeddingProvider** — converts text to vectors. Async-first, with `embed()` and `embed_batch()` methods. Built-in providers: sentence-transformers (local), OpenAI (API), LangChain adapter (any LangChain `Embeddings` instance).
- **VectorStore** — stores and searches vectors. Async-first. Built-in stores: LocalVectorStore (in-process usearch HNSW), PineconeVectorStore (Pinecone cloud), DatabricksVectorStore (Databricks Vector Search).

**SearchEngine** orchestrates them: embed text via provider → store vectors via store → search by embedding queries. `GroverAsync` creates a `SearchEngine` internally and wires it to the `EventBus`.

**Construction-time validation.** `SearchEngine` validates component compatibility at construction time rather than at query time. If a vector store exposes a `dimension` property (duck-typed via `getattr`), the engine checks it matches the embedding provider's `dimensions`. A declared `model_name` is cross-checked against the provider's `model_name`. This catches model swaps and dimension mismatches before any data is indexed.

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

## Integration async patterns

All five integration classes (`GroverBackend`, `GroverMiddleware`, `GroverRetriever`, `GroverLoader`, `GroverStore`) accept either `Grover` (sync) or `GroverAsync` (native async). The constructor detects which was passed via `isinstance(grover, GroverAsync)` and sets an `_is_async` flag.

**When `GroverAsync` is passed:**
- Async methods (`aread`, `abatch`, `_aget_relevant_documents`, `alazy_load`) call `GroverAsync` natively with `await`.
- Sync methods (`read`, `batch`, `_get_relevant_documents`, `lazy_load`) wrap async via `asyncio.run()` for convenience. This raises `RuntimeError` if called from within an existing event loop — use async methods instead.

**When `Grover` is passed:**
- Sync methods work directly (unchanged from before).
- Async methods raise `TypeError` directing the user to pass `GroverAsync` or use sync methods.

**Graph operations** (`dependencies`, `dependents`, `impacts`) are sync on both `Grover` and `GroverAsync` because they operate on an in-memory `rustworkx` graph. `GroverMiddleware` does NOT pass `coroutine` to `StructuredTool.from_function` for graph tools — they are always sync.

**Formatting helpers** are extracted as module-level functions shared between sync and async code paths. This prevents logic drift and ensures identical output format regardless of which path is used.

**GroverRetriever** is a Pydantic model — it uses `Union[Grover, GroverAsync]` (not `X | Y` with `from __future__ import annotations`) because Pydantic needs the runtime `Union` for validation. The `_is_async` flag is a `@property` rather than an instance variable since Pydantic models manage their own `__init__`.

## Adding a new vector store

1. Create `src/grover/search/stores/your_store.py`.
2. Implement the `VectorStore` protocol: `upsert()`, `search()`, `delete()`, `fetch()`, `connect()`, `close()`, and `index_name` property.
3. Add any applicable capability protocols (e.g., `SupportsMetadataFilter`).
4. Import-guard any optional dependencies.
5. Add the store to `src/grover/search/stores/__init__.py`.
6. Add tests in `tests/test_your_store.py`.

The store is passed to `Grover(vector_store=...)` at construction time.

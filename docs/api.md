# API Reference

This is the complete API reference for Grover. For a quick overview, see the [Home](index.md).

---

## Grover / GroverAsync

The main entry points. `Grover` is a thread-safe synchronous wrapper around `GroverAsync`. Both expose the same API — `Grover` methods are synchronous, `GroverAsync` methods are `async`.

```python
from grover import Grover, GroverAsync
```

### Constructor

```python
Grover(*, indexing_mode=IndexingMode.BACKGROUND, debounce_delay=0.1)
GroverAsync(*, indexing_mode=IndexingMode.BACKGROUND, debounce_delay=0.1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `indexing_mode` | `IndexingMode` | Controls how file mutations are dispatched to indexing handlers. `BACKGROUND` (default): mutations are debounced per-path and processed in background tasks so `write()`/`edit()` return immediately. `MANUAL`: all scheduling is suppressed; only an explicit call to `index()` populates the graph and search index. |
| `debounce_delay` | `float` | Seconds to wait before dispatching a debounced task (default `0.1`). Only applies in `BACKGROUND` mode. Multiple rapid writes to the same path within this window are coalesced into a single analysis pass. |

Search providers (`embedding_provider`, `search_provider`) are configured per-mount via `add_mount()`, not on the constructor. See [Mount / Unmount](#mount--unmount) below.

### Mount / Unmount

```python
g.add_mount(path, filesystem=None, *, engine=None, session_factory=None,
            dialect="sqlite", file_model=None, file_version_model=None,
            file_chunk_model=None, db_schema=None, mount_type=None,
            permission=Permission.READ_WRITE, label="", hidden=False,
            embedding_provider=None, search_provider=None)
g.unmount(path)
```

Mount a storage backend at a virtual path. You can pass either:

- A `filesystem` object (e.g., `LocalFileSystem`, `DatabaseFileSystem`)
- An `engine` (SQLAlchemy `AsyncEngine`) — Grover will create a `DatabaseFileSystem` automatically
- A `session_factory` — same as engine, but you control session creation
- A `Mount` object directly

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Virtual mount path (e.g., `"/project"`) |
| `filesystem` | `GroverFileSystem | None` | `None` | Pre-created backend instance |
| `engine` | `AsyncEngine | None` | `None` | SQLAlchemy async engine (creates DatabaseFileSystem) |
| `session_factory` | `Callable[..., AsyncSession] | None` | `None` | Custom session factory |
| `dialect` | `str` | `"sqlite"` | Database dialect (`"sqlite"`, `"postgresql"`, `"mssql"`) |
| `file_model` | `type | None` | `None` | Custom SQLModel file table class |
| `file_version_model` | `type | None` | `None` | Custom SQLModel file version table class |
| `file_chunk_model` | `type | None` | `None` | Custom SQLModel file chunk table class |
| `db_schema` | `str | None` | `None` | Database schema name |
| `mount_type` | `str | None` | `None` | Mount type label (auto-detected if `None`) |
| `permission` | `Permission` | `READ_WRITE` | `Permission.READ_WRITE` or `Permission.READ_ONLY` |
| `label` | `str` | `""` | Human-readable mount label |
| `hidden` | `bool` | `False` | Hidden mounts are excluded from listing and indexing |
| `embedding_provider` | `EmbeddingProvider | None` | `None` | Embedding provider for search (e.g., `OpenAIEmbedding`). Required for vector search. |
| `search_provider` | `SearchProvider | None` | `None` | Search backend (e.g., `LocalVectorStore`, `PineconeVectorStore`). Required for vector search. |

**Provider injection**: `add_mount()` auto-creates a `RustworkxGraph` as graph provider for non-hidden mounts. Search providers are never auto-created — you must pass both `embedding_provider` and `search_provider` to enable vector search. If either is missing, search operations return `success=False`.

For user-scoped mounts, pass a `UserScopedFileSystem` as the filesystem (see [architecture.md](architecture.md#user-scoped-file-systems)).

### Filesystem Operations

All filesystem methods accept an optional `user_id` keyword argument. On user-scoped mounts (using `UserScopedFileSystem`), `user_id` is **required** — paths are automatically namespaced per user (e.g., `/notes.md` → `/{user_id}/notes.md` in the backend). On regular mounts, `user_id` is accepted but ignored.

```python
g.read(path, *, user_id=None) -> ReadResult
g.write(path, content, *, user_id=None) -> WriteResult
g.edit(path, old, new, *, user_id=None) -> EditResult
g.delete(path, permanent=False, *, user_id=None) -> DeleteResult
g.list_dir(path="/", *, user_id=None) -> ListDirResult
g.exists(path, *, user_id=None) -> ExistsResult
g.move(src, dest, *, user_id=None, follow=False) -> MoveResult
g.copy(src, dest, *, user_id=None) -> WriteResult
```

| Method | Description |
|--------|-------------|
| `read(path)` | Read file content. Returns `ReadResult` with `success`, `content`, `total_lines`, etc. |
| `write(path, content)` | Write content to a file. Creates the file if it doesn't exist, creates a new version if it does. Returns `WriteResult` with `success`, `created`, `version`. |
| `edit(path, old, new)` | Find-and-replace within a file. Returns `EditResult` with `success` and `version`. |
| `delete(path, permanent=False)` | Delete a file. Default is soft-delete (moves to trash). Pass `permanent=True` for permanent deletion. Returns `DeleteResult`. |
| `list_dir(path)` | List directory entries. Returns candidates with `path` and `is_directory`. On user-scoped mounts, the user's root listing includes a virtual `@shared/` entry. |
| `exists(path)` | Check if a path exists. Returns `ExistsResult` with `exists: bool`. |
| `move(src, dest, *, follow=False)` | Move a file or directory. Default (`follow=False`) creates a clean break — new file record at dest, source soft-deleted, no version history carryover. `follow=True` does an in-place rename — same file record, versions follow, share paths updated. Returns `MoveResult`. |
| `copy(src, dest)` | Copy a file to a new path. Returns `WriteResult`. |

#### Move semantics: `follow=True` vs `follow=False`

The `follow` parameter controls how `move()` handles identity and history:

**`follow=False` (default) — clean break.** Creates a brand-new file record at the destination. The source is soft-deleted. Version history stays with the old path (accessible via trash/restore). Use this when you want a fresh start at the new location.

```python
g.move("/project/old.py", "/project/new.py")
# /project/old.py → soft-deleted (in trash)
# /project/new.py → new file record, version 1
# Version history for old.py is preserved in trash
```

**`follow=True` — in-place rename.** The file record itself is updated to the new path. Version history, file ID, and shares all follow the file. Use this when renaming or reorganizing and you want continuity.

```python
g.move("/project/old.py", "/project/new.py", follow=True)
# /project/old.py → gone (record updated, not deleted)
# /project/new.py → same file record, same versions, same shares
# Share paths are automatically updated
```

### Search / Query

```python
g.glob(pattern, path="/") -> GlobResult
g.grep(pattern, path="/", *, ...) -> GrepResult
g.tree(path="/", *, max_depth=None) -> TreeResult
```

| Method | Description |
|--------|-------------|
| `glob(pattern, path)` | Find files matching a glob pattern. Supports `*` (single segment), `**` (recursive), `?` (single char), `[seq]` (character class), `[!seq]` (negated). Returns `GlobResult` with `candidates` (list of `FileSearchCandidate`). |
| `grep(pattern, path, ...)` | Search file contents with regex. Returns `GrepResult` with `candidates` (list of `FileSearchCandidate`). Each candidate's evidence includes `GrepEvidence` with `line_matches`. |
| `tree(path, max_depth)` | List all entries recursively. Returns `TreeResult` with `entries`, `total_files`, `total_dirs`. |

**grep options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | required | Regex pattern (or literal if `fixed_string=True`) |
| `path` | `str` | `"/"` | Directory or file to search |
| `glob_filter` | `str | None` | `None` | Only search files matching this glob pattern |
| `case_sensitive` | `bool` | `True` | Case-sensitive matching |
| `fixed_string` | `bool` | `False` | Treat pattern as literal string, not regex |
| `invert` | `bool` | `False` | Return non-matching lines |
| `word_match` | `bool` | `False` | Match whole words only (`\b` boundaries) |
| `context_lines` | `int` | `0` | Lines of context before/after each match |
| `max_results` | `int` | `1000` | Maximum matches returned (0 = unlimited) |
| `max_results_per_file` | `int` | `0` | Maximum matches per file (0 = unlimited) |
| `count_only` | `bool` | `False` | Return count in message, no match details |
| `files_only` | `bool` | `False` | One match per file (file listing mode) |

### Versioning

```python
g.list_versions(path) -> VersionResult
g.get_version_content(path, version) -> GetVersionContentResult
g.restore_version(path, version) -> RestoreResult
g.verify_versions(path) -> VerifyVersionResult
g.verify_all_versions(mount_path=None) -> list[VerifyVersionResult]
```

| Method | Description |
|--------|-------------|
| `list_versions(path)` | List all versions of a file. Returns `VersionResult` (a `FileSearchResult` subclass) with `candidates` — each candidate's path is `"{file_path}@{version}"` and evidence is `VersionEvidence` with `version`, `content_hash`, `size_bytes`, `created_at`, `created_by`. Versions with `created_by="external"` are synthetic snapshots auto-inserted when an external edit was detected. |
| `get_version_content(path, version)` | Retrieve the content of a specific version. Returns `GetVersionContentResult`. |
| `restore_version(path, version)` | Restore a file to a previous version (creates a new version with the old content). Returns `RestoreResult`. |
| `verify_versions(path)` | Verify the version chain integrity for a single file. Reconstructs every version and checks SHA256 hashes. Returns `VerifyVersionResult` with per-version pass/fail details in `errors: list[VersionChainError]`. |
| `verify_all_versions(mount_path=None)` | Verify version chains for all files, optionally filtered to a specific mount. Returns `list[VerifyVersionResult]`. |

### Trash

```python
g.list_trash(*, user_id=None) -> TrashResult
g.restore_from_trash(path, *, user_id=None) -> RestoreResult
g.empty_trash(*, user_id=None) -> DeleteResult
```

| Method | Description |
|--------|-------------|
| `list_trash()` | List all soft-deleted files across all mounts. On user-scoped mounts, scoped to the requesting user's files only. |
| `restore_from_trash(path)` | Restore a previously deleted file by its original path. On user-scoped mounts, only the file owner can restore. Returns `RestoreResult`. |
| `empty_trash()` | Permanently delete trashed files. On user-scoped mounts, only deletes the requesting user's trashed files. Returns `DeleteResult`. |

### Sharing

Available on mounts whose backend implements `SupportsReBAC` (e.g., `UserScopedFileSystem`). Share files or directories with other users.

```python
g.share(path, grantee_id, permission="read", *, user_id) -> ShareResult
g.unshare(path, grantee_id, *, user_id) -> ShareResult
g.list_shares(path, *, user_id) -> ShareSearchResult
g.list_shared_with_me(*, user_id) -> ShareSearchResult
```

| Method | Description |
|--------|-------------|
| `share(path, grantee_id, permission, *, user_id)` | Share a file or directory. `permission` is `"read"` or `"write"`. Only the file owner can create shares. Returns `ShareResult`. |
| `unshare(path, grantee_id, *, user_id)` | Remove a share. Returns `ShareResult`. |
| `list_shares(path, *, user_id)` | List all shares on a given path. Returns `ShareSearchResult` (a `FileSearchResult` subclass) with `candidates` containing `ShareEvidence`. |
| `list_shared_with_me(*, user_id)` | List all files shared with the current user. Paths are returned with `@shared/{owner}/` prefix. Returns `ShareSearchResult`. |

Shared files are accessible via the `@shared/` virtual namespace:

```python
# Alice shares a file with Bob
g.share("/ws/notes.md", "bob", user_id="alice")

# Bob reads it via @shared/
content = g.read("/ws/@shared/alice/notes.md", user_id="bob")

# Bob can browse @shared/ like a directory
shared_owners = g.list_dir("/ws/@shared", user_id="bob")
alice_files = g.list_dir("/ws/@shared/alice", user_id="bob")
```

Directory shares grant access to all descendants (prefix matching).

### Reconciliation

```python
g.reconcile(mount_path=None) -> ReconcileResult
```

Synchronize the database with the actual filesystem state. Only available for backends that implement `SupportsReconcile` (currently `LocalFileSystem`).

Returns a `ReconcileResult` with fields: `created`, `updated`, `deleted`, `chain_errors` (all `int`). The `chain_errors` count is the total number of version records that failed integrity verification across all reconciled files.

### Graph Queries

```python
g.dependencies(path) -> GraphResult
g.dependents(path) -> GraphResult
g.impacts(path, max_depth=3) -> GraphResult
g.path_between(source, target) -> GraphResult
g.contains(path) -> GraphResult
```

| Method | Description |
|--------|-------------|
| `dependencies(path)` | Files that this file depends on (outgoing edges). |
| `dependents(path)` | Files that depend on this file (incoming edges). |
| `impacts(path, max_depth=3)` | Transitive reverse reachability — all files that could be affected by a change to this file. BFS with cycle detection, bounded by `max_depth`. |
| `path_between(source, target)` | Shortest path between two files using Dijkstra (weight-aware). Returns `None` if no path exists. |
| `contains(path)` | Chunks (functions, classes) contained in this file. Returns nodes connected by `"contains"` edges. |

### Connection Operations

```python
g.add_connection(source_path, target_path, connection_type, *, weight=1.0) -> ConnectionResult
g.delete_connection(source_path, target_path, *, connection_type=None) -> ConnectionResult
g.list_connections(path, *, direction="both", connection_type=None) -> ConnectionListResult
```

Connections are persisted through the filesystem layer (not directly on the graph). The in-memory graph is updated via background worker processing after the DB transaction commits. This makes the filesystem the single source of truth for all persistent edge data.

| Method | Description |
|--------|-------------|
| `add_connection(source, target, type)` | Create or update a connection. Returns `ConnectionResult`. The connection path identity is `source[type]target`. |
| `delete_connection(source, target)` | Delete a connection. If `connection_type` is `None`, deletes all connections between source and target. |
| `list_connections(path)` | List connections for a path. `direction` can be `"out"`, `"in"`, or `"both"` (default). Returns `ConnectionListResult` with `connections: list`. |

`ConnectionResult` fields: `success`, `message`, `path`, `source_path`, `target_path`, `connection_type`.

### Graph Algorithms

These convenience methods delegate to the graph backend. They raise `CapabilityNotSupportedError` if the backend doesn't support the required capability protocol.

Graph operations resolve to the per-mount graph for the given path. `pagerank()` and `find_nodes()` accept an optional `path` parameter to select which mount's graph to use (defaults to the first visible mount).

```python
g.pagerank(*, personalization=None, path=None) -> GraphResult
g.ancestors(path) -> GraphResult
g.descendants(path) -> GraphResult
g.meeting_subgraph(paths, *, max_size=50) -> GraphResult
g.neighborhood(path, *, max_depth=2, direction="both", edge_types=None) -> GraphResult
g.find_nodes(*, path=None, **attrs) -> GraphResult
```

| Method | Protocol | Description |
|--------|----------|-------------|
| `pagerank(personalization=None, path=None)` | `SupportsCentrality` | PageRank scores for all nodes. Optional `personalization` dict biases the random walk. `path` selects which mount's graph. |
| `ancestors(path)` | `SupportsTraversal` | All transitive predecessors of a node. |
| `descendants(path)` | `SupportsTraversal` | All transitive successors of a node. |
| `meeting_subgraph(paths, max_size=50)` | `SupportsSubgraph` | Subgraph connecting multiple nodes via shortest paths, scored by PageRank. Pruned to `max_size` nodes. |
| `neighborhood(path, max_depth=2, direction="both", edge_types=None)` | `SupportsSubgraph` | BFS neighborhood around a node. `direction`: `"out"`, `"in"`, or `"both"`. |
| `find_nodes(path=None, **attrs)` | `SupportsFiltering` | Find nodes by attribute predicates. `path` selects which mount's graph. Callable values are used as predicates; non-callable values are matched by equality. |

### Search

```python
g.vector_search(query, k=10, *, path="/", user_id=None) -> VectorSearchResult
g.lexical_search(query, k=10, *, path="/", user_id=None) -> LexicalSearchResult
g.hybrid_search(query, k=10, *, alpha=0.5, path="/", user_id=None) -> FileSearchResult
g.search(query, *, path="/", glob=None, grep=None, k=10, user_id=None) -> FileSearchResult
```

| Method | Description |
|--------|-------------|
| `vector_search(query, k)` | Semantic search using embedding + vector store. Requires `embedding_provider` and `search_provider` on the mount. Returns `VectorSearchResult`. |
| `lexical_search(query, k)` | BM25/full-text keyword search via the filesystem's DB-backed lexical search. Returns `LexicalSearchResult`. |
| `hybrid_search(query, k, alpha)` | Combines vector and lexical results. `alpha` controls the blend: 1.0 = pure vector, 0.0 = pure lexical. Falls back to whichever is available. |
| `search(query, ...)` | Composable search pipeline: optional glob/grep filters followed by vector search. Use `glob` and `grep` to pre-filter before vector search. |

Search is routed through per-mount filesystem providers. When `path="/"`, results are aggregated across all mounts. When `path` targets a specific mount, search is scoped to that mount.

Returns `success=False` results if the required providers (`search_provider`, `embedding_provider`) are not configured on the mount.

### Index and Persistence

```python
g.index(mount_path=None) -> dict[str, int]
g.flush()
g.save()
g.close()
```

| Method | Description |
|--------|-------------|
| `index(mount_path)` | Walk the filesystem, analyze all files, build the knowledge graph and search index. Pass a `mount_path` to index a single mount, or `None` for all visible mounts. Returns stats: `{"files_scanned": N, "chunks_created": N, "edges_added": N, "files_skipped": N}`. Runs inline regardless of `indexing_mode`. |
| `flush()` | Wait for all pending background indexing to complete. In `BACKGROUND` mode, drains the debounce queue and awaits all active analysis tasks. In `MANUAL` mode, this is a no-op. Call before querying if you need guaranteed consistency after recent writes. |
| `save()` | Persist the search index to disk. Automatically drains pending events before saving. Graph edges are persisted through `ConnectionService` at write time, not during `save()`. |
| `close()` | Save state and shut down all subsystems. Automatically drains pending events before saving. Idempotent. |

### Properties and Methods

```python
g.get_graph(path=None) -> GraphProvider   # Per-mount knowledge graph
```

`get_graph(path)` returns the graph provider for the mount owning `path`. If `path` is `None`, returns the first available mount's graph. Each mount has its own isolated `RustworkxGraph` instance, auto-injected at mount time. The graph is accessed through `mount.filesystem.graph_provider`.

---

## Key Types

### Ref

```python
from grover import Ref
```

Immutable identity for any Grover entity. A thin wrapper around a single path string that supports four synthetic path formats:

| Entity | Format | Example |
|--------|--------|---------|
| File | plain path | `/src/auth.py` |
| Chunk | `file#symbol` | `/src/auth.py#login` |
| Version | `file@N` | `/src/auth.py@3` |
| Connection | `source[type]target` | `/src/auth.py[imports]/src/utils.py` |

**Type checks** (mutually exclusive):

| Property | Returns `True` when |
|----------|-------------------|
| `ref.is_file` | Plain file path (no suffix) |
| `ref.is_chunk` | Path contains `#symbol` |
| `ref.is_version` | Path contains `@N` |
| `ref.is_connection` | Path contains `[type]` |

**Decomposition properties:**

| Property | File | Chunk | Version | Connection |
|----------|------|-------|---------|------------|
| `base_path` | path | file path | file path | source path |
| `chunk` | `None` | symbol name | `None` | `None` |
| `version` | `None` | `None` | version int | `None` |
| `source` | `None` | `None` | `None` | source path |
| `target` | `None` | `None` | `None` | target path |
| `connection_type` | `None` | `None` | `None` | type string |

**Factory classmethods:**

```python
Ref.for_chunk("/src/auth.py", "login")           # Ref('/src/auth.py#login')
Ref.for_version("/src/auth.py", 3)               # Ref('/src/auth.py@3')
Ref.for_connection("/a.py", "/b.py", "imports")   # Ref('/a.py[imports]/b.py')
```

### IndexingMode

```python
from grover import IndexingMode
```

Controls how file mutation events are dispatched to indexing handlers.

| Value | Behavior |
|-------|----------|
| `IndexingMode.BACKGROUND` | Default. Events are debounced per-path and dispatched in background `asyncio.Task` instances. `write()` and `edit()` return immediately; indexing happens asynchronously. Call `flush()` before querying if you need guaranteed consistency. |
| `IndexingMode.MANUAL` | All event dispatch is suppressed. Only an explicit call to `index()` populates the graph and search engine. Useful for batch import scenarios where you want to write many files first, then index once. |

```python
# Background mode (default) — writes are fast, index in the background
g = Grover()

# Manual mode — full control over when indexing happens
g = Grover(indexing_mode=IndexingMode.MANUAL)
```

### SearchResult (internal)

```python
from grover.providers.search.types import SearchResult
```

Internal type used by the filesystem's `SearchMethodsMixin` and vector store backends. The public `Grover.search()` API returns `FileSearchResult` (see [Result Types](#result-types) below), which wraps these into `FileSearchCandidate` objects with `VectorEvidence`.

```python
@dataclass(frozen=True)
class SearchResult:
    ref: Ref                    # The matched file or chunk
    score: float                # Cosine similarity (0–1)
    content: str                # The matched text
    parent_path: str | None     # Parent file path (for chunks)
```

### Result Types

Grover has two result families:

- **`FileOperationResult`** — base for content operations (read, write, edit, delete, etc.). Enriched base with `path`, `content`, `message`, `success`, `line_start`, `line_offset`, `version`.
- **`FileSearchResult`** — base for search/query results. Contains `candidates: list[FileSearchCandidate]` where each candidate has a `path` and `evidence` list. Supports set algebra (`&`, `|`, `-`, `>>`), `rebase()`, `remap_paths()`.

All result types live in `grover.results` (canonical location).

```python
from grover import (
    # Operation results
    FileOperationResult, ReadResult, WriteResult, EditResult,
    DeleteResult, MkdirResult, MoveResult, RestoreResult,
    GetVersionContentResult, ShareResult, ConnectionResult,
    FileInfoResult, ExistsResult, ReconcileResult,
    ChunkResult, ChunkListResult, ConnectionListResult,
    # Search results
    FileSearchResult, FileSearchCandidate, Evidence,
    GlobResult, GrepResult, TreeResult, ListDirResult,
    TrashResult, VersionResult, ShareSearchResult,
    VectorSearchResult, LexicalSearchResult, HybridSearchResult,
    GraphResult, LineMatch,
    # Filter operators
    FilterValue, eq, ne, gt, gte, lt, lte, in_, not_in, and_, or_, exists,
)
```

Every result has a `success: bool` and `message: str` field. Always check `success` before using other fields — this is the primary error handling pattern in Grover:

```python
result = g.read("/project/missing.py")
if result.success:
    print(result.content)       # safe to access
else:
    print(result.message)       # e.g., "File not found: /project/missing.py"
    # result.content is "" — don't use it

result = g.write("/project/hello.py", "content")
if result.success:
    print(f"Version {result.version}, created={result.created}")
```

This design is intentional: agents running in loops should handle failures gracefully without try/except blocks. Operations never raise exceptions for expected failures (missing files, permission errors, etc.) — they return a result with `success=False` and a descriptive `message`.

#### FileOperationResult subclasses

| Type | Key Fields (beyond base) |
|------|------------|
| `ReadResult` | `total_lines`, `lines_read`, `truncated` |
| `WriteResult` | `created` (bool) |
| `EditResult` | (base fields sufficient) |
| `DeleteResult` | `permanent` (bool), `total_deleted` |
| `MkdirResult` | `created_dirs` (list) |
| `MoveResult` | `old_path`, `new_path` |
| `RestoreResult` | `restored_version` |
| `GetVersionContentResult` | (base has `content` + `version`) |
| `ShareResult` | `grantee_id`, `permission`, `granted_by` |
| `ConnectionResult` | `source_path`, `target_path`, `connection_type` |
| `FileInfoResult` | `is_directory`, `mime_type`, `size_bytes`, `created_at`, `updated_at`, `permission`, `mount_type`. `get_info()` always returns this (never `None`); check `success` for not-found. |
| `ExistsResult` | `exists` (bool) |
| `ReconcileResult` | `created`, `updated`, `deleted`, `chain_errors` (all int) |
| `ChunkResult` | `count` (int) |
| `ChunkListResult` | `chunks` (list) |
| `ConnectionListResult` | `connections` (list) |
| `VerifyVersionResult` | `versions_checked`, `versions_passed`, `versions_failed`, `errors: list[VersionChainError]` |

`VersionChainError` is a frozen dataclass with fields: `version`, `expected_hash`, `actual_hash`, `error`.

#### FileSearchResult subclasses

Each search result contains `candidates: list[FileSearchCandidate]`, where each candidate has a `path: str` and `evidence: list[Evidence]`.

| Type | Evidence Type | Key Evidence Fields |
|------|--------------|-------------------|
| `GlobResult` | `GlobEvidence` | `is_directory`, `size_bytes`, `mime_type` |
| `GrepResult` | `GrepEvidence` | `line_matches: list[LineMatch]` |
| `TreeResult` | (mixed) | `total_files`, `total_dirs` on result |
| `ListDirResult` | `ListDirEvidence` | `is_directory`, `size_bytes`, `mime_type` |
| `TrashResult` | `TrashEvidence` | `deleted_at`, `deleted_by` |
| `VersionResult` | `VersionEvidence` | `version`, `content_hash`, `size_bytes`, `created_at`, `created_by` |
| `ShareSearchResult` | `ShareEvidence` | `grantee_id`, `permission`, `granted_by`, `expires_at` |
| `VectorSearchResult` | `VectorEvidence` | `snippet` |
| `LexicalSearchResult` | `LexicalEvidence` | `score`, `snippet` |
| `HybridSearchResult` | (mixed) | Vector + lexical evidence |
| `GraphResult` | `GraphEvidence` | `edge_type`, `direction`, `weight` |

#### Query results (frozen, tuple-based)

| Type | Key Fields |
|------|------------|
| `LineMatch` | `line_number`, `line_content` |

---

## Filesystem Layer

```python
from grover.backends import (
    LocalFileSystem,
    DatabaseFileSystem,
    GroverFileSystem,
    SupportsReBAC,
    SupportsReconcile,
)
from grover.mount import MountRegistry
from grover.permissions import Permission
```

### LocalFileSystem

```python
LocalFileSystem(workspace_dir, *, data_dir=None)
```

Stores files on disk at `workspace_dir`. Metadata and version history live in a SQLite database at `data_dir` (defaults to `~/.grover/{workspace_slug}/`).

Implements: `GroverFileSystem`, `SupportsReconcile`.

### DatabaseFileSystem

```python
DatabaseFileSystem(*, dialect="sqlite", file_model=None,
                   file_version_model=None, schema=None)
```

Pure-database storage. All content lives in the `File.content` column. Stateless — requires a session to be injected by VFS.

Implements: `GroverFileSystem`.

### Permission

```python
from grover.permissions import Permission

Permission.READ_WRITE  # Full access (default)
Permission.READ_ONLY   # Reads and listings only
```

**Read-only enforcement**: When a mount (or path within a mount via `read_only_paths`) is set to `READ_ONLY`, all mutations return a failed Result with `success=False` and a descriptive message — no exceptions are raised. This applies to: `write`, `edit`, `delete`, `mkdir`, `move`, `copy`, `restore_version`, `restore_from_trash`, `share`, `unshare`, `add_connection`, `delete_connection`. Multi-mount operations (`empty_trash`, `reconcile`, `index`) silently skip read-only mounts. Read and query operations (`read`, `glob`, `grep`, `list_dir`, `tree`, `list_connections`, `list_shares`, graph queries, search) are unaffected.

### Protocols

| Protocol | Methods |
|----------|---------|
| `GroverFileSystem` | CRUD (`read`, `write`, `edit`, `delete`, `mkdir`, `move`, `copy`), queries (`list_dir`, `exists`, `get_info`, `glob`, `grep`, `tree`), versioning (`list_versions`, `get_version_content`, `restore_version`, `verify_versions`, `verify_all_versions`), trash (`list_trash`, `restore_from_trash`, `empty_trash`), search (`vector_search`, `lexical_search`, `search_add_batch`, `search_remove_file`), connections (`add_connection`, `delete_connection`, `list_connections`), chunks (`replace_file_chunks`, `delete_file_chunks`, `list_file_chunks`) |
| `SupportsReBAC` | `share`, `unshare`, `list_shares_on_path`, `list_shared_with_me` |
| `SupportsReconcile` | `reconcile` |

All protocols are `runtime_checkable`. Every backend must implement `GroverFileSystem`. The two opt-in protocols are for features only some backends provide.

### Exceptions

```python
from grover.exceptions import (
    GroverError,                    # Base exception
    PathNotFoundError,              # File or directory not found
    MountNotFoundError,             # No mount matches the path
    StorageError,                   # Backend I/O failure
    ConsistencyError,               # Metadata/content mismatch
    CapabilityNotSupportedError,    # Backend doesn't support this operation
    AuthenticationRequiredError,    # user_id missing on user-scoped mount
)
```

---

## RustworkxGraph

```python
from grover.providers.graph import RustworkxGraph
```

In-memory directed graph backed by `rustworkx.PyDiGraph`. Nodes are file paths (strings), edges have a free-form type string. Implements the `GraphProvider` protocol plus all capability protocols.

### Node Operations

```python
graph.add_node(path, **attrs)
graph.remove_node(path)
graph.has_node(path) -> bool
graph.get_node(path) -> dict
graph.nodes() -> list[str]
```

### Edge Operations

```python
graph.add_edge(source, target, edge_type, weight=1.0, edge_id=None, **attrs)
graph.remove_edge(source, target)
graph.has_edge(source, target) -> bool
graph.get_edge(source, target) -> dict
graph.edges() -> list[tuple[str, str, dict]]
```

### Query Methods

```python
graph.dependents(path) -> list[Ref]       # Incoming edges (predecessors)
graph.dependencies(path) -> list[Ref]     # Outgoing edges (successors)
graph.impacts(path, max_depth=3) -> list[Ref]  # Transitive BFS
graph.path_between(source, target) -> list[Ref] | None  # Dijkstra
graph.contains(path) -> list[Ref]         # "contains" edges only
graph.by_parent(path) -> list[Ref]        # Nodes with matching parent_path
```

### Centrality (`SupportsCentrality`)

```python
graph.pagerank(*, alpha=0.85, personalization=None, max_iter=100, tol=1e-6) -> dict[str, float]
graph.betweenness_centrality(*, normalized=True) -> dict[str, float]
graph.closeness_centrality() -> dict[str, float]
graph.katz_centrality(*, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6) -> dict[str, float]
graph.degree_centrality() -> dict[str, float]
graph.in_degree_centrality() -> dict[str, float]
graph.out_degree_centrality() -> dict[str, float]
```

### Connectivity (`SupportsConnectivity`)

```python
graph.weakly_connected_components() -> list[set[str]]
graph.strongly_connected_components() -> list[set[str]]
graph.is_weakly_connected() -> bool
```

### Traversal (`SupportsTraversal`)

```python
graph.ancestors(path) -> set[str]
graph.descendants(path) -> set[str]
graph.all_simple_paths(source, target, *, cutoff=None) -> list[list[str]]
graph.topological_sort() -> list[str]                   # Raises ValueError on cycles
graph.shortest_path_length(source, target) -> float | None
```

### Subgraph Extraction (`SupportsSubgraph`)

```python
graph.subgraph(paths) -> SubgraphResult
graph.neighborhood(path, *, max_depth=2, direction="both", edge_types=None) -> SubgraphResult
graph.meeting_subgraph(start_paths, *, max_size=50) -> SubgraphResult
graph.common_reachable(paths, *, direction="forward") -> set[str]
graph.remove_file_subgraph(path)  # Remove node + all children
```

### Filtering (`SupportsFiltering`)

```python
graph.find_nodes(**attrs) -> list[str]                  # Callable or equality predicates
graph.find_edges(*, edge_type=None, source=None, target=None) -> list[tuple[str, str, dict]]
graph.edges_of(path, *, direction="both", edge_types=None) -> list[tuple[str, str, dict]]
```

### Node Similarity (`SupportsNodeSimilarity`)

```python
graph.node_similarity(path1, path2, *, method="jaccard") -> float
graph.similar_nodes(path, *, method="jaccard", k=10) -> list[tuple[str, float]]
```

### Properties

```python
graph.node_count -> int
graph.edge_count -> int
graph.is_dag() -> bool
```

### Persistence

```python
await graph.from_sql(session, path_prefix="/mount")  # Load from grover_files + grover_file_connections
```

The graph is a pure in-memory projection. `from_sql()` loads file nodes from `grover_files` and edges from `grover_file_connections`, applying the mount `path_prefix` to convert relative DB paths to absolute graph paths. `to_sql()` remains available as a non-authoritative compatibility internal but is no longer called during `save()` — edge persistence is handled by `ConnectionService`.

### Analyzers

```python
from grover.analyzers import (
    Analyzer,           # Protocol
    AnalyzerRegistry,   # Extension → analyzer mapping
    ChunkFile,          # Extracted code chunk
    EdgeData,           # Extracted dependency edge
)
```

Built-in analyzers:

| Language | Analyzer | Requires |
|----------|----------|----------|
| Python | `PythonAnalyzer` | Nothing (uses stdlib `ast`) |
| JavaScript | `JavaScriptAnalyzer` | `treesitter` extra |
| TypeScript | `TypeScriptAnalyzer` | `treesitter` extra |
| Go | `GoAnalyzer` | `treesitter` extra |

### Graph Protocols

The graph API uses the same protocol pattern as the filesystem layer. `GraphProvider` and all opt-in capability protocols live in `providers/graph/protocol.py`. Check support with `isinstance()`:

```python
from grover import GraphProvider
from grover.providers.graph.protocol import (
    SupportsCentrality,      # PageRank, betweenness, closeness, katz, degree
    SupportsConnectivity,    # Weakly/strongly connected components
    SupportsTraversal,       # Ancestors, descendants, topological sort, shortest paths
    SupportsSubgraph,        # Subgraph extraction, neighborhood, meeting subgraph
    SupportsFiltering,       # Attribute-based node/edge filtering
    SupportsNodeSimilarity,  # Jaccard structural similarity
    SupportsPersistence,     # SQL persistence (to_sql / from_sql)
)

if isinstance(g.get_graph(), SupportsCentrality):
    scores = g.get_graph().pagerank()
```

`RustworkxGraph` implements all protocols. Custom graph backends only need to implement `GraphProvider` plus whichever capabilities they support.

### SubgraphResult

```python
from grover.providers.graph.types import SubgraphResult
```

Frozen dataclass returned by subgraph extraction methods. Deeply immutable — `tuple` for sequences, `MappingProxyType` for scores.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `tuple[str, ...]` | Node paths in the subgraph |
| `edges` | `tuple[tuple[str, str, dict], ...]` | Edges with metadata |
| `scores` | `MappingProxyType[str, float]` | Optional node scores (e.g., PageRank) |

---

## Search

Grover's search layer is built around two provider protocols on the filesystem — **EmbeddingProvider** (text → vectors) and **SearchProvider** (store/search vectors, optional lexical search). There is no `SearchEngine` intermediary; the filesystem's `SearchMethodsMixin` orchestrates providers directly.

```python
from grover import (
    EmbeddingProvider, SearchProvider,
    VectorEntry, IndexConfig, IndexInfo,
    FilterExpression, FilterValue, eq, gt, and_, or_,
)
from grover.providers.search.types import SearchResult, VectorHit  # internal types
```

### SearchProvider Protocol

```python
from grover import SearchProvider

@runtime_checkable
class SearchProvider(Protocol):
    # Vector operations
    async def upsert(self, entries: list[VectorEntry], *, namespace: str | None = None) -> UpsertResult: ...
    async def vector_search(self, vector: list[float], *, k: int = 10, namespace: str | None = None, filter: Any = None, include_metadata: bool = True, score_threshold: float | None = None) -> VectorSearchResult: ...
    async def delete(self, ids: list[str], *, namespace: str | None = None) -> DeleteResult: ...
    async def fetch(self, ids: list[str], *, namespace: str | None = None) -> list[VectorEntry | None]: ...

    # Lexical search (stores that don't support it return empty result)
    async def lexical_search(self, query: str, *, k: int = 10) -> LexicalSearchResult: ...

    # Lifecycle
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
```

`SearchProvider` unifies vector and lexical search. All built-in stores (`LocalVectorStore`, `PineconeVectorStore`, `DatabricksVectorStore`) implement it. Stores that don't support lexical search return an empty `LexicalSearchResult`. The protocol is set on the filesystem as `search_provider` and passed via `add_mount(..., search_provider=...)`.

### EmbeddingProvider Protocol

```python
from grover import EmbeddingProvider

@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
```

### Embedding Providers

**OpenAIEmbedding** — OpenAI API. Requires the `openai` extra.

```python
from grover.providers.embedding import OpenAIEmbedding
provider = OpenAIEmbedding(model="text-embedding-3-small", dimensions=384)
```

**LangChainEmbedding** — wraps any LangChain `Embeddings` instance. Requires the `langchain` extra.

```python
from grover.providers.embedding import LangChainEmbedding
provider = LangChainEmbedding(embeddings=langchain_embeddings, dimensions=384)
```

### Capability Protocols

Vector stores can implement additional capabilities, checked at runtime via `isinstance()`:

| Protocol | Methods | Implemented by |
|----------|---------|----------------|
| `SupportsNamespaces` | `list_namespaces()`, `delete_namespace()` | Pinecone |
| `SupportsMetadataFilter` | `compile_filter()` | Pinecone, Databricks |
| `SupportsIndexLifecycle` | `create_index()`, `delete_index()`, `list_indexes()` | Pinecone, Databricks |
| `SupportsHybridSearch` | `hybrid_search()` | Pinecone, Databricks |
| `SupportsReranking` | `reranked_search()` | Pinecone |
| `SupportsTextSearch` | `text_search()` | (custom stores) |
| `SupportsTextIngest` | `text_upsert()` | (custom stores) |

### Search Provider Backends

All built-in stores implement `SearchProvider`. Pass them to `add_mount(..., search_provider=...)`.

**LocalVectorStore** — in-process usearch HNSW index for local development.

```python
from grover.providers.search import LocalVectorStore
store = LocalVectorStore(dimension=384, metric="cosine")
store.dimension  # 384
```

**PineconeVectorStore** — Pinecone cloud vector database. Requires the `pinecone` extra.

```python
from grover.providers.search import PineconeVectorStore
store = PineconeVectorStore(index_name="my-index", api_key="...", namespace="")
await store.connect()
```

**DatabricksVectorStore** — Databricks Vector Search (Direct Vector Access). Requires the `databricks` extra.

```python
from grover.providers.search import DatabricksVectorStore
store = DatabricksVectorStore(
    index_name="catalog.schema.my_index",
    endpoint_name="my_endpoint",
)
await store.connect()
```

### Filter Expressions

Provider-agnostic filter AST with builder helpers:

```python
from grover import eq, ne, gt, gte, lt, lte, in_, not_in, exists, and_, or_

# Simple comparison
f = eq("language", "python")

# Combined
f = and_(eq("language", "python"), gt("year", 2020))

# Nested
f = or_(eq("type", "code"), and_(eq("type", "doc"), gt("pages", 5)))

# Use in search
results = await store.search([0.1, ...], k=10, filter=f)
```

Filters are compiled to provider-native formats automatically (MongoDB-style dicts for Pinecone, SQL-like strings for Databricks, simple dicts for local).

### Data Types

```python
@dataclass(frozen=True)
class VectorEntry:
    id: str
    vector: list[float]
    metadata: dict[str, Any]

@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any]
    vector: list[float] | None

@dataclass(frozen=True)
class IndexConfig:
    name: str
    dimension: int
    metric: str = "cosine"
    cloud_config: dict[str, Any] = {}

@dataclass(frozen=True)
class IndexInfo:
    name: str
    dimension: int
    metric: str
    vector_count: int = 0
    metadata: dict[str, Any] = {}
```

---

## Models

```python
from grover.models import File, FileVersion, FileChunk, FileConnection
from grover.models.vector import Vector, VectorType
```

SQLModel table classes for direct database access if needed.

| Model | Table | Purpose |
|-------|-------|---------|
| `File` | `grover_files` | File metadata, content, version tracking, soft-delete |
| `FileVersion` | `grover_file_versions` | Version snapshots and diffs |
| `FileChunk` | `grover_file_chunks` | Code chunks (functions, classes) |
| `FileConnection` | `grover_file_connections` | File-to-file connections/edges |
| `Vector` | — | Runtime `list[float]` subclass with dimension and model name tracking (see below) |
| `VectorType` | — | SQLAlchemy `TypeDecorator` — stores Vector as JSON text, validates dimension and model |

#### Vector subscript forms

```python
Vector[1536]                          # dimension only
Vector["text-embedding-3-large"]      # model name only
Vector[1536, "text-embedding-3-large"] # both
Vector.for_provider(my_provider)       # derive from EmbeddingProvider
```

Instances expose `.dimension` and `.model_name` read-only properties (both `int | None` and `str | None`).

`VectorType` accepts optional `dimension` and `model_name` parameters. On bind, it validates that a `Vector` instance's model name matches the column's declared model name (plain `list[float]` values and `Vector` instances without a model name skip the check). `VectorType.from_provider(provider)` creates a `VectorType` with both dimension and model name from an `EmbeddingProvider`.

### Background Indexing

```python
from grover.worker import BackgroundWorker, IndexingMode
```

The `BackgroundWorker` handles debounced background task scheduling for indexing. It is created internally by `GroverAsync` and is not typically used directly. The `IndexingMode` enum controls behavior:

| Mode | Behavior |
|------|----------|
| `IndexingMode.BACKGROUND` | File mutations schedule analysis tasks with per-path debouncing (default) |
| `IndexingMode.MANUAL` | All scheduling is suppressed; call `index()` explicitly |

Facade methods call processing functions directly via the worker instead of emitting events. `flush()` drains all pending work. `close()` and `save()` auto-drain before persisting state.

---

## deepagents Integration

```python
from grover.integrations.deepagents import GroverBackend, GroverMiddleware
```

Requires the `deepagents` extra: `pip install grover[deepagents]`

### GroverBackend

Implements the deepagents `BackendProtocol`. Accepts either `Grover` (sync) or `GroverAsync` (native async).

```python
# From an existing Grover instance (sync)
backend = GroverBackend(grover)

# From a GroverAsync instance (native async)
backend = GroverBackend(grover_async)

# Convenience factories
backend = GroverBackend.from_local("/path/to/workspace")
backend = GroverBackend.from_database(engine)

# Async factories (return GroverAsync-backed backend)
backend = await GroverBackend.from_local_async("/path/to/workspace")
backend = await GroverBackend.from_database_async(engine)
```

| Method | Description |
|--------|-------------|
| `ls_info(path)` | List directory entries as `FileInfo` dicts |
| `read(path, offset, limit)` | Read file content in cat -n format |
| `write(path, content)` | Create-only write (errors if file exists) |
| `edit(path, old, new, replace_all)` | Find-and-replace within a file |
| `grep_raw(pattern, path, glob)` | Literal string search across files |
| `glob_info(pattern, path)` | Find files matching a glob pattern |
| `upload_files(files)` | Batch file upload (list of `(path, bytes)` tuples) |
| `download_files(paths)` | Batch file download (returns bytes) |

All methods have async variants (`als_info`, `aread`, etc.):
- **With `GroverAsync`:** async methods call native async API directly; sync methods wrap via `asyncio.run()`.
- **With `Grover`:** sync methods work directly; async methods raise `TypeError`.

**Key semantics:**
- `write()` uses create-only semantics (`overwrite=False`) — returns error if file exists
- `files_update` is always `None` (Grover is external storage)
- Path validation rejects `..`, `~`, and paths without leading `/`

### GroverMiddleware

A deepagents `AgentMiddleware` that adds Grover-specific tools beyond standard file ops. Accepts either `Grover` or `GroverAsync`.

```python
middleware = GroverMiddleware(grover)
middleware = GroverMiddleware(grover_async)  # tools get native async coroutines
middleware = GroverMiddleware(grover, enable_search=False, enable_graph=False)
```

When `GroverAsync` is passed, non-graph tools include both sync and async (coroutine) implementations for native async execution. Graph tools remain sync-only (in-memory rustworkx).

| Tool | Description |
|------|-------------|
| `list_versions` | Show version history with timestamps, sizes, hashes |
| `get_version_content` | Read a specific past version of a file |
| `restore_version` | Restore a file to a previous version (creates new version) |
| `delete_file` | Soft-delete a file to trash |
| `list_trash` | List all soft-deleted files |
| `restore_from_trash` | Recover a file from trash |
| `search_semantic` | Semantic similarity search (requires embedding provider) |
| `dependencies` | Direct dependencies from the knowledge graph |
| `dependents` | Reverse dependencies from the knowledge graph |
| `impacts` | Transitive impact analysis |

Toggle tool groups with `enable_search=False` (removes `search_semantic`) and `enable_graph=False` (removes `dependencies`, `dependents`, `impacts`).

---

## LangChain Integration

```python
from grover.integrations.langchain import GroverRetriever, GroverLoader
```

Requires the `langchain` extra: `pip install grover[langchain]`

### GroverRetriever

LangChain `BaseRetriever` backed by Grover's semantic search. Works in any LangChain chain or RAG pipeline. Accepts either `Grover` or `GroverAsync`.

```python
# Sync
retriever = GroverRetriever(grover=g, k=10)
docs = retriever.invoke("search query")

# Async (native)
retriever = GroverRetriever(grover=ga, k=10)
docs = await retriever.ainvoke("search query")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grover` | `Grover \| GroverAsync` | required | Grover instance with search index |
| `k` | `int` | `10` | Maximum number of results |

Returns `list[Document]` with:
- `page_content` — concatenated chunk snippets (or file path if no snippets)
- `metadata["path"]` — file path
- `metadata["chunks"]` — number of chunk matches (if any)
- `id` — file path

**Async behavior:**
- **With `GroverAsync`:** `_aget_relevant_documents` calls native async API; `_get_relevant_documents` wraps via `asyncio.run()`.
- **With `Grover`:** `_get_relevant_documents` works directly; `_aget_relevant_documents` raises `TypeError`.

Returns empty list when search index is not available.

### GroverLoader

LangChain `BaseLoader` that streams Grover files as Documents. Generator-based (`lazy_load()`) for memory efficiency. Accepts either `Grover` or `GroverAsync`.

```python
# Sync: load all text files recursively
loader = GroverLoader(grover=g, path="/project")
docs = loader.load()

# Async: stream documents with native async
loader = GroverLoader(grover=ga, path="/project")
async for doc in loader.alazy_load():
    process(doc)

# Load only Python files
loader = GroverLoader(grover=g, path="/project", glob_pattern="*.py")

# Non-recursive (immediate children only)
loader = GroverLoader(grover=g, path="/project", recursive=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grover` | `Grover \| GroverAsync` | required | Grover instance |
| `path` | `str` | `"/"` | Root path to load from |
| `glob_pattern` | `str \| None` | `None` | Filter files by glob pattern (e.g., `"*.py"`) |
| `recursive` | `bool` | `True` | Walk subdirectories recursively |

Returns `list[Document]` (via `load()`) or `Iterator[Document]` (via `lazy_load()`) with:
- `page_content` — file content
- `metadata["path"]` — file path
- `metadata["source"]` — file path (LangChain convention)
- `metadata["size_bytes"]` — file size
- `id` — file path

Binary files are automatically skipped.

**Async behavior:**
- **With `GroverAsync`:** `alazy_load()` is a native async generator; `lazy_load()` collects from `alazy_load()` via `asyncio.run()`.
- **With `Grover`:** `lazy_load()` works directly; `alazy_load()` raises `TypeError`.

---

## LangGraph Integration

```python
from grover.integrations.langchain import GroverStore
```

Requires the `langgraph` extra: `pip install grover[langgraph]`

### GroverStore

LangGraph `BaseStore` implementation for persistent agent memory. Namespace tuples map to directory paths, values are stored as JSON files. Accepts either `Grover` or `GroverAsync`.

```python
store = GroverStore(grover=g, prefix="/data/store")

# Put and get
store.put(("users", "alice"), "prefs", {"theme": "dark"})
item = store.get(("users", "alice"), "prefs")
# item.value == {"theme": "dark"}

# Delete
store.delete(("users", "alice"), "prefs")

# List namespaces
namespaces = store.list_namespaces()

# Search (uses Grover's semantic search when available)
results = store.search(("docs",), query="API reference")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grover` | `Grover \| GroverAsync` | required | Grover instance |
| `prefix` | `str` | `"/store"` | Path prefix for all stored items |

**Namespace-to-path mapping:** Namespace `("users", "alice", "notes")` with key `"idea-1"` maps to `{prefix}/users/alice/notes/idea-1.json`.

**Supported operations:**
- `GetOp` — read a key from a namespace
- `PutOp` — write a value (or delete with `value=None`)
- `SearchOp` — semantic search within a namespace (falls back to listing if no search index)
- `ListNamespacesOp` — list namespaces with match conditions and depth limiting

**Async behavior:**
- **With `GroverAsync`:** `abatch()` calls native async handlers; `batch()` wraps via `asyncio.run()`.
- **With `Grover`:** `batch()` works directly; `abatch()` raises `TypeError`.

---

## Migrations

```python
from grover.migrations import backfill_alpha_refactor
```

### backfill_alpha_refactor

Idempotent migration script for the alpha refactor schema changes. Safe to run multiple times.

```python
from sqlalchemy.ext.asyncio import create_async_engine
from grover.migrations import backfill_alpha_refactor

engine = create_async_engine("sqlite+aiosqlite:///grover.db")
report = await backfill_alpha_refactor(engine)
# {"file_versions_file_path": "added", "file_chunks_path": "renamed", ...}
```

**What it does:**

| Step | Description |
|------|-------------|
| `file_versions_file_path` | Adds `file_path` column to `grover_file_versions`, backfills from joined `grover_files.path` |
| `file_chunks_path` | Renames `chunk_path` → `path` on `grover_file_chunks` |
| `file_chunks_vector` | Adds `vector` column (nullable JSON text) to `grover_file_chunks` |
| `files_vector` | Adds `vector` column (nullable JSON text) to `grover_files` |
| `file_connections_path` | Adds `path` column to `grover_file_connections`, computes from `source_path[type]target_path` |
| `embeddings_dropped` | Drops `grover_embeddings` table |

**Report values:** `"added"`, `"renamed"`, `"exists"`, `"table_missing"`, `"dropped"`, `"not_present"`.

### Schema validation

`add_mount()` automatically checks for schema compatibility when mounting an engine-based backend. If the database has an old schema, it raises `SchemaIncompatibleError` with instructions to run the migration.

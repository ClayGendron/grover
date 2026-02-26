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
Grover(*, data_dir=None, embedding_provider=None, vector_store=None)
GroverAsync(*, data_dir=None, embedding_provider=None, vector_store=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | `str | None` | Directory for internal state (`.grover/`). Auto-detected from the first mounted backend if not set. `GroverAsync` also accepts `Path`. |
| `embedding_provider` | `EmbeddingProvider | None` | Custom embedding provider for search. Falls back to `SentenceTransformerEmbedding` if the `search` extra is installed. Search is disabled if neither is available. |
| `vector_store` | `VectorStore | None` | Custom vector store backend (e.g., `PineconeVectorStore`, `DatabricksVectorStore`). Defaults to `LocalVectorStore` if not set. |

### Mount / Unmount

```python
g.add_mount(path, backend=None, *, engine=None, session_factory=None,
            dialect="sqlite", file_model=None, file_version_model=None,
            db_schema=None, mount_type=None, permission=Permission.READ_WRITE,
            label="", hidden=False)
g.unmount(path)
```

Mount a storage backend at a virtual path. You can pass either:

- A `backend` object (e.g., `LocalFileSystem`, `DatabaseFileSystem`)
- An `engine` (SQLAlchemy `AsyncEngine`) — Grover will create a `DatabaseFileSystem` automatically
- A `session_factory` — same as engine, but you control session creation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Virtual mount path (e.g., `"/project"`) |
| `backend` | `StorageBackend | None` | `None` | Pre-created backend instance |
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

For user-scoped mounts, pass a `UserScopedFileSystem` as the backend (see [architecture.md](architecture.md#user-scoped-file-systems)).

### Filesystem Operations

All filesystem methods accept an optional `user_id` keyword argument. On user-scoped mounts (using `UserScopedFileSystem`), `user_id` is **required** — paths are automatically namespaced per user (e.g., `/notes.md` → `/{user_id}/notes.md` in the backend). On regular mounts, `user_id` is accepted but ignored.

```python
g.read(path, *, user_id=None) -> ReadResult
g.write(path, content, *, user_id=None) -> WriteResult
g.edit(path, old, new, *, user_id=None) -> EditResult
g.delete(path, permanent=False, *, user_id=None) -> DeleteResult
g.list_dir(path="/", *, user_id=None) -> list[dict]
g.exists(path, *, user_id=None) -> bool
g.move(src, dest, *, user_id=None, follow=False) -> MoveResult
g.copy(src, dest, *, user_id=None) -> WriteResult
```

| Method | Description |
|--------|-------------|
| `read(path)` | Read file content. Returns `ReadResult` with `success`, `content`, `total_lines`, etc. |
| `write(path, content)` | Write content to a file. Creates the file if it doesn't exist, creates a new version if it does. Returns `WriteResult` with `success`, `created`, `version`. |
| `edit(path, old, new)` | Find-and-replace within a file. Returns `EditResult` with `success` and `version`. |
| `delete(path, permanent=False)` | Delete a file. Default is soft-delete (moves to trash). Pass `permanent=True` for permanent deletion. Returns `DeleteResult`. |
| `list_dir(path)` | List directory entries. Returns a list of dicts with `path`, `name`, `is_directory`. On user-scoped mounts, the user's root listing includes a virtual `@shared/` entry. |
| `exists(path)` | Check if a path exists. Returns `bool`. |
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
```

| Method | Description |
|--------|-------------|
| `list_versions(path)` | List all versions of a file. Returns `VersionResult` (a `FileSearchResult` subclass) with `candidates` — each candidate's path is `"{file_path}@{version}"` and evidence is `VersionEvidence` with `version`, `content_hash`, `size_bytes`, `created_at`, `created_by`. Versions with `created_by="external"` are synthetic snapshots auto-inserted when an external edit was detected. |
| `get_version_content(path, version)` | Retrieve the content of a specific version. Returns `GetVersionContentResult`. |
| `restore_version(path, version)` | Restore a file to a previous version (creates a new version with the old content). Returns `RestoreResult`. |

### Trash

```python
g.list_trash(*, user_id=None) -> list
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
g.reconcile(mount_path=None) -> dict[str, int]
```

Synchronize the database with the actual filesystem state. Only available for backends that implement `SupportsReconcile` (currently `LocalFileSystem`).

Returns a dict with counts: `{"created": N, "updated": N, "deleted": N}`.

### Graph Queries

```python
g.dependencies(path) -> list[Ref]
g.dependents(path) -> list[Ref]
g.impacts(path, max_depth=3) -> list[Ref]
g.path_between(source, target) -> list[Ref] | None
g.contains(path) -> list[Ref]
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
g.add_connection(source_path, target_path, connection_type, *, weight=1.0, metadata=None) -> ConnectionResult
g.delete_connection(source_path, target_path, *, connection_type=None) -> ConnectionResult
g.list_connections(path, *, direction="both", connection_type=None) -> list[FileConnection]
```

Connections are persisted through the filesystem layer (not directly on the graph). The in-memory graph is updated via `CONNECTION_ADDED` / `CONNECTION_DELETED` events after the DB transaction commits. This makes the filesystem the single source of truth for all persistent edge data.

| Method | Description |
|--------|-------------|
| `add_connection(source, target, type)` | Create or update a connection. Returns `ConnectionResult`. The connection path identity is `source[type]target`. |
| `delete_connection(source, target)` | Delete a connection. If `connection_type` is `None`, deletes all connections between source and target. |
| `list_connections(path)` | List connections for a path. `direction` can be `"out"`, `"in"`, or `"both"` (default). |

`ConnectionResult` fields: `success`, `message`, `path`, `source_path`, `target_path`, `connection_type`.

### Graph Algorithms

These convenience methods delegate to the graph backend. They raise `CapabilityNotSupportedError` if the backend doesn't support the required capability protocol.

Graph operations resolve to the per-mount graph for the given path. `pagerank()` and `find_nodes()` accept an optional `path` parameter to select which mount's graph to use (defaults to the first visible mount).

```python
g.pagerank(*, personalization=None, path=None) -> dict[str, float]
g.ancestors(path) -> set[str]
g.descendants(path) -> set[str]
g.meeting_subgraph(paths, *, max_size=50) -> SubgraphResult
g.neighborhood(path, *, max_depth=2, direction="both", edge_types=None) -> SubgraphResult
g.find_nodes(*, path=None, **attrs) -> list[str]
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
g.search(query, k=10, *, path="/", glob=None, grep=None, user_id=None) -> FileSearchResult
```

Composable search pipeline: optional glob/grep filters followed by vector search. Returns a `FileSearchResult` with `candidates: list[FileSearchCandidate]` — each candidate has a `path` and `evidence: list[Evidence]` explaining why it matched.

Search is routed through per-mount search engines. When `path="/"`, results are aggregated across all mounts. When `path` targets a specific mount or subdirectory, search is scoped to that mount and filtered to the given path prefix. Use `glob` and `grep` parameters to pre-filter before vector search.

Returns `FileSearchResult(success=False, ...)` if no embedding provider is available.

### Index and Persistence

```python
g.index(mount_path=None) -> dict[str, int]
g.save()
g.close()
```

| Method | Description |
|--------|-------------|
| `index(mount_path)` | Walk the filesystem, analyze all files, build the knowledge graph and search index. Pass a `mount_path` to index a single mount, or `None` for all visible mounts. Returns stats: `{"files_scanned": N, "chunks_created": N, "edges_added": N}`. |
| `save()` | Persist graph edges to the database and search index to disk. |
| `close()` | Save state and shut down all subsystems. Idempotent. |

### Properties and Methods

```python
g.get_graph(path=None) -> GraphStore   # Per-mount knowledge graph (replaces old .graph property)
g.fs -> VFS                            # The virtual filesystem (for advanced use)
```

`get_graph(path)` returns the graph for the mount owning `path`. If `path` is `None`, returns the first available mount's graph. Each mount has its own isolated `RustworkxGraph` instance, injected at mount time.

---

## Key Types

### Ref

```python
from grover import Ref, file_ref
```

Immutable (frozen) reference to a file or chunk.

```python
@dataclass(frozen=True)
class Ref:
    path: str                                    # Normalized virtual path
    version: int | str | None = None             # Version identifier
    line_start: int | None = None                # Chunk start line
    line_end: int | None = None                  # Chunk end line
    metadata: Mapping[str, Any] = field(...)     # Read-only, excluded from hash/equality
```

`file_ref(path, version=None)` is a convenience constructor that normalizes the path.

### SearchResult (internal)

```python
from grover.search.types import SearchResult
```

Internal type used by `SearchEngine` and vector store backends. The public `Grover.search()` API returns `FileSearchResult` (see [Result Types](#result-types) below), which wraps these into `FileSearchCandidate` objects with `VectorEvidence`.

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

All result types live in `grover.types` (canonical location).

```python
from grover import (
    # Operation results
    FileOperationResult, ReadResult, WriteResult, EditResult,
    DeleteResult, MkdirResult, MoveResult, RestoreResult,
    GetVersionContentResult, ShareResult, ConnectionResult,
    FileInfoResult,
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
| `FileInfoResult` | `name`, `is_directory`, `mime_type`, `size_bytes`, `created_at`, `updated_at`, `permission`, `mount_type` |

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
from grover.fs import (
    LocalFileSystem,
    DatabaseFileSystem,
    MountRegistry,
    Permission,
    StorageBackend,
    SupportsVersions,
    SupportsTrash,
    SupportsReconcile,
)
```

### LocalFileSystem

```python
LocalFileSystem(workspace_dir, *, data_dir=None)
```

Stores files on disk at `workspace_dir`. Metadata and version history live in a SQLite database at `data_dir` (defaults to `~/.grover/{workspace_slug}/`).

Implements: `StorageBackend`, `SupportsVersions`, `SupportsTrash`, `SupportsReconcile`, `SupportsSearch`, `SupportsFileChunks`.

### DatabaseFileSystem

```python
DatabaseFileSystem(*, dialect="sqlite", file_model=None,
                   file_version_model=None, schema=None)
```

Pure-database storage. All content lives in the `File.content` column. Stateless — requires a session to be injected by VFS.

Implements: `StorageBackend`, `SupportsVersions`, `SupportsTrash`, `SupportsSearch`, `SupportsFileChunks`.

### Permission

```python
from grover.fs import Permission

Permission.READ_WRITE  # Full access (default)
Permission.READ_ONLY   # Reads and listings only
```

**Read-only enforcement**: When a mount (or path within a mount via `read_only_paths`) is set to `READ_ONLY`, all mutations return a failed Result with `success=False` and a descriptive message — no exceptions are raised. This applies to: `write`, `edit`, `delete`, `mkdir`, `move`, `copy`, `restore_version`, `restore_from_trash`, `share`, `unshare`, `add_connection`, `delete_connection`. Multi-mount operations (`empty_trash`, `reconcile`, `index`) silently skip read-only mounts. Read and query operations (`read`, `glob`, `grep`, `list_dir`, `tree`, `list_connections`, `list_shares`, graph queries, search) are unaffected.

### Protocols

| Protocol | Methods |
|----------|---------|
| `StorageBackend` | `open`, `close`, `read`, `write`, `edit`, `delete`, `mkdir`, `move`, `copy`, `list_dir`, `exists`, `get_info`, `glob`, `grep`, `tree` |
| `SupportsVersions` | `list_versions`, `get_version_content`, `restore_version` |
| `SupportsTrash` | `list_trash`, `restore_from_trash`, `empty_trash` |
| `SupportsReconcile` | `reconcile` |
| `SupportsSearch` | `search` — semantic search backed by a per-mount search engine |
| `SupportsFileChunks` | `replace_file_chunks`, `delete_file_chunks`, `list_file_chunks` |

All protocols are `runtime_checkable`. Implement `StorageBackend` for a minimal custom backend; add optional protocols as needed.

### Exceptions

```python
from grover.fs import (
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
from grover.graph import RustworkxGraph
```

In-memory directed graph backed by `rustworkx.PyDiGraph`. Nodes are file paths (strings), edges have a free-form type string. Implements the `GraphStore` protocol.

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
from grover.graph.analyzers import (
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

The graph API uses the same protocol pattern as the filesystem layer. `GraphStore` is the core protocol; capability protocols are opt-in. Check support with `isinstance()`:

```python
from grover.graph.protocols import (
    GraphStore,              # Core: node/edge CRUD + basic queries
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

`RustworkxGraph` implements all protocols. Custom graph backends only need to implement `GraphStore` plus whichever capabilities they support.

### SubgraphResult

```python
from grover.graph.types import SubgraphResult
```

Frozen dataclass returned by subgraph extraction methods. Deeply immutable — `tuple` for sequences, `MappingProxyType` for scores.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `tuple[str, ...]` | Node paths in the subgraph |
| `edges` | `tuple[tuple[str, str, dict], ...]` | Edges with metadata |
| `scores` | `MappingProxyType[str, float]` | Optional node scores (e.g., PageRank) |

---

## Search

Grover's search layer is built around three protocol layers — **EmbeddingProvider** (text → vectors), **VectorStore** (store/search vectors), and **FullTextStore** (BM25 keyword search) — wired together by **SearchEngine**.

```python
from grover import (
    SearchEngine,
    EmbeddingProvider, VectorStore,
    VectorEntry, IndexConfig, IndexInfo,
    FilterExpression, FilterValue, eq, gt, and_, or_,
)
from grover.search.types import SearchResult, VectorHit  # internal types used by SearchEngine
from grover.search.fulltext import FullTextStore, FullTextResult
```

### SearchEngine

```python
SearchEngine(*, vector=None, embedding=None, lexical=None, hybrid=None, model_name=None)
```

Orchestrates embedding, vector storage, and full-text search. This is what `GroverAsync` uses internally. All components are optional — configure only what you need.

**Construction-time validation:** If both `vector` and `embedding` are provided and the vector store exposes a `dimension` property, the engine verifies that the store's dimension matches `embedding.dimensions`. If `model_name` is provided with an `embedding`, the engine verifies they agree. Both checks raise `ValueError` on mismatch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector` | `VectorStore | None` | Vector store for semantic search |
| `embedding` | `EmbeddingProvider | None` | Embedding provider for vectorization |
| `lexical` | `FullTextStore | None` | Full-text store for BM25 keyword search |
| `hybrid` | `object | None` | Hybrid search provider (reserved) |
| `model_name` | `str | None` | Declared model name — validated against embedding provider if both are set |

| Method | Description |
|--------|-------------|
| `add(path, content, *, parent_path=None, session=None)` | Embed and index a single item (vector + FTS) |
| `add_batch(chunks, *, session=None)` | Batch embed and index multiple items |
| `remove(path, *, session=None)` | Remove a single entry by path |
| `remove_file(path, *, session=None)` | Remove a file and all its chunks |
| `search(query, k=10) -> list[SearchResult]` | Embed query and search (vector). Returns internal `SearchResult` type. |
| `lexical_search(query, *, k=10, session=None) -> list[FullTextResult]` | BM25 keyword search |
| `has(path) -> bool` | Check if a path is indexed |
| `content_hash(path) -> str | None` | Get the content hash of an indexed entry |
| `save(dir)` | Persist to disk (delegates to store if supported) |
| `load(dir)` | Load from disk (delegates to store if supported) |
| `connect()` | Connect the underlying store |
| `close()` | Close the underlying store |
| `supported_protocols() -> set[type]` | Return mount-level dispatch protocols based on configured components |

### FullTextStore Protocol

```python
@runtime_checkable
class FullTextStore(Protocol):
    async def index(self, path: str, content: str, *, session=None) -> None: ...
    async def remove(self, path: str, *, session=None) -> None: ...
    async def remove_file(self, path: str, *, session=None) -> None: ...
    async def search(self, query: str, *, k: int = 10, session=None) -> list[FullTextResult]: ...
```

**Implementations:**
- `SQLiteFullTextStore` — FTS5 virtual table with `bm25()` ranking and `snippet()`
- `PostgresFullTextStore` — `to_tsvector`/`tsquery` with `ts_rank_cd` and GIN index
- `MSSQLFullTextStore` — `FREETEXTTABLE` with full-text catalog

```python
from grover.search.fulltext.sqlite import SQLiteFullTextStore
from grover.search.fulltext.postgres import PostgresFullTextStore
from grover.search.fulltext.mssql import MSSQLFullTextStore
```

### EmbeddingProvider Protocol

```python
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

**SentenceTransformerEmbedding** — local models (default). Requires the `search` extra.

```python
from grover.search.providers import SentenceTransformerEmbedding
provider = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
```

**OpenAIEmbedding** — OpenAI API. Requires the `openai` extra.

```python
from grover.search.providers import OpenAIEmbedding
provider = OpenAIEmbedding(model="text-embedding-3-small", dimensions=384)
```

**LangChainEmbedding** — wraps any LangChain `Embeddings` instance. Requires the `langchain` extra.

```python
from grover.search.providers import LangChainEmbedding
provider = LangChainEmbedding(embeddings=langchain_embeddings, dimensions=384)
```

### VectorStore Protocol

```python
@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, entries: list[VectorEntry], *, namespace: str | None = None) -> UpsertResult: ...
    async def search(self, vector: list[float], *, k: int = 10, namespace: str | None = None, filter: FilterExpression | None = None, ...) -> list[VectorHit]: ...
    async def delete(self, ids: list[str], *, namespace: str | None = None) -> DeleteResult: ...
    async def fetch(self, ids: list[str], *, namespace: str | None = None) -> list[VectorEntry | None]: ...
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    @property
    def index_name(self) -> str: ...
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

### Vector Store Backends

**LocalVectorStore** — in-process usearch HNSW index for local development.

```python
from grover.search.stores import LocalVectorStore
store = LocalVectorStore(dimension=384, metric="cosine")
store.dimension  # 384
```

**PineconeVectorStore** — Pinecone cloud vector database. Requires the `pinecone` extra.

```python
from grover.search.stores import PineconeVectorStore
store = PineconeVectorStore(index_name="my-index", api_key="...", namespace="")
await store.connect()
```

**DatabricksVectorStore** — Databricks Vector Search (Direct Vector Access). Requires the `databricks` extra.

```python
from grover.search.stores import DatabricksVectorStore
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

### Events

```python
from grover.events import EventBus, EventType, FileEvent
```

| Event | Emitted When |
|-------|-------------|
| `FILE_WRITTEN` | A file is created or updated |
| `FILE_DELETED` | A file is deleted |
| `FILE_MOVED` | A file is moved or renamed |
| `FILE_RESTORED` | A file is restored from trash |
| `CONNECTION_ADDED` | A connection is created or updated |
| `CONNECTION_DELETED` | A connection is deleted |

`FileEvent` carries optional connection fields for `CONNECTION_ADDED` / `CONNECTION_DELETED` events:

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | `str | None` | Source file of the connection |
| `target_path` | `str | None` | Target file of the connection |
| `connection_type` | `str | None` | Edge type string |
| `weight` | `float` | Edge weight (default 1.0) |

---

## deepagents Integration

```python
from grover.integrations.deepagents import GroverBackend, GroverMiddleware
```

Requires the `deepagents` extra: `pip install grover[deepagents]`

### GroverBackend

Implements the deepagents `BackendProtocol`, mapping all file operations to Grover's sync API.

```python
# From an existing Grover instance
backend = GroverBackend(grover)

# Convenience factories
backend = GroverBackend.from_local("/path/to/workspace")
backend = GroverBackend.from_database(engine)
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

All methods have async variants (`als_info`, `aread`, etc.) via `asyncio.to_thread()`.

**Key semantics:**
- `write()` uses create-only semantics (`overwrite=False`) — returns error if file exists
- `files_update` is always `None` (Grover is external storage)
- Path validation rejects `..`, `~`, and paths without leading `/`

### GroverMiddleware

A deepagents `AgentMiddleware` that adds Grover-specific tools beyond standard file ops.

```python
middleware = GroverMiddleware(grover)
middleware = GroverMiddleware(grover, enable_search=False, enable_graph=False)
```

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

LangChain `BaseRetriever` backed by Grover's semantic search. Works in any LangChain chain or RAG pipeline.

```python
retriever = GroverRetriever(grover=g, k=10)
docs = retriever.invoke("search query")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grover` | `Grover` | required | Grover instance with search index |
| `k` | `int` | `10` | Maximum number of results |

Returns `list[Document]` with:
- `page_content` — concatenated chunk snippets (or file path if no snippets)
- `metadata["path"]` — file path
- `metadata["score"]` — max chunk similarity score (0–1)
- `metadata["chunks"]` — number of chunk matches (if any)
- `id` — file path

Has async variant via `asyncio.to_thread()`. Returns empty list when search index is not available.

### GroverLoader

LangChain `BaseLoader` that streams Grover files as Documents. Generator-based (`lazy_load()`) for memory efficiency.

```python
# Load all text files recursively
loader = GroverLoader(grover=g, path="/project")
docs = loader.load()

# Load only Python files
loader = GroverLoader(grover=g, path="/project", glob_pattern="*.py")

# Non-recursive (immediate children only)
loader = GroverLoader(grover=g, path="/project", recursive=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grover` | `Grover` | required | Grover instance |
| `path` | `str` | `"/"` | Root path to load from |
| `glob_pattern` | `str | None` | `None` | Filter files by glob pattern (e.g., `"*.py"`) |
| `recursive` | `bool` | `True` | Walk subdirectories recursively |

Returns `list[Document]` (via `load()`) or `Iterator[Document]` (via `lazy_load()`) with:
- `page_content` — file content
- `metadata["path"]` — file path
- `metadata["source"]` — file path (LangChain convention)
- `metadata["size_bytes"]` — file size
- `id` — file path

Binary files are automatically skipped.

---

## LangGraph Integration

```python
from grover.integrations.langchain import GroverStore
```

Requires the `langgraph` extra: `pip install grover[langgraph]`

### GroverStore

LangGraph `BaseStore` implementation for persistent agent memory. Namespace tuples map to directory paths, values are stored as JSON files.

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
| `grover` | `Grover` | required | Grover instance |
| `prefix` | `str` | `"/store"` | Path prefix for all stored items |

**Namespace-to-path mapping:** Namespace `("users", "alice", "notes")` with key `"idea-1"` maps to `{prefix}/users/alice/notes/idea-1.json`.

**Supported operations:**
- `GetOp` — read a key from a namespace
- `PutOp` — write a value (or delete with `value=None`)
- `SearchOp` — semantic search within a namespace (falls back to listing if no search index)
- `ListNamespacesOp` — list namespaces with match conditions and depth limiting

Has async variant (`abatch`) via `asyncio.to_thread()`.

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

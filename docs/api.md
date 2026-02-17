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
Grover(*, data_dir=None, embedding_provider=None)
GroverAsync(*, data_dir=None, embedding_provider=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | `str | Path | None` | Directory for internal state (`.grover/`). Auto-detected from the first mounted backend if not set. |
| `embedding_provider` | `EmbeddingProvider | None` | Custom embedding provider for search. Falls back to `SentenceTransformerProvider` if the `search` extra is installed. Search is disabled if neither is available. |

### Mount / Unmount

```python
g.mount(path, backend=None, *, engine=None, session_factory=None,
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

### Search / Query

```python
g.glob(pattern, path="/") -> GlobResult
g.grep(pattern, path="/", *, ...) -> GrepResult
g.tree(path="/", *, max_depth=None) -> TreeResult
```

| Method | Description |
|--------|-------------|
| `glob(pattern, path)` | Find files matching a glob pattern. Supports `*` (single segment), `**` (recursive), `?` (single char), `[seq]` (character class), `[!seq]` (negated). Returns `GlobResult` with `entries` (list of `FileInfo`). |
| `grep(pattern, path, ...)` | Search file contents with regex. Returns `GrepResult` with `matches` (list of `GrepMatch`). |
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
g.list_versions(path) -> ListVersionsResult
g.get_version_content(path, version) -> GetVersionContentResult
g.restore_version(path, version) -> RestoreResult
```

| Method | Description |
|--------|-------------|
| `list_versions(path)` | List all versions of a file. Returns `ListVersionsResult` with a list of `VersionInfo`. Versions with `created_by="external"` are synthetic snapshots auto-inserted when an external edit was detected. |
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
g.list_shares(path, *, user_id) -> ListSharesResult
g.list_shared_with_me(*, user_id) -> ListSharesResult
```

| Method | Description |
|--------|-------------|
| `share(path, grantee_id, permission, *, user_id)` | Share a file or directory. `permission` is `"read"` or `"write"`. Only the file owner can create shares. Returns `ShareResult`. |
| `unshare(path, grantee_id, *, user_id)` | Remove a share. Returns `ShareResult`. |
| `list_shares(path, *, user_id)` | List all shares on a given path. Returns `ListSharesResult`. |
| `list_shared_with_me(*, user_id)` | List all files shared with the current user. Paths are returned with `@shared/{owner}/` prefix. Returns `ListSharesResult`. |

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

### Search

```python
g.search(query, k=10) -> list[SearchResult]
```

Semantic similarity search over indexed content. Returns up to `k` results sorted by relevance.

Raises `RuntimeError` if no embedding provider is available.

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

### Properties

```python
g.graph -> Graph   # The knowledge graph instance
g.fs -> VFS        # The virtual filesystem (for advanced use)
```

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
    path: str                          # Normalized virtual path
    version: int | str | None = None   # Version identifier
    line_start: int | None = None      # Chunk start line
    line_end: int | None = None        # Chunk end line
    metadata: dict[str, Any] = {}      # Excluded from hash/equality
```

`file_ref(path, version=None)` is a convenience constructor that normalizes the path.

### SearchResult

```python
from grover import SearchResult
```

```python
@dataclass(frozen=True)
class SearchResult:
    ref: Ref                    # The matched file or chunk
    score: float                # Cosine similarity (0–1)
    content: str                # The matched text
    parent_path: str | None     # Parent file path (for chunks)
```

### Result Types

All filesystem operations return structured result objects rather than raising exceptions.

```python
from grover.fs import (
    ReadResult, WriteResult, EditResult, DeleteResult,
    ListResult, MoveResult, RestoreResult,
    ListVersionsResult, GetVersionContentResult,
    GlobResult, GrepResult, GrepMatch, TreeResult,
    FileInfo, VersionInfo,
)
```

Every result has a `success: bool` and `message: str` field. Check `success` to determine if the operation succeeded.

| Type | Key Fields |
|------|------------|
| `ReadResult` | `content`, `file_path`, `total_lines`, `truncated`, `offset` |
| `WriteResult` | `file_path`, `created` (bool), `version` (int) |
| `EditResult` | `file_path`, `version` (int) |
| `DeleteResult` | `file_path`, `permanent` (bool), `total_deleted` |
| `ListResult` | `entries` (list of `FileInfo`), `path` |
| `RestoreResult` | `file_path`, `restored_version`, `current_version` |
| `ListVersionsResult` | `versions` (list of `VersionInfo`) |
| `GetVersionContentResult` | `content` |
| `GlobResult` | `entries` (list of `FileInfo`), `pattern`, `path` |
| `GrepResult` | `matches` (list of `GrepMatch`), `pattern`, `path`, `files_searched`, `files_matched`, `truncated` |
| `GrepMatch` | `file_path`, `line_number`, `line_content`, `context_before`, `context_after` |
| `TreeResult` | `entries` (list of `FileInfo`), `path`, `total_files`, `total_dirs` |
| `FileInfo` | `path`, `name`, `is_directory`, `size_bytes`, `mime_type`, `version` |
| `VersionInfo` | `version`, `content_hash`, `size_bytes`, `created_at`, `created_by` |
| `MoveResult` | `old_path`, `new_path` |
| `ShareResult` | `share` (`ShareInfo | None`) |
| `ListSharesResult` | `shares` (list of `ShareInfo`) |
| `ShareInfo` | `path`, `grantee_id`, `permission`, `granted_by`, `created_at`, `expires_at` |

---

## Filesystem Layer

```python
from grover.fs import (
    LocalFileSystem,
    DatabaseFileSystem,
    VFS,
    MountConfig,
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

Implements: `StorageBackend`, `SupportsVersions`, `SupportsTrash`, `SupportsReconcile`.

### DatabaseFileSystem

```python
DatabaseFileSystem(*, dialect="sqlite", file_model=None,
                   file_version_model=None, schema=None)
```

Pure-database storage. All content lives in the `File.content` column. Stateless — requires a session to be injected by VFS.

Implements: `StorageBackend`, `SupportsVersions`, `SupportsTrash`.

### Permission

```python
from grover.fs import Permission

Permission.READ_WRITE  # Full access (default)
Permission.READ_ONLY   # Reads and listings only
```

### Protocols

| Protocol | Methods |
|----------|---------|
| `StorageBackend` | `open`, `close`, `read`, `write`, `edit`, `delete`, `mkdir`, `move`, `copy`, `list_dir`, `exists`, `get_info`, `glob`, `grep`, `tree` |
| `SupportsVersions` | `list_versions`, `get_version_content`, `restore_version` |
| `SupportsTrash` | `list_trash`, `restore_from_trash`, `empty_trash` |
| `SupportsReconcile` | `reconcile` |

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

## Graph

```python
from grover.graph import Graph
```

In-memory directed graph backed by `rustworkx.PyDiGraph`. Nodes are file paths (strings), edges have a free-form type string.

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

### Subgraph Operations

```python
graph.remove_file_subgraph(path)  # Remove node + all children
```

### Properties

```python
graph.node_count -> int
graph.edge_count -> int
graph.is_dag() -> bool
```

### Persistence

```python
await graph.to_sql(session)    # Save to grover_edges table
await graph.from_sql(session)  # Load from grover_edges table
```

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

---

## Search

```python
from grover.search import (
    SearchIndex,
    SearchResult,
    EmbeddingProvider,
    SentenceTransformerProvider,
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
```

### SearchIndex

```python
SearchIndex(provider: EmbeddingProvider)
```

| Method | Description |
|--------|-------------|
| `add(path, content, parent_path=None)` | Embed and index a single item |
| `add_batch(chunks: list[EmbeddableChunk])` | Batch index multiple items |
| `remove(path)` | Remove all vectors for a path |
| `remove_file(path)` | Alias for `remove` |
| `search(query, k=10) -> list[SearchResult]` | Similarity search |
| `has(path) -> bool` | Check if a path is indexed |
| `save(dir)` | Persist index to disk |
| `load(dir)` | Load index from disk |

### EmbeddingProvider Protocol

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
```

### SentenceTransformerProvider

```python
SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
```

Default provider. Uses `sentence-transformers` and produces 384-dimensional vectors. Runs on CPU, no API key needed. Requires the `search` extra.

---

## Models

```python
from grover.models import File, FileVersion, GroverEdge, Embedding
```

SQLModel table classes for direct database access if needed.

| Model | Table | Purpose |
|-------|-------|---------|
| `File` | `grover_files` | File metadata, content, version tracking, soft-delete |
| `FileVersion` | `grover_file_versions` | Version snapshots and diffs |
| `GroverEdge` | `grover_edges` | Graph edge persistence |
| `Embedding` | `grover_embeddings` | Embedding change detection metadata |

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

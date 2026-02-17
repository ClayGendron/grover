# Grover Baseline: Current Feature Set (v0.1 alpha)

Complete inventory of what Grover can and cannot do today.

## Public API

### Entry Points

- `Grover` — sync facade (RLock + daemon thread event loop)
- `GroverAsync` — primary async class, wires all subsystems
- Both expose identical methods

### Lifecycle

| Method | Description |
|--------|-------------|
| `mount(path, backend, ...)` | Mount backends at virtual paths |
| `unmount(path)` | Unmount and clean up |
| `close()` | Cleanup and shutdown |
| `save()` | Persist graph and search state |

Mount options: `engine`, `session_factory`, `dialect`, `file_model`, `file_version_model`, `db_schema`, `mount_type`, `permission`, `label`, `hidden`.

---

## Filesystem Operations

### Core CRUD

| Method | Description |
|--------|-------------|
| `read(path, offset, limit)` | Read file content with pagination |
| `write(path, content, overwrite)` | Write/create files with automatic versioning |
| `edit(path, old, new)` | Find-and-replace with versioning |
| `delete(path, permanent)` | Soft-delete (default) or permanent |
| `exists(path)` | Check existence |
| `list_dir(path)` | List directory entries with metadata |
| `move(src, dest, follow)` | Move with two semantics (see below) |
| `copy(src, dest)` | Copy file to new path |

**Move semantics:**
- `follow=False` (default): clean break, no version history carryover
- `follow=True`: in-place rename, preserves version chain, updates share paths

### Search/Query

| Method | Description |
|--------|-------------|
| `glob(pattern, path)` | Glob with `*`, `**`, `?`, `[seq]`, `[!seq]` |
| `grep(pattern, path, ...)` | Regex content search with extensive options |
| `tree(path, max_depth)` | Recursive directory listing |

Grep options: `glob_filter`, `case_sensitive`, `fixed_string`, `invert`, `word_match`, `context_lines`, `max_results`, `max_results_per_file`, `count_only`, `files_only`.

### Versioning

| Method | Description |
|--------|-------------|
| `list_versions(path)` | All versions with metadata (hash, size, created_by, created_at) |
| `get_version_content(path, version)` | Retrieve specific version content |
| `restore_version(path, version)` | Restore to previous version |

- Automatic snapshot every 20 versions
- External edit detection with synthetic version insertion (`created_by="external"`)
- Forward diff storage between snapshots

### Trash Management

| Method | Description |
|--------|-------------|
| `list_trash()` | List soft-deleted files (owner-scoped on user-scoped mounts) |
| `restore_from_trash(path)` | Restore deleted file by original path |
| `empty_trash()` | Permanently delete trashed files |

### Sharing (User-Scoped Mounts Only)

| Method | Description |
|--------|-------------|
| `share(path, grantee_id, permission)` | Share file/directory |
| `unshare(path, grantee_id)` | Remove share |
| `list_shares(path)` | Show all shares on a path |
| `list_shared_with_me()` | Show files shared with current user |

- Virtual `@shared/{owner}/{path}` namespace
- Permission model: `"read"` or `"write"` with directory-level prefix matching
- Optional expiration via `expires_at`

### Reconciliation

| Method | Description |
|--------|-------------|
| `reconcile(mount_path)` | Sync database with actual filesystem state (LocalFileSystem only) |

Returns `{"created": N, "updated": N, "deleted": N}`.

---

## Knowledge Graph

### Query Methods

| Method | Description |
|--------|-------------|
| `dependencies(path)` | What does this file import/depend on? (outgoing edges) |
| `dependents(path)` | What depends on this file? (incoming edges) |
| `impacts(path, max_depth)` | Transitive: what's affected if this file changes? (BFS) |
| `path_between(source, target)` | Shortest path via Dijkstra |
| `contains(path)` | Chunks/symbols inside this file |

### Edge Types (Conventions)

- `"imports"` — file imports another
- `"contains"` — file contains a chunk (function, class)
- `"references"` — general reference
- `"inherits"` — class inheritance

### Architecture

- In-memory `rustworkx.PyDiGraph` with string-path-keyed nodes
- Nodes = file paths from `grover_files` table
- Edges = `grover_edges` table (single table, free-form type strings)
- Edge metadata: dict, weight, immutable UUID
- Persistence: `to_sql()` upserts in-memory edges, deletes stale DB edges

---

## Semantic Search

### API

| Method | Description |
|--------|-------------|
| `search(query, k)` | Top-k vector similarity search |

Returns `list[SearchResult]`:
- `ref: Ref` — reference to matched chunk/file
- `score: float` — cosine similarity (0-1)
- `content: str` — matched text
- `parent_path: str | None` — parent file if chunk

### Index

- HNSW via `usearch` library
- Persistent (`search.usearch` binary + `search_meta.json`)
- Thread-safe via lock
- Deduplication on re-add

### Embedding Providers

- Default: `SentenceTransformerProvider` (all-MiniLM-L6-v2, 80 MB, CPU)
- Protocol-based: custom providers injectable
- Optional (search disabled if no provider)

---

## Code Analyzers

| Language | Engine | Dependency |
|----------|--------|------------|
| Python | stdlib `ast` | None |
| JavaScript/TypeScript | tree-sitter | `tree-sitter-javascript` / `tree-sitter-typescript` |
| Go | tree-sitter | `tree-sitter-go` |

Extracts:
- `chunks: list[ChunkFile]` — functions, classes, methods (path, content, line ranges)
- `edges: list[EdgeData]` — dependency relationships (source, target, type, metadata)

Triggered by `FILE_WRITTEN` event. Chunks stored in `/.grover/chunks/`.

`AnalyzerRegistry` maps extensions to analyzers. Falls back to plaintext if no language-specific analyzer.

---

## Storage Backends

### LocalFileSystem

- Files on disk at `{workspace_dir}/{path}`
- Metadata + versions in SQLite at `~/.grover/{slug}/grover.db` (or custom `data_dir`)
- Capabilities: `StorageBackend`, `SupportsVersions`, `SupportsTrash`, `SupportsReconcile`
- External edit detection (IDE/git modifications)
- Safe for concurrent filesystem access

### DatabaseFileSystem

- Pure SQL storage (all content in database)
- Stateless, safe for concurrent web requests
- Supports SQLite, PostgreSQL, MSSQL via dialect-aware SQL
- Capabilities: `StorageBackend`, `SupportsVersions`, `SupportsTrash`

### UserScopedFileSystem

- Subclass of `DatabaseFileSystem` with per-user path namespacing
- Paths: `/notes.md` -> stored as `/{user_id}/notes.md`
- Virtual `@shared/{owner}/{path}` namespace
- Capabilities: `StorageBackend`, `SupportsVersions`, `SupportsTrash`, `SupportsReBAC`
- Requires explicit `user_id` on all operations

---

## Mount System

- Multiple backends simultaneously at virtual paths
- Longest-prefix matching for path resolution
- Permission boundaries (READ_WRITE, READ_ONLY)
- Hidden mounts (excluded from listing/indexing)
- Auto-created internal `/.grover` mount for graph edges, search metadata, chunks

---

## Event Bus

| Event | Triggers |
|-------|----------|
| `FILE_WRITTEN` | Re-analysis, graph update, search re-index |
| `FILE_DELETED` | Remove from graph, search index |
| `FILE_MOVED` | Remove old path, re-analyze at new path |
| `FILE_RESTORED` | Re-analyze file |

Async handlers. Exceptions logged, not propagated. Eventually consistent.

---

## Result Types

All operations return structured dataclasses:

`ReadResult`, `WriteResult`, `EditResult`, `DeleteResult`, `MoveResult`, `GlobResult`, `GrepResult`, `TreeResult`, `ListVersionsResult`, `ShareResult`, `ListSharesResult`

Each includes `success: bool` plus operation-specific fields.

---

## Data Models

| Table | Key Fields |
|-------|-----------|
| `grover_files` | path (unique), parent_path, owner_id, content, content_hash, size_bytes, current_version, deleted_at, original_path |
| `grover_file_versions` | file_id, version, is_snapshot, content (full or diff), content_hash, created_by |
| `grover_edges` | source_path, target_path, type, weight, metadata_json, is_derived, stale |
| `grover_file_shares` | path, grantee_id, permission, granted_by, expires_at |
| `grover_embeddings` | file_path, source_type, source_hash, model_name, dimensions |

---

## Known Limitations

1. **Single-tenant by default** — user-scoped mounts require explicit setup
2. **Synchronous event handlers** — can block file operations if slow
3. **Text-only versioning** — binary files get snapshots only, no diffs
4. **In-memory graph** — no persistence between sessions until `save()`
5. **Manual graph edges** — only auto-created by analyzers; manual additions need explicit calls
6. **Limited analyzer coverage** — Python/JS/TS/Go only
7. **External edit detection** — LocalFileSystem only
8. **No garbage collection** — deleted chunks remain until explicit cleanup
9. **Chunk path stability** — based on symbol name; renames break continuity
10. **No lazy loading** — entire graph loaded into memory on startup

---

## Not Yet Implemented (Planned)

- Full multi-tenancy (user_id scoping across all tables)
- Permission-aware graph and search queries
- Background workers for re-indexing
- Dangling edge auto-creation (placeholder files)
- Public/anonymous shares
- Share invitations and acceptance workflow
- Fuzzy matching and field-specific search
- Hybrid graph + search queries
- Graph visualization / export
- IDE integration (VSCode, LSP)
- CLI tool (`grover` command)
- MCP Server
- LangChain/LangGraph integration

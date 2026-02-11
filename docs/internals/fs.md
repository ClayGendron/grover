# Filesystem Internals

This document covers the internal design of Grover's filesystem layer: how writes happen, how sessions are managed, how versions are stored, and why certain ordering decisions were made.

For the high-level architecture diagram and component relationships, see [fs_architecture.md](../fs_architecture.md).

---

## Table of Contents

- [Write Order of Operations](#write-order-of-operations)
- [Delete Order of Operations](#delete-order-of-operations)
- [Session Lifecycle](#session-lifecycle)
- [Capability Protocols](#capability-protocols)
- [Version Snapshotting](#version-snapshotting)
- [Soft Delete and Trash](#soft-delete-and-trash)
- [Reconciliation](#reconciliation)
- [LocalFileSystem vs DatabaseFileSystem](#localfilesystem-vs-databasefilesystem)
- [Path Validation and Security](#path-validation-and-security)

---

## Write Order of Operations

### The Rule

All mutating operations (`write`, `edit`) follow the same sequence:

```
1. Save version record to session  (not yet committed)
2. Write content to storage         (write_content callback)
3. Flush the session                (session.flush)
4. VFS commits on _session_for exit (session.commit)
```

If step 2 fails, the VFS context manager calls `session.rollback()`, which discards the version record from step 1. The database is clean.

If step 4 fails after step 2 succeeds, the content exists in storage but the database has no record of it. This is an **orphan file** — invisible to the system and harmless.

### Why Not Commit First?

The opposite ordering — commit the session, then write content — creates **phantom metadata**: the database says a file exists at version N with hash X, but the content is missing or stale on disk. This is actively broken state:

- `read()` returns `None` content for a file the database says exists.
- `get_version_content()` fails hash verification.
- The rollback in the `except` block is useless because the commit already happened.

### Failure Mode Comparison

| Scenario | Content-First (correct) | Commit-First (wrong) |
|----------|------------------------|---------------------|
| Storage write fails | Rollback discards version record. **Clean.** | DB committed with version record pointing to missing content. **Broken.** |
| Commit fails after write | Orphan file on disk, DB has old metadata. **Inert.** | N/A — commit happened first. |
| Both succeed | File and metadata consistent. **Clean.** | File and metadata consistent. **Clean.** |

An orphan file on disk is strictly better than phantom metadata because:

- The system cannot see an orphan file (no DB record references it).
- Orphan files can be garbage-collected by `reconcile()`.
- Phantom metadata causes user-visible errors on every subsequent read.

### How It Applies to Each Backend

**LocalFileSystem**: Content is written to disk via atomic temp-file + rename. The commit then persists the version record and metadata to SQLite. If the disk write fails, the rollback removes the version record. If the SQLite commit fails, the disk file exists but is invisible — no user impact.

**DatabaseFileSystem**: `write_content()` updates the `File.content` column on the same session object. Both the content update and the version record are part of the same transaction, so they commit or roll back together atomically when VFS exits `_session_for()`.

### History

This ordering has been the intended design since the initial filesystem implementation. It was accidentally reversed in commit `f7e039a` ("Fix high-severity FS issues H1-H8") which swapped to commit-first with the comment "C1: prevents desync." That was incorrect and has been reverted. **Do not change this ordering.**

### Code Locations

Write and edit orchestration lives in `operations.py`. The pattern is:

```python
# operations.py — write_file() and edit_file()
await write_content(path, content, session)
await session.flush()
```

Sessions are always injected by VFS via `_session_for(mount)`. The VFS context manager handles commit on success and rollback on error. Backends and operations only call `session.flush()` to make changes visible within the transaction.

---

## Delete Order of Operations

Delete follows the same **content-before-commit** principle as writes:

### LocalFileSystem

1. DB soft-delete: set `deleted_at`, move to trash path → `session.flush()`
2. Disk unlink: remove the file from disk
3. Return `DeleteResult(success=True)`
4. VFS commits the session via `_session_for`

### Failure Modes

| Scenario | What happens | State |
|----------|-------------|-------|
| Unlink fails | Return `DeleteResult(success=False)`, session rolls back | Clean — file exists on disk and in DB |
| VFS commit fails after unlink | Disk file gone, DB rolled back (file appears "alive") | Phantom metadata — `reconcile()` fixes it |
| Both succeed | File trashed in DB, gone from disk | Clean |

The commit-failure case is rare (SQLite WAL + FULL sync) and reconcile handles it.

### DatabaseFileSystem

No disk involved — soft-delete is metadata-only (set `deleted_at`, move path to trash). Everything happens within the same DB transaction.

---

## Session Lifecycle

Sessions are managed by VFS and injected into backends. Every operation creates, uses, commits, and closes its own session.

### VFS `_session_for(mount)`

`VFS._session_for(mount)` is an async context manager that provides sessions to backend operations.

```python
@asynccontextmanager
async def _session_for(self, mount):
    if not mount.has_session_factory:
        yield None  # Non-SQL backends get None
        return

    session = mount.session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

SQL backends **fail fast** if session is None:

```python
def _require_session(self, session):
    if session is None:
        raise GroverError("LocalFileSystem requires a session")
    return session
```

### Backend Session Usage

All backend methods accept `session: AsyncSession | None = None` (keyword-only). SQL backends call `_require_session()` at the top of each method. Backends never create, commit, or close sessions — they only call `session.flush()` after mutations to make changes visible within the transaction. VFS handles the full session lifecycle (create → inject → commit/rollback → close).

**LocalFileSystem**: Uses the injected session for all metadata and version operations. Disk I/O is independent of the session.

**DatabaseFileSystem**: Stateless — holds no session factory, no cached session, and no mutable state. All DB operations use the injected session.

### SQLite Configuration (LocalFileSystem)

The SQLite engine is configured with these pragmas on every connection:

| Pragma | Value | Purpose |
|--------|-------|---------|
| `journal_mode` | `WAL` | Write-ahead logging; concurrent reads during writes |
| `synchronous` | `FULL` | fsync on every commit for durability |
| `busy_timeout` | `5000` | Wait 5 seconds on lock contention before raising `SQLITE_BUSY` |
| `foreign_keys` | `ON` | Enforce foreign key constraints |

---

## Capability Protocols

Backends implement the core `StorageBackend` protocol plus optional capability protocols. VFS checks capabilities at runtime using `isinstance()`.

### Protocol Hierarchy

```
StorageBackend (core — 12 methods)
    open, close, read, write, edit, delete, mkdir, move, copy, list_dir, exists, get_info
├── SupportsVersions (opt-in)
│     list_versions → ListVersionsResult
│     get_version_content → GetVersionContentResult
│     restore_version → RestoreResult
├── SupportsTrash (opt-in)
│     list_trash → ListResult
│     restore_from_trash → RestoreResult
│     empty_trash → DeleteResult
└── SupportsReconcile (opt-in)
      reconcile → dict[str, int]
```

### Implementation

| Backend | StorageBackend | SupportsVersions | SupportsTrash | SupportsReconcile |
|---------|:---:|:---:|:---:|:---:|
| LocalFileSystem | Y | Y | Y | Y |
| DatabaseFileSystem | Y | Y | Y | N |
| Custom (minimal) | Y | - | - | - |

### Unsupported Capability Handling

- **Targeted path operations** (e.g., `list_versions("/path")`) → VFS raises `CapabilityNotSupportedError`
- **Aggregate endpoints** (e.g., `list_trash()`, `empty_trash()`) → VFS silently skips unsupported mounts
- **`delete(permanent=False)`** on non-trash backends → Returns `DeleteResult(success=False)` with guidance to use `permanent=True`

`GroverAsync` catches `CapabilityNotSupportedError` and returns appropriate `Result(success=False, message=...)`. The agent loop always gets Results, never unhandled exceptions from normal operations.

---

## Version Snapshotting

Every `write()` and `edit()` creates a `FileVersion` record via `VersioningService`. Versions use a **snapshot + forward diff** strategy to balance storage efficiency with reconstruction speed.

### Snapshot Interval

```
SNAPSHOT_INTERVAL = 20
```

A version is stored as a **full snapshot** when any of these conditions are true:

- It is version 1 (the initial write).
- `version_num % SNAPSHOT_INTERVAL == 0` (every 20th version).
- There is no previous content to diff against (`old_content is None`).

All other versions store a **forward unified diff** from the previous content to the new content.

### Storage Format

| `is_snapshot` | `content` column contains |
|---------------|--------------------------|
| `True` | Full file text |
| `False` | Unified diff (compatible with `unidiff` library) |

The `content_hash` field always stores the SHA-256 of the **reconstructed** content (not the diff itself). This enables integrity verification on reconstruction.

### Reconstructing a Version

To retrieve version N:

1. Find the nearest snapshot at or before version N.
2. Fetch the chain: all versions from that snapshot through N, ordered ascending.
3. Start with the snapshot's full content.
4. Apply each subsequent diff in order using `apply_diff()`.
5. Verify: SHA-256 of the result must match the `content_hash` stored in version N.

```
v1 (snapshot) → v2 (diff) → v3 (diff) → ... → v20 (snapshot) → v21 (diff) → ...
```

To reconstruct v23: start from v20 (nearest snapshot), apply diffs for v21, v22, v23.

### Diff Utilities (`fs/diff.py`)

- `compute_diff(old, new)` — Generates a unified diff via `difflib.unified_diff`. Handles missing-newline-at-EOF markers required by the `unidiff` parser.
- `apply_diff(base, diff)` — Parses a unified diff with `unidiff.PatchSet` and applies hunks in reverse order. Returns the base unchanged if the diff is empty.
- `reconstruct_version(chain)` — Takes an ordered list of `(is_snapshot, content)` tuples, replays from the first snapshot, and returns the final text.

---

## Soft Delete and Trash

### How Soft Delete Works

When `delete(path, permanent=False)` is called (the default):

1. The file's `original_path` is set to its current `path`.
2. The file's `path` is rewritten to a trash path: `/__trash__/{file_id}/{name}`.
3. `deleted_at` is set to the current timestamp.
4. If the target is a directory, all children undergo the same transformation.
5. The session is flushed (VFS commits after return).

The trash path format uses the file's UUID to prevent collisions when multiple files with the same name are deleted.

### LocalFileSystem Specifics

Before soft-deleting, `LocalFileSystem.delete()`:

1. Reads the file content from disk.
2. If no DB record exists (the file was created outside Grover, e.g. by git or an
   IDE), creates a DB record and version 1 snapshot as a backup.
3. Calls `delete_file()` to perform the soft-delete in the DB.
4. Physically removes the file from disk (content-before-commit).

On restore (`restore_from_trash`), the content is written back to disk from the version history.

### Trash Operations

| Operation | Behavior |
|-----------|----------|
| `list_trash()` | Returns all files where `deleted_at IS NOT NULL`, showing `original_path` |
| `restore_from_trash(path)` | Looks up by `original_path`, clears `deleted_at`, restores `path`. Recursively restores children for directories. |
| `empty_trash()` | Permanently deletes all trashed files: removes version records, content, and file records. |

### Read Protection

`read()` rejects any path under `/__trash__/`. Trashed files are only accessible through `list_trash()` and `restore_from_trash()`.

---

## Reconciliation

`LocalFileSystem` implements `SupportsReconcile` with a `reconcile()` method that synchronizes disk state with DB state:

1. Walk all files on disk within the workspace.
2. For each disk file not in the DB → create a DB record (created).
3. For each DB file not on disk → soft-delete the DB record (deleted).
4. For each file that exists in both → compare content hashes and update if changed (updated).

VFS delegates `reconcile()` to capable backends, aggregating results across mounts.

---

## LocalFileSystem vs DatabaseFileSystem

| Aspect | LocalFileSystem | DatabaseFileSystem |
|--------|----------------|-------------------|
| **Content storage** | Files on disk at `workspace_dir` | `File.content` column in DB |
| **Metadata storage** | SQLite at `data_dir/` | External DB (PostgreSQL, MSSQL, etc.) |
| **`File.content` column** | Always `NULL` | Contains actual file text |
| **Instance state** | Workspace dir, SQLite engine, session factory | Dialect, file model, file version model (immutable) |
| **Session handling** | Session injected by VFS, flush only | Session injected by VFS, flush only |
| **Atomic writes** | Temp file + fsync + rename | Standard DB transaction |
| **IDE/git visibility** | Files are real on disk | No physical files |
| **`list_dir()` behavior** | Merges DB records with disk scan | DB records only |
| **Path security** | Symlink detection, workspace boundary enforcement | N/A (virtual paths only) |
| **Delete behavior** | Backs up content before delete, removes from disk | Metadata-only delete |
| **Restore behavior** | Writes content back to disk from version history | Metadata restoration (content already in DB) |
| **Capabilities** | Versions, Trash, Reconcile | Versions, Trash |

### When to Use Which

**LocalFileSystem** is the default for local development. Files live on disk where IDEs, git, and other tools can interact with them directly. The SQLite database provides versioning and metadata without interfering with normal filesystem access.

**DatabaseFileSystem** is designed for server deployments, cloud environments, and cases where all state should live in a single database. There are no physical files to manage, and the database's ACID properties handle all consistency guarantees.

---

## Path Validation and Security

### Normalization (`normalize_path`)

All paths are normalized before use:

- Ensures leading `/`.
- Collapses multiple slashes (`//` → `/`).
- Resolves `.` and `..` via `posixpath.normpath`.
- Strips trailing slashes (except root `/`).
- Applies Unicode NFC normalization.

### Validation (`validate_path`)

Rejects paths that contain:

- Null bytes (`\x00`).
- Control characters (ASCII 1-31 except common whitespace).
- Paths longer than 4096 characters.
- Empty or whitespace-only paths.

### LocalFileSystem Path Security

`_resolve_path_sync()` provides additional protection:

1. Normalizes the virtual path.
2. Joins with `workspace_dir` to get the candidate physical path.
3. Walks each path component, checking for symlinks at every level.
4. Resolves the final path and verifies it is within `workspace_dir` via
   `resolved.relative_to(workspace_dir.resolve())`.
5. Raises `PermissionError` if any symlink is detected or the resolved path
   escapes the workspace boundary.

This prevents both `../` traversal and symlink-based escape attacks.

# Filesystem Architecture

## Component Relationships

```mermaid
graph TD
    Caller["Caller (Grover facade)"]

    subgraph VFS
        UFS["VFS"]
        UFS_perm["_check_writable()"]
        UFS_route["resolve path → mount + rel_path"]
        UFS_session["_session_for(mount)"]
        UFS_emit["_emit(FileEvent)"]
        UFS_cap["_get_capability(backend, protocol)"]
    end

    subgraph Mounts
        MR["MountRegistry"]
        MC_proj["MountConfig /project"]
        MC_grover["MountConfig /.grover"]
        MC_sf["session_factory (SQL mounts)"]
        Perm["Permission (RW / RO)"]
    end

    subgraph Services
        Meta["MetadataService"]
        Ver["VersioningService"]
        Dir["DirectoryService"]
        Trash["TrashService"]
    end

    subgraph Operations
        Ops["operations.py"]
        Ops_write["write_file()"]
        Ops_edit["edit_file()"]
        Ops_delete["delete_file()"]
        Ops_move["move_file()"]
        Ops_copy["copy_file()"]
    end

    subgraph Backends
        Proto["StorageBackend (Protocol)"]
        CapVer["SupportsVersions (Protocol)"]
        CapTrash["SupportsTrash (Protocol)"]
        CapRecon["SupportsReconcile (Protocol)"]

        subgraph LocalFileSystem
            LFS["LocalFileSystem"]
            LFS_disk["_resolve_path → disk I/O"]
            LFS_db["SQLite at data_dir/"]
            LFS_session["session injected by VFS"]
        end

        subgraph DatabaseFileSystem
            DFS["DatabaseFileSystem"]
            DFS_content["content in File.content column"]
            DFS_session["stateless (dialect + models only)"]
        end
    end

    subgraph Models
        GF["File (grover_files)"]
        FV["FileVersion (grover_file_versions)"]
    end

    subgraph Support
        Dialect["dialect.py (upsert_file)"]
        Utils["utils.py (normalize_path, validate_path, ...)"]
        Types["types.py (ReadResult, WriteResult, ...)"]
        EB["EventBus"]
    end

    Caller --> UFS
    UFS --> UFS_perm --> Perm
    UFS --> UFS_route --> MR
    UFS --> UFS_session
    UFS --> UFS_cap
    MR --> MC_proj
    MR --> MC_grover

    MC_proj -->|"local mode"| LFS
    MC_grover -->|"local mode"| LFS
    MC_proj -->|"database mode (engine=)"| DFS
    MC_proj --> MC_sf

    UFS_session -->|"SQL mounts"| MC_sf

    UFS --> UFS_emit --> EB

    LFS -.implements.-> Proto
    LFS -.implements.-> CapVer
    LFS -.implements.-> CapTrash
    LFS -.implements.-> CapRecon
    DFS -.implements.-> Proto
    DFS -.implements.-> CapVer
    DFS -.implements.-> CapTrash

    LFS -->|composes| Meta
    LFS -->|composes| Ver
    LFS -->|composes| Dir
    LFS -->|composes| Trash
    DFS -->|composes| Meta
    DFS -->|composes| Ver
    DFS -->|composes| Dir
    DFS -->|composes| Trash

    LFS -->|delegates to| Ops
    DFS -->|delegates to| Ops

    Ops --> Ops_write
    Ops --> Ops_edit
    Ops --> Ops_delete
    Ops --> Ops_move
    Ops --> Ops_copy
    Ops --> Utils
    Ops --> Types

    Meta --> GF
    Ver --> FV
    Dir --> Dialect
```

**No base class.** Both `LocalFileSystem` and `DatabaseFileSystem` implement the `StorageBackend` protocol directly, composing shared services (`MetadataService`, `VersioningService`, `DirectoryService`, `TrashService`) and delegating to standalone orchestration functions in `operations.py`.

**DB mounts** are created via `engine=` or `session_factory=` on `GroverAsync.mount()`. The engine form auto-creates a session factory, detects the SQL dialect, and ensures tables exist. This produces a stateless `DatabaseFileSystem` instance (immutable config only — dialect, file model, schema) paired with a `session_factory` stored on `MountConfig`. VFS creates sessions from the factory per-operation and passes them to DFS via `session=`.

## Capability Protocols

Backends implement the core `StorageBackend` protocol plus optional capability protocols:

```
StorageBackend (core — 12 methods)
    session: AsyncSession | None = None on all methods
├── SupportsVersions (opt-in — 3 methods)
│     list_versions, get_version_content, restore_version
├── SupportsTrash (opt-in — 3 methods)
│     list_trash, restore_from_trash, empty_trash
└── SupportsReconcile (opt-in — 1 method)
      reconcile
```

VFS uses `isinstance(backend, protocol)` to check capabilities at runtime. Behavior for unsupported capabilities:

| VFS Method | Capability | Unsupported Behavior |
|-----------|-----------|---------------------|
| `list_versions(path)` | `SupportsVersions` | Raise `CapabilityNotSupportedError` |
| `get_version_content(path, ver)` | `SupportsVersions` | Raise `CapabilityNotSupportedError` |
| `restore_version(path, ver)` | `SupportsVersions` | Raise `CapabilityNotSupportedError` |
| `list_trash()` | `SupportsTrash` | Skip mount silently (aggregation) |
| `restore_from_trash(path)` | `SupportsTrash` | Raise `CapabilityNotSupportedError` |
| `empty_trash()` | `SupportsTrash` | Skip mount silently (aggregation) |
| `delete(permanent=False)` | No `SupportsTrash` | `DeleteResult(success=False)` |

`GroverAsync` catches `CapabilityNotSupportedError` and returns `Result(success=False)` — the agent loop always gets Results, never unhandled exceptions.

## Request Flow: `write("/project/hello.py", content)` — Local Mount

```mermaid
sequenceDiagram
    participant C as Caller
    participant UFS as VFS
    participant MR as MountRegistry
    participant P as Permission
    participant EB as EventBus
    participant LFS as LocalFileSystem
    participant Ops as operations.py
    participant DB as SQLite
    participant Disk as Disk I/O

    C->>UFS: write("/project/hello.py", content)
    UFS->>P: _check_writable("/project/hello.py")
    P-->>UFS: OK

    UFS->>MR: resolve("/project/hello.py")
    MR-->>UFS: (MountConfig /project, "/hello.py")

    UFS->>UFS: _session_for(mount) → creates session from mount.session_factory
    UFS->>LFS: write("/hello.py", content, session=session)

    LFS->>Ops: write_file(path, content, session, metadata=, versioning=, ...)

    Note over Ops: validate_path + normalize_path (NFC)
    Note over Ops: is_text_file check

    Ops->>DB: metadata.get_file() → SELECT existing file
    alt new file
        Ops->>DB: directories.ensure_parent_dirs (upsert, sets parent_path)
        Ops->>DB: INSERT File (with parent_path)
        Ops->>DB: versioning.save_version (snapshot)
    else existing file
        Ops->>DB: UPDATE File (version++)
        Ops->>LFS: read_content (old content)
        LFS->>Disk: read old bytes
        Ops->>DB: versioning.save_version (diff or snapshot)
    end

    Ops->>LFS: write_content("/hello.py", content, session)
    LFS->>Disk: atomic write (tmpfile + rename)
    Ops->>DB: session.flush()

    LFS-->>UFS: WriteResult(success=True)

    Note over UFS: _session_for exits: session.commit()

    UFS->>EB: emit(FILE_WRITTEN, path, content)
    Note over EB: handlers update graph + search

    UFS-->>C: WriteResult(success=True)
```

## Request Flow: `write("/data/hello.py", content)` — DB Mount

```mermaid
sequenceDiagram
    participant C as Caller
    participant UFS as VFS
    participant MR as MountRegistry
    participant P as Permission
    participant EB as EventBus
    participant DFS as DatabaseFileSystem
    participant Ops as operations.py
    participant DB as External DB

    C->>UFS: write("/data/hello.py", content)
    UFS->>P: _check_writable("/data/hello.py")
    P-->>UFS: OK

    UFS->>MR: resolve("/data/hello.py")
    MR-->>UFS: (MountConfig /data, "/hello.py")

    UFS->>UFS: _session_for(mount) → creates session from mount.session_factory
    UFS->>DFS: write("/hello.py", content, session=session)

    DFS->>Ops: write_file(path, content, session, metadata=, versioning=, ...)

    Note over Ops: validate_path + normalize_path (NFC)
    Note over Ops: is_text_file check

    Ops->>DB: metadata.get_file() → SELECT existing file
    alt new file
        Ops->>DB: directories.ensure_parent_dirs (upsert)
        Ops->>DB: INSERT File
        Ops->>DB: versioning.save_version (snapshot)
    else existing file
        Ops->>DB: UPDATE File (version++)
        Ops->>DFS: read_content (old content from DB)
        Ops->>DB: versioning.save_version (diff or snapshot)
    end

    Ops->>DFS: write_content → UPDATE File.content
    Ops->>DB: session.flush()

    DFS-->>UFS: WriteResult(success=True)

    Note over UFS: _session_for exits: session.commit()

    UFS->>EB: emit(FILE_WRITTEN, path, content)
    Note over EB: handlers update graph + search

    UFS-->>C: WriteResult(success=True)
```

## Session Lifecycle

Sessions are managed by VFS and injected into backends. Every operation creates, uses, commits, and closes its own session.

```mermaid
sequenceDiagram
    participant Caller
    participant UFS as VFS
    participant MC as MountConfig
    participant Backend as Backend (LFS / DFS)

    Caller->>UFS: write(path, content)
    UFS->>UFS: resolve path → mount

    alt SQL mount (has session_factory)
        UFS->>MC: session_factory() → new session
        UFS->>Backend: write(path, content, session=session)
        Note over Backend: Backend uses injected session
        Backend->>Backend: do work → services + operations → session.flush()
        Backend-->>UFS: WriteResult
        UFS->>UFS: session.commit() (on _session_for exit)
    else Non-SQL mount (no session_factory)
        UFS->>Backend: write(path, content, session=None)
        Backend-->>UFS: WriteResult
    end

    UFS-->>Caller: WriteResult
```

Non-SQL backends receive `session=None` and handle storage independently. SQL backends **fail fast** if session is None.

## Explicit Lifecycle (`open` / `close`)

Backends define `open()` and `close()` methods instead of context managers:

- `open()` — Called at mount time. LocalFS initializes its SQLite engine. DFS is a no-op.
- `close()` — Called on unmount/shutdown. LocalFS disposes its SQLite engine. DFS is a no-op.
- VFS `close()` calls `backend.close()` on all mounts.

There are no `__aenter__`/`__aexit__` methods on VFS, GroverAsync, or Grover.

## DB Mount Setup: `engine=` API

```mermaid
sequenceDiagram
    participant App
    participant GA as GroverAsync
    participant DFS as DatabaseFileSystem
    participant MC as MountConfig
    participant MR as MountRegistry
    participant Engine as AsyncEngine

    App->>GA: mount("/data", engine=engine)
    GA->>Engine: detect dialect (engine.dialect.name)
    GA->>Engine: ensure tables (File, FileVersion)
    GA->>GA: async_sessionmaker(engine) → session_factory
    GA->>DFS: DatabaseFileSystem(dialect, file_model, ...)
    Note over DFS: Stateless — no session, no engine ref
    GA->>MC: MountConfig(backend=DFS, session_factory=sf)
    GA->>MR: add_mount(config)
```

## Soft-Delete / Restore (Directories)

```mermaid
sequenceDiagram
    participant C as Caller
    participant Ops as operations.py / TrashService
    participant DB as Database

    rect rgb(255, 245, 238)
        Note over C,DB: Soft-delete directory
        C->>Ops: delete("/mydir")
        Ops->>DB: SELECT File WHERE path = "/mydir"
        Ops->>DB: SELECT children WHERE path LIKE "/mydir/%"
        loop each child
            Ops->>DB: SET child.original_path, child.path = trash, child.deleted_at = now
        end
        Ops->>DB: SET dir.original_path, dir.path = trash, dir.deleted_at = now
        Ops->>DB: session.flush()
    end

    rect rgb(240, 255, 240)
        Note over C,DB: Restore directory
        C->>Ops: restore_from_trash("/mydir")
        Ops->>DB: SELECT File WHERE original_path = "/mydir" AND deleted_at IS NOT NULL
        Ops->>DB: Restore dir: path = original_path, clear deleted_at
        Ops->>DB: SELECT children WHERE original_path LIKE "/mydir/%" AND deleted_at IS NOT NULL
        loop each child
            Ops->>DB: Restore child: path = original_path, clear deleted_at
        end
        Ops->>DB: session.flush()
    end
```

## Versioning Strategy

```mermaid
graph LR
    subgraph Version Chain
        V1["v1: SNAPSHOT (full content)"]
        V2["v2: forward diff"]
        V3["v3: forward diff"]
        V4["v4: forward diff"]
        V5["..."]
        V20["v20: SNAPSHOT"]
        V21["v21: forward diff"]
    end

    V1 --> V2 --> V3 --> V4 --> V5 --> V20 --> V21

    subgraph Reconstruct v3
        R1["Start from v1 snapshot"]
        R2["Apply v2 diff"]
        R3["Apply v3 diff"]
    end

    R1 --> R2 --> R3
```

Snapshots are stored every 20 versions (`SNAPSHOT_INTERVAL = 20`) and always for version 1. A `UniqueConstraint("file_id", "version")` prevents duplicate version records. Content integrity is verified via SHA-256 hash on reconstruction.

## Composition Stack

```
Services (stateful, hold model refs, receive session per-call):
  metadata.py    — MetadataService (get_file, exists, get_info, file_to_info)
  versioning.py  — VersioningService (save_version, delete_versions, list_versions, get_version_content)
  directories.py — DirectoryService (ensure_parent_dirs, mkdir)
  trash.py       — TrashService (list_trash, restore_from_trash, empty_trash)

Orchestration (stateless functions):
  operations.py  — write_file(), edit_file(), delete_file(), move_file(), copy_file(), list_dir_db()
                   Takes services + content callbacks as parameters.

Concrete Backends (no base class, implement protocol directly):
  LocalFileSystem    — composes all services + operations, owns disk I/O
  DatabaseFileSystem — composes all services + operations, content in DB
```

## Mount Resolution

```mermaid
graph LR
    Path["/project/src/app.py"]

    subgraph MountRegistry
        M1["/.grover → LocalFileSystem (internal)"]
        M2["/project → LocalFileSystem (workspace)"]
    end

    Path -->|longest prefix match| M2
    M2 -->|relative path| Rel["/src/app.py"]
    Rel -->|_resolve_path| Disk["~/{workspace}/src/app.py"]
```

## Database Schema

```mermaid
erDiagram
    grover_files {
        string id PK
        string path UK "indexed"
        string parent_path "indexed for list_dir"
        string name
        boolean is_directory
        string mime_type
        string content "NULL for LocalFS"
        string content_hash
        int size_bytes
        int current_version
        string original_path "set on soft-delete"
        datetime created_at
        datetime updated_at
        datetime deleted_at "NULL = active"
    }

    grover_file_versions {
        string id PK
        string file_id FK "indexed"
        int version "UNIQUE(file_id, version)"
        boolean is_snapshot
        string content "snapshot or unified diff"
        string content_hash "SHA-256"
        int size_bytes
        string created_by
        datetime created_at
    }

    grover_files ||--o{ grover_file_versions : "has versions"
```

## SQLite Pragmas (LocalFileSystem)

| Pragma | Value | Purpose |
|--------|-------|---------|
| `journal_mode` | `WAL` | Concurrent reads during writes; verified on connect |
| `synchronous` | `FULL` | Durability — fsync on every commit |
| `busy_timeout` | `5000` | Wait 5s on contention instead of immediate SQLITE_BUSY |
| `foreign_keys` | `ON` | Enforce FK constraints |

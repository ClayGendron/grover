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
    end

    subgraph Mounts
        MR["MountRegistry"]
        MC_proj["MountConfig /project"]
        MC_grover["MountConfig /.grover"]
        MC_sf["session_factory (all mounts)"]
        Perm["Permission (RW / RO)"]
    end

    subgraph Backends
        Proto["StorageBackend (Protocol)"]

        subgraph BaseFileSystem
            BFS["BaseFileSystem (ABC)"]
            BFS_crud["read / write / edit / delete"]
            BFS_query["exists / get_info"]
            BFS_dir["mkdir / list_dir / move / copy"]
            BFS_ver["_save_version / list_versions / get_version_content / restore_version"]
            BFS_trash["list_trash / restore_from_trash / empty_trash"]
        end

        subgraph LocalFileSystem
            LFS["LocalFileSystem"]
            LFS_disk["_resolve_path → disk I/O"]
            LFS_db["SQLite at ~/.grover/{slug}/"]
            LFS_override["overrides: read, delete, list_dir, mkdir, restore_from_trash"]
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
        Utils["utils.py (normalize_path, validate_path, split_path, replace, is_text_file, is_binary_file, format_read_output)"]
        Types["types.py (ReadResult, WriteResult, EditResult, DeleteResult, ...)"]
        EB["EventBus"]
    end

    Caller --> UFS
    UFS --> UFS_perm --> Perm
    UFS --> UFS_route --> MR
    UFS --> UFS_session
    MR --> MC_proj
    MR --> MC_grover

    %% Local mode: both mounts point to LocalFileSystem instances
    MC_proj -->|"local mode"| LFS
    MC_grover -->|"local mode"| LFS

    %% Database mode: mount points to DatabaseFileSystem instead
    MC_proj -->|"database mode (engine=)"| DFS
    MC_proj --> MC_sf

    UFS_session -->|"all mounts"| MC_sf

    UFS --> UFS_emit --> EB

    LFS -->|inherits| BFS
    DFS -->|inherits| BFS
    LFS -.implements.-> Proto
    DFS -.implements.-> Proto

    BFS --> BFS_crud
    BFS --> BFS_query
    BFS --> BFS_dir
    BFS --> BFS_ver
    BFS --> BFS_trash
    BFS_crud --> Utils
    BFS_crud --> Types
    BFS_ver --> GF
    BFS_ver --> FV
    BFS_dir --> Dialect

    LFS --> LFS_disk
    LFS --> LFS_db
    LFS --> LFS_override
    LFS --> LFS_session
```

**DB mounts** are created via `engine=` or `session_factory=` on `GroverAsync.mount()`. The engine form auto-creates a session factory, detects the SQL dialect, and ensures tables exist. This produces a stateless `DatabaseFileSystem` instance (immutable config only — dialect, file model, schema) paired with a `session_factory` stored on `MountConfig`. VFS creates sessions from the factory per-operation and passes them to DFS via `session=`.

## Request Flow: `write("/project/hello.py", content)` — Local Mount

```mermaid
sequenceDiagram
    participant C as Caller
    participant UFS as VFS
    participant MR as MountRegistry
    participant P as Permission
    participant EB as EventBus
    participant LFS as LocalFileSystem
    participant BFS as BaseFileSystem
    participant DB as SQLite
    participant Disk as Disk I/O

    C->>UFS: write("/project/hello.py", content)
    UFS->>P: _check_writable("/project/hello.py")
    P-->>UFS: OK

    UFS->>MR: resolve("/project/hello.py")
    MR-->>UFS: (MountConfig /project, "/hello.py")

    UFS->>UFS: _session_for(mount) → creates session from mount.session_factory
    UFS->>LFS: write("/hello.py", content, session=session)

    Note over BFS: validate_path + normalize_path (NFC)
    Note over BFS: is_text_file check

    BFS->>DB: SELECT existing file
    alt new file
        BFS->>DB: _ensure_parent_dirs (upsert, sets parent_path)
        BFS->>DB: INSERT File (with parent_path)
        BFS->>DB: INSERT FileVersion (snapshot)
    else existing file
        BFS->>DB: UPDATE File (version++)
        BFS->>LFS: _read_content (old content)
        LFS->>Disk: read old bytes
        BFS->>DB: INSERT FileVersion (diff or snapshot)
    end

    BFS->>LFS: _write_content("/hello.py", content, session)
    LFS->>Disk: atomic write (tmpfile + rename)
    BFS->>DB: session.flush()

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
    participant BFS as BaseFileSystem
    participant DB as External DB

    C->>UFS: write("/data/hello.py", content)
    UFS->>P: _check_writable("/data/hello.py")
    P-->>UFS: OK

    UFS->>MR: resolve("/data/hello.py")
    MR-->>UFS: (MountConfig /data, "/hello.py")

    UFS->>UFS: _session_for(mount) → creates session from mount.session_factory
    UFS->>DFS: write("/hello.py", content, session=session)

    Note over BFS: validate_path + normalize_path (NFC)
    Note over BFS: is_text_file check

    BFS->>DB: SELECT existing file
    alt new file
        BFS->>DB: _ensure_parent_dirs (upsert)
        BFS->>DB: INSERT File
        BFS->>DB: INSERT FileVersion (snapshot)
    else existing file
        BFS->>DB: UPDATE File (version++)
        BFS->>DFS: _read_content (old content from DB)
        BFS->>DB: INSERT FileVersion (diff or snapshot)
    end

    BFS->>DFS: _write_content → UPDATE File.content
    BFS->>DB: session.flush()

    DFS-->>UFS: WriteResult(success=True)

    Note over UFS: _session_for exits: session.commit()

    UFS->>EB: emit(FILE_WRITTEN, path, content)
    Note over EB: handlers update graph + search

    UFS-->>C: WriteResult(success=True)
```

## Session Lifecycle

Sessions are managed by VFS and injected into backends.

```mermaid
sequenceDiagram
    participant Caller
    participant UFS as VFS
    participant MC as MountConfig
    participant Backend as Backend (LFS / DFS)

    Caller->>UFS: write(path, content)
    UFS->>UFS: resolve path → mount

    Note over UFS: All mounts have session_factory set
    UFS->>MC: session_factory() → new session
    UFS->>Backend: write(path, content, session=session)
    Note over Backend: Backend uses injected session
    Backend->>Backend: do work → _write_content → session.flush()

    Backend-->>UFS: WriteResult
    UFS->>UFS: session.commit() (on _session_for exit)
    UFS-->>Caller: WriteResult
```

### Transaction Mode

In transaction mode (`async with grover:`), VFS reuses sessions per mount across operations for all mount types (local and DB). The commit happens on context-manager exit rather than after each operation.

```mermaid
sequenceDiagram
    participant App
    participant UFS as VFS
    participant Backend as Backend (LFS / DFS)

    App->>UFS: begin_transaction()

    App->>UFS: write("/mount/a.py", content)
    UFS->>UFS: _session_for(mount) → reuse txn session
    UFS->>Backend: write(path, content, session=txn_session)
    Backend->>Backend: flush (not commit)

    App->>UFS: write("/mount/b.py", content)
    UFS->>UFS: _session_for(mount) → same txn session
    UFS->>Backend: write(path, content, session=txn_session)
    Backend->>Backend: flush (not commit)

    App->>UFS: commit_transaction()
    UFS->>UFS: txn_session.commit() + close()
```

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
    participant BFS as BaseFileSystem
    participant DB as SQLite

    rect rgb(255, 245, 238)
        Note over C,DB: Soft-delete directory
        C->>BFS: delete("/mydir")
        BFS->>DB: SELECT File WHERE path = "/mydir"
        BFS->>DB: SELECT children WHERE path LIKE "/mydir/%"
        loop each child
            BFS->>DB: SET child.original_path, child.path = trash, child.deleted_at = now
        end
        BFS->>DB: SET dir.original_path, dir.path = trash, dir.deleted_at = now
        BFS->>DB: COMMIT
    end

    rect rgb(240, 255, 240)
        Note over C,DB: Restore directory
        C->>BFS: restore_from_trash("/mydir")
        BFS->>DB: SELECT File WHERE original_path = "/mydir" AND deleted_at IS NOT NULL
        BFS->>DB: Restore dir: path = original_path, clear deleted_at
        BFS->>DB: SELECT children WHERE original_path LIKE "/mydir/%" AND deleted_at IS NOT NULL
        loop each child
            BFS->>DB: Restore child: path = original_path, clear deleted_at
        end
        BFS->>DB: COMMIT
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

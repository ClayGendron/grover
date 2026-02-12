# Filesystem Design Analysis

How Grover's architecture compares to established filesystem patterns, where principles apply, and what lives at which layer.

---

## 1. What Grover Already Does Right

### VFS as Dispatch Layer (Linux VFS Pattern)

The Linux VFS uses **vtable-based dispatch** — each object (superblock, inode, file) carries a pointer to an operations struct, and the VFS calls generic functions that dispatch to the correct filesystem. Grover's `StorageBackend` protocol + `VFS.resolve()` → backend dispatch is exactly this pattern, just expressed in Python protocols instead of C function pointer structs.

| Linux VFS | Grover |
|-----------|--------|
| `file_system_type` with `mount()` callback | `StorageBackend` protocol with `open()` |
| `super_operations` (per-filesystem ops) | `StorageBackend` methods (read, write, etc.) |
| `inode_operations` + `file_operations` | Unified in `StorageBackend` (simpler, appropriate for the use case) |
| `mount()` registers a filesystem | `MountRegistry.add_mount()` |
| VFS dispatches to the correct `s_op`/`i_op`/`f_op` | `VFS.resolve()` → `mount.backend.method()` |

### Capability Protocols (Optional Feature Detection)

Grover's `SupportsVersions`, `SupportsTrash`, `SupportsReconcile` — checked at runtime via `isinstance()` — mirrors how Linux VFS handles optional filesystem features. In Linux, if a filesystem doesn't support (say) symlinks, it sets `inode_operations.symlink = NULL` and the VFS returns `EOPNOTSUPP`. Grover does the same thing but more explicitly: capability protocols make it a first-class pattern rather than null-checking.

This is actually *better* than the Linux approach because it's type-safe and self-documenting. PyFilesystem2 does something similar with its feature flags but less cleanly.

### Longest-Prefix Mount Resolution

`MountRegistry.resolve()` uses longest-prefix matching — exactly how Linux resolves mount points during `path_walk()`. When the kernel walks a path, it checks each dentry against the mount table and follows the deepest matching mount. Grover's implementation in `mounts.py` does the same thing in a single pass over `_mounts`.

### Cross-Mount Operations as Copy+Delete

When Linux encounters a `rename()` across mount boundaries, it returns `EXDEV` — the kernel refuses to do it. Userspace tools (mv, cp) then fall back to copy + delete. Grover's VFS does the same decomposition directly for cross-mount move: read from source backend → write to dest backend → delete from source. This is the established POSIX pattern.

### Content-Before-Commit (Crash Consistency)

Grover's write ordering (content to storage first, then commit metadata) matches the crash-safe pattern used by OverlayFS (xattr-then-rename for atomicity) and ZFS (COW with atomic transaction group pointer flip). The principle is universal: **never let metadata claim something exists if the data isn't there yet.**

### Composition Over Inheritance (Stackable FS Pattern)

Grover's shared services (MetadataService, VersioningService, TrashService, DirectoryService) composed into backends mirror the stackable filesystem approach from Wrapfs/eCryptfs. Instead of a class hierarchy, backends compose the services they need. This is the same principle — layered functionality — expressed through composition rather than filesystem stacking.

### EventBus (Application-Level Change Notification)

OS-level file watchers (inotify, FSEvents, ReadDirectoryChangesW) are all unreliable — they drop events under load, can't watch recursively (inotify), and have platform-specific quirks. Grover sidesteps this entirely by emitting events from the VFS layer itself. Since all mutations go through the VFS, the EventBus is a complete and reliable change stream. This is the right choice for an application-level filesystem.

### Forward Diffs with Periodic Snapshots (Fossil-Adjacent)

Grover's versioning strategy (snapshot at v1, forward diffs, periodic re-snapshot at `SNAPSHOT_INTERVAL=20`) is close to Fossil SCM's approach. Fossil stores blobs + deltas in SQLite with delta chaining. Grover's schema (`grover_files` + `grover_file_versions` with `is_snapshot` flag) is structurally similar. The SHA256 verification on reconstruction adds an integrity check that Fossil gets via content-addressable hashing.

---

## 2. What Lives Where (VFS vs Backend)

### VFS Responsibilities (Router / Policy / Lifecycle)

| Responsibility | Why VFS | Grover Implementation |
|---|---|---|
| **Mount resolution** | The VFS is the only layer that sees the full virtual namespace | `MountRegistry.resolve()` |
| **Permission enforcement** | Policy is above storage — backends shouldn't decide who can write | `VFS._check_write_permission()` |
| **Session lifecycle** | Transactions span operations, not storage — VFS owns commit/rollback | `VFS._session_for()` context manager |
| **Event emission** | Events are about *what happened in the virtual namespace*, not about storage internals | `VFS._emit()` after operations |
| **Capability gating** | The VFS decides what to do when a backend lacks a capability | `VFS._get_capability()` → `CapabilityNotSupportedError` |
| **Cross-mount coordination** | Only the VFS sees both mounts | Cross-mount move/copy in VFS |

### Backend Responsibilities (Storage / Content / Format)

| Responsibility | Why Backend | Grover Implementation |
|---|---|---|
| **Content storage/retrieval** | How bytes are stored (disk vs DB column) is a backend concern | `LocalFileSystem._read_content()` / `DatabaseFileSystem._read_content()` |
| **Path safety** | LocalFS must prevent path traversal; DatabaseFS doesn't have this concern | `LocalFileSystem._safe_resolve()` |
| **Content addressing** | How to map a path to actual storage (disk path vs DB query) | Backend-specific |
| **Format-specific operations** | Binary detection, mime types, encoding — depend on how content is stored | Shared via `utils.py` but called by backends |

### Shared Services (Composed Into Backends)

| Service | Why Shared | Notes |
|---|---|---|
| **MetadataService** | File records are the same schema regardless of storage | SQL-specific; non-SQL backends would skip this |
| **VersioningService** | Diff/snapshot logic is independent of where content lives | SQL-specific |
| **TrashService** | Soft-delete semantics are uniform | SQL-specific |
| **DirectoryService** | Directory creation logic is uniform for SQL backends | SQL-specific |

### Orchestration (operations.py)

The `operations.py` pattern — pure functions that take services + callbacks — sits between VFS and backend. This is analogous to Linux's `generic_file_read_iter()` and `generic_file_write_iter()` — generic implementations that most filesystems delegate to but can override. In Grover, both `LocalFileSystem` and `DatabaseFileSystem` use the same `write_file()` orchestration but provide different `ContentWriter` callbacks.

---

## 3. Where Principles Could Be Applied (Future Evolution)

These are not problems — they're opportunities identified from the research, organized from high-level to technical.

### Reverse Deltas for Latest-Version Optimization

**Layer: VersioningService (backend-level)**

Grover uses forward deltas: v1 is a snapshot, v2-v19 are diffs from the previous version. Reading the latest version of a file that's been edited 15 times requires loading the v1 snapshot and applying 14 diffs. Git (and most production systems) use **reverse deltas**: the latest version is stored as a full copy, and older versions are stored as backwards diffs. This makes the common case (read latest) O(1) instead of O(n).

The `SNAPSHOT_INTERVAL=20` mitigates this — worst case is 19 diff applications — but reverse deltas would eliminate the cost entirely. This is the single highest-impact optimization the research suggests, and it would be localized to `versioning.py` and `diff.py` without touching the VFS or backends.

### Optimistic Concurrency / Version Conflict Detection

**Layer: VFS**

The sync facade uses an `RLock`, but for multiple async clients there's no conflict detection. ZFS uses transaction groups with atomic pointer flips. DeltaV (WebDAV) uses explicit checked-in/checked-out states. A lighter-weight approach for Grover: **optimistic concurrency** — include the expected version number in write requests, and reject the write if the file has been modified since. This is what the `version` field on `Ref` could enable.

This would live in the VFS layer (policy enforcement), not in backends.

### Mount Lifecycle Events

**Layer: VFS / MountRegistry**

Linux 5.2+ introduced the "create then attach" mount API: `fsopen()` → `fsmount()` → `move_mount()`. This separates configuration from visibility. Grover could support runtime mount/unmount with lifecycle events (MOUNT_ADDED, MOUNT_REMOVED on the EventBus). This would let the graph and search layers react to new mounts without restarting.

### Backend Health / Degraded Mode

**Layer: VFS**

NFS has grace periods and stale handle detection. When a backend goes offline (DB connection lost, disk unmounted), the VFS currently has no degraded mode. A future `BackendHealth` protocol could let the VFS detect and handle backend failures — fail fast for the affected mount, continue serving others. This is a VFS concern, not a backend concern.

### Content-Addressable Deduplication

**Layer: VersioningService or a new shared service**

Git, Venti, and Restic all use content-addressable storage for deduplication. If two files have identical content, store it once. Grover already computes SHA256 hashes for every version — the building block is there. A dedup layer would sit between content writing and version storage, checking if a blob with the same hash already exists.

### Persistent Event Log

**Layer: EventBus (cross-cutting)**

Currently events are fire-and-forget in memory. If Grover crashes mid-update, the graph and search index may be inconsistent with the filesystem. A persistent event log (append to SQLite) would enable replay on startup — similar to ZFS's ZIL (intent log) or a write-ahead log.

### Graph Versioning / Temporal Queries

**Layer: Graph (not FS)**

The research on DeltaV baselines (snapshot a collection of version-controlled resources) suggests graph versioning: "what did the dependency graph look like at version N?" This is a graph-layer concern, not an FS concern. It would require versioning edges alongside file versions.

---

## 4. Summary

| Pattern | Status | Layer | Priority |
|---------|--------|-------|----------|
| VFS dispatch via protocols | Implemented | VFS | -- |
| Capability protocols | Implemented | Protocol | -- |
| Longest-prefix mount resolution | Implemented | MountRegistry | -- |
| Cross-mount as copy+delete | Implemented | VFS | -- |
| Content-before-commit | Implemented | operations.py | -- |
| Composition over inheritance | Implemented | Backends | -- |
| Application-level events | Implemented | EventBus | -- |
| Forward diffs + snapshots | Implemented | Versioning | -- |
| **Reverse deltas** | Not yet | Versioning | High |
| **Optimistic concurrency** | Not yet | VFS | Medium |
| **Runtime mount lifecycle** | Not yet | VFS/Registry | Medium |
| **Backend health/degraded mode** | Not yet | VFS | Low |
| **Content-addressable dedup** | Not yet (hash exists) | Versioning | Low |
| **Persistent event log** | Not yet | EventBus | Low |
| **Graph versioning** | Not yet | Graph | Low |

Grover's architecture is well-aligned with established filesystem design. The core patterns (VFS dispatch, mount resolution, capability protocols, session ownership, write ordering, composition) are all correct and match what the Linux VFS, PyFilesystem2, and other mature systems do. The "not yet" items are all evolutionary — none require architectural changes, and most are localized to specific layers.

---

## 5. Cross-Mount Memory Safety

All cross-mount data currently flows through Python RAM — `read()` returns the full content as a `str`, then `write()` passes that string to the destination backend. For text files this is fine (a 10,000-line source file is ~300-500KB), and real systems do the same thing (`mv` across mount boundaries on Linux reads the entire file into a userspace buffer). But there are alternatives if memory-bounded transfers become necessary.

### Option 1: Streaming via Async Iterators (Recommended)

Add `read_stream()` that yields chunks and `write_stream()` that consumes an async iterator:

```python
async for chunk in src_backend.read_stream(path, session=src_sess):
    await dest_backend.write_chunk(path, chunk, session=dest_sess)
```

Memory usage is bounded by chunk size, not file size. This is how fsspec works. Fits naturally as a `SupportsStreaming` capability protocol — the existing `read()`/`write()` with `str` stay unchanged for the normal single-mount case. Cleanest fit with Grover's architecture.

### Option 2: File-Like Objects

PyFilesystem2's core primitive is `openbin()` — returns a file-like object, not content. The VFS pipes source to dest without holding the whole thing:

```python
src_file = await src_backend.open(path, mode="r")
dest_file = await dest_backend.open(dest, mode="w")
```

Same idea as streaming iterators but uses Python's standard file-like interface. Downside: async file-like objects are awkward in Python — no stdlib `AsyncTextIOBase`.

### Option 3: Temp File as Intermediary

Zero protocol changes. Source writes to a temp file on disk, dest reads from it. Trades RAM for disk. However, the initial `read()` still loads full content into RAM, so this only helps if `read()` itself is also changed to write directly to the temp file — which brings you back to needing a streaming interface.

### Option 4: Shared Content Store (Content-Addressable)

If both backends share a content store (same SQLite DB or shared blob directory), a cross-mount "move" transfers just the *reference* — no data copy at all:

```python
hash = src_backend.get_content_hash(path)
dest_backend.link_content(dest_path, hash)  # points to same blob
src_backend.delete(path)
```

This is how Git handles things internally. Grover already computes SHA256 hashes for every file, so the building block exists. But requires a shared content-addressed blob store accessible by both backends — a significant architectural addition.

### SQL Backend Constraint: Streaming Is Not Possible

None of the streaming options above apply to `DatabaseFileSystem`. SQL TEXT/BLOB columns require the complete value at INSERT/UPDATE time — there is no "append to a column" operation in SQL. This constraint exists at the database level (SQLite, PostgreSQL, MSSQL) and cannot be worked around by application-level streaming.

| Source → Dest | Streaming helps? | Why |
|---|---|---|
| LocalFS → LocalFS | Yes | Disk to disk, chunk by chunk |
| LocalFS → DatabaseFS | No | DB INSERT needs full content |
| DatabaseFS → LocalFS | No | SQLAlchemy returns full TEXT value as a Python string |
| DatabaseFS → DatabaseFS | No | Full string on both sides |

`DatabaseFileSystem` would **not** implement `SupportsStreaming`. The full content must pass through RAM for any operation involving a SQL-backed backend. The only alternatives are:

- **Don't store content in SQL** — move blobs to disk or object storage, keep only metadata in the DB. But that's what `LocalFileSystem` already is.
- **Compress before storing** — zlib the content before writing to a BLOB column (like Fossil does). Smaller RAM footprint, but still the full value in memory. Saves storage, not peak memory.
- **Cap file size** — reject writes above a threshold. Doesn't reduce memory usage, just prevents the worst case.

This constraint matches Grover's design intent. The architecture already has the right answer for each scenario: large files go to `LocalFileSystem` (content on disk), structured/shared data goes to `DatabaseFileSystem` (content in SQL). The memory question resolves itself through mount selection.

### Assessment

For text-focused Grover, the current full-content approach is adequate. Streaming only benefits cross-mount transfers where **both** endpoints support it — practically, LocalFS-to-LocalFS or LocalFS-to-a-future-object-store backend. Any path involving `DatabaseFileSystem` requires full content in RAM regardless. If memory-bounded transfers are needed later, **Option 1 (async iterators as a `SupportsStreaming` capability)** is the best fit — it follows the existing capability protocol pattern, doesn't change the core `StorageBackend` interface, and only backends that support chunked I/O implement it.

---

## Sources

This analysis was synthesized from research into:

- [Linux VFS Documentation](https://www.kernel.org/doc/html/latest/filesystems/vfs.html)
- [OverlayFS Documentation](https://docs.kernel.org/filesystems/overlayfs.html)
- [FUSE-over-io_uring](https://www.kernel.org/doc/html/next/filesystems/fuse/fuse-io-uring.html)
- [9P2000 Protocol Spec](https://ericvh.github.io/9p-rfc/rfc9p2000.html)
- [New Mount API (LWN)](https://lwn.net/Articles/759499/)
- [Mount Namespaces and Propagation (LWN)](https://lwn.net/Articles/690679/)
- [fsspec Documentation](https://filesystem-spec.readthedocs.io/)
- [PyFilesystem2 Documentation](https://docs.pyfilesystem.org/)
- [Wrapfs Stackable Filesystem](https://wrapfs.filesystems.org/)
- [Fossil Repository Schema](https://fossil-scm.org/home/doc/f51856be/src/schema.c)
- [Git Pack Heuristics](https://git-scm.com/docs/pack-heuristics)
- [Restic CDC](https://restic.net/blog/2015-09-12/restic-foundation1-cdc/)
- [Venti: Archival Storage](http://doc.cat-v.org/plan_9/4th_edition/papers/venti/)
- [WebDAV DeltaV RFC 3253](https://datatracker.ietf.org/doc/html/rfc3253)

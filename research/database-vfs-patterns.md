# Database VFS Patterns

## Database-Native File Systems

### PostgreSQL: Large Objects and TOAST

**TOAST (The Oversized Attribute Storage Technique)** automatically handles values exceeding a single database page (~8 KB). Out-of-line values are chunked into ~2 KB rows in a secondary TOAST table. Transparent to the application — use `TEXT` or `BYTEA` columns. Supports compression (LZ4 or pglz), up to 1 GB per field.

**Large Objects (`pg_largeobject`)** provide a streaming API for data up to 4 TB. Data chunked into `LOBLKSIZE` pages in system catalog. Support random-access read/write. However, all large objects share a single system table (32 TB limit total) and require explicit lifecycle management.

**Relevance to Grover:** TOAST-backed `TEXT`/`BYTEA` columns are the pragmatic choice for files up to several MB (covers most code files and documents). Grover's current `content` column approach aligns well. Large Objects add complexity without clear benefit.

### SQLite: VFS Layer

SQLite's VFS is a formal abstraction with three core objects:

1. `sqlite3_vfs` — the VFS itself (registration, file opening, deletion)
2. `sqlite3_file` — an open file handle
3. `sqlite3_io_methods` — I/O operations (read, write, truncate, sync, lock/unlock)

Supports **shims** — wrapper VFS layers that intercept operations before delegating. Enables transparent encryption (CEVFS, SQLite3MultipleCiphers), compression (ZIPVFS, sqlite_zstd_vfs), replication (LiteFS), and quota management.

Python access via **APSW** (Another Python SQLite Wrapper) allows subclassing VFS and VFSFile in pure Python.

**Relevance to Grover:** Conceptual reference. Grover operates at the application layer (Python API + SQLModel), not the storage engine layer (page I/O). The shim pattern (layered wrappers) is analogous to Grover's composition model. APSW Python VFS could be interesting for encryption-at-rest.

### MySQL/MariaDB: BLOB Storage

InnoDB stores BLOBs inline when small, overflows to external pages beyond ~768 bytes. Conventional wisdom: store paths in DB, content on filesystem/object storage, because BLOB-heavy tables cause backup bloat, memory pressure, and replication overhead.

**Relevance to Grover:** Confirms storing content in DB is viable for code files (<1 MB). Grover's LocalFileSystem (disk + SQLite metadata) follows the recommended pattern. DatabaseFileSystem stores content in DB, fine for code/document use cases.

### MSSQL: FILESTREAM and FileTable

**FILESTREAM** stores `varbinary(max)` data as actual NTFS files while maintaining transactional consistency. Database Engine manages the physical file namespace. Data participates in SQL Server transactions, backup, and replication, but lives on filesystem for streaming performance.

**FileTable** extends FILESTREAM to expose a full Windows file namespace. Files stored in SQL Server appear as regular files via a Windows share. Windows file system operations are intercepted and reflected as relational data changes.

**Relevance to Grover:** FILESTREAM/FileTable is the closest industrial precedent — bridges the gap between database and filesystem. The separation of content storage (filesystem) from metadata management (database) with transactional control mirrors Grover's LocalFileSystem design. FileTable's concept of exposing database content as a navigable hierarchy is very close to Grover's VFS mount model.

---

## Version-Controlled Database Storage

### Git Internals: Content-Addressable Object Model

Four object types identified by SHA-1 hash:
- **Blobs** — file content (no filename)
- **Trees** — directory listings mapping names to blob/tree hashes
- **Commits** — immutable snapshots (root tree + parent commits + metadata)
- **Tags** — named references to commits

These form a **Merkle DAG** — directed acyclic graph where every node is identified by its content hash. Properties: deduplication (identical content stored once), immutability (changes produce new hashes), integrity (hash verification), history independence.

**Relevance to Grover:** Gold standard for versioned file storage. Grover's snapshot + forward diffs approach is simpler but less space-efficient. Content-addressing would add deduplication but complicate the schema. For AI agent use cases where files are frequently modified and versions need fast reconstruction, Grover's diff-based approach is likely more practical.

### DVC (Data Version Control)

Git-adjacent data versioning for ML. `dvc add` computes MD5 hash, moves file to `.dvc/cache/<hash>` (CAS), creates `.dvc` pointer file committed to Git. Actual data pushed to remote storage (S3, GCS, Azure, SSH). Acquired by lakeFS in November 2025.

**Relevance to Grover:** The "metadata in Git, content in CAS cache" pattern could inspire a future optimization where file content is stored in a content-addressed blob table and version records store only hashes.

### LakeFS: Git-Like Branching for Data Lakes

Metadata layer on top of object storage providing Git-like versioning. Core innovation: **zero-copy branching** (metadata-only operation, no data copied).

Internal architecture centers on **Graveler** versioning engine:
- **Ranges** — leaf nodes (SSTables) containing sorted key-value records, addressed by content hash
- **MetaRanges** — a special range containing all ranges, representing a complete keyspace view at a commit point
- Forms a **two-layer Merkle tree**: modifying files in one range rewrites only that range and its MetaRange; all others reused (>99% reuse)

Diffs computed by comparing range hashes — diff time proportional to difference size, not total data.

Committed metadata (immutable SSTables on object store) separated from uncommitted/staged metadata (mutable, stored in PostgreSQL/DynamoDB).

**Relevance to Grover:** The two-layer Merkle tree is elegant for efficient versioning at scale. Separation of committed (immutable) vs. uncommitted (mutable) metadata maps to Grover's model. Zero-copy branching could be valuable if Grover adds branching — branches as metadata-only references to existing version chains.

### Dolt: Version-Controlled SQL Database

MySQL-compatible database where every table is version-controlled. Built on **Prolly Trees** (Probabilistic B-trees):

- Content-addressed B-tree where chunk boundaries are determined by hash function
- **History independence** — same keys always produce same tree structure regardless of insertion order. Enables structural sharing.
- **Efficient diffing** — compare two versions by traversing only subtrees with different hashes. O(d) diff where d = difference size.
- **Chunk size control** — CDF-based formula produces normally-distributed chunks averaging 4 KB.
- **Key-only hashing** — chunk boundaries depend only on keys (not values), so value updates never shift boundaries.

Version history stored as a commit graph (Merkle DAG) of Prolly Trees. Branching creates a new pointer. Merging uses efficient three-way diffs.

Version control exposed through SQL: system tables (`dolt_log`, `dolt_diff`, `dolt_status`) and stored procedures (`dolt_commit`, `dolt_merge`, `dolt_branch`).

**Relevance to Grover:** Prolly Trees are the most interesting data structure for Grover's future — they solve the tension between B-tree performance and content-addressable versioning. However, implementing them is a major undertaking. More practical takeaway: Dolt's approach of exposing version control through the query interface (virtual tables) could inspire a similar pattern in Grover.

### Project Nessie

Transactional catalog adding Git-like semantics (branches, tags, commits) to Apache Iceberg table metadata. Versions at the catalog pointer level, not content level.

**Relevance to Grover:** Less directly applicable. The concept of versioning at the pointer/reference level is relevant — could enable cheaper branching than content-level versioning.

### TerminusDB

In-memory graph database with native Git-like revision control. Built on succinct data structures (Rust). Uses delta encoding — changes stored as append-only deltas. Database state is immutable — every change creates a new delta.

**Relevance to Grover:** Delta encoding on immutable storage is close to Grover's diff-based versioning. Key difference: TerminusDB operates on graph triples (RDF), Grover on file content.

---

## FUSE-Based Database File Systems

| Project | Database | Language | Status |
|---------|----------|----------|--------|
| MySQLFS | MySQL | C | Proof of concept |
| pgfs | PostgreSQL | Rust | Proof of concept |
| sqlfs | SQLite | Python | Maintained |
| Fuse::DBI | Any DBI | Perl | Legacy |
| SQLiteFS | SQLite | Haskell | Proof of concept |

Common pattern: one table for directory entries (path, parent, type), one for content (BLOBs), FUSE callbacks translating FS operations to SQL queries.

**Relevance to Grover:** Validates the concept but not production quality. Key lesson: **performance is the main challenge** — every file operation becomes a database round-trip. Grover avoids this by providing a Python API that can batch and optimize, rather than implementing raw POSIX syscalls.

---

## Content-Addressable Storage (CAS)

### How It Works

Data identified by cryptographic hash of content. Operations: `put(content) -> hash` and `get(hash) -> content`.

Properties:
- **Deduplication** — identical content stored once
- **Immutability** — content at a given hash never changes
- **Integrity** — retrieval by hash allows verification
- **History independence** — same content always produces same identifier

SHA-256 or SHA-3 recommended. SHA-1 (Git) deprecated due to collision vulnerabilities.

### IPFS: Distributed CAS

Merkle DAG where every node identified by Content Identifier (CID). Files chunked into ~256 KB pieces, each hashed individually, assembled into a tree. Root CID identifies entire file.

### Perkeep (formerly Camlistore): Personal CAS

Four composable layers:
1. **Storage** — `Get`/`Put` blobs by content hash (local disk, S3, Google Storage). Composable with sharding/replication.
2. **Index** — key-value index built on storage. Rebuildable from scratch.
3. **Search** — queries the index.
4. **UI** — web, CLI, API.

Handles mutability via **permanodes** (stable identity blobs) and **claims** (immutable, timestamped mutations). Current state reconstructed by applying all claims in order.

**Relevance to Grover:** Perkeep's permanode/claims model maps to Grover's `File` + `FileVersion` — file identity (path) is mutable, version content is immutable.

### Python CAS Libraries

- **HashFS** — content-addressable file management on disk
- **Fsdb** — CAS designed for many large files
- **FVS** — file versioning using hash-based deduplication

### CAS Value for Grover

1. **Deduplication** — agents frequently revert files; CAS stores identical content once
2. **Integrity verification** — content hashes provide free corruption detection
3. **Efficient version chains** — versions sharing content share storage
4. Strong candidate for v0.2/v0.3 enhancement, especially with branching

---

## Copy-on-Write and Snapshot Patterns

### ZFS/Btrfs Model

- Writes never overwrite — modified blocks go to new locations, pointers updated atomically
- Snapshots are O(1) — metadata freeze of current pointer tree
- Space grows only with divergence
- Btrfs send/receive generates binary diffs between snapshots

**Caveat:** CoW filesystems and databases conflict — databases do their own write management, CoW causes write amplification. Disable CoW for database directories.

### Database-Backed VFS Snapshot Strategies

1. **Full snapshot** — copy all file records. O(n). Simple but expensive.
2. **Pointer-based snapshot** — record snapshot ID; copy-on-write only for modified files. Divergence-proportional storage.
3. **Metadata-only snapshot** — with CAS, snapshot = mapping of `{path -> content_hash}`. O(1) creation. Most efficient.

Metadata-only composes naturally with CAS. For Grover's current diff-based versioning, pointer-based snapshots are the practical choice.

### Branching and Merging

Requirements:
1. **Branch creation** — record pointer to current state
2. **Isolated writes** — branch-specific overlay, not shared state
3. **Merging** — three-way diff (ancestor, source, destination), flag conflicts

LakeFS and Dolt demonstrate feasibility at scale with Merkle-tree storage. For Grover, simpler approach: branches as named snapshot references with copy-on-write for modifications.

---

## Multi-Tenant File Systems in Databases

### Tenancy Models

1. **Pool model (shared everything)** — one DB, one schema, one table. Isolation via RLS or `WHERE tenant_id = ?`. Cheapest, highest leakage risk.
2. **Bridge model (schema-per-tenant)** — shared DB, separate schemas. Better isolation, moderate overhead.
3. **Silo model (database-per-tenant)** — dedicated DB per tenant. Strongest isolation, highest cost.

### PostgreSQL Row-Level Security (RLS)

```sql
-- Add tenant column
ALTER TABLE grover_files ADD COLUMN tenant_id UUID NOT NULL;

-- Enable RLS
ALTER TABLE grover_files ENABLE ROW LEVEL SECURITY;

-- Create policy using session variable
CREATE POLICY tenant_isolation ON grover_files
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Set tenant context per request
SET app.current_tenant = '<tenant-uuid>';
```

Session variable pattern: set once per request, all queries auto-filtered. Database-layer enforcement.

**Performance:** RLS adds overhead (extra WHERE clause). Index on `tenant_id` essential. Complex policies should be encapsulated in functions.

### Grover's Current Approach

Application-level isolation via `UserScopedFileSystem`:
- Prefixes paths with `/{user_id}/`
- Scopes all queries to `owner_id`
- Equivalent to pool model with application-enforced isolation

This works across both SQLite and PostgreSQL. PostgreSQL RLS could be added as optional defense-in-depth — the `owner_id` column already exists, and a RLS policy would add database-enforced guarantees.

---

## Agent-Specific File Systems

### Turso AgentFS

SQLite-backed filesystem specifically for AI agents:
- Three interfaces: POSIX-like VFS (dentry + inode tables), key-value store (agent state), append-only audit log (tool call tracking)
- Single-file portability: entire agent runtime is one SQLite file
- FUSE integration: mount as real filesystem
- Philosophy: "Treat agent state like a filesystem, but implement it as a database"

| Aspect | AgentFS | Grover |
|--------|---------|--------|
| Storage | Single SQLite file | SQLite or PostgreSQL, local or pure-DB |
| Versioning | Snapshot-by-copy | Built-in version chain with diffs |
| Multi-user | No | Yes (UserScopedFileSystem, sharing) |
| Knowledge graph | No | Yes (rustworkx, code analyzers) |
| Semantic search | No | Yes (usearch HNSW index) |
| Audit | Append-only tool call log | Event bus + file events |
| FUSE | Yes | No (Python API) |

Grover is more feature-rich. AgentFS's simplicity (one file = one agent) is appealing for single-agent use cases.

---

## Sources

- [PostgreSQL TOAST and BLOBs | EDB](https://www.enterprisedb.com/postgres-tutorials/postgresql-toast-and-working-blobsclobs-explained)
- [PostgreSQL Large Objects](https://www.postgresql.org/docs/current/lo-intro.html)
- [SQLite VFS](https://sqlite.org/vfs.html)
- [APSW VFS Documentation](https://rogerbinns.github.io/apsw/vfs.html)
- [LiteFS | Fly Blog](https://fly.io/blog/introducing-litefs/)
- [FILESTREAM (SQL Server)](https://learn.microsoft.com/en-us/sql/relational-databases/blob/filestream-sql-server)
- [FileTable (SQL Server)](https://learn.microsoft.com/en-us/sql/relational-databases/blob/filetables-sql-server)
- [Git Objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- [Merkle DAG | IPFS](https://docs.ipfs.tech/concepts/merkle-dag/)
- [DVC Documentation](https://doc.dvc.org/start)
- [lakeFS Architecture](https://docs.lakefs.io/v1.73/understand/architecture/)
- [lakeFS Versioning Internals](https://docs.lakefs.io/v1.60/understand/how/versioning-internals/)
- [Prolly Trees | DoltHub](https://www.dolthub.com/blog/2024-03-03-prolly-trees/)
- [Dolt Architecture](https://docs.dolthub.com/architecture/architecture)
- [Perkeep Documentation](https://perkeep.org/doc/)
- [Multi-Tenant RLS | AWS](https://aws.amazon.com/blogs/database/multi-tenant-data-isolation-with-postgresql-row-level-security/)
- [PostgreSQL RLS](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [AgentFS | Turso](https://turso.tech/blog/agentfs)
- [AgentFS GitHub](https://github.com/tursodatabase/agentfs)

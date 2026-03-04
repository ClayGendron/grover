# Grover Cloud Architecture Research

## Question

How should Grover be deployed as a cloud SaaS product? Postgres-focused (one DB for SQL, vector, and full-text) or remote filesystem with Linux servers?

## Recommendation: Postgres-Focused

Grover's `DatabaseFileSystem` is already stateless and session-injected — designed for this. PostgreSQL covers all three layers:

| Grover Layer | PostgreSQL Feature | Status |
|---|---|---|
| File storage + versioning | Tables (`grover_files`, `grover_file_versions`) | Built |
| Vector search | pgvector extension | Needs `PgVectorStore` |
| Full-text search | tsvector + GIN indexes | Built (`PostgresFullTextStore`) |
| Chunks, connections, shares | Regular tables | Built |
| Multi-tenancy | `UserScopedFileSystem` path-prefix isolation | Built |

One database. One connection pool. ACID across all layers. Managed Postgres available everywhere (RDS, Cloud SQL, Supabase, Neon, Azure).

## Why Not a Remote Filesystem

`LocalFileSystem` on a Linux server means:

- **Session affinity required** — user files live on a specific server, kills horizontal scaling
- **Storage coupled to compute** — can't scale independently
- **Replication is your problem** — Postgres gives streaming replication for free
- **No cross-layer ACID** — disk write + DB write aren't in the same transaction
- **`reconcile()` solves a problem that doesn't exist in cloud** — you control the write path, nobody edits files on disk behind your back

## Graph Strategy: Build Per-Request, Not Global

The in-memory `RustworkxGraph` is the only truly stateful component. Strategy:

**Build from DB when needed, discard after.** The graph is a read model — `grover_file_connections` in Postgres is the source of truth. `from_sql()` reconstructs it from one `SELECT`. Most requests (file CRUD, search, versioning) don't touch the graph at all.

Why not a long-lived global graph:
- Multiple pods with separate in-memory graphs drift out of sync
- Requires cross-pod invalidation (Redis pub/sub, etc.) for consistency
- Memory bloat from idle workspace graphs

Scaling ladder if per-request construction becomes slow:

1. **Request-scoped reuse** — build once per request if multiple graph calls happen
2. **Short-TTL process cache** — cache per-mount with 5-30s TTL, invalidate on local writes via `EventBus`
3. **External graph DB** — `GraphStore` protocol supports alternative backends (Neo4j, Memgraph) if needed

## Missing Piece: `PgVectorStore`

The only significant new code for launch. Would:

- Store vectors on `grover_files` / `grover_file_chunks` (existing `Vector[N]` type handles this)
- Use `SELECT ... ORDER BY embedding <=> $1 LIMIT $k` for search
- Participate in the same `AsyncSession` as file writes (ACID vector updates)
- Implement `SupportsMetadataFilter` by compiling filters to SQL WHERE clauses

## Phased Rollout

**Phase 1 — MVP:** PostgreSQL + pgvector + tsvector. Graph from DB per-request. FastAPI creating `AsyncSession` per-request into `GroverAsync`.

**Phase 2 — Scale:** Pinecone/Databricks for large vector corpora (already built, one-line swap). Push `grep` into Postgres. Graph caching. PgBouncer connection pooling.

**Phase 3 — Enterprise:** MSSQL for Azure customers (already built). Schema-based DB isolation. External graph DB if needed.

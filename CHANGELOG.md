# Changelog

All notable changes to Grover will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.0.11] — 2026-04-02

### Added

- **v2 rewrite** — Complete rewrite of Grover around the "everything is a file" philosophy. New unified `grover_objects` table replaces the four separate tables (files, chunks, versions, connections). All entities are path-addressable with dot-prefixed metadata directories (`.chunks/`, `.versions/`, `.connections/`).
- **`GroverFileSystem` base class** — Concrete async base class with mount routing, session management, and path rebasing. Subclasses override `_*_impl` methods for storage.
- **`DatabaseFileSystem`** — Full SQL-backed implementation with CRUD, glob, grep, tree, versioning (snapshot + forward diffs), soft-delete, cascading operations, and LIKE wildcard escaping for path safety.
- **CLI query engine** — Hand-rolled tokenizer, parser, AST, executor, and renderer. Unix-like pipeline syntax with `|`, `&`, `intersect()`, `except()`. Commands: `read`, `write`, `edit`, `rm`, `mv`, `cp`, `mkdir`, `mkconn`, `ls`, `tree`, `glob`, `grep`, `search`, `lsearch`, `vsearch`, graph traversal, and ranking.
- **Graph algorithms** — All 10 graph algorithms implemented on `RustworkxGraph`: ancestors, descendants, neighborhood, meeting subgraph, min meeting subgraph, PageRank, betweenness/closeness/degree centrality, and HITS. Cross-validated against NetworkX at 10K nodes.
- **BM25 lexical search** — Hand-rolled BM25 scorer with SQL-hybrid pipeline for `lexical_search`. No external dependencies.
- **`EmbeddingProvider` and `VectorStore` protocols** — Pluggable embedding and vector search with `DatabricksVectorStore` and `LangChainEmbeddingProvider` implementations.
- **User-scoped filesystem** — Per-user path-prefix isolation via `user_scoped=True` on `DatabaseFileSystem`. Strict scope/unscope at DB boundary.
- **`GroverAsync` and `Grover` facades** — Async facade for app servers, sync wrapper for scripts and notebooks. `raise_on_error` flag with classified exception hierarchy (`NotFoundError`, `MountError`, `WriteConflictError`, `ValidationError`, `GraphError`).
- **Composable result types** — `GroverResult` with `Candidate` and `Detail` objects. Set algebra (`&`, `|`, `-`), enrichment chains (`sort`, `top`, `filter`, `kinds`).

### Changed

- **README rewritten** — New README focused on the v2 direction with code-first examples, design principles, API table, and namespace diagram.
- **Type checker upgraded** — Migrated from ty 0.0.16 to 0.0.27 with `# ty: ignore` inline comments replacing `# type: ignore`.
- **BM25 comparison tests moved to scripts** — `test_bm25_comparison.py` (requires `rank_bm25`) moved to `scripts/bm25_comparison.py`. Replaced with standalone `test_bm25.py`.

## [0.0.10] — 2026-03-19

### Added

- **Unified `GroverResult` type** — All facade operations (`write_files`, `read`, `edit`, `delete`, `move_files`, `copy_files`, `read_files`, `write_chunks`, `reconcile`) now return `GroverResult` with typed `Detail` objects providing per-file success/failure tracking, messages, and metadata.
- **Batch `move_files()`, `copy_files()`, `read_files()`** — New facade methods for bulk operations with per-file result tracking.
- **`ValidatedSQLModel` base class** — Runtime Pydantic validation for SQLModel instances constructed from non-database sources (DataFrames, dicts). DB models opt in via `ValidatedSQLModel` inheritance; `model_from_mount()` on `GroverContext` resolves the correct model class per mount.
- **`move --follow` support** — `move_files()` gains a `follow` parameter to update graph connections after moves.
- **`list_dir` returns candidates** — Directory listing now returns typed result objects.
- **`ReconcileDetail`** — New detail type for per-file reconcile tracking (added/updated/deleted).

### Changed

- **Version/trash facade methods removed** — `versions()`, `rollback()`, `trash()`, `restore()`, `empty_trash()` removed from facade, backends, and protocols. Deletions now cascade automatically through the database.
- **`read()` simplified** — `offset` and `limit` parameters removed from `read()` across facade, backends, and utilities. Content slicing is no longer a backend concern.
- **`write_chunks` refactored** — Follows the same thin-facade pattern as `write_files`, with chunking logic pushed into `ChunkProvider`.
- **`Ref` internals unified** — Internal ref parsing consolidated into `models/internal/ref.py` with enhanced decomposition.
- **Mount naming and DB model base classes simplified** — Cleaner mount initialization, streamlined dialect helpers.

### Fixed

- **ty 0.0.23 compatibility** — Protocol mismatches and or-pattern union errors resolved for latest type checker.
- **`GroverResult` migration alignment** — Backend return types, batch type exports, and test expectations aligned with unified result model.

## [0.0.9] — 2026-03-17

### Changed

- **`vector` field renamed to `embedding`** — `FileModel.embedding`, `FileChunkModel.embedding`, and `FileVersionModel.embedding` replace the old `vector` field. Pydantic's `Vector` type still handles serialization under the hood.
- **`created_at`/`updated_at` default to `None`** — File models no longer eagerly timestamp at construction time. The backend fills timestamps on write if not set. This makes DataFrame-sourced bulk writes cleaner.
- **Mount `path` → `name`** — `add_mount("project", ...)` replaces `add_mount("/project", ...)`. Mount names are simple identifiers (no `/` allowed). `mount.path` is derived internally as `f"/{name}"`.

### Added

- **`tokens` field** — `FileModel` and `FileChunkModel` gain a `tokens: int = 0` field for storing token counts.
- **`FileModelBase.create()` factory** — `FileModel.create("a.py", "code", mount="project", embedding=[0.1, ...], tokens=150)` builds a fully-populated model with computed hash, size, mime type, line count, and timestamps.
- **`write_files()` model flow-through** — Facade preserves all user-set fields (embedding, tokens, owner_id) through to the backend instead of stripping models down to `{path, content}`.

### Fixed

- **Variable shadowing in `_create_engine_mount`** — A `for name in table_names` loop variable overwrote the `name` parameter, causing engine-config mounts to get the wrong name when tables were created.

## [0.0.8] — 2026-03-17

### Added

- **Schema-aware table creation** — When `EngineConfig` provides a `schema`, `add_mount()` now creates the schema if it doesn't exist (PostgreSQL, MSSQL) and creates tables within that schema using `schema_translate_map`. Logs `Schema created: "name"` and `Tables created: ...` when new objects are created.
- **`ensure_schema()`** — Dialect-aware helper in `grover.util.dialect` that creates a database schema if missing. Supports PostgreSQL (`CREATE SCHEMA IF NOT EXISTS`), MSSQL (conditional `CREATE SCHEMA`), and no-ops on SQLite.
- **`check_tables_exist()`** — Helper that returns which table names already exist in a given schema, used to determine whether to log table creation messages.

## [0.0.7] — 2026-03-16

### Added

- **`EngineConfig`** — Frozen dataclass for engine-managed mounts. Accepts `url` (simple) or `engine_factory` (advanced, e.g. custom pool/connect_args via `create_async_engine_factory`). Supports `schema`, `create_tables`, and custom model overrides. Engine is created at mount time and disposed on unmount/close.
- **`SessionConfig`** — Frozen dataclass for app-managed mounts. Wraps an existing session factory; Grover does not dispose the engine. Dialect inferred from the factory's bind or set explicitly.
- **`create_async_engine_factory()`** — Helper that captures `create_async_engine` args and returns a zero-arg callable for deferred engine creation.
- **Engine lifecycle on `Mount`** — `Mount.engine` field tracks Grover-owned engines. `unmount()` and `close()` dispose them automatically.

### Changed

- **`add_mount()` API redesigned** — `engine=`, `session_factory=`, `dialect=`, `file_model=`, `file_version_model=`, `file_chunk_model=`, `db_schema=` parameters replaced by `engine_config=` and `session_config=`. `filesystem` and `mount` are now keyword-only. `path` remains positional.
- **`DatabaseFileSystem` constructor simplified** — Config parameters (dialect, schema, model classes) removed. Only provider kwargs remain. New `_configure()` method applies settings from `EngineConfig`/`SessionConfig` at mount time.
- **`LocalFileSystem` constructor simplified** — Model and schema parameters removed. Accepts only `workspace_dir`, `data_dir`, and provider kwargs.
- **`UserScopedFileSystem` constructor simplified** — Config parameters removed. Accepts only `share_model` and provider kwargs.
- **deepagents integration** — `from_database()` and `from_database_async()` now accept `EngineConfig` instead of raw engine/session_factory.

### Fixed

- **Sync `Grover` in Jupyter** — Pre-created `AsyncEngine` bound to the wrong event loop no longer causes failures. `EngineConfig` defers engine creation to mount time, ensuring it runs on the correct loop.

## [0.0.6] — 2026-03-16

### Changed

- **`max_length=1024` on all indexed string columns** — All `id`, `path`, `file_path`, `source_path`, `target_path`, `owner_id`, and `grantee_id` fields across all five DB models (`grover_files`, `grover_file_chunks`, `grover_file_connections`, `grover_file_shares`, `grover_file_versions`) now have explicit `max_length=1024` for compatibility with databases that require bounded index columns.

## [0.0.5] — 2026-03-16

### Added

- **`FileSearchSet` candidate container** — unordered set with set algebra (`&`, `|`, `-`, `>>`), path transforms (`rebase`, `remap_paths`), and iteration. Used as input filter for search methods.
- **`BatchResult`** — batch operation result type with `succeeded`/`failed` counts.
- **Candidates filtering** — `glob`, `grep`, `vector_search`, `lexical_search`, `hybrid_search` all accept `candidates: FileSearchSet` for pipeline-style filtering.
- **`diff_versions`** — compare two file versions, exposed on `Grover`/`GroverAsync`.
- **`write_file`, `write_files`, `write_chunk`, `write_chunks`** — public API methods for model-based writes.
- **Self-managing graph** — `RustworkxGraph` lazy-loads from DB, TTL-based refresh, `configure_refresh()`.
- **`IndexConfig`** — frozen dataclass for vector index creation, defined in `providers/search/protocol.py`.
- **`parent_path_from_id`** — utility to extract parent file path from chunk IDs (`/a.py#login` → `/a.py`).

### Changed

- **SearchProvider protocol stripped to MVP** — 6 methods: `connect`, `close`, `create_index`, `upsert(files=)`, `delete(files=)`, `vector_search(candidates=)`. Uses domain types (`File`, `BatchResult`, `FileSearchResult`) instead of search-specific types. `lexical_search` moved to filesystem backend (DB-native FTS).
- **`types.py` deleted** — `VectorEntry`, `UpsertResult`, `DeleteResult`, `VectorHit`, `SparseVector`, `TextEntry`, `IndexConfig` (old), `IndexInfo`, `SearchResult` all removed.
- **Opt-in search protocols removed** — `SupportsNamespaces`, `SupportsMetadataFilter`, `SupportsIndexLifecycle`, `SupportsHybridSearch`, `SupportsReranking` deleted from `protocol.py`. Pinecone/Databricks keep these as concrete methods.
- **DatabricksVectorStore rewritten** — stripped to protocol surface only (`connect`, `close`, `create_index`, `upsert`, `delete`, `vector_search`).
- **Internal result types overhauled** — `FileSearchResult` extends `FileSearchSet`, typed `Evidence` subclasses on `File` objects replace 30+ result subclasses. `GraphProvider` returns `FileSearchResult` directly.
- **RustworkxGraph refactored** — adjacency dicts, `_snapshot()` for thread-safe algorithm execution, `.graph` property exposes `PyDiGraph`, public attributes replace getters.
- **Async graph operations** — all query/algorithm methods on `RustworkxGraph` are `async def`. Heavy algorithms use `asyncio.to_thread()`.
- **Graph facade simplified** — `GraphOpsMixin` is pure delegation to `GraphProvider`. `GraphStore` alias removed.
- **`GroverAsync` mixins consolidated** — 8 → 6 mixins (ConnectionMixin absorbed into GraphOpsMixin, VersionTrashMixin into FileOpsMixin).
- **Search method signatures corrected** — `list_dir`, `tree`, `list_versions` reverted to path-based; `vector_search` rebase bug fixed.

### Fixed

- **CI workflow** — `uvx ruff`/`uvx ty` instead of pip-installed tools; trigger on workflow file changes.
- **ty type errors** — evidence lists annotated as `list[Evidence]`, `Any` replaced with concrete types.

## [0.0.4] — 2026-03-04

### Added

- **Background worker** — `BackgroundWorker` with per-path debounced task scheduling, `flush()`/`drain()` lifecycle, and `IndexingMode` (background vs manual). Replaces `EventBus`.
- **Version chain verification** — `verify_chain()`, `verify_versions()`, and `verify_all_versions()` for proactive integrity checking across backends, facade, and sync wrapper.
- **Composable search pipeline** — `vector_search()`, `lexical_search()`, `hybrid_search()` with `SearchProvider` protocol. BM25 full-text search via `FullTextStore` with SQLite/PostgreSQL/MSSQL backends.
- **Connection service** — `ConnectionService` in `DatabaseFileSystem` for filesystem-owned persistent edges. Graph is now a pure in-memory projection loaded from DB via `from_sql()`.
- **Result type algebra** — `FileOperationResult` and `FileSearchResult` base types with set operations (`&`, `|`, `-`, `>>`). Candidates-based search results. `GraphResult` for graph method returns.
- **`Ref` identity type** — Thin frozen wrapper with lazy path decomposition. Factories for chunk, version, and connection refs. Replaces `file_ref()` and `fs/paths.py`.
- **`GroverContext` dataclass** — Shared state for the facade, accessed via `self._ctx`. `GroverAsync` decomposed into 8 mixins (mount, file ops, search ops, graph ops, version/trash, share, connection, index).
- **Native async integrations** — deepagents and LangChain integrations accept `Grover | GroverAsync`.

### Changed

- **Package restructure** — `fs/` → `backends/`, `facade/` → `api/`, `types/` → `results/`, `graph/` and `search/` → `providers/graph/` and `providers/search/`, analyzers promoted to top-level `analyzers/`.
- **Protocol consolidation** — `SupportsVersions`, `SupportsTrash`, `SupportsSearch`, `SupportsConnections`, `SupportsFileChunks` merged into `GroverFileSystem`. Only `SupportsReBAC` and `SupportsReconcile` remain as opt-in. Graph protocols collapsed from 8 → 1, search from 9 → 6, storage from 3 → 1.
- **Filesystem-centric providers** — `DatabaseFileSystem` owns all providers directly. Provider protocols co-located in `providers/<family>/protocol.py`. `Mount` stripped to minimal dataclass (no graph/search).
- **`LocalFileSystem`** simplified to thin `DatabaseFileSystem` subclass (~330 lines).
- **`SharingService`** inlined into `UserScopedFileSystem`.
- **`DatabaseFileSystem`** flattened — internal mixins and services inlined.
- **Graph terminology** — `dependents`/`dependencies` renamed to `predecessors`/`successors`. `impacts`, `ancestors`, `descendants` removed.
- **Type safety** — `Any` replaced with concrete types across the codebase. `FilterValue` type alias added. `ty check src/` passes clean.
- **Backward compat aliases removed** — `mount()` → `add_mount()`, `SentenceTransformerProvider` alias deleted, `GroverEdge` removed, `query_types.py` deleted.
- **Dead code deleted** — `SearchEngine`, `FullTextStore`, `EventBus`, `Embedding` model, `VectorStore` protocol, mount dispatch protocols, `vfs.py`.

### Fixed

- **Path traversal vulnerability** in `UserScopedFileSystem._resolve_path`.
- **Read-only permission enforcement** across all mutation paths.

## [0.0.3] — 2026-02-19

### Added

- **Graph protocol hierarchy** — `GraphStore` core protocol + 7 capability protocols (`SupportsCentrality`, `SupportsConnectivity`, `SupportsTraversal`, `SupportsSubgraph`, `SupportsFiltering`, `SupportsNodeSimilarity`, `SupportsPersistence`), following the same `@runtime_checkable` pattern as the filesystem layer.
- **Graph algorithms** — Centrality (PageRank, betweenness, closeness, katz, degree), connectivity (weakly/strongly connected components), and traversal (ancestors, descendants, topological sort, shortest paths, all simple paths) on `RustworkxGraph`.
- **Subgraph extraction** — `subgraph()`, `neighborhood()` (BFS with direction/edge-type filters), `meeting_subgraph()` (pairwise shortest paths + PageRank scoring + pruning), and `common_reachable()`.
- **Graph filtering** — `find_nodes()` with callable predicates or equality matching, `find_edges()` by type/source/target, `edges_of()` with direction filtering.
- **Node similarity** — Jaccard coefficient via `node_similarity()` and `similar_nodes()` (top-k).
- **`SubgraphResult` type** — Frozen dataclass with deep immutability (`tuple` fields, `MappingProxyType` scores).
- **Public API surface** — `GraphStore` and `SubgraphResult` exported from `grover`. Convenience wrappers on `Grover`/`GroverAsync` for `pagerank`, `ancestors`, `descendants`, `meeting_subgraph`, `neighborhood`, `find_nodes` with `isinstance`-based capability checking.

### Changed

- **`Graph` → `RustworkxGraph`** — Renamed with no backward-compatible alias. All imports migrated.
- **`GroverAsync.graph` is now a public attribute** typed as `GraphStore` (was `self._graph`).
- Removed `SentenceTransformerProvider` backward-compat alias — use `SentenceTransformerEmbedding`.

## [0.0.2] — 2026-02-17

### Added

- **User-scoped file systems** — `UserScopedFileSystem` backend with per-user path namespacing, owner-scoped trash, and `@shared` virtual directory for cross-user access.
- **Sharing service** — Path-based share/unshare with permission resolution (read-only, read-write), expiration support, and directory inheritance.
- **External edit detection** — Synthetic version insertion to preserve version chain integrity when files change outside Grover.
- **Move with `follow` semantics** — `follow=True` renames in place; `follow=False` creates a clean break.
- **deepagents integration** — `GroverBackend` (BackendProtocol) and `GroverMiddleware` (10 tools for version, search, graph, and trash operations).
- **LangChain/LangGraph integration** — `GroverRetriever`, `GroverLoader`, and `GroverStore` for RAG pipelines and persistent agent memory.
- **Public API additions** — `user_id`, `share`, `unshare`, `list_shares`, `list_shared_with_me`, `move`, `copy`, `overwrite`, `replace_all`, `offset`/`limit` parameters threaded through the full stack.
- **Authorization hardening** — Fixed 6 bypass vulnerabilities in `UserScopedFileSystem`.

### Changed

- Bumped minimum Python requirement from 3.10 to 3.12.
- Scoped CI triggers: tests run on `src/`/`tests/` changes, docs build on `docs/` changes.

### Fixed

- `_list_shared_dir` now supports file-level shares via filtered fallback.
- SQL `LIKE` wildcards properly escaped in `update_share_paths`.
- Loader non-recursive `size_bytes` calculation and binary file skip behavior.

## [0.0.1] — 2026-02-11

Initial alpha release.

### Added

- **Two storage backends** — `LocalFileSystem` (disk + SQLite) for local dev, `DatabaseFileSystem` (pure SQL) for web apps and shared knowledge bases.
- **Mount-based VFS** — Routes operations to the right backend by path prefix; mount multiple backends side by side.
- **Automatic versioning** — Diff-based storage (snapshots + forward diffs) with SHA-256 integrity checks.
- **Soft-delete trash** — Restore or permanently delete.
- **File operations** — read, write, edit, delete, move, copy, mkdir, list_dir, exists.
- **Search operations** — glob (pattern matching), grep (regex search with context lines), tree (directory listing with depth limits).
- **Capability protocols** — Backends declare support via `SupportsVersions`, `SupportsTrash`, `SupportsReconcile`, checked at runtime.
- **Dialect-aware SQL** — SQLite, PostgreSQL, and MSSQL.
- **Reconciliation** — Sync disk state with database for `LocalFileSystem`.
- **Sync and async APIs** — `Grover` (sync facade) and `GroverAsync` (async core).
- **Event-driven architecture** — EventBus for internal consistency.
- **Result types** — Structured return types for all operations.

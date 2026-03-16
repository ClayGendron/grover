# Changelog

All notable changes to Grover will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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

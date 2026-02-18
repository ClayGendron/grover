# Changelog

All notable changes to Grover will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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

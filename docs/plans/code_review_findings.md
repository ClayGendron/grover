# Grover Code Review Findings

**Date:** 2026-02-26
**Version:** 0.0.3 (alpha)

## Reviewers

Four independent sub-agents reviewed the Grover codebase, each bringing a distinct domain lens:

1. **Senior AI Software Engineer** -- overall architecture, DX, API design, integration patterns
2. **Senior FastAPI Contributor** -- async patterns, session management, connection pooling, error handling
3. **Senior Filesystem/VFS Designer** -- write ordering, durability, versioning, mount composition, concurrency
4. **Senior LangGraph/Deepagents Contributor** -- integration correctness, tool surface, retriever design, store semantics

All findings are prioritized into four tiers. Tier 1 items have the highest impact-to-effort ratio and should be addressed first. MCP server findings have been excluded per request.

---

## Tier 1 -- Critical Path (Fix Before Beta)

### 1. Decouple Analysis/Indexing from Writes

**Priority:** Tier 1
**Source:** AI Software Engineer, Filesystem Designer

**Problem statement:**
Every `write()` and `edit()` call triggers the full analysis pipeline synchronously inline via the event bus. The write completes, emits a `FILE_WRITTEN` event, and the event handler `_on_file_written` calls `_analyze_and_integrate`, which performs AST parsing, chunk extraction, graph updates, embedding generation, vector upserts, connection DB writes, and search indexing -- all before the `write()` call returns to the user. For a simple file save, this creates significant latency.

**Code references:**

The event is emitted at the end of `write()` in `FileOpsMixin`:

`/Users/claygendron/Git/Repos/grover/src/grover/facade/file_ops.py`, lines 80-88:
```python
if result.success:
    await self._ctx.emit(
        FileEvent(
            event_type=EventType.FILE_WRITTEN,
            path=path,
            content=content,
            user_id=user_id,
        )
    )
```

The event bus dispatches synchronously (sequentially) to all handlers:

`/Users/claygendron/Git/Repos/grover/src/grover/events.py`, lines 80-92:
```python
async def emit(self, event: FileEvent) -> None:
    """Dispatch *event* to all registered handlers for its type."""
    for handler in self._handlers[event.event_type]:
        try:
            await handler(event)
        except Exception:
            logger.warning(
                "Handler %r failed for %s on %s",
                handler,
                event.event_type.value,
                event.path,
                exc_info=True,
            )
```

The handler reads the file (if content not in the event), then runs the full pipeline:

`/Users/claygendron/Git/Repos/grover/src/grover/facade/indexing.py`, lines 31-43:
```python
async def _on_file_written(self, event: FileEvent) -> None:
    if self._ctx.meta_fs is None:
        return
    if "/.grover/" in event.path:
        return
    content = event.content
    if content is None:
        result = await self.read(event.path)  # type: ignore[attr-defined]
        if not result.success:
            return
        content = result.content
    if content is not None:
        await self._analyze_and_integrate(event.path, content, user_id=event.user_id)
```

`_analyze_and_integrate` is the heavyweight:

`/Users/claygendron/Git/Repos/grover/src/grover/facade/indexing.py`, lines 152-273 -- graph removal, node creation, AST analysis, chunk DB writes, connection DB writes with individual sessions per edge, event emission per edge, and search engine indexing.

**Impact:**
- A single `write()` of a Python file with 10 imports takes ~50-200ms for the analysis pipeline (AST parse + graph ops + N+1 DB sessions for connections + embedding), on top of the actual write I/O.
- In batch scenarios (e.g., `index()` calling `_analyze_and_integrate` per file), this compounds to seconds or minutes.
- The user perceives the write as slow even though the file itself saved quickly.
- No ability to prioritize writes over indexing under load.

**Suggested approach:**
- Make the event bus support a "deferred" dispatch mode: queue events and process them in a background `asyncio.Task`.
- Add a configurable `indexing_mode` parameter: `"inline"` (current behavior, useful for tests), `"background"` (default for production), `"manual"` (index only on explicit `index()` calls).
- The background task should batch events and debounce rapid successive writes to the same file.
- Keep `await emit()` non-blocking: enqueue the event and return immediately.

---

### 2. Batch Sessions in `_analyze_and_integrate`

**Priority:** Tier 1
**Source:** FastAPI Contributor, Filesystem Designer

**Problem statement:**
`_analyze_and_integrate` opens a new database session (via `self._ctx.session_for(mount)`) for almost every individual operation: one for `search_engine.remove_file`, one for `replace_file_chunks`, one for `delete_outgoing_connections`, one *per dependency edge* for `add_connection`, and one for `search_engine.add_batch`. For a Python file with 10 imports, this creates 14+ separate session open/commit/close cycles.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/facade/indexing.py`, lines 172-271:

Session 1 -- remove old search entries (line 173):
```python
async with self._ctx.session_for(mount) as sess:
    await search_engine.remove_file(path, session=sess)
```

Session 2 -- replace chunks (line 197):
```python
async with self._ctx.session_for(mount) as sess:
    await mount.filesystem.replace_file_chunks(
        path, chunk_dicts, session=sess, user_id=user_id
    )
```

Session 3 -- delete stale connections (line 225):
```python
async with self._ctx.session_for(mount) as sess:
    await conn_svc.delete_outgoing_connections(sess, path)
```

Sessions 4 through N -- one per dependency edge (lines 233-252):
```python
for edge in dep_edges:
    _w: float = (
        float(edge.metadata.get("weight", 1.0))
        if edge.metadata
        else 1.0
    )
    async with self._ctx.session_for(mount) as sess:
        await mount.filesystem.add_connection(
            edge.source,
            edge.target,
            edge.edge_type,
            weight=_w,
            metadata=dict(edge.metadata) if edge.metadata else None,
            session=sess,
        )
    # Emit event AFTER session commits (post-commit ordering)
    await self._ctx.emit(
        FileEvent(
            event_type=EventType.CONNECTION_ADDED,
            path=f"{edge.source}[{edge.edge_type}]{edge.target}",
            ...
        )
    )
```

Final session -- add search batch (line 264):
```python
async with self._ctx.session_for(mount) as sess:
    await search_engine.add_batch(embeddable, session=sess)
```

Each `session_for` call (defined at `/Users/claygendron/Git/Repos/grover/src/grover/facade/context.py`, lines 53-68) creates a new session, yields it, commits, and closes:
```python
@asynccontextmanager
async def session_for(self, mount: Mount) -> AsyncGenerator[AsyncSession | None]:
    if mount.session_factory is None:
        yield None
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

**Impact:**
- N+1 session anti-pattern: for a file with M dependency edges, creates M+4 separate sessions with M+4 commits.
- Each commit is a round-trip to the database. For PostgreSQL backends, this is a network round-trip per edge.
- SQLite with WAL mode handles this reasonably, but it still creates unnecessary overhead.
- The per-edge session also means a failure on edge 7 of 10 leaves edges 1-6 committed and 7-10 missing -- partial state.

**Suggested approach:**
- Open a single session for the entire `_analyze_and_integrate` call.
- Batch all chunk writes, connection deletes, connection adds, and search operations into that one session.
- Commit once at the end. If any step fails, roll back everything.
- Defer connection events until after the batch commit (collect them in a list, emit after commit).
- This also naturally composes with finding #1 (background indexing) since the entire pipeline becomes a single transactional unit.

---

### 3. Built-in Hybrid Search Fusion

**Priority:** Tier 1
**Source:** AI Software Engineer, LangGraph Contributor

**Problem statement:**
The `hybrid_search` method in `SearchOpsMixin` runs both vector and lexical searches but "fuses" them using the set union operator (`|`), which simply concatenates candidates. There is no score normalization, no Reciprocal Rank Fusion (RRF), and no weighted blending. The `alpha` parameter is accepted but effectively unused -- it does not influence the result ordering.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/facade/search_ops.py`, lines 365-410:
```python
async def hybrid_search(
    self,
    query: str,
    k: int = 10,
    *,
    alpha: float = 0.5,
    path: str = "/",
    user_id: str | None = None,
) -> FileSearchResult:
    """Hybrid search combining vector and lexical results.

    *alpha* controls the blend: 1.0 = pure vector, 0.0 = pure lexical.
    Falls back to whichever is available if only one is configured.
    """
    path = normalize_path(path)

    vec_result: FileSearchResult | None = None
    lex_result: FileSearchResult | None = None

    has_vector = any(
        mount.search is not None
        and mount.search.vector is not None
        and mount.search.embedding is not None
        for mount in self._ctx.registry.list_visible_mounts()
    )
    has_lexical = any(
        mount.search is not None and mount.search.lexical is not None
        for mount in self._ctx.registry.list_visible_mounts()
    )

    if has_vector:
        vec_result = await self.vector_search(query, k=k, path=path, user_id=user_id)
    if has_lexical:
        lex_result = await self.lexical_search(query, k=k, path=path, user_id=user_id)

    if vec_result is not None and lex_result is not None:
        return vec_result | lex_result   # <-- naive union, no fusion
    if vec_result is not None:
        return vec_result
    if lex_result is not None:
        return lex_result

    return FileSearchResult(
        success=False,
        message="Hybrid search not available: no vector or lexical search configured",
    )
```

The `|` operator on `FileSearchResult` (defined in `types/search.py`) merges candidates by path but does not re-rank or normalize scores. The `alpha` parameter on line 370 is dead code.

Additionally, `SearchEngine` itself has a `_hybrid` slot (line 50 of `_engine.py`) that is always `None`:
```python
def __init__(
    self,
    *,
    vector: VectorStore | None = None,
    embedding: EmbeddingProvider | None = None,
    lexical: FullTextStore | None = None,
    hybrid: object | None = None,
) -> None:
    ...
    self._hybrid = hybrid
```

**Impact:**
- The `alpha` parameter is misleading -- users think they can control the vector vs. lexical blend, but it has no effect.
- Results from hybrid search are unranked: lexical results just get appended after vector results rather than being interleaved by relevance.
- State of the art RAG systems expect RRF or weighted fusion. Without it, Grover's hybrid search is worse than running either search independently.

**Suggested approach:**
- Implement Reciprocal Rank Fusion (RRF) as the default fusion strategy. RRF is simple (`1/(k+rank)` per result list, sum scores across lists), works without score normalization, and is the industry standard (used by Elasticsearch, Pinecone, Weaviate).
- Make `alpha` actually weight the contribution: `alpha * vector_rrf_score + (1-alpha) * lexical_rrf_score`.
- Expose the fused score in `VectorEvidence` or a new `HybridEvidence` type so the caller can see it.
- Wire the `SearchEngine._hybrid` slot to a concrete fusion implementation.

---

### 4. Proactive Version Chain Integrity Verification

**Priority:** Tier 1
**Source:** Filesystem Designer

**Problem statement:**
Grover stores file versions as a chain of snapshots (every 20 versions) with forward unified diffs in between. If any diff in the chain is corrupted (truncated, encoding error, database corruption), `reconstruct_version` will produce wrong content for every version after the corruption point. The hash check at the end catches this after the fact, but the user gets a `ConsistencyError` with no way to recover.

There is no proactive mechanism to verify the chain is intact, no way to repair a broken chain, and no alert when corruption is detected during normal operations.

**Code references:**

The diff chain reconstruction in `VersioningService.get_version_content`:

`/Users/claygendron/Git/Repos/grover/src/grover/fs/versioning.py`, lines 110-165:
```python
async def get_version_content(
    self,
    session: AsyncSession,
    file: FileBase,
    version: int,
) -> str | None:
    ...
    # Collect all versions from snapshot through target
    chain_result = await session.execute(
        select(fv_model)
        .where(
            fv_model.file_id == file.id,
            fv_model.version >= snapshot.version,
            fv_model.version <= version,
        )
        .order_by(fv_model.version.asc())
    )
    chain = chain_result.scalars().all()
    ...
    entries = [(v.is_snapshot, v.content) for v in chain]
    content = reconstruct_version(entries)

    # Verify SHA256 against the target version's stored hash
    expected_hash = chain[-1].content_hash
    actual_hash = hashlib.sha256(content.encode()).hexdigest()
    if actual_hash != expected_hash:
        raise ConsistencyError(
            f"Version {version} of file: content hash mismatch "
            f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...)"
        )
```

The `reconstruct_version` function applies diffs sequentially:

`/Users/claygendron/Git/Repos/grover/src/grover/fs/diff.py`, lines 102-121:
```python
def reconstruct_version(snapshots_and_diffs: list[tuple[bool, str]]) -> str:
    if not snapshots_and_diffs:
        return ""
    first_is_snap, content = snapshots_and_diffs[0]
    if not first_is_snap:
        msg = "First entry must be a snapshot"
        raise ValueError(msg)
    result = content
    for is_snap, diff_text in snapshots_and_diffs[1:]:
        result = diff_text if is_snap else apply_diff(result, diff_text)
    return result
```

The snapshot interval is defined at:

`/Users/claygendron/Git/Repos/grover/src/grover/fs/diff.py`, line 14:
```python
SNAPSHOT_INTERVAL: int = 20
```

**Impact:**
- A single corrupted diff (versions 2-19 of a chain) makes all 18 subsequent versions unrecoverable until the next snapshot.
- The `ConsistencyError` is raised at read time, not at write time, so the corruption may go undetected for many writes.
- No self-healing: even though the current file content is correct on disk (for `LocalFileSystem`), the version chain is permanently broken.
- For `DatabaseFileSystem` where content is only in the DB, corruption could mean total data loss for affected versions.

**Suggested approach:**
- Add a `verify_chain()` method to `VersioningService` that walks the chain and validates each version's hash matches after reconstruction.
- Call `verify_chain()` during `reconcile()` to catch corruption proactively.
- When corruption is detected, attempt self-healing by inserting a new snapshot from the current file content, repairing the chain going forward.
- Consider reducing `SNAPSHOT_INTERVAL` from 20 to 10 to limit the blast radius of a corrupted diff.
- Add a `--verify` flag to `index()` that checks all version chains as part of the indexing pass.

---

### 5. Add Lock to `RustworkxGraph`

**Priority:** Tier 1
**Source:** FastAPI Contributor, Filesystem Designer

**Problem statement:**
`RustworkxGraph` has no concurrency protection. The three mutable data structures (`_graph: PyDiGraph`, `_path_to_idx: dict`, `_idx_to_path: dict`) are all modified by `add_node`, `remove_node`, `add_edge`, `remove_edge`, and `remove_file_subgraph` without any locking. When concurrent writes trigger event handlers that modify the graph simultaneously (e.g., two files being written at the same time), the internal state can become inconsistent: `_path_to_idx` and `_idx_to_path` may desynchronize from the underlying `PyDiGraph`.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/graph/_rustworkx.py`, lines 29-33:
```python
def __init__(self) -> None:
    self._graph: rustworkx.PyDiGraph = rustworkx.PyDiGraph()
    self._path_to_idx: dict[str, int] = {}
    self._idx_to_path: dict[int, str] = {}
```

The `add_node` method mutates both dicts and the graph without protection (lines 38-48):
```python
def add_node(self, path: str, **attrs: object) -> None:
    if path in self._path_to_idx:
        idx = self._path_to_idx[path]
        existing: dict[str, Any] = self._graph[idx]
        existing.update(attrs)
    else:
        data: dict[str, Any] = {"path": path, **attrs}
        idx = self._graph.add_node(data)
        self._path_to_idx[path] = idx
        self._idx_to_path[idx] = path
```

The `remove_node` method (lines 50-55) does `del self._path_to_idx[path]` followed by `del self._idx_to_path[idx]`. If another coroutine calls `add_node` between these two deletes, the index maps will be inconsistent.

The `remove_file_subgraph` method (lines 232-266) iterates over the graph, collects children, and removes them one by one. A concurrent `add_edge` during iteration could reference a node index that was just removed.

Note: although Python's GIL protects dict operations from data corruption, asyncio coroutines can interleave between any two `await` points. The event handlers in `IndexMixin._on_file_written` (which call graph methods) are async and can interleave with other event handlers. The real risk is logical inconsistency, not memory corruption.

**Impact:**
- Under concurrent writes, the graph can end up with orphaned edges (pointing to removed nodes), stale entries in `_path_to_idx` for paths that were removed, or duplicate entries.
- The `from_sql` method (line 891) does `self._graph = rustworkx.PyDiGraph()` which is a complete replacement. If a read happens during `from_sql`, the reader sees a partially-loaded graph.
- In the sync `Grover` wrapper, the `_run` method holds an `RLock` which serializes all operations, but in the async `GroverAsync` used by web servers, there is no such protection.

**Suggested approach:**
- Add an `asyncio.Lock` to `RustworkxGraph` and acquire it around all mutating operations.
- For read-heavy workloads, consider `asyncio.Lock` for writes and allowing concurrent reads (since reads are non-mutating). A simple `asyncio.Lock` is sufficient for the alpha phase.
- Alternatively, make all graph mutations go through a single serialized queue (an `asyncio.Queue` consumed by a single task), which also composes well with finding #1 (background indexing).

---

### 6. Native Async Path for All Integrations

**Priority:** Tier 1
**Source:** LangGraph Contributor, FastAPI Contributor

**Problem statement:**
All three integration layers (deepagents backend, deepagents middleware, LangChain retriever/loader/store) use `Grover` (the sync wrapper) rather than `GroverAsync`. This means every async call goes through `asyncio.to_thread(sync_method)`, which itself calls `asyncio.run_coroutine_threadsafe(async_method, private_loop).result()`. This is a sync-over-async-over-sync sandwich that wastes threads, blocks the calling event loop, and prevents true async concurrency.

**Code references:**

deepagents backend -- all async methods are `asyncio.to_thread` wrappers around sync:

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/deepagents/_backend.py`, lines 124-125:
```python
async def als_info(self, path: str) -> list[FileInfo]:
    return await asyncio.to_thread(self.ls_info, path)
```

Lines 160-166:
```python
async def aread(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    return await asyncio.to_thread(self.read, file_path, offset, limit)
```

This pattern repeats for `awrite` (line 187-188), `aedit` (line 225-232), `agrep_raw` (line 271-277), `aglob_info` (line 308-309), `aupload_files` (line 342-343), `adownload_files` (line 371-372).

deepagents middleware -- all tools use sync `Grover` methods directly:

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/deepagents/_middleware.py`, line 93:
```python
result = grover.list_versions(path)
```

No async alternatives are provided for any middleware tools.

LangChain retriever:

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_retriever.py`, lines 76-83:
```python
async def _aget_relevant_documents(
    self,
    query: str,
    *,
    run_manager: "AsyncCallbackManagerForRetrieverRun | None" = None,
) -> list[Document]:
    """Async variant -- delegates to sync via thread executor."""
    return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=None)
```

LangChain store:

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_store.py`, lines 75-76:
```python
async def abatch(self, ops: Iterable[Op]) -> list[Result]:
    return await asyncio.to_thread(self.batch, list(ops))
```

The call chain for an async operation is:
1. Integration calls `asyncio.to_thread(self.sync_method, ...)`
2. `sync_method` calls `self.grover.some_method(...)` (sync `Grover`)
3. `Grover._run()` calls `asyncio.run_coroutine_threadsafe(coro, self._loop).result()` (blocks thread)
4. The private event loop runs the actual async operation

So every async call occupies a thread pool thread that is entirely blocked waiting on another event loop.

**Impact:**
- Thread pool exhaustion: in a web server handling 50 concurrent requests, each async call blocks a thread. Default `asyncio.to_thread` thread pool is 40 threads -- trivially exhaustible.
- Latency overhead: at least 2 thread context switches per operation (main loop -> thread pool -> private loop -> thread pool -> main loop).
- The integrations cannot benefit from true async I/O (connection multiplexing, pipelining, etc.).
- In LangGraph's async graph execution, this becomes the bottleneck since LangGraph expects truly async tools.

**Suggested approach:**
- Accept `GroverAsync` as an alternative to `Grover` in all integration constructors.
- When `GroverAsync` is provided, call its methods directly (they are already async).
- When `Grover` is provided, fall back to the current `asyncio.to_thread` pattern for backward compatibility.
- This is a non-breaking change: existing users with `Grover` continue to work, new users in async contexts pass `GroverAsync` directly.

---

## Tier 2 -- Important (Pre-Beta Polish)

### 7. Reduce Public API Surface

**Priority:** Tier 2
**Source:** AI Software Engineer

**Problem statement:**
The package `__init__.py` exports 58 names at version 0.0.3. Most users need `Grover`, `GroverAsync`, `Mount`, `Ref`, and a handful of result types. Filter functions (`eq`, `gt`, `gte`, `lt`, `lte`, `ne`, `in_`, `not_in`, `exists`, `and_`, `or_`), store-level types (`VectorEntry`, `IndexConfig`, `IndexInfo`, `UpsertResult`, `SearchDeleteResult`), and protocol types (`SupportsHybridSearch`, `SupportsIndexLifecycle`, `SupportsMetadataFilter`, etc.) should not be top-level exports.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/__init__.py`, lines 86-158 -- `__all__` with 58 entries.

**Impact:**
- Overwhelming for new users: `from grover import ` tab-completion shows 58 choices.
- Increases coupling: downstream code importing internal types from the top-level package makes refactoring harder.
- Documentation burden: every export needs docstrings and version stability guarantees.

**Suggested approach:**
- Keep only the top ~15 exports in `__init__.py`: `Grover`, `GroverAsync`, `Mount`, `Ref`, `file_ref`, `SearchEngine`, the main result types (`ReadResult`, `WriteResult`, `EditResult`, `DeleteResult`, `FileSearchResult`), `EmbeddingProvider`, `VectorStore`, `GraphStore`.
- Move filter functions to `grover.search.filters` (already there, just remove from top-level re-export).
- Move protocol types to `grover.search.protocols` (already there).
- Move store-level types to `grover.search.types` (already there).

---

### 8. Simplify `add_mount` / Add Factory Methods

**Priority:** Tier 2
**Source:** AI Software Engineer

**Problem statement:**
`add_mount` accepts 14+ parameters with complex conditional logic. The common case (mount a local directory) requires understanding `LocalFileSystem`, `AsyncEngine`, session factories, and model types. There are no convenience factory methods on `Grover` or `GroverAsync` for the 80% use case.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/facade/mounting.py`, lines 50-69 -- `add_mount` signature with 18 parameters.

`/Users/claygendron/Git/Repos/grover/src/grover/_grover.py`, lines 116-150 -- sync `add_mount` mirrors all parameters.

**Impact:**
- New users face a steep learning curve just to mount a directory.
- Easy to misconfigure: providing both `engine` and `session_factory` raises, model type mismatches are detected late.

**Suggested approach:**
- Add `Grover.from_local(workspace_dir, data_dir=None)` factory that creates a `Grover` with a `LocalFileSystem` mounted at `/`.
- Add `Grover.from_database(engine)` factory that creates a `Grover` with a `DatabaseFileSystem`.
- Add `Grover.from_config(dict | Path)` for YAML/JSON-driven setup (Tier 4 item, but the factory pattern should be established now).

---

### 9. Expand Middleware Tool Surface

**Priority:** Tier 2
**Source:** LangGraph Contributor

**Problem statement:**
`GroverMiddleware` exposes 10 tools (list_versions, get_version_content, restore_version, delete_file, list_trash, restore_from_trash, search_semantic, dependencies, dependents, impacts). Missing: `tree` (directory overview), `move`/`copy` (file management), `glob` (pattern search), `grep` (text search), `lexical_search` (BM25), `hybrid_search`, `index` (re-index). An agent using only `GroverMiddleware` cannot navigate the filesystem without the backend's `ls_info`.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/deepagents/_middleware.py`, lines 62-80 -- tool list construction:
```python
tool_list: list[BaseTool] = [
    self._create_list_versions_tool(),
    self._create_get_version_content_tool(),
    self._create_restore_version_tool(),
    self._create_delete_file_tool(),
    self._create_list_trash_tool(),
    self._create_restore_from_trash_tool(),
]
if enable_search:
    tool_list.append(self._create_search_semantic_tool())
if enable_graph:
    tool_list.extend([
        self._create_dependencies_tool(),
        self._create_dependents_tool(),
        self._create_impacts_tool(),
    ])
```

**Impact:**
- Agents using `GroverMiddleware` have no filesystem navigation tools (tree, glob).
- No text search tools (grep, lexical_search) -- only semantic search.
- Cannot move or copy files, only delete them.
- Forces agents to use the backend's lower-level `ls_info`/`read`/`write` for basic operations that Grover could provide with richer context.

**Suggested approach:**
- Add `tree`, `glob`, `grep` tools to the middleware (gated by an `enable_navigation` flag).
- Add `move`, `copy` tools (gated by `enable_file_management`).
- Add `lexical_search` and `hybrid_search` tools (gated by the existing `enable_search` flag).

---

### 10. Add Hybrid/BM25 Search + Score Exposure to `GroverRetriever`

**Priority:** Tier 2
**Source:** LangGraph Contributor

**Problem statement:**
`GroverRetriever` only supports vector search. It does not expose BM25/lexical search, hybrid search, or similarity scores. LangChain's retriever ecosystem expects scores for reranking and filtering.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_retriever.py`, lines 55-74:
```python
def _get_relevant_documents(
    self,
    query: str,
    *,
    run_manager: "CallbackManagerForRetrieverRun | None" = None,
) -> list[Document]:
    try:
        result = self.grover.vector_search(query, k=self.k)
    except Exception:
        return []
    if not result.success:
        return []
    return [self._path_to_document(path, result) for path in result.paths]
```

No score information is included in the returned `Document.metadata` (lines 86-101):
```python
@staticmethod
def _path_to_document(path: str, result: "VectorSearchResult") -> Document:
    metadata: dict[str, object] = {
        "path": path,
    }
    snippets = list(result.snippets(path))
    if snippets:
        metadata["chunks"] = len(snippets)
    page_content = "\n\n".join(snippets) if snippets else path
    return Document(
        page_content=page_content,
        metadata=metadata,
        id=path,
    )
```

**Impact:**
- No score metadata means downstream rerankers and threshold filters cannot work.
- Users who need BM25 search must bypass the retriever and call Grover directly.
- Hybrid search (the best quality option) is completely inaccessible through the LangChain integration.

**Suggested approach:**
- Add `search_type` parameter to `GroverRetriever`: `"vector"` (default), `"lexical"`, `"hybrid"`.
- Include `score` in `Document.metadata` for all search types.
- For hybrid, include both `vector_score` and `lexical_score` in metadata.

---

### 11. Fix Error Codes and Silent Swallowing in Integrations

**Priority:** Tier 2
**Source:** LangGraph Contributor, FastAPI Contributor

**Problem statement:**
(a) `GroverBackend.upload_files` uses wrong error codes: `UnicodeDecodeError` returns `"invalid_path"` (should be `"invalid_content"` or similar), and write failures return `"permission_denied"` regardless of the actual error.

(b) `GroverRetriever._get_relevant_documents` catches all exceptions and returns an empty list silently. The user gets no indication that search failed.

(c) `GroverBackend.ls_info` catches all exceptions and returns an empty list.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/deepagents/_backend.py`, lines 323-339:
```python
try:
    content = data.decode("utf-8")
except UnicodeDecodeError:
    responses.append(FileUploadResponse(path=file_path, error="invalid_path"))
    continue
try:
    result = self.grover.write(file_path, content, overwrite=False)
except Exception:
    responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
    continue
```

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_retriever.py`, lines 66-69:
```python
try:
    result = self.grover.vector_search(query, k=self.k)
except Exception:
    return []
```

**Impact:**
- Misleading error codes break integration contract expectations: deepagents may retry a "permission_denied" error when the real issue is binary content.
- Silent failures make debugging extremely difficult in production.

**Suggested approach:**
- Map `UnicodeDecodeError` to a `"binary_content"` or `"encoding_error"` code.
- Map write failures to the actual error type (check for `PermissionError`, `MountNotFoundError`, etc.).
- In `GroverRetriever`, log the exception and optionally raise it based on a configurable `raise_on_error` flag.

---

### 12. Missing Parent Directory `fsync`

**Priority:** Tier 2
**Source:** Filesystem Designer

**Problem statement:**
`LocalFileSystem._write_content` uses atomic write (`tmpfile` -> `fsync` -> `rename`), which is correct. However, it does not `fsync` the parent directory after the rename. On Linux (ext4, XFS), this means a power failure after the rename but before the directory's metadata is flushed can result in the file entry being lost.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/fs/local_fs.py`, lines 306-328:
```python
async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
    actual_path = await self._resolve_path(path)

    def _do_write() -> None:
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=actual_path.parent,
            prefix=".tmp_",
            suffix=actual_path.suffix,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            Path(tmp_path).replace(actual_path)
        except Exception:
            tmp = Path(tmp_path)
            if tmp.exists():
                tmp.unlink()
            raise

    await asyncio.to_thread(_do_write)
```

Line 320 fsyncs the file. Line 321 renames. But the parent directory is never fsynced.

**Impact:**
- On power failure after rename but before directory metadata is flushed, the file entry can be lost. The DB would say the file exists, but the disk file would be missing.
- macOS (APFS) handles this differently and is less susceptible, but Linux deployments are at risk.

**Suggested approach:**
- After `Path(tmp_path).replace(actual_path)`, add:
  ```python
  dir_fd = os.open(str(actual_path.parent), os.O_RDONLY)
  try:
      os.fsync(dir_fd)
  finally:
      os.close(dir_fd)
  ```

---

### 13. Connection Pool Configuration

**Priority:** Tier 2
**Source:** FastAPI Contributor

**Problem statement:**
`LocalFileSystem._ensure_db` creates an `AsyncEngine` with default connection pool settings (`pool_size=5`, `max_overflow=10` for non-SQLite). For SQLite with `aiosqlite`, the default is `StaticPool` (single connection). There is no way for users to configure pool size, overflow, recycling, or timeout.

For `DatabaseFileSystem` mounts, the engine is passed in by the user but the `_create_engine_mount` helper in `MountMixin` does not validate or document pool configuration requirements.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/fs/local_fs.py`, lines 192-195:
```python
self._engine = create_async_engine(
    f"sqlite+aiosqlite:///{db_path}",
    echo=False,
)
```

No pool configuration parameters. SQLAlchemy defaults apply.

**Impact:**
- For PostgreSQL/MSSQL backends in web apps, the default pool size of 5 can be exhausted quickly under concurrent load.
- No connection recycling means stale connections can accumulate.
- No documentation tells users what pool settings to use.

**Suggested approach:**
- Accept optional `pool_size`, `max_overflow`, `pool_recycle` parameters in `LocalFileSystem.__init__` and pass them through to `create_async_engine`.
- Document recommended pool settings for each database backend in `docs/api.md`.
- Consider adding a `create_engine` helper that sets sensible defaults per dialect.

---

## Tier 3 -- Quality of Life

### 14. `GroverStore` Correctness Fixes

**Priority:** Tier 3
**Source:** LangGraph Contributor

**Problem statement:**
Three correctness issues in `GroverStore`:

(a) **Wrong timestamps**: `_handle_get` returns `datetime.now(UTC)` as both `created_at` and `updated_at` instead of the file's actual timestamps (lines 93-99).

(b) **Permanent deletes**: `_handle_put` with `value=None` calls `self.grover.delete(path, permanent=True)` (line 107). LangGraph expects soft-delete semantics. Permanent delete destroys version history.

(c) **Unscoped search**: `_handle_search` calls `self.grover.vector_search(op.query, ...)` which searches the entire Grover index, not just the store's namespace prefix (lines 119-121).

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_store.py`, lines 93-99, 105-107, 119-121.

**Impact:**
- Wrong timestamps break LangGraph's item freshness logic.
- Permanent deletes are surprising and destructive.
- Unscoped search returns results from outside the store's namespace.

**Suggested approach:**
- Use `get_info` to fetch the file's actual `created_at`/`updated_at`.
- Use `delete(path, permanent=False)` (soft-delete).
- Pass `path=self.prefix` to `vector_search` to scope results to the store's namespace.

---

### 15. Chunk-Level Loading in `GroverLoader`

**Priority:** Tier 3
**Source:** LangGraph Contributor

**Problem statement:**
`GroverLoader` yields one `Document` per file. It does not support chunk-level loading, which is Grover's key differentiator. RAG pipelines get better retrieval quality with function/class-level documents than whole-file documents.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/integrations/langchain/_loader.py`, lines 59-101 -- `lazy_load` yields one `Document` per file, no chunk support.

**Impact:**
- Users must implement their own chunking on top of `GroverLoader`, missing the opportunity to use Grover's AST-aware chunk extraction.

**Suggested approach:**
- Add a `chunk_level: bool = False` parameter. When True, yield one `Document` per chunk (function, class) with metadata including `line_start`, `line_end`, `chunk_name`, `parent_path`.

---

### 16. Cross-Mount Move Atomicity

**Priority:** Tier 3
**Source:** Filesystem Designer

**Problem statement:**
Cross-mount moves in `FileOpsMixin.move` use a read-write-delete sequence with separate sessions. If the write succeeds but the delete fails, the file exists in both mounts. The error message says "Copied but failed to delete source" but the caller may not realize data duplication occurred.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/facade/file_ops.py`, lines 304-347:
```python
# Cross-mount move: read -> write -> delete (non-atomic)
async with self._ctx.session_for(src_mount) as src_sess:
    read_result = await src_mount.filesystem.read(...)
...
async with self._ctx.session_for(dest_mount) as dest_sess:
    write_result = await dest_mount.filesystem.write(...)
...
async with self._ctx.session_for(src_mount) as src_sess:
    delete_result = await src_mount.filesystem.delete(...)
if not delete_result.success:
    return MoveResult(
        success=False,
        message=f"Copied but failed to delete source: {delete_result.message}",
    )
```

**Impact:**
- Data duplication on partial failure.
- No compensation logic to clean up the destination if the delete fails.

**Suggested approach:**
- Return `success=True` with a warning in `message` when copy succeeds but delete fails (the user's intent was to move, and the data is at the destination).
- Add a `moved_but_source_retained` flag to `MoveResult` so callers can detect this case.
- Document the non-atomic nature of cross-mount moves in `docs/api.md`.

---

### 17. Concurrent Write Handling

**Priority:** Tier 3
**Source:** FastAPI Contributor

**Problem statement:**
When two concurrent writes create the same file path, both will attempt to `session.add(new_file)` with the same `path`. The second write will hit a `UniqueConstraint` violation on the `path` column. The error propagates as a raw `IntegrityError` with a cryptic SQLAlchemy message.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/fs/operations.py`, lines 233-258 -- the `else` branch (new file creation):
```python
new_file = file_model(
    path=path,
    name=name,
    parent_path=split_path(path)[0],
    ...
)
session.add(new_file)
```

No retry or conflict detection logic.

**Impact:**
- Users get a cryptic `IntegrityError` instead of a clear "file already exists" message.
- The VFS session is rolled back, but no cleanup occurs.

**Suggested approach:**
- Catch `IntegrityError` on flush and retry with the `existing` branch (treat as overwrite if `overwrite=True`).
- Or use a `SELECT ... FOR UPDATE` pattern to lock the path row before the check.

---

### 18. `LocalFileSystem.read()` Orchestration Asymmetry

**Priority:** Tier 3
**Source:** Filesystem Designer

**Problem statement:**
`LocalFileSystem.read()` is implemented directly on the class (58 lines of custom logic: path resolution, binary check, similar-file suggestions) rather than delegating to the shared `read_file` orchestration function in `operations.py`. By contrast, `write`, `edit`, `delete`, and `move` all delegate to the shared orchestration functions.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/fs/local_fs.py`, lines 358-407 -- custom `read` implementation.

Compare with `write` (lines 409-434) which delegates to `write_file` from `operations.py`.

The shared `read_file` function at `/Users/claygendron/Git/Repos/grover/src/grover/fs/operations.py`, lines 109-143 is not used by `LocalFileSystem`.

**Impact:**
- If `read_file` in `operations.py` gains new behavior (e.g., external edit detection on read, caching, permission checks), `LocalFileSystem.read()` will not benefit.
- Two code paths to maintain for the same operation.
- The `LocalFileSystem.read()` skips the trash path check that `read_file` has (line 125-126 of operations.py).

**Suggested approach:**
- Refactor `LocalFileSystem.read()` to delegate to `read_file` from `operations.py`, extending it with the local-specific checks (binary detection, similar-file suggestions) as pre/post hooks.

---

### 19. Observability

**Priority:** Tier 3
**Source:** AI Software Engineer, FastAPI Contributor

**Problem statement:**
No structured logging, no metrics, no tracing. The codebase uses basic `logging.getLogger(__name__)` with unstructured messages. There are no timing metrics for write/read operations, no counters for events, no spans for the analysis pipeline.

**Code references:**
Logging is used throughout (e.g., `indexing.py` line 19, `events.py` line 13, `local_fs.py` line 87) but only for debug and warning messages with no structured fields.

**Impact:**
- In production, diagnosing performance issues requires guesswork.
- No ability to monitor graph size, search index size, event queue depth, or session pool utilization.

**Suggested approach:**
- Add optional OpenTelemetry integration: instrument `write`, `read`, `_analyze_and_integrate`, `search`, and `emit` with spans.
- Add lightweight counters (no dependency required): `_ctx.metrics.writes_total`, `_ctx.metrics.events_emitted`, `_ctx.metrics.analysis_duration_ms`.
- Use `structlog` or equivalent for structured log fields (path, duration, version, user_id).

---

### 20. `GroverToolkit` Bundle Class

**Priority:** Tier 3
**Source:** LangGraph Contributor

**Problem statement:**
`GroverBackend` and `GroverMiddleware` must be configured separately and their tools combined manually. There is no single `GroverToolkit` class that bundles all available tools (file ops from the backend + version/search/graph tools from the middleware) into a single list, which is the standard LangChain pattern.

**Code references:**
No `GroverToolkit` class exists. Users must manually combine:
```python
backend = GroverBackend(g)
middleware = GroverMiddleware(g)
all_tools = backend_tools + middleware.tools  # manual assembly
```

**Impact:**
- More boilerplate for users.
- No single entry point for "give me all Grover tools".

**Suggested approach:**
- Create `GroverToolkit(grover, *, enable_search=True, enable_graph=True, enable_versions=True)` that internally creates both backend and middleware and exposes a unified `tools` list.

---

### 21. Version Pruning / Retention Policy

**Priority:** Tier 3
**Source:** Filesystem Designer

**Problem statement:**
Version records grow without bound. There is no retention policy, no pruning command, and no way to limit the number of versions per file. A file edited 1000 times has 1000 version records (50 snapshots + 950 diffs).

**Code references:**
`/Users/claygendron/Git/Repos/grover/src/grover/fs/versioning.py` -- no retention/pruning methods exist.

The snapshot interval is 20 (`/Users/claygendron/Git/Repos/grover/src/grover/fs/diff.py`, line 14), meaning every 20 versions there is a full snapshot.

**Impact:**
- Database size grows linearly with edit count.
- No way for users to set "keep last N versions" or "keep versions from the last 30 days".

**Suggested approach:**
- Add `prune_versions(file_path, keep_latest=N)` to `VersioningService`.
- Add `retention_policy` to `Mount` configuration: `max_versions`, `max_age_days`.
- Pruning should preserve at least one snapshot and all versions after it.

---

### 22. Graph Memory Management

**Priority:** Tier 3
**Source:** Filesystem Designer

**Problem statement:**
`RustworkxGraph` keeps all nodes and edges in memory. Each node is a Python dict with `path`, `parent_path`, and other attributes. Each edge is a dict with `id`, `source`, `target`, `type`, `weight`, `metadata`. For a 100K-file codebase, this can consume approximately 200MB of memory.

**Code references:**

`/Users/claygendron/Git/Repos/grover/src/grover/graph/_rustworkx.py`, lines 29-33 -- all in-memory, no eviction.

**Impact:**
- Memory usage scales linearly with codebase size.
- No eviction strategy: once loaded, nodes and edges stay in memory forever.
- For web applications with many mounted codebases, memory can become a constraint.

**Suggested approach:**
- Add lazy loading: only load the graph from DB when first accessed.
- Add LRU eviction for mount-level graphs: unmount the graph for inactive mounts.
- Document memory requirements in `docs/api.md` (approximate MB per 10K files).

---

### 23. Simpler Result View Layer

**Priority:** Tier 3
**Source:** AI Software Engineer

**Problem statement:**
Result types (`FileSearchResult`, `GrepResult`, etc.) use a candidates/evidence model that is powerful but verbose for the common case. Simple operations like "get me the list of file paths" require `result.paths`, but getting matched line numbers from a grep requires navigating `result.candidates -> evidence -> line_matches`. There is no `result.as_dicts()` helper for the 80% case.

**Code references:**
Result types are defined in `/Users/claygendron/Git/Repos/grover/src/grover/types/search.py` and `/Users/claygendron/Git/Repos/grover/src/grover/types/operations.py`.

**Impact:**
- Integration code (like the middleware tools) has to write verbose evidence-unpacking loops.
- JSON serialization for API responses requires manual conversion.

**Suggested approach:**
- Add `result.as_dicts()` -> `list[dict]` that returns a simplified view.
- Add `result.to_json()` -> `str` for direct serialization.
- Keep the full candidates/evidence model for advanced use cases.

---

## Tier 4 -- Roadmap (Future Releases)

### 24. Git-Aware Versioning

**Priority:** Tier 4
**Source:** Filesystem Designer

**Problem statement:**
`LocalFileSystem` maintains its own version chain in SQLite, completely independent of git. For codebases that use git, this means version history is duplicated and the two systems can diverge. There is no way to map a Grover version to a git commit or vice versa.

**Code references:**
`/Users/claygendron/Git/Repos/grover/src/grover/fs/versioning.py` -- no git awareness.

**Impact:**
- Users who use git (most users) get no benefit from Grover's versioning beyond fine-grained undo.
- The external edit detection (`check_external_edit` in `operations.py`, lines 58-106) inserts synthetic versions for git-initiated changes, but does not record the git commit hash.

**Suggested approach:**
- Add optional `git_commit_hash` field to `FileVersion` model.
- On write, if the workspace is a git repo, record the current HEAD commit hash.
- Add `versions_since_commit(commit_hash)` query.
- Consider a `GitAwareVersioningService` that delegates to git for old versions and Grover for recent ones.

---

### 25. Additional Language Analyzers

**Priority:** Tier 4
**Source:** AI Software Engineer

**Problem statement:**
The analyzer registry supports Python (stdlib `ast`), JavaScript/TypeScript (tree-sitter), and Go (tree-sitter). Rust, Java, C#, Ruby, and other popular languages are not supported. Without an analyzer, files in these languages are indexed as whole files rather than at the function/class chunk level.

**Code references:**
`/Users/claygendron/Git/Repos/grover/src/grover/graph/analyzers/` -- only `python.py`, `javascript.py`, `go.py`.

**Impact:**
- Codebases in unsupported languages get lower-quality graph and search results.
- The analyzer protocol (`_base.py`) is well-defined, so adding new analyzers is straightforward.

**Suggested approach:**
- Add Rust and Java analyzers using tree-sitter (the infrastructure is already set up for tree-sitter in the JS/TS/Go analyzers).
- Community contributions can add more languages following the same pattern.

---

### 26. `Grover.from_config()` / `from_env()`

**Priority:** Tier 4
**Source:** AI Software Engineer

**Problem statement:**
All Grover configuration is done programmatically. There is no support for configuration files (YAML, JSON, TOML) or environment variables. This makes deployment harder (requires code changes for different environments) and prevents declarative setup.

**Code references:**
`/Users/claygendron/Git/Repos/grover/src/grover/_grover.py`, lines 65-84 -- constructor takes Python objects only.

**Impact:**
- Harder to configure in containerized deployments.
- No way to share configuration across team members without code changes.

**Suggested approach:**
- Add `Grover.from_config(path_or_dict)` that reads a config file and constructs the instance.
- Add `Grover.from_env()` that reads `GROVER_DATA_DIR`, `GROVER_EMBEDDING_PROVIDER`, `GROVER_DATABASE_URL`, etc.
- Keep the programmatic API as the primary interface; config is a convenience layer.

---

### 27. Streaming/Reactive API

**Priority:** Tier 4
**Source:** AI Software Engineer

**Problem statement:**
There is no way to watch for file changes in real time. The event bus is internal-only. For use cases like live dashboards, IDE plugins, or multi-agent coordination, a streaming API is needed.

**Code references:**
`/Users/claygendron/Git/Repos/grover/src/grover/events.py` -- `EventBus` is internal, handlers are registered programmatically.

**Impact:**
- Multi-agent systems cannot coordinate through Grover's event system.
- IDE integrations cannot react to file changes in real time.

**Suggested approach:**
- Add `async for event in grover.watch(path="/", event_types=...)` that yields `FileEvent` objects.
- Implement using an internal `asyncio.Queue` that the event bus pushes to.
- Add an optional WebSocket or SSE endpoint for non-Python consumers.

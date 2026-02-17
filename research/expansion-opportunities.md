# Expansion Opportunities

Detailed analysis of each identified opportunity for expanding Grover's feature set, organized by priority.

---

## High Priority

### 1. deepagents BackendProtocol Implementation

**What:** Implement Grover as a `BackendProtocol` for LangChain's deepagents framework, allowing any LangGraph deep agent to use Grover as its file storage backend.

**Why:** deepagents is the most directly comparable framework, and its `BackendProtocol` maps cleanly to Grover's existing operations. This is the fastest path to getting Grover into production agent systems.

**Interface mapping:**

| deepagents method | Grover equivalent |
|-------------------|-------------------|
| `ls_info(path)` | `list_dir(path)` |
| `read(path, offset, limit)` | `read(path, offset, limit)` |
| `write(path, content)` | `write(path, content)` |
| `edit(path, old, new, replace_all)` | `edit(path, old, new)` |
| `grep_raw(pattern, path, glob)` | `grep(pattern, path, glob_filter=glob, fixed_string=True)` |
| `glob_info(pattern, path)` | `glob(pattern, path)` |

**Additional value:** Grover-specific tools exposed as custom middleware:
- `search_semantic(query, k)` — vector similarity search
- `list_versions(path)` / `restore_version(path, v)` — version management
- `dependencies(path)` / `dependents(path)` / `impacts(path)` — graph queries
- `list_trash()` / `restore_from_trash(path)` — soft-delete management

**Effort:** Medium. Protocol adapter + middleware. No changes to Grover core.

**Dependencies:** deepagents as optional dependency.

---

### 2. MCP Server

**What:** Build a Model Context Protocol server that exposes Grover's versioned filesystem, semantic search, and graph queries as MCP tools.

**Why:** MCP is the emerging standard for AI tool access. A Grover MCP server would make Grover accessible to Claude Desktop, Cursor, VS Code, and any MCP-compatible client — instantly. No competing MCP server offers versioning, semantic search, or graph queries.

**Tools to expose:**

| Tool | Description |
|------|-------------|
| `read_file` | Read with pagination and version support |
| `write_file` | Write with automatic versioning |
| `edit_file` | Find-and-replace with versioning |
| `delete_file` | Soft-delete with trash |
| `list_directory` | Directory listing with metadata |
| `search_files` | Glob pattern matching |
| `search_content` | Regex content search (grep) |
| `search_semantic` | Vector similarity search |
| `file_history` | List versions of a file |
| `restore_version` | Rollback to previous version |
| `diff_versions` | Compare two versions |
| `dependencies` | What does this file depend on? |
| `dependents` | What depends on this file? |
| `impacts` | What's affected if this changes? |
| `restore_from_trash` | Recover deleted files |

**Effort:** Medium-high. Requires MCP protocol implementation, server lifecycle management.

**Dependencies:** MCP SDK (Python or Node.js bridge).

---

### 3. Agent Memory Layer

**What:** Position Grover as the storage backend for agent memory systems, compatible with Letta's context repository model.

**Why:** Letta's MemFS demonstrates strong demand for file-based agent memory with versioning. Grover already has everything MemFS provides (versioned files, directory hierarchy) plus things it lacks (semantic search over memory, dependency graph of memory relationships, database-backed persistence, multi-user sharing).

**Approach:**
- Define memory file conventions compatible with Letta's MemFS format (markdown + frontmatter)
- Support the three-tier memory model: History (immutable), Memory (persistent/mutable), Scratchpad (ephemeral)
- Use Grover's VFS mounts to map tiers to different backends (e.g., scratchpad -> ephemeral, memory -> persistent DB)
- Search over memory files via Grover's semantic search
- Track memory relationships via the knowledge graph (e.g., "this memory references that conversation")

**Effort:** Medium. Mostly conventions and documentation, plus optional helper utilities.

**Dependencies:** None for core. Letta compatibility requires studying their `.af` format.

---

## Medium Priority

### 4. Content-Addressable Storage (CAS)

**What:** Add an optional CAS layer where file content is stored by SHA-256 hash in a dedicated blob table. Version records reference content by hash instead of storing inline content.

**Why:**
- **Deduplication** — agents frequently revert files to previous states; CAS stores identical content once
- **Integrity verification** — content hashes provide free corruption detection
- **Efficient reverts** — `restore_version` becomes "point to old hash" (no content copy)
- **Foundation for branching** — CAS makes metadata-only snapshots possible (branch = `{path -> hash}` mapping)

**Schema addition:**
```
grover_blobs:
  - hash (PK, SHA-256)
  - content (TEXT or BYTEA)
  - size_bytes
  - created_at
```

`grover_file_versions.content` would store either:
- Full content (if CAS disabled, backward compatible)
- Content hash reference (if CAS enabled)

**Effort:** Medium. New table, migration, adapter layer in VersioningService.

**Dependencies:** None. Internal optimization.

---

### 5. Zero-Copy Branching

**What:** Add branch support where branches are named references to a snapshot of the current version state. Writes on a branch use copy-on-write semantics.

**Why:** Enables critical agent workflows:
- **Speculative edits** — agent tries an approach on a branch, merges or discards
- **Agent sandboxing** — give an agent a branch; if it messes up, discard the branch
- **Parallel exploration** — multiple agents work on branches simultaneously, merge results
- **Checkpointing** — save a named point in time, continue working, revert if needed

**Design sketch:**
```
grover_branches:
  - name (PK)
  - parent_branch (FK, nullable)
  - created_at
  - base_snapshot_id (FK to grover_snapshots)

grover_snapshots:
  - id (PK)
  - created_at
  - metadata_json

grover_snapshot_entries:
  - snapshot_id (FK)
  - path
  - version
  - content_hash (if CAS enabled)
```

Branch creation = create snapshot entry + branch record (O(n) files, but metadata only).
Write on branch = copy-on-write (create branch-specific version).
Merge = three-way diff between ancestor, source, dest.

**Effort:** High. New tables, branching logic in VFS, merge algorithm.

**Dependencies:** Strongly benefits from CAS (#4). Could work without it but less efficient.

---

### 6. Enhanced Semantic Code Search

**What:** Improve search accuracy by generating natural language descriptions of code chunks before embedding, following Greptile's research findings.

**Why:** Greptile's research demonstrates:
- Raw code embeddings produce poor search results
- **Code-to-NL translation before embedding yields 12% better accuracy**
- Function-level chunking (which Grover already does) outperforms file-level
- Combining semantic search with structural understanding (which Grover has via graph) is the winning approach

**Approach:**
1. Add a `CodeDescriber` protocol with method `describe(code: str, language: str) -> str`
2. Default implementation: template-based (extract function name, params, docstring, return type -> NL sentence)
3. Optional LLM-based implementation: send code to LLM, get NL description
4. In the extractor pipeline: for code chunks, generate NL description and embed that instead of (or in addition to) raw code
5. Store both embeddings (code + NL description) for hybrid retrieval

**Effort:** Medium. New protocol, extractor enhancement, dual embedding storage.

**Dependencies:** Optional LLM API for high-quality descriptions.

---

### 7. Sandbox Integration Adapters

**What:** Lightweight adapters for running Grover inside E2B, Modal, and Fly.io Sprites sandboxes.

**Why:** No sandbox currently provides versioned file access, semantic search, or graph queries. Grover running inside a sandbox fills all these gaps.

**Approach:**
- **E2B adapter**: `DatabaseFileSystem` connecting to external PostgreSQL. Agent workspace persists across ephemeral sandbox sessions.
- **Fly.io Sprites adapter**: `LocalFileSystem` using the Sprite's persistent storage. Grover adds versioning/search to the already-persistent FS.
- **Generic Docker adapter**: Dockerfile + entrypoint that initializes Grover with configurable backend.

**Effort:** Low per adapter (mostly configuration + documentation).

**Dependencies:** Access to sandbox platforms for testing.

---

## Long Term

### 8. AFS Reference Implementation

**What:** Align Grover with the "Everything is Context" paper's formal model, positioning it as the reference implementation of the Agentic File System concept.

**Why:** The AFS paper validates Grover's core philosophy and provides a theoretical framework. Being the reference implementation gives Grover academic credibility and positions it as the canonical tool in this emerging space.

**Alignment work:**
- Map Grover's three-tier storage (mount types) to AFS's History/Memory/Scratchpad tiers
- Add transaction logging (already have event bus; formalize into audit log)
- Add metadata governance (file metadata policies, access tracking)
- Publish alignment document showing how Grover implements each AFS principle

**Effort:** Medium. Mostly design alignment and documentation, some new features.

**Dependencies:** None.

---

### 9. Multi-Agent Collaboration

**What:** Extend `UserScopedFileSystem` for agent-to-agent file sharing with controlled permissions and version visibility.

**Why:** Multi-agent systems are growing (CrewAI, AutoGen, LangGraph multi-agent). Agents need to share work products, and Grover's existing sharing infrastructure is designed for users but works for agents too.

**Enhancements:**
- Agent identity model (agents as first-class users with `agent_id`)
- Workspace sharing between agents (agent A shares `/analysis/` with agent B)
- Shared version history (both agents see the version chain)
- Conflict resolution for concurrent writes
- Event notifications across agents ("agent A modified a file you're watching")

**Effort:** Medium-high. Identity model, concurrent write handling, cross-agent events.

**Dependencies:** Builds on existing UserScopedFileSystem and sharing infrastructure.

---

### 10. Cloud-Native Backend (S3/GCS)

**What:** Add S3/GCS object storage as a Grover storage backend, separating data (object store) from metadata (database).

**Why:** Enables enterprise scale. Object storage is cheap, durable, and globally distributed. Metadata stays in PostgreSQL for fast queries. Pattern proven by JuiceFS, lakeFS.

**Architecture:**
- Metadata + versions in PostgreSQL
- File content in S3/GCS, keyed by content hash (CAS)
- On read: look up content hash in DB, fetch from object store
- On write: store content in object store, record hash in DB
- Caching layer for recently accessed content

**Effort:** High. New backend implementation, caching layer, deployment complexity.

**Dependencies:** CAS (#4) strongly recommended. AWS/GCP SDK dependencies.

---

### 11. FUSE Mount

**What:** Expose Grover mounts as FUSE filesystems, allowing agents and users to use standard Unix tools against Grover-backed files.

**Why:** AgentFS demonstrates demand. FUSE access means `git`, `grep`, `find`, editors, and any other tool works against Grover files without modification.

**Approach:**
- Use `fusepy` or `pyfuse3` Python library
- Map FUSE callbacks to Grover operations (read -> `read()`, write -> `write()`, readdir -> `list_dir()`)
- Version access via special `.versions/` virtual directory per file
- Read-only mode for safety, optional read-write

**Effort:** High. FUSE is complex (locking, caching, error handling, performance).

**Dependencies:** OS-level FUSE support (Linux, macOS via macFUSE).

---

### 12. PostgreSQL Row-Level Security

**What:** Add optional RLS policies for PostgreSQL deployments as defense-in-depth alongside existing `owner_id` application-level scoping.

**Why:** For SaaS deployments with strict compliance requirements, database-enforced isolation provides guarantees that application-level isolation alone cannot. If a bug in the application layer bypasses `owner_id` filtering, RLS catches it.

**Implementation:**
```sql
ALTER TABLE grover_files ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON grover_files
    USING (owner_id = current_setting('app.current_user')::text);
-- Repeat for grover_file_versions, grover_file_shares, grover_edges
```

Application sets `SET app.current_user = '<user_id>'` per connection.

**Effort:** Low. SQL migration + connection setup code. Optional feature, no impact on SQLite.

**Dependencies:** PostgreSQL only. Requires database migration management.

---

## Priority Matrix

| # | Opportunity | Value | Effort | Dependencies |
|---|-----------|-------|--------|--------------|
| 1 | deepagents Backend | High | Medium | deepagents optional dep |
| 2 | MCP Server | High | Medium-High | MCP SDK |
| 3 | Agent Memory Layer | High | Medium | None |
| 4 | Content-Addressable Storage | Medium-High | Medium | None |
| 5 | Zero-Copy Branching | Medium-High | High | Benefits from #4 |
| 6 | Enhanced Semantic Search | Medium | Medium | Optional LLM API |
| 7 | Sandbox Adapters | Medium | Low | Sandbox access |
| 8 | AFS Reference Impl | Medium | Medium | None |
| 9 | Multi-Agent Collaboration | Medium | Medium-High | Existing sharing |
| 10 | Cloud-Native Backend | Medium | High | #4, AWS/GCP SDKs |
| 11 | FUSE Mount | Low-Medium | High | OS FUSE support |
| 12 | PostgreSQL RLS | Low-Medium | Low | PostgreSQL only |

**Recommended sequence:** 1 -> 2 -> 3 -> 4 -> 5 -> 6 (high-value items first, building foundational capabilities before advanced features).

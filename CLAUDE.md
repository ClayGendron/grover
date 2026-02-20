# CLAUDE.md — Instructions for Claude Code

## What is Grover?

Grover is a Python toolkit that gives AI agents three integrated capabilities over codebases and documents:

1. **Versioned filesystem** — mount-based VFS with local disk + SQLite or pure-database backends, automatic versioning (snapshot + forward diffs), soft-delete trash, and rollback.
2. **Knowledge graph** — in-memory rustworkx directed graph of file dependencies, auto-populated by code analyzers (Python AST, JS/TS/Go via tree-sitter).
3. **Semantic search** — pluggable vector stores (local usearch, Pinecone, Databricks) with pluggable embedding providers (sentence-transformers, OpenAI, LangChain).

All three layers share a single identity model: **everything is a file path**. Graph nodes and search entries are keyed by file paths. Chunks (functions, classes) are stored as DB rows in `grover_file_chunks` but represented in the graph as nodes with synthetic path identifiers. An EventBus keeps layers in sync — write a file and the graph rebuilds, chunk records update, and embeddings re-index automatically.

**Status:** Alpha (v0.1.0). Core API is functional and tested. Expect breaking changes.

## Repository layout

```
src/grover/
├── __init__.py             # Public exports: __version__, Grover, GroverAsync, Ref, query types, file_ref
├── _grover.py              # Grover — sync facade (RLock + daemon thread event loop)
├── _grover_async.py        # GroverAsync — primary async class, wires all subsystems
├── ref.py                  # Ref frozen dataclass (path + version + line range)
├── events.py               # EventBus, EventType, FileEvent
├── fs/                     # Filesystem layer
│   ├── protocol.py         # StorageBackend + capability protocols (SupportsVersions, SupportsFileChunks, etc.)
│   ├── vfs.py              # VFS — mount router, session lifecycle, event emission
│   ├── local_fs.py         # LocalFileSystem — disk + SQLite (desktop/code editing)
│   ├── database_fs.py      # DatabaseFileSystem — pure DB, stateless (web apps/enterprise)
│   ├── operations.py       # Pure orchestration functions (read, write, edit, delete, move, copy)
│   ├── user_scoped_fs.py   # UserScopedFileSystem — DBFS subclass with user-scoping, sharing, @shared
│   ├── mounts.py           # MountRegistry, MountConfig
│   ├── sharing.py          # SharingService (path-based share CRUD, permission resolution)
│   ├── metadata.py         # MetadataService (file lookup, hashing)
│   ├── versioning.py       # VersioningService (diff storage, reconstruction)
│   ├── chunks.py           # ChunkService (DB-backed chunk CRUD)
│   ├── trash.py            # TrashService (soft-delete, restore, owner-scoped)
│   ├── directories.py      # DirectoryService (hierarchy ops)
│   ├── diff.py             # Unified diff compute/apply/reconstruct
│   ├── utils.py            # Path normalization, validation, mime detection, replacers
│   ├── types.py            # Result dataclasses (ReadResult, WriteResult, ShareResult, etc.)
│   ├── query_types.py      # New query response types (GlobQueryResult, GrepQueryResult, SearchQueryResult)
│   ├── exceptions.py       # GroverError hierarchy (+ AuthenticationRequiredError)
│   ├── permissions.py      # Permission enum (READ_WRITE, READ_ONLY)
│   └── dialect.py          # Dialect-aware SQL (upsert for SQLite/PostgreSQL/MSSQL)
├── integrations/
│   ├── deepagents/         # deepagents (LangGraph) integration
│   │   ├── __init__.py     # Exports: GroverBackend, GroverMiddleware (with import guard)
│   │   ├── _backend.py     # GroverBackend — BackendProtocol implementation
│   │   └── _middleware.py   # GroverMiddleware — AgentMiddleware with 10 tools
│   └── langchain/          # LangChain/LangGraph integration
│       ├── __init__.py     # Exports: GroverRetriever, GroverLoader, GroverStore (with import guard)
│       ├── _retriever.py   # GroverRetriever — BaseRetriever backed by semantic search
│       ├── _loader.py      # GroverLoader — BaseLoader for RAG document ingestion
│       └── _store.py       # GroverStore — LangGraph BaseStore for persistent memory
├── graph/
│   ├── __init__.py         # Exports: RustworkxGraph, GraphStore, SubgraphResult, capability protocols
│   ├── _rustworkx.py       # RustworkxGraph — rustworkx wrapper (CRUD, algorithms, persistence)
│   ├── protocols.py        # GraphStore + 7 capability protocols (SupportsCentrality, etc.)
│   ├── types.py            # SubgraphResult frozen dataclass, subgraph_result factory
│   └── analyzers/          # Language-specific code analyzers
│       ├── _base.py        # Analyzer protocol, ChunkFile, EdgeData
│       ├── __init__.py     # AnalyzerRegistry (maps extensions → analyzers)
│       ├── python.py       # PythonAnalyzer (stdlib ast)
│       ├── javascript.py   # JS/TS analyzers (tree-sitter)
│       └── go.py           # GoAnalyzer (tree-sitter)
├── search/
│   ├── _engine.py          # SearchEngine — orchestrator (EmbeddingProvider + VectorStore)
│   ├── protocols.py        # EmbeddingProvider, VectorStore, capability protocols
│   ├── types.py            # VectorEntry, VectorSearchResult, SearchResult, etc.
│   ├── filters.py          # Filter AST, operators, provider compilers
│   ├── extractors.py       # Text extraction (chunks → EmbeddableChunk)
│   ├── providers/
│   │   ├── __init__.py     # Provider exports (with import guards)
│   │   ├── openai.py       # OpenAIEmbedding (AsyncOpenAI)
│   │   ├── sentence_transformers.py  # SentenceTransformerEmbedding (async+sync)
│   │   └── langchain.py    # LangChainEmbedding adapter
│   └── stores/
│       ├── __init__.py     # Store exports (with import guards)
│       ├── local.py        # LocalVectorStore (usearch HNSW, VectorStore protocol)
│       ├── pinecone.py     # PineconeVectorStore (PineconeAsyncio, all capabilities)
│       └── databricks.py   # DatabricksVectorStore (Direct Vector Access, asyncio.to_thread)
└── models/                 # SQLModel database models
    ├── files.py            # File, FileVersion (grover_files, grover_file_versions) — FileBase.owner_id for user scoping
    ├── chunks.py           # FileChunk, FileChunkBase (grover_file_chunks) — DB-backed chunk storage
    ├── shares.py           # FileShare, FileShareBase (grover_file_shares) — path-based sharing between users
    ├── edges.py            # GroverEdge (grover_edges)
    └── embeddings.py       # Embedding (grover_embeddings)

tests/                      # pytest + pytest-asyncio (asyncio_mode = "auto")
├── conftest.py             # Shared fixtures (in-memory SQLite, async sessions)
├── test_grover.py          # Sync Grover integration tests
├── test_grover_async.py    # Async GroverAsync integration tests
├── test_public_api_contracts.py  # Public API surface and Result type contracts
├── test_base_fs.py         # DatabaseFileSystem CRUD, versioning, trash
├── test_local_fs.py        # LocalFileSystem disk I/O, safety, concurrency
├── test_database_fs.py     # DatabaseFileSystem stateless session injection
├── test_vfs.py             # VFS routing, permissions, events, session mgmt
├── test_capabilities.py    # Capability protocol detection and gating
├── test_graph.py           # RustworkxGraph node/edge ops, queries, traversal
├── test_graph_protocols.py # GraphStore protocol satisfaction, SubgraphResult immutability
├── test_graph_algorithms.py # Centrality, connectivity, traversal algorithm tests
├── test_graph_subgraph.py  # Subgraph extraction, neighborhood, meeting subgraph tests
├── test_graph_filtering.py # Filtering, edges_of, node similarity tests
├── test_analyzers.py       # Python/JS/TS/Go code analysis + chunk extraction
├── test_search.py          # Extractors, SearchResult, EmbeddingProvider protocol
├── test_search_engine.py   # SearchEngine orchestrator tests
├── test_local_vector_store.py # LocalVectorStore (VectorStore protocol) tests
├── test_search_protocols.py   # Protocols, types, filter AST tests
├── test_embedding_providers.py # OpenAI, SentenceTransformer, LangChain provider tests
├── test_pinecone_store.py  # PineconeVectorStore (all capabilities) mocked tests
├── test_databricks_store.py # DatabricksVectorStore (Direct Vector Access) mocked tests
├── test_events.py          # EventBus registration, dispatch, error handling
├── test_models.py          # SQLModel CRUD, defaults, upserts
├── test_diff.py            # Diff compute/apply/reconstruct round-trips
├── test_mounts.py          # MountConfig, MountRegistry, path resolution
├── test_fs_types.py        # Result dataclass fields and defaults
├── test_query_types.py     # New query response types (frozen immutability, defaults, construction)
├── test_query_api.py       # Facade API returns new query types (glob→GlobQueryResult, grep→GrepQueryResult, search→SearchQueryResult)
├── test_search_ops.py      # glob, grep, tree search operations
├── test_fs_utils.py        # Path utils, binary detection, replacers
├── test_dialect.py         # Multi-DB dialect support (SQLite, PostgreSQL, MSSQL)
├── test_configurable_model.py  # Custom table names, model subclassing
├── test_ref.py             # Ref immutability, equality, repr
├── test_user_scoped_fs.py  # UserScopedFileSystem isolation tests (path resolution, sharing, trash)
├── test_user_scoping.py    # User-scoped VFS integration tests — @shared access, trash scoping
├── test_sharing.py         # SharingService CRUD, permission resolution, expiration
├── test_move_follow.py     # Move with follow=True/False semantics
├── test_external_edit.py   # External edit detection — synthetic version insertion
├── test_file_chunks.py     # FileChunk model, ChunkService, SupportsFileChunks protocol
├── test_chunk_migration.py # Chunk storage migration (DB rows, graph cleanup, vector metadata)
├── test_per_mount.py       # Per-mount graph/search injection, isolation, routing, persistence
├── test_deepagents_backend.py   # GroverBackend (BackendProtocol) tests
├── test_deepagents_middleware.py # GroverMiddleware (AgentMiddleware) tests
├── test_langchain_retriever.py  # GroverRetriever (BaseRetriever) tests
├── test_langchain_loader.py     # GroverLoader (BaseLoader) tests
└── test_langchain_store.py      # GroverStore (LangGraph BaseStore) tests

docs/
├── api.md                  # Full API reference (all methods, types, protocols)
├── architecture.md         # Design patterns (composition, protocols, write ordering, events)
├── internals/fs.md         # Filesystem internals (write ordering, sessions, versioning, trash)
└── fs_architecture.md      # Component diagram
```

## Documentation map

| Question | Where to look |
|----------|---------------|
| How do I use the API? | `README.md` (quick start) → `docs/api.md` (full reference) |
| Why is the code structured this way? | `docs/architecture.md` |
| How does the filesystem layer work internally? | `docs/internals/fs.md` |
| What's the implementation roadmap? | `grover_implementation_plan.md` |
| How do I contribute? | `CONTRIBUTING.md` |
| What design decisions were made and why? | `docs/architecture.md` + memory file `design-decisions.md` |

## Key architectural rules

These are load-bearing decisions. Do not change them without explicit user discussion:

1. **Everything is a file** — no non-file entities, no URIs. Graph nodes = file paths. Chunks = DB rows in `grover_file_chunks` (represented in graph as nodes with synthetic path identifiers).
2. **Content-before-commit write ordering** — write content to storage THEN commit the DB session. Never reverse this. See `docs/internals/fs.md` for full rationale.
3. **Composition over inheritance** — no `BaseFileSystem` ABC. Backends compose shared services (MetadataService, VersioningService, etc.). Orchestration lives in `operations.py` as pure functions.
4. **Capability protocols** — `StorageBackend` + optional `SupportsVersions`, `SupportsTrash`, `SupportsReconcile`. Runtime-checked via `isinstance()`.
5. **Sessions owned by VFS** — backends never create, commit, or close sessions. They call `session.flush()` only. VFS handles the lifecycle.

## Tooling

| Tool | Command | Notes |
|------|---------|-------|
| **Lint** | `uvx ruff check src/ tests/` | Auto-fix: `uvx ruff check --fix src/ tests/` |
| **Format** | `uvx ruff format src/ tests/` | Check only: `uvx ruff format --check src/ tests/` |
| **Type check** | `uvx ty check src/` | Run on `src/` only — skip root-level `fs/` directory |
| **Tests** | `uv run pytest` | Coverage: `uv run pytest --cov` |
| **Install** | `uv pip install -e ".[all]"` | Use `uv pip`, NOT `uv sync` |
| **Dev deps** | `uv pip install --group dev` | pytest, pytest-asyncio, pytest-cov, pre-commit |

All four checks (lint, format, type check, tests) must pass before committing.

## Task workflow — MANDATORY

**This workflow is not optional.** You MUST follow it for every task unless the user explicitly says to skip it (e.g., "just do it", "skip the workflow", "no plan needed"). If you are unsure whether the user wants you to skip it, follow it. Do not write production code without a plan. Do not skip research. Do not skip code review.

The process has two modes depending on whether an approved plan already exists for the task.

---

### Mode A: No plan exists — Research and plan first

Do NOT write any production code in this mode. The only output is a plan document.

#### Step A1: Research

The goal is to deeply understand the problem before proposing a solution. Do all of the following:

1. **Consult memory.** Read the auto memory files for this project first. Check for past decisions, known pitfalls, user preferences, and pointers to relevant docs. If memory references a decision, trace it back to the doc file and read the full context. This prevents re-litigating settled questions and surfaces constraints you'd otherwise miss.

2. **Explore the codebase.** Read the source files, tests, and docs that are relevant to the task. Understand existing patterns, naming conventions, and how similar features are implemented. Use the Explore agent for broad searches and Glob/Grep for targeted lookups. Read enough to know:
   - What files you'll need to change or create
   - What patterns you must follow (composition, protocols, result types, etc.)
   - What tests already exist for the area you're touching
   - What could break

3. **Research externally.** If the task involves unfamiliar libraries, APIs, techniques, or design patterns, use WebSearch and WebFetch. Read official docs, not just blog posts. Understand the tool before using it.

4. **Ask the user questions.** Use the AskUserQuestion tool to ask focused, specific questions. Do not assume you understand the user's intent — verify it. You must ask about:
   - **The problem:** What are they trying to solve and why? What's the motivation?
   - **Desired behavior:** What should happen in the normal case? What about edge cases?
   - **Scope:** What's in scope and what's explicitly out of scope?
   - **Preferences:** If there are multiple valid approaches, which does the user prefer and why?
   - **Fit:** How does this relate to the broader project vision and roadmap?

   Ask multiple rounds of questions if needed. It is far better to ask too many questions than to build the wrong thing.

5. **Verify understanding.** Before moving to planning, summarize your understanding of the task back to the user. State what you think the goal is, what approach you're leaning toward, and what constraints you've identified. Get explicit confirmation that you have it right.

#### Step A2: Build a plan

1. Enter plan mode. Write the plan document. The plan must be structured so that each phase maps directly to one cycle of Mode B (implement → test → review → commit). The plan must include:

   **Header section:**
   - **Goal:** One sentence stating what this plan achieves.
   - **Context:** What you learned during research. What challenges exist. What the user told you. What constraints came from memory or docs.
   - **Approach:** The chosen design and why. What alternatives were considered and why they were rejected.

   **Phases section — each phase must include ALL of the following:**

   For every phase, spell out:

   - **Phase title and summary:** What this phase accomplishes. One sentence.
   - **Files to create or modify:** Exact file paths. For new files, describe what they contain. For existing files, describe what changes.
   - **Implementation details:** Specific enough that the code can be written without further design decisions. Name the classes, methods, data structures. Describe the logic. If there are non-obvious decisions, explain them here.
   - **Tests to write:** Exact test file path and a list of test cases by name/description. Include:
     - Happy-path tests for the core functionality
     - Edge case tests (empty input, missing files, permission errors, etc.)
     - Integration tests that exercise the feature end-to-end through the public API
   - **Code review focus areas:** What should the reviewer pay special attention to? What are the riskiest parts? What edge cases are most likely to have bugs?
   - **Doc updates:** Which docs need updating after this phase (if any).
   - **Acceptance criteria:** How do you know this phase is done? What must be true?

   Phases must be ordered so that each phase builds on the previous one. Each phase must leave the repo in a working state with all tests passing.

   **Open questions section:**
   - Anything still unresolved. If there are open questions, resolve them with the user before starting implementation.

2. Present the plan to the user for approval. Do not begin Mode B until the user approves.

---

### Mode B: Plan exists — Implement phase by phase

Execute each phase from the approved plan in order. Every phase goes through all three steps below. Do not skip steps. Do not combine phases.

#### Step B1: Implement

1. **Check memory for relevant context.** Before writing code, consult memory for any gotchas, user preferences, or past decisions that affect this phase. This is especially important when a phase touches areas worked on in previous sessions.

2. **Re-read the plan for this phase.** Read the exact phase spec from the plan. Check the files to modify, the implementation details, and the test list. Do not deviate from the plan without telling the user why.

3. **Write the production code.** Follow the plan's implementation details. Match existing patterns in the codebase (composition, protocols, result types, naming conventions). If you discover something the plan didn't anticipate, note it but keep going — raise it in the review step.

4. **Write the tests.** Write every test listed in the plan for this phase. Also add any additional tests you discover are needed during implementation. Tests must include:
   - Unit tests for new functions/methods
   - Edge case tests (the plan lists specific ones — write those plus any you find)
   - At least one integration test that exercises the feature through the public `Grover`/`GroverAsync` API

5. **Run all quality checks.** All four must pass:
   ```
   uv run pytest                              # all tests pass
   uvx ruff check src/ tests/                 # no lint errors
   uvx ruff format --check src/ tests/        # no format errors
   uvx ty check src/                          # no type errors
   ```
   Fix any failures before proceeding. Do not move to Step B2 with failing checks.

#### Step B2: Code review

1. **Launch a code review sub-agent.** Use the Task tool (general-purpose type) to spawn a reviewer. Give it:
   - The list of all files created or modified in this phase
   - The phase spec from the plan (acceptance criteria, focus areas)
   - Explicit instructions to do ALL of the following:

   The reviewer MUST:
   - **Read every changed/new file** line by line. Understand the logic, not just skim.
   - **Check against the plan.** Does the code implement what the plan says? Are there deviations?
   - **Check against existing patterns.** Does it follow the codebase's conventions (composition, protocols, result types, error handling)?
   - **Identify bugs and weak points.** Look for off-by-one errors, missing null checks, race conditions, unhandled exceptions, incorrect edge case behavior.
   - **Write and run throwaway test scripts.** The reviewer must write standalone Python scripts (NOT pytest test files) that import the new code, exercise it end-to-end, and probe the weak points and edge cases identified above. These are disposable verification scripts — they run inline via the Bash tool, not added to the test suite. The scripts should cover scenarios the pytest tests might miss: unusual inputs, failure modes, boundary conditions, ordering dependencies. Run each script and report pass/fail with details.
   - **Check for regressions.** Run `uv run pytest` and verify all existing tests still pass.
   - **Report findings.** Return a clear list of issues: what's wrong, where, and how to fix it. Distinguish between blocking issues (must fix) and suggestions (nice to have).

2. **Fix all blocking issues** the reviewer found. If the fix is non-trivial or changes the design, note it for the docs step.

3. **Re-run all quality checks** after fixes:
   ```
   uv run pytest
   uvx ruff check src/ tests/
   uvx ruff format --check src/ tests/
   uvx ty check src/
   ```
   All must pass. If the reviewer's scripts uncovered real bugs, make sure the fixes are covered by the existing pytest tests. If not, add a pytest test case for the bug.

#### Step B3: Update docs, record knowledge, and commit

1. **Update documentation.** Check each of these and update if this phase's changes affect them:
   - `docs/api.md` — new or changed public methods, types, or protocols
   - `docs/architecture.md` — new or changed design patterns, architectural decisions
   - `docs/internals/fs.md` — changes to filesystem internals (write ordering, sessions, versioning, trash)
   - `README.md` — new user-facing features, changed installation, changed quick start
   - `CONTRIBUTING.md` — changed dev workflow, new tooling, new test patterns
   - `CLAUDE.md` — changed repo structure, new files, changed tooling commands

2. **Record key decisions to memory.** If this phase involved a design choice, discovered constraint, or user preference worth preserving across sessions:
   - Write the full rationale in the appropriate documentation file (docs are the source of truth).
   - Add a brief pointer in the auto memory file linking to that doc, so future sessions find it fast.
   - If the user explicitly asked to remember something (e.g., "always use X", "never do Y"), save it to memory immediately.

3. **Commit.** Stage all changed files (production code, tests, docs) and commit with a clear message describing what this phase accomplished. One commit per phase.

4. **Move to the next phase.** Go back to Step B1 for the next phase in the plan.

---

### General rules

These apply at all times, in both modes:

- **This workflow is mandatory.** Do not skip steps. Do not write production code without a plan. Do not skip code review. The only exception is if the user explicitly tells you to skip the workflow.
- **Never write code without understanding the context first.** Read before you write. Consult memory before you read.
- **Tests are not optional.** Every phase must include tests. Every test must pass before committing. If you can't make tests pass, do not commit — fix the issue or raise it with the user.
- **Code review is not optional.** Every phase gets a sub-agent review with real integration tests. Do not self-review and call it done.
- **One commit per phase.** Each commit should leave the repo in a fully working state with all tests passing. No "WIP" commits.
- **Keep existing patterns.** Match the style, naming conventions, and architectural patterns already in the codebase. When in doubt, look at how similar things are done in the repo. Do not introduce new patterns without discussing it in the plan.
- **Don't over-engineer.** Solve the current problem. Don't add abstractions, config options, or features that aren't in the plan.
- **Don't deviate silently.** If you discover during implementation that the plan needs to change, tell the user. Do not quietly do something different from what was approved.
- **Documentation is the source of truth.** Key decisions, patterns, and rationale belong in the repo's doc files — not only in memory. Memory is a quick-access index that points to docs; docs are where knowledge actually lives.

### What goes where

| What you learned | Where to record it |
|-----------------|-------------------|
| Architectural decision or pattern | `docs/architecture.md` |
| Filesystem internal behavior | `docs/internals/fs.md` |
| API change or new public method | `docs/api.md` |
| Repo structure or tooling change | `CLAUDE.md` (this file) |
| User-facing feature or behavior | `README.md` |
| Quick pointer to any of the above | Auto memory (brief entry + link to the doc) |
| User preference or durable instruction | Auto memory (immediately) |

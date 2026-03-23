# Concepts and Resources for the Grover Rewrite

A study guide for implementing Grover's "everything is a file" architecture. Ordered by priority — what to study first has the most impact on getting the design right.

---

## 1. Virtual Filesystem Design

The core of the redesign. Understanding how an OS dispatches `read()`, `write()`, `list()` through a uniform interface to different backends is what makes Grover's architecture work.

### What to understand

- **VFS dispatch** — how a single `read()` syscall routes through the VFS layer to ext4, procfs, or FUSE depending on the mount point. The key insight: each filesystem fills in a struct of function pointers (`inode_operations`, `file_operations`). This is exactly what `GroverFileSystem` is — a protocol with method slots that each backend fills in.
- **Synthetic filesystems** — how procfs and sysfs present non-file data (process state, kernel parameters) as readable/writable files. `read /proc/42/status` doesn't read bytes from disk — the kernel generates the content on demand. Grover's `.chunks/`, `.versions/`, `.connections/` work the same way.
- **FUSE (Filesystem in Userspace)** — how userspace programs implement filesystems by handling a small set of callbacks. FUSE is the closest analogy to what Grover does. Study the callback surface: `getattr`, `readdir`, `open`, `read`, `write`, `unlink`, `mkdir`.
- **Plan 9 namespaces** — per-process mount tables, `bind`/`mount`, how `walk` resolves paths across mount points. This is the conceptual foundation for Grover's `MountRegistry` routing by path prefix.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **The Design and Implementation of the FreeBSD Operating System** (McKusick et al.) | [oreilly.com/library/view/design-and-implementation/9780133761825](https://www.oreilly.com/library/view/design-and-implementation/9780133761825/) | Chapter 8 (Local Filesystems) — clearest explanation of VFS internals |
| **FUSE documentation** (Linux kernel) | [docs.kernel.org/filesystems/fuse](https://docs.kernel.org/filesystems/fuse/) | Official kernel docs on FUSE architecture and I/O modes |
| **fusepy** — Python FUSE bindings | [github.com/fusepy/fusepy](https://github.com/fusepy/fusepy) / [pypi.org/project/fusepy](https://pypi.org/project/fusepy/) | Small, readable Python code showing the full FUSE callback surface |
| **"The Use of Name Spaces in Plan 9"** (Pike, Presotto, Thompson et al.) | [9p.io/sys/doc/names.html](https://9p.io/sys/doc/names.html) | The paper on Plan 9 namespaces — per-process mounts, bind, the namespace as the API |
| **"The Organization of Networks in Plan 9"** (Presotto, Winterbottom) | [9p.io/sys/doc/net/net.html](https://9p.io/sys/doc/net/net.html) | How Plan 9 turns network protocols into filesystem operations |

---

## 2. Tree Data Modeling in Relational Databases

The entire namespace is a tree stored in SQL. Getting the data model and indexing right determines whether tree queries (cascading deletes, subtree listings, ancestor lookups) are fast or full table scans.

### What to understand

- **Adjacency list** — each row stores `parent_path`. Simple, good for reads, but recursive queries (ancestors, descendants) require CTEs or multiple queries.
- **Materialized path** — the full path stored as a string, subtree queries via `WHERE path LIKE '/src/auth.py/%'`. Fast prefix queries, natural fit for Grover's path-as-identity model. B-tree indexes handle prefix `LIKE` efficiently (no leading wildcard).
- **Closure table** — stores all ancestor-descendant pairs explicitly. Fast for any tree query but expensive on writes. Likely overkill for Grover.
- **Nested set** — encode tree position as left/right integers. Fast reads, very expensive writes. Not suitable for mutable trees.
- **Index behavior with `LIKE`** — how SQLite and Postgres use B-tree indexes for `LIKE 'prefix%'` queries. This is critical for efficient subtree operations on the materialized path.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **SQL Antipatterns** (Bill Karwin) | [pragprog.com/titles/bksqla/sql-antipatterns](https://pragprog.com/titles/bksqla/sql-antipatterns/) | Chapter 3 covers all four tree models with clear tradeoffs |
| **Trees and Hierarchies in SQL for Smarties** (Joe Celko) | [shop.elsevier.com/books/joe-celkos-trees-and-hierarchies...](https://shop.elsevier.com/books/joe-celkos-trees-and-hierarchies-in-sql-for-smarties/celko/978-0-12-387733-8) | Dense but authoritative — the reference on tree structures in SQL |

---

## 3. Protocol and Interface Design

`GroverFileSystem` is the most important API surface in the system. Every backend implements it, every facade method calls it, every CLI command maps to it. Getting this right means the system composes cleanly; getting it wrong means everything is a special case.

### What to understand

- **Deep modules** — a module should have a simple interface that hides significant complexity. A wide, shallow interface (many methods, each trivial) is a design smell. `GroverFileSystem` should be deep: a small number of methods that each do meaningful work depending on path and kind.
- **The narrow waist** — IP is the narrow waist of the internet. Everything above speaks IP, everything below carries IP. `GroverFileSystem` is Grover's narrow waist. Study why a small, stable interface at the center enables everything above (facade, CLI, MCP) and everything below (database, local disk, external backends) to evolve independently.
- **Liskov substitution** — any backend must be substitutable without callers knowing. Return types, error behavior, and side effects must be consistent across implementations. If `DatabaseFileSystem.read()` returns a `GroverResult` with `success=False` for missing files, every backend must do the same — not raise exceptions.
- **Interface segregation** — a protocol shouldn't force implementors to stub out methods they don't support. But taken too far, you get 10 micro-protocols. The balance: one protocol with clear method groups, with unsupported operations returning `success=False`.
- **9P as a case study** — 14 message types for an entire distributed OS. Study why `walk`, `open`, `read`, `write`, `clunk`, `stat`, `create`, `remove` are sufficient, and what that tells you about the minimum viable interface.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **A Philosophy of Software Design** (John Ousterhout) | [web.stanford.edu/~ouster/cgi-bin/book.php](https://web.stanford.edu/~ouster/cgi-bin/book.php) | The best book on interface design. Short, opinionated, practical. Chapter 4 (Modules Should Be Deep) is directly relevant. **Read this first.** |
| **9P protocol specification** | [ericvh.github.io/9p-rfc/rfc9p2000.html](https://ericvh.github.io/9p-rfc/rfc9p2000.html) | The complete 9P2000 protocol — 14 message types for everything |
| **Python `typing.Protocol`** | [docs.python.org/3/library/typing.html#typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) | Official docs on structural subtyping in Python |
| **PEP 544 — Protocols: Structural subtyping** | [peps.python.org/pep-0544](https://peps.python.org/pep-0544/) | The PEP behind `Protocol` — covers runtime checking, generic protocols |

---

## 4. Async Python — Sessions, Concurrency, Coordination

Grover is async-first. Session lifecycle, the BackgroundWorker, and concurrent operations are where subtle bugs hide — the kind that pass all tests and break under real use.

### What to understand

- **SQLAlchemy async session lifecycle** — when `AsyncSession` is created, flushed, committed, closed. How `async with session` scoping works. The difference between `flush()` (sends SQL, doesn't commit) and `commit()` (finalizes transaction). Why backends should only `flush()` and let the VFS layer own commit/rollback.
- **Structured concurrency** — `asyncio.TaskGroup` (Python 3.11+) ensures all spawned tasks complete or cancel cleanly. No orphaned tasks, no silent failures. Contrast with raw `asyncio.create_task()` where exceptions can be silently swallowed.
- **Producer-consumer with backpressure** — `write()` produces work (analyze, chunk, embed), the `BackgroundWorker` consumes it. Understanding debounce semantics: if the same file is written 5 times in 100ms, analyze once. How to drain pending work on shutdown (`flush()`).
- **The "session per operation" pattern** — each high-level operation gets its own session scope. Don't share sessions across concurrent operations. Don't pass sessions between tasks.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **SQLAlchemy async documentation** | [docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) | AsyncSession, AsyncEngine, eager loading, concurrent safety |
| **Python Concurrency with asyncio** (Matthew Fowler) | [manning.com/books/python-concurrency-with-asyncio](https://www.manning.com/books/python-concurrency-with-asyncio) | Practical coverage of TaskGroups, structured concurrency, real patterns |
| **trio design docs** — structured concurrency | [trio.readthedocs.io/en/stable/design.html](https://trio.readthedocs.io/en/stable/design.html) | The conceptual foundation for structured concurrency — even if you use asyncio, the ideas transfer |

---

## 5. CLI Design for Composability

The CLI is the primary agent interface. It needs to work for humans typing commands AND for LLMs generating bash. This is a specific design discipline with well-established conventions.

### What to understand

- **The Unix filter pattern** — read paths from stdin, write paths to stdout, errors to stderr. Exit code 0 for success, non-zero for failure. Every command must work as a pipeline stage. This is what makes `grover search "auth" | grover glob "*.py" | grover pred` possible.
- **Output formats** — human-readable by default, `--json` for structured output agents can parse. Consistent format across all commands. `jq`-compatible JSON.
- **Progressive disclosure** — top-level `--help` is brief (list commands), subcommand `--help` is detailed (flags, examples), `read /path/.api/endpoint` is complete schema. The agent discovers capabilities by navigating, not by reading documentation.
- **Idempotency and safety** — read commands are safe (no side effects), write commands are clearly marked. Destructive operations (`rm`, `edit`) should require explicit paths, never operate on implicit state.
- **Streaming large results** — `glob` or `grep` on a large namespace can return thousands of paths. Stream results line-by-line (like `find`), don't buffer everything into memory.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **Command Line Interface Guidelines** | [clig.dev](https://clig.dev) | The definitive guide to CLI design. Covers output, errors, signals, interactivity, composition. **Read this second.** |
| **Typer** — Python CLI framework | [typer.tiangolo.com](https://typer.tiangolo.com) | Type-hint-based CLI framework built on Click. By the creator of FastAPI. |

---

## 6. Search and Retrieval Fundamentals

Grover combines four retrieval modalities (glob, grep, semantic, structural). Understanding their strengths, weaknesses, and how they compose determines whether the search stack is genuinely useful or just a collection of features.

### What to understand

- **What embeddings are good and bad at** — good: semantic similarity, paraphrase detection, topic matching. Bad: exact string matching, negation ("files that DON'T mention auth"), structured queries. This tells you when `search` adds value vs. when `grep` is the right tool.
- **Reciprocal Rank Fusion (RRF)** — the standard algorithm for combining results from multiple retrievers. Uses rank, not score: `score = 1/(k + rank)`. Simple to implement, works well in practice. Understand why rank-based fusion is more robust than score-based fusion (scores aren't comparable across retrievers).
- **Inverted indexes and full-text search** — how grep/lexical search works under the hood. SQLite FTS5 for full-text search. When FTS is better than `LIKE '%pattern%'` (FTS uses an inverted index; `LIKE` with leading wildcard is a full scan).
- **Graph algorithms for search** — BFS (predecessors/successors), DFS (ancestors/descendants), PageRank (importance), betweenness centrality (bridges). Not the math — the practical question: when would an agent use each one? PageRank finds important files. Betweenness finds files that connect clusters. Neighborhood finds related files.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **Introduction to Information Retrieval** (Manning, Raghavan, Schütze) | [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book/) | Free online (HTML + PDF). Chapters 6 (scoring/ranking) and 8 (evaluation) are most relevant |
| **Qdrant blog: Hybrid Search Revamped** | [qdrant.tech/articles/hybrid-search](https://qdrant.tech/articles/hybrid-search/) | Practical guide to hybrid search — fusion methods (RRF, DBSF), late interaction reranking |
| **rustworkx documentation** | [rustworkx.org](https://www.rustworkx.org) | API reference for the graph library Grover uses — algorithms, traversal, centrality |

---

## 7. Plugin and Extension Architecture

Analyzers, backends, search providers, embedding providers — Grover has many extension points. Getting the plugin boundary right means third parties can extend it without understanding the internals.

### What to understand

- **Protocols over ABCs** — `typing.Protocol` gives you structural subtyping (duck typing with type checking). A backend doesn't need to inherit from a base class — it just needs to have the right methods. This is how Go interfaces work, and it's the right model for Grover's extension points.
- **Entry points for plugin discovery** — Python's `importlib.metadata` entry points let third parties register plugins via `pyproject.toml`. `pip install grover-jira` auto-registers a Jira backend that Grover discovers at runtime. This is how pytest discovers plugins.
- **Dependency injection without a framework** — constructors that accept protocols, not concrete types. `add_mount(path, backend=my_backend)` is DI. No container, no decorator magic. Just pass the implementation in.
- **The plugin sandwich** — the framework calls the plugin (via the protocol), the plugin calls utility functions the framework provides (path normalization, result construction), but the plugin never calls the framework's core (facade, mount registry). This prevents circular dependencies.

### Resources

| Resource | URL | Notes |
|---|---|---|
| **Architecture Patterns with Python** (Percival & Gregory) | [cosmicpython.com](https://www.cosmicpython.com) | Free online. Covers dependency injection, repository pattern, hexagonal architecture |
| **pytest: Writing plugins** | [docs.pytest.org/en/stable/how-to/writing_plugins.html](https://docs.pytest.org/en/stable/how-to/writing_plugins.html) | How pytest discovers and loads plugins via entry points — study the pattern |
| **Python packaging: Entry points** | [packaging.python.org/en/latest/specifications/entry-points](https://packaging.python.org/en/latest/specifications/entry-points/) | Official spec on entry point groups, registration, and discovery |

---

## Reading order

If time is limited, read these three first — they set the foundation for everything else:

1. **A Philosophy of Software Design** (Ousterhout) — [web.stanford.edu/~ouster/cgi-bin/book.php](https://web.stanford.edu/~ouster/cgi-bin/book.php) — will change how you design the GroverFileSystem protocol
2. **Command Line Interface Guidelines** — [clig.dev](https://clig.dev) — will set up the CLI correctly from day one
3. **"The Use of Name Spaces in Plan 9"** — [9p.io/sys/doc/names.html](https://9p.io/sys/doc/names.html) — will deepen your intuition for the filesystem metaphor

Then pick up the domain-specific resources as you implement each layer.

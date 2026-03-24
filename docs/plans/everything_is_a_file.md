# Grover: Everything is a File

A design document for rethinking Grover as a virtual filesystem for knowledge — where documents, relationships, versions, and search results are all navigable paths that any agent can explore with `read`, `write`, `ls`, and `grep`.

---

## 1. The Problem

Enterprise knowledge is fragmented. Documents live in SharePoint. Conversations live in Slack. Tickets live in Jira. Code lives in Git. Relationships between them — who wrote what, what references what, what decision led to what change — exist only in people's heads or scattered across systems with incompatible APIs.

AI agents make this worse before they make it better. Every data source becomes an MCP server with 10-20 tools. Three MCP servers can consume **72% of a 200K-token context window** before the agent processes a single user message (documented in production). At **30+ tools**, LLM tool selection accuracy degrades measurably. At 100+, it fails entirely.

The industry response has been to build more tools, more connectors, more integrations. The result is tool sprawl — hundreds of specialized endpoints that agents struggle to discover, select, and compose.

There is a better approach, and it was invented in 1969.

## 2. Everything is a File

Unix's defining insight was that radically different things — disk drives, terminals, network connections, running processes — could all be accessed through a single interface: the file. Open it, read it, write it, close it. The path is the address. The content is the data. Standard tools (`grep`, `ls`, `cat`, `find`) work on everything.

Plan 9 from Bell Labs (1992) took this further. The network became a filesystem (`/net/tcp/clone`). The window system became a filesystem (`/dev/draw`). Process control became a filesystem (`/proc/42/ctl`). The entire protocol had **14 message types**: walk, open, read, write, stat, create, remove, clunk, and a handful of others. That was sufficient to model every resource in a distributed operating system.

The key insight is not that files are the right abstraction for everything. It's that **a small, universal interface eliminates the need for special-purpose APIs**. When everything speaks the same protocol, tools compose. When everything lives in a single namespace, discovery is navigation. When operations are uniform, agents don't need instructions — they already know how to use a filesystem.

### 2.1 The Three Layers

The Wikipedia article on "Everything is a file" identifies three layers of the Unix principle, in order of importance:

1. **Uniform handle** — represent objects as file descriptors, not abstract handles or names. Any descriptor can be passed to the same operations.
2. **Standard operations** — operate on objects with read/write, returning byte streams interpreted by applications. No type-specific APIs.
3. **Namespace presence** — objects exist in a global filesystem namespace. Discovery is `ls`. Addressing is a path.

Grover implements all three:

- **Layer 1 (Uniform handle):** Every entity — document, chunk, version, connection — is a kinded object in the database. The `Ref` type wraps any path and can be passed to any operation (`read`, `write`, `delete`, `list`, `stat`).
- **Layer 2 (Standard operations):** A small set of verbs (`read`, `write`, `delete`, `list`, `stat`, `edit`, `move`, `copy`, `mkdir`, `mkconn`, `glob`, `grep`, `search`) works on any path regardless of entity type. The path determines the semantics.
- **Layer 3 (Namespace):** All entities live in a single hierarchical namespace. Chunks are children of their file. Versions are children of their file. Connections are children of their source file. Discovery is `ls`. Search is `glob` and `grep`. Addressing is always a path.

## 3. The Namespace

Grover's namespace is a **virtual overlay** — paths are logical, not necessarily physical. File bytes can live on disk (local mounts) or in the database (DB mounts). Metadata nodes (`.chunks/`, `.versions/`, `.connections/`, `.api/`) always live in the database. Operations like `read`, `ls`, `glob`, `grep`, and graph queries operate on the logical namespace, not the physical filesystem.

This means:
- A local mount has real files on disk + metadata in SQLite
- A database mount has everything in one table
- Both present the same namespace, same operations, same behavior
- An agent can't tell which backend it's talking to — and doesn't need to

### 3.1 Path Structure

Every entity in Grover has a path. The hierarchy encodes relationships:

```
/
├── documents/
│   ├── quarterly-report-q4.pdf                         (file)
│   │   ├── .chunks/
│   │   │   ├── executive-summary                       (chunk)
│   │   │   └── revenue-analysis                        (chunk)
│   │   ├── .versions/
│   │   │   ├── 1                                       (version)
│   │   │   └── 2                                       (version)
│   │   └── .connections/
│   │       ├── references/
│   │       │   └── documents/budget-2025.xlsx           (connection)
│   │       ├── authored-by/
│   │       │   └── people/jane-smith                    (connection)
│   │       └── discussed-in/
│   │           └── threads/slack-proj-123               (connection)
│   └── budget-2025.xlsx                                (file)
├── people/
│   └── jane-smith                                      (file)
│       └── .connections/
│           ├── authored/
│           │   └── documents/quarterly-report-q4.pdf    (connection)
│           └── member-of/
│               └── teams/finance                        (connection)
├── threads/
│   └── slack-proj-123                                  (file)
│       ├── .chunks/
│       │   ├── msg-001                                 (chunk)
│       │   └── msg-002                                 (chunk)
│       └── .connections/
│           └── references/
│               └── documents/quarterly-report-q4.pdf    (connection)
├── tickets/
│   └── JIRA-4521                                       (file)
│       └── .connections/
│           ├── blocks/
│           │   └── tickets/JIRA-4519                    (connection)
│           └── assigned-to/
│               └── people/jane-smith                    (connection)
└── src/
    └── auth.py                                         (file)
        ├── .chunks/
        │   ├── login                                   (chunk)
        │   └── AuthService                             (chunk)
        ├── .versions/
        │   ├── 1                                       (version)
        │   └── 2                                       (version)
        └── .connections/
            ├── imports/
            │   └── src/utils.py                        (connection)
            └── calls/
                └── src/db.py                           (connection)
```

### 3.2 Path Conventions

Metadata children of a file use dot-prefixed directories, following the Unix convention for hidden files (`.git`, `.ssh`, `.config`):

| Metadata type | Path pattern | Example |
|---|---|---|
| **Chunk** | `<file>/.chunks/<name>` | `/src/auth.py/.chunks/login` |
| **Version** | `<file>/.versions/<N>` | `/src/auth.py/.versions/3` |
| **Connection** | `<file>/.connections/<type>/<target>` | `/src/auth.py/.connections/imports/src/utils.py` |
| **API endpoint** | `<mount>/.api/<action>` | `/jira/.api/ticket` |

This design has several properties:

- **Standard path parsing.** No special sigils (`#`, `@`, `[]`). Every path is `/`-separated. Standard path libraries work unchanged.
- **Glob works natively.** `glob("/src/**/.chunks/*")` finds all chunks. `glob("/**/.connections/imports/**")` finds all import edges. `glob("/**/.versions/*")` finds all versions.
- **`ls -a` semantics.** `ls /src/auth.py` hides metadata children by default. `ls -a /src/auth.py` reveals `.chunks/`, `.versions/`, `.connections/`. This matches Unix hidden file convention.
- **No collision with user content.** Nobody creates source files or documents called `.chunks`.

### 3.3 `parent_path` Derivation

`parent_path` is **stored metadata, computed at write time** — not trivially derived by splitting on `/`. For files and directories, it's the standard filesystem parent. For metadata nodes, it's the owning file — determined by finding the dot-directory marker:

| Path | parent_path | How derived |
|---|---|---|
| `/src/auth.py` | `/src` | Standard: split on `/`, drop last segment |
| `/src` | `/` | Standard: split on `/`, drop last segment |
| `/src/auth.py/.chunks/login` | `/src/auth.py` | Marker: everything before `/.chunks/` |
| `/src/auth.py/.versions/3` | `/src/auth.py` | Marker: everything before `/.versions/` |
| `/src/auth.py/.connections/imports/src/utils.py` | `/src/auth.py` | Marker: everything before `/.connections/` |
| `/jira/.api/ticket` | `/jira` | Marker: everything before `/.api/` |

Connection paths deserve special attention. The target path (`/src/utils.py`) can contain `/`, which means "split on `/`, drop last segment" would give `/src/auth.py/.connections/imports/src` — wrong. The correct parent is always derived using the `/.connections/` marker. This is handled by `Ref.base_path`:

```python
@property
def base_path(self) -> str:
    for marker in ("/.chunks/", "/.versions/", "/.connections/", "/.api/"):
        idx = self.path.find(marker)
        if idx >= 0:
            return self.path[:idx]
    return self.path
```

This is marker-aware parsing, but it's still simple and unambiguous — no regex, no bracket matching, no ambiguity about where tokens start and end. The `parent_path` column is written once at creation time and indexed for efficient tree queries.

Chunks, versions, and connections are structurally children of their parent file. Cascading deletes, permission inheritance, and tree traversal all follow from this hierarchy.

## 4. The Unified Data Model

### 4.1 Kinded Object Model

All entities in the namespace share a common identity model: a `path`, a `kind`, and a `parent_path`. Whether these live in one table or multiple tables is an implementation choice — what matters is that the logical model is uniform and every object is addressable by path.

The single-table approach is shown here as a natural fit for the "everything is a file" philosophy, but the namespace and API design do not depend on it. The core invariants are: (1) every object has a unique path, (2) every object has a kind, (3) `parent_path` enables tree queries and cascading operations.

Files, chunks, versions, and connections share one logical model. Shares remain separate (they are ACLs — metadata about objects, not objects themselves).

```
grover_objects
──────────────────────────────────────────────────────
id              TEXT PRIMARY KEY
path            TEXT UNIQUE NOT NULL, INDEXED
parent_path     TEXT NOT NULL, INDEXED
kind            TEXT NOT NULL, INDEXED
                  -- file | directory | chunk | version | connection | api

-- Content payload
content         TEXT NULL
content_hash    TEXT NULL
mime_type       TEXT DEFAULT 'text/plain'

-- Metrics (nullable, kind-dependent)
lines           INTEGER DEFAULT 0
size_bytes      INTEGER DEFAULT 0
tokens          INTEGER DEFAULT 0

-- Chunk-specific
line_start      INTEGER NULL
line_end        INTEGER NULL

-- Version-specific
version_number  INTEGER NULL
is_snapshot     BOOLEAN NULL
created_by      TEXT NULL

-- Connection-specific
source_path     TEXT NULL, INDEXED
target_path     TEXT NULL, INDEXED
connection_type TEXT NULL
weight          REAL DEFAULT 1.0

-- Common
embedding       VECTOR NULL
owner_id        TEXT NULL, INDEXED
original_path   TEXT NULL
created_at      TIMESTAMP
updated_at      TIMESTAMP
deleted_at      TIMESTAMP NULL
```

**Why one table:**

- **Uniform operations.** `read()`, `write()`, `delete()`, `list()`, `stat()` become a single code path with kind-aware behavior, not four separate service classes.
- **Cascading deletes.** `DELETE FROM grover_objects WHERE parent_path = '/src/auth.py'` removes all chunks, versions, and connections in one query.
- **Permission inheritance.** A share on `/src/auth.py` covers all its metadata children via path-prefix check.
- **Unified search.** `grep` can search across file content, chunk content, and version content in one query.
- **Unified glob.** `glob("/**/.connections/imports/**")` finds all import edges across the entire namespace.

**Why shares stay separate:**

Shares are ACLs — they describe who can access what, not what exists. They reference paths in the objects table but don't participate in the namespace hierarchy. This matches Unix: permissions are metadata on inodes, not entries in the directory tree.

### 4.2 The `Ref` Type

`Ref` is the uniform handle — the file descriptor equivalent. It wraps a path and can be passed to any operation:

```python
@dataclass(frozen=True)
class Ref:
    path: str

    @property
    def is_chunk(self) -> bool:
        return "/.chunks/" in self.path

    @property
    def is_version(self) -> bool:
        return "/.versions/" in self.path

    @property
    def is_connection(self) -> bool:
        return "/.connections/" in self.path

    @property
    def is_file(self) -> bool:
        return not (self.is_chunk or self.is_version or self.is_connection)

    @property
    def base_path(self) -> str:
        """The parent file this metadata belongs to."""
        for marker in ("/.chunks/", "/.versions/", "/.connections/"):
            idx = self.path.find(marker)
            if idx >= 0:
                return self.path[:idx]
        return self.path
```

No regex. No sigil parsing. No ambiguity. Just path segments.

## 5. Uniform Operations

### 5.1 The Protocol

The `GroverFileSystem` protocol shrinks to core operations that work on any path:

```python
class GroverFileSystem(Protocol):
    # Core CRUD — works on ANY path/kind
    async def read(self, path, *, session) -> GroverResult: ...
    async def write(self, path, content, *, session) -> GroverResult: ...
    async def delete(self, path, *, session) -> GroverResult: ...
    async def stat(self, path, *, session) -> GroverResult: ...
    async def exists(self, path, *, session) -> GroverResult: ...
    async def list(self, path, *, session) -> GroverResult: ...
    async def edit(self, path, old, new, *, session) -> GroverResult: ...
    async def move(self, src, dest, *, session) -> GroverResult: ...
    async def copy(self, src, dest, *, session) -> GroverResult: ...
    async def mkdir(self, path, *, session) -> GroverResult: ...
    async def mkconn(self, source, type, target, *, session) -> GroverResult: ...

    # Batch variants
    async def read_files(self, paths, *, session) -> GroverResult: ...
    async def write_files(self, files, *, session) -> GroverResult: ...
    async def move_files(self, pairs, *, session) -> GroverResult: ...
    async def copy_files(self, pairs, *, session) -> GroverResult: ...

    # Query — returns FileSearchResult
    async def glob(self, pattern, *, session) -> FileSearchResult: ...
    async def grep(self, pattern, *, session) -> FileSearchResult: ...
    async def tree(self, path, *, session) -> GroverResult: ...
    async def search(self, query, *, k) -> FileSearchResult: ...
    async def lsearch(self, query, *, session) -> FileSearchResult: ...

    # Graph algorithms — operate on paths
    async def predecessors(self, candidates, *, session) -> FileSearchResult: ...
    async def successors(self, candidates, *, session) -> FileSearchResult: ...
    async def ancestors(self, candidates, *, session) -> FileSearchResult: ...
    async def descendants(self, candidates, *, session) -> FileSearchResult: ...
    async def neighborhood(self, candidates, *, depth, session) -> FileSearchResult: ...
    async def meeting_subgraph(self, candidates, *, session) -> FileSearchResult: ...
    async def min_meeting_subgraph(self, candidates, *, session) -> FileSearchResult: ...
    async def pagerank(self, candidates, *, session) -> FileSearchResult: ...
    # ... centrality, hits, etc.
```

**What disappeared:**

- `add_connection` / `delete_connection` / `list_connections` → `mkconn()` / `delete()` / `list()`
- `replace_file_chunks` / `delete_file_chunks` / `list_file_chunks` / `write_chunks` → `write()` / `delete()` / `list()`
- `search_add_batch` / `search_remove_file` → internal to the write/delete pipeline
- `vector_search` / `lexical_search` → `search()` / `lsearch()`

### 5.2 Operation Semantics by Kind

Each operation adapts its behavior based on the path's kind:

| Operation | file | directory | chunk | version | connection | api |
|---|---|---|---|---|---|---|
| `read` | content | error | chunk content | reconstructed snapshot | metadata | schema |
| `write` | create/update + auto-version | mkdir | create/update chunk | error (read-only) | create edge | trigger action |
| `delete` | soft-delete + cascade | rmdir | remove chunk | prune version | remove edge | error |
| `list` | metadata children | content children | error (leaf) | error (leaf) | error (leaf) | error (leaf) |
| `edit` | string replace | error | string replace | error | error | error |
| `stat` | metadata | metadata | metadata | metadata | metadata | metadata |
| `move` | rename + cascade | rename + cascade | re-parent | error | error | error |
| In search results | yes (default) | yes (default) | opt-in | opt-in | no | never |
| Embeddable | yes | no | yes | yes | no | never |
| Versioned | yes | no | no | N/A | no | never |

Versions are **read-only** — created as a side effect of writing their parent file. This matches the procfs pattern: `/proc/42/status` is generated by the kernel, not written by the user.

`.api/` nodes are **control plane only** — discoverable (`ls -a`, `read`), actionable (`write`), but never indexed, never embedded, never versioned, never in search results.

### 5.3 Creation Primitives

Like Unix has `mkdir`, `mkfifo`, `mknod` — specialized creation commands for specific entity kinds:

```bash
grover mkdir  /documents/q1-reports/       # create directory
grover mkconn /src/auth.py imports /src/utils.py  # create connection
# creates: /src/auth.py/.connections/imports/src/utils.py
```

Under the hood, `mkconn` is `write()` with `kind=connection` — the same way `mkdir` is `mknod` with `S_IFDIR`. The named command is ergonomic sugar.

### 5.4 Visibility Defaults: Files First, Metadata Opt-In

By default, operations return **files and directories only**. Metadata (chunks, versions, connections) and control plane (`.api/`) nodes are accessible but must be explicitly requested or directly addressed. This prevents agents and users from drowning in metadata results.

| Operation | Default behavior | Opt-in for metadata |
|---|---|---|
| `ls /path` | Files and directories only | `ls -a /path` includes `.chunks/`, `.versions/`, `.connections/`, `.api/` |
| `glob "**"` | Matches files and directories | `glob "**" --all` includes metadata paths |
| `grep "pattern"` | Searches file content only | `grep "pattern" --chunks` includes chunk content |
| `search "query"` | Matches files only (kind=file) | `search "query" --kinds chunk,version` includes metadata |
| `tree /path` | Files and directories only | `tree -a /path` shows full tree including metadata |
| `read /path/.chunks/foo` | Always works — explicit paths always resolve | N/A |
| `predecessors`, `successors`, etc. | Returns files (traverses connections internally) | N/A |

The principle: **if you name it explicitly, you get it. If you search broadly, you get files.** An agent doing `grep "timeout"` gets the 12 files that matter, not 500 chunk results for every function in the codebase. But `read /src/auth.py/.chunks/login` or `ls -a /src/auth.py` work exactly as expected when the agent specifically wants metadata.

`.api/` nodes follow stricter rules — they are **never** included in search results, never embedded, never versioned, never returned by graph traversal. They are control plane: discoverable via `ls -a` and `read`, actionable via `write`, but invisible to the knowledge layer.

## 6. The Agent Interface

### 6.1 The MCP Problem

Model Context Protocol tools consume **550-1,400 tokens each** for their schema definitions, loaded into the context window on every interaction. A team reported three MCP servers consuming **143,000 of 200,000 tokens** (72%) before the agent processed a single user message. At 30+ tools, selection accuracy degrades. At 100+, failure is virtually guaranteed.

The industry response has been:
- **Dynamic tool loading** (Claude Code's MCP Tool Search) — retrieve tool schemas on demand, reducing overhead by 85%
- **CLI patterns** (Apideck) — replace tool schemas with a single prompt pointing to a CLI binary, reducing overhead by 96%
- **Filesystem-based tool discovery** (Anthropic's "Code Execution with MCP") — convert tools into files agents explore on demand, reducing overhead by **98.7%**

Anthropic's own recommendation is filesystem-based discovery. Grover *is* that filesystem.

### 6.2 One Tool

Grover exposes itself as a **single MCP tool** — a CLI that accepts filesystem commands:

```
Tool: grover
Description: Enterprise knowledge filesystem. Use standard filesystem
commands to explore, search, and manage knowledge.
Run `grover --help` for available commands.
```

That's ~40 tokens. The agent discovers capabilities progressively:

```bash
$ grover --help
Commands: read, write, edit, rm, mv, cp, mkdir, mkconn,
          ls, stat, tree, glob, grep, search, lsearch,
          pred, succ, anc, desc, nbr, meet, rank

$ grover search --help
Usage: grover search <query> [--k N]
Semantic search across all mounted content.
Returns ranked paths with relevance scores.
```

Each `--help` call costs 50-200 tokens, loaded **only when needed**. The agent keeps 95%+ of its context window for reasoning.

For `.api/` endpoints, `--help` and `read` return the same schema through different interfaces:

```bash
# CLI convention — human at a terminal
$ grover write /jira/.api/ticket --help

# Filesystem convention — agent navigating the namespace
$ grover read /jira/.api/ticket
```

Both return the write schema. `--help` is a CLI feature (the CLI calls `read()` on the path before executing). `read` is the filesystem primitive. Same content, two access patterns — like `man ls` and `ls --help` coexisting alongside the filesystem itself.

### 6.3 The CLI

Every command takes paths in (arguments or stdin) and produces paths out (stdout). Unix pipes compose them:

```bash
# CRUD
grover read   /documents/quarterly-report-q4.pdf
grover write  /documents/new-report.md < content.md
grover edit   /src/auth.py "old_function" "new_function"
grover rm     /documents/outdated.pdf
grover mv     /documents/draft.md /documents/final.md
grover cp     /src/auth.py /src/auth_backup.py
grover mkdir  /documents/q1-reports/
grover mkconn /src/auth.py imports /src/utils.py
grover ls     /documents/quarterly-report-q4.pdf
grover stat   /documents/quarterly-report-q4.pdf
grover tree   /documents/

# Search
grover glob   "documents/**/*.pdf"
grover grep   "revenue" --glob "*.pdf"
grover search "Q4 financial performance"
grover lsearch "quarterly report"

# Graph traversal
grover pred   /documents/quarterly-report-q4.pdf
grover succ   /documents/quarterly-report-q4.pdf
grover anc    /src/auth.py
grover desc   /src/auth.py
grover nbr    /documents/quarterly-report-q4.pdf --depth 2
grover meet   /src/auth.py /src/db.py
grover rank
```

### 6.4 Composable Pipelines

Every command outputs paths, so composition is free:

```bash
# "Find all documents about authentication, narrow to PDFs,
#  get everything they reference"
grover search "authentication" | grover glob "*.pdf" | grover succ

# "What are the most central files related to revenue?"
grover grep "revenue" | grover rank

# "Find TODO items, then show what depends on those files"
grover grep "TODO" | grover pred

# "Semantic search, intersected with a glob, expanded to neighbors"
grover search "error handling" | grover glob "src/api/**" | grover nbr

# "Who wrote documents related to the Q4 budget?"
grover search "Q4 budget" | grover succ --type "authored-by"

# "What tickets reference files that were recently changed?"
grover glob "/**/.versions/[0-9]*" | grover pred --type "references"
```

This is the Unix pipe model applied to enterprise search. Each stage refines or expands a set of paths. The `FileSearchResult` with its set algebra (`&`, `|`, `-`, `>>`) is the programmatic equivalent.

### 6.5 `ls` at Every Level

With the hierarchical namespace, `ls` shows different things at different levels — like `ls /proc/1234`:

```bash
$ grover ls /documents/
quarterly-report-q4.pdf
budget-2025.xlsx

$ grover ls /documents/quarterly-report-q4.pdf
.chunks/
.versions/
.connections/

$ grover ls /documents/quarterly-report-q4.pdf/.chunks/
executive-summary
revenue-analysis

$ grover ls /documents/quarterly-report-q4.pdf/.connections/
references/
authored-by/
discussed-in/

$ grover ls /documents/quarterly-report-q4.pdf/.connections/authored-by/
people/jane-smith
```

## 7. Agentic Search

### 7.1 The False Dichotomy

The industry frames search as **RAG vs. agentic search**. RAG pre-computes embeddings and retrieves in one shot. Agentic search (as Claude Code does) iteratively greps and reads. The research consensus is that neither alone is sufficient:

- **RAG alone** fails on multi-hop questions, misses structural relationships, and drifts from reality during active editing.
- **Agentic search alone** is slow on large corpora and cannot find semantically similar content that uses different terminology.

The winning pattern, validated by Sourcegraph Cody and A-RAG (arXiv:2602.03442), is **pre-computed structural indices exposed as on-demand search tools** the agent composes iteratively.

### 7.2 Grover's Search Stack

Grover provides four retrieval modalities, all returning `FileSearchResult` (composable via set algebra):

| Modality | Command | What it finds | Pre-computed? |
|---|---|---|---|
| **Pattern** | `glob` | Files matching path patterns | No (on-demand) |
| **Keyword** | `grep` | Files containing text matches | No (on-demand) |
| **Semantic** | `search` | Files similar to a natural language query | Yes (embeddings) |
| **Structural** | `pred`/`succ`/`nbr`/`meet`/`rank` | Files related by graph edges | Yes (graph) |

The agent decides which to use and in what order. A typical search loop:

1. `grover search "authentication logic"` — semantic search, broad net
2. `grover glob "src/**/*.py"` — filter to Python source
3. Result 1 `&` Result 2 — intersect
4. `grover nbr` (piped) — expand to graph neighbors
5. `grover read` (piped) — load the relevant files into context

This is A-RAG's three principles in filesystem form: autonomous strategy (agent chooses), iterative execution (multi-round), interleaved tool use (compose results).

### 7.3 Composable Results

`FileSearchResult` supports set operations that mirror Unix pipe composition:

```python
# Programmatic equivalent of CLI pipes
semantic = await g.search("authentication")
python_files = await g.glob("src/**/*.py")
candidates = semantic & python_files              # intersection
expanded = candidates | await g.neighborhood(candidates)  # union with neighbors
final = expanded >> await g.pagerank(expanded)    # re-rank by centrality
```

Each operation returns the same type. Each result can feed into any other operation. This is the composable search primitive that makes a small number of operations sufficient for complex queries.

### 7.4 Multi-Modal Retrieval Fusion

For queries that benefit from combining modalities, Grover can internally fuse results using Reciprocal Rank Fusion (RRF):

```python
# Behind the scenes, `search` could run multiple retrievers
# and fuse with RRF — or the agent can do it explicitly:
keyword = await g.grep("login")
semantic = await g.search("user authentication")
structural = await g.successors(FileSearchSet.from_paths(["/src/auth.py"]))
fused = keyword | semantic | structural  # union, then rank by overlap
```

The key insight from Sourcegraph's production system: keyword search finds exact references, semantic search finds conceptually related content, and graph retrieval finds structural dependencies. Each retriever is complementary. Together they cover what any single retriever misses.

## 8. The Analyzer Plugin Model

### 8.1 Analyzers as Content Processors

Analyzers take file content in and produce chunks and connections out. They are the bridge between raw content and the knowledge graph. The current code analyzers (Python AST, JavaScript/TypeScript, Go via tree-sitter) become one family among many:

| Analyzer Family | Input | Chunks produced | Connections produced |
|---|---|---|---|
| **Python** | `.py` files | Functions, classes, methods | imports, calls, inherits |
| **JavaScript/TypeScript** | `.js`, `.ts` files | Functions, classes, components | imports, calls, exports |
| **Go** | `.go` files | Functions, types, methods | imports, calls, implements |
| **Markdown** | `.md` files | Headings, sections | references (links), includes |
| **PDF** | `.pdf` files | Pages, sections, tables | references (citations) |
| **Email** | `.eml`, threads | Messages | replies-to, references, involves |
| **Slack** | Conversations | Messages | replies-to, references, involves, reacts-to |
| **Jira** | Tickets | Description, comments | blocks, assigned-to, references, related-to |
| **CSV/JSON** | Structured data | Rows, records | foreign-key relationships |

Each analyzer implements a simple interface:

```python
class Analyzer(Protocol):
    def can_analyze(self, path: str, mime_type: str) -> bool: ...
    def analyze(self, path: str, content: str) -> AnalysisResult: ...

@dataclass
class AnalysisResult:
    chunks: list[ChunkDescriptor]
    connections: list[ConnectionDescriptor]
```

The `BackgroundWorker` calls the appropriate analyzer after a `write()`, then calls `write()` again for each chunk and `mkconn()` for each connection. The VFS doesn't care what kind of content triggered the analysis.

### 8.2 Connection Types as a Vocabulary

Connection types are free-form strings, but a shared vocabulary emerges across analyzers:

| Connection type | Meaning | Used by |
|---|---|---|
| `imports` | Source depends on target | Code analyzers |
| `calls` | Source invokes target | Code analyzers |
| `inherits` | Source extends target | Code analyzers |
| `implements` | Source implements target | Code analyzers |
| `references` | Source mentions/cites target | All analyzers |
| `authored-by` | Source was created by target | Document/ticket analyzers |
| `assigned-to` | Source is assigned to target | Ticket analyzers |
| `blocks` | Source blocks target | Ticket analyzers |
| `replies-to` | Source is a reply to target | Communication analyzers |
| `involves` | Source involves target (person) | Communication analyzers |
| `member-of` | Source belongs to target (group) | People/team analyzers |
| `discussed-in` | Source was discussed in target | Cross-analyzer |
| `contains` | Structural containment (in-memory only) | All analyzers |

The connection type appears in the path: `/src/auth.py/.connections/imports/src/utils.py`. Glob patterns can query by type: `glob("/**/.connections/authored-by/**")` finds all authorship edges.

## 9. Mounts and Integration

### 9.1 Mount Architecture

Following Plan 9's insight that **mounts are dependency injection**, different data sources are mounted at different points in the namespace:

```python
g = Grover()

# Mount a local codebase
g.add_mount("/src", backend="local", root="/path/to/repo")

# Mount a database-backed document store
g.add_mount("/documents", backend="database", connection_string="...")
```

Each mount provides the same `GroverFileSystem` interface. The facade routes operations to the correct backend by path prefix.

### 9.2 Integration Model: Write Your Own Sync, Grover Handles the Rest

Grover does not try to be an API gateway or a universal connector. Instead, it provides a **flexible write API** that any sync pipeline can target. The people building enterprise knowledge bases are already writing ETL pipelines — adding "write the output to Grover" is the easy part:

```python
# Your pipeline already fetches from Jira, Slack, ADO, etc.
# Grover is just the destination.

# Sync Jira tickets
for ticket in jira_client.search("project = PROJ"):
    g.write(f"/jira/{ticket.key}", ticket.description)
    for comment in ticket.comments:
        g.write(f"/jira/{ticket.key}/.chunks/{comment.id}", comment.body)
    g.mkconn(f"/jira/{ticket.key}", "assigned-to", f"/people/{ticket.assignee}")
    if ticket.parent:
        g.mkconn(f"/jira/{ticket.key}", "child-of", f"/jira/{ticket.parent}")

# Sync Slack threads
for msg in slack_client.conversations_history(channel="proj-discussion"):
    g.write(f"/slack/proj-discussion/.chunks/{msg.ts}", msg.text)
    if msg.thread_ts:
        g.mkconn(f"/slack/proj-discussion/.chunks/{msg.ts}",
                 "replies-to",
                 f"/slack/proj-discussion/.chunks/{msg.thread_ts}")

# Sync ADO work items
for item in ado_client.get_work_items(project="MyProject"):
    g.write(f"/ado/{item.id}", item.description)
    for link in item.relations:
        g.mkconn(f"/ado/{item.id}", link.type, f"/ado/{link.target_id}")
```

Once content is written to Grover, the full pipeline activates automatically:
1. The `BackgroundWorker` runs the appropriate analyzer
2. Chunks are extracted (if not already provided)
3. Connections are added to the knowledge graph
4. Content is embedded for semantic search
5. Versions are tracked

**Grover doesn't need to know about Jira's API, Slack's pagination, or ADO's authentication.** It just needs content at a path. The sync pipeline is the user's responsibility — and that's the right boundary, because every enterprise has different sources, different schemas, different sync requirements.

This means the integration surface is just the standard filesystem operations:

| What you're syncing | Grover operations used |
|---|---|
| Documents/tickets/pages | `write()` to create files |
| Comments/messages/sections | `write()` to create chunks under the parent |
| Relationships (blocks, assigned-to, references) | `mkconn()` to create connections |
| Deletions/archival | `delete()` to remove |
| Updates | `write()` with `overwrite=True` or `edit()` for patches |

No special ingest API. No bulk import format. No connector framework to learn. Just `write`, `mkconn`, and `delete` — the same operations an agent uses interactively.

### 9.3 The `.api/` Directory: Data Plane and Control Plane

For external systems, the namespace splits into two planes within the same tree — **synced data** (searchable, in the graph) and **live API** (real-time pass-through):

```
/jira/
├── .api/                              # control plane (live API)
│   ├── search                         # write JQL, read results
│   ├── ticket                         # read = schema, write = create
│   ├── comment                        # read = schema, write = create
│   └── transition                     # read = schema, write = transition
├── PROJ-4521                          # data plane (synced, searchable)
│   ├── .chunks/
│   │   ├── comment-001
│   │   └── comment-002
│   └── .connections/
│       ├── assigned-to/
│       │   └── people/jane-smith
│       └── blocks/
│           └── jira/PROJ-4519
├── PROJ-4522                          # data plane (synced, searchable)
└── PROJ-4523                          # data plane (synced, searchable)
```

This is Plan 9's architecture. In Plan 9, `/net/tcp/` has a `clone` control file (open to create connections) alongside numbered data directories (existing connections). Control and data coexist in the same namespace but have different semantics:

| | Data plane (synced files) | Control plane (`.api/`) |
|---|---|---|
| Storage | Local DB (`grover_objects`) | Pass-through to external API |
| Searchable | Yes (grep, glob, semantic) | No (paths are listable, not indexed) |
| In knowledge graph | Yes (connections, traversal) | No |
| Embeddable | Yes (vectors) | No |
| Versioned | Yes | No |
| Latency | Fast (local query) | Slow (API call) |
| Freshness | Sync interval | Real-time |
| Offline | Yes | No |

**Discovery is just `ls` and `read`:**

```bash
$ grover ls /jira/.api/
search
ticket
comment
transition

$ grover read /jira/.api/ticket
Create a Jira ticket.

Required:
  --project      Project key (e.g., PROJ)
  --summary      Ticket title
  --type         bug | story | task | epic

Optional:
  --description  Body (markdown)
  --assignee     Username
  --priority     low | medium | high | critical
  --labels       Comma-separated

$ grover ls /slack/.api/
post
search
react

$ grover read /slack/.api/post
Post a message.

Required:
  --channel      Channel name
  --message      Message text (markdown)

Optional:
  --thread       Thread timestamp (reply to existing)
```

No `--help` flag needed. `read` on an `.api/` path IS the schema. `ls` on `.api/` IS the discovery. It's files all the way down.

**Write-back through `.api/`:**

```bash
# Create a Jira ticket (control plane → live API)
$ grover write /jira/.api/ticket --project PROJ --summary "Fix auth timeout" --type bug
/jira/PROJ-4524

# The result is immediately available in the data plane:
# /jira/PROJ-4524 is created as a synced file in the DB,
# searchable, connectable, in the graph

# Post a Slack message
$ grover write /slack/.api/post --channel proj-discussion \
    --message "Filed PROJ-4524 for the auth timeout"
/slack/proj-discussion/.chunks/msg-1710934200

# Create a GitHub issue
$ grover write /github/.api/issue --title "Fix auth timeout" --body "See PROJ-4524"
/github/issues/247

# Live search via API (bypasses local data, hits Jira directly)
$ grover write /jira/.api/search --jql "project = PROJ AND status = Open"
/jira/PROJ-4521
/jira/PROJ-4524
```

**Contrast with the data plane:**

```bash
# Data plane — fast, regex, offline, full graph/search
grover grep "timeout" /jira/                    # SQL regex on local synced content
grover search "auth timeout issues"             # vector search on embeddings
grover pred /jira/PROJ-4521                    # graph traversal across mounts
grover glob "/jira/PROJ-*"                     # pattern match on synced paths

# Control plane — real-time, API-native, limited
grover write /jira/.api/search --jql "..."     # live JQL query
grover write /jira/.api/ticket --summary "..." # create ticket via API
grover read /jira/.api/ticket                  # read the create schema
```

**The `.api/` directory is entirely optional.** A sync-only mount has no `.api/` — it's pure data plane. The sync pipeline writes files using `g.write()`, and the agent searches them with `grep`/`search`/`glob`. The `.api/` directory is added by backend plugins that want to expose live interaction, or even manually by users who want to expose custom operations.

This means `.api/` directories can represent any external interface:

```
/jira/.api/ticket          # Jira REST API
/slack/.api/post           # Slack Web API
/github/.api/issue         # GitHub REST API
/internal/.api/deploy      # Your custom deployment script
/db/.api/query             # A SQL query endpoint
```

Each is just a path. `read` returns the schema. `write` triggers the action. The filesystem is the universal API surface.

### 9.4 Integration Responsibility

| Grover's job | User's sync pipeline / backend plugin's job |
|---|---|
| Store content in unified namespace | Fetch data from external systems |
| Build the knowledge graph (connections) | Map external relationships to connections |
| Index content for search (embeddings) | Handle authentication, pagination, rate limits |
| Version changes automatically | Determine sync frequency and scope |
| Enforce permissions (ReBAC) | Map external permissions to Grover shares |
| Compose results across mounts | Handle service-specific error recovery |
| Provide uniform read/search/graph interface | Provide `.api/` schemas and API translation |

### 9.5 Cross-Mount Connections

Connections can span mounts. `/jira/PROJ-4521/.connections/references/src/auth.py` creates an edge from a Jira ticket to a source file. The graph is global — `predecessors("/src/auth.py")` returns results from any mount.

This is how enterprise knowledge becomes navigable. The agent doesn't need separate tools for "search Jira", "search Slack", "search code". It searches one namespace. Connections between systems — which today exist only in people's heads — become explicit, queryable edges in the graph.

### 9.6 End-to-End Agent Workflow

A single agent with a single Grover tool — searching synced data, traversing the graph, and taking actions via `.api/`:

```bash
# 1. Search synced knowledge (data plane — fast, local)
grover search "authentication timeout"

# 2. Read the code and related tickets
grover read /src/auth.py
grover grep "timeout" /jira/

# 3. Trace the relationship chain across systems
grover pred /src/auth.py
# /jira/PROJ-4521 (references this file)
# /slack/proj-discussion/.chunks/msg-001 (discusses this file)

# 4. Fix the code (data plane — local file)
grover edit /src/auth.py "timeout=30" "timeout=120"

# 5. Create a ticket (control plane — live API)
grover write /jira/.api/ticket --project PROJ \
    --summary "Fix auth timeout" --type bug
# returns: /jira/PROJ-4524 (synced back to data plane)

# 6. Connect the ticket to the code
grover mkconn /jira/PROJ-4524 references /src/auth.py

# 7. Notify the team (control plane — live API)
grover write /slack/.api/post --channel proj-discussion \
    --message "Filed PROJ-4524 and pushed a fix for the auth timeout"

# 8. The full chain is now navigable:
grover desc /src/auth.py
# /jira/PROJ-4524 (references this file)
# /jira/PROJ-4521 (references this file)
# /slack/proj-discussion/.chunks/msg-001 (discusses this file)
```

One tool. One namespace. The agent reads and searches the data plane (synced, fast, offline-capable) and takes actions through the control plane (`.api/`, live, real-time). Same verbs everywhere: `read`, `write`, `grep`, `search`, `pred`, `mkconn`.

## 10. Competitive Positioning

### 10.1 The Landscape

No existing system combines all of Grover's capabilities:

| Capability | Elasticsearch | Glean | Pinecone | Microsoft Graph | SharePoint | Neo4j | AgentFS | **Grover** |
|---|---|---|---|---|---|---|---|---|
| Virtual filesystem | No | No | No | No | Partial | No | Yes | **Yes** |
| Knowledge graph | No | Yes | No | Yes | No | Yes | No | **Yes** |
| Semantic search | Yes | Yes | Yes | Yes | Weak | Plugin | No | **Yes** |
| Versioning | No | No | No | Partial | Yes | No | Partial | **Yes** |
| Multi-tenant | Partial | Yes | Partial | Yes | Yes | No | No | **Yes** |
| Embeddable library | Partial | No | Yes | No | No | Yes | Yes | **Yes** |
| Writable | Yes | No | Yes | Partial | Yes | Yes | Yes | **Yes** |
| Composable search | Partial | No | No | No | No | Yes (Cypher) | No | **Yes** |
| Single-tool MCP | No | No | No | No | No | No | No | **Yes** |
| Write-back to services | No | No | No | Partial | Partial | No | No | **Yes** |
| Pluggable analyzers | No | Proprietary | No | Proprietary | No | No | No | **Yes** |

### 10.2 The Unique Position

**AgentFS** (Turso) is the closest — a SQLite-backed VFS for AI agents. But it has no graph, no semantic search, no analyzers, no versioning. It's a filesystem. Grover is a **knowledge filesystem** — the filesystem is the interface, but the value is in the graph, search, and analysis layers that operate on the content.

**Glean** and **Microsoft Graph** have the enterprise search capabilities but are cloud services, not embeddable libraries. You can't `pip install` them and run them inside your own agent.

**Elasticsearch** and **Pinecone** are search infrastructure — they store vectors and match queries, but they don't understand relationships, don't version content, and don't present results as a navigable namespace.

**Neo4j** has the graph but not the filesystem, not the search, not the versioning. And its query language (Cypher) requires specialized knowledge — it's the opposite of a universal interface.

### 10.3 The One-Line Pitch

*"A virtual filesystem for knowledge — mount any data source, and everything about it appears as navigable paths that any agent can explore with `read`, `write`, `ls`, and `grep`."*

Or more concisely: **Knowledge as a filesystem.**

## 11. Architecture Summary

### 11.0 Design Principles

1. **Virtual overlay namespace** — paths are logical, not physical. The namespace is the interface; storage is an implementation detail.
2. **Kinded object model** — every entity has a path, a kind, and a parent. The kind determines operation semantics.
3. **Files-first defaults** — queries return files by default. Metadata is accessible but opt-in.
4. **Metadata as opt-in** — chunks, versions, and connections are always reachable by explicit path but hidden from broad queries.
5. **Small universal API** — `read`, `write`, `delete`, `list`, `stat`, `glob`, `grep`, `search`, `mkconn` + graph traversal. That's the whole interface.
6. **CLI/MCP as the primary agent surface** — one tool, progressive discovery, Unix pipes for composition.

### 11.1 What Stays

- **Ref** — the immutable path wrapper (simplified: no sigil parsing, just segment matching)
- **GroverResult / FileSearchResult** — the composable result types with set algebra
- **BackgroundWorker** — debounced async processing after writes
- **RustworkxGraph** — the in-memory graph provider, loaded from DB on mount
- **Search providers** — pluggable vector stores (local, Pinecone, Databricks)
- **Embedding providers** — pluggable embedding (OpenAI, LangChain)
- **Content-before-commit** — the write ordering invariant
- **Sessions owned by VFS** — backends never create/commit/close sessions
- **Mount + MountRegistry** — routing by path prefix
- **GroverAsync + Grover sync wrapper** — the facade with mixins
- **SupportsReBAC / SupportsReconcile** — opt-in capability protocols

### 11.2 What Changes

| Current | Proposed |
|---|---|
| 4 tables (files, chunks, versions, connections) | Unified kinded object model (likely 1 table, but implementation choice) + shares |
| Type-specific methods (add_connection, replace_file_chunks, ...) | Uniform operations (read, write, delete, list, stat) |
| Sigil-based paths (`#`, `@`, `[]`) | Hierarchy-based paths (`.chunks/`, `.versions/`, `.connections/`) |
| `Ref` with regex/sigil parsing | `Ref` with simple segment matching |
| `Ref.transform()` dispatching to 4 types | Unnecessary — backend dispatches by kind |
| ChunkService, ConnectionService, VersionProvider | Unified in DatabaseFileSystem CRUD |
| Separate graph loading (from_sql on connection table) | Same table, `WHERE kind = 'connection'` |
| Code-only analyzers | Pluggable analyzer families (code, documents, communications, tickets) |
| Python API only | CLI + MCP single-tool + Python API |

### 11.3 What's New

- **CLI** — filesystem commands that compose via Unix pipes
- **MCP single-tool interface** — one tool, progressive discovery via `--help`
- **`mkconn`** — connection creation primitive (like `mkdir`)
- **Kind-based dispatching** — `read()`, `write()`, `delete()` adapt behavior by entity kind
- **Files-first defaults** — `ls`, `glob`, `grep`, `search` return files by default; metadata is opt-in (`-a`, `--chunks`, `--kinds`); `.api/` is never in search results
- **Non-code analyzers** — PDF, Markdown, email, Slack, Jira, CSV/JSON
- **Sync-first integration model** — users write their own sync pipelines using `write()` / `mkconn()` / `delete()` — no connector framework, no special ingest API
- **`.api/` directories** — data plane (synced, searchable) and control plane (live API pass-through) coexist in the same namespace. `ls .api/` for discovery, `read .api/ticket` for schema, `write .api/ticket` for action
- **Optional backend plugins** — for deeper integration (`.api/` endpoints, write-back), third parties implement `GroverFileSystem`

## 12. Key Design Decisions

### 12.1 Paths are logical, not physical (virtual overlay)

**Decision:** The namespace is a virtual overlay. Paths are logical addresses, not filesystem locations. File content may live on disk (local mounts) or in the database (DB mounts). Metadata nodes (`.chunks/`, `.versions/`, `.connections/`, `.api/`) always live in the database, never as physical files on disk.

**Rationale:** Grover's current `LocalFileSystem` stores real files on disk. A real file cannot literally have `/.chunks/` children on the physical filesystem. The solution is explicit: the namespace is virtual, and the backend determines where bytes live. For local mounts, `read("/src/auth.py")` reads from disk; `read("/src/auth.py/.chunks/login")` reads from SQLite. Both look identical to the agent. This is the same model as Linux's VFS layer — one namespace, multiple underlying storage systems.

### 12.2 `parent_path` is stored metadata, not derived from path

**Decision:** `parent_path` is computed at write time using marker-aware parsing (`/.chunks/`, `/.versions/`, `/.connections/`, `/.api/`) and stored as an indexed column. It is not derived by splitting on `/` and dropping the last segment.

**Rationale:** For files and directories, the filesystem parent and the logical parent are the same. For metadata nodes, they diverge. The parent of `/src/auth.py/.chunks/login` is `/src/auth.py`, not `/src/auth.py/.chunks`. The parent of `/src/auth.py/.connections/imports/src/utils.py` is `/src/auth.py`, not `/src/auth.py/.connections/imports/src`. Connection target paths can contain `/`, making naive path splitting ambiguous. Storing `parent_path` explicitly avoids this entirely and enables efficient tree queries via index.

### 12.3 Files-first visibility defaults

**Decision:** All query operations (`ls`, `glob`, `grep`, `search`, `tree`) default to returning files and directories only. Metadata nodes (chunks, versions, connections) require explicit opt-in (`-a`, `--chunks`, `--kinds`). `.api/` nodes are never in search results. Direct reads of any path always work regardless of defaults.

**Rationale:** Without this, an agent doing `grep "timeout"` would get every chunk (function, class, section) that matches — potentially hundreds of results for a handful of relevant files. The default must be useful without configuration. Files are the primary abstraction; metadata is supporting detail. This mirrors Unix: `ls` hides dotfiles by default, `find` skips hidden directories by default. The agent can always opt in when it needs metadata depth.

### 12.4 Dot-prefix for metadata directories (unchanged from prior)

**Decision:** Use `.chunks`, `.versions`, `.connections` (dot-prefixed).

**Rationale:** Plan 9 does not use dot-prefix hidden files — it uses dedicated directories and kernel device prefixes. But Grover is not Plan 9. The dot-prefix convention is deeply embedded in Unix culture (`.git`, `.ssh`, `.config`) and in LLM training data. It provides a natural "show/hide" toggle (`ls` vs `ls -a`). It prevents collision with user content. It signals "this is metadata" to anyone who understands Unix conventions.

### 12.5 Connections live under the source file

**Decision:** `/src/auth.py/.connections/imports/src/utils.py` — connections are children of the source.

**Rationale:** An edge has to live somewhere in a tree namespace. The source file is the natural owner because: (1) analyzers produce connections by analyzing the source file's content, (2) `delete("/src/auth.py")` should cascade to its outgoing connections, (3) `ls -a /src/auth.py/.connections/` answers "what does this file depend on?" which is the most common question. Incoming connections are found via `predecessors()` graph traversal, not namespace navigation.

### 12.6 Versions are read-only

**Decision:** `write("/src/auth.py/.versions/3")` returns an error. Versions are created as a side effect of `write("/src/auth.py")`.

**Rationale:** Versions are like `/proc/42/status` — generated by the system, not written by the user. They are an audit trail of what the file looked like at a point in time. Allowing writes to versions would create confusion about what "the current content" is and undermine the versioning guarantee.

### 12.7 `kind` column vs. path inference

**Decision:** Store `kind` as an explicit column, not derived from path format.

**Rationale:** An explicit kind column enables efficient queries (`WHERE kind = 'connection'` for graph loading), is self-documenting, and survives potential future path format changes. The model validator ensures path format and kind agree.

### 12.8 Nullable kind-specific columns vs. JSON metadata

**Decision:** Keep kind-specific fields as real columns (source_path, target_path, line_start, etc.), not JSON.

**Rationale:** `source_path` and `target_path` need indexes for graph traversal. `line_start`/`line_end` are useful for chunk queries. JSON metadata is harder to index, harder to query, and harder to validate. The trade-off is null columns for most rows, but SQLite and Postgres handle sparse columns efficiently. Note: this is an implementation choice, not a core design constraint — the namespace and API design work regardless of the storage schema.

### 12.9 Sync pipelines as the primary integration model

**Decision:** External data enters Grover through user-written sync pipelines that call `write()` / `mkconn()` / `delete()`. Backend plugins with write-back are optional, not required.

**Rationale:** Every enterprise has different data sources, different schemas, different sync requirements. Building a connector framework (like Glean or Airbyte) is a massive scope expansion that delays the core value. Instead, Grover's standard filesystem operations ARE the integration API. Users already write ETL pipelines — targeting Grover is just `g.write(path, content)` at the end. This keeps Grover focused on what it's good at (namespace, graph, search, versioning) and lets users own the data ingestion, which they need to customize anyway. Backend plugins remain available for teams that want deeper integration (write-back, live schema discovery), but they're a convenience layer, not a prerequisite.

### 12.10 `.api/` directories for live API interaction

**Decision:** External service APIs are exposed as `.api/` directories within the mount namespace. `ls` discovers endpoints, `read` returns schemas, `write` triggers actions. Synced data and live API coexist in the same tree.

**Rationale:** This is Plan 9's data/control separation applied to external services. Plan 9's `/net/tcp/` has `clone` (control — open to create connections) alongside numbered directories (data — existing connections). The same namespace, different semantics. For Grover, `.api/` paths are the control plane — they don't store content, aren't searchable, and aren't in the graph. They're pass-through to the live API. The synced files alongside them are the data plane — local, searchable, in the graph. This separation means: (1) schema discovery is just `read` on a path, not a special `--help` mechanism, (2) agents discover APIs the same way they discover files — by navigating, (3) the `.api/` directory is optional — mounts without it are pure data plane, (4) `.api/` paths can represent any external interface (REST APIs, deployment scripts, SQL endpoints), and (5) the cost is ~0 tokens upfront because schemas are loaded on demand via `read`.

### 12.11 Shares as a separate table

**Decision:** `grover_shares` remains its own table, not merged into `grover_objects`.

**Rationale:** Shares are ACLs — they describe access control, not entities in the namespace. A share on `/documents/report.pdf` grants access to that path and its children. The share itself is not addressable at a path, not searchable, not versioned. It's metadata about the namespace, not part of it. This matches Unix: file permissions are stored in the inode, not as entries in the directory.

## 13. References

### Academic Papers
- *"From 'Everything is a File' to 'Files Are All You Need'"* (arXiv:2601.11672) — filesystem abstraction as universal agent interface
- *"Everything is Context"* (arXiv:2512.05470) — file-system abstraction for context engineering (AIGNE framework)
- *"A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces"* (arXiv:2602.03442) — composable multi-modal retrieval
- *"RAG-MCP"* (arXiv:2505.03275) — tool selection accuracy degrades at scale, RAG-based retrieval restores it
- *"Code-Craft: Hierarchical Graph-Based Code Summarization"* (arXiv:2504.08975) — dependency-aware code retrieval
- *"cAST: Structural Chunking via Abstract Syntax Tree"* (arXiv:2506.15655) — AST-aware chunking

### Industry Sources
- Anthropic, *"Code Execution with MCP"* — filesystem-based tool discovery, 98.7% token reduction
- Anthropic, *"Writing Effective Tools for AI Agents"* — tool design principles
- Anthropic, *"Effective Context Engineering for AI Agents"* — context management strategies
- Boris Cherny (Claude Code creator) — "agentic search is a fancy word for glob and grep"
- Sourcegraph, *"Lessons from Building AI Coding Assistants"* — hybrid retrieval with Repo-level Semantic Graph
- Turso, *"AgentFS"* — SQLite-backed VFS for AI agents
- LangChain, *"How Agents Can Use Filesystems for Context Engineering"*
- Block, *"Block's Playbook for Designing MCP Servers"* — design from workflows, not endpoints
- Apideck, *"Your MCP Server Is Eating Your Context Window"* — CLI alternative to MCP tools

### Systems
- Plan 9 from Bell Labs — 9P protocol, per-process namespaces, synthetic filesystems
- llm9p — Plan 9 protocol for LLM interaction
- Wanix — Plan 9's spirit in WebAssembly
- AIOS — LLM-based Semantic File System
- Redox OS — "everything is a URL"

## 14. Decisions & Progress

### 14.1 `Ref` deferred

**Decision:** `Ref` (§4.2) will not be implemented in the initial build. Operations accept and return plain `str` paths. `Ref` can be introduced later as an ergonomic wrapper if needed.

**Rationale:** The path utilities in `paths.py` already provide everything `Ref` would — `parse_kind()`, `base_path()`, `parent_path()`, `decompose_connection()`, and the path constructors. A frozen dataclass wrapping a string adds a layer of indirection without enabling anything new. If a uniform handle becomes valuable (e.g., for caching parsed properties or for type-safe API boundaries), it can be added without changing the data model or protocol — it's a presentation concern, not a storage or dispatch concern.

### 14.2 Initial scaffolding — 2026-03-23

**Commit:** `3f0dd20` — *Add design docs for Grover v2 rewrite*

| File | What it implements |
|------|-------------------|
| `paths.py` | §3 — Path normalization, validation, kind detection, parent/base resolution, path constructors (`chunk_path`, `version_path`, `connection_path`, `api_path`), `decompose_connection` |
| `models.py` | §4 — `ValidatedSQLModel` base, `GroverObjectBase` with all kinded columns, `GroverObject` concrete table (`grover_objects`) with auto-derived `parent_path`, `kind`, `name`, content metrics, timestamps |
| `vector.py` | Embedding column — `Vector` type with dimension/model-name enforcement, `VectorType` SQLAlchemy decorator (JSON serialization, dimension validation on read/write) |

### 14.3 Composable result types + protocol — 2026-03-23

**Commits:** `f04444b` through `47459d3`

#### Result types (`results.py`)

One result type for everything. Every Grover operation returns `GroverResult`. CRUD returns it with one candidate, queries with many, graph ops with re-ranked/expanded candidates.

- **`Detail`** — flat provenance record. Fields: `operation`, `score`, `success`, `message`, `metadata: dict`. No subclasses. Frozen.
- **`Candidate`** — read-only projection of a `GroverObject`. Only `path` is required; `id`, `kind`, `lines`, `size_bytes`, `tokens` are optional (defaulting to `None`). `name` is a computed property via `split_path()`. `details` is `tuple[Detail, ...]` for true immutability. `score` property returns last non-null detail score. `score_for(operation)` looks up a specific operation's score.
- **`GroverResult`** — carries `success`, `message`, `candidates`. `_grover` back-reference (Pydantic `PrivateAttr`, excluded from JSON) enables method chaining. Set algebra (`&`, `|`, `-`). Enrichment chains (`sort`, `top`, `filter`, `kinds`). CRUD/query/graph chain stubs delegate to facade.

#### Protocol (`protocol.py`)

`GroverFileSystem` — the narrow waist. Every backend implements it.

**Chainable CRUD** (accept `path` or `candidates`, one batched query):
`read`, `stat`, `edit`, `ls`, `delete`

**Path-only CRUD** (no chain use case today):
`write` (with `overwrite`), `move`, `copy`, `mkdir`, `mkconn`

**Search** — three explicit methods:
`semantic_search`, `vector_search`, `lexical_search` (all with `k=15`)

**Query:**
`glob`, `grep` (with `case_sensitive`, `max_results`), `tree`

**Graph** — traversal accepts `path` or `candidates`:
`predecessors`, `successors`, `ancestors`, `descendants`, `neighborhood`

**Graph** — set operations require `candidates`:
`meeting_subgraph`, `min_meeting_subgraph`

**Graph** — centrality/ranking (optional `candidates`):
`pagerank`, `betweenness_centrality`, `closeness_centrality`, `degree_centrality`, `in_degree_centrality`, `out_degree_centrality`, `hits`

All methods: `*, session: AsyncSession | None = None` keyword-only.

#### Key design decisions made

**No `operations` tracking on `GroverResult`.** Provenance lives on `Detail` per candidate, not as hidden state on the result. This follows the Unix principle: data is data, the pipeline is the program. `sort()` defaults to `candidate.score` (last detail score), or accepts explicit `operation=` or `key=`.

**No `Ref` type.** Plain `str` paths everywhere. `paths.py` provides all parsing utilities.

**No `exists` method.** `stat` covers the use case.

**No `open`/`close` lifecycle on protocol.** Session lifecycle owned by facade, not backend.

**`ls` not `list`.** Avoids shadowing Python's `list` builtin, which Pydantic needs to resolve `list[Candidate]` annotations.

**Connection fields removed from `Candidate`.** `source_path`, `target_path`, `connection_type` are derivable from the path via `decompose_connection()`. Connection metadata goes in `Detail.metadata`. `weight` and `distance` kept as graph metrics.

**`Candidate.details` is `tuple[Detail, ...]`** not `list[Detail]`. Frozen model + immutable container = true immutability.

**Merge uses `_first_set` (None-coalescing, not falsy-coalescing).** `lines=0`, `content=""`, `score=0.0` are valid values that won't be replaced by the other candidate's value in set algebra.

**`top(k)` raises `ValueError` for `k < 1`.** No silent empty results.

#### Directory restructure

Old v1 code archived to `src_old/` and `tests_old/`. New v2 code promoted to `src/` and `tests/`. `pyproject.toml` updated to exclude old dirs from lint/test/coverage.

#### Tests

80 tests covering: Detail/Candidate/GroverResult construction, frozen model enforcement, JSON serialization round-trips (`model_dump`, `model_dump_json`), set algebra (intersection, union, difference, detail merging, success propagation, `_grover` propagation), enrichment chains (sort by score/operation/key, top, filter, kinds), chain stubs without bound grover, merge edge cases (zero metrics, empty content, left id preservation, None fallback), `_first_set` direct tests, required field validation, datetime round-trip, duplicate path behavior.

### 14.4 Concrete base class with mount routing — 2026-03-23

**Commits:** `670a2b3` through `9db89e3`

Converted `GroverFileSystem` from a Protocol (interface only) to a **concrete async base class** that owns mount routing, session management, and path rebasing. Subclasses override `_*_impl` methods for storage.

#### `base.py` — GroverFileSystem

Public methods are **routers** — they resolve the terminal filesystem via longest-prefix mount matching, delegate to `_*_impl` methods, then rebase paths before returning.

**Mount management:**
- `add_mount(path, filesystem)` / `remove_mount(path)` — path validation (normalized, no root mount, no duplicates)
- `_match_mount(path)` — longest-prefix match with boundary check (`/web` doesn't match `/webinar`)
- `_resolve_terminal(path)` — walks the mount chain to find the terminal filesystem, accumulating prefix

**Three routing strategies:**
- `_route_single(op, path, candidates)` — resolves one path or dispatches candidates by group
- `_route_two_path(op, ops)` — same-mount batch or cross-mount transfer (read → write → delete)
- `_route_fanout(op, candidates)` — queries self + all mounts in parallel, merges results

**Session management:** `_use_session()` async context manager — commits on success, rolls back on error.

**Result helpers:** `_rebase_result()` restores absolute paths, `_exclude_mounted_paths()` prevents shadow results, `_merge_results()` unions multiple results.

**40+ `_*_impl` stubs** — all raise `NotImplementedError` (excluded from coverage). Subclasses override for their storage backend.

#### Key decisions

**`GroverFileSystem` is a class, not a protocol.** The routing, session, and rebasing logic is shared infrastructure — putting it in a protocol would force every backend to duplicate it. Subclasses override only `_*_impl` methods.

**`_grover` back-reference removed from `GroverResult`.** The chaining stubs on GroverResult (`result.predecessors()`, etc.) were removed. Operations are called on the filesystem, not on results. Pydantic `PrivateAttr` for back-references introduced serialization and testing complexity.

**`Candidate` model relaxed.** Only `path` is required. `id`, `kind`, `lines`, `size_bytes`, `tokens` are all optional (defaulting to `None`). This lets graph and routing code construct lightweight candidates without needing database metadata. Fields are hydrated from the database when needed.

### 14.5 Graph subpackage — 2026-03-23/24

**Commits:** `6766923`, `b0fab6c`

Created `src/grover/graph/` subpackage implementing the in-memory knowledge graph (§11.1 "RustworkxGraph").

#### `graph/protocol.py` — GraphProvider

Runtime-checkable Protocol defining the graph interface. All methods are async. Mutations require `session: AsyncSession` for TTL-based freshness checks. Query methods accept `GroverResult` as candidates input for composability.

**Mutations:** `add_node`, `remove_node`, `has_node`, `add_edge`, `remove_edge`, `has_edge`, `nodes` property
**Traversal:** `predecessors`, `successors`, `ancestors`, `descendants`, `neighborhood`
**Subgraph:** `meeting_subgraph`, `min_meeting_subgraph`
**Centrality:** `pagerank`, `betweenness_centrality`, `closeness_centrality`, `degree_centrality`, `in_degree_centrality`, `out_degree_centrality`, `hits`
**Persistence:** `ensure_fresh(session)`

#### `graph/rustworkx.py` — RustworkxGraph

Sole implementation of `GraphProvider`. Stores topology as adjacency dicts (`_out`, `_in`, `_edge_types`) — no per-edge tuple allocation.

**Constructor:** Takes `model: type[GroverObjectBase]` (the mount's concrete table class) and optional `ttl` (default 1 hour).

**Persistence:** `ensure_fresh()` checks `_loaded_at` against TTL (~117ns hot path). `_load()` queries `GroverObject` rows where `kind="connection"`, using `sqlmodel.select` for typed queries. Build-then-swap: assembles new state in local variables, assigns atomically.

**Snapshot isolation:** `_snapshot()` returns `(frozenset, dict[str, frozenset])` for thread-safe reads via `asyncio.to_thread()`.

**Result helpers:**
- `_relationship_candidates()` — builds candidates from `{path: [related_paths]}` with `Detail(operation=..., metadata={"paths": [...]})`
- `_subgraph_candidates()` — builds node + connection candidates (uses `connection_path()` for edge paths)
- `_score_candidates()` — builds scored candidates from `{path: score}`, sorted descending

**Light reads:** `predecessors()` and `successors()` run inline (no thread overhead), wrapped in try/except returning `GroverResult(success=False)` on error.

**`UnionFind`** — path-compressed union-find with rank balancing, used by `meeting_subgraph` (to be implemented).

#### Key decisions

**Adjacency only in memory.** `_edge_types` stores `dict[tuple[str, str], str]` — single connection type per edge pair. Known limitation: can't represent multiple edge types between the same source/target. Acceptable for now.

**All mutations are async.** Every mutation calls `await self.ensure_fresh(session)` before operating. This ensures the graph is loaded before the first mutation and stays fresh within TTL.

**TTL-based freshness.** `_loaded_at` timestamp + `_ttl` checked via `time.monotonic()`. Default 1 hour. Avoids re-querying the database on every operation.

**Model as constructor parameter.** `RustworkxGraph(model=GroverObject)` — each mount can have a different concrete table class. The `_load()` method uses `select(self._model).where(self._model.kind == "connection")`.

**Heavy algorithms not yet migrated.** `ancestors`, `descendants`, `neighborhood`, `meeting_subgraph`, `min_meeting_subgraph`, and all centrality methods exist as protocol stubs but are not yet implemented in `RustworkxGraph`. The proven implementations exist in `src_old/grover/providers/graph/rustworkx.py` (~1098 lines) and will be migrated as needed.

**Base class wiring not yet done.** The `_*_impl` stubs in `base.py` for graph operations still raise `NotImplementedError`. Step 10 of the migration plan (adding `self._graph` attribute and delegation) is pending.

### 14.6 Test coverage — 2026-03-24

**Commit:** `6fc1ff2`

Raised test coverage from **68% to 99%** (921 statements, 8 missed). Tests went from 308 passing + 3 failing → **462 passing, 0 failing**.

| File | Before | After |
|------|--------|-------|
| `base.py` | 69% | 99% |
| `graph/__init__.py` | 0% | 100% |
| `graph/protocol.py` | 0% | 100% |
| `graph/rustworkx.py` | 0% | 99% |
| `models.py` | 97% | 97% |
| `paths.py` | 99% | 99% |
| `results.py` | 100% | 100% |
| `vector.py` | 100% | 100% |

**New test files:**
- `tests/test_graph.py` — 60+ tests: UnionFind, RustworkxGraph mutations, persistence (mocked session), snapshot/build, result helpers, predecessors/successors, error handling
- `tests/test_base.py` — 86 tests: constructor validation, session management, result helpers, candidate dispatch, all three routing strategies, cross-mount transfer, all public method routing
- `tests/conftest.py` — shared helpers: `DummySession`, `dummy_session_factory`, `tracking_session_factory`, `candidate()`, `make_fs()`

**Fixes:**
- 3 failing tests in `test_results.py` updated for relaxed `Candidate` model (id/kind now optional)
- `test_routing.py` refactored to use shared conftest helpers

**8 uncovered lines** (all defensive/internal): SQLModel ORM lifecycle guards (models.py 47, 49, 150), unreachable-after-normalization guards (paths.py 170, 225; base.py 81-82), rare UnionFind rank branch (rustworkx.py 58).


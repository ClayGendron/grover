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

This was not an afterthought. Dennis Ritchie's own account (*"The Evolution of the Unix Time-sharing System"*, AT&T Bell Labs Technical Journal, 1984) reveals that the filesystem was the **first component designed**, predating even the process model. Thompson, Canaday, and Ritchie developed the basic filesystem structure; Ritchie contributed the idea of *device files* — the insight that I/O devices should appear as files in the same namespace. The i-list (a linear array of inodes, each describing a file with "a short, unambiguous name related in a simple way to the protection, addressing, and other information needed to access the file") was the foundational data structure. This separated **identity** (inode number) from **naming** (directory entries) — a distinction that remains central to every filesystem design, including Grover's.

The Unix architecture established a three-level I/O model that is still widely misunderstood: **file descriptors** (per-process integers) → **open file descriptions** (kernel-wide structures holding the offset and flags, shared by `dup()` and `fork()`) → **inodes** (the actual file objects). A filename is not a file — it is a pointer in a parent directory to an inode. This is what enables hard links, what makes `unlink()` not always delete data, and what makes `rename()` atomic (it's a directory entry update, not a data copy). Grover's architecture maps cleanly to this model: the agent's path string (the name) → the session/operation context (the open description) → the `grover_objects` row (the inode).

Plan 9 from Bell Labs (1992) took this further. The network became a filesystem (`/net/tcp/clone`). The window system became a filesystem (`/dev/draw`). Process control became a filesystem (`/proc/42/ctl`). The entire protocol had **14 message types**: walk, open, read, write, stat, create, remove, clunk, and a handful of others. That was sufficient to model every resource in a distributed operating system.

Rob Pike et al. (*"The Use of Name Spaces in Plan 9"*, 5th ACM SIGOPS European Workshop, 1992) articulated why this works: "The integration of devices into the hierarchical file system was the best idea in UNIX... Plan 9 pushes the concepts much further and shows that file systems, when used inventively, have plenty of scope for productive research." Plan 9's key innovations — per-process mutable namespaces via `bind()`/`mount()`, union directories (multiple trees overlaid at one point), and the 9P wire protocol — demonstrated that a small set of file operations is sufficient to model every resource in a distributed system. Grover's mount architecture (§9), cross-mount search (§9.5), and single-tool CLI (§6.2) are direct descendants of these ideas.

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
| **Version** | `<file>/.versions/<N>` | `/src/auth.py/.versions/3` (cf. ext3cow's `@epoch` syntax) |
| **Connection** | `<file>/.connections/<type>/<target>` | `/src/auth.py/.connections/imports/src/utils.py` |
| **API endpoint** | `<mount>/.api/<action>` | `/jira/.api/ticket` |

This design has several properties:

- **Standard path parsing.** No special sigils (`#`, `@`, `[]`). Every path is `/`-separated. Standard path libraries work unchanged.
- **Glob works natively.** `glob("/src/**/.chunks/*")` finds all chunks. `glob("/**/.connections/imports/**")` finds all import edges. `glob("/**/.versions/*")` finds all versions.
- **`ls -a` semantics.** `ls /src/auth.py` hides metadata children by default. `ls -a /src/auth.py` reveals `.chunks/`, `.versions/`, `.connections/`. This matches Unix hidden file convention.
- **No collision with user content.** Nobody creates source files or documents called `.chunks`.
- **Prior art in versioning filesystems.** The `.versions/<N>` pattern follows ext3cow's `@epoch` syntax (`/path/to/file@1234567890`), which exposed historical versions as path-addressable entities in a real Linux filesystem. NILFS2 and the Elephant File System used similar approaches. Encoding version access in the path — rather than requiring a separate API — is a proven pattern for making time-travel discoverable and composable with standard tools.

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

This is the same problem Rob Pike solved in *"Lexical File Names in Plan 9, or, Getting Dot-Dot Right."* In Unix, symbolic links turn the namespace from a tree into a directed graph, making `..` ambiguous (physical parent vs. the directory you "came from"). Plan 9 solved this by tracking a `Cname` (canonical name) on each open channel — the absolute pathname used to reach the file. When `..` is evaluated, the kernel lexically strips the last component from the Cname, then validates the result. Grover's metadata paths create an analogous problem: `/.connections/imports/src/utils.py` contains `/` characters that make naive parent derivation ambiguous. Grover's solution — marker-aware parsing stored at write time — is the same insight as Plan 9's Cname: the system records *how you got there* rather than trying to derive it from the path string after the fact.

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

**Path-as-key vs. inode-as-key trade-off.** The schema uses `path TEXT UNIQUE` as the primary lookup key — not the `id` column. This is the same choice made by libsqlfs and other database-backed filesystems. The trade-off is well-understood: path-as-key makes read/stat/exists O(1) on the unique index and requires no joins, but `move()` on a directory with N descendants requires updating N rows (every child's `path` and `parent_path`). An inode-based scheme (id-as-primary-key with a separate path→id mapping table) would make rename O(1) but require joins for every path lookup. For Grover's workload — read-heavy, write-moderate, rename-rare — path-as-key is the right choice. If bulk renames become a bottleneck, the mitigation is batched `UPDATE ... WHERE path LIKE prefix%` within a single transaction, which is already the implementation in `_move_impl`.

### 12.9 Sync pipelines as the primary integration model

**Decision:** External data enters Grover through user-written sync pipelines that call `write()` / `mkconn()` / `delete()`. Backend plugins with write-back are optional, not required.

**Rationale:** Every enterprise has different data sources, different schemas, different sync requirements. Building a connector framework (like Glean or Airbyte) is a massive scope expansion that delays the core value. Instead, Grover's standard filesystem operations ARE the integration API. Users already write ETL pipelines — targeting Grover is just `g.write(path, content)` at the end. This keeps Grover focused on what it's good at (namespace, graph, search, versioning) and lets users own the data ingestion, which they need to customize anyway. Backend plugins remain available for teams that want deeper integration (write-back, live schema discovery), but they're a convenience layer, not a prerequisite.

### 12.10 `.api/` directories for live API interaction

**Decision:** External service APIs are exposed as `.api/` directories within the mount namespace. `ls` discovers endpoints, `read` returns schemas, `write` triggers actions. Synced data and live API coexist in the same tree.

**Rationale:** This is Plan 9's data/control separation applied to external services. Plan 9's `/net/tcp/` has `clone` (control — open to create connections) alongside numbered directories (data — existing connections). The same namespace, different semantics. For Grover, `.api/` paths are the control plane — they don't store content, aren't searchable, and aren't in the graph. They're pass-through to the live API. The synced files alongside them are the data plane — local, searchable, in the graph. This separation means: (1) schema discovery is just `read` on a path, not a special `--help` mechanism, (2) agents discover APIs the same way they discover files — by navigating, (3) the `.api/` directory is optional — mounts without it are pure data plane, (4) `.api/` paths can represent any external interface (REST APIs, deployment scripts, SQL endpoints), and (5) the cost is ~0 tokens upfront because schemas are loaded on demand via `read`.

### 12.11 Shares as a separate table

**Decision:** `grover_shares` remains its own table, not merged into `grover_objects`.

**Rationale:** Shares are ACLs — they describe access control, not entities in the namespace. A share on `/documents/report.pdf` grants access to that path and its children. The share itself is not addressable at a path, not searchable, not versioned. It's metadata about the namespace, not part of it. This matches Unix: file permissions are stored in the inode, not as entries in the directory.

### 12.12 Trailing slash normalization

**Decision:** Trailing slashes are stripped during path normalization. `read("/src/auth.py/")` and `read("/src/auth.py")` are identical. No POSIX-style "trailing slash forces directory resolution" semantics.

**Rationale:** POSIX mandates that a trailing slash forces the preceding component to resolve as a directory — `stat("/tmp/link/")` follows a symlink and fails with `ENOTDIR` if the target is a regular file, while `stat("/tmp/link")` succeeds. This subtlety has caused real bugs (the behavior difference between `unlink("/tmp/link")` and `unlink("/tmp/link/")` is a classic POSIX gotcha). Grover has no symlinks and dispatches by `kind`, not by path syntax. Preserving trailing slash semantics would add complexity without enabling anything — the `kind` column already distinguishes files from directories. Normalizing away trailing slashes eliminates an entire class of ambiguity.

### 12.13 Rename atomicity

**Decision:** `move()` is atomic within a single mount (one SQL transaction). Cross-mount `move()` is a three-step transfer (read → write → delete) that is not atomic to concurrent readers.

**Rationale:** POSIX guarantees that `rename()` is atomic for concurrent observers — at no point does the file appear to not exist. Within a single mount, Grover achieves this because all path updates happen in one SQL transaction (either all rows update or none do). Cross-mount moves cannot be atomic because they involve two independent backends. This mirrors the POSIX constraint that `rename()` cannot cross filesystem boundaries (`EXDEV`). The ext4 delayed-allocation incident of 2008 — where the implicit `write(tmp); rename(tmp, real)` atomicity guarantee was violated, causing data loss for essentially all applications that relied on it — demonstrates that atomicity guarantees must be explicit and documented, not implied. Grover's guarantee: same-mount `move()` is atomic; cross-mount `move()` is best-effort with read-before-delete ordering (content is never lost, but may briefly exist at both paths).

### 12.14 No hard links

**Decision:** Every object has exactly one path and exactly one parent. No hard links.

**Rationale:** Hard links would allow a single object to appear at multiple paths, turning the namespace from a tree into a DAG. This would break: (1) `parent_path` uniqueness — an object with two parents has no single `parent_path`, (2) cascading deletes — deleting one parent path shouldn't delete the object if another link exists, requiring reference counting, (3) the `kind` invariant — the kind is path-derived and must agree with the stored kind, and (4) any algorithm that assumes an acyclic namespace (tree queries, permission inheritance, `ls -R`). Unix itself forbids hard links to directories for exactly this reason — they create cycles that break `find`, `du`, `rm -r`, and `fsck`. Connections serve the use case that hard links might otherwise address: "this object is related to that path." Connections are explicit, typed, directional, and live under the source — they don't compromise namespace invariants.

### 12.15 Durability guarantees

**Decision:** After `write()` returns successfully, the data is durable (survives process crash). The specific guarantee depends on the backend.

**Rationale:** Research on crash consistency (MIT's FSCQ, the ALICE tool) found that "every single piece of software tested except SQLite in one mode had at least one crash bug." Grover builds on SQLite (for local/embedded use) and PostgreSQL (for server deployments) — both provide well-understood durability guarantees when used correctly. For SQLite in WAL mode with `PRAGMA synchronous=FULL` (the default), a committed transaction survives power loss. For PostgreSQL, `synchronous_commit=on` (the default) provides the same guarantee. Grover's "content-before-commit" write ordering (§12.1, FS Write Ordering invariant) ensures that a crash during `write()` never creates phantom metadata (DB says file exists, content is missing). The worst case on crash is an orphan file on disk that is invisible to the system — inert and harmless. This ordering was validated independently: the ext4 delayed-allocation incident of 2008 demonstrated that commit-before-content causes real data loss at scale.

### 12.16 No mount propagation

**Decision:** Mounts are private. Adding or removing a mount in one context has no effect on other contexts. No shared/slave/private/unbindable propagation model.

**Rationale:** Linux introduced mount propagation (shared subtrees) in 2.6.15 to handle how mount events flow between mount namespaces. The four propagation types (shared, slave, private, unbindable), their peer group mechanics, and their interactions form a complex state machine that even kernel developers find subtle — LWN.net devoted a multi-part series to explaining the semantics. systemd's decision to default everything to shared propagation has far-reaching consequences for container isolation. Grover deliberately avoids this complexity. Each `GroverFileSystem` instance has its own mount registry (`MountRegistry`). `add_mount()` and `remove_mount()` affect only that instance. This follows Plan 9's simpler model — per-process namespaces where `bind()` and `mount()` affect only the calling process — rather than Linux's shared subtree model. The trade-off is that coordinating mount state across multiple Grover instances requires explicit application-level logic, but this is the right default for a library where each agent or service typically has its own `Grover()` instance.

## 13. References

### Academic Papers
- *"From 'Everything is a File' to 'Files Are All You Need'"* (arXiv:2601.11672) — filesystem abstraction as universal agent interface
- *"Everything is Context"* (arXiv:2512.05470) — file-system abstraction for context engineering (AIGNE framework)
- *"A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces"* (arXiv:2602.03442) — composable multi-modal retrieval
- *"RAG-MCP"* (arXiv:2505.03275) — tool selection accuracy degrades at scale, RAG-based retrieval restores it
- *"Code-Craft: Hierarchical Graph-Based Code Summarization"* (arXiv:2504.08975) — dependency-aware code retrieval
- *"cAST: Structural Chunking via Abstract Syntax Tree"* (arXiv:2506.15655) — AST-aware chunking

### Filesystem Theory & Systems Research
- Dennis Ritchie, *"The Evolution of the Unix Time-sharing System"* (AT&T Bell Labs Technical Journal, 1984) — filesystem-first design, inode architecture, device files as the key Unix insight
- Rob Pike et al., *"The Use of Name Spaces in Plan 9"* (5th ACM SIGOPS European Workshop, 1992) — per-process namespaces, union directories, 9P protocol
- Rob Pike, *"Lexical File Names in Plan 9, or, Getting Dot-Dot Right"* — canonical name tracking for correct `..` resolution in non-tree namespaces
- IEEE Std 1003.1-2017 (POSIX), *Section 4.13: Pathname Resolution* — formal path resolution semantics, trailing slash rules, symlink limits
- Henson, Val, *"ext3cow: A Time-Shifting File System for Regulatory Compliance"* — version-via-path encoding (`@epoch` syntax), prior art for `.versions/<N>` pattern
- Peterson, Zachary, Randal Burns, *"Ext3cow: The Design, Implementation, and Analysis of Metadata for a Time-Shifting File System"* (USENIX FAST, 2005)
- Konishi et al., *"The Linux Implementation of a Log-Structured File System"* (NILFS2) — continuous snapshotting, checkpoint-based versioning
- Pillai et al., *"All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications"* (OSDI 2014, ALICE tool) — "every application except SQLite had crash bugs"
- Chen et al., *"Using Crash Hoare Logic for Certifying the FSCQ File System"* (SOSP 2015) — first machine-checked proof of filesystem crash consistency
- Sigurbjarnarson et al., *"Push-Button Verification of File Systems via Crash Refinement"* (OSDI 2016, Yggdrasil) — automated verification without manual proofs

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
- ext3cow — time-shifting filesystem with `@epoch` path syntax for version access
- NILFS2 — log-structured filesystem with continuous snapshotting
- libsqlfs — SQLite-backed FUSE filesystem (path-as-key schema, prior art for database-backed VFS)
- fsspec — Python filesystem spec with `AbstractFileSystem`, protocol registry, URL chaining (complementary: uniform interface across backends at the I/O level)

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

### 14.7 Complete graph algorithms — 2026-03-25

**Commits:** `dfa9c5a`, `61f0f3f`

Implemented all 10 remaining graph algorithms on `RustworkxGraph`, migrated from v1 and adapted to v2 result types (`GroverResult`/`Candidate`/`Detail`). All algorithms cross-validated against NetworkX at scale (10K nodes, 30K edges) — exact match to machine epsilon.

#### Algorithms implemented

**Traversal (heavy, threaded via `asyncio.to_thread`):**
- `ancestors()` / `_ancestors_impl()` — transitive backward reachability via `rustworkx.ancestors()`
- `descendants()` / `_descendants_impl()` — transitive forward reachability via `rustworkx.descendants()`
- `neighborhood()` — bounded undirected BFS with full snapshot isolation (`_snapshot()` + `_in` + `_edge_types` copied before traversal)

**Subgraph (threaded):**
- `meeting_subgraph()` / `_meeting_subgraph_impl()` — multi-source BFS with `UnionFind` to detect wavefront convergence, then `_strip_leaves()` to prune non-seed leaf nodes. O(V+E).
- `min_meeting_subgraph()` / `_min_meeting_impl()` — calls `meeting_subgraph`, then iteratively removes non-seed, non-articulation-point nodes using `rustworkx.articulation_points()` on an undirected `PyGraph` projection.

**Centrality (threaded, generic dispatcher):**
- `_run_centrality()` / `_centrality_impl()` — generic pattern: `ensure_fresh → snapshot → build PyDiGraph → call rustworkx function → filter by candidates → score_candidates()`.
- `pagerank()` — delegates to `_run_centrality` with `rustworkx.pagerank`
- `betweenness_centrality()` — `rustworkx.digraph_betweenness_centrality` (normalized=True)
- `closeness_centrality()` — `rustworkx.closeness_centrality`
- `degree_centrality()` — `rustworkx.digraph_degree_centrality`
- `in_degree_centrality()` — `rustworkx.in_degree_centrality`
- `out_degree_centrality()` — `rustworkx.out_degree_centrality`
- `hits()` / `_hits_impl()` — custom implementation returning both hub and authority scores in `Detail.metadata`. `score` parameter (`"authority"` or `"hub"`) controls `Detail.score` and sort order.

#### Helper methods added

- `_strip_leaves(kept, edges_out, edges_in, protected)` — iterative leaf removal for meeting subgraph. Removes non-protected nodes that have no successors or no predecessors within the kept set, cascading until stable.

#### Key decisions and fixes

**`neighborhood` uses full snapshot isolation.** Copies `_out`, `_in`, and `_edge_types` before BFS so concurrent mutations between `ensure_fresh()` and result construction cannot cause inconsistent state or `KeyError` in `_subgraph_candidates`.

**`pagerank` collapsed to one-liner.** Was a redundant hand-rolled copy of `_run_centrality` + `_centrality_impl`. Now just `return await self._run_centrality("pagerank", rustworkx.pagerank, ...)`.

**HITS `max_iter` default bumped from 100 to 1000.** `rustworkx.hits()` raises an exception on non-convergence (doesn't return partial results). Sparse graphs at 5K+ nodes regularly need 300-500 iterations. 1000 prevents surprise failures.

**HITS edgeless early-return filters to graph-only paths.** Previously included non-graph candidate paths with zero scores; now consistent with the normal path where non-graph candidates are silently dropped.

**HITS `set(hubs) | set(auths)` eliminated.** Both dicts always have identical keys after `rustworkx.hits()`. Sort now uses `sorted(primary, key=primary.__getitem__, reverse=True)`.

**HITS `score` parameter.** `"authority"` (default) ranks by how many hubs point to a node. `"hub"` ranks by how many authorities a node points to. Both values always in `Detail.metadata`.

**`min_meeting_subgraph` uses `_edge_types` directly.** Previously decomposed connection paths from the meeting result to recover edge topology. Now reads the authoritative `_edge_types` dict filtered to the meeting node set.

**`min_meeting_subgraph` uses `decompose_connection()` to identify nodes.** Previously relied on `c.weight is None` convention. Now uses the intrinsic path structure (`/.connections/` marker) to distinguish nodes from edges.

#### Performance vs NetworkX (10K nodes, 30K edges)

| Algorithm | Grover (ms) | NetworkX (ms) | Speedup |
|---|---:|---:|---|
| betweenness | 1,557 | 137,784 | **89x faster** |
| closeness | 780 | 27,009 | **35x faster** |
| pagerank | 40 | 17 | 0.4x (graph rebuild overhead) |
| hits | 82 | 38 | 0.5x (graph rebuild overhead) |
| ancestors/descendants | 35-41 | 4 | 0.1x (graph rebuild dominates) |
| degree centrality | 36 | 1 | 0.03x (trivial compute, rebuild dominates) |
| predecessors/successors | 0.01 | 0.00 | N/A (both sub-ms) |
| neighborhood (d=2) | 0.4 | 0.2 | ~parity |

For algorithms where computation dominates (betweenness, closeness), rustworkx's Rust backend delivers massive wins that scale with graph size. For cheap algorithms (degree, ancestors), the per-call `PyDiGraph` rebuild (~15-35ms at 10K nodes) dominates.

#### Test coverage

138 graph tests (was 65), 534 total (was 464). Graph subpackage at **99%** coverage (3 unreachable defensive lines: `_strip_leaves` duplicate-in-queue guard, `min_meeting_subgraph` except block unreachable because `meeting_subgraph` catches first).

| File | Coverage |
|------|----------|
| `graph/__init__.py` | 100% |
| `graph/protocol.py` | 100% |
| `graph/rustworkx.py` | 99% |
| **Total project** | **99%** (1173 stmts, 10 missed) |

### 14.8 DatabaseFileSystem write implementation — 2026-03-30

Implemented `backends/database.py` — the SQL-backed `_write_impl` for `grover_objects`. 6-step pipeline: validate, check chunk parents, resolve parent dirs, fetch existing, process writes, commit dirs.

Added `versioning.py` — forward unified diffs with periodic snapshots (every 10 versions). `plan_file_write` on `GroverObjectBase` plans version rows and final file state. `create_version_row` constructs version objects. `_reconstruct_file_version` replays diff chains from snapshots.

#### Key decisions

**Two-query fast path for overwrites.** Step 4 fetches file rows (4a) then latest version hash by constructed path (4b), both via the unique `path` index. When file hash and version hash agree, `plan_file_write` skips `_reconstruct_file_version` entirely — diff computed directly from current content. Broken intermediate version rows are not detected on the fast path (accepted behavior). 100k overwrites on PostgreSQL: 678s → ~160s.

**No manual batching or savepoints.** SQLAlchemy 2.0 `insertmanyvalues` auto-paginates INSERTs per dialect parameter limits. All-or-nothing failure semantics — session rolls back on flush error.

**Content-before-commit ordering.** Parent dirs only commit (Step 6) if file writes succeed (Step 5). Failed writes never leave orphan directories.

**`VersionWritePlan` as decision-complete plan.** `plan_file_write` returns a frozen dataclass with all version rows, final content/hash/metrics, and a `chain_verified` flag. `_update_existing` in `database.py` checks this flag and falls back to `_fetch_version_chain` + re-plan when the fast path can't verify integrity.

**`--scale` test flag.** Pressure tests default to 1k rows, ramp with `--scale 100000` for ad-hoc stress testing against PostgreSQL.

### 14.9 CLI query engine + planner — 2026-03-31

Implemented the first runnable CLI query layer on top of `GroverFileSystem`. The architecture now matches the direction described earlier in this document:

```text
CLI-style query string
    -> hand-rolled tokenizer / parser
    -> frozen AST
    -> ordered query plan
    -> executor
    -> existing eager GroverFileSystem methods
    -> GroverResult
    -> text renderer
```

#### What was added

**New query package (`src/grover/query/`):**
- `ast.py` — frozen AST nodes for pipelines, unions, subqueries, visibility overrides, and terminal render modes
- `parser.py` — hand-rolled `match/case` tokenizer and parser with fail-fast syntax errors
- `executor.py` — lowers AST nodes into explicit `GroverFileSystem` method calls and local `GroverResult` transforms
- `render.py` — default text renderers for `read`, `ls`, `tree`, `stat`, query lists, and mutation summaries
- `__init__.py` — public query package exports

**New `GroverFileSystem` entrypoints (`base.py`):**
- `parse_query(query)` — returns a `QueryPlan` with the AST and ordered public method calls
- `run_query(query, initial=None)` — executes the parsed query and returns `GroverResult`
- `cli(query, initial=None)` — executes the query and returns rendered text

**Demo script:**
- `examples/cli_query_demo.py` — seeds an in-memory `DatabaseFileSystem` and prints the query string, planned methods, and rendered output for representative pipelines and flags

**Tests:**
- `tests/test_query_cli.py` — parser, plan output, execution, rendering, unions, intersection / difference stages, piped copies, and stage-local flags

#### Query language implemented

The initial DSL is deliberately small and Unix-shaped:

```text
search "phrase" --k 5 | glob "/src/*.py" | read
grep "TODO" --max-results 20 | meetinggraph --min | pagerank | top 3
grep "import" & grep "DEBUG"
glob "/src/*.py" | intersect (grep "auth")
glob "/src/*.py" | except (grep "deprecated")
```

**Operators:**
- `|` — pipeline; passes a structured `GroverResult` into the next stage as `candidates`
- `&` — union; evaluates both branches and unions the results
- `()` — explicit grouping
- `intersect(...)` — set intersection stage
- `except(...)` — set difference stage

**Command families currently supported:**
- CRUD: `read`, `stat`, `rm`, `edit`, `write`, `mkdir`, `mv`, `cp`, `mkconn`
- Navigation / query: `ls`, `tree`, `glob`, `grep`, `search`, `lsearch`, `vsearch`
- Graph: `pred`, `succ`, `anc`, `desc`, `nbr`, `meetinggraph`, `meetinggraph --min`
- Ranking: `pagerank`, `betweenness`, `closeness`, `degree`, `indegree`, `outdegree`, `hits`
- Local transforms: `sort`, `top`, `kinds`

#### Key decisions

**Pipeline means structured result passing, not raw byte streams.** The surface deliberately looks like Unix, but `|` is not a shell stdout/stdin pipe. It threads `GroverResult` objects through the pipeline. This keeps the UX Unix-like while preserving typed execution and detail chaining.

**No stdin in v1, but the executor shape allows it later.** `run_query()` / `cli()` already accept an optional `initial` result. That is the intended insertion point for future stdin support or shell-wrapper input hydration.

**Broad query operations hide metadata by default.** `glob`, `grep`, `search`, `lsearch`, `vsearch`, and `tree` default to file/directory views. Explicit opt-in is via `--all` or `--include chunks,versions,connections,api`.

**Graph traversal defaults differ from broad search defaults.** For the CLI query language, graph traversal and graph ranking keep returning connection candidates by default. This is a deliberate UX choice for graph-oriented pipelines even though earlier generic design notes leaned more aggressively toward files-only output.

**No permanent delete in the CLI surface.** `rm` lowers to `delete(permanent=False)`. Permanent deletion remains intentionally unavailable through the query language.

**Stage-local flags are the only way to pass method args.** Examples: `search "phrase" --k 5`, `grep "TODO" --max-results 10`, `nbr /src/auth.py --depth 3`, `tree /src --include connections`. The parser rejects unknown flags and duplicate flags immediately.

**Fail fast on syntax and flag errors.** Unknown verbs, unknown flags, duplicate flags, missing values, and malformed subqueries all raise `QuerySyntaxError` during parsing rather than deferring failure to execution.

**Intersection and difference are spelled as filter stages, not extra infix symbols.** `&` is reserved for union. Intersection and difference are expressed as `intersect(...)` and `except(...)`, which keeps the language closer to Unix filter composition than a dense symbolic algebra.

**Command names are compact and Unix-leaning.** The query string favors short aliases (`pred`, `succ`, `nbr`, `rm`, `cp`, `mv`, `meetinggraph`) rather than Python API spellings.

**The shell binary name remains `grover`, but internal query strings do not repeat it.** The intended shell UX is:

```bash
grover 'search "auth" | meetinggraph --min | pagerank'
```

not:

```bash
grover search "auth" | grover meetinggraph --min | grover pagerank
```

**Piped bulk mutation semantics follow `GroverResult`.**
- `... | read` reads every candidate
- `... | edit "old" "new"` edits every candidate
- `... | rm` soft-deletes every candidate
- `... | cp /destroot` copies each candidate under `/destroot`, preserving relative paths
- `... | mv /destroot` moves each candidate under `/destroot`, preserving relative paths
- `... | mkconn imports /target` creates one connection per input candidate to the target
- `write` and `mkdir` remain standalone commands for now

**The parser was refactored to a command registry.** After the initial implementation proved the language shape, the parser was reorganized around declarative `CommandSpec` records: aliases, allowed flags, and AST builders live in one table. Adding new args like `--k`, `--depth`, or `--include` is now a registry change instead of another parser branch.

#### Rendering defaults implemented

`cli(query)` currently selects a default text renderer from the final stage:

- `read` — file content view
- `write` / `edit` / `rm` / `mv` / `cp` / `mkdir` / `mkconn` — mutation summary
- `ls` — flat list
- `tree` — tree view
- `stat` — metadata block
- query / traversal / ranking pipelines — path list with scores when the final candidates carry scores

#### Current boundaries

**This is an in-process query engine first, not a full shell entrypoint yet.** The implemented public surface is `GroverFileSystem.parse_query()`, `run_query()`, and `cli()`. A real console-script wrapper still needs to be added if Grover should be invocable directly from the shell.

**`search` depends on a configured semantic provider.** The query engine supports `search`, but demo / out-of-the-box examples use `lsearch` because a plain in-memory `DatabaseFileSystem` does not have a semantic backend attached.

**Help output is not yet generated from the registry.** The command registry now contains enough metadata to drive generated help in a future pass, but the current work stops at parsing, planning, execution, and rendering.

### 14.10 DatabaseFileSystem CRUD + query + graph buildout — 2026-03-30/31

**Commits:** `e0f051f` through `b8894de`

Between the `write` implementation (§14.8) and the CLI query engine (§14.9), the full DatabaseFileSystem surface was built out across 7 commits. This section covers the CRUD operations, query operations, graph wiring, and cross-cutting infrastructure that completed the backend.

#### CRUD operations (`database.py`, grew to ~1800 lines)

**Commit `2ba55fa` — ls, delete, mkconn, mkdir:**
- `_ls_impl` — kind-aware: directories list files/dirs children, files list metadata children (`.chunks/`, `.versions/`, `.connections/`)
- `_delete_impl` — soft-delete (`deleted_at` timestamp) and permanent delete with cascading via `_cascade_delete` / `_cascade_delete_permanent`. Batched cascade queries replace N+1 per-item pattern.
- `_mkconn_impl` — connection creation with source validation, constructs `<source>/.connections/<type>/<target>` path, delegates to `_write_impl`
- `_mkdir_impl` — directory creation delegating to `_write_impl` with `kind=directory`
- `_write_impl` widened to accept directory and connection kinds (only files get versioning)

**Commit `e5a22ea` — edit, copy, move, model validation:**
- `_edit_impl` — read → three-level replace engine (exact / line-trimmed / block-anchor via `replace.py`) → batch write
- `_copy_impl` — batch read sources → batch write to destinations
- `_move_impl` — atomic same-mount rename, cascades descendants, rebuilds connection paths, updates incoming `target_path` references. Rejects occupied destinations.
- `delete` gained `cascade=False` mode — non-cascading delete rejects non-empty objects (rmdir semantics)
- `delete` root guard — `delete("/")` always errors
- Model validation: directory content coerced to `None`, file content defaults to `""`, `update_content()` raises on directories

**Commit `e5a22ea` also added `replace.py`** (330 lines) — fuzzy string matching with three strategies: exact match, line-trimmed (ignoring leading/trailing whitespace), and block-anchor (find surrounding context and replace within). Ported from v1.

#### Query operations

**Commit `e0f051f` — patterns.py + base.py routing refactor:**
- `patterns.py` (145 lines) — glob/grep pattern matching utilities: `glob_to_sql_like()` for SQL pre-filtering, `compile_glob()` for Python regex post-filtering
- `base.py` refactored with improved mount routing and session management
- `_with_candidates` helper added to `results.py`

**Commit `c529777` — glob, grep, tree:**
- `_glob_impl` — SQL LIKE pre-filter via `glob_to_sql_like()`, Python regex post-filter via `compile_glob()`. Files/dirs only by default. Candidate filtering preserves detail chain.
- `_grep_impl` — regex content search across files, line-by-line matching with `line_matches` metadata. Hydrates only candidates missing content. Supports `case_sensitive` and `max_results`.
- `_tree_impl` — recursive directory listing with `max_depth` via SQL slash counting. Validates path exists and is a directory.

#### Graph wiring

**Commit `3ad8565` — graph operations + versioning tests:**
- All 14 graph `_*_impl` methods in `DatabaseFileSystem` now delegate to the internal `RustworkxGraph` instance
- Graph cache invalidated (`self._graph.invalidate()`) after connection mutations (`mkconn`, `delete`, `move`, `write`) for immediate consistency
- `RustworkxGraph._load()` fixed to exclude soft-deleted connections (`WHERE deleted_at IS NULL`)
- Versioning module got dedicated unit tests covering `compute_diff`, `apply_diff`, `reconstruct_version`, `create_version` — coverage from 88% to 100%

#### Cross-cutting improvements

**Commit `ef1679a` — detail preservation:**
- Prior details from incoming candidates were lost in graph and centrality methods because each `_impl` built fresh `Candidate` objects
- Solution: `inject_details()` on `GroverResult` prepends prior details onto overlapping result candidates in `_dispatch_candidates` — one line in the routing layer covers all public methods automatically
- Removed `prior_details` plumbing from `_read_impl`, `_grep_impl`, `_glob_impl`, and `to_candidate`

**Commit `b8894de` — signature cleanup:**
- `path`, `pattern`, and `query` parameters on `_*_impl` methods had `= ""` defaults that masked potential bugs. Defaults removed — callers always provide values.

#### Test coverage

1,421 tests at end of buildout (was 534 at §14.7). 96% overall coverage. Key new test files:
- `tests/test_database.py` — 77+ database operation tests
- `tests/test_database_graph.py` — 514-line graph integration test suite
- `tests/test_replace.py` — 49 tests (47% → 99% coverage for `replace.py`)
- `tests/test_patterns.py` — 917-line pattern matching test suite
- `tests/test_method_chaining.py` — 474-line detail preservation test suite
- `tests/test_versioning.py` — dedicated versioning unit tests

### 14.11 Async version planning — 2026-03-31

**Commit:** `704e403`

`plan_file_write` calls in `_update_existing` moved to `asyncio.to_thread()` so SHA256 hashing, diff computation, and version chain reconstruction no longer block the event loop. `_insert_new` made `async` for consistency with all other session-accepting methods.

Small change (8 insertions, 9 deletions) but important for write-heavy workloads — version planning is CPU-bound (hashing + diffing), and prior to this change it blocked the asyncio loop during every file update.

### 14.12 EmbeddingProvider, VectorStore protocols, and search implementations — 2026-03-31

**Commit:** `12880bf`

Introduced the embedding and vector search layer that completes the three-modality search stack described in §7.2 (pattern via glob, keyword via grep/lsearch, semantic via search/vsearch).

#### EmbeddingProvider protocol (`embedding.py`, 159 lines)

Async-first protocol for text-to-vector embedding. Returns `Vector` instances (from `vector.py`) with dimension and model-name tracking so embeddings carry provenance from creation through database storage.

**Methods:** `embed(text) → Vector`, `embed_batch(texts) → list[Vector]`, `dimensions` property, `model_name` property.

**`LangChainEmbeddingProvider`** — adapter wrapping any `langchain_core.embeddings.Embeddings` instance. Adapts `list[float]` results into properly-typed `Vector` instances. Optional dependency via `pip install grover[langchain]`.

#### VectorStore protocol (`vector_store.py`, 58 lines)

Async protocol for vector similarity search backends. Three operations:
- `query(vector, *, k, paths) → list[VectorHit]` — nearest neighbor search, optionally constrained to specific paths
- `upsert(items: list[VectorItem]) → None` — insert or update vectors
- `delete(paths: list[str]) → None` — remove vectors by path

`VectorItem` (path + vector pair for upsert) and `VectorHit` (path + score result) are frozen dataclasses.

#### DatabricksVectorStore (`databricks_store.py`, 163 lines)

Production implementation of `VectorStore` using Databricks Direct Vector Access indexes. All SDK calls wrapped in `asyncio.to_thread()` because the Databricks SDK is synchronous. Batched upserts (1,000 items per batch). Optional dependency via `pip install grover[databricks]`.

#### DatabaseFileSystem search wiring

`DatabaseFileSystem` constructor now accepts optional `embedding_provider: EmbeddingProvider` and `vector_store: VectorStore` parameters.

- `_semantic_search_impl` — embed query text via `EmbeddingProvider`, then delegate to `_vector_search_impl`. Returns error result if no embedding provider or vector store configured.
- `_vector_search_impl` — query the `VectorStore` for nearest neighbours. Constrains to candidate paths when piped. Returns `Candidate` objects with `Detail(operation="vector_search", score=...)`.

#### Key decisions

**Protocols, not base classes.** Both `EmbeddingProvider` and `VectorStore` are `runtime_checkable` `Protocol` classes. Implementations don't need to inherit — duck typing works. This keeps the dependency tree clean: `embedding.py` doesn't import LangChain, `vector_store.py` doesn't import Databricks.

**`Vector` carries provenance.** Embedding results include dimension count and model name (from `vector.py`), so vectors stored in the database are self-describing. Dimension mismatches are caught at write time.

**Graceful degradation.** `semantic_search` and `vector_search` return error results (not exceptions) when providers aren't configured. The CLI query engine already handles this — `search` works only with a configured provider, `lsearch` works out of the box.

#### Tests

- `tests/test_embedding.py` (187 tests) — protocol compliance, LangChain adapter, batch embedding, dimension tracking
- `tests/test_vector_store.py` (323 tests) — protocol compliance, DatabricksVectorStore (mocked SDK), query/upsert/delete, batch splitting

### 14.13 BM25 implementation and benchmarks — 2026-03-31

**Commit:** `aa4606f`

Added a hand-rolled BM25 scorer module and in-memory index, the engine behind `_lexical_search_impl` in DatabaseFileSystem.

#### BM25Scorer (`bm25.py`, 453 lines)

Pure Python, no numpy. Matches `rank-bm25` BM25Okapi scoring semantics. Designed for use with SQL-backed corpora where full-corpus IDF is computed via COUNT queries and document content is pre-filtered before scoring.

**Architecture:**
- Corpus-level statistics (`corpus_size`, `avg_doc_length`) and per-term document frequencies provided externally — typically from SQL COUNT queries — so the scorer never needs the full corpus in memory
- Standard BM25 IDF: `log((N - n + 0.5) / (n + 0.5))` with epsilon floor for common terms (matching BM25Okapi)
- Pre-computed constants (`_k1_plus_1`, `_length_norm_base`, `_length_norm_scale`) eliminate repeated arithmetic during scoring

**Key methods:**
- `set_idf(doc_freqs)` — pre-compute IDF from document-frequency counts, apply epsilon floor for common terms
- `score_document(query_terms, doc_tokens)` — score a single document
- `score_batch(query_terms, documents)` — score multiple documents (for ad hoc SQL pre-filter batches)
- `score_batch_term_frequencies(query_terms, term_freq_docs, doc_lengths)` — score from precomputed term frequencies

**Tokenizer:** `tokenize()` — lowercase split on `\W+`, drop empties. `tokenize_query()` — same with 50-term cap (`QUERY_TERM_LIMIT`).

#### BM25Index

Prepared index for repeated in-memory searches over a stable corpus. Pre-computes postings, document norms, and full-corpus IDF once, so each query only visits matching documents.

**Key methods:**
- `score_sparse(query_terms)` — return `{doc_idx: score}` for matching documents only (posting list intersection)
- `score_batch(query_terms)` — dense scores aligned to document order
- `topk(query_terms, k)` — heap-based top-k via `heapq.nlargest`

#### Lexical search in DatabaseFileSystem

`_lexical_search_impl` uses the `BM25Scorer` in a SQL-hybrid pipeline:
1. Tokenize query (capped at 50 terms)
2. Compute IDF against full corpus via SQL COUNT queries
3. Pre-filter candidates with SQL LIKE + term-count sort (capped at `BM25_PRE_FILTER_LIMIT = 1,000`)
4. Score pre-filtered documents with `BM25Scorer`
5. Return top-k results as `GroverResult` with `Detail(operation="lexical_search", score=...)`

Searches files and chunks — versions excluded (they duplicate file content).

#### Benchmarks

- `scripts/bench_bm25.py` — raw scorer throughput benchmarks
- `scripts/bench_bm25_comparison.py` — cross-validated against `rank-bm25` BM25Okapi at scale, verifying score-exact matching

#### Tests

- `tests/test_bm25_comparison.py` (400 tests) — comprehensive comparison suite validating Grover's BM25 against `rank-bm25` reference implementation. Covers IDF computation, single-document scoring, batch scoring, edge cases (empty docs, unknown terms, repeated terms), and the BM25Index prepared-index path.
- `tests/test_lexical_search.py` (366 tests) — integration tests for `_lexical_search_impl` against `DatabaseFileSystem` with seeded content

#### Key decisions

**Hand-rolled, not a dependency.** `rank-bm25` is a small library but adds a numpy dependency and doesn't support the SQL-hybrid workflow (external IDF, pre-filtered document sets). The hand-rolled implementation is ~450 lines of pure Python, exactly matches BM25Okapi scoring, and integrates cleanly with the SQL pre-filter pipeline.

**SQL pre-filter before scoring.** Full BM25 over the entire corpus would require loading all content into Python. Instead, SQL LIKE narrows to candidate documents containing query terms, sorted by term-count heuristic, capped at 1,000 rows. The BM25 scorer then ranks within this pre-filtered set. This keeps memory bounded and query times fast for large corpora.

**BM25Index for benchmarks, BM25Scorer for production.** The index pre-computes postings for repeated queries over stable data — useful for benchmarks and REPL exploration. The scorer takes external statistics — useful for the SQL-hybrid production path where IDF comes from COUNT queries and documents come from pre-filtered SQL results.

### 14.14 User-scoped filesystem support — 2026-04-01

**Commit:** `5996428`

Added per-user path-prefix isolation to the entire GroverFileSystem stack. A `DatabaseFileSystem` with `user_scoped=True` transparently prefixes all paths with `/{user_id}/` in storage, allowing multiple users to share the same logical paths (e.g., both users have `/docs/README.md`).

#### Design: unscoped paths everywhere, scope at the DB boundary

The API contract is strict: **all public methods and `_*_impl` calls always receive unscoped paths + `user_id`.** Scoping (prepending `/{user_id}/`) and unscoping (stripping it) happen inside each terminal `_*_impl` method at the DB boundary, never across method calls. This prevents double-scoping bugs and makes behavior consistent — a user with id `123` can have a directory named `123` without ambiguity (`/123/123/file.py` in storage).

**Terminal methods** (read, write, ls, delete, stat, tree, move, glob, grep, search, graph ops) — scope inputs at top, unscope result at bottom.

**Delegating methods** (edit→read+write, copy→read+write, mkdir→write, mkconn→write) — NO scoping. Pass unscoped paths + `user_id` to inner `_*_impl` calls. Each inner call handles its own scope cycle. Intermediate results use unscoped paths throughout.

#### Mount topology

```python
g = Grover()
g.add_mount('/snhu', DatabaseFileSystem(engine=shared_engine))                    # shared
g.add_mount('/user', DatabaseFileSystem(engine=user_engine, user_scoped=True))    # per-user

await g.semantic_search("auth", user_id="123")
# /snhu: searches everything (ignores user_id for now)
# /user: searches only /123/** paths in DB
```

#### `user_id` threading

`user_id: str | None = None` added as a keyword-only parameter to:
- All 30+ public methods on `GroverFileSystem`
- All 6 routing methods (`_route_single`, `_dispatch_candidates`, `_route_fanout`, `_route_two_path`, `_cross_mount_transfer`, `_route_write_batch`) — explicit parameter, not `**kwargs`
- All 28 `_*_impl` stubs
- All query executor functions (`execute_query` → `_execute_node` → `_execute_stage` → all helpers)
- `run_query()` and `cli()` — `user_id` is a parameter, NOT part of the query string

Non-scoped filesystems ignore `user_id` harmlessly. Scoped filesystems raise `ValueError` if `user_id` is missing.

#### Path scoping utilities (`paths.py`)

- `validate_user_id(user_id)` — rejects empty, `/`, `\\`, `@`, `\0`, `..`, >255 chars
- `scope_path(path, user_id)` — always prepends (not idempotent): `/docs/README` + `"123"` → `/123/docs/README`
- `unscope_path(path, user_id)` — always strips (raises on mismatch). For connection paths, uses `decompose_connection()` to parse, strips prefix from both source and target, reconstructs with `connection_path()`

#### Connection paths — scoped source AND target

Both endpoints are scoped in the connection path:
```
/123/src/main.py/.connections/imports/123/src/auth.py
```

`decompose_connection()` naturally gives `source="/123/src/main.py"`, `target="/123/src/auth.py"` — matches DB columns exactly. No special-casing needed in `_normalize_and_derive`. Unscoping strips both:
```
/src/main.py/.connections/imports/src/auth.py
```

`_scope_objects()` rebuilds connection paths from scoped `source_path` + `target_path` to keep the `path`, `source_path`, and `target_path` fields consistent.

#### DatabaseFileSystem scoping helpers

Five private methods on `DatabaseFileSystem`:
- `_scope_path(path, user_id)` — no-op when not user_scoped or user_id is None
- `_scope_candidates(candidates, user_id)` — scope all candidate paths
- `_scope_objects(objects, user_id)` — scope `path`, `source_path`, `target_path`, rebuild connection paths, set `owner_id`
- `_unscope_result(result, user_id)` — calls `result.strip_user_scope(user_id)`
- `_require_user_id(user_id)` — raises if user_scoped and no user_id

#### `owner_id` column

`owner_id` is the DB column on `GroverObjectBase` (already existed). `user_id` is the calling parameter. When `user_scoped=True`, `_scope_objects()` sets `owner_id = user_id` on all new objects during writes.

#### BM25 lexical search — per-user corpus stats

Both `_fetch_lexical_docs()` and `_fetch_corpus_stats()` add `WHERE path LIKE '/{user_id}/%'` when user-scoped. Without this, IDF scores would be computed against the global corpus (all users), diluting user-specific relevance.

#### VectorStore — `user_id` column filtering

`VectorStore.query()` gained `user_id: str | None = None`. `VectorItem` gained `owner_id: str | None = None` for upsert storage. `DatabricksVectorStore` uses native column filtering (`filters={"owner_id": user_id}`).

#### Graph — query-time filtering, not load-time

`RustworkxGraph` loads ALL users' connections (single shared graph). Filtering happens at query time:
- `_visible_nodes(user_id)` — returns only nodes with `/{user_id}/` prefix when user-scoped
- `_snapshot(user_id)` — returns filtered nodes + edges
- All traversal and centrality methods use filtered snapshots, so `pagerank()` with `user_id="123"` computes only over user 123's graph
- Graph stays shared in memory — no per-user reload penalty

#### Key decisions

**Strict scoping, not idempotent.** `scope_path` always prepends, `unscope_path` always strips (raises on mismatch). This avoids the ambiguity of a user named `123` having a directory named `123` — the system always knows exactly where it is in the scope/unscope lifecycle.

**Terminal vs delegating method separation.** Terminal methods own their scope cycle. Delegating methods pass unscoped paths through. This prevents double-scoping and double-unscoping bugs that occurred in v1's `UserScopedFileSystem`.

**No subclass.** User scoping is a `user_scoped=True` flag on `DatabaseFileSystem`, not a separate `UserScopedFileSystem` class. This avoids the v1 pain points of polymorphic dispatch (where `copy()` called `self.write()` which re-entered the override).

**`user_id` per-method-call, not per-constructor.** One `Grover` instance serves multiple users. The `user_id` identifies the calling user on each operation. This is the web app model, not Plan 9's per-process namespace model.

**Graph loads all, filters at query time.** Loading per-user would cause reloads on every user switch. Loading all and filtering at query time keeps the graph warm and shared. The path-prefix isolation guarantees traversal stays within the user's namespace naturally.

**ReBAC on shared mounts is a separate future concern.** The `/snhu` shared mount currently ignores `user_id`. Future work: role/group-based access control on shared content, using the `SupportsReBAC` protocol and shares table. The `user_id` parameter is already threaded through — adding permission checks is an additive change.

#### Test coverage

42 new tests in `tests/test_user_scoping.py`:
- Path utility tests (validate_user_id, scope_path, unscope_path, roundtrips)
- Result unscoping (files, connections)
- Write + read isolation between two users
- `owner_id` set on writes
- `user_id` required when scoped, ignored when not scoped
- `ls`, `glob`, `grep`, `tree`, `delete`, `move`, `copy`, `edit` — all scoped
- `mkconn` — connection path scoping, source/target verification
- Graph isolation — predecessors, pagerank stay within user namespace
- BM25 lexical search — per-user corpus

1463 total tests (was 1421), all passing. ruff and ty clean.

### 14.13 LIKE wildcard escaping in path-based SQL queries — 2026-04-01

**Decision:** All SQL `LIKE` clauses that interpolate stored path values must use `_escape_like()` and pass `escape="\\"`. Paths may legally contain `%` and `_` characters (they pass `validate_path()`), but these are LIKE wildcards in SQL. Without escaping, a path like `/data/100%/` produces `WHERE path LIKE '/data/100%/%'`, which matches unintended rows.

**Security finding:** `_move_impl` used unescaped path values in two LIKE patterns (descendant fetch and incoming-connection update) with no Python-level post-filter. A `move()` on a path containing `%` would rewrite paths of unrelated files, corrupting data. `_fetch_children_batched` had the same LIKE issue but was partially mitigated by a `startswith()` post-filter.

**Fix:** Three LIKE calls in `database.py` updated to use `_escape_like(path) + "/%"` with `escape="\\"`:

| Location | Pattern | Risk |
|----------|---------|------|
| `_fetch_children_batched` | `path.like(p + "/%")` | Mitigated by `startswith` post-filter, fixed for defense in depth |
| `_move_impl` (descendants) | `path.like(op.src + "/%")` | **Critical** — results used directly for path rewriting |
| `_move_impl` (connections) | `target_path.like(op.src + "/%")` | **Critical** — results used directly to update connection targets |

The codebase already had the correct tools — `_escape_like()` (line 46) and correct usage in `_glob_impl` and `_tree_impl`. The inconsistency was the bug.

**`user_id` is trusted** — the four `WHERE path LIKE '/{user_id}/%'` clauses in user-scoped queries are not escaped because `user_id` is assigned by the application's authentication layer, not by end users.

#### Test coverage

5 new tests in `TestLikeWildcardSafety` (`tests/test_database.py`):
- `move()` with `%` in path does not affect sibling directories
- `move()` with `_` in path does not affect sibling directories
- `delete(cascade=True)` with `%` in path only deletes correct subtree
- `ls()` on directory with `%` in name returns only own children
- `move()` with `%` in path correctly scopes connection target rewrites

1468 total tests (was 1463), all passing. ruff and ty clean.

### 14.15 GroverAsync and Grover facade layer — 2026-04-01

**Commits:** (this session)

Added the two-tier facade layer described in §11.1: `GroverAsync` for async app servers and `Grover` for sync data pipelines. Both sit on top of `GroverFileSystem` as storageless routers (`storage=False`) that delegate to mounted backends.

#### `storage` flag on GroverFileSystem (`base.py`)

`GroverFileSystem.__init__` gained `storage: bool = True`. When `storage=False`:
- Engine/session_factory are not required (`_session_factory = None`)
- `_use_session()` raises `RuntimeError` if called (guard against accidental self-storage)
- `_route_fanout()` skips `_query_self()` and fans out to mounts only
- `_route_single()` returns a clean `"No mount found for path: {path}"` error when `_resolve_terminal` returns `self`

This eliminates the need for subclass overrides — the base class handles both roles via the flag.

#### `raise_on_error` flag on GroverFileSystem (`base.py`)

`GroverFileSystem.__init__` gained `raise_on_error: bool = False`. When `True`, `_error()` raises a classified `GroverError` subclass instead of returning `GroverResult(success=False)`.

`_error()` was changed from `@staticmethod` to an instance method and now accepts three input types:
- `str` — creates `GroverResult(success=False, errors=[str])`, optionally raises
- `list[str]` — creates `GroverResult(success=False, errors=list)`, optionally raises
- `GroverResult` — passes through if successful, raises if failed and `raise_on_error=True`

The third form handles the ~12 batch operations in `database.py` that accumulate errors across items and construct `GroverResult` directly. All batch result returns are now wrapped with `self._error(result)`.

`add_mount()` propagates `_raise_on_error` to child filesystems, so the flag cascades through the mount tree.

#### Exception hierarchy (`exceptions.py`)

```
GroverError(Exception)           — base, carries .result: GroverResult | None
├── NotFoundError                — "Not found:", "Not a directory:"
├── MountError                   — "No mount found"
├── WriteConflictError           — "Already exists", "Cannot write", "Cannot delete"
├── ValidationError              — "requires", "Invalid", "Duplicate", "Source not found"
└── GraphError                   — "failed:" (graph algorithm errors)
```

`_classify_error(message, errors, result)` maps error message patterns to the appropriate subclass. Falls back to `GroverError` for unrecognized messages.

#### GroverAsync (`client.py`)

Thin subclass of `GroverFileSystem(storage=False)`:

```python
class GroverAsync(GroverFileSystem):
    def __init__(self) -> None:
        super().__init__(storage=False)
```

Rich `add_mount(name, *, filesystem=, engine=, session_factory=, engine_url=, ...)`:
- Four configuration paths: `filesystem`, `engine`, `session_factory`, `engine_url`
- `engine_url` auto-creates engine and tables (`create_tables=True` default)
- Injects `embedding_provider` and `vector_store` onto `DatabaseFileSystem` mounts
- Mount path: `normalize_path(f"/{name}")`

Lifecycle: `remove_mount(name)` disposes engine, `close()` disposes all engines and clears mounts.

#### Grover sync wrapper (`client.py`)

Follows the v1 daemon thread + event loop + RLock pattern:

```python
class Grover:
    def __init__(self) -> None:
        self._async = GroverAsync()
        self._async._raise_on_error = True  # propagates to all mounts
```

Since `raise_on_error=True` cascades to all mounted filesystems, errors raise inside the async code and propagate through `_run()`. No post-hoc checking needed.

**Return types:**
- Single-path ops (`read`, `write`, `edit`, `delete`, `stat`, `mkdir`, `mkconn`) return `Candidate` (unwrapped via `.file`)
- Multi-result ops (search, graph, listing, move, copy) return `GroverResult` (set algebra preserved)
- `cli` returns `str`, `parse_query` returns `QueryPlan`

#### Key decisions

**`storage` flag, not subclass override.** The base class `_route_fanout` and `_route_single` check `self._storage` directly. No overrides needed in `GroverAsync` — the flag makes the routing logic self-aware.

**`raise_on_error` propagates via `add_mount`.** The flag is set on the router and cascades to children during mounting. This means `Grover` only sets it once on its internal `GroverAsync`, and all mounted `DatabaseFileSystem` instances inherit it automatically.

**Overloaded `_error()` instead of a second method.** `_error(GroverResult)` checks an existing result and raises if needed. This keeps one integration point for all error paths — both the 40+ `self._error("message")` calls and the 12 batch `GroverResult` constructions.

**`Grover.read()` returns `Candidate`, not `str`.** Pipelines often need metadata (path, kind, size) alongside content. `Candidate` provides both via `.content` and `.path`. Returning raw `str` would lose provenance.

#### Test coverage

77 new tests across two files:
- `tests/test_client.py` — 31 tests: GroverAsync construction, mount lifecycle, routing, fanout, query engine, storage flag
- `tests/test_grover_sync.py` — 46 tests: exception hierarchy, `_classify_error`, `raise_on_error` flag, Grover sync CRUD, error raising (`NotFoundError`, `MountError`, `WriteConflictError`), search/listing return types, set algebra, query engine

1556 total tests (was 1468), all passing. 92% overall coverage. ruff and ty clean.

### 14.16 Test coverage to 99% and CI enforcement — 2026-04-01

Systematic coverage audit and gap closure, raising overall coverage from 91% to 99.11% (1779 tests, all passing).

#### CI pipeline changes

- `.github/workflows/test.yml`: Added `--cov --cov-report=term-missing --cov-fail-under=99` to the test step, gated to Python 3.13 only (avoids triple-counting across the matrix)
- `pyproject.toml`: Added `fail_under = 99` to `[tool.coverage.report]` so local runs also enforce the threshold

#### Dead code removal

Removed unreachable defensive guards that duplicated guarantees from `normalize_path` / `posixpath.normpath`:

- `base.py`: `startswith("/")` check after `normalize_path` (always adds leading `/`)
- `paths.py`: Trailing slash strip after `normpath` (normpath already strips it), empty segment check after `normalize_path` (normpath collapses them)
- `replace.py`: `lines_to_check <= 0` guard in `block_anchor_replacer` (anchor loop guarantees ≥3 lines)
- `query/executor.py`: Unreachable `case _: raise AssertionError` match fallbacks in `_execute_stage` and `_execute_transfer` (exhaustive match over all AST node types)
- `query/parser.py`: Unreachable `case _: raise AssertionError` match fallbacks in `_planned_methods` and `_render_mode`

#### New test files (4)

- `tests/test_query_parser.py` — 67 tests: tokenizer (quoted strings, escapes, unterminated, pipe/amp/paren tokens), parser syntax (empty query, unexpected tokens, grouped expressions, intersect/except subqueries), all builder commands (stat, delete, edit, write, mkdir, move, copy, mkconn, glob, grep, search, lsearch, vsearch, meetinggraph, graph traversal, sort, top, kinds), visibility/overwrite/kind-name parsing, flag splitting edge cases, `_render_mode` for every AST node type
- `tests/test_query_executor.py` — 48 tests: every `_execute_stage` branch (stat, delete, edit, write, mkdir, move, copy, mkconn, ls, tree, glob, grep, semantic/vector/lexical search, graph traversal, meeting graph, rank, sort, top, kinds, intersect, except), transfer with piped empty/non-empty candidates, visibility filtering, grep reading non-file candidates, union node merging
- `tests/test_query_render.py` — 16 tests: every render mode (content, action, stat, query_list, ls, tree), error rendering, multi-file content headers, all action verbs, ranked vs unranked list, stat metadata fields, unhandled mode assertion
- `tests/test_databricks_store.py` — 4 tests: ImportError guard when SDK missing, user_id filter injection, connect with host/token kwargs

#### Extended test files (8)

- `tests/test_embedding.py` — `_HAS_LANGCHAIN=False` ImportError guard
- `tests/test_patterns.py` — `compile_glob`/`match_glob` on invalid regex (`[z-a]`)
- `tests/test_base.py` — empty write batch, edit/move/copy without required args
- `tests/test_bm25_comparison.py` — BM25Scorer edge cases: empty IDF, zero-length doc, empty query terms, mismatched lengths, BM25Index with empty documents
- `tests/test_graph.py` — user-scoped neighborhood/meeting_subgraph, min_meeting_subgraph pruning and error handling
- `tests/test_grover_sync.py` — move, copy, mkconn, all graph methods (predecessors through hits), lexical_search, semantic/vector search error paths
- `tests/test_models.py` — `_stored_version_payload` error paths, `_reconstruct_file_version` (missing snapshot/version/hash mismatch), plan_file_write on directory, update_content on directory, null bytes in version_diff, both content+version_diff rejected, explicit content_hash
- `tests/test_client.py` — vector_store injection on existing filesystem

#### Coverage by module

| Module | Before | After |
|--------|--------|-------|
| `query/render.py` | 63% | 100% |
| `query/executor.py` | 65% | 99% |
| `query/parser.py` | 70% | 99% |
| `models.py` | 83% | 98% |
| `client.py` | 85% | 100% |
| `bm25.py` | 92% | 100% |
| `databricks_store.py` | 93% | 100% |
| `patterns.py` | 96% | 100% |
| `base.py` | 98% | 100% |
| `paths.py` | 99% | 100% |
| `replace.py` | 99% | 100% |
| `embedding.py` | 97% | 100% |
| **TOTAL** | **91%** | **99.11%** |

1779 total tests (was 1556), all passing. 99.11% coverage. ruff clean.

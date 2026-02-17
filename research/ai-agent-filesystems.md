# AI Agent Filesystems and File Management

## AI Agent Frameworks with File/Storage Abstractions

### LangChain Deep Agents

The most directly comparable system to Grover. Released late 2025, introduced a pluggable filesystem backend with a mount-like composite pattern.

- `BackendProtocol` with methods: `ls_info()`, `read()`, `grep_raw()`, `glob_info()`, `write()`, `edit()`
- Built-in backends: StateBackend (ephemeral), FilesystemBackend (local disk), StoreBackend (persistent via Redis/Postgres), sandbox backends (Modal, Daytona, Deno)
- `CompositeBackend` routes operations by path prefix — validates Grover's mount architecture
- Key insight from their blog: "a single interface through which an agent can flexibly store, retrieve, and update an infinite amount of context"

See [deepagents-analysis.md](deepagents-analysis.md) for full technical breakdown.

### LlamaIndex

Published "Files Are All You Need" — argues files are the primary AI agent interface. Identifies three critical capabilities: document parsing, scalable hybrid search (semantic + keyword), and file generation/editing. Remains focused on indexing/retrieval, not VFS.

### CrewAI

Provides 100+ open-source tools for agents but treats file operations as individual tools rather than a unified filesystem abstraction. No VFS layer.

### OpenHands (formerly OpenDevin)

Sandboxed Docker container architecture. Workspace directory mounted into sandbox. V1 SDK decouples into four packages (SDK, Tools, Workspace, Server). State managed through an event log recording commands, edits, and results. No VFS abstraction — raw filesystem access within containers.

### AutoGPT

Goal-driven automation with file read/write as individual tools. No filesystem abstraction layer.

### Gaps Across All Frameworks

- **No versioning.** Every framework treats files as current-state-only.
- **No knowledge graph.** `glob` and `grep` exist but no structural understanding of file relationships.
- **No semantic search over agent workspace.** Agents can `grep` but cannot do similarity search.
- **Fragmented ecosystems.** Each framework reinvents file operations independently.

---

## MCP (Model Context Protocol) File Servers

### Official MCP Filesystem Server

Node.js server providing: `read_file`, `write_file`, `create_directory`, `directory_tree`, `edit_file`, `get_file_info`, `list_directory`, `move_file`, `search_files`. Access restricted to predefined "allowed directories" with dynamic Roots-based updates.

### Broader MCP Ecosystem

- **Git MCP Server**: clone, commit, branch, diff, log, status, push, pull, merge, rebase
- **GitHub MCP Server**: full GitHub API (repos, PRs, issues, code search)
- **Third-party FS servers**: enhanced versions with large file handling, streaming writes, regex search, backup/recovery
- **GitMCP**: transforms GitHub projects into MCP-accessible documentation hubs

### MCP Security Concerns

CVE-2025-53109 and CVE-2025-53110 exposed directory traversal bypasses in the filesystem server. Security is a known weak point.

### Gaps in MCP

- **No versioning.** Thin wrapper over OS filesystem. Writes are destructive.
- **No semantic understanding.** `search_files` is pattern-matching only.
- **No relationship awareness.** Files treated as isolated entities.
- **Stateless.** No persistent state about managed files.

### Grover MCP Server Opportunity

A Grover MCP Server would expose:
- `write_file` with automatic versioning and rollback via `restore_version`
- `search_semantic` for meaning-based file search
- `query_graph` for "what files depend on X?" queries
- `diff` and `history` for version comparison
- Safe delete with trash and restore

This would make Grover accessible to Claude Desktop, Cursor, VS Code, and any MCP-compatible client.

---

## AI Code Agents and File Management

### Cursor

Sophisticated codebase indexing pipeline:
1. Files chunked locally
2. Hashes organized into Merkle tree (checked every 10 minutes for changes)
3. Embeddings generated server-side
4. Vectors stored in Turbopuffer (remote vector DB)
5. Path segments encrypted for privacy

Semantic search queries the vector DB, then actual code is read locally and sent to the LLM. No VFS layer — operates directly on OS filesystem.

### Aider

Deepest Git integration of any code agent:
- Every AI-suggested change gets an automatic commit
- **Repository map** uses tree-sitter to extract symbol definitions, building a structural overview of the entire repo
- Effectively a lightweight dependency graph
- Operates directly on local filesystem and Git

### Sourcegraph Cody

Search-first RAG architecture:
- SCIP-based semantic analysis for code navigation
- Vector embeddings for semantic search
- Multi-repository awareness
- Up to 1M-token context via Gemini
- Enterprise-only since mid-2025

### Augment Code

"Semantic dependency graph" indexing 450K+ file monorepos:
- Quantized vector search achieving sub-200ms latency on 100M+ line codebases
- 8x memory reduction through quantization
- Cloud-hosted (Google Cloud, BigTable, PubSub)

### Greptile

Semantic code search with key findings:
- **Raw code embeddings produce poor results** — code has different semantic properties than natural language
- **Code-to-NL translation before embedding yields 12% better results**
- Function-level chunking dramatically outperforms file-level chunking
- Combining semantic search with structural understanding (dependency graphs) is the winning approach

### Devin (Cognition Labs)

Sandboxed environment with shell, editor, and browser. Memory layer stores "vectorised snapshots of the code base plus a full replay timeline of every command, file diff, and browser tab."

### Gaps Across Code Agents

- **No reusable VFS library.** Every agent builds custom file management.
- **Versioning is Git-only or proprietary.** No application-level versioning without Git.
- **Dependency graphs are not reusable.** Aider's repo map, Augment's graph, Greptile's codegraph are all internal.
- **Semantic search is always cloud-hosted.** No lightweight local-first option.

### Grover's Unique Value for Code Agents

Grover provides all three capabilities (versioning, dependency graph, semantic search) as a single local-first library:
- File versioning with diff-based storage (lighter than Git for agent workspaces)
- Code dependency analysis (Python AST, JS/TS/Go via tree-sitter, extensible)
- Local HNSW semantic search (usearch, no cloud dependency)

---

## Sources

- [LangChain Deep Agents](https://github.com/langchain-ai/deepagents)
- [LangChain: How Agents Can Use Filesystems for Context Engineering](https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/)
- [LlamaIndex: Files Are All You Need](https://www.llamaindex.ai/blog/files-are-all-you-need)
- [MCP Filesystem Server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [OpenHands SDK (ICLR 2025)](https://arxiv.org/html/2511.03690v1)
- [Aider Repository Map](https://aider.chat/docs/repomap.html)
- [Cursor Codebase Indexing](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast)
- [Augment Code Real-Time Index](https://www.augmentcode.com/blog/a-real-time-index-for-your-codebase-secure-personal-scalable)
- [Greptile: Semantic Codebase Search](https://www.greptile.com/blog/semantic-codebase-search)
- [Sourcegraph Cody Architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase)
- [CrewAI Open Source](https://www.crewai.com/open-source)
- [Devin AI Agents 101](https://devin.ai/agents101)

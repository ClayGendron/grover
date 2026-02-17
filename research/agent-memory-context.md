# Agent Memory, Context Engineering, and Sandboxed Environments

## AI Agent Memory Systems

### Letta (formerly MemGPT)

The leader in agent memory systems. Architecture:

**Memory Blocks** — persistent, structured context sections pinned to the system prompt. Agents actively manage blocks (read, update, search). Blocks can be shared across multiple agents.

**Context Repositories (MemFS)** — git-backed filesystem of markdown files for agent memory. Agents manage their own progressive disclosure by:
- Reorganizing file hierarchies
- Updating frontmatter descriptions
- Moving files in/out of `system/` to control what gets pinned to context
- Automatic versioning with commit messages

**Agent File (.af)** — open file format for serializing stateful agents with persistent memory. Enables sharing, checkpointing, and version control of agents across frameworks.

**Three-tier memory model:**
1. **History** — immutable, global (conversation history)
2. **Memory** — persistent, mutable, agent-specific (learned facts, preferences)
3. **Scratchpad** — ephemeral, task-bounded (working notes)

### The AFS Paper ("Everything is Context", December 2025)

Formalizes the "agentic file system" concept. Key ideas:

- **Everything is a file** — documents, memories, tools, human inputs are nodes in a governed namespace
- **Five principles:** abstraction, modularity, separation of concerns, traceability/verifiability, composability
- **Three context tiers:** History (permanent), Memory (persistent/mutable), Scratchpad (transient)
- **File operations governed by:** metadata, transaction logs, access policies
- Implemented in the open-source AIGNE framework (early stage)

**This paper explicitly validates Grover's "everything is a file" philosophy** and extends it to the context engineering domain.

### LangChain Context Engineering

Uses the filesystem as overflow storage:
- Large tool results written to files
- Agents selectively `grep` for relevant sections
- Filesystem prevents context window saturation
- "Models are specifically trained to understand traversing filesystems"

### Gaps in Memory Systems

- **Letta's MemFS is memory-only.** No general-purpose file storage, versioning of arbitrary files, or code analysis.
- **No system combines agent memory with code understanding.** Letta handles memory; code agents handle code. None does both.
- **AFS paper is theoretical.** AIGNE implementation is early-stage. No production-ready library implements the full vision.

### Grover as Agent Memory Infrastructure

Grover could serve as the storage layer for agent memory systems. Its versioned filesystem already implements many AFS concepts:

| AFS Concept | Grover Implementation |
|-------------|----------------------|
| Everything is a file | Core principle — file paths are identity |
| Traceability | Version chain with diffs |
| Mounting heterogeneous sources | VFS mount system |
| Event-driven sync | EventBus: write file -> graph rebuilds -> embeddings re-index |
| Access policies | UserScopedFileSystem with sharing |

A Grover-backed agent memory system would add what Letta's MemFS lacks:
- Semantic search over memory files
- Dependency graph of memory relationships
- Database-backed persistence with multi-user sharing

---

## Context Engineering: The New Paradigm

### RAG to Context Evolution

RAG is transforming from "retrieval-augmented generation" into a "context engine." Focus has shifted from retrieval algorithms to systematic design of the end-to-end retrieval-context assembly-model reasoning pipeline.

### Martin Fowler on Context Engineering for Coding Agents

Confirms this is the dominant concern for production agent systems. The key challenge: assembling the right context from large codebases for each agent interaction.

### Grover as a Context Engine

Grover's event-driven architecture is inherently a context engineering pipeline:

```
File Write -> EventBus -> Graph Update -> Search Re-index
```

The three integrated layers (filesystem, graph, search) provide the "coherent infrastructure" that the AFS paper identifies as missing from existing systems.

---

## Sandboxed Execution Environments

### E2B

Market leader ($21M funding). Firecracker microVMs start in under 200ms. Complete filesystem operations within each sandbox. Adopted by Hugging Face, Perplexity, Manus, Fortune 500 companies. ~$0.05/hour per vCPU.

### Modal

Optimized for resource efficiency with aggressive container spin-down, causing 2-5+ second cold starts. Better for batch workloads than interactive agent sessions.

### Fly.io Sprites

Stateful sandbox environments with checkpoint/restore. Each Sprite gets a persistent filesystem surviving shutdown. Sub-second startup. Fly.io has declared ephemeral sandboxes "obsolete," pushing toward stateful sandboxes.

### OpenHands

Docker containers with workspace mounting. Sandbox exposes filesystem, terminal, and web interface. Event-sourced state management records all actions.

### Gaps in Sandboxed Environments

- **No versioned file access.** Agents use raw filesystem operations. No undo, history, or diff tracking.
- **No semantic search.** Must bring own search infrastructure.
- **Basic state persistence.** E2B sandboxes are ephemeral (24-hour max). Fly.io persists filesystem without versioning or querying.
- **No cross-sandbox knowledge sharing.** Each sandbox is isolated. Knowledge can't be queried across sessions.

### Grover in Sandboxes

Grover could run inside any sandbox, providing versioned operations, analysis, and search as a portable library:

- **E2B**: `DatabaseFileSystem` (pure DB) persists agent work across ephemeral sessions by connecting to an external database
- **Fly.io Sprites**: Grover adds versioning and search to Sprites' persistent filesystem
- **Multi-agent coordination**: `UserScopedFileSystem` with sharing enables agents in separate sandboxes to share files with controlled permissions

---

## Emerging Patterns

### Pattern 1: "Everything is a File" as Universal Agent Interface

Multiple independent sources converge:
- AFS paper (Dec 2025) formalizes it academically
- LangChain Deep Agents implements it with pluggable backends
- LlamaIndex argues "files are all you need"
- Letta's MemFS uses files for agent memory
- AgentFS builds agent runtime on SQLite filesystem

**Strong validation of Grover's foundational design.**

### Pattern 2: Graph-RAG for Codebases

- **Code-Graph-RAG** — knowledge graphs from codebases via tree-sitter, stored in Memgraph, queried via LLM-generated Cypher. Works as MCP server with Claude Code.
- **Microsoft GraphRAG** — graph-based retrieval outperforms flat vector search for complex reasoning.
- **Augment Code** — semantic dependency graphs of entire codebases.
- **Greptile** — codegraphs for code review (82% bug catch rate).

Grover already has this: rustworkx-based knowledge graph with auto-populated code analyzers.

### Pattern 3: Semantic Code Search Requires Special Treatment

Greptile's research:
- Raw code embeddings produce poor results
- **Translating code to NL descriptions before embedding yields 12% better results**
- Function-level chunking outperforms file-level chunking
- Combining semantic search with structural understanding is the winning approach

**Enhancement opportunity for Grover:** Add NL-description generation before embedding.

### Pattern 4: Context Engineering as Paradigm

The focus has shifted from "how to retrieve" to "how to assemble context." Grover's event-driven three-layer architecture (FS + graph + search) is inherently a context engineering pipeline.

### Pattern 5: Agent-Native Filesystems Are Emerging

The space is nascent but growing:
- **AgentFS** (Turso) — SQLite-backed, audit trail, FUSE
- **Deep Agents backends** — pluggable, composite routing
- **Letta MemFS** — git-backed memory filesystem
- **AFS paper** — formal theoretical framework

None provides Grover's full stack. They each address one or two concerns; Grover addresses all of them.

---

## Sources

- [Letta Memory Overview](https://docs.letta.com/guides/agents/memory/)
- [Letta Context Repositories (MemFS)](https://www.letta.com/blog/context-repositories)
- [Letta Agent File (.af)](https://www.letta.com/blog/agent-file)
- [Letta Memory Blocks](https://www.letta.com/blog/memory-blocks)
- [AFS Paper: Everything is Context](https://arxiv.org/abs/2512.05470)
- [RAG-to-Context 2025 Review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [Martin Fowler: Context Engineering for Coding Agents](https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html)
- [E2B Sandbox Platform](https://e2b.dev/)
- [E2B $21M Funding](https://siliconangle.com/2025/07/28/e2b-shares-vision-sandboxed-cloud-environments-every-ai-agent-raising-21m-funding/)
- [Fly.io Sprites.dev](https://simonwillison.net/2026/Jan/9/sprites-dev/)
- [Modal Sandbox Comparison](https://modal.com/blog/top-code-agent-sandbox-products)
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LangChain: Filesystems for Context Engineering](https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/)

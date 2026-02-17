# deepagents Analysis (v0.4.1)

## What It Is

**deepagents** is an open-source Python package built on **LangGraph** and **LangChain** that enables creating sophisticated AI agents with four key capabilities:

1. **Planning tool** — built-in to-do list management
2. **Sub-agents** — spawn ephemeral, isolated task-specific agents
3. **File system abstraction** — virtual filesystem with multiple storage backends
4. **Structured prompting** — detailed, context-aware system instructions

Inspired by Claude Code. Designed for "deep agents" that plan and act over complex, multi-step tasks.

## Architecture

### Three-Layer Design

1. **Backends Layer** — pluggable storage implementations (`BackendProtocol`)
2. **Middleware Layer** — cross-cutting concerns (filesystem, memory, skills, subagents, summarization)
3. **Agent Layer** — LangChain agent wrapping all middleware with LLM integration

### Entry Point

`create_deep_agent()` in `graph.py` — configures and returns a `CompiledStateGraph` (LangGraph runnable).

```python
create_deep_agent(
    model: str | BaseChatModel | None = None,  # Default: claude-sonnet-4-5
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph
```

## Filesystem Backends

### BackendProtocol

All backends implement a unified interface:

**File Operations:**
- `ls_info(path)` — list directory contents
- `read(file_path, offset, limit)` — read files with pagination
- `write(file_path, content)` — create new files
- `edit(file_path, old_string, new_string, replace_all)` — in-place string replacement
- `grep_raw(pattern, path, glob)` — search files (literal text, not regex)
- `glob_info(pattern, path)` — find files by glob pattern
- `upload_files(files)` / `download_files(paths)` — batch operations

**Response Types:**
- `WriteResult` — `{error, path, files_update}`
- `EditResult` — `{error, path, files_update, occurrences}`
- `FileInfo` — `{path, is_dir, size, modified_at}`
- `GrepMatch` — `{path, line, text}`

### Backend Implementations

#### 1. StateBackend (`backends/state.py`)
- Stores files in LangGraph agent state (ephemeral)
- Lives only for the duration of a conversation thread
- Automatically checkpointed after each agent step
- **Use case:** local, stateless agent conversations

#### 2. FilesystemBackend (`backends/filesystem.py`)
- Reads/writes files directly from the filesystem
- Two modes:
  - `virtual_mode=False` — absolute paths allowed (no security)
  - `virtual_mode=True` — virtual path root, blocks traversal (`..`, `~`)
- Max file size limits (10MB default)
- **Use case:** local development CLIs

#### 3. LocalShellBackend (`backends/local_shell.py`)
- Extends FilesystemBackend with unrestricted shell execution
- Implements `SandboxBackendProtocol` (adds `execute()` method)
- Commands run with user's permissions, no sandboxing
- **Use case:** trusted dev environments only

#### 4. StoreBackend (`backends/store.py`)
- Adapter for LangGraph's `BaseStore` (persistent, cross-thread storage)
- Uses namespaces for isolation (e.g., per-user, per-assistant)
- Factory function for namespace resolution: `NamespaceFactory`
- **Use case:** production multi-user systems, persistent storage

#### 5. CompositeBackend (`backends/composite.py`)
- Routes file operations by path prefix to different backends
- Example: `/temp/` -> StateBackend, `/memories/` -> StoreBackend
- Longest-first prefix matching
- Aggregates root directory listings from all backends
- **Use case:** mixed storage strategies (temporary vs. persistent)

#### 6. BaseSandbox (`backends/sandbox.py`)
- Abstract base for sandboxed execution environments
- Implements all `BackendProtocol` methods by delegating to `execute()`
- Uses shell commands with base64 encoding for safety
- Foundation for container/VM backends

### SandboxBackendProtocol

Extension of `BackendProtocol`:
- Adds `execute(command: str) -> ExecuteResponse`
- Returns: `{output, exit_code, truncated}`
- Designed for isolated execution environments (containers, VMs, remote hosts)

## Middleware Stack

### Built-In Middleware

1. **TodoListMiddleware** — `write_todos()`, `add_todo()`, `update_todo()` tools for planning
2. **FilesystemMiddleware** — file operations (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`)
3. **SubAgentMiddleware** — `task()` tool for spawning sub-agents
4. **SummarizationMiddleware** — context management via conversation history offloading (triggers at 85% context utilization)
5. **AnthropicPromptCachingMiddleware** — token optimization via Claude's prompt caching
6. **PatchToolCallsMiddleware** — tool call reliability fixes

### Optional Middleware

7. **MemoryMiddleware** — loads `AGENTS.md` memory files (persistent context)
8. **SkillsMiddleware** — loads `SKILL.md` files (on-demand workflows)
9. **HumanInTheLoopMiddleware** — pauses for human approval at specified tool calls

## Agent Tools Exposed

The `FilesystemMiddleware` exposes these tools to agents:

| Tool | Description |
|------|-------------|
| `ls(path)` | List files in directory |
| `read_file(file_path, offset, limit)` | Read files with pagination |
| `write_file(file_path, content)` | Create new files |
| `edit_file(file_path, old_string, new_string)` | Replace strings in files |
| `glob(pattern, path)` | Find files by glob pattern |
| `grep(pattern, glob, output_mode)` | Search files for literal text |
| `execute(command)` | Run shell commands (if sandbox backend) |

## Security Model

| Backend | Isolation | Risk | Safeguard |
|---------|-----------|------|-----------|
| StateBackend | Ephemeral | Low | Conversation-scoped only |
| FilesystemBackend | None | High | Only in dev; use `virtual_mode=True` |
| LocalShellBackend | None | Critical | Dev only; enable Human-in-the-Loop |
| StoreBackend | Namespace | Medium | Proper namespace factory needed |
| CompositeBackend | Via routes | Varies | Depends on routed backends |

## Integration Points with Grover

### What deepagents has that Grover doesn't
- Agent orchestration (LangGraph state machine, tool calling loop)
- Sub-agent spawning with isolated context
- Context window management (summarization, offloading)
- Todo/planning tools
- Shell execution in sandboxes
- Middleware pipeline for cross-cutting concerns

### What Grover has that deepagents doesn't
- **Versioning** — deepagents stores files, not versions. No history, no diffs, no rollback.
- **Trash/restore** — deletions in deepagents are permanent.
- **Knowledge graph** — no dependency tracking, no code analysis.
- **Semantic search** — no embedding-based search over stored files.
- **Multi-user with sharing** — `UserScopedFileSystem` with `@shared` namespace.
- **Event-driven sync** — write a file, graph rebuilds, embeddings re-index.

### Integration Strategy

Grover can be wrapped as a custom `BackendProtocol` implementation:

```python
class GroverBackend:
    """deepagents BackendProtocol backed by Grover."""

    def __init__(self, grover: Grover):
        self.grover = grover

    async def ls_info(self, path: str) -> list[FileInfo]: ...
    async def read(self, file_path: str, offset: int, limit: int) -> ReadResponse: ...
    async def write(self, file_path: str, content: str) -> WriteResult: ...
    async def edit(self, file_path: str, old: str, new: str, replace_all: bool) -> EditResult: ...
    async def grep_raw(self, pattern: str, path: str, glob: str) -> list[GrepMatch]: ...
    async def glob_info(self, pattern: str, path: str) -> list[FileInfo]: ...
```

This would give any LangGraph deep agent access to Grover's versioned filesystem, graph, and search — with zero changes to deepagents itself.

Additional Grover-specific tools could be exposed as custom middleware:

- `search_semantic(query, k)` — vector similarity search
- `list_versions(path)` / `restore_version(path, version)` — version management
- `dependencies(path)` / `dependents(path)` / `impacts(path)` — graph queries
- `trash()` / `restore_from_trash(path)` — soft-delete management

## Source Locations

```
.venv/lib/python3.13/site-packages/deepagents/
├── __init__.py              # Public API
├── graph.py                 # create_deep_agent()
├── backends/
│   ├── protocol.py          # BackendProtocol
│   ├── state.py             # StateBackend
│   ├── filesystem.py        # FilesystemBackend
│   ├── local_shell.py       # LocalShellBackend
│   ├── store.py             # StoreBackend
│   ├── composite.py         # CompositeBackend
│   └── sandbox.py           # BaseSandbox
└── middleware/
    ├── filesystem.py         # FilesystemMiddleware
    ├── todo_list.py          # TodoListMiddleware
    ├── sub_agent.py          # SubAgentMiddleware
    ├── summarization.py      # SummarizationMiddleware
    ├── memory.py             # MemoryMiddleware
    └── skills.py             # SkillsMiddleware
```

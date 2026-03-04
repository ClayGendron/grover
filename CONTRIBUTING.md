# Contributing to Grover

Thanks for your interest in contributing to Grover! This guide covers how to set up your development environment, run the test suite, and submit changes.

## Getting started

### Prerequisites

- **Python 3.12+** (3.13 recommended)
- **[uv](https://docs.astral.sh/uv/)** for package management

### Setup

Clone the repo and install in editable mode with dev dependencies:

```bash
git clone https://github.com/ClayGendron/grover.git
cd grover
uv venv
uv pip install -e ".[all]"
uv pip install --group dev
```

This installs Grover with all optional extras (search, tree-sitter, PostgreSQL) plus the development tools (pytest, pre-commit, etc.).

### Verify your setup

```bash
uvx ruff check src/
uvx ty check src/
uv run pytest
```

All three should pass cleanly.

## Development workflow

### Making changes

1. Create a branch from `main`.
2. Make your changes.
3. Run the quality checks (see below).
4. Open a pull request against `main`.

Issues are encouraged for larger changes or design discussions, but not required for straightforward fixes or additions. If you're unsure whether something warrants an issue, go ahead and open a PR — we can discuss there.

### Quality checks

Every PR should pass all three checks:

**Linting** (ruff):

```bash
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
```

To auto-fix lint issues and formatting:

```bash
uvx ruff check --fix src/ tests/
uvx ruff format src/ tests/
```

**Type checking** (ty):

```bash
uvx ty check src/
```

**Tests** (pytest):

```bash
uv run pytest
```

To run with coverage:

```bash
uv run pytest --cov
```

### Writing tests

Tests live in `tests/` and use pytest with `pytest-asyncio` (async mode is set to `auto`). A few conventions:

- Test files mirror the source structure: `src/grover/backends/local.py` is tested in `tests/test_local_fs.py`.
- Shared fixtures are in `tests/conftest.py` (in-memory SQLite engines, async sessions, etc.).
- Use the `@pytest.mark.slow` and `@pytest.mark.integration` markers when appropriate.
- Prefer in-memory SQLite and fake backends for unit tests; use temp directories for LocalFileSystem tests.

### Commit messages

Keep commit messages concise and descriptive. Focus on the *why*, not the *what*:

```
Fix version reconstruction when snapshot is at boundary

The snapshot interval check was off-by-one, causing reconstruction
to miss the boundary snapshot and fall back to a stale one.
```

## Project structure

```
src/grover/
├── client.py             # Async core (GroverAsync)
├── sync_client.py        # Sync wrapper (Grover)
├── ref.py                # Ref frozen dataclass
├── worker.py             # BackgroundWorker for async indexing
├── exceptions.py         # Error hierarchy
├── permissions.py        # Permission enum
├── mount.py              # Mount dataclass + MountRegistry
├── api/                  # GroverAsync mixin implementations
│   ├── context.py        # GroverContext shared state
│   ├── mounting.py       # Mount lifecycle
│   ├── file_ops.py       # File CRUD
│   ├── search_ops.py     # Search and query operations
│   └── ...
├── backends/             # Filesystem implementations
│   ├── protocol.py       # GroverFileSystem + capability protocols
│   ├── database.py       # Pure-database backend
│   ├── local.py          # Disk + SQLite backend
│   └── user_scoped.py    # User-scoped multi-tenant backend
├── providers/            # Pluggable provider families
│   ├── graph/            # rustworkx graph provider
│   ├── search/           # Vector stores (local, Pinecone, Databricks)
│   ├── embedding/        # Embedding providers (OpenAI, LangChain)
│   ├── storage/          # Disk storage provider
│   ├── versioning/       # Version + diff providers
│   └── chunks/           # Code chunk provider
├── results/              # Result types (operations + search)
├── models/               # SQLModel database models
├── analyzers/            # Language-specific code analyzers
├── util/                 # Shared utilities (paths, content, patterns)
└── integrations/         # Framework integrations (deepagents, langchain)
```

For a deeper dive into patterns and design principles, see [docs/architecture.md](docs/architecture.md).

## Tooling reference

| Tool | Purpose | Command |
|------|---------|---------|
| **ruff** | Linting and formatting | `uvx ruff check src/` / `uvx ruff format src/` |
| **ty** | Type checking | `uvx ty check src/` |
| **pytest** | Test runner | `uv run pytest` |
| **uv** | Package management | `uv pip install ...` |

Note: ruff and ty are invoked via `uvx` (not installed as pip packages). Tests and package commands use `uv run` to ensure the correct virtualenv.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold a welcoming, inclusive, and respectful environment for everyone.

In short: be kind, be constructive, and assume good intent. If you experience or witness unacceptable behavior, please reach out to the maintainers.

## Questions?

If you're not sure about something, open an issue or start a discussion. We're happy to help.

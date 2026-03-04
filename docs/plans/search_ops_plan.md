# Plan: Add `glob`, `grep`, and `tree` to Grover Filesystem

## Context

The filesystem layer has comprehensive CRUD, versioning, and trash operations but no search/query capabilities. `list_dir` only shows immediate children — there's no way to recursively find files by pattern, search content, or view a full directory tree. These are essential for navigating and querying data stored in Grover.

**Correctness guarantee**: Glob uses SQL LIKE as a performance optimization only — `fnmatch` post-filtering ensures every glob pattern works correctly. Grep uses Python's `re` module for full regex support. No "best effort" approximations.

## Files to Modify

| File | Changes |
|------|---------|
| `src/grover/fs/types.py` | Add `GlobResult`, `GrepMatch`, `GrepResult`, `TreeResult` dataclasses |
| `src/grover/fs/utils.py` | Add `glob_to_sql_like()` and `match_glob()` helpers |
| `src/grover/fs/base.py` | Add `glob()`, `grep()`, `tree()` methods with shared SQL logic |
| `src/grover/fs/local_fs.py` | Override all three to scan disk (like `list_dir` does) |
| `src/grover/fs/protocol.py` | Add protocol signatures (no `session` param) |
| `src/grover/fs/vfs.py` | Add cross-mount aggregation (like `list_trash`) |
| `src/grover/fs/__init__.py` | Export new types |
| `tests/test_search_ops.py` | New test file for all three operations |

---

## Step 1: New Result Types (`types.py`)

```python
@dataclass
class GlobResult:
    success: bool
    message: str
    entries: list[FileInfo] = field(default_factory=list)
    pattern: str = ""
    path: str = "/"

@dataclass
class GrepMatch:
    file_path: str
    line_number: int        # 1-indexed
    line_content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)

@dataclass
class GrepResult:
    success: bool
    message: str
    matches: list[GrepMatch] = field(default_factory=list)
    pattern: str = ""
    path: str = "/"
    files_searched: int = 0
    files_matched: int = 0
    truncated: bool = False

@dataclass
class TreeResult:
    success: bool
    message: str
    entries: list[FileInfo] = field(default_factory=list)
    path: str = "/"
    total_files: int = 0
    total_dirs: int = 0
```

---

## Step 2: Utils (`utils.py`)

Add two helpers for glob pattern matching:

- **`glob_to_sql_like(pattern, base_path)`** — Translate glob to SQL LIKE for DB pre-filtering. `*` → `%`, `?` → `_`, `**` → `%`. Returns `None` for `[seq]` patterns (caller loads all paths under base_path instead). This is a **performance optimization only** — correctness comes from the `fnmatch` post-filter.
- **`match_glob(path, pattern, base_path)`** — Authoritative match check. Uses `fnmatch` for simple patterns, `PurePosixPath.match()` for `**` patterns (Python 3.13 native support).

---

## Step 3: BaseFileSystem (`base.py`)

New section `# Search / Query Operations` after Directory Operations.

### `glob(pattern, path="/", *, session) → GlobResult`
1. Validate/normalize `path`. Verify it exists and is a directory (if not `/`).
2. Compose full pattern from `base_path + pattern`.
3. Try `glob_to_sql_like()` for SQL pre-filter optimization.
4. Query `grover_files` with `WHERE path LIKE ... AND deleted_at IS NULL`.
5. **Post-filter with `match_glob()` in Python** — this ensures full correctness for all glob patterns.
6. Convert to `FileInfo` via `_file_to_info()`.

### `grep(pattern, path="/", *, ..., session) → GrepResult`

Full signature with all options:

```python
async def grep(
    self,
    pattern: str,
    path: str = "/",
    *,
    glob_filter: str | None = None,    # limit to files matching glob (e.g. "*.py")
    case_sensitive: bool = True,        # case-sensitive matching
    fixed_string: bool = False,         # treat pattern as literal (no regex)
    invert: bool = False,              # return non-matching lines (-v)
    word_match: bool = False,          # match whole words only (-w)
    context_lines: int = 0,            # lines before/after each match (-C)
    max_results: int = 100,            # global match cap
    max_results_per_file: int | None = None,  # per-file match cap (-m)
    count_only: bool = False,          # return counts, no match content (-c)
    files_only: bool = False,          # return file paths only, stop at first match (-l)
    session: AsyncSession,
) -> GrepResult:
```

Implementation:
1. Validate/normalize `path`.
2. Build the compiled regex:
   - If `fixed_string`: `re.escape(pattern)`
   - If `word_match`: wrap in `\b...\b`
   - If not `case_sensitive`: `re.IGNORECASE`
   - Catch `re.error` → error result.
3. Get candidate files: use `self.glob()` if `glob_filter` provided, otherwise query all non-deleted non-directory files under `path`.
4. For each file:
   - Read via `_read_content()`, skip binary extensions.
   - Search line-by-line with compiled regex.
   - If `invert`: include lines that do NOT match.
   - If `files_only`: stop after first match in file, record path only.
   - If `count_only`: count matches per file, don't populate `line_content`.
   - If `max_results_per_file`: break inner loop at that cap.
   - Build `GrepMatch` with context lines (`context_before`, `context_after`).
5. Respect `max_results` global cap. Track `files_searched` / `files_matched`. Set `truncated` if capped.

### `tree(path="/", *, max_depth, session) → TreeResult`
1. Validate/normalize `path`. Verify exists and is directory.
2. Query `WHERE path.startswith(base + "/") AND deleted_at IS NULL`.
3. Filter by depth: count `/` separators in path relative to base. `depth <= max_depth`.
4. Sort by path, count files/dirs.

---

## Step 4: LocalFileSystem Overrides (`local_fs.py`)

Overrides are needed because disk may contain files not in the DB (created by git, IDE, etc.) — same reason `list_dir` has an override.

### `glob` override
- Use `pathlib.Path.glob(pattern)` on disk via `asyncio.to_thread()`.
- Skip dotfiles (matching `list_dir` convention).
- Convert to virtual paths via `_to_virtual_path()`.
- Enrich with DB metadata where available.

### `grep` override
- Get candidates from disk via `self.glob()` or `Path.rglob("*")`.
- Use disk-based `is_binary_file()` check (more accurate than extension-only).
- Read content via `_read_content()` (reads from disk).
- All grep options (fixed_string, invert, word_match, count_only, files_only, max_results_per_file) work identically — the search logic is shared, only file discovery differs.

### `tree` override
- Walk disk with `os.walk()` in `asyncio.to_thread()` (natural depth control).
- Skip dotfiles/dirs. Enrich with DB metadata.
- Sort by path.

---

## Step 5: Protocol (`protocol.py`)

Add under new `# Search / Query Operations` section:

```python
async def glob(self, pattern: str, path: str = "/") -> GlobResult: ...

async def grep(
    self,
    pattern: str,
    path: str = "/",
    *,
    glob_filter: str | None = None,
    case_sensitive: bool = True,
    fixed_string: bool = False,
    invert: bool = False,
    word_match: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
    max_results_per_file: int | None = None,
    count_only: bool = False,
    files_only: bool = False,
) -> GrepResult: ...

async def tree(self, path: str = "/", *, max_depth: int | None = None) -> TreeResult: ...
```

No `session` in protocol signatures (VFS injects sessions — existing convention).

---

## Step 6: VFS (`vfs.py`)

All three methods support **cross-mount aggregation** when `path="/"` (iterate all mounts, aggregate results, prefix paths) — same pattern as `list_trash`. When `path` falls within a specific mount, dispatch to that single mount.

- **glob**: Aggregate `entries` across mounts, prefix with `_prefix_file_info`.
- **grep**: Aggregate `matches` across mounts, prefix `file_path`. Pass `remaining = max_results - accumulated` to each mount to respect the global cap. Propagate `truncated`.
- **tree**: At root, include mount roots as directory entries. Pass `max_depth - 1` to children (the mount root consumes one level).

---

## Step 7: Exports (`__init__.py`)

Add `GlobResult`, `GrepMatch`, `GrepResult`, `TreeResult` to imports and `__all__`.

---

## Implementation Order

1. `types.py` — no dependencies
2. `utils.py` — glob helpers
3. `base.py` — `tree` (simplest), then `glob`, then `grep`
4. `local_fs.py` — overrides
5. `protocol.py` — signatures
6. `vfs.py` — aggregation
7. `__init__.py` — exports
8. Tests

## Verification

1. **Type checking**: `uvx ty check src/`
2. **Linting**: `uvx ruff check src/`
3. **Unit tests**: `python -m pytest tests/test_search_ops.py -v`
4. **Full suite** (no regressions): `python -m pytest tests/ -v`

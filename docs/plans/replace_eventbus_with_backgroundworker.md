# Plan: Replace EventBus with BackgroundWorker

## Context

The EventBus is pub/sub machinery wrapping what is really just "run this function in the background." There's exactly one handler per event type, no fan-out, no dynamic subscription. The useful parts are per-path debouncing and lifecycle management (drain/flush). The rest — EventType enum, FileEvent dataclass with optional fields for two different concerns, handler registration, dispatch routing, nested emit detection — is ceremony.

The user's design principles:
- **FS is source of truth.** All data lives in the filesystem (disk or DB).
- **Functions return immediately.** Background tasks update derived stores (graph, search, chunks).
- **Two kinds of background work:** file processing (analyze, chunk, index) and connection projection (update in-memory graph). These are different concerns that were forced into one FileEvent type.
- **Mount-level handler registration** for external store integration (future follow-up, not in this plan).
- **Graph lazy composition** is a separate follow-up task.

### What changes

Replace `EventBus` + `FileEvent` + `EventType` with a `BackgroundWorker` that does debounced background task scheduling. Facade methods call processing functions directly via the worker instead of emitting events. The `_analyze_and_integrate` pipeline adds graph edges directly after DB commit instead of through deferred connection events.

### Files affected

**Source (8 files):**
- `src/grover/worker.py` — new (BackgroundWorker + IndexingMode)
- `src/grover/events.py` — delete
- `src/grover/facade/context.py` — worker replaces event_bus
- `src/grover/facade/indexing.py` — handlers become processing methods
- `src/grover/facade/file_ops.py` — emit → worker.schedule
- `src/grover/facade/connections.py` — emit → worker.schedule_immediate
- `src/grover/facade/version_trash.py` — emit → worker.schedule
- `src/grover/_grover_async.py` — create worker, remove handler registration

**Tests (6 files):**
- `tests/test_worker.py` — new (replaces test_events.py)
- `tests/test_events.py` — delete
- `tests/test_background_indexing.py` — update refs
- `tests/test_connection_service.py` — update refs
- `tests/test_session_batching.py` — update refs
- `tests/test_chunk_migration.py` — update refs

**Docs/config (4 files):**
- `src/grover/__init__.py` — update IndexingMode import path
- `src/grover/_grover.py` — update IndexingMode import path
- `CLAUDE.md` — update repo layout (events.py → worker.py)
- `docs/architecture.md` — update event dispatch section

---

## Phase 1: Create BackgroundWorker

### Summary
Create `BackgroundWorker` as a standalone module alongside the existing EventBus. Nothing else changes — EventBus still works.

### Files to create

**`src/grover/worker.py`** (~100 lines):

```python
class IndexingMode(Enum):
    BACKGROUND = "background"
    MANUAL = "manual"

class BackgroundWorker:
    """Debounced background task scheduler.

    Replaces EventBus. Facade methods schedule work directly
    instead of emitting events through pub/sub.
    """

    def __init__(self, *, indexing_mode, debounce_delay=0.1): ...

    def schedule(self, key: str, coro_factory: Callable[[], Coroutine]) -> None:
        """Schedule debounced background work keyed by path.
        Coalesces rapid calls to same key — only latest runs."""

    def schedule_immediate(self, coro: Coroutine) -> None:
        """Run immediately in background, no debounce."""

    def cancel(self, key: str) -> None:
        """Cancel pending debounced work for key.
        Used by delete() to cancel pending writes for a deleted file."""

    async def drain(self) -> None:
        """Fire all pending work immediately, await all active tasks.
        Loops until settled (tasks may schedule new work)."""

    @property
    def indexing_mode(self) -> IndexingMode: ...

    @property
    def pending_count(self) -> int: ...
```

Key implementation details:
- `schedule()` is **sync** (not async) — it just sets a timer via `loop.call_later()` and returns instantly. This is why the caller doesn't wait.
- `schedule_immediate()` is also sync — `loop.create_task()` and return.
- Both are no-ops when `indexing_mode == MANUAL`.
- `_create_task()` wraps coroutines in try/except — exceptions logged, never propagated. Same fault isolation as current EventBus.
- Internal state: `_pending: dict[str, tuple[Callable, TimerHandle | None]]`, `_active_tasks: set[Task]` — same structure as EventBus.

**`tests/test_worker.py`** — unit tests for BackgroundWorker in isolation:

| Test | What it verifies |
|------|-----------------|
| `test_schedule_runs_in_background` | Scheduled work executes after drain |
| `test_debounce_coalesces` | Rapid calls to same key → only latest runs |
| `test_debounce_different_keys` | Different keys run independently |
| `test_cancel_removes_pending` | cancel(key) prevents execution |
| `test_schedule_immediate_no_debounce` | Runs immediately without waiting for delay |
| `test_manual_mode_is_noop` | schedule/schedule_immediate do nothing in MANUAL |
| `test_drain_fires_all_pending` | drain() flushes everything |
| `test_drain_handles_cascading_work` | Work scheduled during drain is also drained |
| `test_exception_isolation` | Failing task doesn't crash worker or other tasks |
| `test_pending_count` | Tracks pending + active correctly |

### Acceptance criteria
- `BackgroundWorker` exists and passes all unit tests
- EventBus still works, nothing else changes
- All existing tests pass

---

## Phase 2: Replace EventBus with BackgroundWorker

### Summary
Convert all emit sites to use `worker.schedule()` / `schedule_immediate()`. Convert all handlers to direct processing methods. Simplify `_analyze_and_integrate` to update graph directly instead of through deferred events. Wire BackgroundWorker into GroverAsync.

### Files to modify

#### `src/grover/_grover_async.py`

- Import `BackgroundWorker` from `worker` instead of `EventBus` from `events`
- Create `BackgroundWorker` in `__init__` instead of `EventBus`
- **Remove all 6 `event_bus.register()` calls** — no handler registration needed
- Pass `worker` to GroverContext instead of `event_bus`

#### `src/grover/facade/context.py`

- Replace `event_bus: EventBus` field with `worker: BackgroundWorker`
- **Delete `emit()` method** — no longer needed
- Update `drain()` to call `self.worker.drain()`
- Update imports

#### `src/grover/facade/file_ops.py`

Replace every `await self._ctx.emit(FileEvent(...))` with direct worker calls:

| Method | Current emit | New call |
|--------|-------------|----------|
| `write()` | `emit(FILE_WRITTEN, path, content, user_id)` | `worker.schedule(path, lambda p=path, c=content, u=user_id: self._process_write(p, c, u))` |
| `edit()` | `emit(FILE_WRITTEN, path, user_id)` | `worker.schedule(path, lambda p=path, u=user_id: self._process_write(p, None, u))` |
| `delete()` | `emit(FILE_DELETED, path, user_id)` | `worker.cancel(path)` then `worker.schedule_immediate(self._process_delete(path, user_id))` |
| `move()` same-mount | `emit(FILE_MOVED, dest, old_path=src)` | `worker.cancel(src)` then `worker.schedule_immediate(self._process_move(src, dest, user_id))` |
| `move()` cross-mount | `emit(FILE_MOVED, dest, old_path=src)` | same as above |
| `copy()` same-mount | `emit(FILE_WRITTEN, dest)` | `worker.schedule(dest, lambda p=dest, u=user_id: self._process_write(p, None, u))` |
| `copy()` cross-mount | `emit(FILE_WRITTEN, dest)` | same as above |

Note: lambdas use default args (`p=path`) to avoid late-binding closure issues.

Remove `FileEvent` and `EventType` imports.

#### `src/grover/facade/connections.py`

| Method | Current emit | New call |
|--------|-------------|----------|
| `add_connection()` | `emit(CONNECTION_ADDED, ...)` | `worker.schedule_immediate(self._process_connection_added(source, target, type, weight))` |
| `delete_connection()` | `emit(CONNECTION_DELETED, ...)` | `worker.schedule_immediate(self._process_connection_deleted(source, target))` |

Remove `FileEvent` and `EventType` imports.

#### `src/grover/facade/version_trash.py`

| Method | Current emit | New call |
|--------|-------------|----------|
| `restore_version()` | `emit(FILE_RESTORED, path)` | `worker.schedule(path, lambda p=path, u=user_id: self._process_write(p, None, u))` |
| `restore_from_trash()` | `emit(FILE_RESTORED, path)` | same |

Remove `FileEvent` and `EventType` imports.

#### `src/grover/facade/indexing.py` — the big refactor

**Rename handlers to processing methods with direct parameters:**

| Old handler | New method | Parameters |
|------------|-----------|------------|
| `_on_file_written(event)` | `_process_write(path, content, user_id)` | Extracts path/content/user_id from params instead of FileEvent |
| `_on_file_deleted(event)` | `_process_delete(path, user_id)` | Same logic, direct params |
| `_on_file_moved(event)` | `_process_move(old_path, new_path, user_id)` | **Simplified: calls `_process_delete(old) + _process_write(new)`** |
| `_on_file_restored(event)` | *(deleted — callers use `_process_write` directly)* | |
| `_on_connection_added(event)` | `_process_connection_added(source, target, type, weight)` | Direct graph.add_edge() |
| `_on_connection_deleted(event)` | `_process_connection_deleted(source, target)` | Direct graph.remove_edge() |

**Simplify `_analyze_and_integrate`** — the biggest win:

Current flow (lines 165-266):
1. Collect `deferred_events: list[FileEvent]` during session
2. After session commits, emit CONNECTION_ADDED events
3. `_on_connection_added` handler adds edges to graph

New flow:
1. Collect `edges_to_project: list[tuple]` during session
2. After session commits, **call `graph.add_edge()` directly**
3. No events, no handlers, no nested dispatch

This eliminates: the deferred events list, the post-commit emit loop, the `_dispatching` flag concept, and the nested emit handling in EventBus.

**`_process_move` simplification:**

Current `_on_file_moved` duplicates delete cleanup logic (lines 72-97). New `_process_move` reuses:
```python
async def _process_move(self, old_path, new_path, user_id=None):
    if self._ctx.meta_fs is None:
        return
    if old_path and "/.grover/" not in old_path:
        await self._process_delete(old_path)
    if "/.grover/" not in new_path:
        await self._process_write(new_path, None, user_id)
```

Remove `FileEvent`, `EventType` imports. Remove `from grover.events import ...`.

### Tests to update

**`tests/test_background_indexing.py`:**
- Remove `from grover.events import EventBus, EventType, FileEvent, IndexingMode`
- Add `from grover.worker import BackgroundWorker, IndexingMode`
- `TestBackgroundDispatch` class (lines 55-280): **Rewrite entirely** — these test EventBus directly. Replace with equivalent BackgroundWorker tests (or delete if covered by test_worker.py).
- Integration tests (TestIntegrationBackgroundIndexing, ~line 282+): Update `_ctx.event_bus` refs to `_ctx.worker`. These tests write files and check graph/search state — the pattern (write → flush → assert) stays the same.
- `IndexingMode.MANUAL` test: same pattern, different import path.

**`tests/test_connection_service.py`:**
- Remove FileEvent/EventType imports
- The `_collecting_handler` pattern (line 70) and all `event_bus.register()` calls (lines 372, 518, 594): **Remove**. Instead of collecting events and asserting on FileEvent fields, assert on graph state directly (graph.has_edge, graph.get_edge).
- The `TestFileEventConnectionFields` class (line 823): **Delete entirely** — tests FileEvent dataclass fields which no longer exist.
- Integration test fixtures that return `tuple[GroverAsync, list[FileEvent], ...]`: Simplify to `tuple[GroverAsync, ...]` — no event collection needed.

**`tests/test_session_batching.py`:**
- Update `FileEvent` import → remove
- Line 228: `check_db_handler` registers on `event_bus` → rewrite to check DB state directly
- Line 315: Constructs `FileEvent(FILE_DELETED, ...)` → call `_process_delete()` directly
- Lines 380-381: Constructs `FileEvent(FILE_MOVED, ...)` → call `_process_move()` directly

**`tests/test_chunk_migration.py`:**
- Line 12: Remove `FileEvent`/`EventType` imports
- Lines 369-382: `TestFileEventUserId` class → **Delete entirely** (tests FileEvent fields).

### Acceptance criteria
- No code references `EventBus`, `FileEvent`, or `EventType` except `events.py` itself
- All facade methods use `worker.schedule()` / `schedule_immediate()` / `cancel()`
- `_analyze_and_integrate` updates graph directly (no deferred events)
- `_process_move` reuses `_process_delete` + `_process_write` (no duplication)
- `flush()` still works (delegates to `worker.drain()`)
- `IndexingMode.MANUAL` still suppresses background work
- All tests pass

### Code review focus areas
- Lambda late-binding: verify default args (`p=path`) prevent closure bugs
- `schedule()` vs `schedule_immediate()`: writes/restores are debounced, deletes/moves/connections are immediate
- `cancel()` before `schedule_immediate()` on delete/move: ensures pending writes don't run after delete
- `_process_move` reuse: verify old-path cleanup + new-path analysis matches current behavior
- `_analyze_and_integrate` graph update ordering: edges added AFTER session commits (post-commit ordering preserved)

---

## Phase 3: Delete EventBus + clean up + docs

### Summary
Remove `events.py`, update all import paths for `IndexingMode`, delete `test_events.py`, update documentation.

### Files to modify

**Delete:**
- `src/grover/events.py`
- `tests/test_events.py`

**Update imports:**
- `src/grover/__init__.py` — change `from grover.events import IndexingMode` → `from grover.worker import IndexingMode`
- `src/grover/_grover.py` — update IndexingMode import if referencing events
- `src/grover/_grover_async.py` — remove any remaining events import (should be clean from Phase 2)
- `src/grover/facade/context.py` — remove events import if remaining

**Update docs:**
- `CLAUDE.md` — repo layout: replace `events.py` line with `worker.py # BackgroundWorker, IndexingMode`
- `docs/architecture.md` — rewrite "Event dispatch and indexing modes" section: describe BackgroundWorker, schedule/schedule_immediate, debounce, drain lifecycle. Remove EventBus/FileEvent/handler registration language.
- `docs/api.md` — update any EventBus references

**Update memory:**
- `MEMORY.md` — update "Background indexing" entry: BackgroundWorker replaces EventBus, no FileEvent/EventType, facade methods schedule work directly.

### Acceptance criteria
- `events.py` and `test_events.py` are deleted
- No file in the repo imports from `grover.events`
- `from grover import IndexingMode` still works (re-exported from `__init__.py`)
- All docs reflect the new architecture
- All quality checks pass: `uv run pytest`, `uvx ruff check src/ tests/`, `uvx ruff format --check src/ tests/`, `uvx ty check src/`

---

## Verification

After all phases:
```bash
uv run pytest                           # all tests pass
uvx ruff check src/ tests/              # no lint errors
uvx ruff format --check src/ tests/     # no format errors
uvx ty check src/                       # no type errors
```

Spot-check:
- Write a file → result returns immediately → `flush()` → graph/search updated
- Edit a file → debounced with previous write (same path, 0.1s window)
- Delete a file → cancels pending write → immediate cleanup
- Move a file → cancels pending work for old path → immediate cleanup + re-analyze
- `IndexingMode.MANUAL` → no background work, explicit `index()` still works
- Connection add → immediate graph update (no debounce)

# Handle External File Edits in Version History

## Context

When a file tracked by Grover is modified externally (by an IDE, git, shell, etc.), the version chain silently breaks. Here's the exact failure:

1. Grover creates `/app.py` → v1 (snapshot: `"hello"`)
2. Grover edits `/app.py` → v2 (diff: v1→v2, content: `"hello world"`)
3. **VS Code edits `/app.py`** on disk → disk now contains `"hello world!!!"`
4. Grover edits `/app.py` → v3 (diff is computed from **disk content** `"hello world!!!"` → new content)

**The bug**: `_save_version` stores a diff from disk content → new content. But `reconstruct_version(v3)` replays v1_snapshot + v2_diff + v3_diff. Since v3's diff assumes `"hello world!!!"` as its base (the external edit), but reconstruction uses `"hello world"` (v2) as the base, the result is **wrong** and hash verification **fails** with `ConsistencyError`.

**Root cause**: `base.py:369` reads `old_content` from `_read_content()` which returns the **disk state** for LFS. If the disk diverged from the last known version, the diff chain breaks because there's no version record for the intermediate state.

**The fix**: Before creating a new version, compare the disk content's hash against `File.content_hash`. If they differ, insert a **synthetic snapshot version** capturing the external edit. This keeps the diff chain intact.

---

## Design Decisions

**Detect-on-mutate, not file watching**: Check for external edits at `write()`/`edit()` time only. No new background processes, no new dependencies (watchfiles/watchdog). File watching can be layered on later as a separate enhancement.

**Synthetic versions as snapshots**: When an external edit is detected, store the current disk content as a **full snapshot** (not a diff). This avoids the cost of reconstructing the previous version from the chain. Passing `old_content=""` to `_save_version` naturally produces a snapshot via the `not old_content` branch.

**`created_by="external"`**: Synthetic versions use this marker so they're distinguishable in version history. No model changes needed — `created_by` is already a free-form string.

**Works for both LFS and DFS**: `File.content_hash` is only updated through Grover's `write()`/`edit()` code paths, so it serves as a "last Grover-written hash" marker regardless of backend. For LFS, `_read_content()` reads from disk — an IDE or git edit changes the disk but not `content_hash` → mismatch detected. For DFS, `_read_content()` reads `File.content` from the DB — an external SQL UPDATE changes `content` but not `content_hash` → mismatch detected.

**No events for now**: External edits are silently versioned. Event emission (e.g., `FILE_EXTERNALLY_MODIFIED`) can be layered on later.

---

## Changes

### 1. `src/grover/fs/base.py` — Add `_record_external_edit` + restructure `write()`/`edit()`

**New method** (~20 lines):

```python
async def _record_external_edit(
    self,
    session: AsyncSession,
    file: F,
    current_content: str,
) -> bool:
    """If storage content differs from last known version, record a synthetic version.

    Returns True if an external edit was detected and recorded.
    """
    if not file.content_hash:
        return False

    current_hash = hashlib.sha256(current_content.encode()).hexdigest()
    if current_hash == file.content_hash:
        return False

    # External edit detected — save full snapshot of current state
    file.current_version += 1
    content_bytes = current_content.encode()
    file.content_hash = current_hash
    file.size_bytes = len(content_bytes)
    file.updated_at = datetime.now(UTC)

    await self._save_version(
        session, file, "", current_content, "external",
    )
    return True
```

**`write()` — restructure existing-file path** (lines 346-383):

Move `_read_content` call before the version increment, add external edit check:

```python
if existing:
    # ... directory/overwrite/deleted checks unchanged ...

    # Read current content early — needed for external edit check AND diff
    old_content = await self._read_content(path, session)

    # Detect and record external edits before creating the new version
    if old_content is not None:
        await self._record_external_edit(session, existing, old_content)

    # Update metadata (unchanged)
    existing.current_version += 1
    existing.content_hash = content_hash
    existing.size_bytes = size_bytes
    existing.updated_at = datetime.now(UTC)

    # Save version (unchanged — old_content is still disk content)
    if old_content is not None:
        await self._save_version(
            session, existing, old_content, content, created_by,
        )

    # Content-before-commit ordering preserved
    await self._write_content(path, content, session)
    await session.flush()
```

**`edit()` — add external edit check** (after line 446):

```python
content = await self._read_content(path, session)
if content is None:
    return EditResult(success=False, message=f"File content not found: {path}")

# Detect and record external edits before applying the edit
await self._record_external_edit(session, file, content)

# ... rest of edit() unchanged (replace, increment, save_version, write) ...
```

### 2. `tests/test_base_fs.py` — Add external edit tests

New test class `TestExternalEditDetection` (~6-8 tests):

| Test | Scenario |
|------|----------|
| `test_write_after_external_edit_creates_synthetic_version` | Modify disk between v1 and Grover write → v2 (external snapshot) + v3 (Grover write) |
| `test_edit_after_external_edit_creates_synthetic_version` | Modify disk between v1 and Grover edit → v2 (external) + v3 (edit) |
| `test_version_reconstruction_after_external_edit` | Verify `get_version_content(v2)` returns external content, `get_version_content(v3)` returns Grover content |
| `test_external_version_is_snapshot` | Verify synthetic version has `is_snapshot=True` |
| `test_external_version_created_by` | Verify `created_by="external"` |
| `test_no_external_edit_no_synthetic_version` | Normal write without disk modification → no extra version |
| `test_multiple_external_edits_between_operations` | Two external edits → one synthetic version (captures net result) |
| `test_hash_verification_passes_after_external_edit` | Full chain reconstruction with hash check succeeds |

### 3. `docs/internals/fs.md` — Document external edit handling

Add a new section "External Edit Detection" after "Write Order of Operations" covering:
- How detection works (hash comparison)
- What happens (synthetic snapshot version)
- Why snapshots (avoids reconstruction cost)
- Example version chain with external edit

---

## Key Files

| File | Change |
|------|--------|
| `src/grover/fs/base.py` | Add `_record_external_edit`, restructure `write()` and `edit()` |
| `tests/test_base_fs.py` | Add `TestExternalEditDetection` class (~8 tests) |
| `tests/test_local_fs.py` | Add 1-2 LFS-specific external edit tests (disk I/O) |
| `docs/internals/fs.md` | Document external edit detection |

---

## Version Chain Example

```
v1: snapshot  "hello"                        (Grover write)
v2: diff      v1→"hello world"               (Grover edit)
    ── VS Code edits file on disk to "hello world!!!" ──
v3: snapshot  "hello world!!!"               (external, auto-detected)
v4: diff      v3→"hello world!!! # updated"  (Grover edit)
```

Reconstruction of any version works correctly because v3 is a snapshot — diffs don't need to bridge the gap from v2 to the external state.

---

## Verification

1. `uv run pytest tests/ -x -q` — all tests pass (including new external edit tests)
2. `uvx ruff check src/` — no new lint errors
3. Manual test scenario:
   - Create file via Grover
   - Edit the disk file directly (outside Grover)
   - Write via Grover again
   - Verify `list_versions` shows 3 versions (original, external, new)
   - Verify `get_version_content(v2)` returns the external edit content
   - Verify hash verification passes on all versions

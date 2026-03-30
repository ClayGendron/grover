"""Integration tests for DatabaseFileSystem against in-memory SQLite."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from grover.backends.database import DatabaseFileSystem
from grover.base import GroverFileSystem
from grover.models import GroverObject
from grover.results import Candidate, GroverResult


def _stored_payload(obj: GroverObject) -> str:
    return obj.content if obj.is_snapshot else obj.version_diff

# ------------------------------------------------------------------
# Part 1: Write + Read
# ------------------------------------------------------------------


class TestWriteAndRead:
    async def test_write_and_read_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            w = await db._write_impl("/hello.txt", "hello world", session=s)
        assert w.success
        assert w.content == "hello world"
        assert w.file.kind == "file"
        assert w.file.path == "/hello.txt"

        async with db._use_session() as s:
            r = await db._read_impl("/hello.txt", session=s)
        assert r.success
        assert r.content == "hello world"

    async def test_write_creates_parent_dirs(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a/b/c/file.py", "code", session=s)
            # Parents should exist
            for p in ["/a", "/a/b", "/a/b/c"]:
                obj = await db._get_object(p, s)
                assert obj is not None, f"Missing parent: {p}"
                assert obj.kind == "directory"

    async def test_write_overwrite_false_rejects_existing(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v1", session=s)
        async with db._use_session() as s:
            r = await db._write_impl("/file.txt", "v2", overwrite=False, session=s)
        assert not r.success
        assert "overwrite=False" in r.error_message

    async def test_write_overwrite_updates_content(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v1", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v2", overwrite=True, session=s)
        async with db._use_session() as s:
            r = await db._read_impl("/file.txt", session=s)
        assert r.content == "v2"

    async def test_write_chunk(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "full content", session=s)
            w = await db._write_impl("/src/auth.py/.chunks/login", "def login():", session=s)
        assert w.success
        assert w.file.kind == "chunk"

        async with db._use_session() as s:
            r = await db._read_impl("/src/auth.py/.chunks/login", session=s)
        assert r.content == "def login():"

    async def test_write_chunk_requires_existing_parent_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/ghost.py/.chunks/login", "def login():", session=s)
        assert not r.success
        assert "Chunk parent file not found" in r.error_message

    async def test_write_chunk_allows_companion_file_in_same_batch(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl(
                objects=[
                    GroverObject(path="/src/auth.py", content="full content"),
                    GroverObject(path="/src/auth.py/.chunks/login", content="def login():"),
                ],
                session=s,
            )
        assert r.success
        assert r.paths == ("/src/auth.py", "/src/auth.py/.chunks/login")

        async with db._use_session() as s:
            file_obj = await db._get_object("/src/auth.py", s)
            chunk_obj = await db._get_object("/src/auth.py/.chunks/login", s)
        assert file_obj is not None
        assert chunk_obj is not None

    async def test_public_write_preserves_same_batch_semantics_when_query_chunking(self, db: DatabaseFileSystem):
        db.DIALECT_PARAMETER_BUDGETS = {}
        db.PARAMETER_BUDGET_FALLBACK = 1

        r = await db.write(
            objects=[
                GroverObject(path="/src/auth.py/.chunks/login", content="def login():"),
                GroverObject(path="/src/auth.py", content="full content"),
            ]
        )
        assert r.success
        assert r.paths == ("/src/auth.py/.chunks/login", "/src/auth.py")

        async with db._use_session() as s:
            file_obj = await db._get_object("/src/auth.py", s)
            chunk_obj = await db._get_object("/src/auth.py/.chunks/login", s)
        assert file_obj is not None
        assert chunk_obj is not None

    async def test_write_rejects_version_path(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/file.txt/.versions/1", "nope", session=s)
        assert not r.success
        assert "version" in r.error_message.lower()

    async def test_write_rejects_connection_path(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/a.py/.connections/imports/b.py", "nope", session=s)
        assert not r.success
        assert "connection" in r.error_message.lower()

    async def test_read_nonexistent(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._read_impl("/nope.txt", session=s)
        assert not r.success
        assert "Not found" in r.error_message

    async def test_read_with_candidates(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "aaa", session=s)
            await db._write_impl("/b.py", "bbb", session=s)
        candidates = GroverResult(candidates=[
            Candidate(path="/a.py"),
            Candidate(path="/b.py"),
            Candidate(path="/nope.py"),
        ])
        async with db._use_session() as s:
            r = await db._read_impl(candidates=candidates, session=s)
        assert len(r.candidates) == 2
        assert r.paths == ("/a.py", "/b.py")
        assert not r.success  # nope.py not found → errors

    async def test_content_metrics(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            w = await db._write_impl("/file.txt", "line1\nline2\nline3", session=s)
        assert w.file.lines == 3
        assert w.file.size_bytes == len(b"line1\nline2\nline3")

    async def test_write_under_existing_file_ancestor_rejected(self, db: DatabaseFileSystem):
        """Writing /a.txt/b.txt when /a.txt is a file must fail."""
        async with db._use_session() as s:
            await db._write_impl("/a.txt", "i am a file", session=s)

        async with db._use_session() as s:
            r = await db._write_impl("/a.txt/b.txt", "child", session=s)
        assert not r.success
        assert "not directory" in r.error_message.lower()

        # Child must NOT be persisted
        async with db._use_session() as s:
            child = await db._get_object("/a.txt/b.txt", s)
        assert child is None

    async def test_write_under_existing_file_ancestor_batch_rejected(self, db: DatabaseFileSystem):
        """Batch write where one file's ancestor is an existing file."""
        async with db._use_session() as s:
            await db._write_impl("/blocker.py", "file", session=s)

        objects = [
            GroverObject(path="/blocker.py/sub/child.txt", content="bad"),
            GroverObject(path="/safe/other.txt", content="good"),
        ]
        async with db._use_session() as s:
            r = await db._write_impl(objects=objects, session=s)
        assert not r.success
        assert "not directory" in r.error_message.lower()

    async def test_write_revives_soft_deleted_ancestor_dirs(self, db: DatabaseFileSystem):
        deleted_at = datetime.now(UTC)
        async with db._use_session() as s:
            s.add(GroverObject(path="/archive", kind="directory", deleted_at=deleted_at))
            s.add(GroverObject(path="/archive/nested", kind="directory", deleted_at=deleted_at))

        async with db._use_session() as s:
            r = await db._write_impl("/archive/nested/report.txt", "report", session=s)
        assert r.success

        async with db._use_session() as s:
            archive = await db._get_object("/archive", s, include_deleted=True)
            nested = await db._get_object("/archive/nested", s, include_deleted=True)
        assert archive is not None
        assert nested is not None
        assert archive.deleted_at is None
        assert nested.deleted_at is None

    async def test_failed_write_does_not_revive_soft_deleted_ancestor_dirs(self, db: DatabaseFileSystem):
        """P1 regression: if all writes fail, revived dirs must stay deleted."""
        deleted_at = datetime.now(UTC)
        async with db._use_session() as s:
            s.add(GroverObject(path="/ghost", kind="directory", deleted_at=deleted_at))
            s.add(GroverObject(path="/ghost/deep", kind="directory", deleted_at=deleted_at))

        # Every write fails (overwrite=False on existing file)
        async with db._use_session() as s:
            s.add(GroverObject(path="/ghost/deep/file.txt", content="existing"))
        async with db._use_session() as s:
            r = await db._write_impl(
                "/ghost/deep/file.txt", "conflict", overwrite=False, session=s,
            )
        assert not r.success

        # Ancestor dirs must still be soft-deleted
        async with db._use_session() as s:
            ghost = await db._get_object("/ghost", s, include_deleted=True)
            deep = await db._get_object("/ghost/deep", s, include_deleted=True)
        assert ghost.deleted_at is not None, "Revived dir committed despite failed write"
        assert deep.deleted_at is not None, "Revived dir committed despite failed write"

    async def test_failed_write_does_not_create_parent_dirs(self, db: DatabaseFileSystem):
        """If the flush fails, the session rolls back — no parent dirs created."""
        with pytest.raises(RuntimeError, match="simulated insert failure"):
            async with db._use_session() as s:
                original_flush = s.flush

                async def failing_flush(*args, **kwargs):
                    raise RuntimeError("simulated insert failure")

                s.flush = failing_flush
                await db._write_impl("/brand_new/dir/file.txt", "content", session=s)

        async with db._use_session() as s:
            parent = await db._get_object("/brand_new/dir", s)
            grandparent = await db._get_object("/brand_new", s)
        assert parent is None, "Parent dir created despite failed write"
        assert grandparent is None, "Grandparent dir created despite failed write"

    async def test_write_overwrite_false_revives_soft_deleted_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            s.add(
                GroverObject(
                    path="/revive.txt",
                    content="old",
                    deleted_at=datetime.now(UTC),
                )
            )

        async with db._use_session() as s:
            r = await db._write_impl("/revive.txt", "new", overwrite=False, session=s)
        assert r.success
        assert r.content == "new"

        async with db._use_session() as s:
            obj = await db._get_object("/revive.txt", s, include_deleted=True)
        assert obj is not None
        assert obj.deleted_at is None
        assert obj.content == "new"


class TestStat:
    async def test_stat_delegates_to_read(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "some content", session=s)
        async with db._use_session() as s:
            r = await db._stat_impl("/file.txt", session=s)
        assert r.success
        assert r.file.content == "some content"
        assert r.file.lines == 1
        assert r.file.path == "/file.txt"

    async def test_stat_nonexistent(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._stat_impl("/nope.txt", session=s)
        assert not r.success


class TestAutoVersioning:
    async def test_overwrite_creates_version(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v1", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v2", session=s)

        async with db._use_session() as s:
            v1 = await db._get_object("/file.txt/.versions/1", s)
            v2 = await db._get_object("/file.txt/.versions/2", s)
        assert v1 is not None
        assert v2 is not None
        assert v1.is_snapshot is True
        assert v1.content == "v1"
        assert v1.version_diff is None
        assert v2.is_snapshot is False
        assert v2.content is None
        assert v2.version_diff is not None
        assert v2.content_hash == hashlib.sha256(b"v2").hexdigest()

    async def test_multiple_overwrites_increment_versions(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v1", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v2", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v3", session=s)

        # Both version records should exist
        async with db._use_session() as s:
            r1 = await db._read_impl("/file.txt/.versions/1", session=s)
            r2 = await db._read_impl("/file.txt/.versions/2", session=s)
        assert r1.success
        assert r2.success

        # Current content is v3
        async with db._use_session() as s:
            r = await db._read_impl("/file.txt", session=s)
        assert r.content == "v3"

    async def test_version_reconstruction(self, db: DatabaseFileSystem):
        """Verify we can reconstruct any version from forward diffs."""
        from grover.versioning import reconstruct_version

        async with db._use_session() as s:
            await db._write_impl("/file.txt", "line1\n", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "line1\nline2\n", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "line1\nline2\nline3\n", session=s)

        # Live content is "line1\nline2\nline3\n"
        # Version 1 is the first full file state.
        # Version 2 is stored as a forward diff from v1 -> v2.
        async with db._use_session() as s:
            v1_obj = await db._get_object("/file.txt/.versions/1", s)
            v2_obj = await db._get_object("/file.txt/.versions/2", s)

        # Reconstruct version 1: snapshot — just the stored content
        reconstructed_v1 = reconstruct_version([(v1_obj.is_snapshot, _stored_payload(v1_obj))])
        assert reconstructed_v1 == "line1\n"

        # Reconstruct version 2: start from v1 snapshot, apply v2 forward diff
        reconstructed_v2 = reconstruct_version([
            (v1_obj.is_snapshot, _stored_payload(v1_obj)),
            (v2_obj.is_snapshot, _stored_payload(v2_obj)),
        ])
        assert reconstructed_v2 == "line1\nline2\n"

    async def test_periodic_snapshot(self, db: DatabaseFileSystem):
        """Every SNAPSHOT_INTERVAL versions is a full snapshot."""
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "v0", session=s)
        for i in range(1, 11):
            async with db._use_session() as s:
                await db._write_impl("/file.txt", f"v{i}", session=s)

        async with db._use_session() as s:
            v10 = await db._get_object("/file.txt/.versions/10", s)
        assert v10 is not None
        assert v10.is_snapshot is True
        assert v10.version_diff is None

    async def test_external_edit_creates_synthetic_snapshot(self, db: DatabaseFileSystem):
        await db.write("/app.py", "v1")

        async with db._engine.begin() as conn:
            await conn.execute(text(
                "UPDATE grover_objects SET content='external' WHERE path='/app.py'"
            ))

        r = await db.write("/app.py", "v2")
        assert r.success

        async with db._use_session() as s:
            file_obj = await db._get_object("/app.py", s)
            v2 = await db._get_object("/app.py/.versions/2", s)
            v3 = await db._get_object("/app.py/.versions/3", s)

        assert file_obj is not None
        assert file_obj.version_number == 3
        assert file_obj.content == "v2"
        assert v2 is not None
        assert v2.created_by == "external"
        assert v2.is_snapshot is True
        assert v2.content == "external"
        assert v3 is not None
        assert v3.is_snapshot is False
        assert v3.content is None
        assert v3.version_diff is not None

    async def test_missing_current_version_creates_repair_snapshot(self, db: DatabaseFileSystem):
        await db.write("/repair.txt", "v1")
        await db.write("/repair.txt", "v2")

        async with db._use_session() as s:
            bad = await db._get_object("/repair.txt/.versions/2", s)
            assert bad is not None
            await s.delete(bad)

        r = await db.write("/repair.txt", "v3")
        assert r.success

        async with db._use_session() as s:
            file_obj = await db._get_object("/repair.txt", s)
            v3 = await db._get_object("/repair.txt/.versions/3", s)
            v4 = await db._get_object("/repair.txt/.versions/4", s)

        assert file_obj is not None
        assert file_obj.version_number == 4
        assert v3 is not None
        assert v3.created_by == "repair"
        assert v3.is_snapshot is True
        assert v3.content == "v2"
        assert v4 is not None
        assert v4.is_snapshot is False
        assert v4.version_diff is not None

    async def test_chunk_write_does_not_version(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.py", "full", session=s)
            await db._write_impl("/file.py/.chunks/fn", "def fn(): pass", session=s)
        async with db._use_session() as s:
            await db._write_impl("/file.py/.chunks/fn", "def fn(): return 1", session=s)

        # No version should be created for chunk overwrites
        async with db._use_session() as s:
            r = await db._read_impl("/file.py/.chunks/fn/.versions/1", session=s)
        assert not r.success


class TestNestedMountPaths:
    """Write through a mount, read back — paths must be absolute."""

    async def test_write_and_read_through_nested_mount(self, engine):
        root = GroverFileSystem(engine=engine)
        child = DatabaseFileSystem(engine=engine)
        await root.add_mount("/data", child)

        # Write through the mount
        w = await root.write(
            objects=[
                GroverObject(path="/data/docs/readme.txt", content="hello"),
                GroverObject(path="/data/src/app.py", content="import os"),
            ],
        )
        assert w.success
        assert w.paths == ("/data/docs/readme.txt", "/data/src/app.py")

        # Read back through the mount — paths must be absolute with mount prefix
        r = await root.read("/data/docs/readme.txt")
        assert r.success
        assert r.file.path == "/data/docs/readme.txt"
        assert r.content == "hello"

        r2 = await root.read("/data/src/app.py")
        assert r2.success
        assert r2.file.path == "/data/src/app.py"
        assert r2.content == "import os"

    async def test_write_and_read_through_two_level_mount(self, engine):
        root = GroverFileSystem(engine=engine)
        mid = GroverFileSystem(engine=engine)
        leaf = DatabaseFileSystem(engine=engine)
        await root.add_mount("/org", mid)
        await mid.add_mount("/team", leaf)

        w = await root.write("/org/team/plan.md", "# Plan")
        assert w.success
        assert w.file.path == "/org/team/plan.md"

        r = await root.read("/org/team/plan.md")
        assert r.success
        assert r.file.path == "/org/team/plan.md"
        assert r.content == "# Plan"


class TestBatchWriteAtScale:
    """Stress-test batched writes against real SQLite parameter limits."""

    async def test_batch_write_single_call(self, db: DatabaseFileSystem):
        n = 1_000
        objects = [
            GroverObject(path=f"/data/file_{i:05d}.txt", content=f"content {i}")
            for i in range(n)
        ]

        r = await db.write(objects=objects)

        assert r.success, r.error_message
        assert len(r.candidates) == n

        # Spot-check first, last, and a middle file
        mid_i = n // 2
        async with db._use_session() as s:
            first = await db._get_object("/data/file_00000.txt", s)
            mid = await db._get_object(f"/data/file_{mid_i:05d}.txt", s)
            last = await db._get_object(f"/data/file_{n - 1:05d}.txt", s)
        assert first is not None and first.content == "content 0"
        assert mid is not None and mid.content == f"content {mid_i}"
        assert last is not None and last.content == f"content {n - 1}"

        # Parent dir should exist
        async with db._use_session() as s:
            data_dir = await db._get_object("/data", s)
        assert data_dir is not None
        assert data_dir.kind == "directory"

    async def test_batch_write_across_nested_dirs(self, db: DatabaseFileSystem):
        import random

        rng = random.Random(42)
        dirs = [f"/d{i}/sub{j}" for i in range(10) for j in range(10)]
        n = 1_000
        objects = [
            GroverObject(
                path=f"{rng.choice(dirs)}/file_{i:05d}.txt",
                content=f"content {i}",
            )
            for i in range(n)
        ]

        r = await db.write(objects=objects)

        assert r.success, r.error_message
        assert len(r.candidates) == n

        # Every top-level and nested dir should have been auto-created
        async with db._use_session() as s:
            for d in dirs:
                obj = await db._get_object(d, s)
                assert obj is not None, f"Missing dir: {d}"
                assert obj.kind == "directory"
            # Top-level parents too
            for i in range(10):
                obj = await db._get_object(f"/d{i}", s)
                assert obj is not None, f"Missing parent: /d{i}"
                assert obj.kind == "directory"

    async def test_batch_overwrites_creates_versions(self, db: DatabaseFileSystem):
        n = 1_000
        objects_v1 = [
            GroverObject(path=f"/src/f_{i:05d}.py", content=f"v1_{i}")
            for i in range(n)
        ]
        r1 = await db.write(objects=objects_v1)
        assert r1.success, r1.error_message

        objects_v2 = [
            GroverObject(path=f"/src/f_{i:05d}.py", content=f"v2_{i}")
            for i in range(n)
        ]
        r2 = await db.write(objects=objects_v2)
        assert r2.success, r2.error_message

        # Spot-check: current content is v2, version 1 exists
        async with db._use_session() as s:
            live = await db._get_object("/src/f_00500.py", s)
            ver = await db._get_object("/src/f_00500.py/.versions/1", s)
        assert live is not None and live.content == "v2_500"
        assert ver is not None and ver.kind == "version"


# ------------------------------------------------------------------
# Part 5: Fast-path versioning
# ------------------------------------------------------------------


class TestFastPathVersioning:
    """Tests for the two-query fast path that skips version reconstruction."""

    async def test_overwrite_uses_fast_path(self, db: DatabaseFileSystem):
        """Normal overwrite: version increments by 1 (no repair inserted)."""
        await db.write("/fp.txt", "v1")
        await db.write("/fp.txt", "v2")

        async with db._use_session() as s:
            f = await db._get_object("/fp.txt", s)
        assert f.version_number == 2  # 1→2, not 1→repair→3

        async with db._use_session() as s:
            v2 = await db._get_object("/fp.txt/.versions/2", s)
        assert v2 is not None
        assert v2.is_snapshot is False
        assert v2.created_by == "auto"

    async def test_broken_intermediate_chain_not_repaired_on_fast_path(self, db: DatabaseFileSystem):
        """Accepted behavior: broken intermediate rows are not repaired.

        The fast path only checks the latest version hash against the file
        hash. If an intermediate version row is deleted, the chain is broken
        but the fast path does not detect it — by design.
        """
        await db.write("/chain.txt", "v1")
        await db.write("/chain.txt", "v2")
        await db.write("/chain.txt", "v3")

        # Delete intermediate version row
        async with db._use_session() as s:
            v2 = await db._get_object("/chain.txt/.versions/2", s)
            assert v2 is not None
            await s.delete(v2)

        # Write v4 — fast path, no repair
        r = await db.write("/chain.txt", "v4")
        assert r.success

        async with db._use_session() as s:
            f = await db._get_object("/chain.txt", s)
            v2_after = await db._get_object("/chain.txt/.versions/2", s)
        # Version advances directly, no repair snapshot inserted
        assert f.version_number == 4
        assert v2_after is None  # still missing

    async def test_missing_latest_version_triggers_slow_path(self, db: DatabaseFileSystem):
        """When the latest version row is missing, Step 4b returns no hash.

        This triggers the slow path: _fetch_version_chain loads the chain,
        plan_file_write detects the gap, and a repair snapshot is created.
        """
        await db.write("/slow.txt", "v1")
        await db.write("/slow.txt", "v2")

        # Delete the latest version row (v2)
        async with db._use_session() as s:
            v2 = await db._get_object("/slow.txt/.versions/2", s)
            assert v2 is not None
            await s.delete(v2)

        # Write v3 — slow path should create repair snapshot
        r = await db.write("/slow.txt", "v3")
        assert r.success

        async with db._use_session() as s:
            f = await db._get_object("/slow.txt", s)
            v3 = await db._get_object("/slow.txt/.versions/3", s)
        # Repair snapshot for v2's content + new v4 diff
        assert f.version_number == 4
        assert v3 is not None
        assert v3.created_by == "repair"
        assert v3.is_snapshot is True

    async def test_external_edit_still_detected_with_fast_path(self, db: DatabaseFileSystem):
        """External SQL edits are detected regardless of fast path."""
        await db.write("/ext.txt", "v1")

        async with db._engine.begin() as conn:
            await conn.execute(text(
                "UPDATE grover_objects SET content='hacked' WHERE path='/ext.txt'"
            ))

        r = await db.write("/ext.txt", "v2")
        assert r.success

        async with db._use_session() as s:
            f = await db._get_object("/ext.txt", s)
            v2 = await db._get_object("/ext.txt/.versions/2", s)
        assert f.version_number == 3
        assert v2 is not None
        assert v2.created_by == "external"
        assert v2.is_snapshot is True


class TestFetchVersionChain:
    async def test_returns_bounded_rows(self, db: DatabaseFileSystem):
        """_fetch_version_chain returns only rows within SNAPSHOT_INTERVAL."""
        from grover.versioning import SNAPSHOT_INTERVAL

        # Create 15 versions
        for i in range(1, 16):
            await db.write("/bounded.txt", f"v{i}")

        async with db._use_session() as s:
            f = await db._get_object("/bounded.txt", s)
            assert f.version_number == 15

            rows = await db._fetch_version_chain("/bounded.txt", 15, s)

        version_numbers = sorted(r.version_number for r in rows)
        lower_bound = max(1, 15 - SNAPSHOT_INTERVAL + 1)
        # Should only have versions from lower_bound to 15
        assert version_numbers[0] >= lower_bound
        assert version_numbers[-1] == 15
        assert len(rows) <= SNAPSHOT_INTERVAL

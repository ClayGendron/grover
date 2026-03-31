"""Integration tests for DatabaseFileSystem against in-memory SQLite."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from grover.backends.database import DatabaseFileSystem
from grover.base import GroverFileSystem
from grover.models import GroverObject
from grover.results import Candidate, Detail, EditOperation, GroverResult, TwoPathOperation


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

    async def test_write_accepts_connection_path(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/a.py/.connections/imports/b.py", "nope", session=s)
        assert r.success
        assert r.file.kind == "connection"

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


class TestEdit:
    async def test_edit_single_file(self, db: DatabaseFileSystem):
        await db.write("/file.py", "def hello():\n    return 'world'\n")
        async with db._use_session() as s:
            r = await db._edit_impl("/file.py", edits=[
                EditOperation(old="'world'", new="'earth'"),
            ], session=s)
        assert r.success
        assert "'earth'" in r.content

    async def test_edit_multiple_edits(self, db: DatabaseFileSystem):
        await db.write("/file.py", "x = 1\ny = 2\nz = 3\n")
        async with db._use_session() as s:
            r = await db._edit_impl("/file.py", edits=[
                EditOperation(old="x = 1", new="x = 10"),
                EditOperation(old="z = 3", new="z = 30"),
            ], session=s)
        assert r.success
        assert r.content == "x = 10\ny = 2\nz = 30\n"

    async def test_edit_replace_all(self, db: DatabaseFileSystem):
        await db.write("/file.txt", "foo bar foo baz foo")
        async with db._use_session() as s:
            r = await db._edit_impl("/file.txt", edits=[
                EditOperation(old="foo", new="qux", replace_all=True),
            ], session=s)
        assert r.success
        assert r.content == "qux bar qux baz qux"

    async def test_edit_string_not_found(self, db: DatabaseFileSystem):
        await db.write("/file.txt", "hello world")
        async with db._use_session() as s:
            r = await db._edit_impl("/file.txt", edits=[
                EditOperation(old="missing", new="replacement"),
            ], session=s)
        assert not r.success
        assert "not found" in r.error_message.lower()

    async def test_edit_nonexistent_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._edit_impl("/nope.txt", edits=[
                EditOperation(old="a", new="b"),
            ], session=s)
        assert not r.success

    async def test_edit_creates_version(self, db: DatabaseFileSystem):
        await db.write("/ver.py", "v1 content")
        await db.edit("/ver.py", old="v1", new="v2")
        async with db._use_session() as s:
            obj = await db._get_object("/ver.py", s)
        assert obj.version_number == 2
        assert obj.content == "v2 content"

    async def test_edit_batch_via_candidates(self, db: DatabaseFileSystem):
        await db.write("/a.py", "old_name = 1")
        await db.write("/b.py", "old_name = 2")
        candidates = GroverResult(candidates=[
            Candidate(path="/a.py"),
            Candidate(path="/b.py"),
        ])
        async with db._use_session() as s:
            r = await db._edit_impl(candidates=candidates, edits=[
                EditOperation(old="old_name", new="new_name"),
            ], session=s)
        assert r.success
        assert len(r.candidates) == 2
        r2 = await db.read("/a.py")
        assert r2.content == "new_name = 1"

    async def test_edit_fuzzy_whitespace_match(self, db: DatabaseFileSystem):
        """Line-trimmed replacer handles indentation differences."""
        await db.write("/indent.py", "    def foo():\n        pass\n")
        async with db._use_session() as s:
            r = await db._edit_impl("/indent.py", edits=[
                EditOperation(old="def foo():\n    pass", new="def foo():\n    return 1"),
            ], session=s)
        assert r.success
        assert "return 1" in r.content

    async def test_edit_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/app.py", "timeout = 30")
        r = await root.edit("/code/app.py", old="30", new="120")
        assert r.success
        r2 = await root.read("/code/app.py")
        assert r2.content == "timeout = 120"


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


# ------------------------------------------------------------------
# Part 6: ls
# ------------------------------------------------------------------


class TestLs:
    async def test_ls_directory_returns_files(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "auth", session=s)
            await db._write_impl("/src/utils.py", "utils", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/src", session=s)
        assert r.success
        assert set(r.paths) == {"/src/auth.py", "/src/utils.py"}

    async def test_ls_directory_returns_subdirs(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/data/docs/readme.txt", "hello", session=s)
            await db._write_impl("/data/src/app.py", "code", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/data", session=s)
        assert r.success
        assert set(r.paths) == {"/data/docs", "/data/src"}

    async def test_ls_file_returns_metadata_children(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "code", session=s)
            await db._write_impl("/auth.py/.chunks/login", "def login():", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/auth.py", session=s)
        assert r.success
        paths = set(r.paths)
        assert "/auth.py/.chunks/login" in paths
        assert "/auth.py/.versions/1" in paths

    async def test_ls_directory_hides_metadata_kinds(self, db: DatabaseFileSystem):
        """ls on a directory should not return chunks/versions of child files."""
        async with db._use_session() as s:
            await db._write_impl("/src/app.py", "code", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/src", session=s)
        assert r.success
        assert r.paths == ("/src/app.py",)
        for c in r.candidates:
            assert c.kind in ("file", "directory")

    async def test_ls_root(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.txt", "a", session=s)
            await db._write_impl("/b.txt", "b", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/", session=s)
        assert r.success
        assert set(r.paths) == {"/a.txt", "/b.txt"}

    async def test_ls_empty_directory(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            s.add(GroverObject(path="/empty", kind="directory"))

        async with db._use_session() as s:
            r = await db._ls_impl("/empty", session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_ls_nonexistent_path(self, db: DatabaseFileSystem):
        """Single-path ls on a nonexistent path returns empty (unknown kind, not in DB)."""
        async with db._use_session() as s:
            r = await db._ls_impl("/nope", session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_ls_excludes_deleted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.txt", "a", session=s)
            await db._write_impl("/b.txt", "b", session=s)

        async with db._use_session() as s:
            obj = await db._get_object("/b.txt", s)
            obj.deleted_at = datetime.now(UTC)

        async with db._use_session() as s:
            r = await db._ls_impl("/", session=s)
        assert r.success
        assert r.paths == ("/a.txt",)

    async def test_ls_with_candidates_known_kind(self, db: DatabaseFileSystem):
        """Candidates with kind set skip the DB kind lookup."""
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "a", session=s)
            await db._write_impl("/lib/b.py", "b", session=s)

        candidates = GroverResult(candidates=[
            Candidate(path="/src", kind="directory"),
            Candidate(path="/lib", kind="directory"),
        ])
        async with db._use_session() as s:
            r = await db._ls_impl(candidates=candidates, session=s)
        assert r.success
        assert set(r.paths) == {"/src/a.py", "/lib/b.py"}

    async def test_ls_with_candidates_unknown_kind(self, db: DatabaseFileSystem):
        """Candidates with kind=None trigger a DB lookup."""
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "a", session=s)

        candidates = GroverResult(candidates=[Candidate(path="/src")])
        async with db._use_session() as s:
            r = await db._ls_impl(candidates=candidates, session=s)
        assert r.success
        assert r.paths == ("/src/a.py",)

    async def test_ls_with_candidates_mixed_files_and_dirs(self, db: DatabaseFileSystem):
        """Batch ls on a mix of files and directories."""
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "code", session=s)
            await db._write_impl("/src/auth.py/.chunks/login", "chunk", session=s)
            await db._write_impl("/lib/utils.py", "utils", session=s)

        candidates = GroverResult(candidates=[
            Candidate(path="/src", kind="directory"),
            Candidate(path="/src/auth.py", kind="file"),
        ])
        async with db._use_session() as s:
            r = await db._ls_impl(candidates=candidates, session=s)
        assert r.success
        paths = set(r.paths)
        # Directory child
        assert "/src/auth.py" in paths
        # File metadata children
        assert "/src/auth.py/.chunks/login" in paths
        assert "/src/auth.py/.versions/1" in paths

    async def test_ls_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/code", db)

        await root.write("/code/src/app.py", "code")
        r = await root.ls("/code/src")
        assert r.success
        assert r.paths == ("/code/src/app.py",)

    async def test_ls_rejects_both_path_and_candidates(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._ls_impl(
                "/src",
                candidates=GroverResult(candidates=[Candidate(path="/lib", kind="directory")]),
                session=s,
            )
        assert not r.success


# ------------------------------------------------------------------
# Part 7: Delete
# ------------------------------------------------------------------


class TestDelete:
    async def test_soft_delete_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "content", session=s)

        async with db._use_session() as s:
            r = await db._delete_impl("/file.txt", session=s)
        assert r.success
        assert "/file.txt" in r.paths

        # Not readable
        async with db._use_session() as s:
            r = await db._read_impl("/file.txt", session=s)
        assert not r.success

        # Still in DB
        async with db._use_session() as s:
            obj = await db._get_object("/file.txt", s, include_deleted=True)
        assert obj is not None
        assert obj.deleted_at is not None

    async def test_permanent_delete(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.txt", "content", session=s)

        async with db._use_session() as s:
            r = await db._delete_impl("/file.txt", permanent=True, session=s)
        assert r.success

        async with db._use_session() as s:
            obj = await db._get_object("/file.txt", s, include_deleted=True)
        assert obj is None

    async def test_soft_delete_cascades_to_metadata(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "v1", session=s)
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "v2", session=s)
        async with db._use_session() as s:
            await db._write_impl("/auth.py/.chunks/login", "chunk", session=s)

        async with db._use_session() as s:
            r = await db._delete_impl("/auth.py", session=s)
        assert r.success
        # Result includes the file + cascaded children
        assert len(r.candidates) > 1

        async with db._use_session() as s:
            v1 = await db._get_object("/auth.py/.versions/1", s, include_deleted=True)
            chunk = await db._get_object("/auth.py/.chunks/login", s, include_deleted=True)
        assert v1 is not None and v1.deleted_at is not None
        assert chunk is not None and chunk.deleted_at is not None

    async def test_permanent_delete_cascades_to_metadata(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "code", session=s)
            await db._write_impl("/auth.py/.chunks/fn", "chunk", session=s)

        async with db._use_session() as s:
            await db._delete_impl("/auth.py", permanent=True, session=s)

        async with db._use_session() as s:
            obj = await db._get_object("/auth.py", s, include_deleted=True)
            chunk = await db._get_object("/auth.py/.chunks/fn", s, include_deleted=True)
            version = await db._get_object("/auth.py/.versions/1", s, include_deleted=True)
        assert obj is None
        assert chunk is None
        assert version is None

    async def test_soft_delete_directory_cascades_all(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "a", session=s)
            await db._write_impl("/src/b.py", "b", session=s)

        async with db._use_session() as s:
            await db._delete_impl("/src", session=s)

        async with db._use_session() as s:
            a = await db._get_object("/src/a.py", s, include_deleted=True)
            b = await db._get_object("/src/b.py", s, include_deleted=True)
        assert a is not None and a.deleted_at is not None
        assert b is not None and b.deleted_at is not None

    async def test_permanent_delete_directory_cascades_all(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "a", session=s)
            await db._write_impl("/src/b.py", "b", session=s)

        async with db._use_session() as s:
            await db._delete_impl("/src", permanent=True, session=s)

        async with db._use_session() as s:
            a = await db._get_object("/src/a.py", s, include_deleted=True)
            b = await db._get_object("/src/b.py", s, include_deleted=True)
            src = await db._get_object("/src", s, include_deleted=True)
        assert a is None
        assert b is None
        assert src is None

    async def test_delete_nonexistent(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._delete_impl("/nope.txt", session=s)
        assert not r.success
        assert "Not found" in r.error_message

    async def test_delete_with_candidates(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.txt", "a", session=s)
            await db._write_impl("/b.txt", "b", session=s)

        candidates = GroverResult(candidates=[
            Candidate(path="/a.txt"),
            Candidate(path="/b.txt"),
        ])
        async with db._use_session() as s:
            r = await db._delete_impl(candidates=candidates, session=s)
        assert r.success

        async with db._use_session() as s:
            a = await db._get_object("/a.txt", s)
            b = await db._get_object("/b.txt", s)
        assert a is None
        assert b is None

    async def test_delete_connection(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "a", session=s)
        async with db._use_session() as s:
            await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)

        conn_path = "/a.py/.connections/imports/b.py"
        async with db._use_session() as s:
            r = await db._delete_impl(conn_path, session=s)
        assert r.success

        async with db._use_session() as s:
            obj = await db._get_object(conn_path, s)
        assert obj is None

    async def test_write_revives_soft_deleted_file(self, db: DatabaseFileSystem):
        """Soft-deleted files can be overwritten (revived)."""
        async with db._use_session() as s:
            await db._write_impl("/revive.txt", "v1", session=s)
        async with db._use_session() as s:
            await db._delete_impl("/revive.txt", session=s)

        async with db._use_session() as s:
            r = await db._write_impl("/revive.txt", "v2", session=s)
        assert r.success
        assert r.content == "v2"

    async def test_delete_rejects_both_path_and_candidates(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._delete_impl(
                "/a.txt",
                candidates=GroverResult(candidates=[Candidate(path="/b.txt")]),
                session=s,
            )
        assert not r.success

    async def test_delete_root_rejected(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._delete_impl("/", session=s)
        assert not r.success
        assert "root" in r.error_message.lower()

    async def test_non_cascade_delete_empty_dir(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._mkdir_impl("/empty", session=s)
        async with db._use_session() as s:
            r = await db._delete_impl("/empty", cascade=False, session=s)
        assert r.success

    async def test_non_cascade_delete_nonempty_dir_rejected(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/nonempty/file.txt", "x", session=s)
        async with db._use_session() as s:
            r = await db._delete_impl("/nonempty", cascade=False, session=s)
        assert not r.success
        assert "not empty" in r.error_message.lower()

        # Directory and child must still exist
        async with db._use_session() as s:
            d = await db._get_object("/nonempty", s)
            f = await db._get_object("/nonempty/file.txt", s)
        assert d is not None
        assert f is not None

    async def test_non_cascade_delete_file_no_metadata(self, db: DatabaseFileSystem):
        """A file with no metadata children (chunks/connections) can be
        non-cascade deleted.  Note: it still has a version row, so this
        tests that versions do block non-cascade delete."""
        async with db._use_session() as s:
            await db._write_impl("/bare.txt", "x", session=s)
        # File has a version row — non-cascade should reject
        async with db._use_session() as s:
            r = await db._delete_impl("/bare.txt", cascade=False, session=s)
        assert not r.success
        assert "not empty" in r.error_message.lower()

    async def test_non_cascade_delete_file_with_chunks_rejected(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/chunked.py", "code", session=s)
            await db._write_impl("/chunked.py/.chunks/fn", "def fn():", session=s)
        async with db._use_session() as s:
            r = await db._delete_impl("/chunked.py", cascade=False, session=s)
        assert not r.success
        assert "not empty" in r.error_message.lower()

    async def test_cascade_true_still_works(self, db: DatabaseFileSystem):
        """Explicit cascade=True behaves the same as the default."""
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "a", session=s)
        async with db._use_session() as s:
            r = await db._delete_impl("/src", cascade=True, session=s)
        assert r.success
        assert len(r.candidates) > 1

    async def test_non_cascade_batch_mixed(self, db: DatabaseFileSystem):
        """Batch with some empty and some non-empty paths."""
        async with db._use_session() as s:
            await db._mkdir_impl("/ok_dir", session=s)
            await db._write_impl("/full_dir/file.txt", "x", session=s)

        candidates = GroverResult(candidates=[
            Candidate(path="/ok_dir"),
            Candidate(path="/full_dir"),
        ])
        async with db._use_session() as s:
            r = await db._delete_impl(candidates=candidates, cascade=False, session=s)
        # Partial: ok_dir deleted, full_dir rejected
        assert not r.success  # errors present
        assert "/ok_dir" in r.paths
        assert "/full_dir" not in r.paths


# ------------------------------------------------------------------
# Part 8: mkconn
# ------------------------------------------------------------------


class TestMkconn:
    async def test_mkconn_creates_connection(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "code", session=s)

        async with db._use_session() as s:
            r = await db._mkconn_impl("/auth.py", "/utils.py", "imports", session=s)
        assert r.success
        assert r.file.kind == "connection"
        assert r.file.path == "/auth.py/.connections/imports/utils.py"

    async def test_mkconn_stores_correct_fields(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "code", session=s)

        async with db._use_session() as s:
            await db._mkconn_impl("/src/auth.py", "/src/utils.py", "imports", session=s)

        async with db._use_session() as s:
            conn = await db._get_object(
                "/src/auth.py/.connections/imports/src/utils.py", s,
            )
        assert conn is not None
        assert conn.kind == "connection"
        assert conn.source_path == "/src/auth.py"
        assert conn.target_path == "/src/utils.py"
        assert conn.connection_type == "imports"
        assert conn.parent_path == "/src/auth.py"

    async def test_mkconn_source_not_found(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._mkconn_impl("/nope.py", "/utils.py", "imports", session=s)
        assert not r.success
        assert "Source not found" in r.error_message

    async def test_mkconn_duplicate_updates(self, db: DatabaseFileSystem):
        """Writing the same connection again updates it (via write upsert)."""
        async with db._use_session() as s:
            await db._write_impl("/a.py", "a", session=s)
        async with db._use_session() as s:
            await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)
        async with db._use_session() as s:
            r = await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)
        assert r.success

    async def test_mkconn_revives_soft_deleted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "a", session=s)
        async with db._use_session() as s:
            await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)

        conn_path = "/a.py/.connections/imports/b.py"
        async with db._use_session() as s:
            await db._delete_impl(conn_path, session=s)

        async with db._use_session() as s:
            r = await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)
        assert r.success

        async with db._use_session() as s:
            conn = await db._get_object(conn_path, s)
        assert conn is not None
        assert conn.deleted_at is None

    async def test_mkconn_missing_args(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._mkconn_impl(session=s)
        assert not r.success

    async def test_mkconn_rejects_both_args_and_objects(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._mkconn_impl(
                "/a.py", "/b.py", "imports",
                objects=[GroverObject(path="/x.py/.connections/imports/y.py", kind="connection")],
                session=s,
            )
        assert not r.success

    async def test_mkconn_with_objects_batch(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "a", session=s)
            await db._write_impl("/b.py", "b", session=s)

        conns = [
            GroverObject(
                path="/a.py/.connections/imports/b.py",
                kind="connection",
                source_path="/a.py",
                target_path="/b.py",
                connection_type="imports",
            ),
            GroverObject(
                path="/b.py/.connections/calls/a.py",
                kind="connection",
                source_path="/b.py",
                target_path="/a.py",
                connection_type="calls",
            ),
        ]
        async with db._use_session() as s:
            r = await db._mkconn_impl(objects=conns, session=s)
        assert r.success
        assert len(r.candidates) == 2

    async def test_mkconn_objects_validates_sources(self, db: DatabaseFileSystem):
        """Batch mkconn with objects rejects missing sources."""
        conns = [
            GroverObject(
                path="/ghost.py/.connections/imports/b.py",
                kind="connection",
                source_path="/ghost.py",
                target_path="/b.py",
                connection_type="imports",
            ),
        ]
        async with db._use_session() as s:
            r = await db._mkconn_impl(objects=conns, session=s)
        assert not r.success
        assert "Source not found" in r.error_message

    async def test_mkconn_visible_in_ls(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/auth.py", "code", session=s)
        async with db._use_session() as s:
            await db._mkconn_impl("/auth.py", "/utils.py", "imports", session=s)

        async with db._use_session() as s:
            r = await db._ls_impl("/auth.py", session=s)
        assert r.success
        assert "/auth.py/.connections/imports/utils.py" in set(r.paths)

    async def test_mkconn_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/code", db)

        await root.write("/code/auth.py", "code")
        r = await root.mkconn("/code/auth.py", "/code/utils.py", "imports")
        assert r.success
        assert r.file.path == "/code/auth.py/.connections/imports/utils.py"


# ------------------------------------------------------------------
# Part 9: mkdir
# ------------------------------------------------------------------


class TestMkdir:
    async def test_mkdir_creates_directory(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._mkdir_impl("/data", session=s)
        assert r.success
        assert r.file.kind == "directory"

        async with db._use_session() as s:
            obj = await db._get_object("/data", s)
        assert obj is not None
        assert obj.kind == "directory"

    async def test_mkdir_creates_parents(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._mkdir_impl("/a/b/c", session=s)
        assert r.success

        async with db._use_session() as s:
            for p in ["/a", "/a/b", "/a/b/c"]:
                obj = await db._get_object(p, s)
                assert obj is not None, f"Missing: {p}"
                assert obj.kind == "directory"

    async def test_mkdir_existing_is_noop(self, db: DatabaseFileSystem):
        """mkdir on an existing directory succeeds (like mkdir -p)."""
        async with db._use_session() as s:
            await db._mkdir_impl("/data", session=s)
        async with db._use_session() as s:
            r = await db._mkdir_impl("/data", session=s)
        assert r.success

    async def test_mkdir_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/store", db)

        r = await root.mkdir("/store/docs")
        assert r.success
        assert r.file.path == "/store/docs"


# ------------------------------------------------------------------
# Part 10: copy
# ------------------------------------------------------------------


class TestCopy:
    async def test_copy_file(self, db: DatabaseFileSystem):
        await db.write("/orig.py", "content")
        async with db._use_session() as s:
            r = await db._copy_impl(
                ops=[TwoPathOperation(src="/orig.py", dest="/copy.py")],
                session=s,
            )
        assert r.success
        assert r.file.path == "/copy.py"

        # Both exist with same content
        r1 = await db.read("/orig.py")
        r2 = await db.read("/copy.py")
        assert r1.content == "content"
        assert r2.content == "content"

    async def test_copy_nonexistent_source(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._copy_impl(
                ops=[TwoPathOperation(src="/nope.py", dest="/copy.py")],
                session=s,
            )
        assert not r.success

    async def test_copy_overwrite_false_rejects(self, db: DatabaseFileSystem):
        await db.write("/a.py", "a")
        await db.write("/b.py", "b")
        async with db._use_session() as s:
            r = await db._copy_impl(
                ops=[TwoPathOperation(src="/a.py", dest="/b.py")],
                overwrite=False,
                session=s,
            )
        assert not r.success

    async def test_copy_batch(self, db: DatabaseFileSystem):
        await db.write("/x.py", "x")
        await db.write("/y.py", "y")
        async with db._use_session() as s:
            r = await db._copy_impl(
                ops=[
                    TwoPathOperation(src="/x.py", dest="/x_copy.py"),
                    TwoPathOperation(src="/y.py", dest="/y_copy.py"),
                ],
                session=s,
            )
        assert r.success
        assert len(r.candidates) == 2

    async def test_copy_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/app.py", "code")
        r = await root.copy(src="/code/app.py", dest="/code/app_bak.py")
        assert r.success
        r2 = await root.read("/code/app_bak.py")
        assert r2.content == "code"


# ------------------------------------------------------------------
# Part 11: move
# ------------------------------------------------------------------


class TestMove:
    async def test_move_file(self, db: DatabaseFileSystem):
        await db.write("/old.py", "content")
        async with db._use_session() as s:
            r = await db._move_impl(
                ops=[TwoPathOperation(src="/old.py", dest="/new.py")],
                session=s,
            )
        assert r.success

        # Old path gone, new path has content
        r1 = await db.read("/old.py")
        assert not r1.success
        r2 = await db.read("/new.py")
        assert r2.content == "content"

    async def test_move_directory_cascades(self, db: DatabaseFileSystem):
        await db.write("/src/a.py", "a")
        await db.write("/src/b.py", "b")
        async with db._use_session() as s:
            r = await db._move_impl(
                ops=[TwoPathOperation(src="/src", dest="/lib")],
                session=s,
            )
        assert r.success

        r1 = await db.read("/lib/a.py")
        assert r1.content == "a"
        r2 = await db.read("/lib/b.py")
        assert r2.content == "b"
        # Old paths gone
        r3 = await db.read("/src/a.py")
        assert not r3.success

    async def test_move_cascades_metadata(self, db: DatabaseFileSystem):
        """Chunks and versions follow the file."""
        await db.write("/old.py", "v1")
        await db.write("/old.py/.chunks/fn", "def fn():")
        async with db._use_session() as s:
            r = await db._move_impl(
                ops=[TwoPathOperation(src="/old.py", dest="/new.py")],
                session=s,
            )
        assert r.success

        # Chunk moved
        rc = await db.read("/new.py/.chunks/fn")
        assert rc.content == "def fn():"
        # Version moved
        async with db._use_session() as s:
            v = await db._get_object("/new.py/.versions/1", s)
        assert v is not None

    async def test_move_updates_incoming_connection_targets(self, db: DatabaseFileSystem):
        """Connections from other files that target the moved file get updated."""
        await db.write("/a.py", "a")
        await db.write("/b.py", "b")
        async with db._use_session() as s:
            await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)

        # Move the target
        async with db._use_session() as s:
            await db._move_impl(
                ops=[TwoPathOperation(src="/b.py", dest="/c.py")],
                session=s,
            )

        # The connection from /a.py should now point to /c.py
        async with db._use_session() as s:
            old_conn = await db._get_object("/a.py/.connections/imports/b.py", s)
            new_conn = await db._get_object("/a.py/.connections/imports/c.py", s)
        assert old_conn is None
        assert new_conn is not None
        assert new_conn.target_path == "/c.py"

    async def test_move_updates_outgoing_connection_source(self, db: DatabaseFileSystem):
        """When source file moves, its outgoing connections update source_path."""
        await db.write("/a.py", "a")
        await db.write("/b.py", "b")
        async with db._use_session() as s:
            await db._mkconn_impl("/a.py", "/b.py", "imports", session=s)

        # Move the source
        async with db._use_session() as s:
            await db._move_impl(
                ops=[TwoPathOperation(src="/a.py", dest="/z.py")],
                session=s,
            )

        # Connection should now live under /z.py with updated source_path
        async with db._use_session() as s:
            conn = await db._get_object("/z.py/.connections/imports/b.py", s)
        assert conn is not None
        assert conn.source_path == "/z.py"
        assert conn.target_path == "/b.py"

    async def test_move_nonexistent_source(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._move_impl(
                ops=[TwoPathOperation(src="/nope.py", dest="/new.py")],
                session=s,
            )
        assert not r.success

    async def test_move_occupied_dest_rejected(self, db: DatabaseFileSystem):
        await db.write("/a.py", "a")
        await db.write("/b.py", "b")
        async with db._use_session() as s:
            r = await db._move_impl(
                ops=[TwoPathOperation(src="/a.py", dest="/b.py")],
                session=s,
            )
        assert not r.success
        assert "occupied" in r.error_message.lower()
        # Both files unchanged
        assert (await db.read("/a.py")).content == "a"
        assert (await db.read("/b.py")).content == "b"

    async def test_move_through_public_api(self, db: DatabaseFileSystem, engine):
        root = GroverFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/old.py", "content")
        r = await root.move(src="/code/old.py", dest="/code/new.py")
        assert r.success
        r2 = await root.read("/code/new.py")
        assert r2.content == "content"


# ------------------------------------------------------------------
# Part 10: Glob
# ------------------------------------------------------------------


class TestGlob:
    async def test_glob_matches_files(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "code", session=s)
            await db._write_impl("/src/db.py", "code", session=s)
            await db._write_impl("/src/util.js", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/src/*.py", session=s)
        assert r.success
        assert set(r.paths) == {"/src/auth.py", "/src/db.py"}

    async def test_glob_single_star_does_not_cross_segments(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a/b/c.py", "code", session=s)
            await db._write_impl("/a/x.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/a/*.py", session=s)
        assert r.paths == ("/a/x.py",)

    async def test_glob_double_star_crosses_segments(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a/b/c.py", "code", session=s)
            await db._write_impl("/a/x.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/a/**/*.py", session=s)
        assert set(r.paths) == {"/a/b/c.py", "/a/x.py"}

    async def test_glob_question_mark(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "code", session=s)
            await db._write_impl("/ab.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/?.py", session=s)
        assert r.paths == ("/a.py",)

    async def test_glob_character_class(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "code", session=s)
            await db._write_impl("/b.py", "code", session=s)
            await db._write_impl("/c.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/[ab].py", session=s)
        assert set(r.paths) == {"/a.py", "/b.py"}

    async def test_glob_returns_files_and_dirs_only(self, db: DatabaseFileSystem):
        """Metadata kinds (chunks, versions, connections) are excluded."""
        async with db._use_session() as s:
            await db._write_impl("/src/auth.py", "code", session=s)
            await db._write_impl("/src/auth.py/.chunks/login", "chunk", session=s)
            await db._mkconn_impl("/src/auth.py", "/src/db.py", "imports", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/**", session=s)
        kinds = {c.kind for c in r.candidates}
        assert kinds <= {"file", "directory"}

    async def test_glob_no_match(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/hello.txt", "hi", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/*.py", session=s)
        assert r.success
        assert len(r) == 0

    async def test_glob_with_candidates(self, db: DatabaseFileSystem):
        """When candidates are provided, filter them in-memory."""
        cands = GroverResult(candidates=[
            Candidate(path="/src/auth.py", kind="file"),
            Candidate(path="/src/db.py", kind="file"),
            Candidate(path="/docs/readme.md", kind="file"),
        ])
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/src/*.py", candidates=cands, session=s)
        assert set(r.paths) == {"/src/auth.py", "/src/db.py"}

    async def test_glob_with_candidates_preserves_detail_chain(self, db: DatabaseFileSystem):
        """Prior details from incoming candidates survive through glob filtering."""
        prior = Detail(operation="search", score=0.9)
        cands = GroverResult(candidates=[
            Candidate(path="/src/auth.py", kind="file", details=(prior,)),
            Candidate(path="/docs/readme.md", kind="file", details=(prior,)),
        ])
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/src/*.py", candidates=cands, session=s)
        assert len(r) == 1
        c = r.candidates[0]
        assert len(c.details) == 2
        assert c.details[0].operation == "search"
        assert c.details[0].score == 0.9
        assert c.details[1].operation == "glob"

    async def test_glob_empty_pattern_errors(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="", session=s)
        assert not r.success

    async def test_glob_results_sorted_by_path(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/z.py", "z", session=s)
            await db._write_impl("/a.py", "a", session=s)
            await db._write_impl("/m.py", "m", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/*.py", session=s)
        assert r.paths == ("/a.py", "/m.py", "/z.py")

    async def test_glob_excludes_soft_deleted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/keep.py", "keep", session=s)
            await db._write_impl("/gone.py", "gone", session=s)
            await db._delete_impl("/gone.py", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/*.py", session=s)
        assert r.paths == ("/keep.py",)

    async def test_glob_includes_directories(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._glob_impl(pattern="/src", session=s)
        assert "/src" in r.paths
        assert r.candidates[0].kind == "directory"

    async def test_glob_through_public_api(self, db: DatabaseFileSystem, engine):
        root = DatabaseFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/auth.py", "code")
        await root.write("/code/db.py", "code")
        r = await root.glob("**/*.py")
        assert {"/code/auth.py", "/code/db.py"} <= set(r.paths)


# ------------------------------------------------------------------
# Part 11: Grep
# ------------------------------------------------------------------


class TestGrep:
    async def test_grep_finds_matching_lines(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "line one\ntimeout = 30\nline three", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="timeout", session=s)
        assert r.success
        assert len(r) == 1
        assert r.file.path == "/a.py"
        meta = r.file.details[0].metadata
        assert meta["match_count"] == 1
        assert meta["line_matches"][0]["line"] == 2
        assert "timeout" in meta["line_matches"][0]["text"]

    async def test_grep_multiple_matches_in_file(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "TODO: fix\nok\nTODO: refactor", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="TODO", session=s)
        assert r.file.details[0].metadata["match_count"] == 2
        assert r.file.details[0].score == 2.0

    async def test_grep_across_multiple_files(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "import os", session=s)
            await db._write_impl("/b.py", "import sys", session=s)
            await db._write_impl("/c.py", "no imports here", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="^import", session=s)
        assert set(r.paths) == {"/a.py", "/b.py"}

    async def test_grep_case_insensitive(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "Error occurred", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="error", case_sensitive=True, session=s)
        assert len(r) == 0
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="error", case_sensitive=False, session=s)
        assert len(r) == 1

    async def test_grep_max_results(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            for i in range(10):
                await db._write_impl(f"/file{i:02d}.py", "match me", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="match", max_results=3, session=s)
        assert len(r) == 3

    async def test_grep_no_match(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "hello world", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="zzz_no_match", session=s)
        assert r.success
        assert len(r) == 0

    async def test_grep_regex_pattern(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "timeout = 30\ntimeout = 120", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern=r"timeout\s*=\s*\d{3}", session=s)
        assert len(r) == 1
        assert r.file.details[0].metadata["match_count"] == 1
        assert r.file.details[0].metadata["line_matches"][0]["line"] == 2

    async def test_grep_invalid_regex(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="[invalid", session=s)
        assert not r.success

    async def test_grep_empty_pattern_errors(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="", session=s)
        assert not r.success

    async def test_grep_skips_non_file_kinds(self, db: DatabaseFileSystem):
        """Grep only searches file content, not chunks/connections."""
        async with db._use_session() as s:
            await db._write_impl("/a.py", "no match", session=s)
            await db._write_impl("/a.py/.chunks/login", "timeout here", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="timeout", session=s)
        assert len(r) == 0

    async def test_grep_with_candidates_uses_existing_content(self, db: DatabaseFileSystem):
        """When candidates already carry content, no DB hydration needed."""
        cands = GroverResult(candidates=[
            Candidate(path="/a.py", kind="file", content="timeout = 30"),
            Candidate(path="/b.py", kind="file", content="no match"),
        ])
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="timeout", candidates=cands, session=s)
        assert r.paths == ("/a.py",)

    async def test_grep_with_candidates_preserves_detail_chain(self, db: DatabaseFileSystem):
        """Prior details from incoming candidates survive through grep."""
        prior = Detail(operation="glob", score=None)
        cands = GroverResult(candidates=[
            Candidate(path="/a.py", kind="file", content="timeout = 30", details=(prior,)),
        ])
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="timeout", candidates=cands, session=s)
        assert len(r) == 1
        c = r.candidates[0]
        assert len(c.details) == 2
        assert c.details[0].operation == "glob"
        assert c.details[1].operation == "grep"
        assert c.details[1].score == 1.0

    async def test_grep_with_candidates_hydrates_missing_content(self, db: DatabaseFileSystem):
        """Candidates without content get hydrated from DB."""
        async with db._use_session() as s:
            await db._write_impl("/a.py", "timeout = 30", session=s)
        cands = GroverResult(candidates=[
            Candidate(path="/a.py", kind="file"),  # content=None
        ])
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="timeout", candidates=cands, session=s)
        assert r.paths == ("/a.py",)

    async def test_grep_excludes_soft_deleted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a.py", "match me", session=s)
            await db._delete_impl("/a.py", session=s)
        async with db._use_session() as s:
            r = await db._grep_impl(pattern="match", session=s)
        assert len(r) == 0

    async def test_grep_through_public_api(self, db: DatabaseFileSystem, engine):
        root = DatabaseFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/auth.py", "timeout = 30")
        r = await root.grep("timeout")
        assert "/code/auth.py" in r.paths


# ------------------------------------------------------------------
# Part 12: Tree
# ------------------------------------------------------------------


class TestTree:
    async def test_tree_lists_descendants(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "code", session=s)
            await db._write_impl("/src/sub/b.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/src", session=s)
        assert r.success
        paths = set(r.paths)
        assert "/src/a.py" in paths
        assert "/src/sub/b.py" in paths
        assert "/src/sub" in paths

    async def test_tree_max_depth_1(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a/b/c.py", "code", session=s)
            await db._write_impl("/a/x.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/a", max_depth=1, session=s)
        paths = set(r.paths)
        assert "/a/x.py" in paths
        assert "/a/b" in paths
        assert "/a/b/c.py" not in paths

    async def test_tree_max_depth_2(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/a/b/c/d.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/a", max_depth=2, session=s)
        paths = set(r.paths)
        assert "/a/b" in paths
        assert "/a/b/c" in paths
        assert "/a/b/c/d.py" not in paths

    async def test_tree_root(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/top.py", "code", session=s)
            await db._write_impl("/sub/deep.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/", session=s)
        paths = set(r.paths)
        assert "/top.py" in paths
        assert "/sub" in paths
        assert "/sub/deep.py" in paths

    async def test_tree_root_max_depth_1(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/top.py", "code", session=s)
            await db._write_impl("/sub/deep.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/", max_depth=1, session=s)
        paths = set(r.paths)
        assert "/top.py" in paths
        assert "/sub" in paths
        assert "/sub/deep.py" not in paths

    async def test_tree_not_found(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._tree_impl("/nonexistent", session=s)
        assert not r.success
        assert "Not found" in r.error_message

    async def test_tree_not_a_directory(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/file.py", "code", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/file.py", session=s)
        assert not r.success
        assert "Not a directory" in r.error_message

    async def test_tree_excludes_metadata(self, db: DatabaseFileSystem):
        """Chunks, versions, connections are excluded from tree output."""
        async with db._use_session() as s:
            await db._write_impl("/src/a.py", "v1", session=s)
            await db._write_impl("/src/a.py", "v2", session=s)  # creates version
            await db._write_impl("/src/a.py/.chunks/login", "chunk", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/src", session=s)
        kinds = {c.kind for c in r.candidates}
        assert kinds <= {"file", "directory"}

    async def test_tree_empty_directory(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._mkdir_impl("/empty", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/empty", session=s)
        assert r.success
        assert len(r) == 0

    async def test_tree_sorted_by_path(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/d/z.py", "z", session=s)
            await db._write_impl("/d/a.py", "a", session=s)
            await db._write_impl("/d/m.py", "m", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/d", session=s)
        file_paths = [c.path for c in r.candidates if c.kind == "file"]
        assert file_paths == ["/d/a.py", "/d/m.py", "/d/z.py"]

    async def test_tree_excludes_soft_deleted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/d/keep.py", "keep", session=s)
            await db._write_impl("/d/gone.py", "gone", session=s)
            await db._delete_impl("/d/gone.py", session=s)
        async with db._use_session() as s:
            r = await db._tree_impl("/d", session=s)
        assert "/d/keep.py" in r.paths
        assert "/d/gone.py" not in r.paths

    async def test_tree_max_depth_zero_errors(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._tree_impl("/", max_depth=0, session=s)
        assert not r.success
        assert "max_depth" in r.error_message

    async def test_tree_through_public_api(self, db: DatabaseFileSystem, engine):
        root = DatabaseFileSystem(engine=engine)
        await root.add_mount("/code", db)
        await root.write("/code/src/a.py", "code")
        r = await root.tree("/code/src")
        assert "/code/src/a.py" in r.paths

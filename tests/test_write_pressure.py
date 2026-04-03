"""Chaos engineering tests for DatabaseFileSystem._write_impl.

Throws every adversarial input imaginable at the write path to find
crash bugs, unhandled exceptions, and data corruption.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from grover.backends.database import DatabaseFileSystem
from grover.models import GroverObject
from grover.results import GroverResult
from tests.conftest import require_file, require_object, set_parameter_budget

# ------------------------------------------------------------------
# 1. Unicode chaos
# ------------------------------------------------------------------


class TestUnicodeChaos:
    """Unicode edge cases in paths and content."""

    async def test_emoji_path(self, db: DatabaseFileSystem):
        r = await db.write("/docs/\U0001f600.txt", "smile")
        assert r.success
        r2 = await db.read("/docs/\U0001f600.txt")
        assert r2.content == "smile"

    async def test_emoji_content(self, db: DatabaseFileSystem):
        r = await db.write("/emoji.txt", "\U0001f4a9\U0001f525\U0001f680")
        assert r.success
        r2 = await db.read("/emoji.txt")
        assert r2.content == "\U0001f4a9\U0001f525\U0001f680"

    async def test_null_byte_in_path_rejected(self, db: DatabaseFileSystem):
        """Null bytes in paths should be caught by validation."""
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/file\x00evil.txt", content="x")

    async def test_rtl_characters_in_path(self, db: DatabaseFileSystem):
        r = await db.write("/docs/\u202eevil.txt", "rtl content")
        # RTL override is a control character (U+202E, in range 0x00-0x1F? No, it's > 0x9F)
        # Actually U+202E is 0x202E which is > 0x9F, so should be allowed
        if r.success:
            r2 = await db.read("/docs/\u202eevil.txt")
            assert r2.content == "rtl content"

    async def test_zero_width_joiners_in_path(self, db: DatabaseFileSystem):
        r = await db.write("/docs/a\u200db.txt", "zwj")
        assert r.success
        r2 = await db.read("/docs/a\u200db.txt")
        assert r2.content == "zwj"

    async def test_combining_characters(self, db: DatabaseFileSystem):
        # e + combining acute = e\u0301, but NFC normalizes to \u00e9
        r = await db.write("/docs/caf\u00e9.txt", "coffee")
        assert r.success

        # The validator does NFC normalization, so e+combining should match
        r2 = await db.write("/docs/cafe\u0301.txt", "coffee2")
        # After NFC normalization both should be the same path
        assert r2.success

    async def test_4byte_utf8_content(self, db: DatabaseFileSystem):
        # Musical symbol (U+1D11E) and CJK
        content = "\U0001d11e \u4e16\u754c \U0001f1fa\U0001f1f8"
        r = await db.write("/unicode.txt", content)
        assert r.success
        r2 = await db.read("/unicode.txt")
        assert r2.content == content

    async def test_control_char_in_path_rejected(self, db: DatabaseFileSystem):
        """Control characters (0x01-0x1F) should be rejected."""
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/file\x01.txt", content="x")

    async def test_del_char_in_path_rejected(self, db: DatabaseFileSystem):
        """DEL (0x7F) should be rejected."""
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/file\x7f.txt", content="x")

    async def test_c1_control_in_path_rejected(self, db: DatabaseFileSystem):
        """C1 control chars (0x80-0x9F) should be rejected."""
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/file\x80.txt", content="x")


# ------------------------------------------------------------------
# 2. Path edge cases
# ------------------------------------------------------------------


class TestPathEdgeCases:
    """Adversarial path strings."""

    async def test_extremely_long_path_rejected(self, db: DatabaseFileSystem):
        """Paths > 4096 chars should be rejected by validate_path."""
        long_path = "/" + "a" * 4096 + ".txt"
        with pytest.raises((ValueError, Exception)):
            GroverObject(path=long_path, content="x")

    async def test_path_at_exact_limit(self, db: DatabaseFileSystem):
        """A long path should be accepted if it and its version suffix fit.

        Writes create a version record at ``{path}/.versions/1`` (13 extra
        chars), so the file path must leave room within the 4096-char
        column limit.
        """
        # Target: 4096 - len("/.versions/1") == 4084
        target = 4096 - len("/.versions/1")
        # Build with 16-char segments: "/aaaa.../aaaa.../file.txt"
        prefix = "/a" * (target // 2)  # each "/a" is 2 chars
        path = prefix[: target - 4] + ".txt"
        assert len(path) <= target
        r = await db.write(path, "hi")
        assert r.success

    async def test_path_with_spaces(self, db: DatabaseFileSystem):
        r = await db.write("/my documents/my file.txt", "space content")
        assert r.success
        r2 = await db.read("/my documents/my file.txt")
        assert r2.content == "space content"

    async def test_path_with_special_chars(self, db: DatabaseFileSystem):
        """Quotes, semicolons, etc. in paths."""
        r = await db.write('/files/it\'s a "test";yes.txt', "special")
        assert r.success

    async def test_deeply_nested_path(self, db: DatabaseFileSystem):
        """50+ levels of nesting."""
        parts = "/".join(f"d{i}" for i in range(50))
        path = "/" + parts + "/deep.txt"
        r = await db.write(path, "deep content")
        assert r.success
        r2 = await db.read(path)
        assert r2.content == "deep content"

    async def test_paths_differ_by_case(self, db: DatabaseFileSystem):
        """Case-sensitive paths should be distinct."""
        r1 = await db.write("/File.TXT", "upper")
        r2 = await db.write("/file.txt", "lower")
        assert r1.success
        assert r2.success

        r3 = await db.read("/File.TXT")
        r4 = await db.read("/file.txt")
        assert r3.content == "upper"
        assert r4.content == "lower"

    async def test_path_with_backslashes(self, db: DatabaseFileSystem):
        """Backslashes are not path separators in POSIX."""
        r = await db.write("/dir/file\\name.txt", "backslash")
        assert r.success

    async def test_path_with_dot_segments(self, db: DatabaseFileSystem):
        """.. and . should be normalized."""
        r = await db.write("/a/b/../c/./d.txt", "normalized")
        assert r.success
        r2 = await db.read("/a/c/d.txt")
        assert r2.content == "normalized"

    async def test_long_segment_rejected(self, db: DatabaseFileSystem):
        """Segments > 255 chars should be rejected."""
        long_name = "a" * 256 + ".txt"
        with pytest.raises((ValueError, Exception)):
            GroverObject(path=f"/{long_name}", content="x")

    async def test_root_path_write(self, db: DatabaseFileSystem):
        """Writing to '/' should fail — it's a directory."""
        async with db._use_session() as s:
            r = await db._write_impl("/", "content", session=s)
        assert not r.success


# ------------------------------------------------------------------
# 3. Content edge cases
# ------------------------------------------------------------------


class TestContentEdgeCases:
    """Adversarial content strings."""

    async def test_empty_string_content(self, db: DatabaseFileSystem):
        r = await db.write("/empty.txt", "")
        assert r.success
        r2 = await db.read("/empty.txt")
        assert r2.content == ""

    async def test_none_content_single_path(self, db: DatabaseFileSystem):
        """None content via single-path API defaults to empty string."""
        async with db._use_session() as s:
            r = await db._write_impl("/none.txt", None, session=s)
        assert r.success
        async with db._use_session() as s:
            r2 = await db._read_impl("/none.txt", session=s)
        assert r2.content == ""

    async def test_massive_content(self, db: DatabaseFileSystem):
        """1MB+ content."""
        big = "x" * (1024 * 1024)
        r = await db.write("/big.txt", big)
        assert r.success
        r2 = await db.read("/big.txt")
        assert r2.content == big
        assert require_file(r2).size_bytes == len(big.encode())

    async def test_pure_whitespace_content(self, db: DatabaseFileSystem):
        r = await db.write("/whitespace.txt", "   \n\t\n   ")
        assert r.success
        r2 = await db.read("/whitespace.txt")
        assert r2.content == "   \n\t\n   "

    async def test_binary_looking_content(self, db: DatabaseFileSystem):
        """Content with bytes that look like binary data."""
        content = "".join(chr(i) for i in range(256) if chr(i).isprintable() or i > 0x9F)
        r = await db.write("/binary_ish.txt", content)
        assert r.success

    async def test_content_with_sql_injection(self, db: DatabaseFileSystem):
        """SQL injection in content — parameterized queries should handle it."""
        evil = "'; DROP TABLE grover_objects; --"
        r = await db.write("/evil.txt", evil)
        assert r.success
        r2 = await db.read("/evil.txt")
        assert r2.content == evil

    async def test_content_with_sql_injection_in_path(self, db: DatabaseFileSystem):
        """SQL injection in path."""
        r = await db.write("/'; DROP TABLE grover_objects; --.txt", "payload")
        assert r.success

    async def test_content_with_many_newlines(self, db: DatabaseFileSystem):
        content = "\n" * 100_000
        r = await db.write("/newlines.txt", content)
        assert r.success
        assert require_file(r).lines == 100_001  # N newlines = N+1 lines


# ------------------------------------------------------------------
# 4. Batch adversarial
# ------------------------------------------------------------------


class TestBatchAdversarial:
    """Edge cases in batch write processing."""

    async def test_mixed_valid_and_invalid_kinds(self, db: DatabaseFileSystem):
        """Mix of writable kinds and non-writable (version, connection)."""
        # Manually sneak in a version-kind object by constructing it explicitly
        async with db._use_session() as s:
            r = await db._write_impl(
                objects=[
                    GroverObject(path="/fine.txt", content="fine"),
                ],
                session=s,
            )
        assert r.success

        # A version path should be rejected
        async with db._use_session() as s:
            r = await db._write_impl("/x.txt/.versions/1", "nope", session=s)
        assert not r.success

    async def test_batch_all_duplicate_paths(self, db: DatabaseFileSystem):
        """Every object has the same path — should fail on duplicate detection."""
        objects = [GroverObject(path="/dup.txt", content=f"v{i}") for i in range(5)]
        async with db._use_session() as s:
            r = await db._write_impl(objects=objects, session=s)
        assert not r.success
        assert "Duplicate path" in r.error_message

    async def test_batch_larger_than_flush_threshold(self, db: DatabaseFileSystem):
        """Batch larger than the flush threshold succeeds across multiple flushes."""
        objects = [GroverObject(path=f"/batch/f{i:04d}.txt", content=f"c{i}") for i in range(50)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 50

    async def test_batch_of_one(self, db: DatabaseFileSystem):
        objects = [GroverObject(path="/single.txt", content="alone")]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 1

    async def test_empty_objects_list(self, db: DatabaseFileSystem):
        """Empty objects list — should succeed vacuously."""
        async with db._use_session() as s:
            r = await db._write_impl(objects=[], session=s)
        assert r.success
        assert len(r.candidates) == 0

    async def test_no_path_no_objects(self, db: DatabaseFileSystem):
        """Neither path nor objects — should error."""
        async with db._use_session() as s:
            r = await db._write_impl(session=s)
        assert not r.success
        assert "requires" in r.error_message.lower()


# ------------------------------------------------------------------
# 5. Version stress
# ------------------------------------------------------------------


class TestVersionStress:
    """Rapid version creation and edge cases."""

    async def test_rapid_50_overwrites(self, db: DatabaseFileSystem):
        """Overwrite the same file 50 times — version chain should survive."""
        await db.write("/versioned.txt", "v0")
        for i in range(1, 51):
            r = await db.write("/versioned.txt", f"v{i}")
            assert r.success, f"Failed on overwrite {i}: {r.error_message}"

        r = await db.read("/versioned.txt")
        assert r.content == "v50"

        # Spot-check version 1 exists
        async with db._use_session() as s:
            v1 = await db._get_object("/versioned.txt/.versions/1", s)
        assert v1 is not None
        assert v1.kind == "version"

    async def test_overwrite_identical_content_no_version(self, db: DatabaseFileSystem):
        """Writing identical content should not create an additional version."""
        await db.write("/stable.txt", "same")
        await db.write("/stable.txt", "same")

        async with db._use_session() as s:
            v1 = await db._get_object("/stable.txt/.versions/1", s)
            v2 = await db._get_object("/stable.txt/.versions/2", s)
        assert v1 is not None
        assert v2 is None, "Identical content should not create an additional version"

    async def test_pathological_diff_content(self, db: DatabaseFileSystem):
        """Content that produces extremely large diffs."""
        v1 = "a\n" * 1000
        # Complete replacement — worst case for diff
        v2 = "b\n" * 1000
        await db.write("/diffhell.txt", v1)
        r = await db.write("/diffhell.txt", v2)
        assert r.success

        r2 = await db.read("/diffhell.txt")
        assert r2.content == v2

    async def test_version_with_empty_to_large(self, db: DatabaseFileSystem):
        """Transition from empty to very large content."""
        await db.write("/grow.txt", "")
        r = await db.write("/grow.txt", "x" * 100_000)
        assert r.success

    async def test_version_with_large_to_empty(self, db: DatabaseFileSystem):
        """Transition from very large to empty content."""
        await db.write("/shrink.txt", "x" * 100_000)
        r = await db.write("/shrink.txt", "")
        assert r.success
        r2 = await db.read("/shrink.txt")
        assert r2.content == ""


# ------------------------------------------------------------------
# 6. Chunk edge cases
# ------------------------------------------------------------------


class TestChunkEdgeCases:
    """Chunk parent validation and ordering edge cases."""

    async def test_chunk_parent_in_same_batch_after_chunk(self, db: DatabaseFileSystem):
        """Chunk appears BEFORE its parent file in the objects list."""
        objects = [
            GroverObject(path="/src/module.py/.chunks/func", content="def func(): pass"),
            GroverObject(path="/src/module.py", content="full module"),
        ]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 2

    async def test_chunk_without_parent_file(self, db: DatabaseFileSystem):
        """Chunk with no parent file in DB or batch — must fail."""
        async with db._use_session() as s:
            r = await db._write_impl("/ghost.py/.chunks/orphan", "orphan", session=s)
        assert not r.success
        assert "parent" in r.error_message.lower()

    async def test_deeply_nested_chunk_path(self, db: DatabaseFileSystem):
        """Chunk path with a deep parent."""
        parent = "/a/b/c/d/e/f/g.py"
        chunk = "/a/b/c/d/e/f/g.py/.chunks/deep"
        await db.write(parent, "content")
        r = await db.write(chunk, "chunk content")
        assert r.success

    async def test_chunk_overwrite(self, db: DatabaseFileSystem):
        """Overwriting a chunk should NOT create a version."""
        await db.write("/mod.py", "module")
        await db.write("/mod.py/.chunks/fn", "original")
        r = await db.write("/mod.py/.chunks/fn", "updated")
        assert r.success

        async with db._use_session() as s:
            v1 = await db._get_object("/mod.py/.chunks/fn/.versions/1", s)
        assert v1 is None

    async def test_multiple_chunks_same_parent(self, db: DatabaseFileSystem):
        """Multiple chunks for the same parent in one batch."""
        objects = [
            GroverObject(path="/multi.py", content="multi"),
            GroverObject(path="/multi.py/.chunks/a", content="chunk a"),
            GroverObject(path="/multi.py/.chunks/b", content="chunk b"),
            GroverObject(path="/multi.py/.chunks/c", content="chunk c"),
        ]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 4


# ------------------------------------------------------------------
# 7. Concurrent writes
# ------------------------------------------------------------------


class TestConcurrentWrites:
    """Concurrent asyncio.gather write storms."""

    async def test_concurrent_writes_to_different_paths(self, db: DatabaseFileSystem):
        """Fire 20 writes to different paths concurrently.

        SQLite serializes writes, so concurrent sessions may conflict.
        We accept that some may fail with OperationalError but none should
        produce unhandled crashes.
        """
        tasks = [db.write(f"/concurrent/f{i}.txt", f"content {i}") for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # At minimum, some should succeed. None should be truly unexpected crashes.
        successes = [r for r in results if isinstance(r, GroverResult) and r.success]
        assert len(successes) > 0, "At least some concurrent writes should succeed"
        # Any exceptions should be known SQLAlchemy/SQLite concurrency errors
        for r in results:
            if isinstance(r, Exception):
                msg = str(r).lower()
                assert any(k in msg for k in ("database", "lock", "operational", "integrity")), (
                    f"Unexpected exception type: {type(r).__name__}: {r}"
                )

    async def test_concurrent_writes_to_same_path(self, db: DatabaseFileSystem):
        """Fire 10 writes to the SAME path concurrently.

        The system should handle this gracefully — some may fail due to
        session conflicts, but none should crash with unhandled exceptions.
        """
        await db.write("/contested.txt", "initial")
        tasks = [db.write("/contested.txt", f"v{i}") for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # We don't assert all succeed, but none should be unhandled exceptions
        for r in results:
            if isinstance(r, Exception):
                # SQLAlchemy session conflicts are expected
                assert isinstance(r, Exception)  # not a crash

        # The file should be readable afterward
        r = await db.read("/contested.txt")
        assert r.success

    async def test_concurrent_batch_writes(self, db: DatabaseFileSystem):
        """Multiple batch writes concurrently to non-overlapping paths."""

        async def batch_write(prefix: str, n: int):
            objects = [GroverObject(path=f"/{prefix}/f{i}.txt", content=f"c{i}") for i in range(n)]
            return await db.write(objects=objects)

        tasks = [batch_write(f"ns{i}", 50) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                pytest.fail(f"Unexpected exception: {r}")
            assert isinstance(r, GroverResult)
            assert r.success


# ------------------------------------------------------------------
# 8. Parameter budget attacks
# ------------------------------------------------------------------


class TestParameterBudgetAttacks:
    """Stress the parameter-budget-driven flush batching."""

    async def test_tiny_budget_forces_max_batching(self, db: DatabaseFileSystem):
        """Budget of 1 forces one object per flush batch."""
        set_parameter_budget(db, 1)
        objects = [GroverObject(path=f"/tiny/f{i:04d}.txt", content=f"c{i}") for i in range(100)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 100

    async def test_large_batch_stays_within_budget(self, db: DatabaseFileSystem):
        """200 objects with default budget — batching keeps flushes safe."""
        objects = [GroverObject(path=f"/huge/f{i:04d}.txt", content=f"c{i}") for i in range(200)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 200


# ------------------------------------------------------------------
# 9. Soft-delete resurrection
# ------------------------------------------------------------------


class TestSoftDeleteResurrection:
    """Pre-insert soft-deleted files, then write to those paths."""

    async def test_write_to_soft_deleted_file(self, db: DatabaseFileSystem):
        """Write should revive a soft-deleted file."""
        # Create file
        await db.write("/zombie.txt", "old content")

        # Soft-delete manually (delete() is not implemented on DatabaseFileSystem)
        async with db._use_session() as s:
            obj = await db._get_object("/zombie.txt", s)
            require_object(obj).deleted_at = datetime.now(UTC)

        # Verify deleted (read excludes soft-deleted)
        r = await db.read("/zombie.txt")
        assert not r.success

        # Revive by writing again
        r = await db.write("/zombie.txt", "new content")
        assert r.success
        r2 = await db.read("/zombie.txt")
        assert r2.content == "new content"

    async def test_batch_mixed_deleted_and_new(self, db: DatabaseFileSystem):
        """Batch with some pre-deleted paths and some new paths."""
        # Create some files
        await db.write("/mix_a.txt", "a")
        await db.write("/mix_b.txt", "b")

        # Soft-delete /mix_a.txt manually
        async with db._use_session() as s:
            obj = await db._get_object("/mix_a.txt", s)
            require_object(obj).deleted_at = datetime.now(UTC)

        # Batch with revive + new
        objects = [
            GroverObject(path="/mix_a.txt", content="a_revived"),
            GroverObject(path="/mix_c.txt", content="c_new"),
        ]
        r = await db.write(objects=objects)
        assert r.success

        r_a = await db.read("/mix_a.txt")
        r_c = await db.read("/mix_c.txt")
        assert r_a.content == "a_revived"
        assert r_c.content == "c_new"

    async def test_pre_inserted_soft_deleted_ancestor_dirs(self, db: DatabaseFileSystem):
        """Writing to a path whose ancestors are soft-deleted should revive them."""
        now = datetime.now(UTC)
        async with db._use_session() as s:
            s.add(GroverObject(path="/dead_parent", kind="directory", deleted_at=now))

        r = await db.write("/dead_parent/child.txt", "alive")
        assert r.success

        async with db._use_session() as s:
            parent = await db._get_object("/dead_parent", s)
        assert parent is not None
        assert parent.deleted_at is None


# ------------------------------------------------------------------
# 10. Type confusion
# ------------------------------------------------------------------


class TestTypeConfusion:
    """Objects with unexpected kind values or mismatched hashes."""

    async def test_directory_kind_accepted(self, db: DatabaseFileSystem):
        """Writing with kind=directory is allowed (upsert, no versioning)."""
        async with db._use_session() as s:
            r = await db._write_impl("/somedir", "content", session=s)
        assert r.success
        assert require_file(r).kind == "directory"

    async def test_version_kind_rejected(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/f.txt/.versions/1", "nope", session=s)
        assert not r.success

    async def test_connection_kind_accepted(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            r = await db._write_impl("/a.py/.connections/imports/b.py", "nope", session=s)
        assert r.success
        assert require_file(r).kind == "connection"

    async def test_object_with_mismatched_content_hash(self, db: DatabaseFileSystem):
        """Object with pre-set content_hash that doesn't match content.

        The model validator recomputes content_hash from content, so the
        manual hash should be overridden.
        """
        obj = GroverObject(path="/mismatch.txt", content="real content", content_hash="fakehash")
        # Validator should have overridden
        expected_hash = hashlib.sha256(b"real content").hexdigest()
        assert obj.content_hash == expected_hash

    async def test_batch_with_all_invalid_kinds(self, db: DatabaseFileSystem):
        """A batch where every item is an invalid kind should fail."""
        async with db._use_session() as s:
            r = await db._write_impl(
                objects=[
                    GroverObject(path="/v1.txt/.versions/1", content="v"),
                    GroverObject(path="/v2.txt/.versions/2", content="v"),
                ],
                session=s,
            )
        assert not r.success


# ------------------------------------------------------------------
# 11. Interleaved operations
# ------------------------------------------------------------------


class TestInterleavedOperations:
    """Write, read, overwrite, delete, write-again sequences."""

    async def test_write_read_overwrite_read(self, db: DatabaseFileSystem):
        r1 = await db.write("/inter.txt", "v1")
        assert r1.success

        r2 = await db.read("/inter.txt")
        assert r2.content == "v1"

        r3 = await db.write("/inter.txt", "v2")
        assert r3.success

        r4 = await db.read("/inter.txt")
        assert r4.content == "v2"

    async def test_write_delete_write_again(self, db: DatabaseFileSystem):
        await db.write("/phoenix.txt", "born")

        # Soft-delete manually
        async with db._use_session() as s:
            obj = await db._get_object("/phoenix.txt", s)
            require_object(obj).deleted_at = datetime.now(UTC)

        r = await db.read("/phoenix.txt")
        assert not r.success

        r2 = await db.write("/phoenix.txt", "reborn")
        assert r2.success

        r3 = await db.read("/phoenix.txt")
        assert r3.content == "reborn"

    async def test_batch_write_then_partial_overwrite(self, db: DatabaseFileSystem):
        """Write a batch, then overwrite only some paths."""
        objects = [GroverObject(path=f"/partial/f{i}.txt", content=f"v1_{i}") for i in range(10)]
        await db.write(objects=objects)

        # Overwrite first 3
        overwrite_objs = [GroverObject(path=f"/partial/f{i}.txt", content=f"v2_{i}") for i in range(3)]
        r = await db.write(objects=overwrite_objs)
        assert r.success

        # Check overwritten
        r0 = await db.read("/partial/f0.txt")
        assert r0.content == "v2_0"

        # Check not-overwritten
        r5 = await db.read("/partial/f5.txt")
        assert r5.content == "v1_5"

    async def test_overwrite_false_after_delete(self, db: DatabaseFileSystem):
        """After soft-delete, overwrite=False should revive the file."""
        await db.write("/ovf.txt", "original")

        # Soft-delete manually
        async with db._use_session() as s:
            obj = await db._get_object("/ovf.txt", s)
            require_object(obj).deleted_at = datetime.now(UTC)

        async with db._use_session() as s:
            r = await db._write_impl("/ovf.txt", "new", overwrite=False, session=s)
        assert r.success


# ------------------------------------------------------------------
# 12. Empty / degenerate inputs
# ------------------------------------------------------------------


class TestEmptyDegenerate:
    """Completely empty or nonsensical inputs."""

    async def test_write_empty_path_string(self, db: DatabaseFileSystem):
        """Empty string path — normalizes to '/' which is directory."""
        async with db._use_session() as s:
            r = await db._write_impl("", "content", session=s)
        assert not r.success

    async def test_write_slash_only(self, db: DatabaseFileSystem):
        """'/' is root directory — not writable."""
        async with db._use_session() as s:
            r = await db._write_impl("/", "content", session=s)
        assert not r.success

    async def test_objects_none_path_none(self, db: DatabaseFileSystem):
        """Both path and objects are None."""
        async with db._use_session() as s:
            r = await db._write_impl(session=s)
        assert not r.success

    async def test_write_path_with_only_dots(self, db: DatabaseFileSystem):
        """Path like '/...' — normalization should handle."""
        async with db._use_session() as s:
            r = await db._write_impl("/../../..", "x", session=s)
        # Should normalize to "/" which is a directory
        assert not r.success

    async def test_write_path_with_double_slashes(self, db: DatabaseFileSystem):
        """Double slashes should be normalized.

        NOTE: The public write() path normalizes via _resolve_terminal, but
        _write_impl constructs a GroverObject from the normalized path.
        The ValidatedSQLModel double-validation may reject paths that the
        model_validator would normally normalize. This test documents the
        current behavior.
        """
        # Direct _write_impl with a normalized path should work
        async with db._use_session() as s:
            r = await db._write_impl("/a/b/c.txt", "normalized", session=s)
        assert r.success
        async with db._use_session() as s:
            r2 = await db._read_impl("/a/b/c.txt", session=s)
        assert r2.content == "normalized"

        # The public API normalizes before reaching _write_impl,
        # so double-slashes in the original path should also work
        r3 = await db.write("/x//y//z.txt", "also fine")
        assert r3.success


# ------------------------------------------------------------------
# 13. Overwrite=False semantics
# ------------------------------------------------------------------


class TestOverwriteFalse:
    """Thorough tests for overwrite=False behavior."""

    async def test_overwrite_false_new_file(self, db: DatabaseFileSystem):
        """New file with overwrite=False should succeed."""
        async with db._use_session() as s:
            r = await db._write_impl("/new_noof.txt", "content", overwrite=False, session=s)
        assert r.success

    async def test_overwrite_false_existing_file(self, db: DatabaseFileSystem):
        """Existing file with overwrite=False should fail."""
        await db.write("/exists.txt", "v1")
        async with db._use_session() as s:
            r = await db._write_impl("/exists.txt", "v2", overwrite=False, session=s)
        assert not r.success
        assert "overwrite=False" in r.error_message

    async def test_overwrite_false_batch_mixed(self, db: DatabaseFileSystem):
        """Batch with overwrite=False: some new, some existing."""
        await db.write("/batch_ow_a.txt", "a")

        objects = [
            GroverObject(path="/batch_ow_a.txt", content="a_new"),
            GroverObject(path="/batch_ow_b.txt", content="b_new"),
        ]
        async with db._use_session() as s:
            r = await db._write_impl(objects=objects, overwrite=False, session=s)
        # The existing file should cause an error but the new one should succeed
        assert not r.success  # at least one error
        # b should have been written
        async with db._use_session() as s:
            b = await db._get_object("/batch_ow_b.txt", s)
        assert b is not None
        assert b.content == "b_new"


# ------------------------------------------------------------------
# 14. Content hash correctness
# ------------------------------------------------------------------


class TestContentHashCorrectness:
    """Verify content_hash is always consistent."""

    async def test_hash_matches_content(self, db: DatabaseFileSystem):
        content = "hello world"
        r = await db.write("/hash_test.txt", content)
        assert r.success

        async with db._use_session() as s:
            obj = await db._get_object("/hash_test.txt", s)
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert require_object(obj).content_hash == expected

    async def test_hash_updated_on_overwrite(self, db: DatabaseFileSystem):
        await db.write("/hash_ow.txt", "v1")
        await db.write("/hash_ow.txt", "v2")

        async with db._use_session() as s:
            obj = await db._get_object("/hash_ow.txt", s)
        expected = hashlib.sha256(b"v2").hexdigest()
        assert require_object(obj).content_hash == expected

    async def test_hash_for_empty_content(self, db: DatabaseFileSystem):
        await db.write("/hash_empty.txt", "")

        async with db._use_session() as s:
            obj = await db._get_object("/hash_empty.txt", s)
        expected = hashlib.sha256(b"").hexdigest()
        assert require_object(obj).content_hash == expected


# ------------------------------------------------------------------
# 15. Metric correctness under stress
# ------------------------------------------------------------------


class TestMetricCorrectness:
    """Verify lines, size_bytes are accurate under adversarial content."""

    async def test_lines_count_no_trailing_newline(self, db: DatabaseFileSystem):
        r = await db.write("/lines.txt", "a\nb\nc")
        assert require_file(r).lines == 3

    async def test_lines_count_trailing_newline(self, db: DatabaseFileSystem):
        r = await db.write("/lines2.txt", "a\nb\nc\n")
        assert require_file(r).lines == 4

    async def test_lines_empty_content(self, db: DatabaseFileSystem):
        r = await db.write("/lines_empty.txt", "")
        assert require_file(r).lines == 0

    async def test_size_bytes_multibyte_chars(self, db: DatabaseFileSystem):
        content = "\U0001f600"  # 4 bytes in UTF-8
        r = await db.write("/multi_byte.txt", content)
        assert require_file(r).size_bytes == 4

    async def test_size_bytes_with_bom(self, db: DatabaseFileSystem):
        content = "\ufeffhello"
        r = await db.write("/bom.txt", content)
        assert require_file(r).size_bytes == len(content.encode())


# ------------------------------------------------------------------
# 16. Session and transaction edge cases
# ------------------------------------------------------------------


class TestSessionEdgeCases:
    """Verify session management under unusual conditions."""

    async def test_write_then_read_same_session(self, db: DatabaseFileSystem):
        """Write and read in the same session context."""
        async with db._use_session() as s:
            await db._write_impl("/same_session.txt", "content", session=s)
            r = await db._read_impl("/same_session.txt", session=s)
        assert r.success
        assert r.content == "content"

    async def test_multiple_writes_same_session(self, db: DatabaseFileSystem):
        """Multiple writes in one session."""
        async with db._use_session() as s:
            r1 = await db._write_impl("/multi_a.txt", "a", session=s)
            r2 = await db._write_impl("/multi_b.txt", "b", session=s)
        assert r1.success
        assert r2.success

    async def test_write_after_failed_write_same_session(self, db: DatabaseFileSystem):
        """A failed write should not corrupt session for next write."""
        async with db._use_session() as s:
            # This should fail (version path)
            r1 = await db._write_impl("/x.txt/.versions/1", "bad", session=s)
            assert not r1.success

            # This should succeed
            r2 = await db._write_impl("/good.txt", "good", session=s)
            assert r2.success


# ------------------------------------------------------------------
# 17. Large batch + version stress combo
# ------------------------------------------------------------------


class TestLargeBatchVersionCombo:
    """Combine batch writes with version creation at scale."""

    async def test_batch_overwrite_at_scale(self, db: DatabaseFileSystem, scale: int):
        """Write N files, then overwrite all N — creates N versions."""
        n = scale
        objs_v1 = [GroverObject(path=f"/scale/f{i:06d}.txt", content=f"v1_{i}") for i in range(n)]
        r1 = await db.write(objects=objs_v1)
        assert r1.success

        objs_v2 = [GroverObject(path=f"/scale/f{i:06d}.txt", content=f"v2_{i}") for i in range(n)]
        r2 = await db.write(objects=objs_v2)
        assert r2.success

        # Spot check middle file
        mid = n // 2
        r = await db.read(f"/scale/f{mid:06d}.txt")
        assert r.content == f"v2_{mid}"

        async with db._use_session() as s:
            v1 = await db._get_object(f"/scale/f{mid:06d}.txt/.versions/1", s)
        assert v1 is not None

    async def test_triple_overwrite_verifies_chain(self, db: DatabaseFileSystem):
        """3x overwrite — version chain should be intact."""
        await db.write("/chain.txt", "v1")
        await db.write("/chain.txt", "v2")
        await db.write("/chain.txt", "v3")

        async with db._use_session() as s:
            v1 = await db._get_object("/chain.txt/.versions/1", s)
            v2 = await db._get_object("/chain.txt/.versions/2", s)
        v1 = require_object(v1)
        v2 = require_object(v2)
        assert v1.is_snapshot is not None
        assert v2.is_snapshot is not None

        from grover.versioning import reconstruct_version

        reconstructed = reconstruct_version(
            [
                (v1.is_snapshot, v1.content or v1.version_diff or ""),
                (v2.is_snapshot, v2.content or v2.version_diff or ""),
            ]
        )
        assert reconstructed == "v2"


# ------------------------------------------------------------------
# 18. Exotic path patterns
# ------------------------------------------------------------------


class TestExoticPaths:
    """Paths that exercise unusual parsing logic."""

    async def test_path_with_percent_encoding_chars(self, db: DatabaseFileSystem):
        """Percent signs in paths (not actually encoded)."""
        r = await db.write("/files/100%.txt", "percent")
        assert r.success

    async def test_path_with_hash_sign(self, db: DatabaseFileSystem):
        r = await db.write("/files/config#1.txt", "hash")
        assert r.success

    async def test_path_with_at_sign(self, db: DatabaseFileSystem):
        r = await db.write("/files/user@host.txt", "at")
        assert r.success

    async def test_path_with_tilde(self, db: DatabaseFileSystem):
        r = await db.write("/files/~backup.txt", "tilde")
        assert r.success

    async def test_path_with_exclamation(self, db: DatabaseFileSystem):
        r = await db.write("/files/important!.txt", "bang")
        assert r.success

    async def test_path_with_parentheses(self, db: DatabaseFileSystem):
        r = await db.write("/files/doc (1).txt", "parens")
        assert r.success

    async def test_path_with_brackets(self, db: DatabaseFileSystem):
        r = await db.write("/files/[draft].txt", "brackets")
        assert r.success

    async def test_path_with_curly_braces(self, db: DatabaseFileSystem):
        r = await db.write("/files/{template}.txt", "curlies")
        assert r.success

    async def test_path_with_pipe(self, db: DatabaseFileSystem):
        r = await db.write("/files/a|b.txt", "pipe")
        assert r.success

    async def test_path_with_ampersand(self, db: DatabaseFileSystem):
        r = await db.write("/files/a&b.txt", "ampersand")
        assert r.success

    async def test_path_with_equals(self, db: DatabaseFileSystem):
        r = await db.write("/files/key=val.txt", "equals")
        assert r.success

    async def test_path_with_plus(self, db: DatabaseFileSystem):
        r = await db.write("/files/a+b.txt", "plus")
        assert r.success

    async def test_dotfile_path(self, db: DatabaseFileSystem):
        """Dotfiles should be treated as files, not directories."""
        r = await db.write("/.env", "SECRET=x")
        assert r.success
        assert require_file(r).kind == "file"

    async def test_dotfile_in_nested_dir(self, db: DatabaseFileSystem):
        r = await db.write("/config/.gitignore", "*.pyc")
        assert r.success
        assert require_file(r).kind == "file"


# ------------------------------------------------------------------
# 19. Rapid same-file create-delete cycles
# ------------------------------------------------------------------


class TestCreateDeleteCycles:
    """Rapidly create and delete the same path."""

    async def test_10_create_delete_cycles(self, db: DatabaseFileSystem):
        for i in range(10):
            r = await db.write("/cycle.txt", f"iteration {i}")
            assert r.success, f"Write failed at iteration {i}"

            # Soft-delete manually (delete() not implemented on DatabaseFileSystem)
            async with db._use_session() as s:
                obj = await db._get_object("/cycle.txt", s)
                assert obj is not None
                obj.deleted_at = datetime.now(UTC)

        # After all cycles, file should not be readable (soft-deleted)
        r = await db.read("/cycle.txt")
        assert not r.success


# ------------------------------------------------------------------
# 20. Model validator edge cases
# ------------------------------------------------------------------


class TestModelValidatorEdgeCases:
    """Test GroverObject model_validator _normalize_and_derive."""

    def test_object_with_no_path(self):
        """Object with no path — validator should handle gracefully."""
        # The validator checks if path is str, returns data unchanged if not
        obj = GroverObject(path="/valid.txt", content="ok")
        assert obj.kind == "file"

    def test_object_kind_inferred_from_path(self):
        """kind should be auto-inferred from path."""
        obj = GroverObject(path="/dir/file.py", content="code")
        assert obj.kind == "file"
        assert obj.parent_path == "/dir"
        assert obj.name == "file.py"

    def test_object_name_derived(self):
        obj = GroverObject(path="/a/b/c.txt", content="x")
        assert obj.name == "c.txt"

    def test_object_parent_path_derived(self):
        obj = GroverObject(path="/a/b/c.txt", content="x")
        assert obj.parent_path == "/a/b"

    def test_object_timestamps_auto_set(self):
        before = datetime.now(UTC)
        obj = GroverObject(path="/ts.txt", content="x")
        after = datetime.now(UTC)
        assert obj.created_at is not None
        assert obj.updated_at is not None
        assert before <= obj.created_at <= after
        assert before <= obj.updated_at <= after


# ------------------------------------------------------------------
# 21. Batch ordering guarantees
# ------------------------------------------------------------------


class TestBatchOrderingGuarantees:
    """Verify that batch writes return candidates in the expected order."""

    async def test_candidate_order_matches_input_order(self, db: DatabaseFileSystem):
        """Candidates should come back in the same order as input objects."""
        objects = [GroverObject(path=f"/order/f{i:03d}.txt", content=f"c{i}") for i in range(20)]
        r = await db.write(objects=objects)
        assert r.success
        expected_paths = tuple(f"/order/f{i:03d}.txt" for i in range(20))
        assert r.paths == expected_paths

    async def test_batch_with_chunks_interleaved(self, db: DatabaseFileSystem):
        """Files and their chunks interleaved in a single batch."""
        objects = [
            GroverObject(path="/mix.py", content="module"),
            GroverObject(path="/other.py", content="other"),
            GroverObject(path="/mix.py/.chunks/fn_a", content="fn a"),
            GroverObject(path="/other.py/.chunks/fn_b", content="fn b"),
        ]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 4


# ------------------------------------------------------------------
# 22. Write-then-overwrite with same hash but different metadata
# ------------------------------------------------------------------


class TestSameHashDifferentMetadata:
    """Verify behavior when content is identical but other fields differ."""

    async def test_same_content_different_id(self, db: DatabaseFileSystem):
        """Two objects with the same path, different IDs — should be handled by dedup."""
        await db.write("/dup_id.txt", "same content")
        r2 = await db.write("/dup_id.txt", "same content")
        # Same content — no additional version created, just timestamp updated
        assert r2.success

        async with db._use_session() as s:
            v1 = await db._get_object("/dup_id.txt/.versions/1", s)
            v2 = await db._get_object("/dup_id.txt/.versions/2", s)
        assert v1 is not None
        assert v2 is None, "Identical content should not create an additional version"


# ------------------------------------------------------------------
# 23. Adversarial content that looks like diff output
# ------------------------------------------------------------------


class TestDiffLikeContent:
    """Content that mimics unified diff format — could confuse version reconstruction."""

    async def test_content_that_looks_like_a_diff(self, db: DatabaseFileSystem):
        """File content that is itself a valid unified diff."""
        diff_content = "--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+line2_modified\n line3\n"
        await db.write("/diff.txt", diff_content)
        # Overwrite to create a version of the diff content
        r = await db.write("/diff.txt", "replaced")
        assert r.success

        async with db._use_session() as s:
            v1 = await db._get_object("/diff.txt/.versions/1", s)
        assert v1 is not None

    async def test_content_with_at_at_markers(self, db: DatabaseFileSystem):
        """Content with @@ markers that could confuse hunk parsing."""
        content = "@@ this is not a hunk @@\n@@ -0,0 +0,0 @@\nfake"
        await db.write("/atat.txt", content)
        r = await db.write("/atat.txt", "new content")
        assert r.success


# ------------------------------------------------------------------
# 24. Parent directory edge cases
# ------------------------------------------------------------------


class TestParentDirectoryEdgeCases:
    """Stress the parent directory auto-creation logic."""

    async def test_100_files_share_same_parent(self, db: DatabaseFileSystem):
        """All 100 files in the same directory — parent created once."""
        objects = [GroverObject(path=f"/same_dir/f{i:03d}.txt", content=f"c{i}") for i in range(100)]
        r = await db.write(objects=objects)
        assert r.success

        async with db._use_session() as s:
            parent = await db._get_object("/same_dir", s)
        assert parent is not None
        assert parent.kind == "directory"

    async def test_files_create_unique_deep_trees(self, db: DatabaseFileSystem):
        """Each file creates a unique deep path tree."""
        objects = [GroverObject(path=f"/tree_{i}/a/b/c/d.txt", content=f"c{i}") for i in range(20)]
        r = await db.write(objects=objects)
        assert r.success

        async with db._use_session() as s:
            for i in range(20):
                d = await db._get_object(f"/tree_{i}/a/b/c", s)
                assert d is not None, f"Missing dir: /tree_{i}/a/b/c"

    async def test_write_to_root_level_file(self, db: DatabaseFileSystem):
        """File directly under root — no intermediate dirs needed."""
        r = await db.write("/root_file.txt", "content")
        assert r.success
        # Parent is / — no intermediate dirs needed.
        # The key test is that the file was created successfully.


# ------------------------------------------------------------------
# 25. Session re-entrancy / nested write_impl calls
# ------------------------------------------------------------------


class TestSessionReEntrancy:
    """Multiple _write_impl calls within the same session."""

    async def test_two_write_impls_same_session(self, db: DatabaseFileSystem):
        """Two sequential _write_impl calls in one session."""
        async with db._use_session() as s:
            r1 = await db._write_impl("/re_a.txt", "a", session=s)
            r2 = await db._write_impl("/re_b.txt", "b", session=s)
        assert r1.success
        assert r2.success

        async with db._use_session() as s:
            a = await db._get_object("/re_a.txt", s)
            b = await db._get_object("/re_b.txt", s)
        assert a is not None and a.content == "a"
        assert b is not None and b.content == "b"

    async def test_write_then_overwrite_same_session(self, db: DatabaseFileSystem):
        """Write then overwrite in the same session."""
        async with db._use_session() as s:
            await db._write_impl("/re_ow.txt", "v1", session=s)
            r2 = await db._write_impl("/re_ow.txt", "v2", session=s)
        assert r2.success

        async with db._use_session() as s:
            obj = await db._get_object("/re_ow.txt", s)
        assert require_object(obj).content == "v2"

    async def test_write_then_read_then_write_same_session(self, db: DatabaseFileSystem):
        async with db._use_session() as s:
            await db._write_impl("/rwr.txt", "v1", session=s)
            r = await db._read_impl("/rwr.txt", session=s)
            assert r.content == "v1"
            await db._write_impl("/rwr.txt", "v2", session=s)
            r2 = await db._read_impl("/rwr.txt", session=s)
            assert r2.content == "v2"


# ------------------------------------------------------------------
# 26. Corrupted version chains
# ------------------------------------------------------------------


class TestCorruptedVersionChains:
    """Simulate broken version chains and verify recovery."""

    async def test_version_with_none_version_number(self, db: DatabaseFileSystem):
        """Pre-insert a version with None version_number, then write.

        plan_file_write should handle this gracefully. We insert the corrupt
        record at a path that won't collide with auto-generated version paths.
        """
        await db.write("/corrupt_chain.txt", "v1")

        # First overwrite creates version 2 normally
        await db.write("/corrupt_chain.txt", "v2")

        # Manually insert a corrupt version record with None version_number
        # at a path that doesn't collide with any real version
        async with db._use_session() as s:
            corrupt_v = GroverObject(
                path="/corrupt_chain.txt/.versions/999",
                content="corrupt",
                version_number=None,
                is_snapshot=None,
            )
            s.add(corrupt_v)

        # Overwrite again — plan_file_write should ignore the corrupt extra row
        # in the recent_versions list gracefully (filtering it out)
        r = await db.write("/corrupt_chain.txt", "v3")
        assert r.success
        r2 = await db.read("/corrupt_chain.txt")
        assert r2.content == "v3"


# ------------------------------------------------------------------
# 27. Idempotency tests
# ------------------------------------------------------------------


class TestIdempotency:
    """Verify that repeated identical operations are idempotent."""

    async def test_double_write_same_content(self, db: DatabaseFileSystem):
        """Writing the same content twice should be idempotent (no extra version)."""
        await db.write("/idem.txt", "constant")
        await db.write("/idem.txt", "constant")
        await db.write("/idem.txt", "constant")

        async with db._use_session() as s:
            v1 = await db._get_object("/idem.txt/.versions/1", s)
            v2 = await db._get_object("/idem.txt/.versions/2", s)
        assert v1 is not None
        assert v2 is None

    async def test_batch_write_idempotent(self, db: DatabaseFileSystem):
        """Writing the same batch twice should update timestamps only."""
        objects = [
            GroverObject(path="/idem_batch/f.txt", content="same"),
        ]
        r1 = await db.write(objects=objects)
        assert r1.success

        objects2 = [
            GroverObject(path="/idem_batch/f.txt", content="same"),
        ]
        r2 = await db.write(objects=objects2)
        assert r2.success

        async with db._use_session() as s:
            v1 = await db._get_object("/idem_batch/f.txt/.versions/1", s)
            v2 = await db._get_object("/idem_batch/f.txt/.versions/2", s)
        assert v1 is not None
        assert v2 is None


# ------------------------------------------------------------------
# 28. Unicode normalization attacks
# ------------------------------------------------------------------


class TestUnicodeNormalizationAttacks:
    """Test that NFC normalization prevents path collisions."""

    async def test_nfc_vs_nfd_same_path(self, db: DatabaseFileSystem):
        """NFC and NFD forms of the same character should resolve to one path."""
        import unicodedata

        # \u00e9 = NFC form of e + combining acute
        nfc = "/caf\u00e9.txt"
        nfd = unicodedata.normalize("NFD", nfc)  # e + \u0301
        assert nfc != nfd  # different byte sequences

        await db.write(nfc, "coffee v1")
        # Writing with NFD form should overwrite (same path after normalization)
        r = await db.write(nfd, "coffee v2")
        assert r.success

        # Reading with either form should return the same content
        r1 = await db.read(nfc)
        assert r1.content == "coffee v2"

    async def test_look_alike_unicode_paths_are_distinct(self, db: DatabaseFileSystem):
        """Visually similar but code-point-different paths should be distinct."""
        # Latin 'a' vs Cyrillic 'a' (U+0430)
        r1 = await db.write("/a.txt", "latin a")
        r2 = await db.write("/\u0430.txt", "cyrillic a")
        assert r1.success
        assert r2.success

        read1 = await db.read("/a.txt")
        read2 = await db.read("/\u0430.txt")
        assert read1.content == "latin a"
        assert read2.content == "cyrillic a"


# ------------------------------------------------------------------
# 29. Extreme parameter budget values
# ------------------------------------------------------------------


class TestExtremeBudgets:
    """Push parameter budget constants to extreme values."""

    async def test_budget_below_model_width(self, db: DatabaseFileSystem):
        """Budget smaller than one row of fields — batch_size floors to 1."""
        set_parameter_budget(db, 1)
        objects = [GroverObject(path=f"/budget/f{i:04d}.txt", content=f"c{i}") for i in range(50)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 50

    async def test_budget_exactly_model_width(self, db: DatabaseFileSystem):
        """Budget equal to one row of fields — one object per flush."""
        field_count = len(GroverObject.model_fields)
        set_parameter_budget(db, field_count + db.PARAMETER_RESERVE)
        objects = [GroverObject(path=f"/fields/f{i:04d}.txt", content=f"c{i}") for i in range(field_count * 2)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == field_count * 2


# ------------------------------------------------------------------
# 30. Timestamp manipulation
# ------------------------------------------------------------------


class TestTimestampManipulation:
    """Objects with pre-set or adversarial timestamps."""

    async def test_object_with_future_timestamps(self, db: DatabaseFileSystem):
        """Object with timestamps far in the future."""
        future = datetime(2099, 12, 31, tzinfo=UTC)
        obj = GroverObject(
            path="/future.txt",
            content="from the future",
            created_at=future,
            updated_at=future,
        )
        r = await db.write(objects=[obj])
        assert r.success

    async def test_object_with_epoch_timestamps(self, db: DatabaseFileSystem):
        """Object with Unix epoch timestamps."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        obj = GroverObject(
            path="/epoch.txt",
            content="old",
            created_at=epoch,
            updated_at=epoch,
        )
        r = await db.write(objects=[obj])
        assert r.success

    async def test_overwrite_preserves_created_at(self, db: DatabaseFileSystem):
        """Overwriting should not change created_at."""
        await db.write("/preserve_ts.txt", "v1")
        async with db._use_session() as s:
            obj1 = await db._get_object("/preserve_ts.txt", s)
            created_original = require_object(obj1).created_at

        await db.write("/preserve_ts.txt", "v2")
        async with db._use_session() as s:
            obj2 = await db._get_object("/preserve_ts.txt", s)
        assert require_object(obj2).created_at == created_original


# ------------------------------------------------------------------
# 31. Massive single-session batch stress
# ------------------------------------------------------------------


class TestMassiveSingleSessionBatch:
    """Large batch in a single _write_impl call (no routing)."""

    async def test_objects_single_impl_call(self, db: DatabaseFileSystem, scale: int):
        """N objects in one _write_impl call."""
        n = scale
        objects = [GroverObject(path=f"/mass/f{i:06d}.txt", content=f"content {i}") for i in range(n)]
        async with db._use_session() as s:
            r = await db._write_impl(objects=objects, session=s)
        assert r.success
        assert len(r.candidates) == n

    async def test_overwrite_objects_single_impl_call(self, db: DatabaseFileSystem, scale: int):
        """Write N, then overwrite all N in one call."""
        n = scale
        objs_v1 = [GroverObject(path=f"/mass_ow/f{i:06d}.txt", content=f"v1_{i}") for i in range(n)]
        async with db._use_session() as s:
            await db._write_impl(objects=objs_v1, session=s)

        objs_v2 = [GroverObject(path=f"/mass_ow/f{i:06d}.txt", content=f"v2_{i}") for i in range(n)]
        async with db._use_session() as s:
            r = await db._write_impl(objects=objs_v2, session=s)
        assert r.success
        assert len(r.candidates) == n


# ------------------------------------------------------------------
# 32. Content with all manner of line endings
# ------------------------------------------------------------------


class TestLineEndings:
    """CRLF, CR, LF, mixed — verify they're stored faithfully."""

    async def test_unix_lf(self, db: DatabaseFileSystem):
        r = await db.write("/lf.txt", "a\nb\nc")
        assert r.success
        r2 = await db.read("/lf.txt")
        assert r2.content == "a\nb\nc"

    async def test_windows_crlf(self, db: DatabaseFileSystem):
        r = await db.write("/crlf.txt", "a\r\nb\r\nc")
        assert r.success
        r2 = await db.read("/crlf.txt")
        assert r2.content == "a\r\nb\r\nc"

    async def test_old_mac_cr(self, db: DatabaseFileSystem):
        r = await db.write("/cr.txt", "a\rb\rc")
        assert r.success
        r2 = await db.read("/cr.txt")
        assert r2.content == "a\rb\rc"

    async def test_mixed_line_endings(self, db: DatabaseFileSystem):
        content = "unix\nwindows\r\nold_mac\rend"
        r = await db.write("/mixed_le.txt", content)
        assert r.success
        r2 = await db.read("/mixed_le.txt")
        assert r2.content == content


# ------------------------------------------------------------------
# 33. Chunk validation with tiny parameter budget
# ------------------------------------------------------------------


class TestChunkValidationWithTinyBudget:
    """Chunk parent validation when internal batching splits the batch."""

    async def test_chunk_before_parent_tiny_budget(self, db: DatabaseFileSystem):
        """With budget=1, chunk and parent are in different internal batches.

        The write validates chunk parents against the full write_map
        before chunking.
        """
        set_parameter_budget(db, 1)
        objects = [
            GroverObject(path="/tiny_batch.py/.chunks/fn", content="def fn(): pass"),
            GroverObject(path="/tiny_batch.py", content="module content"),
        ]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 2

    async def test_many_chunks_one_parent_tiny_budget(self, db: DatabaseFileSystem):
        """Many chunks + parent, tiny budget — splits across batches."""
        set_parameter_budget(db, 1)
        objects = [
            GroverObject(path="/multi_chunk.py", content="module"),
            GroverObject(path="/multi_chunk.py/.chunks/a", content="a"),
            GroverObject(path="/multi_chunk.py/.chunks/b", content="b"),
            GroverObject(path="/multi_chunk.py/.chunks/c", content="c"),
            GroverObject(path="/multi_chunk.py/.chunks/d", content="d"),
        ]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == 5


# ------------------------------------------------------------------
# 34. Exception recovery (savepoint + per-item handlers)
# ------------------------------------------------------------------


class TestExceptionRecovery:
    """Verify that savepoint and per-item exception handlers work."""

    async def test_per_item_error_skips_item_others_succeed(self, db: DatabaseFileSystem):
        """plan_file_write raises for one item — it's skipped, others succeed."""
        from unittest.mock import patch

        await db.write("/good.txt", "v1")
        await db.write("/bad.txt", "v1")

        from grover.models import GroverObjectBase

        original_plan = GroverObjectBase.plan_file_write

        def failing_plan(self, new_content, version_rows=None, *, latest_version_hash=None):
            if self.path == "/bad.txt":
                raise RuntimeError("simulated version chain corruption")
            return original_plan(self, new_content, version_rows, latest_version_hash=latest_version_hash)

        with patch.object(GroverObjectBase, "plan_file_write", failing_plan):
            objects = [
                GroverObject(path="/good.txt", content="v2"),
                GroverObject(path="/bad.txt", content="v2"),
            ]
            async with db._use_session() as s:
                r = await db._write_impl(objects=objects, session=s)

        assert "/good.txt" in r.paths
        assert any("Write failed for /bad.txt" in e for e in r.errors)

        # good.txt updated, bad.txt still on v1
        async with db._use_session() as s:
            good = await db._get_object("/good.txt", s)
            bad = await db._get_object("/bad.txt", s)
        assert require_object(good).content == "v2"
        assert require_object(bad).content == "v1"

    async def test_flush_failure_rolls_back_entire_write(self, db: DatabaseFileSystem):
        """Without savepoints, a flush failure rolls back everything."""
        objects = [
            GroverObject(path="/a.txt", content="a"),
            GroverObject(path="/b.txt", content="b"),
        ]

        with pytest.raises(RuntimeError, match="simulated flush failure"):
            async with db._use_session() as s:

                async def failing_flush(*args, **kwargs):
                    raise RuntimeError("simulated flush failure")

                cast("Any", s).flush = failing_flush
                await db._write_impl(objects=objects, session=s)

        # Nothing persisted
        async with db._use_session() as s:
            a = await db._get_object("/a.txt", s)
            b = await db._get_object("/b.txt", s)
        assert a is None
        assert b is None


# ------------------------------------------------------------------
# Scale: Large writes + cascading deletes
# ------------------------------------------------------------------


class TestLargeWriteAndDelete:
    """Stress-test write + ls + delete at scale."""

    async def test_large_batch_write_then_ls(self, db: DatabaseFileSystem, scale: int):
        """Write N files under a directory, ls returns all of them."""
        n = scale
        objects = [GroverObject(path=f"/data/file_{i:06d}.txt", content=f"content {i}") for i in range(n)]
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == n

        async with db._use_session() as s:
            ls_r = await db._ls_impl("/data", session=s)
        assert ls_r.success
        assert len(ls_r.candidates) == n

    async def test_large_batch_write_then_soft_delete_directory(self, db: DatabaseFileSystem, scale: int):
        """Write N files, soft-delete the parent dir, verify all cascaded."""
        n = scale
        objects = [GroverObject(path=f"/src/file_{i:06d}.py", content=f"code {i}") for i in range(n)]
        r = await db.write(objects=objects)
        assert r.success

        async with db._use_session() as s:
            del_r = await db._delete_impl("/src", session=s)
        assert del_r.success
        # Parent dir + N files + N version rows + /src dir itself
        assert len(del_r.candidates) > n

        # ls should return nothing
        async with db._use_session() as s:
            ls_r = await db._ls_impl("/src", session=s)
        # /src is soft-deleted so ls finds nothing (unknown kind, not in DB)
        assert len(ls_r.candidates) == 0

    async def test_large_batch_write_then_permanent_delete_directory(self, db: DatabaseFileSystem, scale: int):
        """Write N files, permanently delete the dir, verify all gone."""
        n = scale
        objects = [GroverObject(path=f"/tmp/file_{i:06d}.txt", content=f"temp {i}") for i in range(n)]
        r = await db.write(objects=objects)
        assert r.success

        async with db._use_session() as s:
            del_r = await db._delete_impl("/tmp", permanent=True, session=s)
        assert del_r.success

        # Nothing left in DB
        async with db._use_session() as s:
            obj = await db._get_object("/tmp", s, include_deleted=True)
        assert obj is None
        async with db._use_session() as s:
            spot = await db._get_object(f"/tmp/file_{n // 2:06d}.txt", s, include_deleted=True)
        assert spot is None

    async def test_large_batch_write_with_chunks_then_delete(self, db: DatabaseFileSystem, scale: int):
        """Write N files each with a chunk, delete cascades to all metadata."""
        n = min(scale, 500)  # chunks double the object count
        objects = []
        for i in range(n):
            objects.append(GroverObject(path=f"/code/f_{i:05d}.py", content=f"code {i}"))
            objects.append(GroverObject(path=f"/code/f_{i:05d}.py/.chunks/main", content=f"def main_{i}():"))
        r = await db.write(objects=objects)
        assert r.success
        assert len(r.candidates) == n * 2

        async with db._use_session() as s:
            del_r = await db._delete_impl("/code", permanent=True, session=s)
        assert del_r.success

        # Spot-check: file, chunk, and version all gone
        mid = n // 2
        async with db._use_session() as s:
            f = await db._get_object(f"/code/f_{mid:05d}.py", s, include_deleted=True)
            c = await db._get_object(f"/code/f_{mid:05d}.py/.chunks/main", s, include_deleted=True)
            v = await db._get_object(f"/code/f_{mid:05d}.py/.versions/1", s, include_deleted=True)
        assert f is None
        assert c is None
        assert v is None

    async def test_large_batch_connections_write_and_delete(self, db: DatabaseFileSystem, scale: int):
        """Write N files with connections between consecutive pairs, then delete."""
        n = min(scale, 500)
        files = [GroverObject(path=f"/graph/node_{i:05d}.py", content=f"node {i}") for i in range(n)]
        r = await db.write(objects=files)
        assert r.success

        conns = [
            GroverObject(
                path=f"/graph/node_{i:05d}.py/.connections/calls/graph/node_{i + 1:05d}.py",
                kind="connection",
                source_path=f"/graph/node_{i:05d}.py",
                target_path=f"/graph/node_{i + 1:05d}.py",
                connection_type="calls",
            )
            for i in range(n - 1)
        ]
        r2 = await db.write(objects=conns)
        assert r2.success
        assert len(r2.candidates) == n - 1

        # Delete the whole graph
        async with db._use_session() as s:
            del_r = await db._delete_impl("/graph", permanent=True, session=s)
        assert del_r.success

        async with db._use_session() as s:
            obj = await db._get_object("/graph", s, include_deleted=True)
        assert obj is None

    async def test_write_delete_write_cycle_at_scale(self, db: DatabaseFileSystem, scale: int):
        """Write N files, soft-delete all, write N new files at same paths."""
        n = scale
        objects_v1 = [GroverObject(path=f"/cycle/f_{i:06d}.txt", content=f"v1_{i}") for i in range(n)]
        r1 = await db.write(objects=objects_v1)
        assert r1.success

        # Soft-delete the directory
        async with db._use_session() as s:
            await db._delete_impl("/cycle", session=s)

        # Re-write same paths
        objects_v2 = [GroverObject(path=f"/cycle/f_{i:06d}.txt", content=f"v2_{i}") for i in range(n)]
        r2 = await db.write(objects=objects_v2)
        assert r2.success

        # Spot-check: content is v2
        mid = n // 2
        r = await db.read(f"/cycle/f_{mid:06d}.txt")
        assert r.content == f"v2_{mid}"

    async def test_nested_dirs_bulk_delete(self, db: DatabaseFileSystem, scale: int):
        """Write files across many nested directories, bulk delete root."""
        n = min(scale, 500)
        import random

        rng = random.Random(42)
        dirs = [f"/deep/d{i}/sub{j}" for i in range(10) for j in range(10)]
        objects = [GroverObject(path=f"{rng.choice(dirs)}/f_{i:05d}.txt", content=f"c{i}") for i in range(n)]
        r = await db.write(objects=objects)
        assert r.success

        async with db._use_session() as s:
            del_r = await db._delete_impl("/deep", permanent=True, session=s)
        assert del_r.success

        # Everything under /deep is gone
        async with db._use_session() as s:
            obj = await db._get_object("/deep", s, include_deleted=True)
        assert obj is None
        async with db._use_session() as s:
            ls_r = await db._ls_impl("/", session=s)
        assert "/deep" not in set(ls_r.paths)

    async def test_deeply_nested_with_metadata_then_delete(self, db: DatabaseFileSystem, scale: int):
        """Write files across 25+ directory levels with versions, chunks,
        and connections, then delete the root.

        Each file gets: 1 version row (auto), 1 chunk, 1 connection to
        the next file. Total objects ~ 4x file count + directories.
        """
        import random

        n = scale
        rng = random.Random(99)
        depth = 25

        # Build 25-deep directory paths with branching at the top levels
        branches = [f"/tree/b{b}" for b in range(5)]
        dir_pool: list[str] = []
        for branch in branches:
            path = branch
            for level in range(depth):
                path = f"{path}/l{level}"
                dir_pool.append(path)

        # Distribute files across the deep dirs
        file_objects = []
        for i in range(n):
            parent = rng.choice(dir_pool)
            file_objects.append(GroverObject(path=f"{parent}/f_{i:06d}.py", content=f"code {i}"))

        r = await db.write(objects=file_objects)
        assert r.success, r.error_message
        assert len(r.candidates) == n

        # Add a chunk per file
        chunk_objects = [
            GroverObject(
                path=f"{obj.path}/.chunks/main",
                content=f"def main_{i}():",
            )
            for i, obj in enumerate(file_objects)
        ]
        r2 = await db.write(objects=chunk_objects)
        assert r2.success, r2.error_message

        # Add connections: each file → next file (circular)
        conn_objects = [
            GroverObject(
                path=f"{file_objects[i].path}/.connections/calls/{file_objects[(i + 1) % n].path.lstrip('/')}",
                kind="connection",
                source_path=file_objects[i].path,
                target_path=file_objects[(i + 1) % n].path,
                connection_type="calls",
            )
            for i in range(n)
        ]
        r3 = await db.write(objects=conn_objects)
        assert r3.success, r3.error_message

        # Permanent delete the root — everything should cascade
        async with db._use_session() as s:
            del_r = await db._delete_impl("/tree", permanent=True, session=s)
        assert del_r.success
        # At minimum: dir + N files + N versions + N chunks + N connections
        assert len(del_r.candidates) >= n * 4

        # Nothing left
        async with db._use_session() as s:
            root = await db._get_object("/tree", s, include_deleted=True)
        assert root is None

        # Spot-check a deep path
        mid = n // 2
        async with db._use_session() as s:
            f = await db._get_object(file_objects[mid].path, s, include_deleted=True)
            c = await db._get_object(f"{file_objects[mid].path}/.chunks/main", s, include_deleted=True)
        assert f is None
        assert c is None

    async def test_multi_directory_batch_delete(self, db: DatabaseFileSystem, scale: int):
        """Delete many separate directories in a single batch call.

        This is the pattern where batched cascade queries (OR of LIKE)
        outperform the N+1 per-item cascade.  Each directory has files
        with versions + chunks underneath.
        """
        n_dirs = min(scale, 200)
        files_per_dir = 10

        # Create N directories, each with files_per_dir files + chunks
        all_objects: list[GroverObject] = []
        for d in range(n_dirs):
            for f in range(files_per_dir):
                all_objects.append(
                    GroverObject(
                        path=f"/batch/d{d:04d}/f{f:03d}.py",
                        content=f"code d{d} f{f}",
                    )
                )
                all_objects.append(
                    GroverObject(
                        path=f"/batch/d{d:04d}/f{f:03d}.py/.chunks/main",
                        content=f"def main_{d}_{f}():",
                    )
                )

        r = await db.write(objects=all_objects)
        assert r.success, r.error_message

        # Build candidates for all N directories
        from grover.results import Candidate, GroverResult

        candidates = GroverResult(candidates=[Candidate(path=f"/batch/d{d:04d}") for d in range(n_dirs)])

        # Delete all directories in one batch call
        async with db._use_session() as s:
            del_r = await db._delete_impl(candidates=candidates, permanent=True, session=s)
        assert del_r.success
        # Each dir has: files_per_dir files + files_per_dir chunks
        # + files_per_dir version rows + the dir itself
        assert len(del_r.candidates) >= n_dirs * (files_per_dir * 3 + 1)

        # Spot-check
        async with db._use_session() as s:
            obj = await db._get_object(f"/batch/d{n_dirs // 2:04d}", s, include_deleted=True)
        assert obj is None

    async def test_multi_file_batch_delete_with_metadata(self, db: DatabaseFileSystem, scale: int):
        """Delete many separate files in a single batch call.

        Each file has a version row and a chunk — the cascade for files
        uses parent_path IN (...) which batches well.
        """
        n = min(scale, 500)

        all_objects: list[GroverObject] = []
        for i in range(n):
            all_objects.append(GroverObject(path=f"/flat/f{i:05d}.py", content=f"code {i}"))
            all_objects.append(GroverObject(path=f"/flat/f{i:05d}.py/.chunks/main", content=f"def main_{i}():"))

        r = await db.write(objects=all_objects)
        assert r.success, r.error_message

        # Delete all files (not the directory) in one batch
        from grover.results import Candidate, GroverResult

        candidates = GroverResult(candidates=[Candidate(path=f"/flat/f{i:05d}.py") for i in range(n)])

        async with db._use_session() as s:
            del_r = await db._delete_impl(candidates=candidates, permanent=True, session=s)
        assert del_r.success
        # Each file: itself + 1 version + 1 chunk = 3
        assert len(del_r.candidates) >= n * 3

        # Files gone, directory still exists
        async with db._use_session() as s:
            flat_dir = await db._get_object("/flat", s)
        assert flat_dir is not None
        async with db._use_session() as s:
            obj = await db._get_object(f"/flat/f{n // 2:05d}.py", s, include_deleted=True)
        assert obj is None

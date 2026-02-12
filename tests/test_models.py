"""Tests for database models and diff utilities."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, select

from grover.fs.diff import apply_diff, compute_diff, reconstruct_version
from grover.models import (
    Embedding,
    File,
    FileShare,
    FileShareBase,
    FileVersion,
    GroverEdge,
)

# ---------------------------------------------------------------------------
# Table creation & basic CRUD
# ---------------------------------------------------------------------------


class TestTableCreation:
    def test_grover_files_table_exists(self, engine):
        """grover_files table is created by create_all."""
        assert "grover_files" in engine.dialect.get_table_names(engine.connect())

    def test_grover_file_versions_table_exists(self, engine):
        assert "grover_file_versions" in engine.dialect.get_table_names(engine.connect())

    def test_grover_edges_table_exists(self, engine):
        assert "grover_edges" in engine.dialect.get_table_names(engine.connect())

    def test_grover_embeddings_table_exists(self, engine):
        assert "grover_embeddings" in engine.dialect.get_table_names(engine.connect())


class TestDefaultFactories:
    def test_grover_file_defaults(self, session: Session):
        f = File(path="/hello.txt", name="hello.txt", parent_path="/")
        session.add(f)
        session.commit()
        session.refresh(f)

        assert f.id  # UUID string
        assert f.path == "/hello.txt"
        assert f.current_version == 1
        assert f.deleted_at is None
        assert f.created_at is not None
        assert f.mime_type == "text/plain"
        assert f.is_directory is False
        assert f.content is None
        assert f.content_hash is None
        assert f.original_path is None

    def test_file_version_defaults(self, session: Session):
        fv = FileVersion(file_id="abc", version=1, is_snapshot=True, content="hello")
        session.add(fv)
        session.commit()
        session.refresh(fv)

        assert fv.id
        assert fv.file_id == "abc"
        assert fv.is_snapshot is True
        assert fv.content_hash == ""
        assert fv.size_bytes == 0
        assert fv.created_by is None

    def test_file_version_with_new_fields(self, session: Session):
        fv = FileVersion(
            file_id="abc",
            version=2,
            is_snapshot=False,
            content="diff content",
            content_hash="sha256hash",
            size_bytes=42,
            created_by="agent",
        )
        session.add(fv)
        session.commit()
        session.refresh(fv)

        assert fv.content_hash == "sha256hash"
        assert fv.size_bytes == 42
        assert fv.created_by == "agent"

    def test_grover_edge_defaults(self, session: Session):
        edge = GroverEdge(source_path="/a.py", target_path="/b.py", type="imports")
        session.add(edge)
        session.commit()
        session.refresh(edge)

        assert edge.id
        assert edge.type == "imports"
        assert edge.weight == 1.0

    def test_embedding_defaults(self, session: Session):
        emb = Embedding(file_id="xyz", file_version=1, content_hash="abc123")
        session.add(emb)
        session.commit()
        session.refresh(emb)

        assert emb.id
        assert emb.model_name == ""

    def test_grover_edge_metadata_json_default(self, session: Session):
        edge = GroverEdge(source_path="/a.py", target_path="/b.py", type="imports")
        session.add(edge)
        session.commit()
        session.refresh(edge)

        assert edge.metadata_json == "{}"

    def test_query_round_trip(self, session: Session):
        """Insert and query back a File."""
        f = File(path="/test.py", name="test.py", parent_path="/")
        session.add(f)
        session.commit()

        result = session.exec(select(File).where(File.path == "/test.py")).first()
        assert result is not None
        assert result.path == "/test.py"

    def test_grover_file_directory(self, session: Session):
        d = File(
            path="/src",
            name="src",
            parent_path="/",
            is_directory=True,
        )
        session.add(d)
        session.commit()
        session.refresh(d)

        assert d.is_directory is True

    def test_grover_file_with_content(self, session: Session):
        f = File(
            path="/readme.md",
            name="readme.md",
            parent_path="/",
            content="# Hello",
            content_hash="abc123",
        )
        session.add(f)
        session.commit()
        session.refresh(f)

        assert f.content == "# Hello"
        assert f.content_hash == "abc123"


# ---------------------------------------------------------------------------
# owner_id field
# ---------------------------------------------------------------------------


class TestFileBaseOwnerId:
    def test_file_base_owner_id_default_none(self, session: Session):
        f = File(path="/no_owner.txt", name="no_owner.txt", parent_path="/")
        session.add(f)
        session.commit()
        session.refresh(f)
        assert f.owner_id is None

    def test_file_base_owner_id_set(self, session: Session):
        f = File(
            path="/owned.txt",
            name="owned.txt",
            parent_path="/",
            owner_id="alice",
        )
        session.add(f)
        session.commit()
        session.refresh(f)
        assert f.owner_id == "alice"


# ---------------------------------------------------------------------------
# FileShare model
# ---------------------------------------------------------------------------


class TestFileShare:
    def test_file_share_create(self, session: Session):
        share = FileShare(
            path="/alice/notes.md",
            grantee_id="bob",
            permission="read",
            granted_by="alice",
        )
        session.add(share)
        session.commit()
        session.refresh(share)

        assert share.id  # UUID string
        assert share.path == "/alice/notes.md"
        assert share.grantee_id == "bob"
        assert share.permission == "read"
        assert share.granted_by == "alice"
        assert share.created_at is not None
        assert share.expires_at is None

    def test_file_share_defaults(self, session: Session):
        share = FileShare(
            path="/a/b.txt",
            grantee_id="charlie",
            granted_by="alice",
        )
        session.add(share)
        session.commit()
        session.refresh(share)

        assert share.permission == "read"
        assert share.expires_at is None

    def test_file_share_base_subclass(self, engine):
        """Custom table name via subclassing FileShareBase."""

        class CustomShare(FileShareBase, table=True):
            __tablename__ = "custom_shares"

        SQLModel.metadata.create_all(engine)
        tables = engine.dialect.get_table_names(engine.connect())
        assert "custom_shares" in tables

    def test_file_share_table_exists(self, engine):
        assert "grover_file_shares" in engine.dialect.get_table_names(engine.connect())

    def test_file_share_write_permission(self, session: Session):
        share = FileShare(
            path="/alice/project/",
            grantee_id="bob",
            permission="write",
            granted_by="alice",
        )
        session.add(share)
        session.commit()
        session.refresh(share)
        assert share.permission == "write"


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


class TestFileVersionUniqueConstraint:
    """H4: Duplicate (file_id, version) should be rejected."""

    def test_duplicate_file_version_rejected(self, session: Session):
        from sqlalchemy.exc import IntegrityError

        fv1 = FileVersion(file_id="abc", version=1, is_snapshot=True, content="v1")
        session.add(fv1)
        session.commit()

        fv2 = FileVersion(file_id="abc", version=1, is_snapshot=True, content="v1dup")
        session.add(fv2)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_same_version_different_file_allowed(self, session: Session):
        fv1 = FileVersion(file_id="abc", version=1, is_snapshot=True, content="v1a")
        fv2 = FileVersion(file_id="xyz", version=1, is_snapshot=True, content="v1b")
        session.add(fv1)
        session.add(fv2)
        session.commit()
        # Should not raise


class TestComputeDiff:
    def test_identical_content(self):
        assert compute_diff("hello\n", "hello\n") == ""

    def test_single_line_change(self):
        diff = compute_diff("line1\nline2\n", "line1\nchanged\n")
        assert "-line2\n" in diff
        assert "+changed\n" in diff

    def test_addition(self):
        diff = compute_diff("a\n", "a\nb\n")
        assert "+b\n" in diff

    def test_deletion(self):
        diff = compute_diff("a\nb\n", "a\n")
        assert "-b\n" in diff

    def test_empty_to_content(self):
        diff = compute_diff("", "hello\n")
        assert "+hello\n" in diff

    def test_content_to_empty(self):
        diff = compute_diff("hello\n", "")
        assert "-hello\n" in diff

    def test_empty_to_empty(self):
        assert compute_diff("", "") == ""


class TestApplyDiffBoundsValidation:
    """C6: Validate that out-of-bounds hunks raise ValueError."""

    def test_hunk_source_start_too_large(self):
        """A hunk referencing lines beyond the file should raise."""
        base = "line1\nline2\n"
        # Craft a diff with source_start beyond file length
        bad_diff = "--- a\n+++ b\n@@ -100,1 +100,1 @@\n-old\n+new\n"
        with pytest.raises(ValueError, match="out of bounds"):
            apply_diff(base, bad_diff)

    def test_hunk_source_length_exceeds_file(self):
        """A hunk whose source_start + source_length exceeds file lines should raise."""
        base = "line1\nline2\n"
        # source_start=1, source_length=5 — valid unidiff structure but
        # the base file only has 2 lines, so end_idx=5 > 2
        bad_diff = "--- a\n+++ b\n@@ -1,5 +1,1 @@\n-line1\n-line2\n-line3\n-line4\n-line5\n+new\n"
        with pytest.raises(ValueError, match="out of bounds"):
            apply_diff(base, bad_diff)

    def test_valid_diff_still_works(self):
        """Normal diffs should not be affected by bounds checking."""
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new


class TestApplyDiff:
    def test_empty_diff_returns_base(self):
        assert apply_diff("hello\n", "") == "hello\n"

    def test_round_trip_single_change(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nmodified\nline3\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_addition(self):
        old = "a\nb\n"
        new = "a\nb\nc\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_deletion(self):
        old = "a\nb\nc\n"
        new = "a\nc\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_multiple_hunks(self):
        old = "".join(f"line{i}\n" for i in range(20))
        new_lines = [f"line{i}\n" for i in range(20)]
        new_lines[2] = "changed2\n"
        new_lines[15] = "changed15\n"
        new = "".join(new_lines)
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_no_trailing_newline(self):
        """Files without trailing newlines must round-trip exactly."""
        old = "line1\nline2"
        new = "line1\nchanged"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_add_trailing_newline(self):
        """Adding a trailing newline must be preserved."""
        old = "hello"
        new = "hello\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_remove_trailing_newline(self):
        """Removing a trailing newline must be preserved."""
        old = "hello\n"
        new = "hello"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_empty_to_content(self):
        old = ""
        new = "hello\n"
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new

    def test_round_trip_content_to_empty(self):
        old = "hello\n"
        new = ""
        diff = compute_diff(old, new)
        assert apply_diff(old, diff) == new


class TestReconstructVersion:
    def test_empty_list(self):
        assert reconstruct_version([]) == ""

    def test_snapshot_only(self):
        assert reconstruct_version([(True, "hello\n")]) == "hello\n"

    def test_snapshot_plus_diffs(self):
        v0 = "line1\nline2\nline3\n"
        v1 = "line1\nmodified\nline3\n"
        v2 = "line1\nmodified\nline3\nline4\n"

        d1 = compute_diff(v0, v1)
        d2 = compute_diff(v1, v2)

        result = reconstruct_version([(True, v0), (False, d1), (False, d2)])
        assert result == v2

    def test_first_must_be_snapshot(self):
        with pytest.raises(ValueError, match="snapshot"):
            reconstruct_version([(False, "diff")])

    def test_mid_chain_snapshot(self):
        """A snapshot mid-chain replaces everything accumulated so far."""
        v0 = "original\n"
        v1 = "modified\n"
        v2 = "fresh snapshot\n"
        v3 = "fresh snapshot\nextra line\n"

        d1 = compute_diff(v0, v1)
        d3 = compute_diff(v2, v3)

        result = reconstruct_version(
            [
                (True, v0),
                (False, d1),
                (True, v2),  # mid-chain snapshot resets
                (False, d3),
            ]
        )
        assert result == v3

    def test_multi_version_round_trip(self):
        """Write -> edit 5 times -> reconstruct each intermediate version."""
        versions = ["version 0\nshared line\n"]
        for i in range(1, 6):
            prev = versions[-1]
            new = prev + f"added in v{i}\n"
            versions.append(new)

        # Build snapshot + diffs
        entries: list[tuple[bool, str]] = [(True, versions[0])]
        for i in range(1, len(versions)):
            diff = compute_diff(versions[i - 1], versions[i])
            entries.append((False, diff))

        # Reconstruct each version by replaying from snapshot
        for i in range(len(versions)):
            result = reconstruct_version(entries[: i + 1])
            assert result == versions[i], f"Mismatch at version {i}"

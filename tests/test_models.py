"""Tests for grover.models — GroverObjectBase and GroverObject."""

from __future__ import annotations

import hashlib

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from grover.models import GroverObject, GroverObjectBase
from grover.vector import Vector

# =========================================================================
# Path normalization and validation in the model
# =========================================================================


class TestPathHandling:
    def test_normalizes_path(self):
        obj = GroverObject(path="src/auth.py")
        assert obj.path == "/src/auth.py"

    def test_resolves_double_slashes(self):
        obj = GroverObject(path="/src//auth.py")
        assert obj.path == "/src/auth.py"

    def test_resolves_dot_dot(self):
        obj = GroverObject(path="/src/../auth.py")
        assert obj.path == "/auth.py"

    def test_null_byte_rejected(self):
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/foo\x00bar")

    def test_control_char_rejected(self):
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/foo\x01bar")

    def test_del_rejected(self):
        with pytest.raises((ValueError, Exception)):
            GroverObject(path="/foo\x7fbar")

    def test_empty_path_is_root(self):
        obj = GroverObject(path="")
        assert obj.path == "/"


# =========================================================================
# Kind inference
# =========================================================================


class TestKindInference:
    def test_file_from_extension(self):
        obj = GroverObject(path="/src/auth.py")
        assert obj.kind == "file"

    def test_directory(self):
        obj = GroverObject(path="/src")
        assert obj.kind == "directory"

    def test_chunk(self):
        obj = GroverObject(path="/src/auth.py/.chunks/login")
        assert obj.kind == "chunk"

    def test_version(self):
        obj = GroverObject(path="/src/auth.py/.versions/3")
        assert obj.kind == "version"

    def test_connection(self):
        obj = GroverObject(path="/a.py/.connections/imports/b.py")
        assert obj.kind == "connection"

    def test_api(self):
        obj = GroverObject(path="/jira/.apis/ticket")
        assert obj.kind == "api"

    def test_dotfile(self):
        obj = GroverObject(path="/.bashrc")
        assert obj.kind == "file"

    def test_extensionless_file(self):
        obj = GroverObject(path="/Makefile")
        assert obj.kind == "file"

    def test_explicit_kind_preserved(self):
        obj = GroverObject(path="/myapp", kind="file")
        assert obj.kind == "file"


# =========================================================================
# Parent path derivation
# =========================================================================


class TestName:
    def test_file(self):
        obj = GroverObject(path="/src/auth.py")
        assert obj.name == "auth.py"

    def test_directory(self):
        obj = GroverObject(path="/src")
        assert obj.name == "src"

    def test_root(self):
        obj = GroverObject(path="/")
        assert obj.name == ""

    def test_chunk(self):
        obj = GroverObject(path="/src/auth.py/.chunks/login")
        assert obj.name == "login"

    def test_version(self):
        obj = GroverObject(path="/src/auth.py/.versions/3")
        assert obj.name == "3"

    def test_connection(self):
        obj = GroverObject(path="/a.py/.connections/imports/src/utils.py")
        assert obj.name == "utils.py"

    def test_api(self):
        obj = GroverObject(path="/jira/.apis/ticket")
        assert obj.name == "ticket"

    def test_explicit_name_preserved(self):
        obj = GroverObject(path="/src/auth.py", name="custom")
        assert obj.name == "custom"


class TestParentPath:
    def test_file(self):
        obj = GroverObject(path="/src/auth.py")
        assert obj.parent_path == "/src"

    def test_root_child(self):
        obj = GroverObject(path="/src")
        assert obj.parent_path == "/"

    def test_root(self):
        obj = GroverObject(path="/")
        assert obj.parent_path == "/"

    def test_chunk_parent_is_owning_file(self):
        obj = GroverObject(path="/src/auth.py/.chunks/login")
        assert obj.parent_path == "/src/auth.py"

    def test_connection_parent_is_owning_file(self):
        obj = GroverObject(path="/a.py/.connections/imports/b.py")
        assert obj.parent_path == "/a.py"

    def test_explicit_parent_preserved(self):
        obj = GroverObject(path="/src/auth.py", parent_path="/custom")
        assert obj.parent_path == "/custom"


# =========================================================================
# Connection decomposition
# =========================================================================


class TestConnectionDecomposition:
    def test_fields_extracted(self):
        obj = GroverObject(path="/src/auth.py/.connections/imports/src/utils.py")
        assert obj.source_path == "/src/auth.py"
        assert obj.target_path == "/src/utils.py"
        assert obj.connection_type == "imports"

    def test_deep_target(self):
        obj = GroverObject(path="/a.py/.connections/calls/deep/nested/path.py")
        assert obj.target_path == "/deep/nested/path.py"

    def test_explicit_connection_fields_preserved(self):
        obj = GroverObject(
            path="/a.py/.connections/imports/b.py",
            source_path="/custom",
            target_path="/custom-target",
            connection_type="custom-type",
        )
        assert obj.source_path == "/custom"
        assert obj.target_path == "/custom-target"
        assert obj.connection_type == "custom-type"

    def test_non_connection_path_no_decomposition(self):
        obj = GroverObject(path="/src/auth.py")
        assert obj.source_path is None
        assert obj.target_path is None
        assert obj.connection_type is None


# =========================================================================
# Content metrics
# =========================================================================


class TestContentMetrics:
    def test_content_with_text(self):
        obj = GroverObject(path="/a.py", content="def login():\n    pass")
        assert obj.content_hash is not None
        assert obj.size_bytes == len(b"def login():\n    pass")
        assert obj.lines == 2
        assert obj.tokens == 0
        assert obj.lexical_tokens == GroverObjectBase._lexical_token_count(
            "def login():\n    pass",
        )

    def test_single_line_no_newline(self):
        obj = GroverObject(path="/a.py", content="hello")
        assert obj.lines == 1

    def test_trailing_newline(self):
        obj = GroverObject(path="/a.py", content="a\nb\n")
        assert obj.lines == 3

    def test_empty_string_content(self):
        obj = GroverObject(path="/empty.txt", content="")
        assert obj.content_hash == hashlib.sha256(b"").hexdigest()
        assert obj.size_bytes == 0
        assert obj.lines == 0

    def test_none_content(self):
        obj = GroverObject(path="/dir")
        assert obj.content is None
        assert obj.content_hash is None
        assert obj.size_bytes == 0
        assert obj.lines == 0

    def test_hash_recomputed_even_if_explicit(self):
        obj = GroverObject(path="/a.py", content="hello", content_hash="bogus")
        assert obj.content_hash == hashlib.sha256(b"hello").hexdigest()

    def test_version_row_factory_uses_reconstructed_metadata(self):
        row = GroverObjectBase.create_version_row(
            file_path="/a.py",
            version_number=2,
            version_content="new",
            prev_content="old",
            created_by="auto",
        )
        assert row.kind == "version"
        assert row.is_snapshot is False
        assert row.content is None
        assert row.version_diff is not None
        assert row.content_hash == hashlib.sha256(b"new").hexdigest()
        assert row.size_bytes == len(b"new")
        assert row.lines == 1
        assert row.tokens == 0
        assert row.lexical_tokens == GroverObjectBase._lexical_token_count("new")

    def test_version_row_preserves_explicit_metadata(self):
        obj = GroverObject(
            path="/a.py/.versions/2",
            kind="version",
            content=None,
            version_diff="diff payload",
            content_hash="abc",
            size_bytes=123,
            lines=7,
            version_number=2,
            is_snapshot=False,
        )
        assert obj.content is None
        assert obj.version_diff == "diff payload"
        assert obj.content_hash == "abc"
        assert obj.size_bytes == 123
        assert obj.lines == 7


# =========================================================================
# Timestamps
# =========================================================================


class TestTimestamps:
    def test_auto_set(self):
        obj = GroverObject(path="/a.py")
        assert obj.created_at is not None
        assert obj.updated_at is not None

    def test_created_at_equals_updated_at(self):
        obj = GroverObject(path="/a.py")
        # Both set from the same `now` in the validator
        assert obj.created_at == obj.updated_at

    def test_explicit_created_at_preserved(self):
        from datetime import UTC, datetime

        ts = datetime(2020, 1, 1, tzinfo=UTC)
        obj = GroverObject(path="/a.py", created_at=ts)
        assert obj.created_at == ts

    def test_deleted_at_defaults_to_none(self):
        obj = GroverObject(path="/a.py")
        assert obj.deleted_at is None


# =========================================================================
# ID generation
# =========================================================================


class TestId:
    def test_auto_generated(self):
        obj = GroverObject(path="/a.py")
        assert obj.id is not None
        assert len(obj.id) == 36  # UUID format

    def test_unique_across_instances(self):
        a = GroverObject(path="/a.py")
        b = GroverObject(path="/b.py")
        assert a.id != b.id

    def test_explicit_id_preserved(self):
        obj = GroverObject(path="/a.py", id="custom-id")
        assert obj.id == "custom-id"


# =========================================================================
# Embedding
# =========================================================================


class TestEmbedding:
    def test_defaults_to_none(self):
        obj = GroverObject(path="/a.py")
        assert obj.embedding is None

    def test_accepts_list(self):
        obj = GroverObject(path="/a.py", embedding=Vector([0.1, 0.2, 0.3]))
        assert isinstance(obj.embedding, Vector)
        assert list(obj.embedding) == [0.1, 0.2, 0.3]

    def test_accepts_vector(self):
        vec = Vector([1.0, 2.0])
        obj = GroverObject(path="/a.py", embedding=vec)
        assert obj.embedding is vec


# =========================================================================
# Base vs concrete class
# =========================================================================


class TestBaseVsConcrete:
    def test_base_is_not_a_table(self):
        assert not GroverObjectBase.model_config.get("table", False)

    def test_concrete_is_a_table(self):
        assert GroverObject.__tablename__ == "grover_objects"

    def test_concrete_inherits_validation(self):
        obj = GroverObject(path="/src/auth.py", content="x")
        assert obj.kind == "file"
        assert obj.content_hash is not None


class TestContentMutation:
    def test_update_content_recomputes_lexical_tokens(self):
        obj = GroverObject(path="/a.py", content="hello")
        original_lexical_tokens = obj.lexical_tokens
        obj.update_content("hello world from grover")
        assert obj.lexical_tokens == GroverObjectBase._lexical_token_count(
            "hello world from grover",
        )
        assert obj.lexical_tokens != original_lexical_tokens


# =========================================================================
# add_prefix / strip_prefix
# =========================================================================


class TestAddStripPrefix:
    def test_strip_prefix_rederives_fields(self):
        obj = GroverObject(path="/data/sub/file.py", content="code")
        obj.strip_prefix("/data/sub")
        assert obj.path == "/file.py"
        assert obj.name == "file.py"
        assert obj.parent_path == "/"

    def test_add_prefix_rederives_fields(self):
        obj = GroverObject(path="/file.py", content="code")
        obj.add_prefix("/data/sub")
        assert obj.path == "/data/sub/file.py"
        assert obj.name == "file.py"
        assert obj.parent_path == "/data/sub"

    def test_roundtrip(self):
        obj = GroverObject(path="/mount/deep/file.py", content="x")
        obj.strip_prefix("/mount")
        assert obj.path == "/deep/file.py"
        obj.add_prefix("/mount")
        assert obj.path == "/mount/deep/file.py"

    def test_empty_prefix_is_noop(self):
        obj = GroverObject(path="/file.py", content="x")
        obj.add_prefix("")
        assert obj.path == "/file.py"
        obj.strip_prefix("")
        assert obj.path == "/file.py"

    def test_strip_to_root(self):
        obj = GroverObject(path="/data", content="x")
        obj.strip_prefix("/data")
        assert obj.path == "/"

    def test_returns_self(self):
        obj = GroverObject(path="/a.py", content="x")
        assert obj.add_prefix("/m") is obj
        assert obj.strip_prefix("/m") is obj

    def test_chunk_kind_preserved(self):
        obj = GroverObject(path="/src/mod.py/.chunks/fn", content="def fn(): pass")
        obj.strip_prefix("/src")
        assert obj.path == "/mod.py/.chunks/fn"
        assert obj.kind == "chunk"
        assert obj.parent_path == "/mod.py"


# DB round-trip
# =========================================================================


class TestDBRoundTrip:
    @pytest.fixture()
    def engine(self):
        e = create_engine("sqlite://")
        SQLModel.metadata.create_all(e)
        return e

    def test_insert_and_load(self, engine):
        obj = GroverObject(path="/src/auth.py", content="def login(): pass")
        expected_hash = obj.content_hash
        with Session(engine) as s:
            s.add(obj)
            s.commit()

        with Session(engine) as s:
            loaded = s.exec(select(GroverObject).where(GroverObject.path == "/src/auth.py")).one()
            assert loaded.path == "/src/auth.py"
            assert loaded.kind == "file"
            assert loaded.parent_path == "/src"
            assert loaded.content == "def login(): pass"
            assert loaded.content_hash == expected_hash

    def test_embedding_round_trip(self, engine):
        obj = GroverObject(path="/a.py", embedding=Vector([0.1, 0.2, 0.3]))
        with Session(engine) as s:
            s.add(obj)
            s.commit()

        with Session(engine) as s:
            loaded = s.exec(select(GroverObject).where(GroverObject.path == "/a.py")).one()
            assert isinstance(loaded.embedding, Vector)
            assert list(loaded.embedding) == [0.1, 0.2, 0.3]

    def test_connection_round_trip(self, engine):
        obj = GroverObject(path="/a.py/.connections/imports/b.py")
        with Session(engine) as s:
            s.add(obj)
            s.commit()

        with Session(engine) as s:
            loaded = s.exec(select(GroverObject).where(GroverObject.path == "/a.py/.connections/imports/b.py")).one()
            assert loaded.kind == "connection"
            assert loaded.source_path == "/a.py"
            assert loaded.target_path == "/b.py"
            assert loaded.connection_type == "imports"


# =========================================================================
# plan_file_write
# =========================================================================


class TestPlanFileWrite:
    """Unit tests for GroverObjectBase.plan_file_write fast/slow path."""

    @staticmethod
    def _make_file(content: str, version_number: int) -> GroverObject:
        """Build a file object with consistent hash/version state."""
        obj = GroverObject(path="/test.txt", content=content)
        obj.version_number = version_number
        return obj

    def test_fast_path_skips_reconstruction(self):
        """When latest_version_hash matches file hash, chain_verified=True."""
        obj = self._make_file("hello", version_number=1)
        plan = obj.plan_file_write(
            "world",
            latest_version_hash=obj.content_hash,
        )
        assert plan.chain_verified is True
        assert plan.final_content == "world"
        assert plan.final_version_number == 2
        assert len(plan.version_rows) == 1
        assert plan.version_rows[0].created_by == "auto"

    def test_no_latest_hash_signals_unverified(self):
        """No latest_version_hash and no version_rows → chain_verified=False."""
        obj = self._make_file("hello", version_number=1)
        plan = obj.plan_file_write("world", latest_version_hash=None)
        assert plan.chain_verified is False

    def test_mismatched_latest_hash_signals_unverified(self):
        """Wrong latest_version_hash → chain_verified=False."""
        obj = self._make_file("hello", version_number=1)
        plan = obj.plan_file_write("world", latest_version_hash="wrong_hash")
        assert plan.chain_verified is False

    def test_same_content_returns_noop_plan(self):
        """Same content → no new version rows."""
        obj = self._make_file("hello", version_number=1)
        plan = obj.plan_file_write(
            "hello",
            latest_version_hash=obj.content_hash,
        )
        assert plan.chain_verified is True
        assert len(plan.version_rows) == 0
        assert plan.final_version_number == 1

    def test_version_number_zero_creates_initial_snapshot(self):
        """First write creates v1 snapshot regardless of hash args."""
        obj = self._make_file("hello", version_number=0)
        plan = obj.plan_file_write("hello")
        assert plan.final_version_number == 1
        assert len(plan.version_rows) == 1
        assert plan.version_rows[0].is_snapshot is True

    def test_external_edit_detected_by_hash_mismatch(self):
        """File hash ≠ stored hash → external snapshot, even with latest_version_hash."""
        obj = self._make_file("original", version_number=1)
        # Simulate external edit: change content without updating hash
        obj.content = "tampered"
        plan = obj.plan_file_write(
            "new_content",
            latest_version_hash=obj.content_hash,  # matches stored, not observed
        )
        assert plan.chain_verified is True  # external path handles it directly
        assert plan.final_version_number == 3  # external=2, new=3
        external_row = plan.version_rows[0]
        assert external_row.created_by == "external"
        assert external_row.is_snapshot is True


# ==================================================================
# Edge case coverage — _stored_version_payload
# ==================================================================


class TestStoredVersionPayload:
    def test_non_version_raises(self):
        obj = GroverObject(path="/a.py", kind="file", content="x")
        with pytest.raises(ValueError, match="non-version object"):
            obj._stored_version_payload()

    def test_missing_payload_raises(self):
        obj = GroverObject(
            path="/a.py@1", kind="version", content=None, version_diff=None, is_snapshot=False,
        )
        with pytest.raises(ValueError, match="missing stored payload"):
            obj._stored_version_payload()


# ==================================================================
# Edge case coverage — _reconstruct_file_version
# ==================================================================


class TestReconstructFileVersion:
    def test_missing_target_version_raises(self):
        with pytest.raises(ValueError, match="Missing version row for v2"):
            GroverObject._reconstruct_file_version([], target_version=2)

    def test_missing_snapshot_raises(self):
        row = GroverObject(
            path="/a.py@1", kind="version", content=None,
            version_diff="some diff", version_number=1, is_snapshot=False,
        )
        with pytest.raises(ValueError, match="Missing snapshot"):
            GroverObject._reconstruct_file_version([row], target_version=1)

    def test_missing_intermediate_version_raises(self):
        snapshot = GroverObject(
            path="/a.py@1", kind="version", content="base",
            version_number=1, is_snapshot=True,
        )
        # v2 is missing, asking for v3
        v3 = GroverObject(
            path="/a.py@3", kind="version", content=None,
            version_diff="diff", version_number=3, is_snapshot=False,
        )
        with pytest.raises(ValueError, match="Missing version row for v2"):
            GroverObject._reconstruct_file_version([snapshot, v3], target_version=3)

    def test_hash_mismatch_raises(self):
        snapshot = GroverObject(
            path="/a.py@1", kind="version", content="base content",
            version_number=1, is_snapshot=True, content_hash="wrong_hash",
        )
        with pytest.raises(ValueError, match="Hash mismatch"):
            GroverObject._reconstruct_file_version([snapshot], target_version=1)


# ==================================================================
# Edge case coverage — plan_file_write and update_content
# ==================================================================


class TestPlanFileWriteEdgeCases:
    def test_directory_raises(self):
        obj = GroverObject(path="/mydir", kind="directory")
        with pytest.raises(ValueError, match="Version planning only applies to files"):
            obj.plan_file_write("content")


class TestUpdateContentEdgeCases:
    def test_directory_raises(self):
        obj = GroverObject(path="/mydir", kind="directory")
        with pytest.raises(ValueError, match="Cannot set content on a directory"):
            obj.update_content("x")


# ==================================================================
# Edge case coverage — model validator
# ==================================================================


class TestValidatorEdgeCases:
    def test_null_bytes_in_version_diff_rejected(self):
        with pytest.raises(ValueError, match="version_diff contains null bytes"):
            GroverObject(
                path="/a.py@1", kind="version", version_diff="has\x00null",
                version_number=1, is_snapshot=False,
            )

    def test_both_content_and_version_diff_rejected(self):
        with pytest.raises(ValueError, match="must not set both content and version_diff"):
            GroverObject(
                path="/a.py@1", kind="version", content="text",
                version_diff="diff", version_number=1, is_snapshot=False,
            )

    def test_version_with_explicit_content_hash(self):
        h = hashlib.sha256(b"hello").hexdigest()
        obj = GroverObject(
            path="/a.py@1", kind="version", content="hello",
            version_number=1, is_snapshot=True, content_hash=h,
        )
        assert obj.content_hash == h

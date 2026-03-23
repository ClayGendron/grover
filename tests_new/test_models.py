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
        assert obj.size_bytes == len("def login():\n    pass".encode())
        assert obj.lines == 2

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
        obj = GroverObject(path="/a.py", embedding=[0.1, 0.2, 0.3])
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


# =========================================================================
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
            loaded = s.exec(
                select(GroverObject).where(GroverObject.path == "/src/auth.py")
            ).one()
            assert loaded.path == "/src/auth.py"
            assert loaded.kind == "file"
            assert loaded.parent_path == "/src"
            assert loaded.content == "def login(): pass"
            assert loaded.content_hash == expected_hash

    def test_embedding_round_trip(self, engine):
        obj = GroverObject(path="/a.py", embedding=[0.1, 0.2, 0.3])
        with Session(engine) as s:
            s.add(obj)
            s.commit()

        with Session(engine) as s:
            loaded = s.exec(
                select(GroverObject).where(GroverObject.path == "/a.py")
            ).one()
            assert isinstance(loaded.embedding, Vector)
            assert list(loaded.embedding) == [0.1, 0.2, 0.3]

    def test_connection_round_trip(self, engine):
        obj = GroverObject(path="/a.py/.connections/imports/b.py")
        with Session(engine) as s:
            s.add(obj)
            s.commit()

        with Session(engine) as s:
            loaded = s.exec(
                select(GroverObject).where(
                    GroverObject.path == "/a.py/.connections/imports/b.py"
                )
            ).one()
            assert loaded.kind == "connection"
            assert loaded.source_path == "/a.py"
            assert loaded.target_path == "/b.py"
            assert loaded.connection_type == "imports"

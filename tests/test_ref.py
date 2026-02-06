"""Tests for Ref dataclass and path normalization."""

from __future__ import annotations

import pytest

from grover.fs.utils import normalize_path, split_path
from grover.ref import Ref, file_ref


class TestRefCreation:
    def test_minimal_ref(self):
        r = Ref(path="/foo.txt")
        assert r.path == "/foo.txt"
        assert r.version is None
        assert r.line_start is None
        assert r.line_end is None
        assert r.metadata == {}

    def test_full_ref(self):
        r = Ref(
            path="/src/main.py",
            version=3,
            line_start=10,
            line_end=20,
            metadata={"lang": "python"},
        )
        assert r.path == "/src/main.py"
        assert r.version == 3
        assert r.line_start == 10
        assert r.line_end == 20
        assert r.metadata == {"lang": "python"}

    def test_string_version(self):
        r = Ref(path="/f.txt", version="abc123")
        assert r.version == "abc123"

    def test_repr_minimal(self):
        r = Ref(path="/foo.txt")
        assert repr(r) == "Ref(path='/foo.txt')"

    def test_repr_full(self):
        r = Ref(path="/f.txt", version=1, line_start=5, line_end=10, metadata={"k": "v"})
        assert "version=1" in repr(r)
        assert "line_start=5" in repr(r)
        assert "line_end=10" in repr(r)
        assert "metadata={'k': 'v'}" in repr(r)


class TestRefImmutability:
    def test_cannot_set_path(self):
        r = Ref(path="/foo.txt")
        with pytest.raises(AttributeError):
            r.path = "/bar.txt"  # type: ignore[misc]

    def test_cannot_set_version(self):
        r = Ref(path="/foo.txt", version=1)
        with pytest.raises(AttributeError):
            r.version = 2  # type: ignore[misc]

    def test_cannot_set_line_start(self):
        r = Ref(path="/foo.txt")
        with pytest.raises(AttributeError):
            r.line_start = 5  # type: ignore[misc]

    def test_metadata_dict_is_mutable(self):
        """Metadata dict itself can be mutated (frozen only protects attribute rebinding)."""
        r = Ref(path="/foo.txt", metadata={"a": 1})
        r.metadata["b"] = 2
        assert r.metadata == {"a": 1, "b": 2}


class TestRefEquality:
    def test_equal_refs(self):
        a = Ref(path="/foo.txt", version=1)
        b = Ref(path="/foo.txt", version=1)
        assert a == b

    def test_different_path(self):
        a = Ref(path="/foo.txt")
        b = Ref(path="/bar.txt")
        assert a != b

    def test_different_version(self):
        a = Ref(path="/foo.txt", version=1)
        b = Ref(path="/foo.txt", version=2)
        assert a != b

    def test_metadata_excluded_from_equality(self):
        a = Ref(path="/foo.txt", metadata={"x": 1})
        b = Ref(path="/foo.txt", metadata={"y": 2})
        assert a == b

    def test_metadata_excluded_from_hash(self):
        a = Ref(path="/foo.txt", metadata={"x": 1})
        b = Ref(path="/foo.txt", metadata={"y": 2})
        assert hash(a) == hash(b)

    def test_hashable_in_set(self):
        refs = {Ref(path="/a"), Ref(path="/a"), Ref(path="/b")}
        assert len(refs) == 2

    def test_hashable_as_dict_key(self):
        d = {Ref(path="/a"): 1}
        assert d[Ref(path="/a")] == 1


class TestFileRef:
    def test_normalizes_path(self):
        r = file_ref("foo.txt")
        assert r.path == "/foo.txt"

    def test_normalizes_double_slashes(self):
        r = file_ref("/foo//bar.txt")
        assert r.path == "/foo/bar.txt"

    def test_resolves_dotdot(self):
        r = file_ref("/foo/../bar.txt")
        assert r.path == "/bar.txt"

    def test_with_version(self):
        r = file_ref("/main.py", version=5)
        assert r.path == "/main.py"
        assert r.version == 5

    def test_returns_ref_instance(self):
        r = file_ref("/x.py")
        assert isinstance(r, Ref)


class TestNormalizePath:
    def test_empty_string(self):
        assert normalize_path("") == "/"

    def test_bare_filename(self):
        assert normalize_path("foo.txt") == "/foo.txt"

    def test_leading_slash(self):
        assert normalize_path("/foo.txt") == "/foo.txt"

    def test_double_slashes(self):
        assert normalize_path("/foo//bar.txt") == "/foo/bar.txt"

    def test_dotdot(self):
        assert normalize_path("/foo/../bar.txt") == "/bar.txt"

    def test_dot(self):
        assert normalize_path("/foo/./bar.txt") == "/foo/bar.txt"

    def test_trailing_slash(self):
        assert normalize_path("/foo/") == "/foo"

    def test_root(self):
        assert normalize_path("/") == "/"

    def test_whitespace_stripped(self):
        assert normalize_path("  /foo.txt  ") == "/foo.txt"


class TestSplitPath:
    def test_nested_file(self):
        assert split_path("/foo/bar.txt") == ("/foo", "bar.txt")

    def test_root_file(self):
        assert split_path("/foo.txt") == ("/", "foo.txt")

    def test_root(self):
        assert split_path("/") == ("/", "")

    def test_normalizes_before_split(self):
        assert split_path("foo//bar.txt") == ("/foo", "bar.txt")

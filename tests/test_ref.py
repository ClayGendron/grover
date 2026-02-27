"""Tests for Ref identity type and path normalization."""

from __future__ import annotations

import pytest

from grover.fs.utils import normalize_path, split_path
from grover.ref import Ref

# ==================================================================
# Construction
# ==================================================================


class TestRefConstruction:
    def test_path_only(self):
        r = Ref(path="/foo.txt")
        assert r.path == "/foo.txt"

    def test_keyword_arg(self):
        r = Ref(path="/src/main.py")
        assert r.path == "/src/main.py"

    def test_repr(self):
        assert repr(Ref(path="/foo.txt")) == "Ref('/foo.txt')"

    def test_repr_chunk(self):
        assert repr(Ref(path="/a.py#login")) == "Ref('/a.py#login')"

    def test_repr_connection(self):
        assert repr(Ref(path="/a.py[imports]/b.py")) == "Ref('/a.py[imports]/b.py')"


# ==================================================================
# Immutability
# ==================================================================


class TestRefImmutability:
    def test_cannot_set_path(self):
        r = Ref(path="/foo.txt")
        with pytest.raises(AttributeError):
            r.path = "/bar.txt"  # type: ignore[misc]

    def test_no_extra_attributes(self):
        r = Ref(path="/foo.txt")
        with pytest.raises((AttributeError, TypeError)):
            r.extra = "nope"  # type: ignore[attr-defined]


# ==================================================================
# Equality and hashing
# ==================================================================


class TestRefEquality:
    def test_equal_same_path(self):
        assert Ref(path="/foo.txt") == Ref(path="/foo.txt")

    def test_not_equal_different_path(self):
        assert Ref(path="/foo.txt") != Ref(path="/bar.txt")

    def test_hashable_in_set(self):
        refs = {Ref(path="/a"), Ref(path="/a"), Ref(path="/b")}
        assert len(refs) == 2

    def test_hashable_as_dict_key(self):
        d = {Ref(path="/a"): 1}
        assert d[Ref(path="/a")] == 1


# ==================================================================
# Type checks
# ==================================================================


class TestIsFile:
    def test_plain_path(self):
        assert Ref(path="/src/auth.py").is_file is True

    def test_chunk_not_file(self):
        assert Ref(path="/src/auth.py#login").is_file is False

    def test_version_not_file(self):
        assert Ref(path="/src/auth.py@3").is_file is False

    def test_connection_not_file(self):
        assert Ref(path="/a.py[imports]/b.py").is_file is False

    def test_root_path(self):
        assert Ref(path="/").is_file is True


class TestIsChunk:
    def test_simple_chunk(self):
        assert Ref(path="/a.py#foo").is_chunk is True

    def test_scoped_chunk(self):
        assert Ref(path="/a.py#Client.connect").is_chunk is True

    def test_plain_not_chunk(self):
        assert Ref(path="/a.py").is_chunk is False

    def test_version_not_chunk(self):
        assert Ref(path="/a.py@3").is_chunk is False

    def test_trailing_hash_not_chunk(self):
        assert Ref(path="/a.py#").is_chunk is False

    def test_leading_hash_not_chunk(self):
        assert Ref(path="#foo").is_chunk is False

    def test_hash_in_dir_only_not_chunk(self):
        assert Ref(path="/dir#v1/file.py").is_chunk is False

    def test_hash_in_dir_and_suffix_is_chunk(self):
        assert Ref(path="/dir#name/file.py#symbol").is_chunk is True


class TestIsVersion:
    def test_simple_version(self):
        assert Ref(path="/a.py@3").is_version is True

    def test_version_zero(self):
        assert Ref(path="/a.py@0").is_version is True

    def test_large_version(self):
        assert Ref(path="/a.py@999").is_version is True

    def test_plain_not_version(self):
        assert Ref(path="/a.py").is_version is False

    def test_chunk_not_version(self):
        assert Ref(path="/a.py#foo").is_version is False

    def test_non_numeric_not_version(self):
        assert Ref(path="/a.py@abc").is_version is False

    def test_leading_at_not_version(self):
        assert Ref(path="@3").is_version is False

    def test_at_in_dir_and_suffix_is_version(self):
        assert Ref(path="/dir@v2/file.py@3").is_version is True


class TestIsConnection:
    def test_simple_connection(self):
        assert Ref(path="/a.py[imports]/b.py").is_connection is True

    def test_plain_not_connection(self):
        assert Ref(path="/a.py").is_connection is False

    def test_empty_type_not_connection(self):
        assert Ref(path="/a.py[]/b.py").is_connection is False

    def test_type_with_slash_not_connection(self):
        assert Ref(path="/a.py[im/ports]/b.py").is_connection is False

    def test_leading_bracket_not_connection(self):
        assert Ref(path="[type]/b.py").is_connection is False

    def test_various_types(self):
        for ct in ("imports", "contains", "calls", "depends_on"):
            assert Ref(path=f"/a.py[{ct}]/b.py").is_connection is True


# ==================================================================
# Decomposition — base_path
# ==================================================================


class TestBasePath:
    def test_plain_file(self):
        assert Ref(path="/a.py").base_path == "/a.py"

    def test_chunk(self):
        assert Ref(path="/a.py#foo").base_path == "/a.py"

    def test_version(self):
        assert Ref(path="/a.py@3").base_path == "/a.py"

    def test_connection_returns_source(self):
        assert Ref(path="/a.py[imports]/b.py").base_path == "/a.py"

    def test_hash_in_dir_chunk(self):
        assert Ref(path="/dir#name/file.py#symbol").base_path == "/dir#name/file.py"

    def test_at_in_dir_version(self):
        assert Ref(path="/dir@v2/file.py@3").base_path == "/dir@v2/file.py"


# ==================================================================
# Decomposition — chunk
# ==================================================================


class TestChunkProperty:
    def test_simple_symbol(self):
        assert Ref(path="/a.py#foo").chunk == "foo"

    def test_scoped_symbol(self):
        assert Ref(path="/a.py#Client.connect").chunk == "Client.connect"

    def test_dunder(self):
        assert Ref(path="/a.py#MyClass.__init__").chunk == "MyClass.__init__"

    def test_plain_is_none(self):
        assert Ref(path="/a.py").chunk is None

    def test_version_is_none(self):
        assert Ref(path="/a.py@3").chunk is None

    def test_connection_is_none(self):
        assert Ref(path="/a.py[imports]/b.py").chunk is None


# ==================================================================
# Decomposition — version
# ==================================================================


class TestVersionProperty:
    def test_simple_version(self):
        assert Ref(path="/a.py@3").version == 3

    def test_version_zero(self):
        assert Ref(path="/a.py@0").version == 0

    def test_large_version(self):
        assert Ref(path="/a.py@999").version == 999

    def test_plain_is_none(self):
        assert Ref(path="/a.py").version is None

    def test_chunk_is_none(self):
        assert Ref(path="/a.py#foo").version is None

    def test_connection_is_none(self):
        assert Ref(path="/a.py[imports]/b.py").version is None


# ==================================================================
# Decomposition — connection
# ==================================================================


class TestConnectionProperties:
    def test_source(self):
        assert Ref(path="/a.py[imports]/b.py").source == "/a.py"

    def test_target(self):
        assert Ref(path="/a.py[imports]/b.py").target == "/b.py"

    def test_connection_type(self):
        assert Ref(path="/a.py[imports]/b.py").connection_type == "imports"

    def test_plain_source_is_none(self):
        assert Ref(path="/a.py").source is None

    def test_plain_target_is_none(self):
        assert Ref(path="/a.py").target is None

    def test_plain_connection_type_is_none(self):
        assert Ref(path="/a.py").connection_type is None

    def test_nested_paths(self):
        r = Ref(path="/src/a.py[calls]/lib/b.py")
        assert r.source == "/src/a.py"
        assert r.target == "/lib/b.py"
        assert r.connection_type == "calls"


# ==================================================================
# Factories
# ==================================================================


class TestFactories:
    def test_for_chunk_simple(self):
        assert Ref.for_chunk("/a.py", "foo") == Ref(path="/a.py#foo")

    def test_for_chunk_scoped(self):
        assert Ref.for_chunk("/a.py", "Client.connect") == Ref(path="/a.py#Client.connect")

    def test_for_version_simple(self):
        assert Ref.for_version("/a.py", 3) == Ref(path="/a.py@3")

    def test_for_version_zero(self):
        assert Ref.for_version("/a.py", 0) == Ref(path="/a.py@0")

    def test_for_connection_simple(self):
        r = Ref.for_connection("/a.py", "/b.py", "imports")
        assert r == Ref(path="/a.py[imports]/b.py")

    def test_for_connection_path(self):
        r = Ref.for_connection("/src/a.py", "/lib/b.py", "calls")
        assert r.path == "/src/a.py[calls]/lib/b.py"


# ==================================================================
# Round-trips
# ==================================================================


class TestRoundTrips:
    def test_chunk_round_trip(self):
        r = Ref.for_chunk("/src/auth.py", "login")
        assert r.base_path == "/src/auth.py"
        assert r.chunk == "login"

    def test_version_round_trip(self):
        r = Ref.for_version("/src/auth.py", 5)
        assert r.base_path == "/src/auth.py"
        assert r.version == 5

    def test_connection_round_trip(self):
        r = Ref.for_connection("/a.py", "/b.py", "imports")
        assert r.source == "/a.py"
        assert r.target == "/b.py"
        assert r.connection_type == "imports"

    def test_for_chunk_is_chunk(self):
        assert Ref.for_chunk("/a.py", "foo").is_chunk is True

    def test_for_version_is_version(self):
        assert Ref.for_version("/a.py", 3).is_version is True

    def test_for_connection_is_connection(self):
        assert Ref.for_connection("/a.py", "/b.py", "imports").is_connection is True

    def test_for_chunk_not_version(self):
        assert Ref.for_chunk("/a.py", "foo").is_version is False

    def test_for_version_not_chunk(self):
        assert Ref.for_version("/a.py", 3).is_chunk is False


# ==================================================================
# Path normalization (from fs.utils — unchanged)
# ==================================================================


class TestNormalizePath:
    @pytest.mark.parametrize(
        ("input_path", "expected"),
        [
            pytest.param("", "/", id="empty-string"),
            pytest.param("foo.txt", "/foo.txt", id="bare-filename"),
            pytest.param("/foo.txt", "/foo.txt", id="leading-slash"),
            pytest.param("/foo//bar.txt", "/foo/bar.txt", id="double-slashes"),
            pytest.param("/foo/../bar.txt", "/bar.txt", id="dotdot"),
            pytest.param("/foo/./bar.txt", "/foo/bar.txt", id="dot"),
            pytest.param("/foo/", "/foo", id="trailing-slash"),
            pytest.param("/", "/", id="root"),
            pytest.param("  /foo.txt  ", "/foo.txt", id="whitespace-stripped"),
            pytest.param("/a/../../b", "/b", id="dotdot-beyond-root"),
            pytest.param("/../../etc/passwd", "/etc/passwd", id="deeply-nested-dotdot"),
            pytest.param("   ", "/", id="whitespace-only"),
        ],
    )
    def test_normalize(self, input_path: str, expected: str):
        assert normalize_path(input_path) == expected


class TestSplitPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            pytest.param("/foo/bar.txt", ("/foo", "bar.txt"), id="nested-file"),
            pytest.param("/foo.txt", ("/", "foo.txt"), id="root-file"),
            pytest.param("/", ("/", ""), id="root"),
            pytest.param("foo//bar.txt", ("/foo", "bar.txt"), id="normalizes-first"),
        ],
    )
    def test_split(self, path: str, expected: tuple[str, str]):
        assert split_path(path) == expected

"""Tests for path format utilities via Ref."""

from __future__ import annotations

from grover.ref import Ref

# ==================================================================
# Ref.for_chunk (replaces build_chunk_ref)
# ==================================================================


class TestForChunk:
    def test_simple_symbol(self):
        assert Ref.for_chunk("/src/auth.py", "login").path == "/src/auth.py#login"

    def test_scoped_symbol(self):
        assert Ref.for_chunk("/src/auth.py", "Client.connect").path == "/src/auth.py#Client.connect"

    def test_root_file(self):
        assert Ref.for_chunk("/main.py", "run").path == "/main.py#run"

    def test_deeply_nested(self):
        assert Ref.for_chunk("/a/b/c/d.py", "foo").path == "/a/b/c/d.py#foo"

    def test_dunder_method(self):
        r = Ref.for_chunk("/src/cls.py", "MyClass.__init__")
        assert r.path == "/src/cls.py#MyClass.__init__"

    def test_dotted_filename(self):
        assert (
            Ref.for_chunk("/src/auth.test.py", "test_login").path == "/src/auth.test.py#test_login"
        )


# ==================================================================
# Ref.for_version (replaces build_version_ref)
# ==================================================================


class TestForVersion:
    def test_simple(self):
        assert Ref.for_version("/src/auth.py", 3).path == "/src/auth.py@3"

    def test_version_zero(self):
        assert Ref.for_version("/src/auth.py", 0).path == "/src/auth.py@0"

    def test_large_version(self):
        assert Ref.for_version("/file.txt", 999).path == "/file.txt@999"


# ==================================================================
# Ref.for_connection (new)
# ==================================================================


class TestForConnection:
    def test_simple(self):
        assert Ref.for_connection("/a.py", "/b.py", "imports").path == "/a.py[imports]/b.py"

    def test_nested_paths(self):
        assert (
            Ref.for_connection("/src/a.py", "/lib/b.py", "calls").path
            == "/src/a.py[calls]/lib/b.py"
        )

    def test_various_types(self):
        for ct in ("contains", "imports", "calls"):
            r = Ref.for_connection("/a.py", "/b.py", ct)
            assert r.path == f"/a.py[{ct}]/b.py"


# ==================================================================
# Parsing — chunk (replaces TestParseRef chunk cases)
# ==================================================================


class TestParseChunk:
    def test_chunk_ref(self):
        r = Ref(path="/src/auth.py#login")
        assert r.is_chunk is True
        assert r.chunk == "login"
        assert r.base_path == "/src/auth.py"

    def test_scoped_chunk_ref(self):
        r = Ref(path="/src/auth.py#Client.connect")
        assert r.chunk == "Client.connect"

    def test_plain_path(self):
        r = Ref(path="/src/auth.py")
        assert r.is_chunk is False
        assert r.chunk is None

    def test_hash_in_dir_and_suffix(self):
        r = Ref(path="/dir#name/file.py#symbol")
        assert r.is_chunk is True
        assert r.chunk == "symbol"
        assert r.base_path == "/dir#name/file.py"

    def test_hash_in_dir_only(self):
        r = Ref(path="/dir#v1/file.py")
        assert r.is_chunk is False
        assert r.chunk is None

    def test_empty_chunk_treated_as_plain(self):
        r = Ref(path="/file.py#")
        assert r.is_chunk is False
        assert r.chunk is None


# ==================================================================
# Parsing — version (replaces TestParseRef version cases)
# ==================================================================


class TestParseVersion:
    def test_version_ref(self):
        r = Ref(path="/src/auth.py@3")
        assert r.is_version is True
        assert r.version == 3
        assert r.base_path == "/src/auth.py"

    def test_version_zero(self):
        assert Ref(path="/src/auth.py@0").version == 0

    def test_invalid_version(self):
        r = Ref(path="/file.py@abc")
        assert r.is_version is False
        assert r.version is None

    def test_at_in_dir_and_suffix(self):
        r = Ref(path="/dir@v2/file.py@3")
        assert r.is_version is True
        assert r.version == 3
        assert r.base_path == "/dir@v2/file.py"

    def test_empty_string(self):
        r = Ref(path="")
        assert r.is_version is False

    def test_root_path(self):
        r = Ref(path="/")
        assert r.is_version is False


# ==================================================================
# Parsing — connection (new)
# ==================================================================


class TestParseConnection:
    def test_connection_ref(self):
        r = Ref(path="/a.py[imports]/b.py")
        assert r.is_connection is True
        assert r.source == "/a.py"
        assert r.target == "/b.py"
        assert r.connection_type == "imports"

    def test_plain_not_connection(self):
        r = Ref(path="/a.py")
        assert r.is_connection is False

    def test_empty_type(self):
        r = Ref(path="/a.py[]/b.py")
        assert r.is_connection is False


# ==================================================================
# is_chunk (replaces TestIsChunkRef)
# ==================================================================


class TestIsChunkRef:
    def test_true(self):
        assert Ref(path="/a.py#foo").is_chunk is True

    def test_false_plain(self):
        assert Ref(path="/a.py").is_chunk is False

    def test_false_version(self):
        assert Ref(path="/a.py@3").is_chunk is False

    def test_false_trailing_hash(self):
        assert Ref(path="/a.py#").is_chunk is False

    def test_false_leading_hash(self):
        assert Ref(path="#foo").is_chunk is False

    def test_false_hash_in_directory(self):
        assert Ref(path="/dir#v1/file.py").is_chunk is False


# ==================================================================
# is_version (replaces TestIsVersionRef)
# ==================================================================


class TestIsVersionRef:
    def test_true(self):
        assert Ref(path="/a.py@3").is_version is True

    def test_false_plain(self):
        assert Ref(path="/a.py").is_version is False

    def test_false_chunk(self):
        assert Ref(path="/a.py#foo").is_version is False

    def test_false_non_numeric(self):
        assert Ref(path="/a.py@abc").is_version is False

    def test_version_zero(self):
        assert Ref(path="/a.py@0").is_version is True

    def test_false_leading_at(self):
        assert Ref(path="@3").is_version is False


# ==================================================================
# is_connection (new)
# ==================================================================


class TestIsConnection:
    def test_true(self):
        assert Ref(path="/a.py[imports]/b.py").is_connection is True

    def test_false_plain(self):
        assert Ref(path="/a.py").is_connection is False

    def test_false_empty_type(self):
        assert Ref(path="/a.py[]/b.py").is_connection is False

    def test_false_leading_bracket(self):
        assert Ref(path="[type]/b.py").is_connection is False


# ==================================================================
# base_path (replaces TestStripRef)
# ==================================================================


class TestBasePath:
    def test_strip_chunk(self):
        assert Ref(path="/src/auth.py#login").base_path == "/src/auth.py"

    def test_strip_version(self):
        assert Ref(path="/src/auth.py@3").base_path == "/src/auth.py"

    def test_plain_unchanged(self):
        assert Ref(path="/src/auth.py").base_path == "/src/auth.py"

    def test_root(self):
        assert Ref(path="/").base_path == "/"

    def test_empty(self):
        assert Ref(path="").base_path == ""

    def test_connection_returns_source(self):
        assert Ref(path="/a.py[imports]/b.py").base_path == "/a.py"


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

    def test_strip_chunk_round_trip(self):
        r = Ref.for_chunk("/a/b.py", "func")
        assert r.base_path == "/a/b.py"

    def test_strip_version_round_trip(self):
        r = Ref.for_version("/a/b.py", 7)
        assert r.base_path == "/a/b.py"

    def test_connection_round_trip(self):
        r = Ref.for_connection("/a.py", "/b.py", "imports")
        assert r.source == "/a.py"
        assert r.target == "/b.py"
        assert r.connection_type == "imports"

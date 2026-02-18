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
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("path", "/bar.txt", id="path"),
            pytest.param("version", 2, id="version"),
            pytest.param("line_start", 5, id="line_start"),
        ],
    )
    def test_cannot_set_field(self, field: str, value: object):
        r = Ref(path="/foo.txt", version=1)
        with pytest.raises(AttributeError):
            setattr(r, field, value)

    def test_metadata_dict_is_immutable(self):
        """Metadata is wrapped in MappingProxyType — truly read-only."""
        r = Ref(path="/foo.txt", metadata={"a": 1})
        with pytest.raises(TypeError):
            r.metadata["b"] = 2  # type: ignore[index]

    def test_metadata_readable_after_creation(self):
        """Metadata values are still readable through the proxy."""
        r = Ref(path="/foo.txt", metadata={"a": 1, "b": "two"})
        assert r.metadata["a"] == 1
        assert r.metadata["b"] == "two"
        assert len(r.metadata) == 2

    def test_metadata_compares_equal_to_dict(self):
        """MappingProxyType compares equal to an equivalent dict."""
        r = Ref(path="/foo.txt", metadata={"x": 42})
        assert r.metadata == {"x": 42}


class TestRefEquality:
    @pytest.mark.parametrize(
        ("a_kwargs", "b_kwargs", "expected_equal"),
        [
            pytest.param(
                {"path": "/foo.txt", "version": 1},
                {"path": "/foo.txt", "version": 1},
                True,
                id="equal-refs",
            ),
            pytest.param(
                {"path": "/foo.txt"},
                {"path": "/bar.txt"},
                False,
                id="different-path",
            ),
            pytest.param(
                {"path": "/foo.txt", "version": 1},
                {"path": "/foo.txt", "version": 2},
                False,
                id="different-version",
            ),
            pytest.param(
                {"path": "/foo.txt", "metadata": {"x": 1}},
                {"path": "/foo.txt", "metadata": {"y": 2}},
                True,
                id="metadata-excluded",
            ),
        ],
    )
    def test_equality(
        self,
        a_kwargs: dict,
        b_kwargs: dict,
        expected_equal: bool,
    ):
        assert (Ref(**a_kwargs) == Ref(**b_kwargs)) is expected_equal

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

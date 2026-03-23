"""Tests for grover.paths — path utilities for the dot-prefix metadata namespace."""

from __future__ import annotations

import pytest

from grover.paths import (
    ConnectionParts,
    METADATA_KIND_MAP,
    METADATA_MARKERS,
    MARKER_KINDS,
    api_path,
    base_path,
    chunk_path,
    connection_path,
    decompose_connection,
    normalize_path,
    parent_path,
    parse_kind,
    split_path,
    validate_path,
    version_path,
)


# =========================================================================
# normalize_path
# =========================================================================


class TestNormalizePath:
    def test_empty_string(self):
        assert normalize_path("") == "/"

    def test_adds_leading_slash(self):
        assert normalize_path("foo.txt") == "/foo.txt"

    def test_resolves_double_slashes(self):
        assert normalize_path("/foo//bar.txt") == "/foo/bar.txt"

    def test_resolves_dot_dot(self):
        assert normalize_path("/foo/../bar.txt") == "/bar.txt"

    def test_resolves_single_dot(self):
        assert normalize_path("/foo/./bar.txt") == "/foo/bar.txt"

    def test_removes_trailing_slash(self):
        assert normalize_path("/foo/") == "/foo"

    def test_root_preserved(self):
        assert normalize_path("/") == "/"

    def test_dot_dot_at_root_clamps(self):
        assert normalize_path("/../../../etc/passwd") == "/etc/passwd"

    def test_metadata_traversal(self):
        assert normalize_path("/.chunks/../../../etc/passwd") == "/etc/passwd"

    def test_nfc_normalization(self):
        # NFD é (e + combining acute) should collapse to NFC é
        nfd = "/caf\u0065\u0301"
        nfc = "/caf\u00e9"
        assert normalize_path(nfd) == normalize_path(nfc)

    def test_whitespace_stripped(self):
        assert normalize_path("  /foo  ") == "/foo"

    def test_only_slashes(self):
        assert normalize_path("///") == "/"


# =========================================================================
# split_path
# =========================================================================


class TestSplitPath:
    def test_root(self):
        assert split_path("/") == ("/", "")

    def test_single_segment(self):
        assert split_path("/foo") == ("/", "foo")

    def test_nested(self):
        assert split_path("/src/auth.py") == ("/src", "auth.py")

    def test_metadata_path_is_literal_split(self):
        # split_path is a pure string operation, not metadata-aware
        assert split_path("/src/auth.py/.chunks/login") == (
            "/src/auth.py/.chunks",
            "login",
        )

    def test_connection_path_is_literal_split(self):
        assert split_path("/src/auth.py/.connections/imports/src/utils.py") == (
            "/src/auth.py/.connections/imports/src",
            "utils.py",
        )


# =========================================================================
# validate_path
# =========================================================================


class TestValidatePath:
    def test_valid_paths(self):
        valid = [
            "/src/auth.py",
            "/",
            "/a",
            "/src/auth.py/.chunks/login",
            "/jira/.apis/ticket",
            "/documents/quarterly-report.pdf",
        ]
        for p in valid:
            ok, msg = validate_path(p)
            assert ok, f"{p!r} should be valid: {msg}"

    def test_null_byte(self):
        ok, _ = validate_path("/foo\x00bar")
        assert not ok

    @pytest.mark.parametrize(
        "ch",
        ["\x01", "\x0b", "\x1f", "\t", "\n", "\r"],
        ids=["SOH", "VT", "US", "tab", "newline", "CR"],
    )
    def test_ascii_control_chars_rejected(self, ch):
        ok, _ = validate_path(f"/foo{ch}bar")
        assert not ok

    def test_del_rejected(self):
        ok, _ = validate_path("/foo\x7fbar")
        assert not ok

    def test_c1_control_rejected(self):
        ok, _ = validate_path("/foo\x9fbar")
        assert not ok

    def test_path_too_long(self):
        ok, _ = validate_path("/" + "a" * 4096)
        assert not ok

    def test_path_at_limit(self):
        # 4096 total: "/" + 15 segments of "a" * 255 joined by "/"
        # Use a path that's long but has valid segment lengths
        path = "/" + "/".join(["a" * 255] * 15)
        assert len(path) <= 4096
        ok, _ = validate_path(path)
        assert ok

    def test_segment_too_long(self):
        ok, _ = validate_path("/" + "a" * 256)
        assert not ok

    def test_segment_at_limit(self):
        ok, _ = validate_path("/" + "a" * 255)
        assert ok


# =========================================================================
# parse_kind
# =========================================================================


class TestParseKind:
    # --- Metadata markers ---

    def test_chunk(self):
        assert parse_kind("/src/auth.py/.chunks/login") == "chunk"

    def test_version(self):
        assert parse_kind("/src/auth.py/.versions/3") == "version"

    def test_connection(self):
        assert parse_kind("/src/auth.py/.connections/imports/src/utils.py") == "connection"

    def test_api(self):
        assert parse_kind("/jira/.apis/ticket") == "api"

    # --- Files with extensions ---

    @pytest.mark.parametrize("path", ["/src/auth.py", "/docs/readme.md", "/data/report.pdf"])
    def test_file_with_extension(self, path):
        assert parse_kind(path) == "file"

    def test_multiple_extensions(self):
        assert parse_kind("/archive.tar.gz") == "file"

    def test_trailing_dot(self):
        # file. has a dot at position > 0
        assert parse_kind("/file.") == "file"

    # --- Dotfiles ---

    @pytest.mark.parametrize("name", [".bashrc", ".gitconfig", ".hidden", ".vimrc"])
    def test_unlisted_dotfiles_are_files(self, name):
        assert parse_kind(f"/home/{name}") == "file"

    def test_listed_dotfile(self):
        assert parse_kind("/.gitignore") == "file"

    def test_dotfile_with_extension(self):
        assert parse_kind("/.env.local") == "file"

    # --- Reserved metadata names as bare directories ---

    @pytest.mark.parametrize("name", [".chunks", ".versions", ".connections", ".apis"])
    def test_reserved_names_are_directories(self, name):
        assert parse_kind(f"/foo/{name}") == "directory"

    # --- Extensionless files (case-insensitive) ---

    @pytest.mark.parametrize("name", ["Makefile", "makefile", "MAKEFILE"])
    def test_makefile_case_insensitive(self, name):
        assert parse_kind(f"/{name}") == "file"

    @pytest.mark.parametrize("name", ["LICENSE", "license", "License"])
    def test_license_case_insensitive(self, name):
        assert parse_kind(f"/{name}") == "file"

    def test_dockerfile(self):
        assert parse_kind("/Dockerfile") == "file"

    # --- Directories ---

    @pytest.mark.parametrize("path", ["/src", "/documents", "/", "/people/teams"])
    def test_directories(self, path):
        assert parse_kind(path) == "directory"

    # --- Marker boundary (no false positives) ---

    def test_similar_name_not_misclassified(self):
        # "/my-connections/" should not trigger the /.connections/ marker
        assert parse_kind("/my-connections/file.sql") == "file"


# =========================================================================
# base_path
# =========================================================================


class TestBasePath:
    def test_file_returns_self(self):
        assert base_path("/src/auth.py") == "/src/auth.py"

    def test_chunk(self):
        assert base_path("/src/auth.py/.chunks/login") == "/src/auth.py"

    def test_version(self):
        assert base_path("/src/auth.py/.versions/3") == "/src/auth.py"

    def test_connection_deep_target(self):
        assert base_path("/src/auth.py/.connections/imports/src/deep/path.py") == "/src/auth.py"

    def test_api(self):
        assert base_path("/jira/.apis/ticket") == "/jira"

    def test_root(self):
        assert base_path("/") == "/"

    def test_bare_metadata_dir(self):
        assert base_path("/src/auth.py/.chunks") == "/src/auth.py"
        assert base_path("/src/auth.py/.versions") == "/src/auth.py"
        assert base_path("/src/auth.py/.connections") == "/src/auth.py"
        assert base_path("/jira/.apis") == "/jira"

    def test_first_marker_wins(self):
        assert base_path("/foo/.chunks/.versions/1") == "/foo"


# =========================================================================
# parent_path
# =========================================================================


class TestParentPath:
    def test_file(self):
        assert parent_path("/src/auth.py") == "/src"

    def test_root_child(self):
        assert parent_path("/src") == "/"

    def test_root_is_own_parent(self):
        assert parent_path("/") == "/"

    def test_chunk(self):
        assert parent_path("/src/auth.py/.chunks/login") == "/src/auth.py"

    def test_version(self):
        assert parent_path("/src/auth.py/.versions/3") == "/src/auth.py"

    def test_connection(self):
        assert parent_path("/src/auth.py/.connections/imports/src/utils.py") == "/src/auth.py"

    def test_api(self):
        assert parent_path("/jira/.apis/ticket") == "/jira"

    def test_bare_metadata_dir(self):
        # /.chunks (no trailing /) falls through to split_path
        assert parent_path("/src/auth.py/.chunks") == "/src/auth.py"


# =========================================================================
# chunk_path
# =========================================================================


class TestChunkPath:
    def test_basic(self):
        assert chunk_path("/src/auth.py", "login") == "/src/auth.py/.chunks/login"

    def test_normalizes_file_path(self):
        assert chunk_path("src/auth.py", "login") == "/src/auth.py/.chunks/login"

    def test_roundtrip_parse_kind(self):
        assert parse_kind(chunk_path("/f.py", "x")) == "chunk"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="chunk_name"):
            chunk_path("/f.py", "")

    def test_slash_in_name_rejected(self):
        with pytest.raises(ValueError, match="chunk_name"):
            chunk_path("/f.py", "a/b")

    def test_metadata_base_rejected(self):
        with pytest.raises(ValueError, match="metadata"):
            chunk_path("/f.py/.chunks/bar", "x")

    def test_reserved_ending_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            chunk_path("/foo/.chunks", "x")


# =========================================================================
# version_path
# =========================================================================


class TestVersionPath:
    def test_basic(self):
        assert version_path("/src/auth.py", 3) == "/src/auth.py/.versions/3"

    def test_roundtrip_parse_kind(self):
        assert parse_kind(version_path("/f.py", 1)) == "version"

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="version_number"):
            version_path("/f.py", -1)

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="version_number"):
            version_path("/f.py", 0)

    def test_metadata_base_rejected(self):
        with pytest.raises(ValueError, match="metadata"):
            version_path("/f.py/.versions/1", 2)


# =========================================================================
# connection_path + decompose_connection (roundtrip)
# =========================================================================


class TestConnectionPath:
    def test_basic(self):
        assert (
            connection_path("/src/auth.py", "/src/utils.py", "imports")
            == "/src/auth.py/.connections/imports/src/utils.py"
        )

    def test_roundtrip(self):
        cases = [
            ("/src/auth.py", "/src/utils.py", "imports"),
            ("/jira/PROJ-1", "/src/auth.py", "references"),
            ("/a.py", "/deep/nested/path.py", "calls"),
        ]
        for s, t, c in cases:
            path = connection_path(s, t, c)
            parts = decompose_connection(path)
            assert parts == ConnectionParts(source=s, target=t, connection_type=c)

    def test_normalizes_target(self):
        p = connection_path("/a.py", "src//utils.py", "imports")
        assert p == "/a.py/.connections/imports/src/utils.py"

    def test_empty_type_rejected(self):
        with pytest.raises(ValueError, match="connection_type"):
            connection_path("/a.py", "/b.py", "")

    def test_slash_in_type_rejected(self):
        with pytest.raises(ValueError, match="connection_type"):
            connection_path("/a.py", "/b.py", "calls/async")

    def test_root_target_rejected(self):
        with pytest.raises(ValueError, match="target"):
            connection_path("/a.py", "/", "imports")

    def test_metadata_base_rejected(self):
        with pytest.raises(ValueError, match="metadata"):
            connection_path("/a.py/.connections/imports/b.py", "/c.py", "calls")


class TestDecomposeConnection:
    def test_basic(self):
        result = decompose_connection("/src/auth.py/.connections/imports/src/utils.py")
        assert result == ConnectionParts(
            source="/src/auth.py",
            target="/src/utils.py",
            connection_type="imports",
        )

    def test_not_a_connection(self):
        assert decompose_connection("/src/auth.py") is None

    def test_type_only_no_target(self):
        assert decompose_connection("/foo/.connections/imports") is None

    def test_deep_target(self):
        result = decompose_connection("/a.py/.connections/calls/src/deep/nested/path.py")
        assert result is not None
        assert result.source == "/a.py"
        assert result.target == "/src/deep/nested/path.py"
        assert result.connection_type == "calls"

    def test_named_access(self):
        result = decompose_connection("/a.py/.connections/imports/b.py")
        assert result is not None
        assert result.source == "/a.py"
        assert result.target == "/b.py"
        assert result.connection_type == "imports"
        # Positional matches named
        assert result[0] == result.source
        assert result[1] == result.target
        assert result[2] == result.connection_type


# =========================================================================
# api_path
# =========================================================================


class TestApiPath:
    def test_basic(self):
        assert api_path("/jira", "ticket") == "/jira/.apis/ticket"

    def test_roundtrip_parse_kind(self):
        assert parse_kind(api_path("/jira", "ticket")) == "api"

    def test_empty_action_rejected(self):
        with pytest.raises(ValueError, match="action"):
            api_path("/jira", "")

    def test_whitespace_action_rejected(self):
        with pytest.raises(ValueError, match="action"):
            api_path("/jira", "   ")

    def test_slash_in_action_rejected(self):
        with pytest.raises(ValueError, match="action"):
            api_path("/jira", "ticket/create")

    def test_metadata_mount_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            api_path("/jira/.apis", "ticket")


# =========================================================================
# Derived constants
# =========================================================================


class TestConstants:
    def test_metadata_markers_derived_from_map(self):
        for name, kind in METADATA_KIND_MAP.items():
            marker = f"/{name}/"
            assert marker in METADATA_MARKERS
            assert MARKER_KINDS[marker] == kind

    def test_marker_count_matches_map(self):
        assert len(METADATA_MARKERS) == len(METADATA_KIND_MAP)
        assert len(MARKER_KINDS) == len(METADATA_KIND_MAP)

    def test_apis_in_markers(self):
        assert "/.apis/" in METADATA_MARKERS
        assert "/.api/" not in METADATA_MARKERS

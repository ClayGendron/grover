"""Tests for grover.patterns — glob pattern matching and SQL LIKE translation."""

from __future__ import annotations

import re

import pytest

from grover.patterns import compile_glob, glob_to_sql_like, match_glob

# =========================================================================
# Single star (*) — matches any characters except /
# =========================================================================


class TestSingleStar:
    """Single * matches zero or more characters within a single path segment."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # Basic extension matching
            ("*.py", "/foo.py", True),
            ("*.py", "/bar.py", True),
            ("*.py", "/baz.txt", False),
            ("*.py", "/.py", True),  # * matches zero chars
            # Should NOT match across /
            ("*.py", "/dir/foo.py", False),
            ("*.py", "/a/b/c.py", False),
            # Star in the middle
            ("foo*bar", "/foobar", True),
            ("foo*bar", "/foo_anything_bar", True),
            ("foo*bar", "/fooXbar", True),
            ("foo*bar", "/foo/bar", False),  # * cannot cross /
            # Star at the beginning
            ("*foo", "/foo", True),
            ("*foo", "/barfoo", True),
            ("*foo", "/dir/barfoo", False),  # cannot cross /
            # Star at the end
            ("foo*", "/foo", True),
            ("foo*", "/foobar", True),
            ("foo*", "/foo.py", True),
            # Multiple stars in one segment
            ("*.*", "/foo.py", True),
            ("*.*", "/a.b", True),
            ("*.*", "/file", False),  # no dot
            ("*.*", "/dir/foo.py", False),  # crosses /
            # Star matching empty string
            ("*.py", "/.py", True),
            ("foo*", "/foo", True),
            ("*bar", "/bar", True),
            # Star in directory + filename
            ("src/*.py", "/src/foo.py", True),
            ("src/*.py", "/src/bar.py", True),
            ("src/*.py", "/src/sub/foo.py", False),
            ("src/*.py", "/other/foo.py", False),
            # Star in directory name
            ("*/foo.py", "/src/foo.py", True),
            ("*/foo.py", "/lib/foo.py", True),
            ("*/foo.py", "/a/b/foo.py", False),  # only one segment
        ],
    )
    def test_single_star(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Double star (**) — matches across directory boundaries
# =========================================================================


class TestDoubleStar:
    """Double ** matches zero or more path segments."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # **/*.ext — any depth
            ("**/*.py", "/foo.py", True),
            ("**/*.py", "/src/foo.py", True),
            ("**/*.py", "/a/b/c/d.py", True),
            ("**/*.py", "/foo.txt", False),
            # **/name — match name anywhere
            ("**/foo.py", "/foo.py", True),
            ("**/foo.py", "/src/foo.py", True),
            ("**/foo.py", "/a/b/c/foo.py", True),
            ("**/foo.py", "/foo.txt", False),
            # prefix/**/suffix
            ("src/**/foo.py", "/src/foo.py", True),
            ("src/**/foo.py", "/src/a/foo.py", True),
            ("src/**/foo.py", "/src/a/b/c/foo.py", True),
            ("src/**/foo.py", "/other/foo.py", False),
            # ** at end
            ("src/**", "/src/foo.py", True),
            ("src/**", "/src/a/b/c.py", True),
            ("src/**", "/src/", True),  # ** at end becomes .* which matches /
            # ** alone
            ("**", "/foo.py", True),
            ("**", "/a/b/c", True),
            # Double star with extension (no slash between)
            ("**.py", "/foo.py", True),
            ("**.py", "/a/b/c.py", True),
            # prefix/** with deeper nesting
            ("a/**/z", "/a/z", True),
            ("a/**/z", "/a/b/z", True),
            ("a/**/z", "/a/b/c/d/z", True),
            ("a/**/z", "/b/z", False),
            # Multiple ** segments
            ("**/src/**/test_*.py", "/src/test_foo.py", True),
            ("**/src/**/test_*.py", "/root/src/tests/test_bar.py", True),
            ("**/src/**/test_*.py", "/root/src/a/b/test_baz.py", True),
        ],
    )
    def test_double_star(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Question mark (?) — single character except /
# =========================================================================


class TestQuestionMark:
    """? matches exactly one character that is not /."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            ("?.py", "/a.py", True),
            ("?.py", "/b.py", True),
            ("?.py", "/.py", False),  # ? requires exactly one char
            ("?.py", "/ab.py", False),  # ? is one char only
            ("?.py", "/dir/a.py", False),  # ? cannot be /
            ("??.py", "/ab.py", True),
            ("??.py", "/a.py", False),
            ("??.py", "/abc.py", False),
            # ? in the middle
            ("f?o.py", "/fao.py", True),
            ("f?o.py", "/fbo.py", True),
            ("f?o.py", "/foo.py", True),
            ("f?o.py", "/fo.py", False),
            ("f?o.py", "/faao.py", False),
            # ? at end
            ("foo.p?", "/foo.py", True),
            ("foo.p?", "/foo.px", True),
            ("foo.p?", "/foo.p", False),
            # ? in directory
            ("sr?/foo.py", "/src/foo.py", True),
            ("sr?/foo.py", "/srv/foo.py", True),
            ("sr?/foo.py", "/s/foo.py", False),
            # Multiple ?
            ("???", "/abc", True),
            ("???", "/ab", False),
            ("???", "/abcd", False),
        ],
    )
    def test_question_mark(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Character classes [seq] and [!seq]
# =========================================================================


class TestCharacterClasses:
    """[seq] matches any char in seq, [!seq] matches any char NOT in seq."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # Basic character class
            ("[abc].py", "/a.py", True),
            ("[abc].py", "/b.py", True),
            ("[abc].py", "/c.py", True),
            ("[abc].py", "/d.py", False),
            # Character range
            ("[a-z].py", "/a.py", True),
            ("[a-z].py", "/z.py", True),
            ("[a-z].py", "/A.py", False),
            ("[0-9].py", "/5.py", True),
            ("[0-9].py", "/a.py", False),
            # Negation
            ("[!abc].py", "/d.py", True),
            ("[!abc].py", "/a.py", False),
            ("[!abc].py", "/b.py", False),
            # Negation with range
            ("[!0-9].py", "/a.py", True),
            ("[!0-9].py", "/5.py", False),
            # Character class in directory
            ("src/[abc].py", "/src/a.py", True),
            ("src/[abc].py", "/src/d.py", False),
            # Character class with star
            ("[st]*.py", "/src.py", True),
            ("[st]*.py", "/test.py", True),
            ("[st]*.py", "/abc.py", False),
        ],
    )
    def test_character_classes(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected

    def test_unclosed_bracket_treated_as_literal(self):
        """An unclosed [ is treated as a literal character."""
        # Pattern "[abc" has no closing ] — [ treated as literal
        assert match_glob("/[abc", "[abc") is True
        assert match_glob("/a", "[abc") is False


# =========================================================================
# Base path handling
# =========================================================================


class TestBasePath:
    """Tests for base_path parameter behavior in all three functions."""

    @pytest.mark.parametrize(
        "pattern, base_path, path, expected",
        [
            # Default base_path is /
            ("*.py", "/", "/foo.py", True),
            ("*.py", "/", "/dir/foo.py", False),
            # Custom base path
            ("*.py", "/src", "/src/foo.py", True),
            ("*.py", "/src", "/other/foo.py", False),
            ("*.py", "/src", "/foo.py", False),
            # Nested base path
            ("*.py", "/src/lib", "/src/lib/foo.py", True),
            ("*.py", "/src/lib", "/src/foo.py", False),
            # Base path with ** pattern
            ("**/*.py", "/src", "/src/foo.py", True),
            ("**/*.py", "/src", "/src/a/b/foo.py", True),
            ("**/*.py", "/src", "/other/foo.py", False),
            # Absolute pattern ignores base
            ("/*.py", "/src", "/foo.py", True),
            ("/*.py", "/src", "/src/foo.py", False),
            ("/src/*.py", "/other", "/src/foo.py", True),
            # Base path normalization (trailing slash removed)
            ("*.py", "/src/", "/src/foo.py", True),
            # Base path normalization (no leading slash)
            ("*.py", "src", "/src/foo.py", True),
        ],
    )
    def test_base_path(
        self, pattern: str, base_path: str, path: str, expected: bool
    ):
        assert match_glob(path, pattern, base_path=base_path) is expected

    def test_compile_glob_with_base_path(self):
        regex = compile_glob("*.py", base_path="/src")
        assert regex is not None
        assert regex.match("/src/foo.py") is not None
        assert regex.match("/other/foo.py") is None

    def test_glob_to_sql_like_with_base_path(self):
        result = glob_to_sql_like("*.py", base_path="/src")
        assert result is not None
        assert result.startswith("/src/")

    def test_glob_to_sql_like_default_base(self):
        result = glob_to_sql_like("*.py")
        assert result is not None
        assert result.startswith("/")


# =========================================================================
# glob_to_sql_like — token translation
# =========================================================================


class TestGlobToSqlLike:
    """Tests for SQL LIKE translation specifics."""

    @pytest.mark.parametrize(
        "pattern, expected_like",
        [
            # Single * → %
            ("*.py", "/%.py"),
            # ** → % (trailing / consumed), then * → another %
            ("**/*.py", "/%%.py"),
            # ? → _
            ("?.py", "/_.py"),
            # Literal text
            ("foo.py", "/foo.py"),
            # Multiple tokens
            ("src/*.py", "/src/%.py"),
            ("**/test_*.py", "/%test\\_%.py"),  # _ in test_ is escaped
            # ** at end
            ("src/**", "/src/%"),
            # ** without trailing /
            ("src/**", "/src/%"),
            # Escaping % in pattern
            ("100%.txt", "/100\\%.txt"),
            # Escaping _ in pattern
            ("my_file.txt", "/my\\_file.txt"),
            # Combined escaping
            ("100%_file.txt", "/100\\%\\_file.txt"),
            # Just **
            ("**", "/%"),
            # **.py (no / after **)
            ("**.py", "/%.py"),
            # Multiple ** segments
            ("**/src/**/*.py", "/%src/%%.py"),
        ],
    )
    def test_like_translation(self, pattern: str, expected_like: str):
        assert glob_to_sql_like(pattern) == expected_like

    def test_returns_none_for_bracket(self):
        """Any pattern with [ returns None."""
        assert glob_to_sql_like("[abc].py") is None
        assert glob_to_sql_like("*.py") is not None

    def test_returns_none_for_unclosed_bracket(self):
        """Even unclosed [ returns None since we check for '[' in pattern."""
        assert glob_to_sql_like("[abc") is None

    def test_returns_none_for_negated_bracket(self):
        assert glob_to_sql_like("[!abc].py") is None

    @pytest.mark.parametrize(
        "pattern, base_path, expected_like",
        [
            ("*.py", "/src", "/src/%.py"),
            ("*.py", "/a/b", "/a/b/%.py"),
            ("/abs.py", "/src", "/abs.py"),  # absolute pattern ignores base
            ("**/*.py", "/src", "/src/%%.py"),
        ],
    )
    def test_like_with_base_path(
        self, pattern: str, base_path: str, expected_like: str
    ):
        assert glob_to_sql_like(pattern, base_path=base_path) == expected_like


# =========================================================================
# compile_glob — regex compilation
# =========================================================================


class TestCompileGlob:
    """Tests for compile_glob specifically."""

    def test_returns_pattern_object(self):
        result = compile_glob("*.py")
        assert isinstance(result, re.Pattern)

    def test_valid_pattern_compiles(self):
        result = compile_glob("**/*.py")
        assert result is not None

    def test_simple_pattern(self):
        regex = compile_glob("foo.py")
        assert regex is not None
        assert regex.match("/foo.py") is not None
        assert regex.match("/bar.py") is None

    def test_with_base_path(self):
        regex = compile_glob("*.py", base_path="/src")
        assert regex is not None
        assert regex.match("/src/foo.py") is not None
        assert regex.match("/foo.py") is None

    def test_absolute_pattern_ignores_base(self):
        regex = compile_glob("/src/*.py", base_path="/other")
        assert regex is not None
        assert regex.match("/src/foo.py") is not None
        assert regex.match("/other/foo.py") is None

    def test_reuse_compiled_pattern(self):
        """Compiled regex can be reused for many paths efficiently."""
        regex = compile_glob("**/*.py")
        assert regex is not None
        paths = ["/a.py", "/src/b.py", "/a/b/c.py", "/foo.txt"]
        results = [regex.match(p) is not None for p in paths]
        assert results == [True, True, True, False]


# =========================================================================
# LIKE / match_glob consistency
# =========================================================================


class TestLikeMatchGlobConsistency:
    """LIKE is a superset — if match_glob returns True, LIKE must also match.

    LIKE may produce false positives (e.g., * matches across / in LIKE),
    but must never produce false negatives.
    """

    @pytest.mark.parametrize(
        "pattern, path",
        [
            # Cases where match_glob is True — LIKE must also match
            ("*.py", "/foo.py"),
            ("**/*.py", "/src/foo.py"),
            ("**/*.py", "/a/b/c.py"),
            ("src/*.py", "/src/foo.py"),
            ("src/**/foo.py", "/src/foo.py"),
            ("src/**/foo.py", "/src/a/b/foo.py"),
            ("**/test_*.py", "/tests/test_foo.py"),
            ("foo*bar.py", "/fooXbar.py"),
            ("?.py", "/a.py"),
            ("foo.py", "/foo.py"),
        ],
    )
    def test_like_superset_true_matches(self, pattern: str, path: str):
        """When match_glob is True, LIKE should also match (no false negatives)."""
        assert match_glob(path, pattern) is True
        like = glob_to_sql_like(pattern)
        if like is not None:
            # Convert LIKE to regex for testing
            like_regex = _like_to_regex(like)
            assert like_regex.match(path) is not None, (
                f"LIKE '{like}' rejected '{path}' but match_glob accepted it"
            )

    @pytest.mark.parametrize(
        "pattern, path",
        [
            # Cases where LIKE matches but match_glob does NOT (false positives OK)
            ("*.py", "/dir/foo.py"),  # * in LIKE is %, crosses /
            ("src/*.py", "/src/sub/foo.py"),  # same issue
        ],
    )
    def test_like_false_positives_ok(self, pattern: str, path: str):
        """LIKE may match paths that match_glob rejects — that's by design."""
        assert match_glob(path, pattern) is False
        like = glob_to_sql_like(pattern)
        if like is not None:
            like_regex = _like_to_regex(like)
            # LIKE will match here — that's expected
            assert like_regex.match(path) is not None

    @pytest.mark.parametrize(
        "pattern, path",
        [
            # Cases where both reject
            ("*.py", "/foo.txt"),
            ("src/*.py", "/other/foo.py"),
            ("foo.py", "/bar.py"),
        ],
    )
    def test_both_reject(self, pattern: str, path: str):
        """When match_glob is False, LIKE should ideally also be False."""
        assert match_glob(path, pattern) is False
        like = glob_to_sql_like(pattern)
        if like is not None:
            like_regex = _like_to_regex(like)
            assert like_regex.match(path) is None


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases: empty patterns, dots, unicode, deeply nested paths, etc."""

    def test_empty_pattern(self):
        """Empty pattern should match root only."""
        # Empty pattern becomes "/" after prepending base
        assert match_glob("/", "") is True
        assert match_glob("/foo", "") is False

    def test_root_pattern(self):
        assert match_glob("/", "/") is True
        assert match_glob("/foo", "/") is False

    def test_pattern_with_spaces(self):
        assert match_glob("/my file.py", "my file.py") is True
        assert match_glob("/my file.py", "my*.py") is True
        assert match_glob("/dir/my file.py", "*/my file.py") is True

    def test_unicode_paths(self):
        assert match_glob("/src/cafe\u0301.py", "**/*.py") is True
        assert match_glob("/src/\u00fcber.py", "**/\u00fcber.py") is True
        assert match_glob("/\u6587\u4ef6.txt", "*.txt") is True

    def test_dot_files(self):
        assert match_glob("/.gitignore", ".gitignore") is True
        assert match_glob("/.gitignore", "*") is True  # * matches dot-prefixed
        assert match_glob("/.gitignore", ".*") is True
        assert match_glob("/src/.env", "**/.env") is True

    def test_multiple_extensions(self):
        assert match_glob("/foo.tar.gz", "*.gz") is True
        assert match_glob("/foo.tar.gz", "*.tar.gz") is True
        assert match_glob("/foo.tar.gz", "*.*.*") is True
        assert match_glob("/foo.tar.gz", "*.py") is False

    def test_deeply_nested_paths(self):
        deep = "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p.py"
        assert match_glob(deep, "**/*.py") is True
        assert match_glob(deep, "a/**/*.py") is True
        assert match_glob(deep, "*.py") is False  # single * can't cross /

    def test_path_with_dots_in_directory(self):
        assert match_glob("/src/v1.2/foo.py", "**/*.py") is True
        assert match_glob("/src/v1.2/foo.py", "src/v1.2/*.py") is True
        assert match_glob("/src/v1.2/foo.py", "src/*/*.py") is True

    def test_pattern_is_exact_path(self):
        """Pattern without any glob characters matches exactly."""
        assert match_glob("/src/foo.py", "/src/foo.py") is True
        assert match_glob("/src/foo.py", "/src/bar.py") is False

    def test_single_segment_path(self):
        assert match_glob("/foo", "foo") is True
        assert match_glob("/foo", "*") is True
        assert match_glob("/foo", "**") is True
        assert match_glob("/foo", "bar") is False

    def test_trailing_slash_in_path(self):
        """Paths are matched as given; trailing slashes are significant."""
        # match_glob does not normalize the path
        assert match_glob("/src/", "src/") is True

    def test_regex_special_chars_escaped(self):
        """Regex metacharacters in the pattern should be escaped."""
        assert match_glob("/foo.py", "foo.py") is True
        assert match_glob("/fooXpy", "foo.py") is False  # . is literal, not regex any
        assert match_glob("/foo(1).py", "foo(1).py") is True
        assert match_glob("/foo+bar.py", "foo+bar.py") is True
        assert match_glob("/foo^bar.py", "foo^bar.py") is True
        assert match_glob("/$100.py", "$100.py") is True

    def test_pattern_with_pipe(self):
        """Pipe | should be treated as literal, not regex alternation."""
        assert match_glob("/a|b.py", "a|b.py") is True
        assert match_glob("/a.py", "a|b.py") is False


# =========================================================================
# Metadata paths — .chunks/, .versions/, .connections/
# =========================================================================


class TestMetadataPaths:
    """Glob matching on Grover's dot-prefix metadata paths."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # Chunk paths
            ("**/.chunks/*", "/src/auth.py/.chunks/login", True),
            ("**/.chunks/*", "/src/auth.py/.chunks/logout", True),
            ("**/.chunks/*", "/src/auth.py", False),
            # Version paths
            ("**/.versions/*", "/src/auth.py/.versions/3", True),
            ("**/.versions/*", "/src/auth.py/.versions/1", True),
            ("**/.versions/*", "/src/auth.py", False),
            # Connection paths — deeper nesting
            (
                "**/.connections/**",
                "/src/auth.py/.connections/imports/src/utils.py",
                True,
            ),
            (
                "**/.connections/imports/**",
                "/src/auth.py/.connections/imports/src/utils.py",
                True,
            ),
            ("**/.connections/**", "/src/auth.py", False),
            # Chunks for a specific file
            (
                "/src/auth.py/.chunks/*",
                "/src/auth.py/.chunks/login",
                True,
            ),
            (
                "/src/auth.py/.chunks/*",
                "/src/auth.py/.chunks/logout",
                True,
            ),
            (
                "/src/auth.py/.chunks/*",
                "/src/other.py/.chunks/login",
                False,
            ),
            # All metadata under a specific file
            (
                "/src/auth.py/.**",
                "/src/auth.py/.chunks/login",
                True,
            ),
            (
                "/src/auth.py/.**",
                "/src/auth.py/.versions/3",
                True,
            ),
            # API paths
            ("**/.apis/*", "/jira/.apis/ticket", True),
            ("**/.apis/*", "/jira/.apis/search", True),
            ("**/.apis/*", "/jira/search", False),
        ],
    )
    def test_metadata_paths(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Real-world patterns
# =========================================================================


class TestRealWorldPatterns:
    """Common glob patterns used in practice."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # All Python files
            ("**/*.py", "/src/grover/client.py", True),
            ("**/*.py", "/tests/test_foo.py", True),
            ("**/*.py", "/setup.cfg", False),
            # All test files
            ("**/test_*.py", "/tests/test_foo.py", True),
            ("**/test_*.py", "/tests/sub/test_bar.py", True),
            ("**/test_*.py", "/tests/foo.py", False),
            ("**/test_*.py", "/tests/test_foo.txt", False),
            # Everything under src/
            ("src/**", "/src/foo.py", True),
            ("src/**", "/src/a/b/c.py", True),
            ("src/**", "/tests/foo.py", False),
            # Specific directory
            ("src/grover/*.py", "/src/grover/client.py", True),
            ("src/grover/*.py", "/src/grover/sub/foo.py", False),
            # All YAML files
            ("**/*.yml", "/config/db.yml", True),
            ("**/*.yml", "/.github/workflows/ci.yml", True),
            ("**/*.yml", "/config/db.yaml", False),
            # Both YAML extensions
            ("**/*.y?ml", "/config/db.yml", False),  # y?ml needs yamlor yml (4 chars)
            # Hidden files
            ("**/.*", "/.gitignore", True),
            ("**/.*", "/src/.env", True),
            ("**/.*", "/src/foo.py", False),
            # All files in any __pycache__
            ("**/__pycache__/*", "/src/__pycache__/foo.pyc", True),
            ("**/__pycache__/*", "/src/a/__pycache__/bar.pyc", True),
            # Specific chunk patterns
            ("/**/.chunks/*", "/src/auth.py/.chunks/login", True),
            # All markdown files at root
            ("/*.md", "/README.md", True),
            ("/*.md", "/src/docs/README.md", False),
            # Node modules
            ("**/node_modules/**", "/app/node_modules/lodash/index.js", True),
            ("**/node_modules/**", "/app/src/index.js", False),
        ],
    )
    def test_real_world(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Adversarial patterns
# =========================================================================


class TestAdversarial:
    """Adversarial and unusual patterns."""

    def test_triple_star(self):
        """*** is parsed as ** + * — ** consumes two, * is next."""
        # _glob_to_regex: first two * become **, then third * becomes [^/]*
        # *** at end of pattern: ** matches i,i+1 → .*, then * matches i+2 → [^/]*
        # Full pattern: /.*[^/]* which is effectively /.*
        regex = compile_glob("***")
        assert regex is not None
        assert regex.match("/foo") is not None
        assert regex.match("/a/b/c") is not None

    def test_double_star_dot_py(self):
        """**.py — ** without trailing / → .* then .py."""
        assert match_glob("/foo.py", "**.py") is True
        assert match_glob("/a/b/c.py", "**.py") is True
        assert match_glob("/foo.txt", "**.py") is False

    def test_very_long_path(self):
        """Very long paths should still work."""
        segments = "/".join(f"dir{i}" for i in range(100))
        path = f"/{segments}/file.py"
        assert match_glob(path, "**/*.py") is True
        assert match_glob(path, "*.py") is False

    def test_many_stars(self):
        """Pattern with many * characters."""
        assert match_glob("/a/b/c/d.py", "*/*/*/*.*") is True
        assert match_glob("/a/b/c.py", "*/*/*/*.*") is False  # only 3 segments

    def test_empty_segments_in_pattern(self):
        """Patterns with // may or may not match depending on implementation."""
        # The pattern is compiled as-is; // creates an empty segment
        regex = compile_glob("src//foo.py")
        assert regex is not None

    def test_star_star_slash_star_star(self):
        """**/** pattern."""
        assert match_glob("/a/b/c", "**/**") is True
        assert match_glob("/a", "**/**") is True

    def test_only_question_marks(self):
        """Pattern of just ? chars."""
        assert match_glob("/abc", "???") is True
        assert match_glob("/ab", "???") is False
        assert match_glob("/abcd", "???") is False

    def test_only_stars(self):
        """Pattern of just * (single star)."""
        assert match_glob("/foo", "*") is True
        assert match_glob("/a/b", "*") is False

    def test_consecutive_single_stars(self):
        """* * with something between them."""
        assert match_glob("/foobar", "*bar") is True
        assert match_glob("/foobar", "foo*") is True

    def test_pattern_with_backslash(self):
        """Backslashes should be treated as literal by regex escaping."""
        # A single backslash in pattern matches a single backslash in path
        assert match_glob("/foo\\bar", "foo\\bar") is True
        assert match_glob("/foo\\bar", "*\\*") is True

    def test_bracket_with_dash_at_edges(self):
        """Dash at start or end of bracket is literal."""
        assert match_glob("/-", "[-a]") is True
        assert match_glob("/a", "[-a]") is True
        assert match_glob("/b", "[-a]") is False

    def test_bracket_with_special_chars(self):
        """Brackets containing regex-special characters."""
        assert match_glob("/.", "[.]") is True
        assert match_glob("/a", "[.]") is False


# =========================================================================
# match_glob return type
# =========================================================================


class TestMatchGlobReturnType:
    """Verify match_glob always returns bool."""

    def test_returns_bool_true(self):
        result = match_glob("/foo.py", "*.py")
        assert result is True
        assert isinstance(result, bool)

    def test_returns_bool_false(self):
        result = match_glob("/foo.py", "*.txt")
        assert result is False
        assert isinstance(result, bool)


# =========================================================================
# compile_glob edge cases
# =========================================================================


class TestCompileGlobEdgeCases:
    def test_empty_pattern_compiles(self):
        regex = compile_glob("")
        assert regex is not None
        assert regex.match("/") is not None

    def test_slash_only_pattern(self):
        regex = compile_glob("/")
        assert regex is not None
        assert regex.match("/") is not None
        assert regex.match("/foo") is None

    def test_anchored_regex(self):
        """Compiled regex should be anchored at start and end."""
        regex = compile_glob("*.py")
        assert regex is not None
        # Should not match if path has extra content
        assert regex.match("/foo.py/bar") is None

    def test_character_class_pattern(self):
        regex = compile_glob("[abc].py")
        assert regex is not None


# =========================================================================
# glob_to_sql_like edge cases
# =========================================================================


class TestGlobToSqlLikeEdgeCases:
    def test_empty_pattern(self):
        result = glob_to_sql_like("")
        assert result == "/"

    def test_slash_only(self):
        result = glob_to_sql_like("/")
        assert result == "/"

    def test_double_star_at_start(self):
        result = glob_to_sql_like("**")
        assert result == "/%"

    def test_no_glob_chars(self):
        result = glob_to_sql_like("src/foo.py")
        assert result == "/src/foo.py"

    def test_absolute_pattern(self):
        result = glob_to_sql_like("/src/foo.py")
        assert result == "/src/foo.py"

    def test_multiple_percent_escapes(self):
        result = glob_to_sql_like("100%_file_%name")
        assert result == "/100\\%\\_file\\_\\%name"

    def test_double_star_then_single_star(self):
        """**/* should produce /%% → /% (after ** consumes trailing /)."""
        result = glob_to_sql_like("**/*")
        assert result == "/%%"


# =========================================================================
# Interaction between * and ** in same pattern
# =========================================================================


class TestStarInteraction:
    """Patterns mixing * and ** in various positions."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # */* — two single-star segments
            ("*/*", "/a/b", True),
            ("*/*", "/a/b/c", False),
            # */** — single segment then anything
            ("*/**", "/a/b", True),
            ("*/**", "/a/b/c/d", True),
            # **/* — anything then single segment
            ("**/*", "/a", True),
            ("**/*", "/a/b", True),
            ("**/*", "/a/b/c", True),
            # */*/* — exactly three segments
            ("*/*/*", "/a/b/c", True),
            ("*/*/*", "/a/b", False),
            ("*/*/*", "/a/b/c/d", False),
            # **/*/* — any depth then two segments
            ("**/*/*", "/a/b", True),
            ("**/*/*", "/a/b/c", True),
            ("**/*/*", "/x/y/a/b/c", True),
        ],
    )
    def test_star_interaction(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Boundary cases for ** with /
# =========================================================================


class TestDoubleStarSlashBehavior:
    """Detailed tests for how ** interacts with / boundaries."""

    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            # **/ at start — zero or more directory levels
            ("**/foo", "/foo", True),  # zero levels
            ("**/foo", "/a/foo", True),  # one level
            ("**/foo", "/a/b/foo", True),  # two levels
            # ** at very end matches any remaining
            ("src/**", "/src/a", True),
            ("src/**", "/src/a/b", True),
            ("src/**", "/src/a/b/c", True),
            # ** between segments
            ("a/**/b", "/a/b", True),
            ("a/**/b", "/a/x/b", True),
            ("a/**/b", "/a/x/y/b", True),
            ("a/**/b", "/b", False),  # needs leading a
            ("a/**/b", "/a/b/c", False),  # trailing c
        ],
    )
    def test_double_star_boundary(self, pattern: str, path: str, expected: bool):
        assert match_glob(path, pattern) is expected


# =========================================================================
# Helpers
# =========================================================================


def _like_to_regex(like: str) -> re.Pattern[str]:
    """Convert a SQL LIKE pattern to a regex for testing purposes.

    Handles:
    - ``%`` → ``.*``
    - ``_`` → ``.``
    - ``\\%`` → literal ``%``
    - ``\\_`` → literal ``_``
    """
    result = ""
    i = 0
    while i < len(like):
        ch = like[i]
        if ch == "\\" and i + 1 < len(like) and like[i + 1] in ("%", "_"):
            result += re.escape(like[i + 1])
            i += 2
            continue
        if ch == "%":
            result += ".*"
        elif ch == "_":
            result += "."
        else:
            result += re.escape(ch)
        i += 1
    return re.compile("^" + result + "$")


# ===========================================================================
# compile_glob / match_glob — invalid regex fallback
# ===========================================================================


class TestInvalidGlobPattern:
    def test_compile_glob_returns_none_on_bad_regex(self):
        from grover.patterns import compile_glob

        assert compile_glob("[z-a]") is None

    def test_match_glob_returns_false_on_bad_regex(self):
        from grover.patterns import match_glob

        assert match_glob("/a.py", "[z-a]") is False

"""Glob pattern matching — SQL LIKE translation and regex-based glob matching."""

from __future__ import annotations

import re

from grover.util.paths import normalize_path


def glob_to_sql_like(pattern: str, base_path: str = "/") -> str | None:
    """Translate a glob pattern to a SQL LIKE clause for pre-filtering.

    Returns None for patterns containing ``[seq]`` character classes,
    which cannot be expressed in LIKE. The caller should fall back to
    loading all paths and filtering with ``match_glob()``.

    This is a performance optimisation only — ``match_glob()`` is the
    authoritative filter.
    """
    if "[" in pattern:
        return None

    # Normalise base so we can prepend it
    base_path = normalize_path(base_path)

    # Build the full virtual pattern
    if pattern.startswith("/"):
        full = pattern
    elif base_path == "/":
        full = "/" + pattern
    else:
        full = base_path + "/" + pattern

    # Translate glob tokens → LIKE tokens
    like = ""
    i = 0
    while i < len(full):
        ch = full[i]
        if ch == "*":
            # ** → % (any depth), * → match within one segment
            if i + 1 < len(full) and full[i + 1] == "*":
                like += "%"
                i += 2
                # Skip trailing /
                if i < len(full) and full[i] == "/":
                    i += 1
                continue
            # Single * — we still use % here because LIKE has no
            # single-segment wildcard.  match_glob post-filters.
            like += "%"
        elif ch == "?":
            like += "_"
        elif ch == "%":
            like += "\\%"
        elif ch == "_":
            like += "\\_"
        else:
            like += ch
        i += 1

    return like


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern to a compiled regex.

    - ``*`` matches any characters except ``/``
    - ``**`` matches any characters including ``/`` (zero or more path segments)
    - ``?`` matches a single character except ``/``
    - ``[seq]`` matches any character in *seq*
    """
    result = ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # **/ → zero or more directory levels
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    result += "(?:.*/)?"
                    i += 3
                else:
                    result += ".*"
                    i += 2
                continue
            result += "[^/]*"
        elif ch == "?":
            result += "[^/]"
        elif ch == "[":
            j = i + 1
            while j < len(pattern) and pattern[j] != "]":
                j += 1
            if j >= len(pattern):
                # Unclosed bracket — treat [ as literal
                result += re.escape(ch)
            else:
                bracket = pattern[i : j + 1]
                # Translate glob negation [!...] to regex negation [^...]
                if bracket.startswith("[!"):
                    bracket = "[^" + bracket[2:]
                result += bracket
                i = j
        else:
            result += re.escape(ch)
        i += 1
    return re.compile("^" + result + "$")


def compile_glob(pattern: str, base_path: str = "/") -> re.Pattern[str] | None:
    """Compile a glob *pattern* into a regex for repeated matching.

    Returns ``None`` if the pattern is malformed.  Use the returned
    regex with ``regex.match(path) is not None`` to test individual
    paths efficiently without recompiling.
    """
    base_path = normalize_path(base_path)

    if pattern.startswith("/"):
        full_pattern = pattern
    elif base_path == "/":
        full_pattern = "/" + pattern
    else:
        full_pattern = base_path + "/" + pattern

    try:
        return _glob_to_regex(full_pattern)
    except re.error:
        return None


def match_glob(path: str, pattern: str, base_path: str = "/") -> bool:
    """Authoritative glob match for a virtual path against *pattern*.

    Handles ``*``, ``?``, ``[seq]``, and ``**`` (recursive).
    Uses a regex translation that correctly prevents ``*`` from
    crossing directory boundaries while allowing ``**`` to match
    across any number of path segments.

    For matching many paths against the same pattern, use
    :func:`compile_glob` to avoid repeated regex compilation.
    """
    regex = compile_glob(pattern, base_path)
    if regex is None:
        return False
    return regex.match(path) is not None

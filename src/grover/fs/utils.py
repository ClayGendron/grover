"""Path utilities, text replacement, binary detection."""

from __future__ import annotations

import posixpath


def normalize_path(path: str) -> str:
    """Normalize a virtual file system path.

    - Ensures leading /
    - Resolves .. and . references
    - Removes double slashes
    - Removes trailing slash (except for root)

    Examples:
        normalize_path("foo.txt") -> "/foo.txt"
        normalize_path("/foo//bar.txt") -> "/foo/bar.txt"
        normalize_path("/foo/../bar.txt") -> "/bar.txt"
        normalize_path("/foo/") -> "/foo"
        normalize_path("") -> "/"
    """
    if not path:
        return "/"

    path = path.strip()

    if not path.startswith("/"):
        path = "/" + path

    path = posixpath.normpath(path)

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return path


def split_path(path: str) -> tuple[str, str]:
    """Split path into (parent_dir, filename).

    Examples:
        split_path("/foo/bar.txt") -> ("/foo", "bar.txt")
        split_path("/foo.txt") -> ("/", "foo.txt")
        split_path("/") -> ("/", "")
    """
    path = normalize_path(path)
    if path == "/":
        return ("/", "")
    return posixpath.split(path)

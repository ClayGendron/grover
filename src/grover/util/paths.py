"""Path utilities — normalization, validation, trash/shared namespace helpers."""

from __future__ import annotations

import posixpath
import unicodedata

# =============================================================================
# Shared Namespace
# =============================================================================

SHARED_SEGMENT = "@shared"

# Reserved filenames (Windows compatibility)
RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


# =============================================================================
# Path Utilities
# =============================================================================


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

    path = unicodedata.normalize("NFC", path)
    path = path.strip()

    if not path.startswith("/"):
        path = "/" + path

    path = posixpath.normpath(path)

    if path != "/" and path.endswith("/"):  # pragma: no cover - normpath already strips
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
        return "/", ""
    return posixpath.split(path)


def validate_path(path: str) -> tuple[bool, str]:
    """Validate a path for security and compatibility issues.

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    if "\x00" in path:
        return False, "Path contains null bytes"

    # Reject ASCII control characters (0x01-0x1f) except \t, \n, \r
    for ch in path:
        code = ord(ch)
        if 0x01 <= code <= 0x1F and ch not in ("\t", "\n", "\r"):
            return False, f"Path contains control character: 0x{code:02x}"

    if len(path) > 4096:
        return False, "Path too long (max 4096 characters)"

    path = normalize_path(path)
    _, name = split_path(path)

    if name and len(name) > 255:
        return False, "Filename too long (max 255 characters)"

    if name:
        name_upper = name.upper()
        base_name = name_upper.split(".")[0] if "." in name_upper else name_upper
        if base_name in RESERVED_NAMES:
            return False, f"Reserved filename: {name}"

    if is_shared_path(path):
        return False, "Path contains reserved '@shared' segment"

    return True, ""


def is_shared_path(path: str) -> bool:
    """Check if a path contains the ``@shared`` virtual namespace segment."""
    normalized = normalize_path(path)
    segments = normalized.split("/")
    return SHARED_SEGMENT in segments


def is_trash_path(path: str) -> bool:
    """Check if a path is in the trash namespace."""
    return path.startswith("/__trash__/")


def to_trash_path(path: str, file_id: str) -> str:
    """Convert a path to its trash namespace equivalent."""
    path = normalize_path(path)
    return f"/__trash__/{file_id}{path}"


def from_trash_path(trash_path: str) -> str:
    """Extract the original path from a trash namespace path."""
    if not is_trash_path(trash_path):
        return trash_path

    # Format: /__trash__/{uuid}/original/path
    rest = trash_path[len("/__trash__/") :]
    slash_idx = rest.find("/")
    if slash_idx == -1:
        return "/"
    return rest[slash_idx:]

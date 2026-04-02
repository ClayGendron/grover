"""Path utilities for Grover's dot-prefix metadata namespace.

Grover uses dot-prefixed directories (.chunks/, .versions/, .connections/, .apis/)
to organize metadata as children of their parent file. These utilities handle
path normalization, kind detection, parent resolution, and path construction
for the unified namespace.

Path conventions:

    /src/auth.py                                    file
    /src/auth.py/.chunks/login                      chunk
    /src/auth.py/.versions/3                        version
    /src/auth.py/.connections/imports/src/utils.py   connection
    /jira/.apis/ticket                               api endpoint
"""

from __future__ import annotations

import posixpath
import unicodedata
from typing import Literal, NamedTuple


class ConnectionParts(NamedTuple):
    source: str
    target: str
    connection_type: str


ObjectKind = Literal["file", "directory", "chunk", "version", "connection", "api"]
MetadataKind = Literal[".chunks", ".versions", ".connections", ".apis"]

METADATA_KIND_MAP: dict[MetadataKind, ObjectKind] = {
    ".chunks": "chunk",
    ".versions": "version",
    ".connections": "connection",
    ".apis": "api",
}

MARKER_KINDS: dict[str, ObjectKind] = {f"/{name}/": kind for name, kind in METADATA_KIND_MAP.items()}

METADATA_MARKERS = tuple(MARKER_KINDS.keys())

EXTENSIONLESS_FILES = frozenset(
    {
        # Build / CI
        "Makefile",
        "GNUmakefile",
        "BSDmakefile",
        "Kbuild",
        "Dockerfile",
        "Containerfile",
        "Jenkinsfile",
        "Vagrantfile",
        "Procfile",
        "Justfile",
        "Taskfile",
        "Earthfile",
        "Snakefile",
        "Tiltfile",
        "Caddyfile",
        "BUILD",
        "WORKSPACE",
        # Ruby ecosystem
        "Rakefile",
        "Gemfile",
        "Brewfile",
        "Podfile",
        "Fastfile",
        "Appfile",
        "Scanfile",
        "Berksfile",
        "Capfile",
        "Guardfile",
        "Thorfile",
        "Dangerfile",
        "Steepfile",
        "Appraisals",
        # Other language build files
        "Emakefile",
        "Nukefile",
        # Documentation / metadata
        "LICENSE",
        "LICENCE",
        "README",
        "CHANGELOG",
        "CHANGES",
        "CONTRIBUTING",
        "AUTHORS",
        "CONTRIBUTORS",
        "PATENTS",
        "NOTICE",
        "CREDITS",
        "HISTORY",
        "NEWS",
        "THANKS",
        "TODO",
        "COPYING",
        "COPYRIGHT",
        "INSTALL",
        "CODEOWNERS",
        "MAINTAINERS",
        "OWNERS",
        "VERSION",
        "MANIFEST",
        # Git
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".gitkeep",
        ".git-blame-ignore-revs",
        ".mailmap",
        # Container / orchestration
        ".dockerignore",
        ".helmignore",
        # Editor / formatting
        ".editorconfig",
        ".clang-format",
        ".clang-tidy",
        ".dir-locals",
        # JS / Node
        ".eslintignore",
        ".eslintrc",
        ".prettierignore",
        ".prettierrc",
        ".stylelintignore",
        ".stylelintrc",
        ".babelrc",
        ".browserslistrc",
        ".npmignore",
        ".npmrc",
        ".nvmrc",
        ".node-version",
        # Python
        ".python-version",
        ".flaskenv",
        # Ruby / other version managers
        ".ruby-version",
        ".java-version",
        ".tool-versions",
        # Environment
        ".env",
        ".envrc",
        # YAML linting
        ".yamllint",
    }
)

_EXTENSIONLESS_FILES_LOWER = frozenset(name.lower() for name in EXTENSIONLESS_FILES)


# ---------------------------------------------------------------------------
# Normalization and validation
# ---------------------------------------------------------------------------


def normalize_path(path: str) -> str:
    """Normalize a virtual filesystem path.

    Ensures leading ``/``, resolves ``..`` and ``.``, removes double slashes
    and trailing slashes (except root).
    """
    if not path:
        return "/"
    path = unicodedata.normalize("NFC", path).strip()
    if not path.startswith("/"):
        path = "/" + path
    path = posixpath.normpath(path)
    return path


def split_path(path: str) -> tuple[str, str]:
    """Split a normalized path into ``(directory, name)``.

    >>> split_path("/src/auth.py")
    ('/src', 'auth.py')
    >>> split_path("/")
    ('/', '')
    """
    path = normalize_path(path)
    if path == "/":
        return "/", ""
    return posixpath.split(path)


def validate_path(path: str) -> tuple[bool, str]:
    """Validate structural correctness of a path.

    Checks (in order):

    1. Null bytes
    2. Control characters (0x01-0x1F, DEL 0x7F, C1 0x80-0x9F)
    3. Total path length (max 4096)
    4. Empty segments after normalization
    5. Segment length (max 255 per segment)

    Returns ``(is_valid, error_message)``.  ``error_message`` is empty when
    valid.

    Note: reserved metadata name protection (``.chunks``, ``.versions``, etc.)
    is enforced at the operation level (``write``, ``mkdir``), not here —
    the same path may be valid for reads but not for writes.
    """
    if "\x00" in path:
        return False, "Path contains null bytes"

    for ch in path:
        code = ord(ch)
        if (0x01 <= code <= 0x1F) or code == 0x7F or (0x80 <= code <= 0x9F):
            return False, f"Path contains control character: U+{code:04X}"

    if len(path) > 4096:
        return False, "Path too long (max 4096 characters)"

    normalized = normalize_path(path)
    if normalized == "/":
        return True, ""

    segments = normalized.split("/")[1:]

    for segment in segments:
        if len(segment) > 255:
            return False, f"Path segment too long (max 255): '{segment[:40]}...'"

    return True, ""


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------


def parse_kind(path: str) -> ObjectKind:
    """Detect the entity kind from a path.

    Resolution order:

    1. Metadata markers (``.chunks/``, ``.versions/``, ``.connections/``,
       ``.apis/``) → ``"chunk"``, ``"version"``, ``"connection"``, ``"api"``
    2. Dot-prefixed name (dotfile) → ``"file"`` (unless reserved metadata name)
    3. Has a file extension (e.g. ``.py``, ``.md``) → ``"file"``
    4. Known extensionless file, case-insensitive (``Makefile``, ``LICENSE``, …)
       → ``"file"``
    5. Otherwise → ``"directory"``
    """
    for marker, kind in MARKER_KINDS.items():
        if marker in path:
            return kind

    _, name = split_path(path)
    if not name:
        return "directory"

    # Dotfiles: names starting with "." are files, unless they're
    # reserved metadata directory names (.chunks, .versions, etc.)
    if name.startswith("."):
        if name in METADATA_KIND_MAP:
            return "directory"
        return "file"

    # Has a file extension (dot after position 0)
    dot = name.rfind(".")
    if dot > 0:
        return "file"

    # Known extensionless file (case-insensitive)
    if name.lower() in _EXTENSIONLESS_FILES_LOWER:
        return "file"

    return "directory"


# ---------------------------------------------------------------------------
# Parent and base path resolution
# ---------------------------------------------------------------------------


def base_path(path: str) -> str:
    """Return the owning file for a metadata path, or *path* itself for files.

    >>> base_path("/src/auth.py/.chunks/login")
    '/src/auth.py'
    >>> base_path("/src/auth.py/.chunks")
    '/src/auth.py'
    >>> base_path("/src/auth.py")
    '/src/auth.py'
    """
    # Full marker match: /.chunks/login → /src/auth.py
    for marker in METADATA_MARKERS:
        idx = path.find(marker)
        if idx >= 0:
            return path[:idx]
    # Bare metadata dir: /.chunks (no child) → /src/auth.py
    _, name = split_path(path)
    if name in METADATA_KIND_MAP:
        return split_path(path)[0]
    return path


def parent_path(path: str) -> str:
    """Compute the parent path for DB storage.

    For metadata nodes the parent is the owning file (marker-aware).
    For files and directories the parent is the standard directory parent.

    >>> parent_path("/src/auth.py/.chunks/login")
    '/src/auth.py'
    >>> parent_path("/src/auth.py/.connections/imports/src/utils.py")
    '/src/auth.py'
    >>> parent_path("/src/auth.py")
    '/src'
    >>> parent_path("/src")
    '/'
    """
    # Metadata nodes: parent is everything before the first marker.
    for marker in METADATA_MARKERS:
        idx = path.find(marker)
        if idx >= 0:
            return path[:idx]
    # Files and directories: standard filesystem parent.
    return split_path(path)[0]


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def _validate_base(path: str) -> str:
    """Normalize *path* and reject metadata paths as construction bases."""
    normalized = normalize_path(path)
    for marker in METADATA_MARKERS:
        if marker in normalized:
            msg = f"Base path must not be a metadata path: {path}"
            raise ValueError(msg)
    _, name = split_path(normalized)
    if name in METADATA_KIND_MAP:
        msg = f"Base path must not end with reserved name '{name}': {path}"
        raise ValueError(msg)
    return normalized


def _validate_name(value: str, label: str) -> str:
    """Validate that *value* is a non-empty, single-segment name."""
    if not value or not value.strip():
        msg = f"{label} must not be empty"
        raise ValueError(msg)
    if "/" in value:
        msg = f"{label} must not contain '/': {value}"
        raise ValueError(msg)
    return value


def chunk_path(file_path: str, chunk_name: str) -> str:
    """Build a chunk path.

    >>> chunk_path("/src/auth.py", "login")
    '/src/auth.py/.chunks/login'

    Raises ``ValueError`` if *file_path* is a metadata path or
    *chunk_name* is empty or contains ``/``.
    """
    base = _validate_base(file_path)
    _validate_name(chunk_name, "chunk_name")
    return f"{base}/.chunks/{chunk_name}"


def version_path(file_path: str, version_number: int) -> str:
    """Build a version path.

    >>> version_path("/src/auth.py", 3)
    '/src/auth.py/.versions/3'

    Raises ``ValueError`` if *version_number* < 1 or *file_path* is a
    metadata path.
    """
    base = _validate_base(file_path)
    if version_number < 1:
        msg = f"version_number must be >= 1, got {version_number}"
        raise ValueError(msg)
    return f"{base}/.versions/{version_number}"


def connection_path(source: str, target: str, connection_type: str) -> str:
    """Build a connection path.

    >>> connection_path("/src/auth.py", "/src/utils.py", "imports")
    '/src/auth.py/.connections/imports/src/utils.py'

    Both *source* and *target* are normalized.  Raises ``ValueError`` if
    *connection_type* is empty or contains ``/``, or if *target* is root.
    """
    src = _validate_base(source)
    _validate_name(connection_type, "connection_type")
    tgt = normalize_path(target)
    if tgt == "/":
        msg = "target must not be root"
        raise ValueError(msg)
    return f"{src}/.connections/{connection_type}/{tgt.lstrip('/')}"


def api_path(mount: str, action: str) -> str:
    """Build an API endpoint path.

    >>> api_path("/jira", "ticket")
    '/jira/.apis/ticket'

    Raises ``ValueError`` if *action* is empty or contains ``/``, or if
    *mount* is a metadata path.
    """
    base = _validate_base(mount)
    _validate_name(action, "action")
    return f"{base}/.apis/{action}"


# ---------------------------------------------------------------------------
# Path decomposition
# ---------------------------------------------------------------------------


def decompose_connection(path: str) -> ConnectionParts | None:
    """Extract source, target, and connection type from a connection path.

    Returns ``None`` if *path* is not a valid connection path.

    >>> decompose_connection("/src/auth.py/.connections/imports/src/utils.py")
    ConnectionParts(source='/src/auth.py', target='/src/utils.py', connection_type='imports')
    """
    marker = "/.connections/"
    idx = path.find(marker)
    if idx < 0:
        return None
    source = path[:idx]
    rest = path[idx + len(marker) :]
    slash = rest.find("/")
    if slash < 0:
        return None
    connection_type = rest[:slash]
    target = "/" + rest[slash + 1 :]
    return ConnectionParts(source=source, target=target, connection_type=connection_type)


# ---------------------------------------------------------------------------
# User scoping
# ---------------------------------------------------------------------------

_UNSAFE_USER_ID_CHARS = frozenset("/\\@\0")


def validate_user_id(user_id: str) -> tuple[bool, str]:
    """Validate that *user_id* is safe for use as a path segment.

    Rejects empty strings, strings containing ``/``, ``\\``, ``@``,
    null bytes, the ``..`` traversal sequence, and strings longer than
    255 characters.

    Returns ``(is_valid, error_message)``.  ``error_message`` is empty
    when valid.
    """
    if not user_id or not user_id.strip():
        return False, "user_id must not be empty"
    if len(user_id) > 255:
        return False, "user_id too long (max 255 characters)"
    if ".." in user_id:
        return False, "user_id must not contain '..'"
    for ch in user_id:
        if ch in _UNSAFE_USER_ID_CHARS:
            return False, f"user_id contains unsafe character: {ch!r}"
    return True, ""


def scope_path(path: str, user_id: str) -> str:
    """Prepend ``/{user_id}`` to *path* for per-user storage.

    Always prepends — callers must pass unscoped paths.

    >>> scope_path("/docs/README.md", "123")
    '/123/docs/README.md'
    >>> scope_path("/", "123")
    '/123'
    """
    valid, err = validate_user_id(user_id)
    if not valid:
        msg = f"Invalid user_id: {err}"
        raise ValueError(msg)
    path = normalize_path(path)
    if path == "/":
        return f"/{user_id}"
    return f"/{user_id}{path}"


def unscope_path(path: str, user_id: str) -> str:
    """Strip the ``/{user_id}`` prefix from a storage path.

    For connection paths, both the source prefix and the embedded target
    prefix are stripped.

    >>> unscope_path("/123/docs/README.md", "123")
    '/docs/README.md'
    >>> unscope_path("/123", "123")
    '/'
    """
    prefix = f"/{user_id}"

    # Connection paths: unscope both source and target portions.
    parts = decompose_connection(path)
    if parts is not None:
        source = _strip_user_prefix(parts.source, prefix)
        target = _strip_user_prefix(parts.target, prefix)
        return connection_path(source, target, parts.connection_type)

    return _strip_user_prefix(path, prefix)


def _strip_user_prefix(path: str, prefix: str) -> str:
    """Strip *prefix* from the start of *path*.

    Raises ``ValueError`` if *path* does not start with *prefix*.
    """
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        return path[len(prefix):]
    msg = f"Path {path!r} does not start with user prefix {prefix!r}"
    raise ValueError(msg)

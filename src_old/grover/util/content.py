"""Content utilities — hashing, text/binary detection, MIME types, read output formatting."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from grover.util.paths import split_path

if TYPE_CHECKING:
    from grover.models.internal.results import FileOperationResult


# =============================================================================
# Content Hashing
# =============================================================================


def compute_content_hash(content: str) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for *content*."""
    encoded = content.encode()
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


# =============================================================================
# Text File Extensions (allowed for write operations)
# =============================================================================

TEXT_EXTENSIONS = {
    # Programming languages
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".m",
    ".mm",
    ".pl",
    ".pm",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".bat",
    ".cmd",
    ".vbs",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    # Data/Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".xml",
    ".csv",
    ".tsv",
    # Documentation
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".text",
    ".asciidoc",
    ".adoc",
    # SQL
    ".sql",
    ".ddl",
    ".dml",
    # Other text formats
    ".log",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".eslintrc",
    ".prettierrc",
    ".babelrc",
    ".npmrc",
    ".nvmrc",
    ".makefile",
    ".dockerfile",
    ".tf",
    ".tfvars",
    ".hcl",
    ".graphql",
    ".gql",
    ".proto",
}

# Files without extensions that are text
TEXT_FILENAMES = {
    "Makefile",
    "Dockerfile",
    "Jenkinsfile",
    "Vagrantfile",
    "Procfile",
    "Gemfile",
    "Rakefile",
    "Brewfile",
    "Podfile",
    "Fastfile",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
    "LICENSE",
    "README",
    "CHANGELOG",
    "CONTRIBUTING",
    "AUTHORS",
}

# Binary file extensions that should not be read
BINARY_EXTENSIONS = {
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".class",
    ".jar",
    ".war",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".bin",
    ".dat",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".wasm",
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".tiff",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".flac",
    ".pdf",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
}


# =============================================================================
# File Detection
# =============================================================================


def is_text_file(filename: str) -> bool:
    """Check if a file is a text file based on extension or name."""
    name = Path(filename).name
    ext = Path(filename).suffix.lower()

    if ext and ext in TEXT_EXTENSIONS:
        return True

    if name in TEXT_FILENAMES:
        return True

    return bool(name.startswith(".") and not ext)


def guess_mime_type(filename: str) -> str:
    """Guess the MIME type of a file based on its name."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "text/plain"


def has_binary_extension(file_path: str) -> bool:
    """Check if a virtual path has a known binary file extension."""
    name = split_path(file_path)[1]
    ext_suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return ext_suffix.lower() in BINARY_EXTENSIONS


def is_binary_file(file_path: str | Path) -> bool:
    """Check if a file is binary based on extension and content.

    Uses two-stage detection:
    1. Check known binary extensions (fast)
    2. Analyze file content for binary indicators (null bytes, non-printable chars)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in BINARY_EXTENSIONS:
        return True

    try:
        with path.open("rb") as f:
            chunk = f.read(4096)

        if not chunk:
            return False

        if b"\x00" in chunk:
            return True

        non_printable = sum(1 for byte in chunk if byte < 9 or (13 < byte < 32))

        return (non_printable / len(chunk)) > 0.3

    except Exception:
        return False


def get_similar_files(
    directory: str | Path,
    filename: str,
    max_suggestions: int = 3,
) -> list[str]:
    """Find similar filenames in a directory for suggestions."""
    try:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []

        filename_lower = filename.lower()

        suggestions = [
            str(entry)
            for entry in dir_path.iterdir()
            if filename_lower in entry.name.lower() or entry.name.lower() in filename_lower
        ]

        return suggestions[:max_suggestions]
    except Exception:
        return []


# =============================================================================
# Read Output Formatting (for LLM display)
# =============================================================================


def format_read_output(result: FileOperationResult) -> str:
    """Format a FileOperationResult with line numbers and ``<file>`` wrapper for LLM display.

    The *result* should contain raw (unformatted) content from
    a backend's ``read()`` method.  This function adds zero-padded line numbers
    and wraps the output in ``<file>...</file>`` tags.
    """
    # Support both new FileOperationResult (content on .file) and legacy ReadResult
    content: str | None
    if hasattr(result, "file") and hasattr(result.file, "content"):
        content = result.file.content
    else:
        content = getattr(result, "content", None)
    if not content:
        return "<file>\n(empty file)\n</file>"

    lines = content.split("\n")

    formatted_lines = [f"{str(i + 1).zfill(5)}| {line}" for i, line in enumerate(lines)]

    formatted = "<file>\n"
    formatted += "\n".join(formatted_lines)
    formatted += f"\n\n(End of file - total {len(lines)} lines)"
    formatted += "\n</file>"
    return formatted

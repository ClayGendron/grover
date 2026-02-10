"""Path utilities, text replacement, binary detection."""

from __future__ import annotations

import mimetypes
import posixpath
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ReadResult

# =============================================================================
# Text File Extensions (allowed for write operations)
# =============================================================================

TEXT_EXTENSIONS = {
    # Programming languages
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".m", ".mm", ".pl", ".pm", ".lua", ".sh", ".bash", ".zsh", ".fish",
    ".ps1", ".psm1", ".bat", ".cmd", ".vbs",
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    # Data/Config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".csv", ".tsv",
    # Documentation
    ".md", ".markdown", ".rst", ".txt", ".text", ".asciidoc", ".adoc",
    # SQL
    ".sql", ".ddl", ".dml",
    # Other text formats
    ".log", ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig",
    ".eslintrc", ".prettierrc", ".babelrc", ".npmrc", ".nvmrc",
    ".makefile", ".dockerfile", ".tf", ".tfvars", ".hcl",
    ".graphql", ".gql", ".proto",
}

# Files without extensions that are text
TEXT_FILENAMES = {
    "Makefile", "Dockerfile", "Jenkinsfile", "Vagrantfile", "Procfile",
    "Gemfile", "Rakefile", "Brewfile", "Podfile", "Fastfile",
    ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig",
    ".env", ".env.local", ".env.development", ".env.production",
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "package.json", "tsconfig.json", "jsconfig.json",
    "LICENSE", "README", "CHANGELOG", "CONTRIBUTING", "AUTHORS",
}

# Reserved filenames (Windows compatibility)
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

# Binary file extensions that should not be read
BINARY_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".class", ".jar", ".war",
    ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
    ".odp", ".bin", ".dat", ".obj", ".o", ".a", ".lib", ".wasm", ".pyc", ".pyo",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".eot",
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

    return True, ""


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
    rest = trash_path[len("/__trash__/"):]
    slash_idx = rest.find("/")
    if slash_idx == -1:
        return "/"
    return rest[slash_idx:]


# =============================================================================
# Binary File Detection
# =============================================================================


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

        non_printable = sum(
            1 for byte in chunk
            if byte < 9 or (13 < byte < 32)
        )

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
# Text Replacement (Smart Edit)
# =============================================================================


@dataclass
class Match:
    """Structured match result from a replacer."""

    start: int
    end: int
    text: str
    method: str
    confidence: float


@dataclass
class ReplaceResult:
    """Result of a replace operation."""

    success: bool
    content: str | None = None
    error: str | None = None
    matches: list[Match] | None = None
    method_used: str | None = None


Replacer = Callable[[str, str], Generator[Match]]


def normalize_line_endings(text: str) -> str:
    """Convert Windows line endings to Unix."""
    return text.replace("\r\n", "\n")


def levenshtein(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if a == "" or b == "":
        return max(len(a), len(b))

    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    return matrix[len(a)][len(b)]


def get_line_number(content: str, position: int) -> int:
    """Get 1-indexed line number for a position in content."""
    return content[:position].count("\n") + 1


def get_context_lines(content: str, start: int, end: int, context: int = 3) -> str:
    """Get matched text with surrounding context lines."""
    lines = content.split("\n")
    start_line = get_line_number(content, start) - 1  # 0-indexed
    end_line = get_line_number(content, end) - 1

    context_start = max(0, start_line - context)
    context_end = min(len(lines), end_line + context + 1)

    result_lines = []
    for i in range(context_start, context_end):
        prefix = ">" if start_line <= i <= end_line else " "
        result_lines.append(f"{i + 1:4d} {prefix} {lines[i]}")

    return "\n".join(result_lines)


# -----------------------------------------------------------------------------
# Replacers
# -----------------------------------------------------------------------------


def simple_replacer(content: str, find: str) -> Generator[Match]:
    """Exact match replacer."""
    start = 0
    while True:
        index = content.find(find, start)
        if index == -1:
            break
        yield Match(
            start=index,
            end=index + len(find),
            text=find,
            method="exact",
            confidence=1.0,
        )
        start = index + len(find)


def line_trimmed_replacer(content: str, find: str) -> Generator[Match]:
    """Match lines after stripping whitespace from each line."""
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    if not find_lines:
        return

    for i in range(len(content_lines) - len(find_lines) + 1):
        matches = True
        for j in range(len(find_lines)):
            if content_lines[i + j].strip() != find_lines[j].strip():
                matches = False
                break

        if matches:
            start_pos = sum(len(content_lines[k]) + 1 for k in range(i))
            end_pos = start_pos
            for k in range(len(find_lines)):
                end_pos += len(content_lines[i + k])
                if k < len(find_lines) - 1:
                    end_pos += 1

            matched_text = "\n".join(content_lines[i : i + len(find_lines)])
            yield Match(
                start=start_pos,
                end=end_pos,
                text=matched_text,
                method="line_trimmed",
                confidence=0.9,
            )


# Thresholds for BlockAnchorReplacer
SINGLE_CANDIDATE_THRESHOLD = 0.6
MULTIPLE_CANDIDATES_THRESHOLD = 0.3


def block_anchor_replacer(content: str, find: str) -> Generator[Match]:
    """Match blocks using first/last lines as anchors with fuzzy middle matching."""
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    if len(find_lines) < 3:
        return

    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    if len(find_lines) < 3:
        return

    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    candidates: list[tuple[int, int]] = []
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                candidates.append((i, j))
                break

    if not candidates:
        return

    def calculate_similarity(start_line: int, end_line: int) -> float:
        actual_block_size = end_line - start_line + 1
        find_block_size = len(find_lines)
        lines_to_check = min(find_block_size - 2, actual_block_size - 2)

        if lines_to_check <= 0:
            return 1.0

        total_similarity = 0.0
        for j in range(1, min(find_block_size - 1, actual_block_size - 1)):
            content_line = content_lines[start_line + j].strip()
            find_line = find_lines[j].strip()
            max_len = max(len(content_line), len(find_line))
            if max_len == 0:
                continue
            distance = levenshtein(content_line, find_line)
            total_similarity += 1 - (distance / max_len)

        return total_similarity / lines_to_check

    def make_match(start_line: int, end_line: int, confidence: float) -> Match:
        start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))
        end_pos = start_pos
        for k in range(start_line, end_line + 1):
            end_pos += len(content_lines[k])
            if k < end_line:
                end_pos += 1

        matched_text = "\n".join(content_lines[start_line : end_line + 1])
        return Match(
            start=start_pos,
            end=end_pos,
            text=matched_text,
            method="block_anchor",
            confidence=confidence,
        )

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        similarity = calculate_similarity(start_line, end_line)
        if similarity >= SINGLE_CANDIDATE_THRESHOLD:
            yield make_match(start_line, end_line, similarity)
        return

    best_match = None
    best_similarity = -1.0

    for start_line, end_line in candidates:
        similarity = calculate_similarity(start_line, end_line)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (start_line, end_line)

    if best_similarity >= MULTIPLE_CANDIDATES_THRESHOLD and best_match:
        yield make_match(best_match[0], best_match[1], best_similarity)


# Replacers in priority order
REPLACERS: list[Replacer] = [
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
]


# -----------------------------------------------------------------------------
# Core Replace Function
# -----------------------------------------------------------------------------


def replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> ReplaceResult:
    """Replace old_string with new_string in content.

    Uses a three-level matching strategy:
    1. Exact match (confidence: 1.0)
    2. Line-trimmed match - ignores whitespace per line (confidence: 0.9)
    3. Block anchor match - matches first/last lines, fuzzy middle (confidence: 0.3-0.6)
    """
    if not old_string:
        return ReplaceResult(
            success=False,
            error="old_string cannot be empty. Use the write tool to create new files.",
        )

    if old_string == new_string:
        return ReplaceResult(
            success=False,
            error="old_string and new_string must be different.",
        )

    content = normalize_line_endings(content)
    old_string = normalize_line_endings(old_string)
    new_string = normalize_line_endings(new_string)

    for replacer in REPLACERS:
        matches = list(replacer(content, old_string))

        if not matches:
            continue

        method = matches[0].method
        is_exact = method == "exact"

        if replace_all and not is_exact:
            return ReplaceResult(
                success=False,
                error=(
                    f"replace_all=True is only allowed with exact matches. "
                    f"Found fuzzy match using '{method}' method."
                ),
            )

        if replace_all and is_exact:
            new_content = content.replace(old_string, new_string)
            return ReplaceResult(
                success=True,
                content=new_content,
                method_used=method,
                matches=matches,
            )

        if len(matches) == 1:
            match = matches[0]
            new_content = content[: match.start] + new_string + content[match.end :]
            return ReplaceResult(
                success=True,
                content=new_content,
                method_used=method,
                matches=[match],
            )

        match_info = []
        for m in matches:
            line_num = get_line_number(content, m.start)
            ctx = get_context_lines(content, m.start, m.end)
            match_info.append(f"Match at line {line_num}:\n{ctx}")

        return ReplaceResult(
            success=False,
            error=(
                f"Found {len(matches)} matches. "
                "Provide more context in old_string to identify a unique match.\n\n"
                + "\n\n---\n\n".join(match_info)
            ),
            matches=matches,
        )

    return ReplaceResult(
        success=False,
        error="old_string not found in file content.",
    )


# =============================================================================
# Read Output Formatting (for LLM display)
# =============================================================================


def format_read_output(result: ReadResult) -> str:
    """Format a ReadResult with line numbers and ``<file>`` wrapper for LLM display.

    The *result* should contain raw (unformatted) content from
    ``BaseFileSystem.read()``.  This function adds zero-padded line numbers
    and wraps the output in ``<file>...</file>`` tags, matching the format
    that was previously embedded in ``_format_read_output``.
    """
    if not result.content:
        return "<file>\n(empty file)\n</file>"

    lines = result.content.split("\n")
    offset = result.offset

    formatted_lines = [
        f"{str(i + offset + 1).zfill(5)}| {line}"
        for i, line in enumerate(lines)
    ]

    formatted = "<file>\n"
    formatted += "\n".join(formatted_lines)

    if result.truncated:
        last_read_line = offset + len(lines)
        formatted += (
            f"\n\n(File has more lines. "
            f"Use 'offset' parameter to read beyond line {last_read_line})"
        )
    else:
        formatted += f"\n\n(End of file - total {result.total_lines or len(lines)} lines)"

    formatted += "\n</file>"
    return formatted

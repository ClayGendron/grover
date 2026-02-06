"""
Shared utilities for the virtual file system.

Contains path normalization, validation, and text file detection
used by both LocalFileSystem and DatabaseFileSystem.
"""

import mimetypes
import posixpath
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

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
    """
    Normalize a virtual file system path.

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

    # Strip whitespace
    path = path.strip()

    # Ensure leading /
    if not path.startswith("/"):
        path = "/" + path

    # Normalize with posixpath (handles .., //, etc.)
    path = posixpath.normpath(path)

    # Remove trailing slash except for root
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return path


def split_path(path: str) -> tuple[str, str]:
    """
    Split path into (parent_dir, filename).

    Examples:
        split_path("/foo/bar.txt") -> ("/foo", "bar.txt")
        split_path("/foo.txt") -> ("/", "foo.txt")
        split_path("/") -> ("/", "")
    """
    path = normalize_path(path)
    if path == "/":
        return ("/", "")
    return posixpath.split(path)


def validate_path(path: str) -> tuple[bool, str]:
    """
    Validate a path for security and compatibility issues.

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    # Check for null bytes
    if "\x00" in path:
        return False, "Path contains null bytes"

    # Check path length
    if len(path) > 4096:
        return False, "Path too long (max 4096 characters)"

    # Normalize and check filename
    path = normalize_path(path)
    _, name = split_path(path)

    # Check filename length
    if name and len(name) > 255:
        return False, "Filename too long (max 255 characters)"

    # Check for reserved names (Windows compatibility)
    if name:
        name_upper = name.upper()
        # Handle extensions like "CON.txt"
        base_name = name_upper.split(".")[0] if "." in name_upper else name_upper
        if base_name in RESERVED_NAMES:
            return False, f"Reserved filename: {name}"

    return True, ""


def is_text_file(filename: str) -> bool:
    """
    Check if a file is a text file based on extension or name.

    Used to validate write operations (only text files allowed).
    """
    name = Path(filename).name
    ext = Path(filename).suffix.lower()

    # Check extension
    if ext and ext in TEXT_EXTENSIONS:
        return True

    # Check filename (for files without extensions or special names)
    if name in TEXT_FILENAMES:
        return True

    # Check if name starts with a dot and has a known config pattern
    return bool(name.startswith(".") and not ext)


def guess_mime_type(filename: str) -> str:
    """
    Guess the MIME type of a file based on its name.

    Returns "text/plain" as default for unknown types.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "text/plain"


def is_trash_path(path: str) -> bool:
    """Check if a path is in the trash namespace."""
    return path.startswith("/__trash__/")


def to_trash_path(path: str, file_id: str) -> str:
    """
    Convert a path to its trash namespace equivalent.

    The file_id is included to ensure uniqueness in trash.
    """
    path = normalize_path(path)
    return f"/__trash__/{file_id}{path}"


def from_trash_path(trash_path: str) -> str:
    """
    Extract the original path from a trash namespace path.

    This is a fallback - normally original_path field should be used.
    """
    if not is_trash_path(trash_path):
        return trash_path

    # Format: /__trash__/{uuid}/original/path
    # Find the UUID end (first / after /__trash__/)
    rest = trash_path[len("/__trash__/"):]
    slash_idx = rest.find("/")
    if slash_idx == -1:
        return "/"
    return rest[slash_idx:]


# =============================================================================
# Binary File Detection
# =============================================================================

def is_binary_file(file_path: str | Path) -> bool:
    """
    Check if a file is binary based on extension and content.

    Uses two-stage detection:
    1. Check known binary extensions (fast)
    2. Analyze file content for binary indicators (null bytes, non-printable chars)

    Args:
        file_path: Path to the file to check

    Returns:
        True if file appears to be binary, False otherwise
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Check known binary extensions
    if ext in BINARY_EXTENSIONS:
        return True

    # Check file content for binary indicators
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)

        if not chunk:
            return False

        # Check for null bytes (strong binary indicator)
        if b"\x00" in chunk:
            return True

        # Count non-printable characters
        non_printable = sum(
            1 for byte in chunk
            if byte < 9 or (13 < byte < 32)
        )

        # If >30% non-printable, consider it binary
        return (non_printable / len(chunk)) > 0.3

    except Exception:
        return False


def get_similar_files(
    directory: str | Path,
    filename: str,
    max_suggestions: int = 3
) -> list[str]:
    """
    Find similar filenames in a directory for suggestions.

    Used when a file is not found to suggest alternatives.

    Args:
        directory: Directory to search in
        filename: The filename that wasn't found
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar file paths (up to max_suggestions)
    """
    import os

    try:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []

        entries = os.listdir(dir_path)
        filename_lower = filename.lower()

        suggestions = [
            str(dir_path / entry)
            for entry in entries
            if filename_lower in entry.lower() or entry.lower() in filename_lower
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
                matrix[i - 1][j] + 1,      # deletion
                matrix[i][j - 1] + 1,      # insertion
                matrix[i - 1][j - 1] + cost  # substitution
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
            confidence=1.0
        )
        start = index + len(find)


def line_trimmed_replacer(content: str, find: str) -> Generator[Match]:
    """Match lines after stripping whitespace from each line."""
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    # Remove trailing empty line if present
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
            # Calculate start position
            start_pos = sum(len(content_lines[k]) + 1 for k in range(i))
            # Calculate end position
            end_pos = start_pos
            for k in range(len(find_lines)):
                end_pos += len(content_lines[i + k])
                if k < len(find_lines) - 1:
                    end_pos += 1  # newline

            matched_text = "\n".join(content_lines[i:i + len(find_lines)])
            yield Match(
                start=start_pos,
                end=end_pos,
                text=matched_text,
                method="line_trimmed",
                confidence=0.9
            )


# Thresholds for BlockAnchorReplacer
SINGLE_CANDIDATE_THRESHOLD = 0.6
MULTIPLE_CANDIDATES_THRESHOLD = 0.3


def block_anchor_replacer(content: str, find: str) -> Generator[Match]:
    """Match blocks using first/last lines as anchors with fuzzy middle matching."""
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    # Need at least 3 lines for anchor matching
    if len(find_lines) < 3:
        return

    # Remove trailing empty line if present
    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    if len(find_lines) < 3:
        return

    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    # Collect candidates where both anchors match
    candidates: list[tuple[int, int]] = []
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue
        # Look for matching last line
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                candidates.append((i, j))
                break  # Only first match of last line

    if not candidates:
        return

    def calculate_similarity(start_line: int, end_line: int) -> float:
        """Calculate similarity of middle lines using Levenshtein distance."""
        actual_block_size = end_line - start_line + 1
        find_block_size = len(find_lines)
        lines_to_check = min(find_block_size - 2, actual_block_size - 2)

        if lines_to_check <= 0:
            return 1.0  # No middle lines to compare

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
        """Create a Match from line indices."""
        start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))
        end_pos = start_pos
        for k in range(start_line, end_line + 1):
            end_pos += len(content_lines[k])
            if k < end_line:
                end_pos += 1

        matched_text = "\n".join(content_lines[start_line:end_line + 1])
        return Match(
            start=start_pos,
            end=end_pos,
            text=matched_text,
            method="block_anchor",
            confidence=confidence
        )

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        similarity = calculate_similarity(start_line, end_line)
        if similarity >= SINGLE_CANDIDATE_THRESHOLD:
            yield make_match(start_line, end_line, similarity)
        return

    # Multiple candidates - find best match
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
    replace_all: bool = False
) -> ReplaceResult:
    """
    Replace old_string with new_string in content.

    Uses a three-level matching strategy:
    1. Exact match (confidence: 1.0)
    2. Line-trimmed match - ignores whitespace per line (confidence: 0.9)
    3. Block anchor match - matches first/last lines, fuzzy middle (confidence: 0.3-0.6)

    Safety rules:
    - old_string cannot be empty
    - old_string must be different from new_string
    - replace_all only works with exact matches
    - Multiple matches returns an error with match locations

    Args:
        content: The file content to search in
        old_string: The text to find and replace
        new_string: The replacement text
        replace_all: If True, replace all occurrences (exact match only)

    Returns:
        ReplaceResult with success status, new content, or error details
    """
    if not old_string:
        return ReplaceResult(
            success=False,
            error="old_string cannot be empty. Use the write tool to create new files."
        )

    if old_string == new_string:
        return ReplaceResult(
            success=False,
            error="old_string and new_string must be different."
        )

    # Normalize line endings
    content = normalize_line_endings(content)
    old_string = normalize_line_endings(old_string)
    new_string = normalize_line_endings(new_string)

    # Try each replacer in order
    for replacer in REPLACERS:
        matches = list(replacer(content, old_string))

        if not matches:
            continue

        # Found matches with this replacer
        method = matches[0].method
        is_exact = method == "exact"

        # Safety: replace_all only with exact matches
        if replace_all and not is_exact:
            return ReplaceResult(
                success=False,
                error=f"replace_all=True is only allowed with exact matches. "
                      f"Found fuzzy match using '{method}' method."
            )

        # Handle replace_all with exact match
        if replace_all and is_exact:
            new_content = content.replace(old_string, new_string)
            return ReplaceResult(
                success=True,
                content=new_content,
                method_used=method,
                matches=matches
            )

        # Single match - do the replacement
        if len(matches) == 1:
            match = matches[0]
            new_content = content[:match.start] + new_string + content[match.end:]
            return ReplaceResult(
                success=True,
                content=new_content,
                method_used=method,
                matches=[match]
            )

        # Multiple matches - return error with locations
        match_info = []
        for m in matches:
            line_num = get_line_number(content, m.start)
            context = get_context_lines(content, m.start, m.end)
            match_info.append(f"Match at line {line_num}:\n{context}")

        return ReplaceResult(
            success=False,
            error=f"Found {len(matches)} matches. Provide more context in old_string to identify a unique match.\n\n"
                  + "\n\n---\n\n".join(match_info),
            matches=matches
        )

    # No matches found with any replacer
    return ReplaceResult(
        success=False,
        error="old_string not found in file content."
    )
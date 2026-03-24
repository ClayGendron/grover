"""Text replacement engine — smart edit with exact, line-trimmed, and block-anchor matching."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass


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
                    f"replace_all=True is only allowed with exact matches. Found fuzzy match using '{method}' method."
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
                "Provide more context in old_string to identify a unique match.\n\n" + "\n\n---\n\n".join(match_info)
            ),
            matches=matches,
        )

    return ReplaceResult(
        success=False,
        error="old_string not found in file content.",
    )

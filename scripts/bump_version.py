"""Bump the project version in pyproject.toml.

Usage:
    uv run python scripts/bump_version.py --patch   # 0.0.1 → 0.0.2
    uv run python scripts/bump_version.py --minor   # 0.0.2 → 0.1.0
    uv run python scripts/bump_version.py --major   # 0.1.0 → 1.0.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
VERSION_RE = re.compile(r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")', re.MULTILINE)


def read_version(text: str) -> tuple[int, int, int]:
    match = VERSION_RE.search(text)
    if not match:
        print("error: could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    parts = match.group(2).split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def bump(major: int, minor: int, patch: int, part: str) -> tuple[int, int, int]:
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump project version")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--major", action="store_const", const="major", dest="part")
    group.add_argument("--minor", action="store_const", const="minor", dest="part")
    group.add_argument("--patch", action="store_const", const="patch", dest="part")
    args = parser.parse_args()

    text = PYPROJECT.read_text()
    old = read_version(text)
    new = bump(*old, args.part)

    old_str = f"{old[0]}.{old[1]}.{old[2]}"
    new_str = f"{new[0]}.{new[1]}.{new[2]}"

    updated = VERSION_RE.sub(rf"\g<1>{new_str}\3", text)
    PYPROJECT.write_text(updated)

    print(f"{old_str} → {new_str}")


if __name__ == "__main__":
    main()

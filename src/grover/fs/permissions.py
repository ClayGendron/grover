"""Permission enum and directory-level gates."""

from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    """Permission level for a mount point or directory."""

    READ_WRITE = "read_write"
    READ_ONLY = "read_only"

"""
File System Permissions.

Defines permission levels for mount points and directories.
"""

from enum import Enum


class Permission(str, Enum):
    """Permission level for a mount point or directory."""
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
"""StorageBackend protocol — runtime-checkable interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .types import (
        DeleteResult,
        EditResult,
        FileInfo,
        ListResult,
        MkdirResult,
        MoveResult,
        ReadResult,
        RestoreResult,
        VersionInfo,
        WriteResult,
    )


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the interface for filesystem backends.

    Both LocalFileSystem and DatabaseFileSystem implement this protocol.
    Using @runtime_checkable allows isinstance() checks at runtime.
    """

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> StorageBackend: ...

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult: ...

    async def list_dir(self, path: str = "/") -> ListResult: ...

    async def exists(self, path: str) -> bool: ...

    async def get_info(self, path: str) -> FileInfo | None: ...

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
    ) -> WriteResult: ...

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
    ) -> EditResult: ...

    async def delete(self, path: str, permanent: bool = False) -> DeleteResult: ...

    async def mkdir(self, path: str, parents: bool = True) -> MkdirResult: ...

    async def move(self, src: str, dest: str) -> MoveResult: ...

    async def copy(self, src: str, dest: str) -> WriteResult: ...

    # =========================================================================
    # Version & Trash Operations
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]: ...

    async def restore_version(self, path: str, version: int) -> RestoreResult: ...

    async def get_version_content(self, path: str, version: int) -> str | None: ...

    async def list_trash(self) -> ListResult: ...

    async def restore_from_trash(self, path: str) -> RestoreResult: ...

    async def empty_trash(self) -> DeleteResult: ...

    async def close(self) -> None: ...

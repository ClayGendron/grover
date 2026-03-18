"""Version provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.file import FileModelBase
    from grover.models.internal.results import FileOperationResult


@runtime_checkable
class VersionProvider(Protocol):
    """Version storage — diff-based with periodic snapshots."""

    async def save_version(
        self,
        session: AsyncSession,
        file: FileModelBase,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None: ...

    async def delete_versions(self, session: AsyncSession, file_path: str) -> None: ...

    async def list_versions(self, session: AsyncSession, file: FileModelBase) -> list: ...

    async def get_version_content(self, session: AsyncSession, file: FileModelBase, version: int) -> str | None: ...

    async def verify_chain(self, session: AsyncSession, file: FileModelBase) -> FileOperationResult: ...

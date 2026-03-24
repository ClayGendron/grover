"""Version provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.file import FileModelBase


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

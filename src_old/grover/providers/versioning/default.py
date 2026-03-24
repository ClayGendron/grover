"""DefaultVersionProvider — version save and delete."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete as sa_delete

from grover.util.content import compute_content_hash

from .diff import SNAPSHOT_INTERVAL, compute_diff

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.file import FileModelBase
    from grover.models.database.version import FileVersionModelBase


class DefaultVersionProvider:
    """Diff-based version storage with periodic snapshots.

    Stores a full snapshot at version 1 and every ``SNAPSHOT_INTERVAL``
    versions.  All other versions are stored as forward unified diffs.
    """

    def __init__(
        self,
        file_model: type[FileModelBase],
        file_version_model: type[FileVersionModelBase],
    ) -> None:
        self._file_model = file_model
        self._file_version_model = file_version_model

    async def save_version(
        self,
        session: AsyncSession,
        file: FileModelBase,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None:
        """Save a version record using diff-based storage."""
        version_num = file.current_version
        is_snap = (version_num % SNAPSHOT_INTERVAL == 0) or (version_num == 1)

        stored_content = new_content if is_snap or not old_content else compute_diff(old_content, new_content)
        content_hash, size_bytes = compute_content_hash(new_content)

        version = self._file_version_model(
            file_path=file.path,
            path=f"{file.path}@{version_num}",
            version=version_num,
            is_snapshot=is_snap or not old_content,
            content=stored_content,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_by=created_by,
        )
        session.add(version)

    async def delete_versions(self, session: AsyncSession, file_path: str) -> None:
        """Delete all version records for a file."""
        fv_model = self._file_version_model
        await session.execute(
            sa_delete(fv_model).where(
                fv_model.file_path == file_path,  # type: ignore[arg-type]
            )
        )

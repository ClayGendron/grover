"""VersioningService — version save, delete, list, reconstruct."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from sqlalchemy import delete as sa_delete
from sqlmodel import select

from .diff import SNAPSHOT_INTERVAL, compute_diff, reconstruct_version
from .exceptions import ConsistencyError
from .types import VersionInfo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase, FileVersionBase


class VersioningService:
    """Diff-based version storage with periodic snapshots.

    Stores a full snapshot at version 1 and every ``SNAPSHOT_INTERVAL``
    versions.  All other versions are stored as forward unified diffs.
    """

    def __init__(
        self,
        file_model: type[FileBase],
        file_version_model: type[FileVersionBase],
    ) -> None:
        self._file_model = file_model
        self._file_version_model = file_version_model

    async def save_version(
        self,
        session: AsyncSession,
        file: FileBase,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None:
        """Save a version record using diff-based storage."""
        version_num = file.current_version
        is_snap = (version_num % SNAPSHOT_INTERVAL == 0) or (version_num == 1)

        if is_snap or not old_content:
            content = new_content
        else:
            content = compute_diff(old_content, new_content)

        new_content_bytes = new_content.encode()
        version = self._file_version_model(
            file_id=file.id,
            version=version_num,
            is_snapshot=is_snap or not old_content,
            content=content,
            content_hash=hashlib.sha256(new_content_bytes).hexdigest(),
            size_bytes=len(new_content_bytes),
            created_by=created_by,
        )
        session.add(version)

    async def delete_versions(self, session: AsyncSession, file_id: str) -> None:
        """Delete all version records for a file."""
        fv_model = self._file_version_model
        await session.execute(
            sa_delete(fv_model).where(
                fv_model.file_id == file_id,  # type: ignore[arg-type]
            )
        )

    async def list_versions(
        self,
        session: AsyncSession,
        file: FileBase,
    ) -> list[VersionInfo]:
        """List all saved versions for a file."""
        fv_model = self._file_version_model
        result = await session.execute(
            select(fv_model)
            .where(fv_model.file_id == file.id)  # type: ignore[arg-type]
            .order_by(fv_model.version.desc())  # type: ignore[unresolved-attribute]
        )
        versions = result.scalars().all()

        return [
            VersionInfo(
                version=v.version,
                content_hash=v.content_hash,
                size_bytes=v.size_bytes,
                created_at=v.created_at,
                created_by=v.created_by,
            )
            for v in versions
        ]

    async def get_version_content(
        self,
        session: AsyncSession,
        file: FileBase,
        version: int,
    ) -> str | None:
        """Get the content of a specific version using diff reconstruction."""
        fv_model = self._file_version_model

        # Find the nearest snapshot at or before the target version
        snapshot_result = await session.execute(
            select(fv_model)
            .where(
                fv_model.file_id == file.id,  # type: ignore[arg-type]
                fv_model.version <= version,  # type: ignore[arg-type]
                fv_model.is_snapshot.is_(True),  # type: ignore[unresolved-attribute]
            )
            .order_by(fv_model.version.desc())  # type: ignore[unresolved-attribute]
            .limit(1)
        )
        snapshot = snapshot_result.scalar_one_or_none()
        if not snapshot:
            return None

        # Collect all versions from snapshot through target
        chain_result = await session.execute(
            select(fv_model)
            .where(
                fv_model.file_id == file.id,  # type: ignore[arg-type]
                fv_model.version >= snapshot.version,  # type: ignore[arg-type]
                fv_model.version <= version,  # type: ignore[arg-type]
            )
            .order_by(fv_model.version.asc())  # type: ignore[unresolved-attribute]
        )
        chain = chain_result.scalars().all()

        if not chain:
            return None

        # The exact target version must exist in the chain
        if chain[-1].version != version:
            return None

        entries = [(v.is_snapshot, v.content) for v in chain]
        content = reconstruct_version(entries)

        # Verify SHA256 against the target version's stored hash
        expected_hash = chain[-1].content_hash
        actual_hash = hashlib.sha256(content.encode()).hexdigest()
        if actual_hash != expected_hash:
            raise ConsistencyError(
                f"Version {version} of file: content hash mismatch "
                f"(expected {expected_hash[:12]}…, got {actual_hash[:12]}…)"
            )

        return content

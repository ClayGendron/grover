"""DefaultVersionProvider — version save, delete, list, reconstruct."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import delete as sa_delete
from sqlmodel import select

from grover.models.internal.ref import File
from grover.models.internal.results import FileOperationResult
from grover.util.content import compute_content_hash

from ...exceptions import ConsistencyError
from .diff import SNAPSHOT_INTERVAL, compute_diff, reconstruct_version

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.file import FileModelBase
    from grover.models.database.version import FileVersionModelBase


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """Internal version metadata (not part of public API)."""

    version: int
    content_hash: str
    size_bytes: int
    created_at: datetime | None = None
    created_by: str | None = None


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

    async def list_versions(
        self,
        session: AsyncSession,
        file: FileModelBase,
    ) -> list[VersionInfo]:
        """List all saved versions for a file."""
        fv_model = self._file_version_model
        result = await session.execute(
            select(fv_model).where(fv_model.file_path == file.path).order_by(fv_model.version.desc())  # type: ignore[unresolved-attribute]
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
        file: FileModelBase,
        version: int,
    ) -> str | None:
        """Get the content of a specific version using diff reconstruction."""
        fv_model = self._file_version_model

        # Find the nearest snapshot at or before the target version
        snapshot_result = await session.execute(
            select(fv_model)
            .where(
                fv_model.file_path == file.path,
                fv_model.version <= version,
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
                fv_model.file_path == file.path,
                fv_model.version >= snapshot.version,
                fv_model.version <= version,
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
        actual_hash = compute_content_hash(content)[0]
        if actual_hash != expected_hash:
            raise ConsistencyError(
                f"Version {version} of file: content hash mismatch "
                f"(expected {expected_hash[:12]}…, got {actual_hash[:12]}…)"
            )

        return content

    async def verify_chain(
        self,
        session: AsyncSession,
        file: FileModelBase,
    ) -> FileOperationResult:
        """Verify the integrity of the entire version chain for a file.

        Reconstructs every version and checks its SHA256 hash against the
        stored hash. Captures all errors rather than raising — callers get
        a full report of which versions are healthy and which are corrupt.
        """
        fv_model = self._file_version_model

        # Load all version records ordered ascending
        all_result = await session.execute(
            select(fv_model).where(fv_model.file_path == file.path).order_by(fv_model.version.asc())  # type: ignore[unresolved-attribute]
        )
        all_versions = all_result.scalars().all()

        if not all_versions:
            return FileOperationResult(
                success=True,
                message="Verified: 0 checked, 0 passed, 0 failed",
                file=File(path=file.path),
            )

        checked = 0
        passed = 0

        for v in all_versions:
            checked += 1

            # Find the nearest snapshot at or before this version
            snapshot_rec = None
            for candidate in all_versions:
                if candidate.version > v.version:
                    break
                if candidate.is_snapshot:
                    snapshot_rec = candidate

            if snapshot_rec is None:
                continue

            # Build the sub-chain from snapshot through target
            chain = [rec for rec in all_versions if snapshot_rec.version <= rec.version <= v.version]

            try:
                entries = [(rec.is_snapshot, rec.content) for rec in chain]
                content = reconstruct_version(entries)
                actual_hash = compute_content_hash(content)[0]
            except Exception:
                continue

            if actual_hash == v.content_hash:
                passed += 1

        failed = checked - passed
        ok = failed == 0
        return FileOperationResult(
            success=ok,
            message=f"Verified: {checked} checked, {passed} passed, {failed} failed",
            file=File(path=file.path),
        )

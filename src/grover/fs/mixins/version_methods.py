"""VersionMethodsMixin — version delegates for DatabaseFileSystem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.types.operations import GetVersionContentResult, RestoreResult, VerifyVersionResult
from grover.types.search import FileSearchCandidate, VersionEvidence, VersionResult

from ..utils import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class VersionMethodsMixin:
    """Delegates version operations to ``self.version_provider``."""

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VersionResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)  # type: ignore[attr-defined]
        if not file:
            return VersionResult(success=False, message=f"File not found: {path}")
        versions = await self.version_provider.list_versions(sess, file)  # type: ignore[attr-defined]
        candidates = [
            FileSearchCandidate(
                path=f"{path}@{v.version}",
                evidence=[
                    VersionEvidence(
                        strategy="version",
                        path=path,
                        version=v.version,
                        content_hash=v.content_hash,
                        size_bytes=v.size_bytes,
                        created_at=v.created_at,
                        created_by=v.created_by,
                    )
                ],
            )
            for v in versions
        ]
        return VersionResult(
            success=True,
            message=f"Found {len(versions)} version(s)",
            candidates=candidates,
        )

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GetVersionContentResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)  # type: ignore[attr-defined]
        if not file:
            return GetVersionContentResult(
                success=False,
                message=f"File not found: {path}",
            )
        content = await self.version_provider.get_version_content(sess, file, version)  # type: ignore[attr-defined]
        if content is None:
            return GetVersionContentResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )
        return GetVersionContentResult(success=True, message="OK", content=content)

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> RestoreResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        path = normalize_path(path)
        vc_result = await self.get_version_content(path, version, session=sess)
        if not vc_result.success or vc_result.content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(  # type: ignore[attr-defined]
            path,
            vc_result.content,
            created_by="restore",
            session=sess,
        )

        return RestoreResult(
            success=True,
            message=f"Restored {path} to version {version}",
            path=path,
            restored_version=version,
            version=write_result.version,
        )

    async def verify_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VerifyVersionResult:
        sess = self._require_session(session)  # type: ignore[attr-defined]
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)  # type: ignore[attr-defined]
        if not file:
            return VerifyVersionResult(
                success=False,
                message=f"File not found: {path}",
                path=path,
            )
        return await self.version_provider.verify_chain(sess, file)  # type: ignore[attr-defined]

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> list[VerifyVersionResult]:
        from sqlmodel import select

        sess = self._require_session(session)  # type: ignore[attr-defined]
        model = self._file_model  # type: ignore[attr-defined]
        result = await sess.execute(
            select(model).where(
                model.deleted_at.is_(None),
                model.is_directory.is_(False),
            )
        )
        results: list[VerifyVersionResult] = []
        for file in result.scalars().all():
            results.append(await self.version_provider.verify_chain(sess, file))  # type: ignore[attr-defined] # noqa: PERF401
        return results

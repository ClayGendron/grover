"""TrashService — soft-delete listing, restore, and empty."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from grover.types.operations import DeleteResult, RestoreResult
from grover.types.search import FileSearchCandidate, TrashEvidence, TrashResult

from .utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase

    from .providers.protocols import VersionProvider

    ContentDeleter = Callable[[str, AsyncSession], Awaitable[None]]
    GetFile = Callable[[AsyncSession, str, bool], Awaitable[FileBase | None]]


class TrashService:
    """Trash management: list, restore, and empty.

    Depends on a ``VersionProvider`` for cleaning up version records
    and a content-delete callback for removing content from storage.
    """

    def __init__(
        self,
        file_model: type[FileBase],
        versioning: VersionProvider,
        delete_content_cb: ContentDeleter,
    ) -> None:
        self._file_model = file_model
        self._versioning = versioning
        self._delete_content_cb = delete_content_cb

    async def list_trash(
        self, session: AsyncSession, *, owner_id: str | None = None
    ) -> TrashResult:
        """List soft-deleted files, optionally scoped to *owner_id*."""
        model = self._file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
        files = result.scalars().all()

        candidates = [
            FileSearchCandidate(
                path=f.original_path or f.path,
                evidence=[
                    TrashEvidence(
                        strategy="trash",
                        path=f.original_path or f.path,
                        deleted_at=f.deleted_at,
                        original_path=f.original_path or f.path,
                    )
                ],
            )
            for f in files
        ]

        return TrashResult(
            success=True,
            message=f"Found {len(candidates)} items in trash",
            candidates=candidates,
        )

    async def restore_from_trash(
        self,
        session: AsyncSession,
        path: str,
        get_file: GetFile,
        *,
        owner_id: str | None = None,
    ) -> RestoreResult:
        """Restore a file from trash, optionally verifying *owner_id*."""
        path = normalize_path(path)

        model = self._file_model
        conditions = [
            model.original_path == path,
            model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
        ]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
        file = result.scalar_one_or_none()

        if not file:
            return RestoreResult(success=False, message=f"File not in trash: {path}")

        original = file.original_path or path

        # If path is occupied, overwrite the occupant (git restore semantics).
        existing = await get_file(session, original, False)
        if existing and existing.id != file.id:
            await self._versioning.delete_versions(session, existing.id)
            await session.delete(existing)
            await session.flush()

        file.path = original
        file.original_path = None
        file.deleted_at = None
        file.updated_at = datetime.now(UTC)

        if file.is_directory:
            children_result = await session.execute(
                select(model).where(
                    model.original_path.startswith(path + "/"),  # type: ignore[union-attr]
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            children = children_result.scalars().all()

            # Remove occupants at children's original paths
            had_occupants = False
            for child in children:
                child_original = child.original_path or child.path
                child_existing = await get_file(session, child_original, False)
                if child_existing and child_existing.id != child.id:
                    await self._versioning.delete_versions(session, child_existing.id)
                    await session.delete(child_existing)
                    had_occupants = True
            if had_occupants:
                await session.flush()

            for child in children:
                child.path = child.original_path or child.path
                child.original_path = None
                child.deleted_at = None
                child.updated_at = datetime.now(UTC)

        await session.flush()

        return RestoreResult(
            success=True,
            message=f"Restored from trash: {path}",
            path=path,
        )

    async def empty_trash(
        self, session: AsyncSession, *, owner_id: str | None = None
    ) -> DeleteResult:
        """Permanently delete trashed files, optionally scoped to *owner_id*."""
        model = self._file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
        files = result.scalars().all()

        count = len(files)
        for file in files:
            await self._versioning.delete_versions(session, file.id)
            await self._delete_content_cb(file.original_path or file.path, session)
            await session.delete(file)

        await session.flush()

        return DeleteResult(
            success=True,
            message=f"Permanently deleted {count} items from trash",
            permanent=True,
            total_deleted=count,
        )

"""Standalone orchestration functions for filesystem operations.

Each function takes services + content callbacks as parameters.
No inheritance, no duplication. Backends compose these freely.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from .metadata import MetadataService, compute_content_hash
from .types import (
    DeleteResult,
    EditResult,
    ListResult,
    MoveResult,
    ReadResult,
    WriteResult,
)
from .utils import (
    guess_mime_type,
    is_text_file,
    is_trash_path,
    normalize_path,
    replace,
    split_path,
    to_trash_path,
    validate_path,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase

    from .directories import DirectoryService
    from .versioning import VersioningService

    ContentReader = Callable[[str, AsyncSession], Awaitable[str | None]]
    ContentWriter = Callable[[str, str, AsyncSession], Awaitable[None]]
    ContentDeleter = Callable[[str, AsyncSession], Awaitable[None]]

logger = logging.getLogger(__name__)

# Constants for read operations
DEFAULT_READ_LIMIT = 2000


async def read_file(
    path: str,
    offset: int,
    limit: int,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    read_content: ContentReader,
) -> ReadResult:
    """Orchestrate a file read: validate → lookup → read → paginate."""
    valid, error = validate_path(path)
    if not valid:
        return ReadResult(success=False, message=error)

    path = normalize_path(path)

    if is_trash_path(path):
        return ReadResult(success=False, message=f"Cannot read from trash: {path}")

    file = await metadata.get_file(session, path)

    if not file:
        return ReadResult(success=False, message=f"File not found: {path}")

    if file.is_directory:
        return ReadResult(
            success=False,
            message=f"Path is a directory, not a file: {path}",
        )

    content = await read_content(path, session)
    if content is None:
        return ReadResult(success=False, message=f"File content not found: {path}")

    return MetadataService.paginate_content(content, path, offset, limit)


async def write_file(
    path: str,
    content: str,
    created_by: str,
    overwrite: bool,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    versioning: VersioningService,
    directories: DirectoryService,
    file_model: type[FileBase],
    read_content: ContentReader,
    write_content: ContentWriter,
) -> WriteResult:
    """Orchestrate a file write: validate → version → write → flush."""
    valid, error = validate_path(path)
    if not valid:
        return WriteResult(success=False, message=error)

    path = normalize_path(path)
    _, name = split_path(path)

    if not is_text_file(name):
        return WriteResult(
            success=False,
            message=(
                f"Cannot write non-text file: {name}. "
                "Use allowed extensions (.py, .js, .json, .md, etc.)"
            ),
        )

    content_hash, size_bytes = compute_content_hash(content)

    existing = await metadata.get_file(session, path, include_deleted=True)

    if existing:
        if existing.is_directory:
            return WriteResult(success=False, message=f"Path is a directory: {path}")

        if not overwrite and not existing.deleted_at:
            return WriteResult(
                success=False,
                message=f"File already exists: {path}",
            )

        # Handle soft-deleted files
        if existing.deleted_at:
            existing.deleted_at = None
            existing.path = existing.original_path or path
            existing.original_path = None

        # Update metadata (increment version first so save_version uses new number)
        existing.current_version += 1
        existing.content_hash = content_hash
        existing.size_bytes = size_bytes
        existing.updated_at = datetime.now(UTC)

        # Save version after incrementing
        old_content = await read_content(path, session)
        if old_content is not None:
            await versioning.save_version(
                session, existing, old_content, content, created_by,
            )

        # Write content first, then flush. If disk write fails the
        # exception propagates to VFS which rolls back the session.
        # See docs/internals/fs.md.
        await write_content(path, content, session)
        await session.flush()
        created = False
        version = existing.current_version
    else:
        await directories.ensure_parent_dirs(session, path)

        new_file = file_model(
            path=path,
            name=name,
            parent_path=split_path(path)[0],
            content_hash=content_hash,
            size_bytes=size_bytes,
            mime_type=guess_mime_type(name),
        )
        session.add(new_file)

        # Save initial snapshot (version 1)
        await versioning.save_version(
            session, new_file, "", content, created_by,
        )

        # Write content first, then flush — see comment above.
        await write_content(path, content, session)
        await session.flush()

        created = True
        version = 1

    return WriteResult(
        success=True,
        message=f"{'Created' if created else 'Updated'}: {path} (v{version})",
        file_path=path,
        created=created,
        version=version,
    )


async def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    created_by: str,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    versioning: VersioningService,
    read_content: ContentReader,
    write_content: ContentWriter,
) -> EditResult:
    """Orchestrate a file edit: validate → read → replace → write → flush."""
    valid, error = validate_path(path)
    if not valid:
        return EditResult(success=False, message=error)

    path = normalize_path(path)

    file = await metadata.get_file(session, path)

    if not file:
        return EditResult(success=False, message=f"File not found: {path}")

    if file.is_directory:
        return EditResult(success=False, message=f"Cannot edit directory: {path}")

    content = await read_content(path, session)
    if content is None:
        return EditResult(success=False, message=f"File content not found: {path}")

    result = replace(content, old_string, new_string, replace_all)

    if not result.success:
        return EditResult(
            success=False,
            message=result.error or "Edit failed",
            file_path=path,
        )

    new_content = result.content
    assert new_content is not None

    # Update metadata
    new_content_bytes = new_content.encode()
    file.current_version += 1
    file.content_hash = hashlib.sha256(new_content_bytes).hexdigest()
    file.size_bytes = len(new_content_bytes)
    file.updated_at = datetime.now(UTC)

    # Save version after incrementing
    await versioning.save_version(session, file, content, new_content, created_by)

    # Write content first, then flush — see write_file comment.
    await write_content(path, new_content, session)
    await session.flush()

    return EditResult(
        success=True,
        message=f"Edit applied to {path} (v{file.current_version})",
        file_path=path,
        version=file.current_version,
    )


async def delete_file(
    path: str,
    permanent: bool,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    versioning: VersioningService,
    file_model: type[FileBase],
    delete_content: ContentDeleter,
) -> DeleteResult:
    """Orchestrate a file delete: validate → soft-delete or permanent."""
    valid, error = validate_path(path)
    if not valid:
        return DeleteResult(success=False, message=error)

    path = normalize_path(path)

    file = await metadata.get_file(session, path)

    if not file:
        return DeleteResult(success=False, message=f"File not found: {path}")

    model = file_model
    if permanent:
        if file.is_directory:
            result = await session.execute(
                select(model).where(
                    model.path.startswith(path + "/"),  # type: ignore[union-attr]
                )
            )
            for child in result.scalars().all():
                await versioning.delete_versions(session, child.id)
                await delete_content(child.path, session)
                await session.delete(child)

        await versioning.delete_versions(session, file.id)
        await delete_content(path, session)
        await session.delete(file)
    else:
        now = datetime.now(UTC)
        if file.is_directory:
            children_result = await session.execute(
                select(model).where(
                    model.path.startswith(path + "/"),  # type: ignore[union-attr]
                    model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                )
            )
            for child in children_result.scalars().all():
                child.original_path = child.path
                child.path = to_trash_path(child.path, child.id)
                child.deleted_at = now

        file.original_path = file.path
        file.path = to_trash_path(file.path, file.id)
        file.deleted_at = now

    await session.flush()

    return DeleteResult(
        success=True,
        message=f"{'Permanently deleted' if permanent else 'Moved to trash'}: {path}",
        file_path=path,
        permanent=permanent,
    )


async def move_file(
    src: str,
    dest: str,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    versioning: VersioningService,
    file_model: type[FileBase],
    read_content: ContentReader,
    write_content: ContentWriter,
    delete_content: ContentDeleter,
) -> MoveResult:
    """Orchestrate a file move within the same backend."""
    valid, error = validate_path(src)
    if not valid:
        return MoveResult(success=False, message=error)

    src = normalize_path(src)
    dest = normalize_path(dest)

    valid, error = validate_path(dest)
    if not valid:
        return MoveResult(success=False, message=error)

    if src == dest:
        return MoveResult(
            success=True,
            message="Source and destination are the same",
            old_path=src,
            new_path=dest,
        )

    src_file = await metadata.get_file(session, src)
    if not src_file:
        return MoveResult(success=False, message=f"Source not found: {src}")

    if src_file.is_directory and dest.startswith(src + "/"):
        return MoveResult(
            success=False,
            message=f"Cannot move directory into itself: {dest} is inside {src}",
        )

    dest_file = await metadata.get_file(session, dest)
    if dest_file:
        if dest_file.is_directory:
            return MoveResult(
                success=False,
                message=f"Destination is a directory: {dest}",
            )
        if src_file.is_directory:
            return MoveResult(
                success=False,
                message=f"Cannot move directory over file: {dest}",
            )

        # Atomic move: all changes in the current session
        content = await read_content(src, session)
        if content is None:
            return MoveResult(success=False, message=f"Source content not found: {src}")
        old_dest_content = await read_content(dest, session) or ""

        # Update dest metadata with source content
        content_bytes = content.encode()
        dest_file.current_version += 1
        dest_file.content_hash = hashlib.sha256(content_bytes).hexdigest()
        dest_file.size_bytes = len(content_bytes)
        dest_file.updated_at = datetime.now(UTC)

        # Save version for dest
        await versioning.save_version(
            session, dest_file, old_dest_content, content, "move",
        )

        # Write content to dest storage
        await write_content(dest, content, session)

        # Soft-delete src
        now = datetime.now(UTC)
        src_file.original_path = src_file.path
        src_file.path = to_trash_path(src_file.path, src_file.id)
        src_file.deleted_at = now

        # Single flush
        await session.flush()

        # Clean up src content (best-effort)
        try:
            await delete_content(src, session)
        except Exception:
            logger.warning("Failed to clean up old content at %s", src)

        return MoveResult(
            success=True,
            message=f"Moved {src} to {dest}",
            old_path=src,
            new_path=dest,
        )

    old_paths: list[str] = [src]

    if src_file.is_directory:
        model = file_model
        result = await session.execute(
            select(model).where(
                model.path.startswith(src + "/"),  # type: ignore[union-attr]
            )
        )
        children = result.scalars().all()

        for desc in children:
            old_paths.append(desc.path)
            new_path = dest + desc.path[len(src):]
            content = await read_content(desc.path, session)
            if content is not None:
                await write_content(new_path, content, session)
            desc.path = new_path
            desc.name = split_path(new_path)[1]
            desc.parent_path = split_path(new_path)[0]

    content = await read_content(src, session)
    if content is not None:
        await write_content(dest, content, session)

    src_file.path = dest
    src_file.name = split_path(dest)[1]
    src_file.parent_path = split_path(dest)[0]
    src_file.updated_at = datetime.now(UTC)

    await session.flush()

    for old_path in old_paths:
        try:
            await delete_content(old_path, session)
        except Exception:
            logger.warning("Failed to clean up old content at %s", old_path)

    return MoveResult(
        success=True,
        message=f"Moved {src} to {dest}",
        old_path=src,
        new_path=dest,
    )


async def copy_file(
    src: str,
    dest: str,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    read_content: ContentReader,
    write_fn: Callable[..., Awaitable[WriteResult]],
) -> WriteResult:
    """Orchestrate a file copy: read src → write to dest."""
    src = normalize_path(src)

    src_file = await metadata.get_file(session, src)
    if not src_file:
        return WriteResult(success=False, message=f"Source not found: {src}")

    if src_file.is_directory:
        return WriteResult(success=False, message="Directory copy not yet implemented")

    content = await read_content(src, session)
    if content is None:
        return WriteResult(success=False, message=f"Source content not found: {src}")

    return await write_fn(dest, content, "copy", overwrite=True, session=session)


async def list_dir_db(
    path: str,
    session: AsyncSession,
    *,
    metadata: MetadataService,
    file_model: type[FileBase],
) -> ListResult:
    """List files and directories from database records."""
    path = normalize_path(path)

    if path != "/":
        dir_file = await metadata.get_file(session, path)
        if not dir_file:
            return ListResult(success=False, message=f"Directory not found: {path}")
        if not dir_file.is_directory:
            return ListResult(success=False, message=f"Not a directory: {path}")

    model = file_model
    if path == "/":
        result = await session.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.parent_path.in_(["", "/"]),  # type: ignore[union-attr]
            )
        )
    else:
        result = await session.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.parent_path == path,  # type: ignore[arg-type]
            )
        )
    entries = [MetadataService.file_to_info(f) for f in result.scalars().all()]

    return ListResult(
        success=True,
        message=f"Listed {len(entries)} items in {path}",
        entries=entries,
        path=path,
    )

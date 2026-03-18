"""Standalone orchestration functions for filesystem operations.

Each function takes services + content callbacks as parameters.
No inheritance, no duplication. Backends compose these freely.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlmodel import select

from grover.models.internal.ref import Directory, File
from grover.models.internal.results import FileOperationResult, GroverResult

from .content import compute_content_hash, guess_mime_type, is_text_file
from .paths import is_trash_path, normalize_path, split_path, to_trash_path, validate_path
from .replace import replace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.file import FileModelBase
    from grover.models.database.version import FileVersionModelBase
    from grover.providers.versioning.protocol import VersionProvider

    ContentReader = Callable[[str, AsyncSession], Awaitable[str | None]]
    ContentWriter = Callable[[str, str, AsyncSession], Awaitable[None]]
    ContentDeleter = Callable[[str, AsyncSession], Awaitable[None]]
    GetFileRecord = Callable[..., Awaitable["FileModelBase | None"]]
    EnsureParentDirs = Callable[[AsyncSession, str, str | None], Awaitable[None]]

logger = logging.getLogger(__name__)

# Constants for read operations
DEFAULT_READ_LIMIT = 2000


def file_to_info(f: FileModelBase) -> File | Directory:
    """Convert a FileModelBase model to a File or Directory."""
    if f.is_directory:
        return Directory(path=f.path, created_at=f.created_at, updated_at=f.updated_at)
    return File(
        path=f.path,
        size_bytes=f.size_bytes,
        mime_type=f.mime_type,
        current_version=f.current_version,
        created_at=f.created_at,
        updated_at=f.updated_at,
    )


async def check_external_edit(
    file: FileModelBase,
    current_content: str,
    session: AsyncSession,
    *,
    versioning: VersionProvider,
) -> bool:
    """Detect and record an external edit as a synthetic snapshot version.

    Compares the hash of *current_content* (the actual storage state) against
    ``file.content_hash`` (the last Grover-written hash).  If they differ, an
    external tool modified the file.  A full snapshot version is inserted with
    ``created_by="external"`` to keep the diff chain intact.

    Returns ``True`` if an external edit was detected and recorded, ``False``
    otherwise.

    This function mutates *file* in-place (increments version, updates hash
    and timestamps) but does **not** flush the session — the caller is
    responsible for flushing.
    """
    if not file.content_hash:
        return False

    current_hash, current_size = compute_content_hash(current_content)
    if current_hash == file.content_hash:
        return False

    # External edit detected — record synthetic snapshot version
    file.current_version += 1
    file.content_hash = current_hash
    file.size_bytes = current_size
    file.updated_at = datetime.now(UTC)

    # Passing old_content="" makes save_version store a full snapshot
    await versioning.save_version(
        session,
        file,
        "",
        current_content,
        "external",
    )

    logger.info(
        "External edit detected for %s — recorded synthetic v%d",
        file.path,
        file.current_version,
    )
    return True


async def read_file(
    path: str,
    offset: int,
    limit: int,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    read_content: ContentReader,
) -> FileOperationResult:
    """Orchestrate a file read: validate → lookup → read → paginate."""
    valid, error = validate_path(path)
    if not valid:
        return FileOperationResult(success=False, message=error)

    path = normalize_path(path)

    if is_trash_path(path):
        return FileOperationResult(success=False, message=f"Cannot read from trash: {path}")

    file = await get_file_record(session, path)

    if not file:
        return FileOperationResult(success=False, message=f"File not found: {path}")

    if file.is_directory:
        return FileOperationResult(
            success=False,
            message=f"Path is a directory, not a file: {path}",
        )

    content = await read_content(path, session)
    if content is None:
        return FileOperationResult(success=False, message=f"File content not found: {path}")

    return paginate_content(content, path, offset, limit)


def paginate_content(
    content: str,
    path: str,
    offset: int,
    limit: int,
) -> FileOperationResult:
    """Paginate file content and return raw text."""
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines == 0 or (total_lines == 1 and lines[0] == ""):
        return FileOperationResult(
            success=True,
            message=(f"File is empty. (total_lines=0, lines_read=0, truncated=False, line_offset={offset})"),
            file=File(path=path, content="", lines=0),
        )

    end_line = min(len(lines), offset + limit)
    output_lines = lines[offset:end_line]

    last_read_line = offset + len(output_lines)
    has_more = total_lines > last_read_line

    return FileOperationResult(
        success=True,
        message=(
            f"Read {len(output_lines)} lines from {path}"
            f" (total_lines={total_lines}, lines_read={len(output_lines)},"
            f" truncated={has_more}, line_offset={offset})"
        ),
        file=File(path=path, content="\n".join(output_lines), lines=total_lines),
    )


async def write_file(
    path: str,
    content: str,
    created_by: str,
    overwrite: bool,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    versioning: VersionProvider,
    ensure_parent_dirs: EnsureParentDirs,
    file_model: type[FileModelBase],
    read_content: ContentReader,
    write_content: ContentWriter,
    owner_id: str | None = None,
) -> FileOperationResult:
    """Orchestrate a file write: validate → version → write → flush."""
    valid, error = validate_path(path)
    if not valid:
        return FileOperationResult(success=False, message=error)

    path = normalize_path(path)
    _, name = split_path(path)

    if not is_text_file(name):
        return FileOperationResult(
            success=False,
            message=(f"Cannot write non-text file: {name}. Use allowed extensions (.py, .js, .json, .md, etc.)"),
        )

    content_hash, size_bytes = compute_content_hash(content)

    existing = await get_file_record(session, path, include_deleted=True)

    if existing:
        if existing.is_directory:
            return FileOperationResult(success=False, message=f"Path is a directory: {path}")

        if not overwrite and not existing.deleted_at:
            return FileOperationResult(
                success=False,
                message=f"File already exists: {path}",
            )

        # Handle soft-deleted files
        if existing.deleted_at:
            existing.deleted_at = None
            existing.path = existing.original_path or path
            existing.original_path = None

        # Read current content first — needed for external edit check AND diff
        old_content = await read_content(path, session)

        # Detect and record external edits before creating the Grover version
        if old_content is not None:
            await check_external_edit(
                existing,
                old_content,
                session,
                versioning=versioning,
            )

        # Update metadata (increment version first so save_version uses new number)
        existing.current_version += 1
        existing.content_hash = content_hash
        existing.size_bytes = size_bytes
        existing.updated_at = datetime.now(UTC)

        # Save version after incrementing
        if old_content is not None:
            await versioning.save_version(
                session,
                existing,
                old_content,
                content,
                created_by,
            )

        # Write content first, then flush. If disk write fails the
        # exception propagates to VFS which rolls back the session.
        # See docs/internals/fs.md.
        await write_content(path, content, session)
        await session.flush()
        created = False
        version = existing.current_version
    else:
        await ensure_parent_dirs(session, path, owner_id)

        now = datetime.now(UTC)
        new_file = file_model(
            path=path,
            parent_path=split_path(path)[0],
            owner_id=owner_id,
            content_hash=content_hash,
            size_bytes=size_bytes,
            mime_type=guess_mime_type(name),
            created_at=now,
            updated_at=now,
        )
        session.add(new_file)

        # Save initial snapshot (version 1)
        await versioning.save_version(
            session,
            new_file,
            "",
            content,
            created_by,
        )

        # Write content first, then flush — see comment above.
        await write_content(path, content, session)
        await session.flush()

        created = True
        version = 1

    return FileOperationResult(
        success=True,
        message=f"{'Created' if created else 'Updated'}: {path} (v{version})",
        file=File(path=path, current_version=version),
    )


async def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    created_by: str,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    versioning: VersionProvider,
    read_content: ContentReader,
    write_content: ContentWriter,
) -> FileOperationResult:
    """Orchestrate a file edit: validate → read → replace → write → flush."""
    valid, error = validate_path(path)
    if not valid:
        return FileOperationResult(success=False, message=error)

    path = normalize_path(path)

    file = await get_file_record(session, path)

    if not file:
        return FileOperationResult(success=False, message=f"File not found: {path}")

    if file.is_directory:
        return FileOperationResult(success=False, message=f"Cannot edit directory: {path}")

    content = await read_content(path, session)
    if content is None:
        return FileOperationResult(success=False, message=f"File content not found: {path}")

    # Detect and record external edits before applying the edit
    await check_external_edit(file, content, session, versioning=versioning)

    result = replace(content, old_string, new_string, replace_all)

    if not result.success:
        return FileOperationResult(
            success=False,
            message=result.error or "Edit failed",
            file=File(path=path),
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

    return FileOperationResult(
        success=True,
        message=f"Edit applied to {path} (v{file.current_version})",
        file=File(path=path, current_version=file.current_version),
    )


async def delete_file(
    path: str,
    permanent: bool,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    versioning: VersionProvider,
    file_model: type[FileModelBase],
    delete_content: ContentDeleter,
) -> FileOperationResult:
    """Orchestrate a file delete: validate → soft-delete or permanent."""
    valid, error = validate_path(path)
    if not valid:
        return FileOperationResult(success=False, message=error)

    path = normalize_path(path)

    file = await get_file_record(session, path)

    if not file:
        return FileOperationResult(success=False, message=f"File not found: {path}")

    model = file_model
    if permanent:
        if file.is_directory:
            result = await session.execute(
                select(model).where(
                    model.path.startswith(path + "/"),
                )
            )
            for child in result.scalars().all():
                await versioning.delete_versions(session, child.path)
                await delete_content(child.path, session)
                await session.delete(child)

        await versioning.delete_versions(session, file.path)
        await delete_content(path, session)
        await session.delete(file)
    else:
        now = datetime.now(UTC)
        if file.is_directory:
            children_result = await session.execute(
                select(model).where(
                    model.path.startswith(path + "/"),
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

    return FileOperationResult(
        success=True,
        message=f"{'Permanently deleted' if permanent else 'Moved to trash'}: {path}",
        file=File(path=path),
    )


async def move_file(
    src: str,
    dest: str,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    versioning: VersionProvider,
    ensure_parent_dirs: EnsureParentDirs,
    file_model: type[FileModelBase],
    file_version_model: type[FileVersionModelBase],
    read_content: ContentReader,
    write_content: ContentWriter,
    delete_content: ContentDeleter,
    follow: bool = False,
) -> FileOperationResult:
    """Orchestrate a file move within the same backend.

    Parameters
    ----------
    follow:
        ``False`` (default) — clean break: new file at *dest*, source deleted.
        Version history does not carry over.

        ``True`` — in-place rename: same file record, versions follow.
    """
    valid, error = validate_path(src)
    if not valid:
        return FileOperationResult(success=False, message=error)

    src = normalize_path(src)
    dest = normalize_path(dest)

    valid, error = validate_path(dest)
    if not valid:
        return FileOperationResult(success=False, message=error)

    if src == dest:
        return FileOperationResult(
            success=True,
            message=f"Source and destination are the same ({src} -> {dest})",
            file=File(path=dest),
        )

    src_file = await get_file_record(session, src)
    if not src_file:
        return FileOperationResult(success=False, message=f"Source not found: {src}")

    if src_file.is_directory and dest.startswith(src + "/"):
        return FileOperationResult(
            success=False,
            message=f"Cannot move directory into itself: {dest} is inside {src}",
        )

    dest_file = await get_file_record(session, dest)
    if dest_file:
        if dest_file.is_directory:
            return FileOperationResult(
                success=False,
                message=f"Destination is a directory: {dest}",
            )
        if src_file.is_directory:
            return FileOperationResult(
                success=False,
                message=f"Cannot move directory over file: {dest}",
            )

        # Overwrite existing dest with source content
        content = await read_content(src, session)
        if content is None:
            return FileOperationResult(success=False, message=f"Source content not found: {src}")
        old_dest_content = await read_content(dest, session) or ""

        # Update dest metadata with source content
        content_bytes = content.encode()
        dest_file.current_version += 1
        dest_file.content_hash = hashlib.sha256(content_bytes).hexdigest()
        dest_file.size_bytes = len(content_bytes)
        dest_file.updated_at = datetime.now(UTC)

        # Save version for dest
        await versioning.save_version(
            session,
            dest_file,
            old_dest_content,
            content,
            "move",
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

        return FileOperationResult(
            success=True,
            message=f"Moved {src} to {dest}",
            file=File(path=dest),
        )

    # ---- No existing dest ----

    if follow:
        # In-place rename: same file record, versions follow
        await ensure_parent_dirs(session, dest, None)
        old_paths: list[str] = [src]

        if src_file.is_directory:
            model = file_model
            result = await session.execute(
                select(model).where(
                    model.path.startswith(src + "/"),
                )
            )
            children = result.scalars().all()

            for desc in children:
                old_paths.append(desc.path)
                new_path = dest + desc.path[len(src) :]
                content = await read_content(desc.path, session)
                if content is not None:
                    await write_content(new_path, content, session)
                desc.path = new_path
                desc.parent_path = split_path(new_path)[0]

        content = await read_content(src, session)
        if content is not None:
            await write_content(dest, content, session)

        src_file.path = dest
        src_file.parent_path = split_path(dest)[0]
        src_file.updated_at = datetime.now(UTC)

        # Update version records to follow the rename
        for old_path in old_paths:
            new_path = dest + old_path[len(src) :] if old_path != src else dest
            ver_result = await session.execute(
                select(file_version_model).where(
                    file_version_model.file_path == old_path,
                )
            )
            for ver in ver_result.scalars().all():
                ver.file_path = new_path
                ver.path = f"{new_path}@{ver.version}"

        await session.flush()

        for old_path in old_paths:
            try:
                await delete_content(old_path, session)
            except Exception:
                logger.warning("Failed to clean up old content at %s", old_path)

        return FileOperationResult(
            success=True,
            message=f"Moved {src} to {dest}",
            file=File(path=dest),
        )

    # follow=False: clean break — new records at dest, source soft-deleted
    now = datetime.now(UTC)
    await ensure_parent_dirs(session, dest, None)

    if src_file.is_directory:
        # Collect children and their content before modifying records
        model = file_model
        result = await session.execute(
            select(model).where(
                model.path.startswith(src + "/"),
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        children = list(result.scalars().all())
        child_contents: list[tuple[Any, str, str | None]] = []
        for child in children:
            c = await read_content(child.path, session)
            new_child_path = dest + child.path[len(src) :]
            child_contents.append((child, new_child_path, c))

        # Soft-delete source children
        for child in children:
            child.original_path = child.path
            child.path = to_trash_path(child.path, child.id)
            child.deleted_at = now

        # Soft-delete source directory
        src_file.original_path = src_file.path
        src_file.path = to_trash_path(src_file.path, src_file.id)
        src_file.deleted_at = now

        # Create new directory
        dest_parent, dest_name = split_path(dest)
        move_now = datetime.now(UTC)
        new_dir = file_model(
            path=dest,
            parent_path=dest_parent,
            owner_id=src_file.owner_id,
            is_directory=True,
            created_at=move_now,
            updated_at=move_now,
        )
        session.add(new_dir)

        # Create new children
        for orig_child, new_child_path, child_content in child_contents:
            cp, cn = split_path(new_child_path)
            if orig_child.is_directory:
                new_child = file_model(
                    path=new_child_path,
                    parent_path=cp,
                    owner_id=src_file.owner_id,
                    is_directory=True,
                    created_at=move_now,
                    updated_at=move_now,
                )
            else:
                new_child = file_model(
                    path=new_child_path,
                    parent_path=cp,
                    owner_id=src_file.owner_id,
                    content_hash=compute_content_hash(child_content or "")[0],
                    size_bytes=compute_content_hash(child_content or "")[1],
                    mime_type=guess_mime_type(cn),
                    created_at=move_now,
                    updated_at=move_now,
                )
            session.add(new_child)
            if child_content is not None and not orig_child.is_directory:
                await versioning.save_version(session, new_child, "", child_content, "move")
                await write_content(new_child_path, child_content, session)

        await session.flush()

        # Clean up old content (best-effort, use original path before soft-delete)
        for orig_child, _np, _c in child_contents:
            orig_path = orig_child.original_path
            if orig_path:
                try:
                    await delete_content(orig_path, session)
                except Exception:
                    logger.debug("Failed to clean up old content for child")
    else:
        # Single file clean break
        content = await read_content(src, session)
        if content is None:
            return FileOperationResult(success=False, message=f"Source content not found: {src}")

        # Soft-delete source
        src_file.original_path = src_file.path
        src_file.path = to_trash_path(src_file.path, src_file.id)
        src_file.deleted_at = now

        # Create new file at dest
        dest_parent, dest_name = split_path(dest)
        content_hash, size_bytes = compute_content_hash(content)
        move_now = datetime.now(UTC)
        new_file = file_model(
            path=dest,
            parent_path=dest_parent,
            owner_id=src_file.owner_id,
            content_hash=content_hash,
            size_bytes=size_bytes,
            mime_type=guess_mime_type(dest_name),
            created_at=move_now,
            updated_at=move_now,
        )
        session.add(new_file)

        await versioning.save_version(session, new_file, "", content, "move")
        await write_content(dest, content, session)
        await session.flush()

        # Clean up src content (best-effort)
        try:
            await delete_content(src, session)
        except Exception:
            logger.warning("Failed to clean up old content at %s", src)

    return FileOperationResult(
        success=True,
        message=f"Moved {src} to {dest}",
        file=File(path=dest),
    )


async def copy_file(
    src: str,
    dest: str,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    read_content: ContentReader,
    write_fn: Callable[..., Awaitable[FileOperationResult]],
) -> FileOperationResult:
    """Orchestrate a file copy: read src → write to dest."""
    src = normalize_path(src)

    src_file = await get_file_record(session, src)
    if not src_file:
        return FileOperationResult(success=False, message=f"Source not found: {src}")

    if src_file.is_directory:
        return FileOperationResult(success=False, message="Directory copy not yet implemented")

    content = await read_content(src, session)
    if content is None:
        return FileOperationResult(success=False, message=f"Source content not found: {src}")

    return await write_fn(dest, content, "copy", overwrite=True, session=session)


async def list_dir_db(
    path: str,
    session: AsyncSession,
    *,
    get_file_record: GetFileRecord,
    file_model: type[FileModelBase],
) -> GroverResult:
    """List files and directories from database records."""

    path = normalize_path(path)

    if path != "/":
        dir_file = await get_file_record(session, path)
        if not dir_file:
            return GroverResult(success=False, message=f"Directory not found: {path}")
        if not dir_file.is_directory:
            return GroverResult(success=False, message=f"Not a directory: {path}")

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
                model.parent_path == path,
            )
        )

    files: list[File] = []
    directories: list[Directory] = []
    for f in result.scalars().all():
        if f.is_directory:
            directories.append(Directory(path=f.path))
        else:
            files.append(
                File(
                    path=f.path,
                    size_bytes=f.size_bytes,
                    mime_type=f.mime_type,
                    current_version=f.current_version,
                    created_at=f.created_at,
                    updated_at=f.updated_at,
                )
            )

    total = len(files) + len(directories)
    return GroverResult(
        success=True,
        message=f"Listed {total} items in {path}",
        files=files,
        directories=directories,
    )

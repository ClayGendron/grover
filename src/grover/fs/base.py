"""BaseFileSystem — shared SQL logic, dialect-aware."""

from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Generic, TypeVar

from sqlalchemy import delete as sa_delete
from sqlmodel import select

from grover.fs.dialect import upsert_file
from grover.fs.diff import SNAPSHOT_INTERVAL, compute_diff, reconstruct_version
from grover.fs.exceptions import ConsistencyError
from grover.models.files import (
    File,
    FileBase,
    FileVersion,
    FileVersionBase,
)

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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Constants for read operations
DEFAULT_READ_LIMIT = 2000

F = TypeVar("F", bound=FileBase)
FV = TypeVar("FV", bound=FileVersionBase)


class BaseFileSystem(ABC, Generic[F, FV]):
    """Base file system with shared logic for both backends.

    Subclasses must implement:
    - _get_session(): Get database session
    - _read_content(path, session): Read file content from storage
    - _write_content(path, content, session): Write content to storage
    - _delete_content(path, session): Delete content from storage
    - _content_exists(path, session): Check if content exists in storage
    """

    def __init__(
        self,
        dialect: str = "sqlite",
        file_model: type[F] | None = None,
        file_version_model: type[FV] | None = None,
        schema: str | None = None,
    ) -> None:
        self.dialect = dialect
        self.schema = schema
        self._file_model: type[F] = file_model or File  # type: ignore[assignment]
        self._file_version_model: type[FV] = file_version_model or FileVersion  # type: ignore[assignment]

    @property
    def file_model(self) -> type[F]:
        """The SQLModel table class used for file records."""
        return self._file_model

    @property
    def file_version_model(self) -> type[FV]:
        """The SQLModel table class used for file version records."""
        return self._file_version_model

    # =========================================================================
    # Abstract Methods - Subclasses must implement
    # =========================================================================

    @abstractmethod
    async def _get_session(self) -> AsyncSession:
        """Return a database session for the current operation.

        LocalFileSystem creates a new session from its async session factory.
        DatabaseFileSystem raises RuntimeError (sessions must be provided
        per-operation via the ``session`` kwarg).
        """
        ...

    @abstractmethod
    async def _read_content(self, path: str, session: AsyncSession) -> str | None:
        """Read raw file content from the storage backend.

        ``path`` is already normalized. Return the file text if it exists,
        or ``None`` if no content is stored at that path. The base class
        handles all metadata lookups — this method only touches bytes.
        """
        ...

    @abstractmethod
    async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
        """Persist raw file content to the storage backend.

        ``path`` is already normalized. For LocalFileSystem this writes to
        disk; for DatabaseFileSystem this updates the ``content`` column on
        the file row. Parent directories are handled by the base class.
        """
        ...

    @abstractmethod
    async def _delete_content(self, path: str, session: AsyncSession) -> None:
        """Remove raw file content from the storage backend.

        Called during permanent deletes and moves. For LocalFileSystem this
        removes the file from disk. For DatabaseFileSystem this is a no-op
        because content lives in the file row that the base class
        already deletes.
        """
        ...

    @abstractmethod
    async def _content_exists(self, path: str, session: AsyncSession) -> bool:
        """Check whether raw content exists in the storage backend.

        ``path`` is already normalized. This checks actual storage (disk or
        DB column), not the file metadata record.
        """
        ...

    @abstractmethod
    async def _commit(self, session: AsyncSession) -> None:
        """Commit or flush the given session.

        LocalFileSystem calls ``session.commit()`` since it owns the session
        lifecycle. DatabaseFileSystem calls ``session.flush()`` to defer the
        real commit to the outer transaction boundary.
        """
        ...

    # =========================================================================
    # Session Resolution
    # =========================================================================

    async def _resolve_session(
        self, session: AsyncSession | None,
    ) -> tuple[AsyncSession, bool]:
        """Return ``(session, owns_it)``.

        If the caller provided a session we reuse it and do NOT
        commit / close it (``owns_it=False``).  Otherwise we create a
        fresh one via ``_get_session()`` (``owns_it=True``).
        """
        if session is not None:
            return session, False
        return await self._get_session(), True

    # =========================================================================
    # Shared Helper Methods
    # =========================================================================

    async def _get_file(
        self,
        session: AsyncSession,
        path: str,
        include_deleted: bool = False,
    ) -> F | None:
        """Get file record by path."""
        path = normalize_path(path)
        model = self._file_model
        query = select(model).where(
            model.path == path,  # type: ignore[arg-type]
        )
        if not include_deleted:
            query = query.where(model.deleted_at.is_(None))  # type: ignore[unresolved-attribute]

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def _save_version(
        self,
        session: AsyncSession,
        file: F,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None:
        """Save a version record using diff-based storage.

        Stores a full snapshot when version == 1, version % SNAPSHOT_INTERVAL == 0,
        or when old_content is empty. Otherwise, stores a forward diff.
        """
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

    async def _delete_versions(self, session: AsyncSession, file_id: str) -> None:
        """Delete all version records for a file."""
        fv_model = self._file_version_model
        await session.execute(
            sa_delete(fv_model).where(
                fv_model.file_id == file_id,  # type: ignore[arg-type]
            )
        )

    async def _ensure_parent_dirs(self, session: AsyncSession, path: str) -> None:
        """Ensure all parent directories exist in the database."""
        parts = path.split("/")
        for i in range(2, len(parts)):
            dir_path = "/".join(parts[:i])
            if not dir_path:
                continue

            dir_name = parts[i - 1]
            parent = "/".join(parts[:i - 1]) or "/"
            await upsert_file(
                session,
                self.dialect,
                values={
                    "id": str(uuid.uuid4()),
                    "path": dir_path,
                    "name": dir_name,
                    "parent_path": parent,
                    "is_directory": True,
                    "current_version": 1,
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                },
                conflict_keys=["path"],
                model=self._file_model,
                schema=self.schema,
                update_keys=["updated_at"],
            )

    def _paginate_content(
        self,
        content: str,
        path: str,
        offset: int,
        limit: int,
    ) -> ReadResult:
        """Paginate file content and return raw text (no formatting)."""
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines == 0 or (total_lines == 1 and lines[0] == ""):
            return ReadResult(
                success=True,
                message="File is empty.",
                content="",
                file_path=path,
                total_lines=0,
                lines_read=0,
                truncated=False,
                offset=offset,
            )

        end_line = min(len(lines), offset + limit)
        output_lines = lines[offset:end_line]

        last_read_line = offset + len(output_lines)
        has_more = total_lines > last_read_line

        return ReadResult(
            success=True,
            message=f"Read {len(output_lines)} lines from {path}",
            content="\n".join(output_lines),
            file_path=path,
            total_lines=total_lines,
            lines_read=len(output_lines),
            truncated=has_more,
            offset=offset,
        )

    # =========================================================================
    # Core Operations: Read
    # =========================================================================

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
        *,
        session: AsyncSession | None = None,
    ) -> ReadResult:
        """Read file content with pagination."""
        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        if is_trash_path(path):
            return ReadResult(success=False, message=f"Cannot read from trash: {path}")

        _session, owns = await self._resolve_session(session)

        try:
            file = await self._get_file(_session, path)

            if not file:
                return ReadResult(success=False, message=f"File not found: {path}")

            if file.is_directory:
                return ReadResult(
                    success=False,
                    message=f"Path is a directory, not a file: {path}",
                )

            content = await self._read_content(path, _session)
            if content is None:
                return ReadResult(success=False, message=f"File content not found: {path}")

            return self._paginate_content(content, path, offset, limit)

        except Exception as e:
            logger.error("Read failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Core Operations: Write
    # =========================================================================

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
    ) -> WriteResult:
        """Write file content."""
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

        content_bytes = content.encode()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        size_bytes = len(content_bytes)

        _session, owns = await self._resolve_session(session)
        try:
            existing = await self._get_file(_session, path, include_deleted=True)

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

                # Update metadata (increment version first so _save_version uses new number)
                existing.current_version += 1
                existing.content_hash = content_hash
                existing.size_bytes = size_bytes
                existing.updated_at = datetime.now(UTC)

                # Save version after incrementing
                old_content = await self._read_content(path, _session)
                if old_content is not None:
                    await self._save_version(
                        _session, existing, old_content, content, created_by,
                    )

                # Write content first, then commit. If disk write fails the
                # except block rolls back the version record. An orphaned file
                # on disk (commit fails after write) is inert; phantom metadata
                # (commit succeeds but write fails) breaks reads. See docs/internals/fs.md.
                await self._write_content(path, content, _session)
                if owns:
                    await self._commit(_session)
                else:
                    await _session.flush()
                created = False
                version = existing.current_version
            else:
                await self._ensure_parent_dirs(_session, path)

                new_file = self._file_model(
                    path=path,
                    name=name,
                    parent_path=split_path(path)[0],
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    mime_type=guess_mime_type(name),
                )
                _session.add(new_file)

                # Save initial snapshot (version 1)
                await self._save_version(
                    _session, new_file, "", content, created_by,
                )

                # Write content first, then commit — see comment above.
                await self._write_content(path, content, _session)
                if owns:
                    await self._commit(_session)
                else:
                    await _session.flush()

                created = True
                version = 1

            return WriteResult(
                success=True,
                message=f"{'Created' if created else 'Updated'}: {path} (v{version})",
                file_path=path,
                created=created,
                version=version,
            )

        except Exception as e:
            logger.error("Write failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Core Operations: Edit
    # =========================================================================

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession | None = None,
    ) -> EditResult:
        """Edit file using smart replacement."""
        valid, error = validate_path(path)
        if not valid:
            return EditResult(success=False, message=error)

        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)

            if not file:
                return EditResult(success=False, message=f"File not found: {path}")

            if file.is_directory:
                return EditResult(success=False, message=f"Cannot edit directory: {path}")

            content = await self._read_content(path, _session)
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
            await self._save_version(_session, file, content, new_content, created_by)

            # Write content first, then commit — see write() comment.
            await self._write_content(path, new_content, _session)
            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            return EditResult(
                success=True,
                message=f"Edit applied to {path} (v{file.current_version})",
                file_path=path,
                version=file.current_version,
            )

        except Exception as e:
            logger.error("Edit failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Core Operations: Delete
    # =========================================================================

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
    ) -> DeleteResult:
        """Delete a file or directory."""
        valid, error = validate_path(path)
        if not valid:
            return DeleteResult(success=False, message=error)

        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)

            if not file:
                return DeleteResult(success=False, message=f"File not found: {path}")

            model = self._file_model
            if permanent:
                if file.is_directory:
                    result = await _session.execute(
                        select(model).where(
                            model.path.startswith(path + "/"),  # type: ignore[union-attr]
                        )
                    )
                    for child in result.scalars().all():
                        await self._delete_versions(_session, child.id)
                        await self._delete_content(child.path, _session)
                        await _session.delete(child)

                await self._delete_versions(_session, file.id)
                await self._delete_content(path, _session)
                await _session.delete(file)
            else:
                now = datetime.now(UTC)
                if file.is_directory:
                    children_result = await _session.execute(
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

            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            return DeleteResult(
                success=True,
                message=f"{'Permanently deleted' if permanent else 'Moved to trash'}: {path}",
                file_path=path,
                permanent=permanent,
            )

        except Exception as e:
            logger.error("Delete failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Directory Operations
    # =========================================================================

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
    ) -> MkdirResult:
        """Create a directory using dialect-aware upsert."""
        valid, error = validate_path(path)
        if not valid:
            return MkdirResult(success=False, message=error)

        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            existing = await self._get_file(_session, path)
            if existing:
                if existing.is_directory:
                    return MkdirResult(
                        success=True,
                        message=f"Directory already exists: {path}",
                        path=path,
                        created_dirs=[],
                    )
                return MkdirResult(success=False, message=f"Path exists as file: {path}")

            dirs_to_create: list[str] = []
            current = path

            while current != "/":
                existing = await self._get_file(_session, current)
                if existing:
                    if not existing.is_directory:
                        return MkdirResult(
                            success=False,
                            message=f"Path exists as file: {current}",
                        )
                    break
                dirs_to_create.insert(0, current)
                if not parents and len(dirs_to_create) > 1:
                    return MkdirResult(
                        success=False,
                        message=f"Parent directory does not exist: {split_path(current)[0]}",
                    )
                current = split_path(current)[0]

            created_dirs: list[str] = []
            for dir_path in dirs_to_create:
                parent, name = split_path(dir_path)
                rowcount = await upsert_file(
                    _session,
                    self.dialect,
                    values={
                        "id": str(uuid.uuid4()),
                        "path": dir_path,
                        "name": name,
                        "parent_path": parent,
                        "is_directory": True,
                        "current_version": 1,
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                    },
                    conflict_keys=["path"],
                    model=self._file_model,
                    schema=self.schema,
                )
                if rowcount > 0:
                    created_dirs.append(dir_path)

            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            if created_dirs:
                return MkdirResult(
                    success=True,
                    message=f"Created directory: {path}",
                    path=path,
                    created_dirs=created_dirs,
                )
            return MkdirResult(
                success=True,
                message=f"Directory already exists: {path}",
                path=path,
                created_dirs=[],
            )

        except Exception as e:
            logger.error("Mkdir failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
    ) -> ListResult:
        """List files and directories at the given path."""
        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            if path != "/":
                dir_file = await self._get_file(_session, path)
                if not dir_file:
                    return ListResult(success=False, message=f"Directory not found: {path}")
                if not dir_file.is_directory:
                    return ListResult(success=False, message=f"Not a directory: {path}")

            model = self._file_model
            if path == "/":
                result = await _session.execute(
                    select(model).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                        model.parent_path.in_(["", "/"]),  # type: ignore[union-attr]
                    )
                )
            else:
                result = await _session.execute(
                    select(model).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                        model.parent_path == path,  # type: ignore[arg-type]
                    )
                )
            entries = [self._file_to_info(f) for f in result.scalars().all()]

            return ListResult(
                success=True,
                message=f"Listed {len(entries)} items in {path}",
                entries=entries,
                path=path,
            )

        except Exception as e:
            logger.error("List dir failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    def _file_to_info(self, f: FileBase) -> FileInfo:
        """Convert a file record to FileInfo."""
        return FileInfo(
            path=f.path,
            name=f.name,
            is_directory=f.is_directory,
            size_bytes=f.size_bytes,
            mime_type=f.mime_type,
            version=f.current_version,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )

    # =========================================================================
    # File Management
    # =========================================================================

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
    ) -> MoveResult:
        """Move a file or directory."""
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

        _session, owns = await self._resolve_session(session)
        try:
            src_file = await self._get_file(_session, src)
            if not src_file:
                return MoveResult(success=False, message=f"Source not found: {src}")

            if src_file.is_directory and dest.startswith(src + "/"):
                return MoveResult(
                    success=False,
                    message=f"Cannot move directory into itself: {dest} is inside {src}",
                )

            dest_file = await self._get_file(_session, dest)
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
                content = await self._read_content(src, _session) or ""
                old_dest_content = await self._read_content(dest, _session) or ""

                # Update dest metadata with source content
                content_bytes = content.encode()
                dest_file.current_version += 1
                dest_file.content_hash = hashlib.sha256(content_bytes).hexdigest()
                dest_file.size_bytes = len(content_bytes)
                dest_file.updated_at = datetime.now(UTC)

                # Save version for dest (records the overwrite in dest's history)
                await self._save_version(
                    _session, dest_file, old_dest_content, content, "move",
                )

                # Write content to dest storage
                await self._write_content(dest, content, _session)

                # Soft-delete src (history retained, recoverable from trash)
                now = datetime.now(UTC)
                src_file.original_path = src_file.path
                src_file.path = to_trash_path(src_file.path, src_file.id)
                src_file.deleted_at = now

                # Single commit — all changes are atomic
                if owns:
                    await self._commit(_session)
                else:
                    await _session.flush()

                # Clean up src content from storage (best-effort, after commit)
                try:
                    await self._delete_content(src, _session)
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
                model = self._file_model
                result = await _session.execute(
                    select(model).where(
                        model.path.startswith(src + "/"),  # type: ignore[union-attr]
                    )
                )
                children = result.scalars().all()

                for desc in children:
                    old_paths.append(desc.path)
                    new_path = dest + desc.path[len(src):]
                    content = await self._read_content(desc.path, _session)
                    if content is not None:
                        await self._write_content(new_path, content, _session)
                    desc.path = new_path
                    desc.name = split_path(new_path)[1]
                    desc.parent_path = split_path(new_path)[0]

            content = await self._read_content(src, _session)
            if content is not None:
                await self._write_content(dest, content, _session)

            src_file.path = dest
            src_file.name = split_path(dest)[1]
            src_file.parent_path = split_path(dest)[0]
            src_file.updated_at = datetime.now(UTC)

            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            for old_path in old_paths:
                try:
                    await self._delete_content(old_path, _session)
                except Exception:
                    logger.warning("Failed to clean up old content at %s", old_path)

            return MoveResult(
                success=True,
                message=f"Moved {src} to {dest}",
                old_path=src,
                new_path=dest,
            )

        except Exception as e:
            logger.error("Move failed for %s -> %s: %s", src, dest, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
    ) -> WriteResult:
        """Copy a file."""
        src = normalize_path(src)

        _session, owns = await self._resolve_session(session)
        try:
            src_file = await self._get_file(_session, src)
            if not src_file:
                return WriteResult(success=False, message=f"Source not found: {src}")

            if src_file.is_directory:
                return WriteResult(success=False, message="Directory copy not yet implemented")

            content = await self._read_content(src, _session)
            if content is None:
                return WriteResult(success=False, message=f"Source content not found: {src}")
        except Exception as e:
            logger.error("Copy failed for %s: %s", src, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise
        finally:
            if owns:
                await _session.close()

        return await self.write(dest, content, created_by="copy", session=session)

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> bool:
        """Check if a file or directory exists."""
        valid, _ = validate_path(path)
        if not valid:
            return False

        path = normalize_path(path)
        if path == "/":
            return True

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)
            return file is not None

        finally:
            if owns:
                await _session.close()

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileInfo | None:
        """Get metadata for a file or directory."""
        valid, _ = validate_path(path)
        if not valid:
            return None

        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)
            if not file:
                return None
            return self._file_to_info(file)

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Version Control
    # =========================================================================

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> list[VersionInfo]:
        """List all saved versions for a file."""
        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)
            if not file:
                return []

            fv_model = self._file_version_model
            result = await _session.execute(
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

        except Exception as e:
            logger.error("List versions failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
    ) -> RestoreResult:
        """Restore file to a previous version."""
        path = normalize_path(path)

        version_content = await self.get_version_content(path, version, session=session)
        if version_content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(
            path, version_content, created_by="restore", session=session,
        )

        return RestoreResult(
            success=True,
            message=f"Restored {path} to version {version}",
            file_path=path,
            restored_version=version,
            current_version=write_result.version,
        )

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
    ) -> str | None:
        """Get the content of a specific version using diff reconstruction."""
        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            file = await self._get_file(_session, path)
            if not file:
                return None

            fv_model = self._file_version_model

            # Find the nearest snapshot at or before the target version
            snapshot_result = await _session.execute(
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
            chain_result = await _session.execute(
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
                    f"Version {version} of {path}: content hash mismatch "
                    f"(expected {expected_hash[:12]}…, got {actual_hash[:12]}…)"
                )

            return content

        except Exception as e:
            logger.error(
                "Get version content failed for %s v%s: %s", path, version, e, exc_info=True
            )
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    # =========================================================================
    # Trash Operations
    # =========================================================================

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> ListResult:
        """List all soft-deleted files."""
        _session, owns = await self._resolve_session(session)
        try:
            model = self._file_model
            result = await _session.execute(
                select(model).where(
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            files = result.scalars().all()

            entries = [
                FileInfo(
                    path=f.original_path or f.path,
                    name=f.name,
                    is_directory=f.is_directory,
                    size_bytes=f.size_bytes,
                    version=f.current_version,
                    created_at=f.created_at,
                    updated_at=f.deleted_at,
                )
                for f in files
            ]

            return ListResult(
                success=True,
                message=f"Found {len(entries)} items in trash",
                entries=entries,
                path="/__trash__",
            )

        except Exception as e:
            logger.error("List trash failed: %s", e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> RestoreResult:
        """Restore a file from trash."""
        path = normalize_path(path)

        _session, owns = await self._resolve_session(session)
        try:
            model = self._file_model
            result = await _session.execute(
                select(model).where(
                    model.original_path == path,  # type: ignore[arg-type]
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            file = result.scalar_one_or_none()

            if not file:
                return RestoreResult(success=False, message=f"File not in trash: {path}")

            original = file.original_path or path

            # If path is occupied, overwrite the occupant (git restore semantics).
            # The occupant's content is preserved in its version history.
            existing = await self._get_file(_session, original)
            if existing and existing.id != file.id:
                await self._delete_versions(_session, existing.id)
                await _session.delete(existing)
                await _session.flush()  # flush delete before path reassignment

            file.path = original
            file.original_path = None
            file.deleted_at = None
            file.updated_at = datetime.now(UTC)

            if file.is_directory:
                children_result = await _session.execute(
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
                    child_existing = await self._get_file(_session, child_original)
                    if child_existing and child_existing.id != child.id:
                        await self._delete_versions(_session, child_existing.id)
                        await _session.delete(child_existing)
                        had_occupants = True
                if had_occupants:
                    await _session.flush()  # flush deletes before path reassignment

                for child in children:
                    child.path = child.original_path or child.path
                    child.original_path = None
                    child.deleted_at = None
                    child.updated_at = datetime.now(UTC)

            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            return RestoreResult(
                success=True,
                message=f"Restored from trash: {path}",
                file_path=path,
            )

        except Exception as e:
            logger.error("Restore from trash failed for %s: %s", path, e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

    async def empty_trash(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> DeleteResult:
        """Permanently delete all files in trash."""
        _session, owns = await self._resolve_session(session)
        try:
            model = self._file_model
            result = await _session.execute(
                select(model).where(
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            files = result.scalars().all()

            count = len(files)
            for file in files:
                await self._delete_versions(_session, file.id)
                await self._delete_content(file.original_path or file.path, _session)
                await _session.delete(file)

            if owns:
                await self._commit(_session)
            else:
                await _session.flush()

            return DeleteResult(
                success=True,
                message=f"Permanently deleted {count} items from trash",
                permanent=True,
                total_deleted=count,
            )

        except Exception as e:
            logger.error("Empty trash failed: %s", e, exc_info=True)
            if owns:
                await _session.rollback()
            raise

        finally:
            if owns:
                await _session.close()

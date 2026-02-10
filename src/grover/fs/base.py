"""BaseFileSystem — shared SQL logic, dialect-aware."""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Generic, TypeVar

from sqlmodel import select

from grover.fs.dialect import upsert_file
from grover.models.files import (
    SNAPSHOT_INTERVAL,
    File,
    FileBase,
    FileVersion,
    FileVersionBase,
    compute_diff,
    reconstruct_version,
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

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Constants for read operations
DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024  # 50KB

F = TypeVar("F", bound=FileBase)
FV = TypeVar("FV", bound=FileVersionBase)


class BaseFileSystem(ABC, Generic[F, FV]):
    """Base file system with shared logic for both backends.

    Subclasses must implement:
    - _get_session(): Get database session
    - _read_content(path): Read file content from storage
    - _write_content(path, content): Write content to storage
    - _delete_content(path): Delete content from storage
    - _content_exists(path): Check if content exists in storage
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
        self.in_transaction = False
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
        DatabaseFileSystem returns a session from its injected factory. The
        caller is responsible for committing via ``_commit`` when done.
        """
        ...

    @abstractmethod
    async def _read_content(self, path: str) -> str | None:
        """Read raw file content from the storage backend.

        ``path`` is already normalized. Return the file text if it exists,
        or ``None`` if no content is stored at that path. The base class
        handles all metadata lookups — this method only touches bytes.
        """
        ...

    @abstractmethod
    async def _write_content(self, path: str, content: str) -> None:
        """Persist raw file content to the storage backend.

        ``path`` is already normalized. For LocalFileSystem this writes to
        disk; for DatabaseFileSystem this updates the ``content`` column on
        the file row. Parent directories are handled by the base class.
        """
        ...

    @abstractmethod
    async def _delete_content(self, path: str) -> None:
        """Remove raw file content from the storage backend.

        Called during permanent deletes and moves. For LocalFileSystem this
        removes the file from disk. For DatabaseFileSystem this is a no-op
        because content lives in the file row that the base class
        already deletes.
        """
        ...

    @abstractmethod
    async def _content_exists(self, path: str) -> bool:
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
        change_summary: str | None = None,
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

        version = self._file_version_model(
            file_id=file.id,
            version=version_num,
            is_snapshot=is_snap or not old_content,
            content=content,
            content_hash=hashlib.sha256(new_content.encode()).hexdigest(),
            size_bytes=len(new_content.encode()),
            created_by=created_by,
            change_summary=change_summary,
        )
        session.add(version)

    async def _ensure_parent_dirs(self, session: AsyncSession, path: str) -> None:
        """Ensure all parent directories exist in the database."""
        parts = path.split("/")
        for i in range(2, len(parts)):
            dir_path = "/".join(parts[:i])
            if not dir_path:
                continue

            dir_name = parts[i - 1]
            await upsert_file(
                session,
                self.dialect,
                values={
                    "id": str(uuid.uuid4()),
                    "path": dir_path,
                    "name": dir_name,
                    "is_directory": True,
                    "current_version": 1,
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                },
                conflict_keys=["path"],
                model=self._file_model,
                schema=self.schema,
            )

    def _format_read_output(
        self,
        content: str,
        path: str,
        offset: int,
        limit: int,
    ) -> ReadResult:
        """Format file content with line numbers and pagination."""
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
            )

        output_lines = []
        bytes_read = 0
        end_line = min(len(lines), offset + limit)

        for i in range(offset, end_line):
            line = lines[i]

            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."

            line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)
            if bytes_read + line_bytes > MAX_BYTES:
                break

            output_lines.append(line)
            bytes_read += line_bytes

        formatted_lines = [
            f"{str(i + offset + 1).zfill(5)}| {line}"
            for i, line in enumerate(output_lines)
        ]

        formatted_content = "<file>\n"
        formatted_content += "\n".join(formatted_lines)

        last_read_line = offset + len(output_lines)
        has_more = total_lines > last_read_line

        if has_more:
            formatted_content += (
                f"\n\n(File has more lines. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        else:
            formatted_content += f"\n\n(End of file - total {total_lines} lines)"

        formatted_content += "\n</file>"

        return ReadResult(
            success=True,
            message=f"Read {len(output_lines)} lines from {path}",
            content=formatted_content,
            file_path=path,
            total_lines=total_lines,
            lines_read=len(output_lines),
            truncated=has_more,
        )

    # =========================================================================
    # Core Operations: Read
    # =========================================================================

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> ReadResult:
        """Read file content with pagination."""
        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        if is_trash_path(path):
            return ReadResult(success=False, message=f"Cannot read from trash: {path}")

        session = await self._get_session()

        try:
            file = await self._get_file(session, path)

            if not file:
                return ReadResult(success=False, message=f"File not found: {path}")

            if file.is_directory:
                return ReadResult(
                    success=False,
                    message=f"Path is a directory, not a file: {path}",
                )

            content = await self._read_content(path)
            if content is None:
                return ReadResult(success=False, message=f"File content not found: {path}")

            return self._format_read_output(content, path, offset, limit)

        finally:
            pass

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

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        size_bytes = len(content.encode())

        session = await self._get_session()
        try:
            existing = await self._get_file(session, path, include_deleted=True)

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
                old_content = await self._read_content(path)
                if old_content is not None:
                    await self._save_version(
                        session, existing, old_content, content, created_by,
                    )

                await self._commit(session)
                created = False
                version = existing.current_version
            else:
                await self._ensure_parent_dirs(session, path)

                new_file = self._file_model(
                    path=path,
                    name=name,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    mime_type=guess_mime_type(name),
                )
                session.add(new_file)

                # Save initial snapshot (version 1)
                await self._save_version(
                    session, new_file, "", content, created_by,
                )
                await self._commit(session)

                created = True
                version = 1

            await self._write_content(path, content)

            return WriteResult(
                success=True,
                message=f"{'Created' if created else 'Updated'}: {path} (v{version})",
                file_path=path,
                created=created,
                version=version,
            )
        finally:
            pass

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
    ) -> EditResult:
        """Edit file using smart replacement."""
        valid, error = validate_path(path)
        if not valid:
            return EditResult(success=False, message=error)

        path = normalize_path(path)

        session = await self._get_session()
        try:
            file = await self._get_file(session, path)

            if not file:
                return EditResult(success=False, message=f"File not found: {path}")

            if file.is_directory:
                return EditResult(success=False, message=f"Cannot edit directory: {path}")

            content = await self._read_content(path)
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
            file.current_version += 1
            file.content_hash = hashlib.sha256(new_content.encode()).hexdigest()
            file.size_bytes = len(new_content.encode())
            file.updated_at = datetime.now(UTC)

            # Save version after incrementing
            await self._save_version(session, file, content, new_content, created_by)

            await self._commit(session)

            await self._write_content(path, new_content)

            return EditResult(
                success=True,
                message=f"Edit applied to {path} (v{file.current_version})",
                file_path=path,
                version=file.current_version,
            )
        finally:
            pass

    # =========================================================================
    # Core Operations: Delete
    # =========================================================================

    async def delete(self, path: str, permanent: bool = False) -> DeleteResult:
        """Delete a file or directory."""
        valid, error = validate_path(path)
        if not valid:
            return DeleteResult(success=False, message=error)

        path = normalize_path(path)

        session = await self._get_session()
        try:
            file = await self._get_file(session, path)

            if not file:
                return DeleteResult(success=False, message=f"File not found: {path}")

            if permanent:
                model = self._file_model
                if file.is_directory:
                    result = await session.execute(
                        select(model).where(
                            model.path.startswith(path + "/"),  # type: ignore[union-attr]
                        )
                    )
                    for child in result.scalars().all():
                        await self._delete_content(child.path)
                        await session.delete(child)

                await self._delete_content(path)
                await session.delete(file)
            else:
                file.original_path = file.path
                file.path = to_trash_path(file.path, file.id)
                file.deleted_at = datetime.now(UTC)

            await self._commit(session)

            return DeleteResult(
                success=True,
                message=f"{'Permanently deleted' if permanent else 'Moved to trash'}: {path}",
                file_path=path,
                permanent=permanent,
            )
        finally:
            pass

    # =========================================================================
    # Directory Operations
    # =========================================================================

    async def mkdir(self, path: str, parents: bool = True) -> MkdirResult:
        """Create a directory using dialect-aware upsert."""
        valid, error = validate_path(path)
        if not valid:
            return MkdirResult(success=False, message=error)

        path = normalize_path(path)

        session = await self._get_session()
        try:
            existing = await self._get_file(session, path)
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
                existing = await self._get_file(session, current)
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
                _, name = split_path(dir_path)
                rowcount = await upsert_file(
                    session,
                    self.dialect,
                    values={
                        "id": str(uuid.uuid4()),
                        "path": dir_path,
                        "name": name,
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

            await self._commit(session)

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
        finally:
            pass

    async def list_dir(self, path: str = "/") -> ListResult:
        """List files and directories at the given path."""
        path = normalize_path(path)

        session = await self._get_session()
        try:
            if path != "/":
                dir_file = await self._get_file(session, path)
                if not dir_file:
                    return ListResult(success=False, message=f"Directory not found: {path}")
                if not dir_file.is_directory:
                    return ListResult(success=False, message=f"Not a directory: {path}")

            model = self._file_model
            result = await session.execute(
                select(model).where(
                    model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                )
            )
            all_files = result.scalars().all()

            prefix = path if path == "/" else path + "/"
            entries: list[FileInfo] = []
            seen: set[str] = set()

            for f in all_files:
                if f.path == path:
                    continue

                if path == "/":
                    if (
                        f.path.startswith("/")
                        and "/" not in f.path[1:]
                        and f.name not in seen
                    ):
                        seen.add(f.name)
                        entries.append(self._file_to_info(f))
                elif f.path.startswith(prefix):
                    remainder = f.path[len(prefix) :]
                    if "/" not in remainder and f.name not in seen:
                        seen.add(f.name)
                        entries.append(self._file_to_info(f))

            return ListResult(
                success=True,
                message=f"Listed {len(entries)} items in {path}",
                entries=entries,
                path=path,
            )
        finally:
            pass

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

    async def move(self, src: str, dest: str) -> MoveResult:
        """Move a file or directory."""
        src = normalize_path(src)
        dest = normalize_path(dest)

        valid, error = validate_path(dest)
        if not valid:
            return MoveResult(success=False, message=error)

        session = await self._get_session()
        try:
            src_file = await self._get_file(session, src)
            if not src_file:
                return MoveResult(success=False, message=f"Source not found: {src}")

            if src_file.is_directory:
                model = self._file_model
                result = await session.execute(
                    select(model).where(
                        model.path.startswith(src + "/"),  # type: ignore[union-attr]
                    )
                )
                for desc in result.scalars().all():
                    new_path = dest + desc.path[len(src) :]
                    content = await self._read_content(desc.path)
                    if content:
                        await self._write_content(new_path, content)
                        await self._delete_content(desc.path)
                    desc.path = new_path

            content = await self._read_content(src)
            if content:
                await self._write_content(dest, content)
                await self._delete_content(src)

            src_file.path = dest
            src_file.name = split_path(dest)[1]
            src_file.updated_at = datetime.now(UTC)

            await self._commit(session)

            return MoveResult(
                success=True,
                message=f"Moved {src} to {dest}",
                old_path=src,
                new_path=dest,
            )
        finally:
            pass

    async def copy(self, src: str, dest: str) -> WriteResult:
        """Copy a file."""
        src = normalize_path(src)

        session = await self._get_session()
        try:
            src_file = await self._get_file(session, src)
            if not src_file:
                return WriteResult(success=False, message=f"Source not found: {src}")

            if src_file.is_directory:
                return WriteResult(success=False, message="Directory copy not yet implemented")

            content = await self._read_content(src)
            if content is None:
                return WriteResult(success=False, message=f"Source content not found: {src}")
        finally:
            pass

        return await self.write(dest, content, created_by="copy")

    async def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        path = normalize_path(path)
        if path == "/":
            return True

        session = await self._get_session()
        file = await self._get_file(session, path)
        return file is not None

    async def get_info(self, path: str) -> FileInfo | None:
        """Get metadata for a file or directory."""
        path = normalize_path(path)

        session = await self._get_session()
        file = await self._get_file(session, path)
        if not file:
            return None
        return self._file_to_info(file)

    # =========================================================================
    # Version Control
    # =========================================================================

    async def list_versions(self, path: str) -> list[VersionInfo]:
        """List all saved versions for a file."""
        path = normalize_path(path)

        session = await self._get_session()
        try:
            file = await self._get_file(session, path)
            if not file:
                return []

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
                    change_summary=v.change_summary,
                )
                for v in versions
            ]
        finally:
            pass

    async def restore_version(self, path: str, version: int) -> RestoreResult:
        """Restore file to a previous version."""
        path = normalize_path(path)

        version_content = await self.get_version_content(path, version)
        if version_content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(path, version_content, created_by="restore")

        return RestoreResult(
            success=True,
            message=f"Restored {path} to version {version}",
            file_path=path,
            restored_version=version,
            current_version=write_result.version,
        )

    async def get_version_content(self, path: str, version: int) -> str | None:
        """Get the content of a specific version using diff reconstruction."""
        path = normalize_path(path)

        session = await self._get_session()
        try:
            file = await self._get_file(session, path)
            if not file:
                return None

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
            return reconstruct_version(entries)
        finally:
            pass

    # =========================================================================
    # Trash Operations
    # =========================================================================

    async def list_trash(self) -> ListResult:
        """List all soft-deleted files."""
        session = await self._get_session()
        try:
            model = self._file_model
            result = await session.execute(
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
        finally:
            pass

    async def restore_from_trash(self, path: str) -> RestoreResult:
        """Restore a file from trash."""
        path = normalize_path(path)

        session = await self._get_session()
        try:
            model = self._file_model
            result = await session.execute(
                select(model).where(
                    model.original_path == path,  # type: ignore[arg-type]
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            file = result.scalar_one_or_none()

            if not file:
                return RestoreResult(success=False, message=f"File not in trash: {path}")

            file.path = file.original_path or path
            file.original_path = None
            file.deleted_at = None
            file.updated_at = datetime.now(UTC)

            await self._commit(session)

            return RestoreResult(
                success=True,
                message=f"Restored from trash: {path}",
                file_path=path,
            )
        finally:
            pass

    async def empty_trash(self) -> DeleteResult:
        """Permanently delete all files in trash."""
        session = await self._get_session()
        try:
            model = self._file_model
            result = await session.execute(
                select(model).where(
                    model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
                )
            )
            files = result.scalars().all()

            count = len(files)
            for file in files:
                await self._delete_content(file.original_path or file.path)
                await session.delete(file)

            await self._commit(session)

            return DeleteResult(
                success=True,
                message=f"Permanently deleted {count} items from trash",
                permanent=True,
                total_deleted=count,
            )
        finally:
            pass

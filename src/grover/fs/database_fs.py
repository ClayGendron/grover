"""DatabaseFileSystem — pure SQL storage, stateless, no base class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlmodel import select

from .directories import DirectoryService
from .exceptions import GroverError
from .metadata import MetadataService
from .operations import (
    copy_file,
    delete_file,
    edit_file,
    list_dir_db,
    move_file,
    read_file,
    write_file,
)
from .trash import TrashService
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    ListResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    WriteResult,
)
from .utils import normalize_path
from .versioning import VersioningService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase, FileVersionBase

logger = logging.getLogger(__name__)


class DatabaseFileSystem:
    """Database-backed file system — stateless, sessions provided per-operation.

    All content is stored in the database — portable and consistent
    across deployments. Works with SQLite, PostgreSQL, MSSQL, etc.

    This class holds only configuration (dialect, models, schema) and
    composed services. It has no session factory, no mutable state,
    and is safe for concurrent use from multiple requests.

    Implements ``StorageBackend``, ``SupportsVersions``, and
    ``SupportsTrash`` protocols.
    """

    def __init__(
        self,
        dialect: str = "sqlite",
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        schema: str | None = None,
    ) -> None:
        from grover.models.files import File, FileVersion

        fm: type[FileBase] = file_model or File  # type: ignore[assignment]
        fvm: type[FileVersionBase] = file_version_model or FileVersion  # type: ignore[assignment]

        self.dialect = dialect
        self.schema = schema
        self._file_model = fm
        self._file_version_model = fvm

        # Composed services
        self.metadata = MetadataService(fm)
        self.versioning = VersioningService(fm, fvm)
        self.directories = DirectoryService(fm, dialect, schema)
        self.trash = TrashService(fm, self.versioning, self._delete_content)

    @property
    def file_model(self) -> type[FileBase]:
        return self._file_model

    @property
    def file_version_model(self) -> type[FileVersionBase]:
        return self._file_version_model

    def _require_session(self, session: AsyncSession | None) -> AsyncSession:
        if session is None:
            raise GroverError("DatabaseFileSystem requires a session")
        return session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """No-op — DFS is stateless."""

    async def close(self) -> None:
        """No-op — DFS has no resources to release."""

    # ------------------------------------------------------------------
    # Content helpers (DB-specific)
    # ------------------------------------------------------------------

    async def _read_content(self, path: str, session: AsyncSession) -> str | None:
        path = normalize_path(path)
        model = self._file_model
        result = await session.execute(
            select(model.content).where(  # type: ignore[arg-type]
                model.path == path,  # type: ignore[arg-type]
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
        path = normalize_path(path)
        model = self._file_model
        result = await session.execute(
            select(model).where(
                model.path == path,  # type: ignore[arg-type]
            )
        )
        file = result.scalar_one_or_none()
        if file:
            file.content = content

    async def _delete_content(self, path: str, session: AsyncSession) -> None:
        pass  # Content lives in the file record

    # ------------------------------------------------------------------
    # Core protocol: StorageBackend
    # ------------------------------------------------------------------

    async def read(
        self, path: str, offset: int = 0, limit: int = 2000,
        *, session: AsyncSession | None = None,
    ) -> ReadResult:
        sess = self._require_session(session)
        return await read_file(
            path, offset, limit, sess,
            metadata=self.metadata,
            read_content=self._read_content,
        )

    async def write(
        self, path: str, content: str, created_by: str = "agent",
        *, overwrite: bool = True,
        session: AsyncSession | None = None,
    ) -> WriteResult:
        sess = self._require_session(session)
        return await write_file(
            path, content, created_by, overwrite, sess,
            metadata=self.metadata,
            versioning=self.versioning,
            directories=self.directories,
            file_model=self._file_model,
            read_content=self._read_content,
            write_content=self._write_content,
        )

    async def edit(
        self, path: str, old_string: str, new_string: str,
        replace_all: bool = False, created_by: str = "agent",
        *, session: AsyncSession | None = None,
    ) -> EditResult:
        sess = self._require_session(session)
        return await edit_file(
            path, old_string, new_string, replace_all, created_by, sess,
            metadata=self.metadata,
            versioning=self.versioning,
            read_content=self._read_content,
            write_content=self._write_content,
        )

    async def delete(
        self, path: str, permanent: bool = False,
        *, session: AsyncSession | None = None,
    ) -> DeleteResult:
        sess = self._require_session(session)
        return await delete_file(
            path, permanent, sess,
            metadata=self.metadata,
            versioning=self.versioning,
            file_model=self._file_model,
            delete_content=self._delete_content,
        )

    async def mkdir(
        self, path: str, parents: bool = True,
        *, session: AsyncSession | None = None,
    ) -> MkdirResult:
        sess = self._require_session(session)
        created_dirs, error = await self.directories.mkdir(
            sess, path, parents, self.metadata.get_file,
        )
        if error is not None:
            return MkdirResult(success=False, message=error)
        path = normalize_path(path)
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

    async def list_dir(
        self, path: str = "/",
        *, session: AsyncSession | None = None,
    ) -> ListResult:
        sess = self._require_session(session)
        return await list_dir_db(
            path, sess,
            metadata=self.metadata,
            file_model=self._file_model,
        )

    async def exists(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> bool:
        sess = self._require_session(session)
        return await self.metadata.exists(sess, path)

    async def get_info(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> FileInfo | None:
        sess = self._require_session(session)
        return await self.metadata.get_info(sess, path)

    async def move(
        self, src: str, dest: str,
        *, session: AsyncSession | None = None,
    ) -> MoveResult:
        sess = self._require_session(session)
        return await move_file(
            src, dest, sess,
            metadata=self.metadata,
            versioning=self.versioning,
            file_model=self._file_model,
            read_content=self._read_content,
            write_content=self._write_content,
            delete_content=self._delete_content,
        )

    async def copy(
        self, src: str, dest: str,
        *, session: AsyncSession | None = None,
    ) -> WriteResult:
        sess = self._require_session(session)
        return await copy_file(
            src, dest, sess,
            metadata=self.metadata,
            read_content=self._read_content,
            write_fn=self.write,
        )

    # ------------------------------------------------------------------
    # Capability: SupportsVersions
    # ------------------------------------------------------------------

    async def list_versions(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> ListVersionsResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return ListVersionsResult(success=True, message="File not found", versions=[])
        versions = await self.versioning.list_versions(sess, file)
        return ListVersionsResult(
            success=True,
            message=f"Found {len(versions)} version(s)",
            versions=versions,
        )

    async def get_version_content(
        self, path: str, version: int,
        *, session: AsyncSession | None = None,
    ) -> GetVersionContentResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return GetVersionContentResult(
                success=False, message=f"File not found: {path}",
            )
        content = await self.versioning.get_version_content(sess, file, version)
        if content is None:
            return GetVersionContentResult(
                success=False, message=f"Version {version} not found for {path}",
            )
        return GetVersionContentResult(success=True, message="OK", content=content)

    async def restore_version(
        self, path: str, version: int,
        *, session: AsyncSession | None = None,
    ) -> RestoreResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        vc_result = await self.get_version_content(path, version, session=sess)
        if not vc_result.success or vc_result.content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(
            path, vc_result.content, created_by="restore", session=sess,
        )

        return RestoreResult(
            success=True,
            message=f"Restored {path} to version {version}",
            file_path=path,
            restored_version=version,
            current_version=write_result.version,
        )

    # ------------------------------------------------------------------
    # Capability: SupportsTrash
    # ------------------------------------------------------------------

    async def list_trash(
        self, *, session: AsyncSession | None = None,
    ) -> ListResult:
        sess = self._require_session(session)
        return await self.trash.list_trash(sess)

    async def restore_from_trash(
        self, path: str,
        *, session: AsyncSession | None = None,
    ) -> RestoreResult:
        sess = self._require_session(session)
        return await self.trash.restore_from_trash(sess, path, self.metadata.get_file)

    async def empty_trash(
        self, *, session: AsyncSession | None = None,
    ) -> DeleteResult:
        sess = self._require_session(session)
        return await self.trash.empty_trash(sess)

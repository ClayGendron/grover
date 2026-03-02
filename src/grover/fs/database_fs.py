"""DatabaseFileSystem — pure SQL storage, stateless, no base class."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlmodel import select

from grover.models.chunks import FileChunk
from grover.models.connections import FileConnection
from grover.models.files import File, FileVersion
from grover.types.operations import (
    ConnectionResult,
    DeleteResult,
    EditResult,
    FileInfoResult,
    GetVersionContentResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    VerifyVersionResult,
    WriteResult,
)
from grover.types.search import (
    FileSearchCandidate,
    GlobEvidence,
    GlobResult,
    GrepEvidence,
    GrepResult,
    ListDirResult,
    TrashResult,
    TreeEvidence,
    TreeResult,
    VersionEvidence,
    VersionResult,
)
from grover.types.search import (
    LineMatch as SearchLineMatch,
)

from .chunks import ChunkService
from .connections import ConnectionService
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
from .utils import (
    compile_glob,
    glob_to_sql_like,
    has_binary_extension,
    normalize_path,
)
from .versioning import VersioningService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunks import FileChunkBase
    from grover.models.connections import FileConnectionBase
    from grover.models.files import FileBase, FileVersionBase

    from .sharing import SharingService

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
        file_chunk_model: type[FileChunkBase] | None = None,
        file_connection_model: type[FileConnectionBase] | None = None,
        schema: str | None = None,
    ) -> None:
        self.dialect = dialect
        self.schema = schema
        self._file_model: type[FileBase] = file_model or File
        self._file_version_model: type[FileVersionBase] = file_version_model or FileVersion
        self._file_chunk_model: type[FileChunkBase] = file_chunk_model or FileChunk
        self._file_connection_model: type[FileConnectionBase] = (
            file_connection_model or FileConnection
        )

        # Composed services
        self.metadata = MetadataService(self._file_model)
        self.versioning = VersioningService(self._file_model, self._file_version_model)
        self.directories = DirectoryService(self._file_model, dialect, schema)
        self.trash = TrashService(self._file_model, self.versioning, self._delete_content)
        self.chunks = ChunkService(self._file_chunk_model)
        self.connections = ConnectionService(self._file_connection_model)

    @property
    def file_model(self) -> type[FileBase]:
        return self._file_model

    @property
    def file_version_model(self) -> type[FileVersionBase]:
        return self._file_version_model

    @property
    def file_chunk_model(self) -> type[FileChunkBase]:
        return self._file_chunk_model

    @property
    def file_connection_model(self) -> type[FileConnectionBase]:
        return self._file_connection_model

    @staticmethod
    def _require_session(session: AsyncSession | None) -> AsyncSession:
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
            select(model.content).where(
                model.path == path,
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
                model.path == path,
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
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ReadResult:
        sess = self._require_session(session)
        return await read_file(
            path,
            offset,
            limit,
            sess,
            metadata=self.metadata,
            read_content=self._read_content,
        )

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        sess = self._require_session(session)
        return await write_file(
            path,
            content,
            created_by,
            overwrite,
            sess,
            metadata=self.metadata,
            versioning=self.versioning,
            directories=self.directories,
            file_model=self._file_model,
            read_content=self._read_content,
            write_content=self._write_content,
            owner_id=owner_id,
        )

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> EditResult:
        sess = self._require_session(session)
        return await edit_file(
            path,
            old_string,
            new_string,
            replace_all,
            created_by,
            sess,
            metadata=self.metadata,
            versioning=self.versioning,
            read_content=self._read_content,
            write_content=self._write_content,
        )

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        sess = self._require_session(session)
        return await delete_file(
            path,
            permanent,
            sess,
            metadata=self.metadata,
            versioning=self.versioning,
            file_model=self._file_model,
            delete_content=self._delete_content,
        )

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        sess = self._require_session(session)
        created_dirs, error = await self.directories.mkdir(
            sess,
            path,
            parents,
            self.metadata.get_file,
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
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListDirResult:
        sess = self._require_session(session)
        return await list_dir_db(
            path,
            sess,
            metadata=self.metadata,
            file_model=self._file_model,
        )

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> bool:
        sess = self._require_session(session)
        return await self.metadata.exists(sess, path)

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfoResult | None:
        sess = self._require_session(session)
        return await self.metadata.get_info(sess, path)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        follow: bool = False,
        sharing: SharingService | None = None,
        user_id: str | None = None,
    ) -> MoveResult:
        sess = self._require_session(session)
        return await move_file(
            src,
            dest,
            sess,
            metadata=self.metadata,
            versioning=self.versioning,
            directories=self.directories,
            file_model=self._file_model,
            read_content=self._read_content,
            write_content=self._write_content,
            delete_content=self._delete_content,
            follow=follow,
            sharing=sharing,
        )

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        sess = self._require_session(session)
        return await copy_file(
            src,
            dest,
            sess,
            metadata=self.metadata,
            read_content=self._read_content,
            write_fn=self.write,
        )

    # ------------------------------------------------------------------
    # Search / Query Operations
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GlobResult:
        sess = self._require_session(session)
        path = normalize_path(path)

        if not pattern:
            return GlobResult(
                success=False,
                message="Empty glob pattern",
                pattern=pattern,
            )

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self.metadata.get_file(sess, path)
            if not dir_file:
                return GlobResult(
                    success=False,
                    message=f"Directory not found: {path}",
                    pattern=pattern,
                )
            if not dir_file.is_directory:
                return GlobResult(
                    success=False,
                    message=f"Not a directory: {path}",
                    pattern=pattern,
                )

        model = self._file_model

        # Try SQL pre-filter for performance
        like_pattern = glob_to_sql_like(pattern, path)
        if like_pattern is not None:
            result = await sess.execute(
                select(model).where(
                    model.path.like(like_pattern, escape="\\"),  # type: ignore[union-attr]
                    model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                )
            )
            candidates = list(result.scalars().all())
        else:
            # Fall back: load all non-deleted files under path
            if path == "/":
                result = await sess.execute(
                    select(model).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
            else:
                result = await sess.execute(
                    select(model).where(
                        model.path.like(path + "/%", escape="\\"),  # type: ignore[union-attr]
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
            candidates = list(result.scalars().all())

        # Post-filter with authoritative match (compile once for all candidates)
        glob_regex = compile_glob(pattern, path)
        matched = [
            f for f in candidates if glob_regex is not None and glob_regex.match(f.path) is not None
        ]

        candidates: list[FileSearchCandidate] = []
        for f in matched:
            info = MetadataService.file_to_info(f)
            candidates.append(
                FileSearchCandidate(
                    path=info.path,
                    evidence=[
                        GlobEvidence(
                            strategy="glob",
                            path=info.path,
                            is_directory=info.is_directory,
                            size_bytes=info.size_bytes,
                            mime_type=info.mime_type,
                        )
                    ],
                )
            )

        return GlobResult(
            success=True,
            message=f"Found {len(candidates)} match(es)",
            candidates=candidates,
            pattern=pattern,
        )

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> GrepResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        context_lines = max(0, context_lines)

        # Compile regex
        try:
            regex_pattern = re.escape(pattern) if fixed_string else pattern
            if word_match:
                regex_pattern = r"\b" + regex_pattern + r"\b"
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(regex_pattern, flags)
        except re.error as e:
            return GrepResult(
                success=False,
                message=f"Invalid regex: {e}",
                pattern=pattern,
            )

        # Get candidate files
        if glob_filter:
            glob_result = await self.glob(glob_filter, path, session=sess)
            if not glob_result.success:
                return GrepResult(
                    success=False,
                    message=glob_result.message,
                    pattern=pattern,
                )
            candidate_paths = list(glob_result.files())
        else:
            model = self._file_model
            if path != "/":
                file = await self.metadata.get_file(sess, path)
                if file and not file.is_directory:
                    candidate_paths = [path]
                elif file and file.is_directory:
                    result = await sess.execute(
                        select(model.path).where(
                            model.path.like(path + "/%", escape="\\"),  # type: ignore[union-attr]
                            model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                            model.is_directory.is_(False),  # type: ignore[union-attr]
                        )
                    )
                    candidate_paths = [row[0] for row in result.all()]
                else:
                    return GrepResult(
                        success=False,
                        message=f"Path not found: {path}",
                        pattern=pattern,
                    )
            else:
                result = await sess.execute(
                    select(model.path).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                        model.is_directory.is_(False),  # type: ignore[union-attr]
                    )
                )
                candidate_paths = [row[0] for row in result.all()]

        result_candidates: list[FileSearchCandidate] = []
        files_searched = 0
        files_matched = 0
        truncated = False
        total_matches = 0

        for file_path in candidate_paths:
            if has_binary_extension(file_path):
                continue

            content = await self._read_content(file_path, sess)
            if content is None:
                continue

            files_searched += 1
            lines = content.split("\n")
            file_line_matches: list[SearchLineMatch] = []

            for i, line in enumerate(lines):
                has_match = regex.search(line) is not None
                if invert:
                    has_match = not has_match

                if has_match:
                    ctx_before: tuple[str, ...] = ()
                    ctx_after: tuple[str, ...] = ()
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        ctx_before = tuple(lines[start:i])
                        end = min(len(lines), i + context_lines + 1)
                        ctx_after = tuple(lines[i + 1 : end])

                    file_line_matches.append(
                        SearchLineMatch(
                            line_number=i + 1,
                            line_content=line,
                            context_before=ctx_before,
                            context_after=ctx_after,
                        )
                    )

                    if max_results_per_file > 0 and len(file_line_matches) >= max_results_per_file:
                        break

            if file_line_matches:
                files_matched += 1
                if files_only:
                    file_line_matches = [file_line_matches[0]]

                result_candidates.append(
                    FileSearchCandidate(
                        path=file_path,
                        evidence=[
                            GrepEvidence(
                                strategy="grep",
                                path=file_path,
                                line_matches=tuple(file_line_matches),
                            )
                        ],
                    )
                )
                total_matches += len(file_line_matches)

                if max_results > 0 and total_matches >= max_results:
                    truncated = True
                    break

        if count_only:
            total = files_matched if files_only else total_matches
            return GrepResult(
                success=True,
                message=f"Count: {total}",
                pattern=pattern,
                files_searched=files_searched,
                files_matched=files_matched,
                truncated=truncated,
            )

        return GrepResult(
            success=True,
            message=f"Found {total_matches} match(es) in {files_matched} file(s)",
            candidates=result_candidates,
            pattern=pattern,
            files_searched=files_searched,
            files_matched=files_matched,
            truncated=truncated,
        )

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> TreeResult:
        sess = self._require_session(session)
        path = normalize_path(path)

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self.metadata.get_file(sess, path)
            if not dir_file:
                return TreeResult(
                    success=False,
                    message=f"Directory not found: {path}",
                )
            if not dir_file.is_directory:
                return TreeResult(
                    success=False,
                    message=f"Not a directory: {path}",
                )

        model = self._file_model
        base_depth = path.count("/") if path != "/" else 0

        # Build query conditions
        conditions = [
            model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
        ]
        if path != "/":
            conditions.append(
                model.path.like(path + "/%", escape="\\"),  # type: ignore[union-attr]
            )
        if max_depth is not None:
            max_slashes = base_depth + max_depth
            slash_count = func.length(model.path) - func.length(func.replace(model.path, "/", ""))
            conditions.append(slash_count <= max_slashes)

        result = await sess.execute(select(model).where(*conditions))
        all_files = list(result.scalars().all())

        candidates: list[FileSearchCandidate] = []
        total_files = 0
        total_dirs = 0

        for f in all_files:
            info = MetadataService.file_to_info(f)
            depth = info.path.count("/") - base_depth
            candidates.append(
                FileSearchCandidate(
                    path=info.path,
                    evidence=[
                        TreeEvidence(
                            strategy="tree",
                            path=info.path,
                            depth=depth,
                            is_directory=info.is_directory,
                        )
                    ],
                )
            )
            if f.is_directory:
                total_dirs += 1
            else:
                total_files += 1

        candidates.sort(key=lambda c: c.path)
        return TreeResult(
            success=True,
            message=f"{total_dirs} directories, {total_files} files",
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # Capability: SupportsVersions
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> VersionResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return VersionResult(success=False, message=f"File not found: {path}")
        versions = await self.versioning.list_versions(sess, file)
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
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return GetVersionContentResult(
                success=False,
                message=f"File not found: {path}",
            )
        content = await self.versioning.get_version_content(sess, file, version)
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
        sess = self._require_session(session)
        path = normalize_path(path)
        vc_result = await self.get_version_content(path, version, session=sess)
        if not vc_result.success or vc_result.content is None:
            return RestoreResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(
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
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return VerifyVersionResult(
                success=False,
                message=f"File not found: {path}",
                path=path,
            )
        return await self.versioning.verify_chain(sess, file)

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> list[VerifyVersionResult]:
        sess = self._require_session(session)
        model = self._file_model
        result = await sess.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        results: list[VerifyVersionResult] = []
        for file in result.scalars().all():
            results.append(await self.versioning.verify_chain(sess, file))  # noqa: PERF401
        return results

    # ------------------------------------------------------------------
    # Capability: SupportsTrash
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> TrashResult:
        sess = self._require_session(session)
        return await self.trash.list_trash(sess, owner_id=owner_id)

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> RestoreResult:
        sess = self._require_session(session)
        return await self.trash.restore_from_trash(
            sess, path, self.metadata.get_file, owner_id=owner_id
        )

    async def empty_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        sess = self._require_session(session)
        return await self.trash.empty_trash(sess, owner_id=owner_id)

    # ------------------------------------------------------------------
    # Capability: SupportsFileChunks
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
    ) -> int:
        sess = self._require_session(session)
        return await self.chunks.replace_file_chunks(sess, file_path, chunks)

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> int:
        sess = self._require_session(session)
        return await self.chunks.delete_file_chunks(sess, file_path)

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> list:
        sess = self._require_session(session)
        return await self.chunks.list_file_chunks(sess, file_path)

    # ------------------------------------------------------------------
    # Capability: SupportsConnections
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
        session: AsyncSession | None = None,
    ) -> ConnectionResult:
        sess = self._require_session(session)
        return await self.connections.add_connection(
            sess,
            source_path,
            target_path,
            connection_type,
            weight=weight,
        )

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> ConnectionResult:
        sess = self._require_session(session)
        return await self.connections.delete_connection(
            sess,
            source_path,
            target_path,
            connection_type=connection_type,
        )

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> list:
        sess = self._require_session(session)
        return await self.connections.list_connections(
            sess,
            path,
            direction=direction,
            connection_type=connection_type,
        )

"""DatabaseFileSystem — pure SQL storage with pluggable providers."""

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
    ConnectionListResult,
    ConnectionResult,
    DeleteResult,
    EditResult,
    ExistsResult,
    FileInfoResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
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
)
from grover.types.search import (
    LineMatch as SearchLineMatch,
)

from .chunks import DefaultChunkProvider
from .connections import ConnectionService
from .directories import DirectoryService
from .exceptions import GroverError
from .mixins import (
    ChunkMethodsMixin,
    GraphMethodsMixin,
    SearchMethodsMixin,
    VersionMethodsMixin,
)
from .operations import (
    copy_file,
    delete_file,
    edit_file,
    file_to_info,
    list_dir_db,
    move_file,
    read_file,
    write_file,
)
from .providers.protocols import SupportsStorageQueries
from .trash import TrashService
from .utils import (
    compile_glob,
    glob_to_sql_like,
    has_binary_extension,
    normalize_path,
    validate_path,
)
from .versioning import DefaultVersionProvider

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.chunks import FileChunkBase
    from grover.models.connections import FileConnectionBase
    from grover.models.files import FileBase, FileVersionBase
    from grover.search.protocols import EmbeddingProvider, VectorStore

    from .providers.protocols import (
        ChunkProvider,
        GraphProvider,
        StorageProvider,
        VersionProvider,
    )
    from .sharing import SharingService

logger = logging.getLogger(__name__)


class DatabaseFileSystem(
    GraphMethodsMixin,
    SearchMethodsMixin,
    VersionMethodsMixin,
    ChunkMethodsMixin,
):
    """Database-backed file system with pluggable providers.

    All content is stored in the database by default — portable and
    consistent across deployments.  When a ``storage_provider`` is set,
    content I/O delegates to it (e.g. ``DiskStorageProvider`` for local disk).

    Providers (keyword-only):

    - ``storage_provider`` — external content I/O + queries (None = DB content)
    - ``graph_provider`` — in-memory graph (None = no graph)
    - ``search_provider`` — vector store (None = no vector search)
    - ``embedding_provider`` — embedding model (None = no embeddings)
    - ``version_provider`` — version management (default: ``DefaultVersionProvider``)
    - ``chunk_provider`` — chunk management (default: ``DefaultChunkProvider``)
    """

    def __init__(
        self,
        dialect: str = "sqlite",
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
        file_connection_model: type[FileConnectionBase] | None = None,
        schema: str | None = None,
        *,
        storage_provider: StorageProvider | None = None,
        graph_provider: GraphProvider | None = None,
        search_provider: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        version_provider: VersionProvider | None = None,
        chunk_provider: ChunkProvider | None = None,
    ) -> None:
        self.dialect = dialect
        self.schema = schema
        self._file_model: type[FileBase] = file_model or File
        self._file_version_model: type[FileVersionBase] = file_version_model or FileVersion
        self._file_chunk_model: type[FileChunkBase] = file_chunk_model or FileChunk
        self._file_connection_model: type[FileConnectionBase] = (
            file_connection_model or FileConnection
        )

        # Pluggable providers
        self.storage_provider = storage_provider
        self.graph_provider = graph_provider
        self.search_provider = search_provider
        self.embedding_provider = embedding_provider
        self.version_provider = version_provider or DefaultVersionProvider(
            self._file_model, self._file_version_model
        )
        self.chunk_provider = chunk_provider or DefaultChunkProvider(self._file_chunk_model)

        # Internal services
        self.directories = DirectoryService(self._file_model, dialect, schema)
        self.trash = TrashService(self._file_model, self.version_provider, self._delete_content)
        self.connections = ConnectionService(self._file_connection_model)

        # Validate search dimensions if both providers set
        self._validate_search_dimensions()

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
    # File record lookup (absorbed from MetadataService)
    # ------------------------------------------------------------------

    async def _get_file_record(
        self,
        session: AsyncSession,
        path: str,
        include_deleted: bool = False,
    ) -> FileBase | None:
        """Look up a file record by path.

        Absorbed from the former ``MetadataService.get_file()``.
        """
        path = normalize_path(path)
        model = self._file_model
        conditions = [model.path == path]
        if not include_deleted:
            conditions.append(
                model.deleted_at.is_(None)  # type: ignore[unresolved-attribute]
            )
        result = await session.execute(select(model).where(*conditions))
        return result.scalar_one_or_none()

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
        if self.storage_provider is not None:
            return await self.storage_provider.read_content(path)
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
        if self.storage_provider is not None:
            await self.storage_provider.write_content(path, content)
            return
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
        if self.storage_provider is not None:
            await self.storage_provider.delete_content(path)
            return
        # Content lives in the file record — no-op for DB storage

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
            get_file_record=self._get_file_record,
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
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
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
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
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
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
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
            self._get_file_record,
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
        if isinstance(self.storage_provider, SupportsStorageQueries):
            return await self.storage_provider.storage_list_dir(path)
        sess = self._require_session(session)
        return await list_dir_db(
            path,
            sess,
            get_file_record=self._get_file_record,
            file_model=self._file_model,
        )

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ExistsResult:
        if self.storage_provider is not None:
            result = await self.storage_provider.exists(path)
            return ExistsResult(exists=result, path=normalize_path(path))
        sess = self._require_session(session)
        valid, _error = validate_path(path)
        if not valid:
            return ExistsResult(exists=False, path=path)
        path = normalize_path(path)
        if path == "/":
            return ExistsResult(exists=True, path=path)
        file = await self._get_file_record(sess, path)
        return ExistsResult(exists=file is not None, path=path)

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileInfoResult:
        if self.storage_provider is not None:
            return await self.storage_provider.get_info(path)
        sess = self._require_session(session)
        valid, error = validate_path(path)
        if not valid:
            return FileInfoResult(success=False, message=error, path=path)
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)
        if not file:
            return FileInfoResult(success=False, message=f"File not found: {path}", path=path)
        return file_to_info(file)

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
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
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
            get_file_record=self._get_file_record,
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
        if isinstance(self.storage_provider, SupportsStorageQueries):
            return await self.storage_provider.storage_glob(pattern, path)
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
            dir_file = await self._get_file_record(sess, path)
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
            db_files = list(result.scalars().all())
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
            db_files = list(result.scalars().all())

        # Post-filter with authoritative match (compile once for all candidates)
        glob_regex = compile_glob(pattern, path)
        matched = [
            f for f in db_files if glob_regex is not None and glob_regex.match(f.path) is not None
        ]

        candidates: list[FileSearchCandidate] = []
        for f in matched:
            info = file_to_info(f)
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
        if isinstance(self.storage_provider, SupportsStorageQueries):
            return await self.storage_provider.storage_grep(
                pattern,
                path,
                glob_filter=glob_filter,
                case_sensitive=case_sensitive,
                fixed_string=fixed_string,
                invert=invert,
                word_match=word_match,
                context_lines=context_lines,
                max_results=max_results,
                max_results_per_file=max_results_per_file,
                count_only=count_only,
                files_only=files_only,
            )
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
                file = await self._get_file_record(sess, path)
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
        if isinstance(self.storage_provider, SupportsStorageQueries):
            return await self.storage_provider.storage_tree(path, max_depth=max_depth)
        sess = self._require_session(session)
        path = normalize_path(path)

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self._get_file_record(sess, path)
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
            info = file_to_info(f)
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
            sess, path, self._get_file_record, owner_id=owner_id
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
    ) -> ConnectionListResult:
        sess = self._require_session(session)
        return await self.connections.list_connections(
            sess,
            path,
            direction=direction,
            connection_type=connection_type,
        )

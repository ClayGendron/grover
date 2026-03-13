"""DatabaseFileSystem — pure SQL storage with pluggable providers."""

from __future__ import annotations

import inspect
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlmodel import select

from grover.exceptions import GroverError
from grover.models.database.chunk import FileChunkModel
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.database.version import FileVersionModel
from grover.models.internal.evidence import (
    GlobEvidence,
    GrepEvidence,
    TrashEvidence,
    TreeEvidence,
    VersionEvidence,
)
from grover.models.internal.evidence import (
    LineMatch as SearchLineMatch,
)
from grover.models.internal.ref import File, FileConnection, Ref
from grover.models.internal.results import FileOperationResult, FileSearchResult
from grover.providers.chunks import DefaultChunkProvider
from grover.providers.search.types import SearchResult, VectorEntry
from grover.providers.versioning import DefaultVersionProvider
from grover.ref import Ref as LegacyRef
from grover.util.content import (
    compute_content_hash,
    guess_mime_type,
    has_binary_extension,
    is_text_file,
)
from grover.util.dialect import upsert_file
from grover.util.operations import (
    check_external_edit,
    copy_file,
    delete_file,
    edit_file,
    file_to_info,
    list_dir_db,
    move_file,
    read_file,
    write_file,
)
from grover.util.paths import normalize_path, split_path, validate_path
from grover.util.patterns import compile_glob, glob_to_sql_like

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.connection import FileConnectionModelBase
    from grover.models.database.file import FileModelBase
    from grover.models.database.version import FileVersionModelBase
    from grover.providers.chunks.protocol import ChunkProvider
    from grover.providers.embedding.protocol import EmbeddingProvider
    from grover.providers.graph.protocol import GraphProvider
    from grover.providers.search.extractors import EmbeddableChunk
    from grover.providers.search.local import LocalVectorStore
    from grover.providers.search.protocol import SearchProvider
    from grover.providers.storage.protocol import StorageProvider
    from grover.providers.versioning.protocol import VersionProvider

logger = logging.getLogger(__name__)


class _BatchFileResult(FileOperationResult):
    """Internal batch write result for write_files (includes per-file results list)."""

    results: list[FileOperationResult] = []  # noqa: RUF012
    succeeded: int = 0
    failed: int = 0


class DatabaseFileSystem:
    """Database-backed file system with pluggable providers.

    All content is stored in the database by default — portable and
    consistent across deployments.  When a ``storage_provider`` is set,
    content I/O delegates to it (e.g. ``DiskStorageProvider`` for local disk).

    Providers (keyword-only):

    - ``storage_provider`` — external content I/O + queries (None = DB content)
    - ``graph_provider`` — in-memory graph (None = no graph)
    - ``search_provider`` — search provider (None = no search)
    - ``embedding_provider`` — embedding model (None = no embeddings)
    - ``version_provider`` — version management (default: ``DefaultVersionProvider``)
    - ``chunk_provider`` — chunk management (default: ``DefaultChunkProvider``)
    """

    def __init__(
        self,
        dialect: str = "sqlite",
        file_model: type[FileModelBase] | None = None,
        file_version_model: type[FileVersionModelBase] | None = None,
        file_chunk_model: type[FileChunkModelBase] | None = None,
        file_connection_model: type[FileConnectionModelBase] | None = None,
        schema: str | None = None,
        *,
        storage_provider: StorageProvider | None = None,
        graph_provider: GraphProvider | None = None,
        search_provider: SearchProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        version_provider: VersionProvider | None = None,
        chunk_provider: ChunkProvider | None = None,
    ) -> None:
        self.dialect = dialect
        self.schema = schema
        self.file_model: type[FileModelBase] = file_model or FileModel
        self.file_version_model: type[FileVersionModelBase] = file_version_model or FileVersionModel
        self.file_chunk_model: type[FileChunkModelBase] = file_chunk_model or FileChunkModel
        self.file_connection_model: type[FileConnectionModelBase] = (
            file_connection_model or FileConnectionModel
        )

        # Pluggable providers
        self.storage_provider = storage_provider
        self.graph_provider = graph_provider
        self.search_provider = search_provider
        self.embedding_provider = embedding_provider
        self.version_provider = version_provider or DefaultVersionProvider(
            self.file_model, self.file_version_model
        )
        self.chunk_provider = chunk_provider or DefaultChunkProvider(self.file_chunk_model)

        # Validate search dimensions if both providers set
        self._validate_search_dimensions()

    @staticmethod
    def _require_session(session: AsyncSession | None) -> AsyncSession:
        if session is None:
            raise GroverError("DatabaseFileSystem requires a session")
        return session

    # ------------------------------------------------------------------
    # Conversion helpers: old result types → new internal types
    # ------------------------------------------------------------------

    @staticmethod
    def _op_to_internal(old: object) -> FileOperationResult:
        """Convert an old FileOperationResult subclass to the new internal type."""
        success = getattr(old, "success", True)
        message = getattr(old, "message", "")
        path = getattr(old, "path", "")
        content = getattr(old, "content", None) or None
        version = getattr(old, "version", 0)
        lines_read = getattr(old, "lines_read", 0)
        f = (
            File(path=path, content=content, current_version=version, lines=lines_read)
            if path
            else None
        )
        return FileOperationResult(success=success, message=message, file=f or File(path=""))

    @staticmethod
    def _search_to_internal(old: object) -> FileSearchResult:
        """Convert an old or new FileSearchResult to the internal type."""
        success = getattr(old, "success", True)
        message = getattr(old, "message", "")
        # New internal type already has .files
        if isinstance(old, FileSearchResult):
            return old
        # Old dataclass-based type has .file_candidates
        old_candidates = getattr(old, "file_candidates", [])
        files = [File(path=c.path, evidence=list(c.evidence)) for c in old_candidates]
        return FileSearchResult(success=success, message=message, files=files)

    # ------------------------------------------------------------------
    # File record lookup (absorbed from MetadataService)
    # ------------------------------------------------------------------

    async def _get_file_record(
        self,
        session: AsyncSession,
        path: str,
        include_deleted: bool = False,
    ) -> FileModelBase | None:
        """Look up a file record by path.

        Absorbed from the former ``MetadataService.get_file()``.
        """
        path = normalize_path(path)
        model = self.file_model
        conditions = [model.path == path]
        if not include_deleted:
            conditions.append(
                model.deleted_at.is_(None)  # type: ignore[unresolved-attribute]
            )
        result = await session.execute(select(model).where(*conditions))
        return result.scalar_one_or_none()

    async def _batch_get_file_records(
        self,
        session: AsyncSession,
        paths: list[str],
        include_deleted: bool = False,
    ) -> dict[str, FileModelBase]:
        """Look up multiple file records by path in a single query.

        Returns a dict mapping path -> record for all found files.
        """
        if not paths:
            return {}
        model = self.file_model
        normalized = [normalize_path(p) for p in paths]
        conditions = [model.path.in_(normalized)]  # type: ignore[arg-type]
        if not include_deleted:
            conditions.append(
                model.deleted_at.is_(None)  # type: ignore[unresolved-attribute]
            )
        result = await session.execute(select(model).where(*conditions))
        return {row.path: row for row in result.scalars().all()}

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
        model = self.file_model
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
        model = self.file_model
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
    # Core protocol: GroverFileSystem
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
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
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await write_file(
            path,
            content,
            created_by,
            overwrite,
            sess,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            ensure_parent_dirs=self._ensure_parent_dirs,
            file_model=self.file_model,
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
    ) -> FileOperationResult:
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
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await delete_file(
            path,
            permanent,
            sess,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            file_model=self.file_model,
            delete_content=self._delete_content,
        )

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        created_dirs, error = await self._mkdir_impl(sess, path, parents)
        if error is not None:
            return FileOperationResult(success=False, message=error)
        path = normalize_path(path)
        f = File(path=path, is_directory=True)
        if created_dirs:
            return FileOperationResult(
                success=True,
                message=f"Created directory: {path}",
                file=f,
            )
        return FileOperationResult(
            success=True,
            message=f"Directory already exists: {path}",
            file=f,
        )

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_list_dir(path)
            return self._search_to_internal(old)
        sess = self._require_session(session)
        return await list_dir_db(
            path,
            sess,
            get_file_record=self._get_file_record,
            file_model=self.file_model,
        )

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        if self.storage_provider is not None:
            found = await self.storage_provider.exists(path)
            p = normalize_path(path)
            return FileOperationResult(
                success=True,
                message="exists" if found else "not found",
                file=File(path=p),
            )
        sess = self._require_session(session)
        valid, _error = validate_path(path)
        if not valid:
            return FileOperationResult(success=True, message="not found", file=File(path=path))
        path = normalize_path(path)
        if path == "/":
            return FileOperationResult(success=True, message="exists", file=File(path=path))
        file = await self._get_file_record(sess, path)
        return FileOperationResult(
            success=True,
            message="exists" if file is not None else "not found",
            file=File(path=path),
        )

    async def get_info(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        if self.storage_provider is not None:
            return await self.storage_provider.get_info(path)
        sess = self._require_session(session)
        valid, error = validate_path(path)
        if not valid:
            return FileOperationResult(success=False, message=error, file=File(path=path))
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)
        if not file:
            return FileOperationResult(
                success=False,
                message=f"File not found: {path}",
                file=File(path=path),
            )
        info = file_to_info(file)
        f = File(
            path=info.path,
            is_directory=info.is_directory,
            lines=0,
            current_version=info.version,
        )
        return FileOperationResult(success=True, message="OK", file=f)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        follow: bool = False,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await move_file(
            src,
            dest,
            sess,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            ensure_parent_dirs=self._ensure_parent_dirs,
            file_model=self.file_model,
            read_content=self._read_content,
            write_content=self._write_content,
            delete_content=self._delete_content,
            follow=follow,
        )

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
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
    ) -> FileSearchResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_glob(pattern, path)
            return self._search_to_internal(old)
        sess = self._require_session(session)
        path = normalize_path(path)

        if not pattern:
            return FileSearchResult(
                success=False,
                message="Empty glob pattern",
            )

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self._get_file_record(sess, path)
            if not dir_file:
                return FileSearchResult(
                    success=False,
                    message=f"Directory not found: {path}",
                )
            if not dir_file.is_directory:
                return FileSearchResult(
                    success=False,
                    message=f"Not a directory: {path}",
                )

        model = self.file_model

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

        files: list[File] = []
        for f in matched:
            info = file_to_info(f)
            files.append(
                File(
                    path=info.path,
                    is_directory=info.is_directory,
                    evidence=[
                        GlobEvidence(
                            operation="glob",
                            is_directory=info.is_directory,
                            size_bytes=info.size_bytes,
                            mime_type=info.mime_type,
                        )
                    ],
                )
            )

        return FileSearchResult(
            success=True,
            message=f"Found {len(files)} match(es)",
            files=files,
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
    ) -> FileSearchResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_grep(
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
            return self._search_to_internal(old)
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
            return FileSearchResult(
                success=False,
                message=f"Invalid regex: {e}",
            )

        # Get candidate files
        if glob_filter:
            glob_result = await self.glob(glob_filter, path, session=sess)
            if not glob_result.success:
                return FileSearchResult(
                    success=False,
                    message=glob_result.message,
                )
            # Extract non-directory paths from glob result
            candidate_paths = [f.path for f in glob_result.files if not f.is_directory]
        else:
            model = self.file_model
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
                    return FileSearchResult(
                        success=False,
                        message=f"Path not found: {path}",
                    )
            else:
                result = await sess.execute(
                    select(model.path).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                        model.is_directory.is_(False),  # type: ignore[union-attr]
                    )
                )
                candidate_paths = [row[0] for row in result.all()]

        result_files: list[File] = []
        files_searched = 0
        files_matched = 0
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

                result_files.append(
                    File(
                        path=file_path,
                        evidence=[
                            GrepEvidence(
                                operation="grep",
                                line_matches=tuple(file_line_matches),
                            )
                        ],
                    )
                )
                total_matches += len(file_line_matches)

                if max_results > 0 and total_matches >= max_results:
                    break

        if count_only:
            total = files_matched if files_only else total_matches
            return FileSearchResult(
                success=True,
                message=f"Count: {total}",
            )

        return FileSearchResult(
            success=True,
            message=f"Found {total_matches} match(es) in {files_matched} file(s)",
            files=result_files,
        )

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_tree(path, max_depth=max_depth)
            return self._search_to_internal(old)
        sess = self._require_session(session)
        path = normalize_path(path)

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self._get_file_record(sess, path)
            if not dir_file:
                return FileSearchResult(
                    success=False,
                    message=f"Directory not found: {path}",
                )
            if not dir_file.is_directory:
                return FileSearchResult(
                    success=False,
                    message=f"Not a directory: {path}",
                )

        model = self.file_model
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

        files: list[File] = []
        total_files = 0
        total_dirs = 0

        for f in all_files:
            info = file_to_info(f)
            depth = info.path.count("/") - base_depth
            files.append(
                File(
                    path=info.path,
                    is_directory=info.is_directory,
                    evidence=[
                        TreeEvidence(
                            operation="tree",
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

        files.sort(key=lambda f: f.path)
        return FileSearchResult(
            success=True,
            message=f"{total_dirs} directories, {total_files} files",
            files=files,
        )

    # ------------------------------------------------------------------
    # Trash operations
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        sess = self._require_session(session)
        model = self.file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await sess.execute(select(model).where(*conditions))
        db_files = result.scalars().all()

        files = [
            File(
                path=f.original_path or f.path,
                evidence=[
                    TrashEvidence(
                        operation="trash",
                        deleted_at=f.deleted_at,
                        original_path=f.original_path or f.path,
                    )
                ],
            )
            for f in db_files
        ]

        return FileSearchResult(
            success=True,
            message=f"Found {len(files)} items in trash",
            files=files,
        )

    async def restore_from_trash(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        path = normalize_path(path)

        model = self.file_model
        conditions = [
            model.original_path == path,
            model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
        ]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await sess.execute(select(model).where(*conditions))
        file = result.scalar_one_or_none()

        if not file:
            return FileOperationResult(success=False, message=f"File not in trash: {path}")

        original = file.original_path or path

        # If path is occupied, overwrite the occupant (git restore semantics).
        existing = await self._get_file_record(sess, original, False)
        if existing and existing.id != file.id:
            await self.version_provider.delete_versions(sess, existing.id)
            await sess.delete(existing)
            await sess.flush()

        file.path = original
        file.original_path = None
        file.deleted_at = None
        file.updated_at = datetime.now(UTC)

        if file.is_directory:
            children_result = await sess.execute(
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
                child_existing = await self._get_file_record(sess, child_original, False)
                if child_existing and child_existing.id != child.id:
                    await self.version_provider.delete_versions(sess, child_existing.id)
                    await sess.delete(child_existing)
                    had_occupants = True
            if had_occupants:
                await sess.flush()

            for child in children:
                child.path = child.original_path or child.path
                child.original_path = None
                child.deleted_at = None
                child.updated_at = datetime.now(UTC)

        await sess.flush()

        return FileOperationResult(
            success=True,
            message=f"Restored from trash: {path}",
            file=File(path=path),
        )

    async def empty_trash(
        self,
        *,
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        model = self.file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await sess.execute(select(model).where(*conditions))
        db_files = result.scalars().all()

        count = len(db_files)
        for file in db_files:
            await self.version_provider.delete_versions(sess, file.id)
            await self._delete_content(file.original_path or file.path, sess)
            await sess.delete(file)

        await sess.flush()

        return FileOperationResult(
            success=True,
            message=f"Permanently deleted {count} items from trash",
        )

    # ------------------------------------------------------------------
    # Connection operations
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        source_path: str,
        target_path: str,
        connection_type: str,
        *,
        weight: float = 1.0,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        model = self.file_connection_model
        path = f"{source_path}[{connection_type}]{target_path}"

        result = await sess.execute(select(model).where(model.path == path))
        existing = result.scalar_one_or_none()

        if existing is not None:
            existing.weight = weight
            await sess.flush()
            return FileOperationResult(
                success=True,
                message="Connection updated",
                file=File(path=path),
            )

        record = model(
            path=path,
            source_path=source_path,
            target_path=target_path,
            type=connection_type,
            weight=weight,
        )
        sess.add(record)
        await sess.flush()
        return FileOperationResult(
            success=True,
            message="Connection created",
            file=File(path=path),
        )

    async def delete_connection(
        self,
        source_path: str,
        target_path: str,
        *,
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        model = self.file_connection_model

        if connection_type is not None:
            path = f"{source_path}[{connection_type}]{target_path}"
            result = await sess.execute(select(model).where(model.path == path))
            row = result.scalar_one_or_none()
            if row is None:
                return FileOperationResult(
                    success=False,
                    message=f"Connection not found: {path}",
                    file=File(path=path),
                )
            await sess.delete(row)
            await sess.flush()
            return FileOperationResult(
                success=True,
                message="Connection deleted",
                file=File(path=path),
            )

        # Delete all connections between source and target
        result = await sess.execute(
            select(model).where(
                model.source_path == source_path,
                model.target_path == target_path,
            )
        )
        rows = list(result.scalars().all())
        if not rows:
            return FileOperationResult(
                success=False,
                message=f"No connections found from {source_path} to {target_path}",
            )
        for row in rows:
            await sess.delete(row)
        await sess.flush()
        return FileOperationResult(
            success=True,
            message=f"Deleted {len(rows)} connection(s)",
        )

    async def delete_connections_for_path(
        self,
        session: AsyncSession,
        path: str,
    ) -> int:
        """Delete all connections where path is source or target. Returns count deleted."""
        model = self.file_connection_model
        result = await session.execute(
            select(model).where((model.source_path == path) | (model.target_path == path))
        )
        rows = list(result.scalars().all())
        for row in rows:
            await session.delete(row)
        if rows:
            await session.flush()
        return len(rows)

    async def delete_outgoing_connections(
        self,
        session: AsyncSession,
        path: str,
    ) -> int:
        """Delete connections where path is the source only. Returns count deleted."""
        model = self.file_connection_model
        result = await session.execute(select(model).where(model.source_path == path))
        rows = list(result.scalars().all())
        for row in rows:
            await session.delete(row)
        if rows:
            await session.flush()
        return len(rows)

    async def list_connections(
        self,
        path: str,
        *,
        direction: str = "both",
        connection_type: str | None = None,
        session: AsyncSession | None = None,
    ) -> FileSearchResult:
        sess = self._require_session(session)
        model = self.file_connection_model
        conditions = []

        if direction == "out":
            conditions.append(model.source_path == path)
        elif direction == "in":
            conditions.append(model.target_path == path)
        else:
            conditions.append((model.source_path == path) | (model.target_path == path))

        if connection_type is not None:
            conditions.append(model.type == connection_type)

        stmt = select(model).where(*conditions)
        result = await sess.execute(stmt)
        db_connections = list(result.scalars().all())
        connections = [
            FileConnection(
                source=Ref(path=c.source_path),
                target=Ref(path=c.target_path),
                type=c.type,
                weight=c.weight,
            )
            for c in db_connections
        ]
        return FileSearchResult(
            success=True,
            message=f"Found {len(connections)} connection(s)",
            connections=connections,
        )

    # ------------------------------------------------------------------
    # Directory helpers (inlined from DirectoryService)
    # ------------------------------------------------------------------

    async def _ensure_parent_dirs(
        self,
        session: AsyncSession,
        path: str,
        owner_id: str | None = None,
    ) -> None:
        """Ensure all parent directories exist in the database."""
        parts = path.split("/")
        for i in range(2, len(parts)):
            dir_path = "/".join(parts[:i])
            if not dir_path:
                continue

            parent = "/".join(parts[: i - 1]) or "/"
            values: dict[str, object] = {
                "id": str(uuid.uuid4()),
                "path": dir_path,
                "parent_path": parent,
                "is_directory": True,
                "current_version": 1,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            if owner_id is not None:
                values["owner_id"] = owner_id
            await upsert_file(
                session,
                self.dialect,
                values=values,
                conflict_keys=["path"],
                model=self.file_model,
                schema=self.schema,
                update_keys=["updated_at"],
            )

    async def _mkdir_impl(
        self,
        session: AsyncSession,
        path: str,
        parents: bool,
        owner_id: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Create a directory using dialect-aware upsert.

        Returns ``(created_dirs, error_message)``.  On success,
        ``error_message`` is ``None``.
        """
        valid, error = validate_path(path)
        if not valid:
            return [], error

        path = normalize_path(path)

        existing = await self._get_file_record(session, path, False)
        if existing:
            if existing.is_directory:
                return [], None  # already exists, no error
            return [], f"Path exists as file: {path}"

        dirs_to_create: list[str] = []
        current = path

        while current != "/":
            existing = await self._get_file_record(session, current, False)
            if existing:
                if not existing.is_directory:
                    return [], f"Path exists as file: {current}"
                break
            dirs_to_create.insert(0, current)
            if not parents and len(dirs_to_create) > 1:
                return [], f"Parent directory does not exist: {split_path(current)[0]}"
            current = split_path(current)[0]

        created_dirs: list[str] = []
        for dir_path in dirs_to_create:
            parent, _name = split_path(dir_path)
            values: dict[str, object] = {
                "id": str(uuid.uuid4()),
                "path": dir_path,
                "parent_path": parent,
                "is_directory": True,
                "current_version": 1,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            if owner_id is not None:
                values["owner_id"] = owner_id
            rowcount = await upsert_file(
                session,
                self.dialect,
                values=values,
                conflict_keys=["path"],
                model=self.file_model,
                schema=self.schema,
            )
            if rowcount > 0:
                created_dirs.append(dir_path)

        await session.flush()
        return created_dirs, None

    # ------------------------------------------------------------------
    # Graph methods (inlined from GraphMethodsMixin)
    # ------------------------------------------------------------------

    def graph_add_node(self, path: str, **attrs: object) -> None:
        if self.graph_provider is None:
            return
        self.graph_provider.add_node(path, **attrs)

    def graph_remove_node(self, path: str) -> None:
        if self.graph_provider is None:
            return
        self.graph_provider.remove_node(path)

    def graph_has_node(self, path: str) -> bool:
        if self.graph_provider is None:
            return False
        return self.graph_provider.has_node(path)

    def graph_get_node(self, path: str) -> dict:
        if self.graph_provider is None:
            return {}
        return self.graph_provider.get_node(path)

    def graph_nodes(self) -> list[str]:
        if self.graph_provider is None:
            return []
        return self.graph_provider.nodes()

    def graph_add_edge(self, source: str, target: str, edge_type: str, **attrs: object) -> None:
        if self.graph_provider is None:
            return
        self.graph_provider.add_edge(source, target, edge_type, **attrs)

    def graph_remove_edge(self, source: str, target: str) -> None:
        if self.graph_provider is None:
            return
        self.graph_provider.remove_edge(source, target)

    def graph_has_edge(self, source: str, target: str) -> bool:
        if self.graph_provider is None:
            return False
        return self.graph_provider.has_edge(source, target)

    async def graph_predecessors(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return FileSearchResult(success=True, message="No graph provider")
        old = await self.graph_provider.predecessors(path, session=session)  # type: ignore[arg-type]
        return self._search_to_internal(old)

    async def graph_successors(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return FileSearchResult(success=True, message="No graph provider")
        old = await self.graph_provider.successors(path, session=session)  # type: ignore[arg-type]
        return self._search_to_internal(old)

    async def graph_path_between(
        self,
        source: str,
        target: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return FileSearchResult(success=True, message="No path found")
        old = await self.graph_provider.path_between(source, target, session=session)  # type: ignore[arg-type]
        return self._search_to_internal(old)

    async def graph_contains(self, path: str, *, session: AsyncSession | None = None) -> list[Ref]:
        if self.graph_provider is None:
            return []
        return await self.graph_provider.contains(path, session=session)  # type: ignore[arg-type]

    def graph_remove_file_subgraph(self, path: str) -> list[str]:
        if self.graph_provider is None:
            return []
        return self.graph_provider.remove_file_subgraph(path)

    @property
    def graph_node_count(self) -> int:
        if self.graph_provider is None:
            return 0
        return self.graph_provider.node_count

    @property
    def graph_edge_count(self) -> int:
        if self.graph_provider is None:
            return 0
        return self.graph_provider.edge_count

    # ------------------------------------------------------------------
    # Chunk methods (inlined from ChunkMethodsMixin)
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await self.chunk_provider.replace_file_chunks(sess, file_path, chunks)

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await self.chunk_provider.delete_file_chunks(sess, file_path)

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await self.chunk_provider.list_file_chunks(sess, file_path)

    async def write_chunk(
        self,
        chunk: FileChunkModelBase,
        *,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await self.chunk_provider.write_chunk(sess, chunk)

    async def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        session: AsyncSession | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        return await self.chunk_provider.write_chunks(sess, chunks)

    # ------------------------------------------------------------------
    # Batch file write (write_file / write_files)
    # ------------------------------------------------------------------

    async def write_file(
        self,
        file: FileModelBase,
        *,
        overwrite: bool = True,
        created_by: str = "agent",
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        """Write a single file from a model instance. Delegates to write_files."""
        sess = self._require_session(session)
        batch = await self.write_files(
            [file],
            overwrite=overwrite,
            created_by=created_by,
            session=sess,
            owner_id=owner_id,
        )
        return batch

    async def write_files(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool = True,
        created_by: str = "agent",
        session: AsyncSession | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        """Batch write files from model instances with minimized DB queries.

        One SELECT IN for existing records, per-file versioning, one flush.
        System manages hashes, sizes, versions, and timestamps.
        """
        sess = self._require_session(session)
        now = datetime.now(UTC)
        results_by_idx: dict[int, FileOperationResult] = {}

        if not files:
            return FileOperationResult(success=True, message="No files to write")

        # Extract and validate paths, tracking original index
        valid_items: list[tuple[int, str, str]] = []  # (idx, path, content)
        for i, f in enumerate(files):
            valid, error = validate_path(f.path)
            if not valid:
                results_by_idx[i] = FileOperationResult(
                    success=False, message=error, file=File(path=f.path)
                )
                continue

            path = normalize_path(f.path)
            _, name = split_path(path)
            if not is_text_file(name):
                results_by_idx[i] = FileOperationResult(
                    success=False,
                    message=f"Cannot write non-text file: {name}",
                    file=File(path=path),
                )
                continue

            valid_items.append((i, path, f.content if f.content else ""))

        # Deduplicate paths: last-write-wins for same path in batch
        seen_paths: dict[str, int] = {}  # path -> index in valid_items
        deduped_items: list[tuple[int, str, str]] = []
        for item in valid_items:
            idx, path, content = item
            if path in seen_paths:
                # Supersede earlier entry
                prev_idx = valid_items[seen_paths[path]][0]
                results_by_idx[prev_idx] = FileOperationResult(
                    success=False,
                    message=f"Superseded by later entry in batch: {path}",
                    file=File(path=path),
                )
            seen_paths[path] = len(deduped_items)
            deduped_items.append(item)
        valid_items = deduped_items

        if not valid_items:
            # All failed validation
            results = [results_by_idx[i] for i in range(len(files))]
            succeeded = sum(1 for r in results if r.success)
            failed = len(results) - succeeded
            return FileOperationResult(
                success=failed == 0,
                message=f"Wrote {succeeded} file(s), {failed} failed",
            )

        # ONE query: batch lookup existing records
        paths = [path for _, path, _ in valid_items]
        existing_map = await self._batch_get_file_records(sess, paths, include_deleted=True)

        # Per-file processing
        for idx, path, content in valid_items:
            try:
                content_hash, size_bytes = compute_content_hash(content)
                existing = existing_map.get(path)

                if existing:
                    if existing.is_directory:
                        results_by_idx[idx] = FileOperationResult(
                            success=False,
                            message=f"Path is a directory: {path}",
                            file=File(path=path),
                        )
                        continue

                    if not overwrite and not existing.deleted_at:
                        results_by_idx[idx] = FileOperationResult(
                            success=False,
                            message=f"File already exists: {path}",
                            file=File(path=path),
                        )
                        continue

                    # Handle soft-deleted files
                    if existing.deleted_at:
                        existing.deleted_at = None
                        existing.path = existing.original_path or path
                        existing.original_path = None

                    # Read current content for external edit check + versioning
                    old_content = await self._read_content(path, sess)

                    if old_content is not None:
                        await check_external_edit(
                            existing,
                            old_content,
                            sess,
                            versioning=self.version_provider,
                        )

                    existing.current_version += 1
                    existing.content_hash = content_hash
                    existing.size_bytes = size_bytes
                    existing.updated_at = now

                    if old_content is not None:
                        await self.version_provider.save_version(
                            sess, existing, old_content, content, created_by
                        )

                    await self._write_content(path, content, sess)
                    results_by_idx[idx] = FileOperationResult(
                        success=True,
                        message=f"Updated: {path} (v{existing.current_version})",
                        file=File(path=path, current_version=existing.current_version),
                    )
                else:
                    # New file
                    await self._ensure_parent_dirs(sess, path, owner_id)

                    _, name = split_path(path)
                    new_file = self.file_model(
                        path=path,
                        parent_path=split_path(path)[0],
                        owner_id=owner_id,
                        content_hash=content_hash,
                        size_bytes=size_bytes,
                        mime_type=guess_mime_type(name),
                    )
                    sess.add(new_file)

                    await self.version_provider.save_version(
                        sess, new_file, "", content, created_by
                    )
                    await self._write_content(path, content, sess)
                    results_by_idx[idx] = FileOperationResult(
                        success=True,
                        message=f"Created: {path} (v1)",
                        file=File(path=path, current_version=1),
                    )
            except Exception as e:
                results_by_idx[idx] = FileOperationResult(
                    success=False,
                    message=f"Write failed: {e}",
                    file=File(path=path),
                )

        await sess.flush()

        results = [results_by_idx[i] for i in range(len(files))]
        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        return _BatchFileResult(
            success=failed == 0,
            message=f"Wrote {succeeded} file(s)" + (f", {failed} failed" if failed else ""),
            results=results,
            succeeded=succeeded,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Version methods (inlined from VersionMethodsMixin)
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)
        if not file:
            return FileSearchResult(success=False, message=f"File not found: {path}")
        versions = await self.version_provider.list_versions(sess, file)
        files = [
            File(
                path=f"{path}@{v.version}",
                current_version=v.version,
                evidence=[
                    VersionEvidence(
                        operation="version",
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
        return FileSearchResult(
            success=True,
            message=f"Found {len(versions)} version(s)",
            files=files,
        )

    async def get_version_content(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)
        if not file:
            return FileOperationResult(
                success=False,
                message=f"File not found: {path}",
            )
        content = await self.version_provider.get_version_content(sess, file, version)
        if content is None:
            return FileOperationResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )
        return FileOperationResult(
            success=True,
            message="OK",
            file=File(path=path, content=content, current_version=version),
        )

    async def restore_version(
        self,
        path: str,
        version: int,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        vc_result = await self.get_version_content(path, version, session=sess)
        if not vc_result.success or not vc_result.file or vc_result.file.content is None:
            return FileOperationResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(
            path,
            vc_result.file.content,
            created_by="restore",
            session=sess,
        )

        new_version = write_result.file.current_version if write_result.file else 0
        return FileOperationResult(
            success=True,
            message=f"Restored {path} to version {version}",
            file=File(path=path, current_version=new_version),
        )

    async def verify_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self._get_file_record(sess, path)
        if not file:
            return FileOperationResult(
                success=False,
                message=f"File not found: {path}",
                file=File(path=path),
            )
        return await self.version_provider.verify_chain(sess, file)

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> list[FileOperationResult]:
        sess = self._require_session(session)
        model = self.file_model
        result = await sess.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[union-attr]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        results: list[FileOperationResult] = []
        for file in result.scalars().all():
            results.append(await self.version_provider.verify_chain(sess, file))  # noqa: PERF401
        return results

    # ------------------------------------------------------------------
    # Search methods (inlined from SearchMethodsMixin)
    # ------------------------------------------------------------------

    def _validate_search_dimensions(self) -> None:
        """Check that embedding dimensions match search store dimensions."""
        if self.embedding_provider is not None and self.search_provider is not None:
            store_dim = getattr(self.search_provider, "dimension", None)
            if store_dim is not None and self.embedding_provider.dimensions != store_dim:
                msg = (
                    f"Dimension mismatch: embedding provider "
                    f"'{self.embedding_provider.model_name}' produces "
                    f"{self.embedding_provider.dimensions}-dim vectors, but vector "
                    f"store expects {store_dim}-dim"
                )
                raise ValueError(msg)

    async def search_add(
        self,
        path: str,
        content: str,
        *,
        parent_path: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed *content* and upsert to the vector store."""
        if self.search_provider is not None and self.embedding_provider is not None:
            vector = await self._search_embed(content)
            entry = VectorEntry(
                id=path,
                vector=vector,
                metadata={
                    "content": content,
                    "parent_path": parent_path,
                    "content_hash": compute_content_hash(content)[0],
                },
            )
            await self.search_provider.upsert([entry])

    async def search_add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Embed a batch of entries and upsert to the vector store."""
        if not entries:
            return

        if self.search_provider is not None and self.embedding_provider is not None:
            texts = [e.content for e in entries]
            vectors = await self._search_embed_batch(texts)

            vector_entries = [
                VectorEntry(
                    id=entry.path,
                    vector=vectors[i],
                    metadata={
                        "content": entry.content,
                        "parent_path": entry.parent_path,
                        "content_hash": compute_content_hash(entry.content)[0],
                        "chunk_name": entry.chunk_name,
                        "line_start": entry.line_start,
                        "line_end": entry.line_end,
                    },
                )
                for i, entry in enumerate(entries)
            ]
            await self.search_provider.upsert(vector_entries)

    async def search_remove(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove a single entry by path from the vector store."""
        if self.search_provider is not None:
            await self.search_provider.delete([path])

    async def search_remove_file(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches."""
        if self.search_provider is not None:
            local_store = self._search_get_local_store()
            if local_store is not None:
                local_store.remove_file(path)
            else:
                await self.search_provider.delete([path])

    async def vector_search(self, query: str, k: int = 10) -> FileSearchResult:
        """Embed *query*, call ``search_provider.vector_search()``, return result."""
        if self.embedding_provider is None:
            return FileSearchResult(
                success=False,
                message="Cannot search: no embedding provider configured",
            )
        if self.search_provider is None:
            return FileSearchResult(
                success=False,
                message="Cannot search: no search provider configured",
            )

        vector = await self._search_embed(query)
        old = await self.search_provider.vector_search(vector, k=k)
        return self._search_to_internal(old)

    async def lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SearchResult]:
        """Lexical search: tries search_provider first, falls back to DB FTS."""
        if self.search_provider is not None:
            provider_result = await self.search_provider.lexical_search(query, k=k)
            if provider_result.success and len(provider_result) > 0:
                results: list[SearchResult] = []
                for c in provider_result.files:
                    for ev in c.evidence:
                        snippet = getattr(ev, "snippet", "")
                        results.append(
                            SearchResult(
                                ref=LegacyRef(path=c.path),
                                score=1.0,
                                content=snippet,
                            )
                        )
                return results

        return await self._db_lexical_search(query, k=k, session=session)

    async def _db_lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SearchResult]:
        """Dialect-aware full-text search against DB content."""
        from sqlalchemy import text

        sess = self._require_session(session)
        model = self.file_model

        results: list[SearchResult] = []

        if self.dialect == "sqlite":
            try:
                table_name = getattr(model, "__tablename__", "grover_files")
                fts_table = f"{table_name}_fts"
                stmt = text(
                    f"SELECT path, content FROM {fts_table} WHERE {fts_table} MATCH :query LIMIT :k"
                )
                rows = await sess.execute(stmt, {"query": query, "k": k})
                results.extend(
                    SearchResult(
                        ref=LegacyRef(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                logger.debug("FTS5 not available, falling back to LIKE")

        elif self.dialect == "postgresql":
            try:
                stmt = (
                    select(model.path, model.content)
                    .where(
                        text("content_tsv @@ plainto_tsquery('english', :query)"),
                        model.deleted_at.is_(None),  # type: ignore[union-attr]
                    )
                    .limit(k)
                )
                rows = await sess.execute(stmt, {"query": query})
                results.extend(
                    SearchResult(
                        ref=LegacyRef(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                logger.debug("PostgreSQL FTS not available, falling back to LIKE")

        elif self.dialect == "mssql":
            try:
                stmt = (
                    select(model.path, model.content)
                    .where(
                        text("FREETEXT(content, :query)"),
                        model.deleted_at.is_(None),  # type: ignore[union-attr]
                    )
                    .limit(k)
                )
                rows = await sess.execute(stmt, {"query": query})
                results.extend(
                    SearchResult(
                        ref=LegacyRef(path=row[0]),
                        score=1.0,
                        content=row[1] or "",
                    )
                    for row in rows
                )
                return results
            except Exception:
                logger.debug("MSSQL FTS not available, falling back to LIKE")

        # Fallback: LIKE search
        like_pattern = f"%{query}%"
        stmt = (
            select(model.path, model.content)
            .where(
                model.content.like(like_pattern),  # type: ignore[union-attr]
                model.deleted_at.is_(None),  # type: ignore[union-attr]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
            .limit(k)
        )
        rows = await sess.execute(stmt)
        results.extend(
            SearchResult(
                ref=LegacyRef(path=row[0]),
                score=0.5,
                content=row[1] or "",
            )
            for row in rows
        )
        return results

    def search_has(self, path: str) -> bool:
        """Return whether *path* is present in the local vector store."""
        local = self._search_get_local_store()
        if local is not None:
            return local.has(path)
        return False

    def search_content_hash(self, path: str) -> str | None:
        """Return the content hash for *path*, or None."""
        local = self._search_get_local_store()
        if local is not None:
            return local.content_hash(path)
        return None

    def search_save(self, directory: str) -> None:
        """Persist the vector store to *directory* (if supported)."""
        save_fn = getattr(self.search_provider, "save", None)
        if save_fn is not None:
            save_fn(directory)

    def search_load(self, directory: str) -> None:
        """Load the vector store from *directory* (if supported)."""
        load_fn = getattr(self.search_provider, "load", None)
        if load_fn is not None:
            load_fn(directory)

    async def search_connect(self) -> None:
        """Connect the underlying vector store."""
        if self.search_provider is not None:
            await self.search_provider.connect()

    async def search_close(self) -> None:
        """Close the underlying vector store."""
        if self.search_provider is not None:
            await self.search_provider.close()

    async def _search_embed(self, text: str) -> list[float]:
        """Embed a single text, handling both sync and async providers."""
        assert self.embedding_provider is not None
        result = self.embedding_provider.embed(text)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _search_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling both sync and async providers."""
        assert self.embedding_provider is not None
        result = self.embedding_provider.embed_batch(texts)
        if inspect.isawaitable(result):
            return await result
        return result

    def _search_get_local_store(self) -> LocalVectorStore | None:
        """Return the store as a LocalVectorStore if it is one."""
        from grover.providers.search.local import LocalVectorStore

        if isinstance(self.search_provider, LocalVectorStore):
            return self.search_provider
        return None

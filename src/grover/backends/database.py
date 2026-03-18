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

from grover.models.database.chunk import FileChunkModel
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel
from grover.models.database.version import FileVersionModel
from grover.models.internal.detail import CopyDetail, DeleteDetail, MoveDetail, ReadDetail, WriteDetail
from grover.models.internal.evidence import (
    GlobEvidence,
    GrepEvidence,
    TrashEvidence,
    VersionEvidence,
)
from grover.models.internal.evidence import (
    LineMatch as SearchLineMatch,
)
from grover.models.internal.ref import Directory, File, FileChunk, FileConnection, FileVersion, Ref
from grover.models.internal.results import (
    BatchResult,
    FileOperationResult,
    FileSearchResult,
    FileSearchSet,
    GroverResult,
)
from grover.providers.chunks import DefaultChunkProvider
from grover.providers.versioning import DefaultVersionProvider
from grover.providers.versioning.diff import SNAPSHOT_INTERVAL, compute_diff
from grover.util.content import (
    compute_content_hash,
    guess_mime_type,
    has_binary_extension,
)
from grover.util.dialect import upsert_file
from grover.util.operations import (
    copy_file,
    delete_file,
    edit_file,
    file_to_info,
    list_dir_db,
    move_file,
    read_file,
    write_file,
)
from grover.util.paths import normalize_path, split_path, to_trash_path, validate_path
from grover.util.patterns import compile_glob, glob_to_sql_like

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.config import EngineConfig, SessionConfig
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
        *,
        storage_provider: StorageProvider | None = None,
        graph_provider: GraphProvider | None = None,
        search_provider: SearchProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        version_provider: VersionProvider | None = None,
        chunk_provider: ChunkProvider | None = None,
    ) -> None:
        # Config defaults — overwritten by _configure() when mounting via
        # EngineConfig / SessionConfig.
        self.dialect: str = "sqlite"
        self.file_model: type[FileModelBase] = FileModel
        self.file_version_model: type[FileVersionModelBase] = FileVersionModel
        self.file_chunk_model: type[FileChunkModelBase] = FileChunkModel
        self.file_connection_model: type[FileConnectionModelBase] = FileConnectionModel

        # Pluggable providers
        self.storage_provider = storage_provider
        self.graph_provider = graph_provider
        self.search_provider = search_provider
        self.embedding_provider = embedding_provider
        self.version_provider = version_provider or DefaultVersionProvider(self.file_model, self.file_version_model)
        self.chunk_provider = chunk_provider or DefaultChunkProvider(self.file_chunk_model)

        # Validate search dimensions if both providers set
        self._validate_search_dimensions()

    def _init_default_providers(self) -> None:
        """Re-create default version/chunk providers from current models.

        Called by ``_configure()`` after model attributes change.
        """
        self.version_provider = DefaultVersionProvider(self.file_model, self.file_version_model)
        self.chunk_provider = DefaultChunkProvider(self.file_chunk_model)

    def _configure(self, config: EngineConfig | SessionConfig, dialect: str) -> None:
        """Apply config from EngineConfig or SessionConfig. Called by add_mount."""

        self.dialect = dialect
        self.file_model = config.file_model
        self.file_version_model = config.file_version_model
        self.file_chunk_model = config.file_chunk_model
        self.file_connection_model = config.file_connection_model
        # Re-initialize default providers with potentially new models
        self._init_default_providers()

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
        f = File(path=path, content=content, current_version=version, lines=lines_read) if path else None
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        op = await read_file(
            path,
            offset,
            limit,
            session,
            get_file_record=self._get_file_record,
            read_content=self._read_content,
        )
        if not op.success:
            return GroverResult(success=False, message=op.message)
        op.file.evidence = [
            ReadDetail(
                operation="read",
                success=True,
                message=op.message,
                offset=offset,
            )
        ]
        return GroverResult(success=True, message=op.message, files=[op.file])

    async def read_files(
        self,
        paths: list[str],
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Batch read — single query lookup, results in input order."""
        if not paths:
            return GroverResult(success=True, message="No files to read")

        normalized = [normalize_path(p) for p in paths]
        records = await self._batch_get_file_records(session, normalized)

        result_files: list[File] = []
        for path in normalized:
            rec = records.get(path)
            if rec is None or rec.is_directory:
                msg = f"File not found: {path}" if rec is None else f"Path is a directory: {path}"
                result_files.append(
                    File(
                        path=path,
                        evidence=[
                            ReadDetail(operation="read", success=False, message=msg),
                        ],
                    )
                )
                continue

            content = rec.content if self.storage_provider is None else await self._read_content(path, session)
            if content is None:
                result_files.append(
                    File(
                        path=path,
                        evidence=[
                            ReadDetail(operation="read", success=False, message=f"Content not found: {path}"),
                        ],
                    )
                )
                continue

            result_files.append(
                File(
                    path=path,
                    content=content,
                    evidence=[
                        ReadDetail(operation="read", success=True, message=f"Read {path}"),
                    ],
                )
            )

        succeeded = sum(1 for f in result_files if all(d.success for d in f.details))
        failed = len(result_files) - succeeded
        return GroverResult(
            success=failed == 0,
            message=f"Read {succeeded} file(s)" + (f", {failed} failed" if failed else ""),
            files=result_files,
        )

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        return await write_file(
            path,
            content,
            created_by,
            overwrite,
            session,
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        op = await edit_file(
            path,
            old_string,
            new_string,
            replace_all,
            created_by,
            session,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            read_content=self._read_content,
            write_content=self._write_content,
        )
        if not op.success:
            return GroverResult(success=False, message=op.message)
        op.file.evidence = [
            WriteDetail(
                operation="edit",
                success=True,
                message=op.message,
                version=op.file.current_version,
            )
        ]
        return GroverResult(success=True, message=op.message, files=[op.file])

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        op = await delete_file(
            path,
            permanent,
            session,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            file_model=self.file_model,
            delete_content=self._delete_content,
        )
        if not op.success:
            return GroverResult(success=False, message=op.message)
        op.file.evidence = [
            DeleteDetail(
                operation="delete",
                success=True,
                message=op.message,
                permanent=permanent,
            )
        ]
        return GroverResult(success=True, message=op.message, files=[op.file])

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        created_dirs, error = await self._mkdir_impl(session, path, parents)
        if error is not None:
            return GroverResult(success=False, message=error)
        path = normalize_path(path)
        d = Directory(path=path)
        if created_dirs:
            return GroverResult(
                success=True,
                message=f"Created directory: {path}",
                directories=[d],
            )
        return GroverResult(
            success=True,
            message=f"Directory already exists: {path}",
            directories=[d],
        )

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_list_dir(path)
            result = self._search_to_internal(old)
            # TODO: migrate storage_provider path to GroverResult
            return GroverResult(success=result.success, message=result.message, files=result.files)
        return await list_dir_db(
            path,
            session,
            get_file_record=self._get_file_record,
            file_model=self.file_model,
        )

    async def exists(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        path = normalize_path(path)
        if path == "/":
            return GroverResult(success=True, message="exists", directories=[Directory(path=path)])

        ref = Ref(path)

        if ref.is_connection:
            model = self.file_connection_model
            record = (await session.execute(select(model).where(model.path == path))).scalar_one_or_none()
            if record is None:
                return GroverResult(success=False, message="not found")
            return GroverResult(
                success=True,
                message="exists",
                connections=[
                    FileConnection(
                        path=record.path,
                        source_path=record.source_path,
                        target_path=record.target_path,
                        type=record.type,
                        weight=record.weight,
                    )
                ],
            )

        if ref.is_chunk:
            model = self.file_chunk_model
            record = (await session.execute(select(model).where(model.path == path))).scalar_one_or_none()
            if record is None:
                return GroverResult(success=False, message="not found")
            return GroverResult(
                success=True,
                message="exists",
                files=[
                    File(
                        path=ref.base_path,
                        chunks=[
                            FileChunk(
                                path=record.path,
                                name=ref.chunk or "",
                                content=record.content,
                                line_start=record.line_start,
                                line_end=record.line_end,
                            )
                        ],
                    )
                ],
            )

        if ref.is_version:
            model = self.file_version_model
            record = (await session.execute(select(model).where(model.path == path))).scalar_one_or_none()
            if record is None:
                return GroverResult(success=False, message="not found")
            return GroverResult(
                success=True,
                message="exists",
                files=[
                    File(
                        path=ref.base_path,
                        versions=[
                            FileVersion(
                                path=record.path,
                                number=record.version,
                            )
                        ],
                    )
                ],
            )

        # File or directory
        file = await self._get_file_record(session, path)
        if file is None:
            return GroverResult(success=False, message="not found")
        if file.is_directory:
            return GroverResult(success=True, message="exists", directories=[Directory(path=path)])
        return GroverResult(
            success=True,
            message="exists",
            files=[
                File(
                    path=file.path,
                    size_bytes=file.size_bytes,
                    mime_type=file.mime_type,
                    current_version=file.current_version,
                    created_at=file.created_at,
                    updated_at=file.updated_at,
                )
            ],
        )

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession,
        follow: bool = False,
        user_id: str | None = None,
    ) -> FileOperationResult:
        return await move_file(
            src,
            dest,
            session,
            get_file_record=self._get_file_record,
            versioning=self.version_provider,
            ensure_parent_dirs=self._ensure_parent_dirs,
            file_model=self.file_model,
            file_version_model=self.file_version_model,
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult:
        return await copy_file(
            src,
            dest,
            session,
            get_file_record=self._get_file_record,
            read_content=self._read_content,
            write_fn=self.write,
        )

    async def move_files(
        self,
        pairs: list[tuple[str, str]],
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Batch move files (no directories). Three-phase, all-or-nothing."""
        if not pairs:
            return GroverResult(success=True, message="No files to move")

        # --- Phase 1: Prepare (no side effects) ---
        normalized: list[tuple[str, str]] = []
        all_src_paths: list[str] = []
        all_dest_paths: list[str] = []

        for src, dest in pairs:
            src, dest = normalize_path(src), normalize_path(dest)
            if src == dest:
                return GroverResult(success=False, message=f"Source and destination are the same: {src}")
            normalized.append((src, dest))
            all_src_paths.append(src)
            all_dest_paths.append(dest)

        if len(set(all_dest_paths)) != len(all_dest_paths):
            return GroverResult(success=False, message="Duplicate destination paths in batch")

        if set(all_src_paths) & set(all_dest_paths):
            path = next(iter(set(all_src_paths) & set(all_dest_paths)))
            return GroverResult(success=False, message=f"Source path also appears as destination: {path}")

        src_records = await self._batch_get_file_records(session, all_src_paths)
        dest_records = await self._batch_get_file_records(session, all_dest_paths, include_deleted=True)

        # Validate all pairs and read source content
        src_contents: dict[str, str] = {}
        for src, dest in normalized:
            src_rec = src_records.get(src)
            if src_rec is None:
                return GroverResult(success=False, message=f"Source not found: {src}")
            if src_rec.is_directory:
                return GroverResult(success=False, message=f"Source is a directory (use single move): {src}")
            dest_rec = dest_records.get(dest)
            if dest_rec and dest_rec.is_directory:
                return GroverResult(success=False, message=f"Destination is a directory: {dest}")

            content = await self._read_content(src, session)
            if content is None:
                return GroverResult(success=False, message=f"Source content not found: {src}")
            src_contents[src] = content

        # --- Phase 2: Mutate (single flush) ---
        file_results: list[File] = []
        version_records = []

        try:
            seen_parents: set[str] = set()
            for _, dest in normalized:
                parent = split_path(dest)[0]
                if parent not in seen_parents:
                    seen_parents.add(parent)
                    await self._ensure_parent_dirs(session, dest)

            now = datetime.now(UTC)

            for src, dest in normalized:
                src_rec = src_records[src]
                dest_rec = dest_records.get(dest)
                content = src_contents[src]
                content_hash, size_bytes = compute_content_hash(content)

                # Soft-delete source
                src_rec.original_path = src_rec.path
                src_rec.path = to_trash_path(src_rec.path, src_rec.id)
                src_rec.deleted_at = now

                if dest_rec is not None:
                    old_dest_content = await self._read_content(dest, session) or ""
                    dest_rec.current_version += 1
                    dest_rec.content_hash = content_hash
                    dest_rec.size_bytes = size_bytes
                    dest_rec.deleted_at = None
                    dest_rec.original_path = None
                    dest_rec.updated_at = now
                    await session.merge(dest_rec)

                    v = dest_rec.current_version
                    is_snap = (v % SNAPSHOT_INTERVAL == 0) or (v == 1)
                    stored = content if is_snap or not old_dest_content else compute_diff(old_dest_content, content)
                    version_records.append(
                        self.file_version_model(
                            file_path=dest,
                            path=f"{dest}@{v}",
                            version=v,
                            is_snapshot=is_snap or not old_dest_content,
                            content=stored,
                            content_hash=content_hash,
                            size_bytes=size_bytes,
                        )
                    )
                    version = v
                else:
                    dest_parent, dest_name = split_path(dest)
                    new_file = self.file_model(
                        path=dest,
                        parent_path=dest_parent,
                        owner_id=src_rec.owner_id,
                        content_hash=content_hash,
                        size_bytes=size_bytes,
                        mime_type=guess_mime_type(dest_name),
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(new_file)
                    version_records.append(
                        self.file_version_model(
                            file_path=dest,
                            path=f"{dest}@1",
                            version=1,
                            is_snapshot=True,
                            content=content,
                            content_hash=content_hash,
                            size_bytes=size_bytes,
                        )
                    )
                    version = 1

                if self.storage_provider is not None:
                    await self.storage_provider.write_content(dest, content)

                file_results.append(
                    File(
                        path=dest,
                        current_version=version,
                        evidence=[
                            MoveDetail(
                                operation="move",
                                success=True,
                                message=f"Moved {src} -> {dest}",
                                source_path=src,
                                version=version,
                            )
                        ],
                    )
                )

            if version_records:
                session.add_all(version_records)
            await session.flush()

        except Exception as e:
            return GroverResult(
                success=False,
                message=str(e),
                files=[
                    File(
                        path=d,
                        evidence=[
                            MoveDetail(
                                operation="move",
                                success=False,
                                message=str(e),
                                source_path=s,
                            )
                        ],
                    )
                    for s, d in normalized
                ],
            )

        return GroverResult(
            success=True,
            message=f"Moved {len(file_results)} file(s)",
            files=file_results,
        )

    async def copy_files(
        self,
        pairs: list[tuple[str, str]],
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Batch copy files (no directories). Delegates to write_files."""
        if not pairs:
            return GroverResult(success=True, message="No files to copy")

        all_src_paths = [normalize_path(src) for src, _ in pairs]
        src_records = await self._batch_get_file_records(session, all_src_paths)

        dest_files: list[FileModelBase] = []
        src_by_dest: dict[str, str] = {}

        for src, dest in pairs:
            src, dest = normalize_path(src), normalize_path(dest)
            src_rec = src_records.get(src)
            if src_rec is None:
                return GroverResult(success=False, message=f"Source not found: {src}")
            if src_rec.is_directory:
                return GroverResult(success=False, message=f"Source is a directory (use single copy): {src}")

            content = await self._read_content(src, session)
            if content is None:
                return GroverResult(success=False, message=f"Source content not found: {src}")

            dest_files.append(self.file_model(path=dest, content=content))
            src_by_dest[dest] = src

        result = await self.write_files(dest_files, overwrite=True, session=session)

        # Map WriteDetail -> CopyDetail
        for f in result.files:
            src_path = src_by_dest.get(f.path, "")
            f.evidence = [
                CopyDetail(
                    operation="copy",
                    success=d.success,
                    message=d.message,
                    source_path=src_path,
                    version=d.version,
                )
                if isinstance(d, WriteDetail)
                else d
                for d in f.evidence
            ]

        result.message = result.message.replace("Wrote", "Copied")
        return result

    # ------------------------------------------------------------------
    # Search / Query Operations
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_glob(pattern, path)
            result = self._search_to_internal(old)
            if candidates is not None:
                allowed = set(candidates.paths)
                result.files = [f for f in result.files if f.path in allowed]
                result.message = f"Found {len(result.files)} match(es) (filtered)"
            return result
        path = normalize_path(path)

        if not pattern:
            return FileSearchResult(
                success=False,
                message="Empty glob pattern",
            )

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self._get_file_record(session, path)
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
            result = await session.execute(
                select(model).where(
                    model.path.like(like_pattern, escape="\\"),  # type: ignore[union-attr]
                    model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                )
            )
            db_files = list(result.scalars().all())
        else:
            # Fall back: load all non-deleted files under path
            if path == "/":
                result = await session.execute(
                    select(model).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
            else:
                result = await session.execute(
                    select(model).where(
                        model.path.like(path + "/%", escape="\\"),  # type: ignore[union-attr]
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                    )
                )
            db_files = list(result.scalars().all())

        # Post-filter with authoritative match (compile once for all candidates)
        glob_regex = compile_glob(pattern, path)
        matched = [f for f in db_files if glob_regex is not None and glob_regex.match(f.path) is not None]

        files: list[File] = []
        for f in matched:
            info = file_to_info(f)
            is_dir = isinstance(info, Directory)
            size = 0 if is_dir else info.size_bytes
            mime = "" if is_dir else info.mime_type
            files.append(
                File(
                    path=info.path,
                    evidence=[
                        GlobEvidence(
                            operation="glob",
                            is_directory=is_dir,
                            size_bytes=size,
                            mime_type=mime,
                        )
                    ],
                )
            )

        if candidates is not None:
            allowed = set(candidates.paths)
            files = [f for f in files if f.path in allowed]

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
        candidates: FileSearchSet | None = None,
        session: AsyncSession,
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
            result = self._search_to_internal(old)
            if candidates is not None:
                allowed = set(candidates.paths)
                result.files = [f for f in result.files if f.path in allowed]
                result.message = f"Found {len(result.files)} match(es) (filtered)"
            return result
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
            glob_result = await self.glob(glob_filter, path, session=session)
            if not glob_result.success:
                return FileSearchResult(
                    success=False,
                    message=glob_result.message,
                )
            # Extract non-directory paths from glob result
            candidate_paths = [
                f.path
                for f in glob_result.files
                if not any(isinstance(e, GlobEvidence) and e.is_directory for e in f.evidence)
            ]
        else:
            model = self.file_model
            if path != "/":
                file = await self._get_file_record(session, path)
                if file and not file.is_directory:
                    candidate_paths = [path]
                elif file and file.is_directory:
                    result = await session.execute(
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
                result = await session.execute(
                    select(model.path).where(
                        model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                        model.is_directory.is_(False),  # type: ignore[union-attr]
                    )
                )
                candidate_paths = [row[0] for row in result.all()]

        # Pre-filter by candidates set
        if candidates is not None:
            allowed = set(candidates.paths)
            candidate_paths = [p for p in candidate_paths if p in allowed]

        result_files: list[File] = []
        files_searched = 0
        files_matched = 0
        total_matches = 0

        for file_path in candidate_paths:
            if has_binary_extension(file_path):
                continue

            content = await self._read_content(file_path, session)
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> GroverResult:
        if self.storage_provider is not None:
            old = await self.storage_provider.storage_tree(path, max_depth=max_depth)
            result = self._search_to_internal(old)
            # TODO: migrate storage_provider path to GroverResult
            return GroverResult(success=result.success, message=result.message, files=result.files)
        path = normalize_path(path)

        # Verify base directory exists (unless root)
        if path != "/":
            dir_file = await self._get_file_record(session, path)
            if not dir_file:
                return GroverResult(success=False, message=f"Directory not found: {path}")
            if not dir_file.is_directory:
                return GroverResult(success=False, message=f"Not a directory: {path}")

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

        result = await session.execute(select(model).where(*conditions))

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

        files.sort(key=lambda f: f.path)
        directories.sort(key=lambda d: d.path)
        return GroverResult(
            success=True,
            message=f"{len(directories)} directories, {len(files)} files",
            files=files,
            directories=directories,
        )

    # ------------------------------------------------------------------
    # Trash operations
    # ------------------------------------------------------------------

    async def list_trash(
        self,
        *,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        model = self.file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
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
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)

        model = self.file_model
        conditions = [
            model.original_path == path,
            model.deleted_at.is_not(None),  # type: ignore[unresolved-attribute]
        ]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
        file = result.scalar_one_or_none()

        if not file:
            return FileOperationResult(success=False, message=f"File not in trash: {path}")

        original = file.original_path or path

        # If path is occupied, overwrite the occupant (git restore semantics).
        existing = await self._get_file_record(session, original, False)
        if existing and existing.id != file.id:
            await self.version_provider.delete_versions(session, existing.id)
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
                child_existing = await self._get_file_record(session, child_original, False)
                if child_existing and child_existing.id != child.id:
                    await self.version_provider.delete_versions(session, child_existing.id)
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

        return FileOperationResult(
            success=True,
            message=f"Restored from trash: {path}",
            file=File(path=path),
        )

    async def empty_trash(
        self,
        *,
        session: AsyncSession,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        model = self.file_model
        conditions = [model.deleted_at.is_not(None)]  # type: ignore[unresolved-attribute]
        if owner_id is not None:
            conditions.append(model.owner_id == owner_id)
        result = await session.execute(select(model).where(*conditions))
        db_files = result.scalars().all()

        count = len(db_files)
        for file in db_files:
            original = file.original_path or file.path
            await self.version_provider.delete_versions(session, original)
            await self._delete_content(original, session)
            await session.delete(file)

        await session.flush()

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
        session: AsyncSession,
    ) -> FileOperationResult:
        model = self.file_connection_model
        path = f"{source_path}[{connection_type}]{target_path}"

        result = await session.execute(select(model).where(model.path == path))
        existing = result.scalar_one_or_none()

        if existing is not None:
            existing.weight = weight
            await session.flush()
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
        session.add(record)
        await session.flush()
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
        session: AsyncSession,
    ) -> FileOperationResult:
        model = self.file_connection_model

        if connection_type is not None:
            path = f"{source_path}[{connection_type}]{target_path}"
            result = await session.execute(select(model).where(model.path == path))
            row = result.scalar_one_or_none()
            if row is None:
                return FileOperationResult(
                    success=False,
                    message=f"Connection not found: {path}",
                    file=File(path=path),
                )
            await session.delete(row)
            await session.flush()
            return FileOperationResult(
                success=True,
                message="Connection deleted",
                file=File(path=path),
            )

        # Delete all connections between source and target
        result = await session.execute(
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
            await session.delete(row)
        await session.flush()
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
        result = await session.execute(select(model).where((model.source_path == path) | (model.target_path == path)))
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
        session: AsyncSession,
    ) -> FileSearchResult:
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
        result = await session.execute(stmt)
        db_connections = list(result.scalars().all())
        connections = [
            FileConnection(
                path=c.path,
                source_path=c.source_path,
                target_path=c.target_path,
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
            )
            if rowcount > 0:
                created_dirs.append(dir_path)

        await session.flush()
        return created_dirs, None

    # ------------------------------------------------------------------
    # Chunk methods (inlined from ChunkMethodsMixin)
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession,
    ) -> FileOperationResult:
        return await self.chunk_provider.replace_file_chunks(session, file_path, chunks)

    async def delete_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession,
    ) -> FileOperationResult:
        return await self.chunk_provider.delete_file_chunks(session, file_path)

    async def list_file_chunks(
        self,
        file_path: str,
        *,
        session: AsyncSession,
    ) -> FileOperationResult:
        return await self.chunk_provider.list_file_chunks(session, file_path)

    async def write_chunk(
        self,
        chunk: FileChunkModelBase,
        *,
        session: AsyncSession,
    ) -> FileOperationResult:
        return await self.chunk_provider.write_chunk(session, chunk)

    async def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        session: AsyncSession,
    ) -> BatchResult:
        return await self.chunk_provider.write_chunks(session, chunks)

    async def write_files(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool = True,
        session: AsyncSession,
    ) -> GroverResult:
        """Batch write files from model instances.

        Three-phase approach:
        1. Prepare — batch-fetch existing records, classify each file (per-file errors)
        2. Mutate  — parent dirs, session adds/merges, versions, storage
        3. Results — only built after flush succeeds

        Results are returned in input order.  Per-file outcomes are carried
        as ``WriteDetail`` objects on each ``File.details``.
        """
        if not files:
            return GroverResult(success=True, message="No files to write")

        # --- Phase 1: Prepare (no side effects) ---
        # Results indexed by input position to preserve ordering.
        file_results: dict[int, File] = {}
        pending: list[tuple[int, FileModelBase, FileModelBase | None]] = []  # (idx, incoming, existing|None)

        all_paths = [f.path for f in files]
        existing_map = await self._batch_get_file_records(session, all_paths, include_deleted=True)

        for idx, f in enumerate(files):
            try:
                existing = existing_map.get(f.path)
                if existing and not overwrite and not existing.deleted_at:
                    file_results[idx] = File(
                        path=f.path,
                        evidence=[
                            WriteDetail(
                                operation="write",
                                success=False,
                                message=f"File already exists: {f.path}",
                            )
                        ],
                    )
                else:
                    # Validate into concrete table model early to catch bad data
                    self.file_model.model_validate(f.model_dump())
                    pending.append((idx, f, existing))
            except Exception as e:
                file_results[idx] = File(
                    path=f.path,
                    evidence=[
                        WriteDetail(
                            operation="write",
                            success=False,
                            message=f"Validation failed: {e}",
                        )
                    ],
                )

        # --- Phase 2: Mutate (all session mutations, no results yet) ---
        records_by_idx: dict[int, FileModelBase] = {}
        version_records = []

        try:
            # Parent dirs — deduplicated across new files
            seen_parents: set[str] = set()
            for _idx, f, existing in pending:
                if existing is None:
                    parent = split_path(f.path)[0]
                    if parent not in seen_parents:
                        seen_parents.add(parent)
                        await self._ensure_parent_dirs(session, f.path, f.owner_id)

            for idx, f, existing in pending:
                record = self.file_model.model_validate(f.model_dump())

                if existing is not None:
                    # Update — copy identity from existing, merge incoming
                    old_content = existing.content or ""
                    record.id = existing.id
                    record.current_version = existing.current_version + 1
                    record.created_at = existing.created_at
                    record.deleted_at = None
                    record.original_path = None
                    await session.merge(record)

                    # Build version record
                    version_num = record.current_version
                    new_content = record.content or ""
                    is_snap = (version_num % SNAPSHOT_INTERVAL == 0) or (version_num == 1)
                    stored = new_content if is_snap or not old_content else compute_diff(old_content, new_content)
                    content_hash, size_bytes = compute_content_hash(new_content)
                    version_records.append(
                        self.file_version_model(
                            file_path=record.path,
                            path=f"{record.path}@{version_num}",
                            version=version_num,
                            is_snapshot=is_snap or not old_content,
                            content=stored,
                            content_hash=content_hash,
                            size_bytes=size_bytes,
                        )
                    )
                else:
                    # Create
                    session.add(record)

                    new_content = record.content or ""
                    content_hash, size_bytes = compute_content_hash(new_content)
                    version_records.append(
                        self.file_version_model(
                            file_path=record.path,
                            path=f"{record.path}@1",
                            version=1,
                            is_snapshot=True,
                            content=new_content,
                            content_hash=content_hash,
                            size_bytes=size_bytes,
                        )
                    )

                records_by_idx[idx] = record

            if version_records:
                session.add_all(version_records)

            # Storage provider writes (content-before-commit)
            if self.storage_provider is not None:
                for record in records_by_idx.values():
                    await self.storage_provider.write_content(record.path, record.content or "")

            await session.flush()

            # --- Phase 3: Results (only after flush succeeds) ---
            for idx, record in records_by_idx.items():
                is_create = any(existing is None for i, _, existing in pending if i == idx)
                version = 1 if is_create else record.current_version
                msg = f"Created: {record.path} (v1)" if is_create else f"Updated: {record.path} (v{version})"
                file_results[idx] = File(
                    path=record.path,
                    current_version=version,
                    evidence=[
                        WriteDetail(
                            operation="write",
                            success=True,
                            message=msg,
                            version=version,
                        )
                    ],
                )

        except Exception as e:
            for idx, f, _ in pending:
                if idx not in file_results:
                    file_results[idx] = File(
                        path=f.path,
                        evidence=[
                            WriteDetail(
                                operation="write",
                                success=False,
                                message=str(e),
                            )
                        ],
                    )

        # Build results in input order
        result_files = [file_results[i] for i in range(len(files))]
        succeeded = sum(1 for f in result_files if all(d.success for d in f.details))
        failed = len(result_files) - succeeded
        return GroverResult(
            success=failed == 0,
            message=f"Wrote {succeeded} file(s)" + (f", {failed} failed" if failed else ""),
            files=result_files,
        )

    # ------------------------------------------------------------------
    # Version methods (inlined from VersionMethodsMixin)
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileSearchResult:
        path = normalize_path(path)
        file = await self._get_file_record(session, path)
        if not file:
            return FileSearchResult(success=False, message=f"File not found: {path}")
        versions = await self.version_provider.list_versions(session, file)
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        file = await self._get_file_record(session, path)
        if not file:
            return FileOperationResult(
                success=False,
                message=f"File not found: {path}",
            )
        content = await self.version_provider.get_version_content(session, file, version)
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        vc_result = await self.get_version_content(path, version, session=session)
        if not vc_result.success or not vc_result.file or vc_result.file.content is None:
            return FileOperationResult(
                success=False,
                message=f"Version {version} not found for {path}",
            )

        write_result = await self.write(
            path,
            vc_result.file.content,
            created_by="restore",
            session=session,
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
        session: AsyncSession,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        file = await self._get_file_record(session, path)
        if not file:
            return FileOperationResult(
                success=False,
                message=f"File not found: {path}",
                file=File(path=path),
            )
        return await self.version_provider.verify_chain(session, file)

    async def verify_all_versions(
        self,
        *,
        session: AsyncSession,
        user_id: str | None = None,
    ) -> list[FileOperationResult]:
        model = self.file_model
        result = await session.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[union-attr]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        results: list[FileOperationResult] = []
        for file in result.scalars().all():
            results.append(await self.version_provider.verify_chain(session, file))  # noqa: PERF401
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
        session: AsyncSession,
    ) -> None:
        """Embed *content* and upsert to the vector store."""
        if self.search_provider is not None and self.embedding_provider is not None:
            vector = await self._search_embed(content)
            file_obj = File(path=path, embedding=vector)
            await self.search_provider.upsert(files=[file_obj])

    async def search_add_batch(
        self,
        entries: list[EmbeddableChunk],
        *,
        session: AsyncSession,
    ) -> None:
        """Embed a batch of entries and upsert to the vector store."""
        if not entries:
            return

        if self.search_provider is not None and self.embedding_provider is not None:
            texts = [e.content for e in entries]
            vectors = await self._search_embed_batch(texts)

            file_objs = [File(path=entry.path, embedding=vectors[i]) for i, entry in enumerate(entries)]
            await self.search_provider.upsert(files=file_objs)

    async def search_remove(
        self,
        path: str,
        *,
        session: AsyncSession,
    ) -> None:
        """Remove a single entry by path from the vector store."""
        if self.search_provider is not None:
            await self.search_provider.delete(files=[path])

    async def search_remove_file(
        self,
        path: str,
        *,
        session: AsyncSession,
    ) -> None:
        """Remove *path* and all entries whose ``parent_path`` matches."""
        if self.search_provider is not None:
            local_store = self._search_get_local_store()
            if local_store is not None:
                local_store.remove_file(path)
            else:
                await self.search_provider.delete(files=[path])

    async def vector_search(
        self, query: str, k: int = 10, *, candidates: FileSearchSet | None = None
    ) -> FileSearchResult:
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
        return await self.search_provider.vector_search(vector, k=k, candidates=candidates)

    async def lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Lexical search: DB-based full-text search."""
        return await self._db_lexical_search(query, k=k, session=session)

    async def _db_lexical_search(
        self,
        query: str,
        *,
        k: int = 10,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Dialect-aware full-text search against DB content."""
        from sqlalchemy import text

        from grover.models.internal.evidence import LexicalEvidence

        model = self.file_model

        def _rows_to_result(rows: Iterable) -> FileSearchResult:
            entries: dict[str, list[LexicalEvidence]] = {}
            for row in rows:
                path = row[0]
                content = row[1] or ""
                snippet = content[:200] + ("..." if len(content) > 200 else "") if content else ""
                ev = LexicalEvidence(operation="lexical_search", snippet=snippet)
                entries.setdefault(path, []).append(ev)
            files = [File(path=p, evidence=list(evs)) for p, evs in entries.items()]
            return FileSearchResult(
                success=True,
                message=f"Found matches in {len(files)} file(s)",
                files=files,
            )

        if self.dialect == "sqlite":
            try:
                table_name = getattr(model, "__tablename__", "grover_files")
                fts_table = f"{table_name}_fts"
                stmt = text(f"SELECT path, content FROM {fts_table} WHERE {fts_table} MATCH :query LIMIT :k")
                rows = await session.execute(stmt, {"query": query, "k": k})
                return _rows_to_result(rows)
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
                rows = await session.execute(stmt, {"query": query})
                return _rows_to_result(rows)
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
                rows = await session.execute(stmt, {"query": query})
                return _rows_to_result(rows)
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
        rows = await session.execute(stmt)
        return _rows_to_result(rows)

    def search_has(self, path: str) -> bool:
        """Return whether *path* is present in the local vector store."""
        local = self._search_get_local_store()
        if local is not None:
            return local.has(path)
        return False

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

    # ------------------------------------------------------------------
    # Graph query delegation
    # ------------------------------------------------------------------

    _NO_GRAPH = FileSearchResult(success=False, message="No graph provider")

    async def predecessors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.predecessors(candidates, session=session)

    async def successors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.successors(candidates, session=session)

    async def ancestors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.ancestors(candidates, session=session)

    async def descendants(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.descendants(candidates, session=session)

    async def neighborhood(
        self,
        candidates: FileSearchSet,
        *,
        max_depth: int = 2,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.neighborhood(candidates, max_depth=max_depth, session=session)

    async def meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.meeting_subgraph(candidates, session=session)

    async def min_meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.min_meeting_subgraph(candidates, session=session)

    async def pagerank(
        self,
        candidates: FileSearchSet,
        *,
        personalization: dict[str, float] | None = None,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.pagerank(candidates, personalization=personalization, session=session)

    async def betweenness_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.betweenness_centrality(candidates, session=session)

    async def closeness_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.closeness_centrality(candidates, session=session)

    async def katz_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.katz_centrality(candidates, session=session)

    async def degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.degree_centrality(candidates, session=session)

    async def in_degree_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.in_degree_centrality(candidates, session=session)

    async def out_degree_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.out_degree_centrality(candidates, session=session)

    async def hits(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        if self.graph_provider is None:
            return self._NO_GRAPH
        return await self.graph_provider.hits(candidates, session=session)

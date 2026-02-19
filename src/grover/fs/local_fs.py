"""LocalFileSystem — disk + SQLite versioning, no base class."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import select

from grover.models.chunks import FileChunk
from grover.models.files import File, FileVersion

from .chunks import ChunkService
from .directories import DirectoryService
from .exceptions import GroverError
from .metadata import MetadataService
from .operations import (
    copy_file,
    delete_file,
    edit_file,
    move_file,
    write_file,
)
from .trash import TrashService
from .types import (
    DeleteResult,
    EditResult,
    FileInfo,
    GetVersionContentResult,
    GlobResult,
    GrepMatch,
    GrepResult,
    ListResult,
    ListVersionsResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    RestoreResult,
    TreeResult,
    WriteResult,
)
from .utils import (
    compile_glob,
    get_similar_files,
    has_binary_extension,
    is_binary_file,
    normalize_path,
    split_path,
    to_trash_path,
    validate_path,
)
from .versioning import VersioningService

if TYPE_CHECKING:
    from grover.models.chunks import FileChunkBase
    from grover.models.files import FileBase, FileVersionBase

    from .sharing import SharingService

logger = logging.getLogger(__name__)


def _workspace_slug(workspace_dir: Path) -> str:
    """Derive a directory-safe slug from a workspace path."""
    try:
        relative = workspace_dir.resolve().relative_to(Path.home())
    except ValueError:
        relative = Path(str(workspace_dir.resolve()).lstrip("/"))
    return str(relative).replace("/", "_").strip("_")


def _default_data_dir(workspace_dir: Path) -> Path:
    """Return the global data directory for a given workspace."""
    return Path.home() / ".grover" / _workspace_slug(workspace_dir)


class LocalFileSystem:
    """Local file system with disk storage and SQLite versioning.

    - Files stored on disk at ``{workspace_dir}/{path}``
    - Metadata and versions in SQLite at ``~/.grover/{slug}/file_versions.db``
    - IDE, git, and other tools can see/edit files directly

    Implements ``StorageBackend``, ``SupportsVersions``, ``SupportsTrash``,
    and ``SupportsReconcile`` protocols.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        file_model: type[FileBase] | None = None,
        file_version_model: type[FileVersionBase] | None = None,
        file_chunk_model: type[FileChunkBase] | None = None,
        schema: str | None = None,
        *,
        graph: Any | None = None,
        search_engine: Any | None = None,
    ) -> None:
        fm: type[FileBase] = file_model or File
        fvm: type[FileVersionBase] = file_version_model or FileVersion
        fcm: type[FileChunkBase] = file_chunk_model or FileChunk

        self.dialect = "sqlite"
        self.schema = schema
        self._file_model = fm
        self._file_version_model = fvm
        self._file_chunk_model = fcm

        # Per-mount graph and search engine (injected at mount time)
        self._graph = graph
        self._search_engine = search_engine

        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir(self.workspace_dir)

        self._engine = None
        self._session_factory = None
        self._init_lock = asyncio.Lock()

        # Composed services
        self.metadata = MetadataService(fm)
        self.versioning = VersioningService(fm, fvm)
        self.directories = DirectoryService(fm, "sqlite", schema)
        self.trash = TrashService(fm, self.versioning, self._delete_content)
        self.chunks = ChunkService(fcm)

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
    def session_factory(self) -> async_sessionmaker | None:
        """The async session factory, available after ``open()``."""
        return self._session_factory

    @property
    def engine(self) -> AsyncEngine | None:
        """The async engine, available after ``open()``."""
        return self._engine

    @staticmethod
    def _require_session(session: AsyncSession | None) -> AsyncSession:
        if session is None:
            raise GroverError("LocalFileSystem requires a session")
        return session

    # ------------------------------------------------------------------
    # Database Management
    # ------------------------------------------------------------------

    async def _ensure_db(self) -> None:
        """Initialize database if needed."""
        if self._session_factory is not None:
            return
        async with self._init_lock:
            if self._session_factory is not None:
                return

            self.data_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.data_dir / "file_versions.db"

            self._engine = create_async_engine(
                f"sqlite+aiosqlite:///{db_path}",
                echo=False,
            )

            @event.listens_for(self._engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
                cursor.execute("PRAGMA journal_mode=WAL")
                result = cursor.fetchone()
                if result[0].lower() != "wal":
                    logging.getLogger(__name__).warning(
                        "WAL mode not active, got: %s",
                        result[0],
                    )
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA synchronous=FULL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            async with self._engine.begin() as conn:
                fm_table = self._file_model.__table__  # type: ignore[unresolved-attribute]
                fv_table = self._file_version_model.__table__  # type: ignore[unresolved-attribute]
                fc_table = self._file_chunk_model.__table__  # type: ignore[unresolved-attribute]
                await conn.run_sync(lambda c: fm_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: fv_table.create(c, checkfirst=True))
                await conn.run_sync(lambda c: fc_table.create(c, checkfirst=True))
                # Create edges table for per-mount graph persistence
                from grover.models.edges import GroverEdge

                edge_table = GroverEdge.__table__  # type: ignore[unresolved-attribute]
                await conn.run_sync(lambda c: edge_table.create(c, checkfirst=True))

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Initialize the SQLite database."""
        await self._ensure_db()

    async def close(self) -> None:
        """Close database engine and release resources."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path_sync(self, virtual_path: str) -> Path:
        """Convert virtual path to an actual disk path within the workspace."""
        virtual_path = normalize_path(virtual_path)
        rel = virtual_path.lstrip("/")
        if not rel:
            return self.workspace_dir

        candidate = self.workspace_dir / rel

        current = self.workspace_dir
        for part in Path(rel).parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError(
                    f"Symlinks not allowed: {virtual_path} contains symlink at "
                    f"{current.relative_to(self.workspace_dir)}"
                )

        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_dir.resolve())
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: {virtual_path} resolves outside workspace"
            ) from None

        return resolved

    async def _resolve_path(self, virtual_path: str) -> Path:
        return await asyncio.to_thread(self._resolve_path_sync, virtual_path)

    def _to_virtual_path(self, physical_path: Path) -> str:
        rel = physical_path.resolve().relative_to(self.workspace_dir.resolve())
        vpath = "/" + str(rel).replace("\\", "/")
        return vpath if vpath != "/." else "/"

    # ------------------------------------------------------------------
    # Content helpers (disk-specific)
    # ------------------------------------------------------------------

    async def _read_content(self, path: str, session: AsyncSession) -> str | None:
        try:
            actual_path = await self._resolve_path(path)
        except (PermissionError, ValueError):
            return None

        def _do_read() -> str | None:
            if not actual_path.exists() or actual_path.is_dir():
                return None
            return actual_path.read_text(encoding="utf-8")

        try:
            return await asyncio.to_thread(_do_read)
        except (UnicodeDecodeError, PermissionError, OSError):
            return None

    async def _write_content(self, path: str, content: str, session: AsyncSession) -> None:
        actual_path = await self._resolve_path(path)

        def _do_write() -> None:
            actual_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=actual_path.parent,
                prefix=".tmp_",
                suffix=actual_path.suffix,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp_path).replace(actual_path)
            except Exception:
                tmp = Path(tmp_path)
                if tmp.exists():
                    tmp.unlink()
                raise

        await asyncio.to_thread(_do_write)

    async def _delete_content(self, path: str, session: AsyncSession) -> None:
        try:
            actual_path = await self._resolve_path(path)
        except (PermissionError, ValueError):
            return

        def _do_delete() -> None:
            try:
                if actual_path.is_dir():
                    shutil.rmtree(actual_path)
                else:
                    actual_path.unlink()
            except FileNotFoundError:
                pass

        await asyncio.to_thread(_do_delete)

    async def _content_exists(self, path: str) -> bool:
        try:
            actual_path = await self._resolve_path(path)
            return await asyncio.to_thread(actual_path.exists)
        except (PermissionError, ValueError):
            return False

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
        """Read file with binary check and similar file suggestions."""
        sess = self._require_session(session)

        valid, error = validate_path(path)
        if not valid:
            return ReadResult(success=False, message=error)

        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return ReadResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            parent_exists = await asyncio.to_thread(actual_path.parent.exists)
            if parent_exists:
                suggestions = await asyncio.to_thread(
                    get_similar_files, actual_path.parent, actual_path.name
                )
                if suggestions:
                    suggestion_text = "\n".join(f"  {s}" for s in suggestions)
                    return ReadResult(
                        success=False,
                        message=f"File not found: {path}\n\nDid you mean?\n{suggestion_text}",
                    )
            return ReadResult(success=False, message=f"File not found: {path}")

        if await asyncio.to_thread(is_binary_file, actual_path):
            return ReadResult(success=False, message=f"Cannot read binary file: {path}")

        if await asyncio.to_thread(actual_path.is_dir):
            return ReadResult(success=False, message=f"Path is a directory, not a file: {path}")

        content = await self._read_content(path, sess)

        if content is None:
            return ReadResult(success=False, message=f"Could not read file: {path}")

        return MetadataService.paginate_content(content, path, offset, limit)

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
        """Delete file, backing up content to the database first.

        Content-before-commit: DB soft-delete → flush → unlink disk → return.
        VFS commits after.
        """
        sess = self._require_session(session)
        norm = normalize_path(path)

        # Read content from disk before anything else
        content = await self._read_content(norm, sess)

        # Ensure a DB record exists so delete can soft-delete it
        if content is not None:
            file = await self.metadata.get_file(sess, norm)
            if file is None:
                # Disk-only file: create a DB record + version 1 snapshot
                await write_file(
                    norm,
                    content,
                    "backup",
                    True,
                    sess,
                    metadata=self.metadata,
                    versioning=self.versioning,
                    directories=self.directories,
                    file_model=self._file_model,
                    read_content=self._read_content,
                    write_content=self._write_content,
                )

        result = await delete_file(
            norm,
            permanent,
            sess,
            metadata=self.metadata,
            versioning=self.versioning,
            file_model=self._file_model,
            delete_content=self._delete_content,
        )

        # Remove from disk regardless of soft/permanent
        if result.success:
            await self._delete_content(norm, sess)

        return result

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        """Create directory in database and on disk."""
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
        actual_path = await self._resolve_path(path)
        await asyncio.to_thread(actual_path.mkdir, parents=True, exist_ok=True)

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
    ) -> ListResult:
        """List directory, including files only on disk."""
        sess = self._require_session(session)
        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return ListResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return ListResult(success=False, message=f"Directory not found: {path}")

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return ListResult(success=False, message=f"Not a directory: {path}")

        entries: list[FileInfo] = []

        def _scan_dir() -> list[tuple[str, bool, int | None]]:
            """Collect (name, is_dir, size) from disk in one thread."""
            items = []
            for item in actual_path.iterdir():
                if item.name.startswith("."):
                    continue
                is_d = item.is_dir()
                sz = item.stat().st_size if item.is_file() else None
                items.append((item.name, is_d, sz))
            return items

        disk_items = await asyncio.to_thread(_scan_dir)

        for name, is_d, disk_size in disk_items:
            item_path = f"{path}/{name}" if path != "/" else f"/{name}"
            item_path = normalize_path(item_path)

            file = await self.metadata.get_file(sess, item_path)

            entries.append(
                FileInfo(
                    path=item_path,
                    name=name,
                    is_directory=is_d,
                    size_bytes=file.size_bytes if file else disk_size,
                    mime_type=file.mime_type if file else None,
                    version=file.current_version if file else 1,
                    created_at=file.created_at if file else None,
                    updated_at=file.updated_at if file else None,
                )
            )

        return ListResult(
            success=True,
            message=f"Listed {len(entries)} items in {path}",
            entries=entries,
            path=path,
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
    ) -> FileInfo | None:
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
                path=path,
            )

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return GlobResult(
                success=False,
                message=str(e),
                pattern=pattern,
                path=path,
            )

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return GlobResult(
                success=False,
                message=f"Directory not found: {path}",
                pattern=pattern,
                path=path,
            )

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return GlobResult(
                success=False,
                message=f"Not a directory: {path}",
                pattern=pattern,
                path=path,
            )

        glob_regex = compile_glob(pattern, path)

        def _collect_and_match() -> list[tuple[Path, str, bool, int | None]]:
            """Walk disk, filter with compiled glob regex."""
            if glob_regex is None:
                return []
            results: list[tuple[Path, str, bool, int | None]] = []
            for root, dirs, files in os.walk(actual_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in dirs:
                    full = Path(root) / name
                    try:
                        vp = self._to_virtual_path(full)
                    except (ValueError, PermissionError):
                        continue
                    if glob_regex.match(vp) is not None:
                        results.append((full, vp, True, None))
                for name in files:
                    if name.startswith("."):
                        continue
                    full = Path(root) / name
                    try:
                        vp = self._to_virtual_path(full)
                    except (ValueError, PermissionError):
                        continue
                    if glob_regex.match(vp) is not None:
                        try:
                            sz = full.stat().st_size
                        except OSError:
                            sz = None
                        results.append((full, vp, False, sz))
            return results

        matched = await asyncio.to_thread(_collect_and_match)

        # Batch metadata lookup
        vpaths = [vp for _, vp, _, _ in matched]
        model = self._file_model
        if vpaths:
            db_result = await sess.execute(
                select(model).where(model.path.in_(vpaths))  # type: ignore[union-attr]
            )
            file_map = {f.path: f for f in db_result.scalars().all()}
        else:
            file_map = {}

        entries: list[FileInfo] = []
        for p, vpath, is_d, size in matched:
            file = file_map.get(vpath)
            entries.append(
                FileInfo(
                    path=vpath,
                    name=p.name,
                    is_directory=is_d,
                    size_bytes=file.size_bytes if file else size,
                    mime_type=file.mime_type if file else None,
                    version=file.current_version if file else 1,
                    created_at=file.created_at if file else None,
                    updated_at=file.updated_at if file else None,
                )
            )

        return GlobResult(
            success=True,
            message=f"Found {len(entries)} match(es)",
            entries=entries,
            pattern=pattern,
            path=path,
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
                path=path,
            )

        # Get candidate files
        if glob_filter:
            glob_result = await self.glob(glob_filter, path, session=sess)
            if not glob_result.success:
                return GrepResult(
                    success=False,
                    message=glob_result.message,
                    pattern=pattern,
                    path=path,
                )
            candidate_vpaths = [e.path for e in glob_result.entries if not e.is_directory]
        else:
            try:
                actual_path = await self._resolve_path(path)
            except PermissionError as e:
                return GrepResult(
                    success=False,
                    message=str(e),
                    pattern=pattern,
                    path=path,
                )

            # Check if path exists
            exists = await asyncio.to_thread(actual_path.exists)
            if not exists:
                return GrepResult(
                    success=False,
                    message=f"Path not found: {path}",
                    pattern=pattern,
                    path=path,
                )

            # Check if path is a file (not a directory)
            is_file = await asyncio.to_thread(actual_path.is_file)
            if is_file:
                candidate_vpaths = [path]
            else:

                def _collect_files() -> list[str]:
                    vpaths = []
                    for root, dirs, files in os.walk(actual_path):
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        for name in files:
                            if name.startswith("."):
                                continue
                            full = Path(root) / name
                            try:
                                vp = self._to_virtual_path(full)
                                vpaths.append(vp)
                            except (ValueError, PermissionError):
                                continue
                    return vpaths

                candidate_vpaths = await asyncio.to_thread(_collect_files)

        matches: list[GrepMatch] = []
        files_searched = 0
        files_matched = 0
        truncated = False

        for file_path in candidate_vpaths:
            if has_binary_extension(file_path):
                continue

            # Also check content-based binary detection and file size on disk
            try:
                actual = await self._resolve_path(file_path)
                stat = await asyncio.to_thread(actual.stat)
                # Skip files larger than 10 MB
                if stat.st_size > 10 * 1024 * 1024:
                    continue
                if await asyncio.to_thread(is_binary_file, actual):
                    continue
            except (PermissionError, ValueError, OSError):
                continue

            content = await self._read_content(file_path, sess)
            if content is None:
                continue

            files_searched += 1
            lines = content.split("\n")
            file_matches: list[GrepMatch] = []

            for i, line in enumerate(lines):
                has_match = regex.search(line) is not None
                if invert:
                    has_match = not has_match

                if has_match:
                    ctx_before = []
                    ctx_after = []
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        ctx_before = lines[start:i]
                        end = min(len(lines), i + context_lines + 1)
                        ctx_after = lines[i + 1 : end]

                    file_matches.append(
                        GrepMatch(
                            file_path=file_path,
                            line_number=i + 1,
                            line_content=line,
                            context_before=ctx_before,
                            context_after=ctx_after,
                        )
                    )

                    if max_results_per_file > 0 and len(file_matches) >= max_results_per_file:
                        break

            if file_matches:
                files_matched += 1
                if files_only:
                    matches.append(file_matches[0])
                else:
                    matches.extend(file_matches)

                if max_results > 0 and len(matches) >= max_results:
                    truncated = True
                    matches = matches[:max_results]
                    break

        if count_only:
            total = files_matched if files_only else len(matches)
            return GrepResult(
                success=True,
                message=f"Count: {total}",
                matches=[],
                pattern=pattern,
                path=path,
                files_searched=files_searched,
                files_matched=files_matched,
                truncated=truncated,
            )

        return GrepResult(
            success=True,
            message=f"Found {len(matches)} match(es) in {files_matched} file(s)",
            matches=matches,
            pattern=pattern,
            path=path,
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

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return TreeResult(success=False, message=str(e), path=path)

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return TreeResult(
                success=False,
                message=f"Directory not found: {path}",
                path=path,
            )

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return TreeResult(
                success=False,
                message=f"Not a directory: {path}",
                path=path,
            )

        def _walk() -> list[tuple[str, bool, int | None]]:
            """Collect (virtual_path, is_dir, size) with depth limit."""
            items = []
            base_depth = len(actual_path.resolve().parts)
            for root, dirs, files in os.walk(actual_path):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                root_path = Path(root).resolve()
                current_depth = len(root_path.parts) - base_depth

                if max_depth is not None and current_depth >= max_depth:
                    dirs[:] = []
                    continue

                # Add directories at this level
                for d in dirs:
                    full = Path(root) / d
                    try:
                        vp = self._to_virtual_path(full)
                        items.append((vp, True, None))
                    except (ValueError, PermissionError):
                        continue

                # Add files at this level
                for name in sorted(files):
                    if name.startswith("."):
                        continue
                    full = Path(root) / name
                    try:
                        vp = self._to_virtual_path(full)
                        sz = full.stat().st_size if full.is_file() else None
                        items.append((vp, False, sz))
                    except (ValueError, PermissionError, OSError):
                        continue

            return items

        disk_items = await asyncio.to_thread(_walk)

        # Batch metadata lookup
        vpaths = [vp for vp, _, _ in disk_items]
        model = self._file_model
        if vpaths:
            db_result = await sess.execute(
                select(model).where(model.path.in_(vpaths))  # type: ignore[union-attr]
            )
            file_map = {f.path: f for f in db_result.scalars().all()}
        else:
            file_map = {}

        entries: list[FileInfo] = []
        total_files = 0
        total_dirs = 0

        for vpath, is_d, disk_size in disk_items:
            file = file_map.get(vpath)
            name = split_path(vpath)[1]
            entries.append(
                FileInfo(
                    path=vpath,
                    name=name,
                    is_directory=is_d,
                    size_bytes=file.size_bytes if file else disk_size,
                    mime_type=file.mime_type if file else None,
                    version=file.current_version if file else 1,
                    created_at=file.created_at if file else None,
                    updated_at=file.updated_at if file else None,
                )
            )
            if is_d:
                total_dirs += 1
            else:
                total_files += 1

        entries.sort(key=lambda e: e.path)

        return TreeResult(
            success=True,
            message=f"{total_dirs} directories, {total_files} files",
            entries=entries,
            path=path,
            total_files=total_files,
            total_dirs=total_dirs,
        )

    # ------------------------------------------------------------------
    # Capability: SupportsSearch
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> list:
        """Semantic search via per-mount search engine. Returns list[SearchResult]."""
        if self._search_engine is None:
            return []
        results = await self._search_engine.search(query, k)
        if path != "/":
            prefix = path.rstrip("/") + "/"
            results = [
                r
                for r in results
                if (r.parent_path or r.ref.path).startswith(prefix)
                or (r.parent_path or r.ref.path) == path.rstrip("/")
            ]
        return results

    # ------------------------------------------------------------------
    # Capability: SupportsVersions
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> ListVersionsResult:
        sess = self._require_session(session)
        path = normalize_path(path)
        file = await self.metadata.get_file(sess, path)
        if not file:
            return ListVersionsResult(success=False, message=f"File not found: {path}", versions=[])
        versions = await self.versioning.list_versions(sess, file)
        return ListVersionsResult(
            success=True,
            message=f"Found {len(versions)} version(s)",
            versions=versions,
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
            file_path=path,
            restored_version=version,
            current_version=write_result.version,
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
    ) -> ListResult:
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
        """Restore a file from trash, writing content back to disk."""
        sess = self._require_session(session)
        result = await self.trash.restore_from_trash(
            sess, path, self.metadata.get_file, owner_id=owner_id
        )
        if not result.success:
            return result

        restored_path = result.file_path or path
        file = await self.metadata.get_file(sess, restored_path)
        if file:
            if file.is_directory:
                model = self._file_model
                children_result = await sess.execute(
                    select(model).where(
                        model.path.startswith(restored_path + "/"),
                        model.deleted_at.is_(None),  # type: ignore[possibly-missing-attribute]
                    )
                )
                for child in children_result.scalars().all():
                    if not child.is_directory:
                        vc = await self.get_version_content(
                            child.path,
                            child.current_version,
                            session=sess,
                        )
                        if vc.success and vc.content is not None:
                            await self._write_content(child.path, vc.content, sess)
            else:
                vc = await self.get_version_content(
                    restored_path,
                    file.current_version,
                    session=sess,
                )
                if vc.success and vc.content is not None:
                    await self._write_content(restored_path, vc.content, sess)

        return result

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
    # Capability: SupportsReconcile
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        *,
        session: AsyncSession | None = None,
    ) -> dict[str, int]:
        """Walk disk, compare with DB, create/update/soft-delete as needed."""
        sess = self._require_session(session)
        stats = {"created": 0, "updated": 0, "deleted": 0}

        # Walk workspace files
        disk_paths: set[str] = set()

        def _walk() -> list[tuple[str, bool]]:
            items = []
            for root, dirs, files in os.walk(self.workspace_dir):
                # Skip dotfiles/dirs
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    if name.startswith("."):
                        continue
                    full = Path(root) / name
                    try:
                        vpath = self._to_virtual_path(full)
                        items.append((vpath, True))
                    except (ValueError, PermissionError):
                        continue
            return items

        async def _noop_write(
            _path: str,
            _content: str,
            _session: AsyncSession,
        ) -> None:
            pass

        items = await asyncio.to_thread(_walk)
        for vpath, _ in items:
            disk_paths.add(vpath)

            file = await self.metadata.get_file(sess, vpath)
            if file is None:
                # File on disk but not in DB — create DB record only
                # (content already exists on disk, no need to rewrite)
                content = await self._read_content(vpath, sess)
                if content is not None:
                    await write_file(
                        vpath,
                        content,
                        "reconcile",
                        True,
                        sess,
                        metadata=self.metadata,
                        versioning=self.versioning,
                        directories=self.directories,
                        file_model=self._file_model,
                        read_content=self._read_content,
                        write_content=_noop_write,
                    )
                    stats["created"] += 1

        # Check DB records against disk
        model = self._file_model
        result = await sess.execute(
            select(model).where(
                model.deleted_at.is_(None),  # type: ignore[unresolved-attribute]
                model.is_directory.is_(False),  # type: ignore[union-attr]
            )
        )
        for file in result.scalars().all():
            if file.path not in disk_paths:
                exists = await self._content_exists(file.path)
                if not exists:
                    # DB record but no disk file — phantom metadata, soft-delete
                    file.original_path = file.path
                    file.path = to_trash_path(file.path, file.id)
                    file.deleted_at = datetime.now(UTC)
                    stats["deleted"] += 1

        await sess.flush()
        return stats

    # ------------------------------------------------------------------
    # Capability: SupportsFileChunks
    # ------------------------------------------------------------------

    async def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[dict],
        *,
        session: AsyncSession | None = None,
        user_id: str | None = None,
    ) -> int:
        sess = self._require_session(session)
        return await self.chunks.replace_file_chunks(sess, file_path, chunks, user_id=user_id)

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

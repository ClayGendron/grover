"""DiskStorageProvider — local disk storage with no DB dependencies.

Implements ``StorageProvider`` (content I/O, queries, and reconciliation).
Session parameters are NOT part of the ``StorageProvider`` interface — the
filesystem injects sessions for DB operations after the storage provider call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grover.models.internal.evidence import (
    GlobEvidence,
    GrepEvidence,
    LineMatch,
    ListDirEvidence,
    TreeEvidence,
)
from grover.models.internal.ref import File
from grover.models.internal.results import FileOperationResult, FileSearchResult
from grover.util.content import has_binary_extension, is_binary_file
from grover.util.paths import normalize_path, validate_path
from grover.util.patterns import compile_glob

logger = logging.getLogger(__name__)


class DiskStorageProvider:
    """Local disk content I/O and queries.

    Constructor
    -----------
    workspace_dir : str | Path
        Root directory for all file operations.
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

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
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: {virtual_path} resolves outside workspace"
            ) from None

        return resolved

    async def _resolve_path(self, virtual_path: str) -> Path:
        return await asyncio.to_thread(self._resolve_path_sync, virtual_path)

    def _to_virtual_path(self, physical_path: Path) -> str:
        rel = physical_path.resolve().relative_to(self.workspace_dir)
        vpath = "/" + str(rel).replace("\\", "/")
        return vpath if vpath != "/." else "/"

    # ------------------------------------------------------------------
    # StorageProvider implementation
    # ------------------------------------------------------------------

    async def read_content(self, path: str) -> str | None:
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

    async def write_content(self, path: str, content: str) -> None:
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

    async def delete_content(self, path: str) -> None:
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

    async def move_content(self, src: str, dest: str) -> None:
        src_path = await self._resolve_path(src)
        dest_path = await self._resolve_path(dest)

        def _do_move() -> None:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            src_path.rename(dest_path)

        await asyncio.to_thread(_do_move)

    async def copy_content(self, src: str, dest: str) -> None:
        src_path = await self._resolve_path(src)
        dest_path = await self._resolve_path(dest)

        def _do_copy() -> None:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)

        await asyncio.to_thread(_do_copy)

    async def exists(self, path: str) -> bool:
        try:
            actual_path = await self._resolve_path(path)
            return await asyncio.to_thread(actual_path.exists)
        except (PermissionError, ValueError):
            return False

    async def mkdir(self, path: str, parents: bool = True) -> None:
        actual_path = await self._resolve_path(path)
        await asyncio.to_thread(actual_path.mkdir, parents=parents, exist_ok=True)

    async def get_info(self, path: str) -> FileOperationResult:
        valid, error = validate_path(path)
        if not valid:
            return FileOperationResult(success=False, message=error or "", file=File(path=path))

        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return FileOperationResult(success=False, message=str(e), file=File(path=path))

        def _stat() -> FileOperationResult:
            if not actual_path.exists():
                return FileOperationResult(
                    success=False,
                    message=f"File not found: {path}",
                    file=File(path=path),
                )
            stat = actual_path.stat()
            is_dir = actual_path.is_dir()
            lines = 0
            if not is_dir:
                try:
                    text = actual_path.read_text(encoding="utf-8")
                    lines = text.count("\n") + (1 if text else 0)
                except (UnicodeDecodeError, PermissionError, OSError):
                    pass
            return FileOperationResult(
                success=True,
                message="OK",
                file=File(
                    path=path,
                    is_directory=is_dir,
                    lines=lines,
                    created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                ),
            )

        return await asyncio.to_thread(_stat)

    # ------------------------------------------------------------------
    # SupportsStorageQueries implementation
    # ------------------------------------------------------------------

    async def storage_glob(self, pattern: str, path: str = "/") -> FileSearchResult:
        path = normalize_path(path)

        if not pattern:
            return FileSearchResult(
                success=False,
                message="Empty glob pattern",
            )

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return FileSearchResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return FileSearchResult(
                success=False,
                message=f"Directory not found: {path}",
            )

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return FileSearchResult(
                success=False,
                message=f"Not a directory: {path}",
            )

        glob_regex = compile_glob(pattern, path)

        def _collect_and_match() -> list[tuple[str, bool, int | None]]:
            if glob_regex is None:
                return []
            results: list[tuple[str, bool, int | None]] = []
            for root, dirs, files in os.walk(actual_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in dirs:
                    full = Path(root) / name
                    try:
                        vp = self._to_virtual_path(full)
                    except (ValueError, PermissionError):
                        continue
                    if glob_regex.match(vp) is not None:
                        results.append((vp, True, None))
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
                        results.append((vp, False, sz))
            return results

        matched = await asyncio.to_thread(_collect_and_match)

        files: list[File] = []
        for vpath, is_d, size in matched:
            files.append(
                File(
                    path=vpath,
                    evidence=[
                        GlobEvidence(
                            operation="glob",
                            is_directory=is_d,
                            size_bytes=size,
                        )
                    ],
                )
            )

        return FileSearchResult(
            success=True,
            message=f"Found {len(files)} match(es)",
            files=files,
        )

    async def storage_grep(self, pattern: str, path: str = "/", **kwargs: Any) -> FileSearchResult:
        path = normalize_path(path)
        case_sensitive: bool = kwargs.get("case_sensitive", True)
        fixed_string: bool = kwargs.get("fixed_string", False)
        invert: bool = kwargs.get("invert", False)
        word_match: bool = kwargs.get("word_match", False)
        context_lines: int = max(0, kwargs.get("context_lines", 0))
        max_results: int = kwargs.get("max_results", 1000)
        max_results_per_file: int = kwargs.get("max_results_per_file", 0)
        count_only: bool = kwargs.get("count_only", False)
        files_only: bool = kwargs.get("files_only", False)
        glob_filter: str | None = kwargs.get("glob_filter")

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
            glob_result = await self.storage_glob(glob_filter, path)
            if not glob_result.success:
                return FileSearchResult(
                    success=False,
                    message=glob_result.message,
                )
            candidate_vpaths = list(glob_result)
        else:
            try:
                actual_path = await self._resolve_path(path)
            except PermissionError as e:
                return FileSearchResult(success=False, message=str(e))

            exists = await asyncio.to_thread(actual_path.exists)
            if not exists:
                return FileSearchResult(
                    success=False,
                    message=f"Path not found: {path}",
                )

            is_file = await asyncio.to_thread(actual_path.is_file)
            if is_file:
                candidate_vpaths = [path]
            else:

                def _collect_files() -> list[str]:
                    vpaths = []
                    for root, dirs, walk_files in os.walk(actual_path):
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        for name in walk_files:
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

        result_files: list[File] = []
        files_searched = 0
        files_matched = 0
        total_matches = 0

        for file_path in candidate_vpaths:
            if has_binary_extension(file_path):
                continue

            try:
                actual = await self._resolve_path(file_path)
                stat = await asyncio.to_thread(actual.stat)
                if stat.st_size > 10 * 1024 * 1024:
                    continue
                if await asyncio.to_thread(is_binary_file, actual):
                    continue
            except (PermissionError, ValueError, OSError):
                continue

            content = await self.read_content(file_path)
            if content is None:
                continue

            files_searched += 1
            lines = content.split("\n")
            file_line_matches: list[LineMatch] = []

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
                        LineMatch(
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

    async def storage_tree(self, path: str = "/", max_depth: int | None = None) -> FileSearchResult:
        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return FileSearchResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return FileSearchResult(success=False, message=f"Directory not found: {path}")

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return FileSearchResult(success=False, message=f"Not a directory: {path}")

        def _walk() -> list[tuple[str, bool, int]]:
            items: list[tuple[str, bool, int]] = []
            base_depth = len(actual_path.resolve().parts)
            for root, dirs, files in os.walk(actual_path):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                root_path = Path(root).resolve()
                current_depth = len(root_path.parts) - base_depth

                if max_depth is not None and current_depth >= max_depth:
                    dirs[:] = []
                    continue

                for d in dirs:
                    full = Path(root) / d
                    try:
                        vp = self._to_virtual_path(full)
                        items.append((vp, True, current_depth + 1))
                    except (ValueError, PermissionError):
                        continue

                for name in sorted(files):
                    if name.startswith("."):
                        continue
                    full = Path(root) / name
                    try:
                        vp = self._to_virtual_path(full)
                        items.append((vp, False, current_depth + 1))
                    except (ValueError, PermissionError, OSError):
                        continue

            return items

        disk_items = await asyncio.to_thread(_walk)

        files: list[File] = []
        for vpath, is_d, depth in sorted(disk_items, key=lambda x: x[0]):
            files.append(
                File(
                    path=vpath,
                    evidence=[
                        TreeEvidence(
                            operation="tree",
                            depth=depth,
                            is_directory=is_d,
                        )
                    ],
                )
            )

        return FileSearchResult(
            success=True,
            message=f"{sum(1 for _, d, _ in disk_items if d)} directories, "
            f"{sum(1 for _, d, _ in disk_items if not d)} files",
            files=files,
        )

    async def storage_list_dir(self, path: str) -> FileSearchResult:
        path = normalize_path(path)

        try:
            actual_path = await self._resolve_path(path)
        except PermissionError as e:
            return FileSearchResult(success=False, message=str(e))

        exists = await asyncio.to_thread(actual_path.exists)
        if not exists:
            return FileSearchResult(success=False, message=f"Directory not found: {path}")

        is_dir = await asyncio.to_thread(actual_path.is_dir)
        if not is_dir:
            return FileSearchResult(success=False, message=f"Not a directory: {path}")

        def _scan_dir() -> list[tuple[str, bool, int | None]]:
            items = []
            for item in actual_path.iterdir():
                if item.name.startswith("."):
                    continue
                is_d = item.is_dir()
                sz = item.stat().st_size if item.is_file() else None
                items.append((item.name, is_d, sz))
            return items

        disk_items = await asyncio.to_thread(_scan_dir)

        files: list[File] = []
        for name, is_d, disk_size in disk_items:
            item_path = f"{path}/{name}" if path != "/" else f"/{name}"
            item_path = normalize_path(item_path)
            files.append(
                File(
                    path=item_path,
                    evidence=[
                        ListDirEvidence(
                            operation="list_dir",
                            is_directory=is_d,
                            size_bytes=disk_size,
                        )
                    ],
                )
            )

        return FileSearchResult(
            success=True,
            message=f"Listed {len(files)} items in {path}",
            files=files,
        )

    # ------------------------------------------------------------------
    # SupportsStorageReconcile implementation
    # ------------------------------------------------------------------

    async def reconcile(self, **kwargs: Any) -> FileOperationResult:
        """Walk disk, compare with DB, create/update/soft-delete as needed.

        This method requires DB services passed via kwargs because reconcile
        is inherently a cross-cutting operation between disk and DB state.

        Required kwargs:
            session: AsyncSession
            get_file_record: async callable(session, path) -> FileBase | None
            version_provider: DefaultVersionProvider
            ensure_parent_dirs: async callable(session, path, owner_id) -> None
            file_model: type[FileBase]
            read_content: async callable(path, session) -> str | None
        """
        from grover.util.operations import write_file

        session = kwargs["session"]
        get_file_record = kwargs["get_file_record"]
        version_provider = kwargs["version_provider"]
        ensure_parent_dirs = kwargs["ensure_parent_dirs"]
        file_model = kwargs["file_model"]
        read_content = kwargs["read_content"]

        created = 0
        updated = 0
        deleted = 0
        chain_errors = 0

        async def _noop_write(
            _path: str,
            _content: str,
            _session: Any,
        ) -> None:
            pass

        def _walk() -> list[tuple[str, bool]]:
            items = []
            for root, dirs, files in os.walk(self.workspace_dir):
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

        items = await asyncio.to_thread(_walk)
        disk_paths: set[str] = set()

        for vpath, _ in items:
            disk_paths.add(vpath)

            file = await get_file_record(session, vpath)
            if file is None:
                content = await self.read_content(vpath)
                if content is not None:
                    await write_file(
                        vpath,
                        content,
                        "reconcile",
                        True,
                        session,
                        get_file_record=get_file_record,
                        versioning=version_provider,
                        ensure_parent_dirs=ensure_parent_dirs,
                        file_model=file_model,
                        read_content=read_content,
                        write_content=_noop_write,
                    )
                    created += 1

        # Check DB records against disk
        from sqlmodel import select

        result = await session.execute(
            select(file_model).where(
                file_model.deleted_at.is_(None),
                file_model.is_directory.is_(False),
            )
        )

        from grover.util.paths import to_trash_path

        for file in result.scalars().all():
            if file.path not in disk_paths:
                disk_exists = await self.exists(file.path)
                if not disk_exists:
                    file.original_path = file.path
                    file.path = to_trash_path(file.path, file.id)
                    file.deleted_at = datetime.now(UTC)
                    deleted += 1

        await session.flush()

        return FileOperationResult(
            success=True,
            message=(
                f"Reconcile: {created} created, {updated} updated, "
                f"{deleted} deleted, {chain_errors} chain errors"
            ),
        )

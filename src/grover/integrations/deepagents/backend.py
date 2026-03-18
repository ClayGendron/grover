"""GroverBackend — deepagents BackendProtocol backed by Grover."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
)

if TYPE_CHECKING:
    from grover.client import Grover, GroverAsync
    from grover.models.config import EngineConfig


def _validate_path(path: str) -> str | None:
    """Validate a virtual path. Returns error string or None if valid."""
    if path.startswith("~"):
        return f"Path traversal not allowed: {path}"
    if not path.startswith("/"):
        return f"Path must start with '/': {path}"
    if any(seg == ".." for seg in path.split("/")):
        return f"Path traversal not allowed: {path}"
    return None


def _require_async(backend: GroverBackend) -> GroverAsync:
    """Raise TypeError if backend is not async-capable. Return narrowed GroverAsync."""
    if not backend._is_async:
        raise TypeError("Async methods require GroverAsync. Pass a GroverAsync instance or use sync methods instead.")
    return cast("GroverAsync", backend.grover)


def _format_ls_info_entries(entries: object) -> list[FileInfo]:
    """Convert a FileSearchResult to a list of FileInfo dicts."""
    from grover.models.internal.evidence import ListDirEvidence

    result: list[FileInfo] = []
    for f in entries.files:  # type: ignore[union-attr]
        is_dir = any(isinstance(e, ListDirEvidence) and e.is_directory for e in f.evidence)
        info: FileInfo = {
            "path": f.path,
            "is_dir": is_dir,
        }
        result.append(info)
    return result


def _format_read_result(file_path: str, result: object, offset: int) -> str:
    """Convert a read result to formatted string."""
    if not result.success:  # type: ignore[union-attr]
        return f"Error: {result.message}"  # type: ignore[union-attr]

    content = result.file.content if hasattr(result, "file") and result.file else None  # type: ignore[union-attr]
    if content is None:
        return f"Error: No content for {file_path}"

    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    return format_content_with_line_numbers(content, start_line=offset + 1)


def _format_grep_result(result: object) -> list[GrepMatch]:
    """Convert a FileSearchResult to a list of GrepMatch dicts."""
    from grover.models.internal.evidence import GrepEvidence

    matches: list[GrepMatch] = []
    for f in result.files:  # type: ignore[union-attr]
        for ev in f.evidence:
            if isinstance(ev, GrepEvidence):
                matches.extend(
                    GrepMatch(
                        path=f.path,
                        line=lm.line_number,
                        text=lm.line_content,
                    )
                    for lm in ev.line_matches
                )
    return matches


def _format_glob_info(result: object) -> list[FileInfo]:
    """Convert a FileSearchResult to a list of FileInfo dicts."""
    from grover.models.internal.evidence import GlobEvidence

    infos: list[FileInfo] = []
    for f in result.files:  # type: ignore[union-attr]
        glob_ev = next((e for e in f.evidence if isinstance(e, GlobEvidence)), None)
        info: FileInfo = {
            "path": f.path,
            "is_dir": glob_ev.is_directory if glob_ev else False,
        }
        if glob_ev and glob_ev.size_bytes is not None:
            info["size"] = glob_ev.size_bytes
        infos.append(info)
    return infos


class GroverBackend(BackendProtocol):
    """deepagents ``BackendProtocol`` backed by Grover or GroverAsync.

    Accepts either a sync :class:`~grover.Grover` or async
    :class:`~grover.GroverAsync` instance:

    - **Grover:** sync methods work directly; async methods raise ``TypeError``.
    - **GroverAsync:** async methods call native async API; sync methods wrap
      via ``asyncio.run()`` (cannot be called from a running event loop).

    Usage::

        from grover import Grover, GroverAsync
        from grover.backends.local import LocalFileSystem
        from grover.integrations.deepagents import GroverBackend

        # Sync
        g = Grover()
        g.add_mount("project", LocalFileSystem(workspace_dir="/tmp/ws"))
        backend = GroverBackend(g)

        # Async
        ga = GroverAsync()
        await ga.add_mount("project", LocalFileSystem(workspace_dir="/tmp/ws"))
        backend = GroverBackend(ga)
    """

    def __init__(self, grover: Grover | GroverAsync) -> None:
        from grover.client import GroverAsync

        self.grover = grover
        self._is_async = isinstance(grover, GroverAsync)

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def from_local(
        cls,
        workspace_dir: str,
        *,
        data_dir: str | None = None,
        **mount_kwargs: Any,
    ) -> GroverBackend:
        """Create a GroverBackend with a LocalFileSystem mounted at ``/``."""
        from grover.backends.local import LocalFileSystem
        from grover.client import Grover

        fs_kwargs: dict[str, Any] = {"workspace_dir": workspace_dir}
        if data_dir is not None:
            fs_kwargs["data_dir"] = data_dir
        g = Grover()
        g.add_mount("", filesystem=LocalFileSystem(**fs_kwargs), **mount_kwargs)
        return cls(g)

    @classmethod
    def from_database(
        cls,
        engine_config: EngineConfig,
        **mount_kwargs: Any,
    ) -> GroverBackend:
        """Create a GroverBackend with a DatabaseFileSystem mounted at ``/``."""
        from grover.client import Grover

        g = Grover()
        g.add_mount("", engine_config=engine_config, **mount_kwargs)
        return cls(g)

    @classmethod
    async def from_local_async(
        cls,
        workspace_dir: str,
        *,
        data_dir: str | None = None,
        **mount_kwargs: Any,
    ) -> GroverBackend:
        """Create a GroverBackend with a GroverAsync + LocalFileSystem at ``/``."""
        from grover.backends.local import LocalFileSystem
        from grover.client import GroverAsync

        fs_kwargs: dict[str, Any] = {"workspace_dir": workspace_dir}
        if data_dir is not None:
            fs_kwargs["data_dir"] = data_dir
        g = GroverAsync()
        await g.add_mount("", filesystem=LocalFileSystem(**fs_kwargs), **mount_kwargs)
        return cls(g)

    @classmethod
    async def from_database_async(
        cls,
        engine_config: EngineConfig,
        **mount_kwargs: Any,
    ) -> GroverBackend:
        """Create a GroverBackend with a GroverAsync + DatabaseFileSystem at ``/``."""
        from grover.client import GroverAsync

        g = GroverAsync()
        await g.add_mount("", engine_config=engine_config, **mount_kwargs)
        return cls(g)

    # ------------------------------------------------------------------
    # ls_info
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        err = _validate_path(path)
        if err:
            return []

        if self._is_async:
            return asyncio.run(self.als_info(path))

        g = cast("Grover", self.grover)
        try:
            entries = g.list_dir(path)
        except Exception:
            return []

        return _format_ls_info_entries(entries)

    async def als_info(self, path: str) -> list[FileInfo]:
        g = _require_async(self)

        err = _validate_path(path)
        if err:
            return []

        try:
            entries = await g.list_dir(path)
        except Exception:
            return []

        return _format_ls_info_entries(entries)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        err = _validate_path(file_path)
        if err:
            return f"Error: {err}"

        if self._is_async:
            return asyncio.run(self.aread(file_path, offset, limit))

        g = cast("Grover", self.grover)
        try:
            result = g.read(file_path, offset=offset, limit=limit)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        return _format_read_result(file_path, result, offset)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        g = _require_async(self)

        err = _validate_path(file_path)
        if err:
            return f"Error: {err}"

        try:
            result = await g.read(file_path, offset=offset, limit=limit)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        return _format_read_result(file_path, result, offset)

    # ------------------------------------------------------------------
    # write (create-only)
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        err = _validate_path(file_path)
        if err:
            return WriteResult(error=err)

        if self._is_async:
            return asyncio.run(self.awrite(file_path, content))

        g = cast("Grover", self.grover)
        try:
            result = g.write(file_path, content, overwrite=False)
        except Exception as e:
            return WriteResult(error=f"Write failed: {e}")

        if not result.success:
            return WriteResult(error=result.message)

        return WriteResult(path=file_path, files_update=None)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        g = _require_async(self)

        err = _validate_path(file_path)
        if err:
            return WriteResult(error=err)

        try:
            result = await g.write(file_path, content, overwrite=False)
        except Exception as e:
            return WriteResult(error=f"Write failed: {e}")

        if not result.success:
            return WriteResult(error=result.message)

        return WriteResult(path=file_path, files_update=None)

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        err = _validate_path(file_path)
        if err:
            return EditResult(error=err)

        if self._is_async:
            return asyncio.run(self.aedit(file_path, old_string, new_string, replace_all))

        g = cast("Grover", self.grover)
        # Pre-read to count occurrences for replace_all
        occurrences = 1
        if replace_all:
            try:
                read_result = g.read(file_path)
                rc = read_result.file.content if read_result.file else None
                if read_result.success and rc is not None:
                    occurrences = rc.count(old_string)
            except Exception:
                occurrences = 1

        try:
            result = g.edit(file_path, old_string, new_string, replace_all=replace_all)
        except Exception as e:
            return EditResult(error=f"Edit failed: {e}")

        if not result.success:
            return EditResult(error=result.message)

        return EditResult(path=file_path, files_update=None, occurrences=occurrences)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        g = _require_async(self)

        err = _validate_path(file_path)
        if err:
            return EditResult(error=err)

        # Pre-read to count occurrences for replace_all
        occurrences = 1
        if replace_all:
            try:
                read_result = await g.read(file_path)
                rc = read_result.file.content if read_result.file else None
                if read_result.success and rc is not None:
                    occurrences = rc.count(old_string)
            except Exception:
                occurrences = 1

        try:
            result = await g.edit(file_path, old_string, new_string, replace_all=replace_all)
        except Exception as e:
            return EditResult(error=f"Edit failed: {e}")

        if not result.success:
            return EditResult(error=result.message)

        return EditResult(path=file_path, files_update=None, occurrences=occurrences)

    # ------------------------------------------------------------------
    # grep_raw
    # ------------------------------------------------------------------

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        search_path = path or "/"
        err = _validate_path(search_path)
        if err:
            return f"Error: {err}"

        if self._is_async:
            return asyncio.run(self.agrep_raw(pattern, path, glob))

        g = cast("Grover", self.grover)
        try:
            result = g.grep(
                pattern,
                search_path,
                fixed_string=True,
                glob_filter=glob,
            )
        except Exception as e:
            return f"Error: {e}"

        if not result.success:
            return f"Error: {result.message}"

        return _format_grep_result(result)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        g = _require_async(self)

        search_path = path or "/"
        err = _validate_path(search_path)
        if err:
            return f"Error: {err}"

        try:
            result = await g.grep(
                pattern,
                search_path,
                fixed_string=True,
                glob_filter=glob,
            )
        except Exception as e:
            return f"Error: {e}"

        if not result.success:
            return f"Error: {result.message}"

        return _format_grep_result(result)

    # ------------------------------------------------------------------
    # glob_info
    # ------------------------------------------------------------------

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        err = _validate_path(path)
        if err:
            return []

        if self._is_async:
            return asyncio.run(self.aglob_info(pattern, path))

        g = cast("Grover", self.grover)
        try:
            result = g.glob(pattern, path)
        except Exception:
            return []

        if not result.success:
            return []

        return _format_glob_info(result)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        g = _require_async(self)

        err = _validate_path(path)
        if err:
            return []

        try:
            result = await g.glob(pattern, path)
        except Exception:
            return []

        if not result.success:
            return []

        return _format_glob_info(result)

    # ------------------------------------------------------------------
    # upload_files / download_files
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if self._is_async:
            return asyncio.run(self.aupload_files(files))

        g = cast("Grover", self.grover)
        responses: list[FileUploadResponse] = []
        for file_path, data in files:
            err = _validate_path(file_path)
            if err:
                responses.append(FileUploadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                responses.append(FileUploadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                result = g.write(file_path, content, overwrite=False)
            except Exception:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
                continue

            if not result.success:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
            else:
                responses.append(FileUploadResponse(path=file_path))

        return responses

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        g = _require_async(self)

        responses: list[FileUploadResponse] = []
        for file_path, data in files:
            err = _validate_path(file_path)
            if err:
                responses.append(FileUploadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                responses.append(FileUploadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                result = await g.write(file_path, content, overwrite=False)
            except Exception:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
                continue

            if not result.success:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
            else:
                responses.append(FileUploadResponse(path=file_path))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self._is_async:
            return asyncio.run(self.adownload_files(paths))

        g = cast("Grover", self.grover)
        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            err = _validate_path(file_path)
            if err:
                responses.append(FileDownloadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                result = g.read(file_path)
            except Exception:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
                continue

            rc = result.file.content if hasattr(result, "file") and result.file else None
            if not result.success or rc is None:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
            else:
                responses.append(
                    FileDownloadResponse(
                        path=file_path,
                        content=rc.encode("utf-8"),
                    )
                )

        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        g = _require_async(self)

        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            err = _validate_path(file_path)
            if err:
                responses.append(FileDownloadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                result = await g.read(file_path)
            except Exception:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
                continue

            rc = result.file.content if hasattr(result, "file") and result.file else None
            if not result.success or rc is None:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
            else:
                responses.append(
                    FileDownloadResponse(
                        path=file_path,
                        content=rc.encode("utf-8"),
                    )
                )

        return responses

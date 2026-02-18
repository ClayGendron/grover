"""GroverBackend — deepagents BackendProtocol backed by Grover."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

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
    from grover._grover import Grover


def _validate_path(path: str) -> str | None:
    """Validate a virtual path. Returns error string or None if valid."""
    if path.startswith("~"):
        return f"Path traversal not allowed: {path}"
    if not path.startswith("/"):
        return f"Path must start with '/': {path}"
    if any(seg == ".." for seg in path.split("/")):
        return f"Path traversal not allowed: {path}"
    return None


class GroverBackend(BackendProtocol):
    """deepagents ``BackendProtocol`` backed by a :class:`~grover.Grover` instance.

    Maps deepagents file operations to Grover's sync API. Uses create-only
    write semantics (``overwrite=False``) and returns ``files_update=None``
    (external backend — Grover persists its own state).

    Usage::

        from grover import Grover
        from grover.fs.local_fs import LocalFileSystem
        from grover.integrations.deepagents import GroverBackend

        g = Grover()
        g.mount("/project", LocalFileSystem(workspace_dir="/tmp/ws"))
        backend = GroverBackend(g)
    """

    def __init__(self, grover: Grover) -> None:
        self.grover = grover

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def from_local(cls, workspace_dir: str, **mount_kwargs: Any) -> GroverBackend:
        """Create a GroverBackend with a LocalFileSystem mounted at ``/``."""
        from grover._grover import Grover
        from grover.fs.local_fs import LocalFileSystem

        g = Grover()
        g.mount("/", LocalFileSystem(workspace_dir=workspace_dir), **mount_kwargs)
        return cls(g)

    @classmethod
    def from_database(
        cls,
        engine: Any,
        session_factory: Any = None,
        **mount_kwargs: Any,
    ) -> GroverBackend:
        """Create a GroverBackend with a DatabaseFileSystem mounted at ``/``."""
        from grover._grover import Grover
        from grover.fs.database_fs import DatabaseFileSystem

        g = Grover()
        g.mount(
            "/",
            DatabaseFileSystem(),
            engine=engine,
            session_factory=session_factory,
            **mount_kwargs,
        )
        return cls(g)

    # ------------------------------------------------------------------
    # ls_info
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        err = _validate_path(path)
        if err:
            return []

        try:
            entries = self.grover.list_dir(path)
        except Exception:
            return []

        result: list[FileInfo] = []
        for entry in entries:
            info: FileInfo = {
                "path": entry["path"],
                "is_dir": entry.get("is_directory", False),
            }
            result.append(info)
        return result

    async def als_info(self, path: str) -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

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

        try:
            result = self.grover.read(file_path, offset=offset, limit=limit)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        if not result.success:
            return f"Error: {result.message}"

        content = result.content
        if content is None:
            return f"Error: No content for {file_path}"

        empty_msg = check_empty_content(content)
        if empty_msg:
            return empty_msg

        # Format as cat -n using deepagents' formatter, starting at offset+1
        return format_content_with_line_numbers(content, start_line=offset + 1)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    # ------------------------------------------------------------------
    # write (create-only)
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        err = _validate_path(file_path)
        if err:
            return WriteResult(error=err)

        try:
            result = self.grover.write(file_path, content, overwrite=False)
        except Exception as e:
            return WriteResult(error=f"Write failed: {e}")

        if not result.success:
            return WriteResult(error=result.message)

        return WriteResult(path=file_path, files_update=None)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

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

        # Pre-read to count occurrences for replace_all
        occurrences = 1
        if replace_all:
            try:
                read_result = self.grover.read(file_path)
                if read_result.success and read_result.content is not None:
                    occurrences = read_result.content.count(old_string)
            except Exception:
                occurrences = 1

        try:
            result = self.grover.edit(file_path, old_string, new_string, replace_all=replace_all)
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
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

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

        try:
            result = self.grover.grep(
                pattern,
                search_path,
                fixed_string=True,
                glob_filter=glob,
            )
        except Exception as e:
            return f"Error: {e}"

        if not result.success:
            return f"Error: {result.message}"

        return [
            {
                "path": m.file_path,
                "line": m.line_number,
                "text": m.line_content,
            }
            for m in result.matches
        ]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    # ------------------------------------------------------------------
    # glob_info
    # ------------------------------------------------------------------

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        err = _validate_path(path)
        if err:
            return []

        try:
            result = self.grover.glob(pattern, path)
        except Exception:
            return []

        if not result.success:
            return []

        infos: list[FileInfo] = []
        for entry in result.entries:
            info: FileInfo = {
                "path": entry.path,
                "is_dir": entry.is_directory,
            }
            if entry.size_bytes is not None:
                info["size"] = entry.size_bytes
            infos.append(info)
        return infos

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.glob_info, pattern, path)

    # ------------------------------------------------------------------
    # upload_files / download_files
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
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
                result = self.grover.write(file_path, content, overwrite=False)
            except Exception:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
                continue

            if not result.success:
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
            else:
                responses.append(FileUploadResponse(path=file_path))

        return responses

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            err = _validate_path(file_path)
            if err:
                responses.append(FileDownloadResponse(path=file_path, error="invalid_path"))
                continue

            try:
                result = self.grover.read(file_path)
            except Exception:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
                continue

            if not result.success or result.content is None:
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
            else:
                responses.append(
                    FileDownloadResponse(
                        path=file_path,
                        content=result.content.encode("utf-8"),
                    )
                )

        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)

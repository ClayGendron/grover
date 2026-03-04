"""GroverLoader — LangChain document loader backed by Grover filesystem."""

from __future__ import annotations

import asyncio
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, cast

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from grover.util.content import has_binary_extension

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from grover.client import Grover, GroverAsync


class GroverLoader(BaseLoader):
    """Load documents from a Grover versioned filesystem.

    Accepts either a sync :class:`~grover.Grover` or async
    :class:`~grover.GroverAsync` instance:

    - **Grover:** ``lazy_load()`` works directly; ``alazy_load()`` raises
      ``TypeError``.
    - **GroverAsync:** ``alazy_load()`` calls native async API;
      ``lazy_load()`` wraps via ``asyncio.run()``.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverLoader

        g = Grover()
        g.add_mount("/project", backend)

        # Load everything
        loader = GroverLoader(grover=g, path="/project")
        docs = loader.load()

        # Load only Python files
        loader = GroverLoader(grover=g, path="/project", glob_pattern="*.py")
        docs = loader.load()

    The loader is generator-based (:meth:`lazy_load`), so it streams
    documents without loading them all into memory at once.
    """

    def __init__(
        self,
        grover: Grover | GroverAsync,
        *,
        path: str = "/",
        glob_pattern: str | None = None,
        recursive: bool = True,
    ) -> None:
        from grover.client import GroverAsync

        self.grover = grover
        self.path = path
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self._is_async = isinstance(grover, GroverAsync)

    def lazy_load(self) -> Iterator[Document]:
        """Yield documents from the Grover filesystem.

        Iterates over files under :attr:`path`, reads each one, and yields
        a :class:`Document` with the file content and metadata.

        Binary files and directories are skipped.  If :attr:`glob_pattern`
        is set, only files whose name matches the pattern are included.
        """
        if self._is_async:

            async def _collect() -> list[Document]:
                return [doc async for doc in self.alazy_load()]

            yield from asyncio.run(_collect())
            return

        g = cast("Grover", self.grover)
        entries = self._list_entries()

        for entry in entries:
            if entry.get("is_directory", False):
                continue

            file_path: str = entry.get("path", "")
            if not file_path:
                continue

            # Skip binary files
            if has_binary_extension(file_path):
                continue

            # Apply glob filter on filename
            if self.glob_pattern is not None:
                name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
                if not fnmatch(name, self.glob_pattern):
                    continue

            # Read file content
            read_result = g.read(file_path)
            if not read_result.success or read_result.content is None:
                continue

            yield Document(
                page_content=read_result.content,
                metadata={
                    "path": file_path,
                    "source": file_path,
                    "size_bytes": entry.get("size_bytes"),
                },
                id=file_path,
            )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async variant — native async when GroverAsync, TypeError otherwise."""
        if not self._is_async:
            raise TypeError(
                "Async methods require GroverAsync. "
                "Pass a GroverAsync instance or use sync methods instead."
            )

        g = cast("GroverAsync", self.grover)
        entries = await self._alist_entries()

        for entry in entries:
            if entry.get("is_directory", False):
                continue

            file_path: str = entry.get("path", "")
            if not file_path:
                continue

            if has_binary_extension(file_path):
                continue

            if self.glob_pattern is not None:
                name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
                if not fnmatch(name, self.glob_pattern):
                    continue

            read_result = await g.read(file_path)
            if not read_result.success or read_result.content is None:
                continue

            yield Document(
                page_content=read_result.content,
                metadata={
                    "path": file_path,
                    "source": file_path,
                    "size_bytes": entry.get("size_bytes"),
                },
                id=file_path,
            )

    def _list_entries(self) -> list[dict[str, Any]]:
        """List file entries based on recursive setting."""
        g = cast("Grover", self.grover)
        result = g.tree(self.path, max_depth=None if self.recursive else 1)
        if not result.success:
            return []
        from grover.results import TreeEvidence

        entries = []
        for c in result.candidates:
            is_dir = any(isinstance(e, TreeEvidence) and e.is_directory for e in c.evidence)
            entries.append({"path": c.path, "is_directory": is_dir, "size_bytes": None})
        return entries

    async def _alist_entries(self) -> list[dict[str, Any]]:
        """Async variant of _list_entries."""
        g = cast("GroverAsync", self.grover)
        result = await g.tree(self.path, max_depth=None if self.recursive else 1)
        if not result.success:
            return []
        from grover.results import TreeEvidence

        entries = []
        for c in result.candidates:
            is_dir = any(isinstance(e, TreeEvidence) and e.is_directory for e in c.evidence)
            entries.append({"path": c.path, "is_directory": is_dir, "size_bytes": None})
        return entries

"""GroverLoader — LangChain document loader backed by Grover filesystem."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from grover.fs.utils import has_binary_extension

if TYPE_CHECKING:
    from collections.abc import Iterator

    from grover._grover import Grover


class GroverLoader(BaseLoader):
    """Load documents from a Grover versioned filesystem.

    Walks a Grover directory tree and yields each text file as a LangChain
    :class:`~langchain_core.documents.Document`.  Binary files are
    automatically skipped.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverLoader

        g = Grover()
        g.mount("/project", backend)

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
        grover: Grover,
        *,
        path: str = "/",
        glob_pattern: str | None = None,
        recursive: bool = True,
    ) -> None:
        self.grover = grover
        self.path = path
        self.glob_pattern = glob_pattern
        self.recursive = recursive

    def lazy_load(self) -> Iterator[Document]:
        """Yield documents from the Grover filesystem.

        Iterates over files under :attr:`path`, reads each one, and yields
        a :class:`Document` with the file content and metadata.

        Binary files and directories are skipped.  If :attr:`glob_pattern`
        is set, only files whose name matches the pattern are included.
        """
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
            read_result = self.grover.read(file_path)
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
        if self.recursive:
            result = self.grover.tree(self.path)
            if not result.success:
                return []
            return [
                {
                    "path": e.path,
                    "is_directory": e.is_directory,
                    "size_bytes": e.size_bytes,
                }
                for e in result.entries
            ]
        else:
            raw = self.grover.list_dir(self.path)
            return [
                {
                    "path": item.get("path", ""),
                    "is_directory": item.get("is_directory", False),
                    "size_bytes": item.get("size_bytes"),
                }
                for item in raw
            ]

"""Internal types — Ref, File, Directory, FileChunk, FileVersion, FileConnection.

``Ref`` is the immutable identity type for any Grover entity.  It parses
synthetic path formats and can ``transform()`` into the appropriate
runtime type (``File``, ``Directory``, or ``FileConnection``).

Chunks and versions are attributes of ``File``, not standalone entities.
A chunk or version path transforms into a ``File`` with the entity nested
in its ``chunks`` or ``versions`` list.

DB models live in ``grover.models.database`` and use the ``Model`` suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from grover.models.internal.detail import Detail
    from grover.models.internal.evidence import Evidence


@dataclass(frozen=True, slots=True)
class Ref:
    """Immutable identity for any Grover entity.

    All four synthetic path formats are supported:

    - File:       ``/src/auth.py``
    - Chunk:      ``/src/auth.py#login``
    - Version:    ``/src/auth.py@3``
    - Connection: ``/src/auth.py[imports]/src/utils.py``

    Use the factory classmethods to build paths; use the properties to
    decompose them; use ``transform()`` to get the appropriate runtime type.
    """

    path: str

    def __repr__(self) -> str:
        return f"Ref({self.path!r})"

    # ------------------------------------------------------------------
    # Type checks (mutually exclusive: connection > chunk > version > file)
    # ------------------------------------------------------------------

    @property
    def is_connection(self) -> bool:
        """Return ``True`` if this path encodes a connection (``source[type]target``)."""
        bracket_open = self.path.rfind("[")
        if bracket_open <= 0:
            return False
        bracket_close = self.path.find("]", bracket_open + 1)
        if bracket_close <= bracket_open + 1:
            return False
        conn_type = self.path[bracket_open + 1 : bracket_close]
        return bool(conn_type) and "/" not in conn_type

    @property
    def is_chunk(self) -> bool:
        """Return ``True`` if this path contains a ``#symbol`` suffix."""
        if self.is_connection:
            return False
        hash_idx = self.path.rfind("#")
        if hash_idx <= 0:
            return False
        suffix = self.path[hash_idx + 1 :]
        return bool(suffix) and "/" not in suffix

    @property
    def is_version(self) -> bool:
        """Return ``True`` if this path contains an ``@N`` version suffix."""
        if self.is_chunk or self.is_connection:
            return False
        at_idx = self.path.rfind("@")
        if at_idx <= 0:
            return False
        try:
            int(self.path[at_idx + 1 :])
            return True
        except ValueError:
            return False

    @property
    def is_file(self) -> bool:
        """Return ``True`` if this is a plain file path (no suffix)."""
        return not (self.is_connection or self.is_chunk or self.is_version)

    # ------------------------------------------------------------------
    # Decomposition — file / chunk / version
    # ------------------------------------------------------------------

    @property
    def base_path(self) -> str:
        """The base file path.

        Strips ``#chunk`` or ``@version`` suffixes.  For connections,
        returns the source path.  For plain files, returns the path
        unchanged.
        """
        if self.is_connection:
            return self.path[: self.path.rfind("[")]
        if self.is_chunk:
            return self.path[: self.path.rfind("#")]
        if self.is_version:
            return self.path[: self.path.rfind("@")]
        return self.path

    @property
    def chunk(self) -> str | None:
        """The symbol name for chunk refs, otherwise ``None``."""
        if not self.is_chunk:
            return None
        return self.path[self.path.rfind("#") + 1 :]

    @property
    def version(self) -> int | None:
        """The version number for version refs, otherwise ``None``."""
        if not self.is_version:
            return None
        return int(self.path[self.path.rfind("@") + 1 :])

    # ------------------------------------------------------------------
    # Decomposition — connection
    # ------------------------------------------------------------------

    @property
    def source(self) -> str | None:
        """The source path for connection refs, otherwise ``None``."""
        if not self.is_connection:
            return None
        return self.path[: self.path.rfind("[")]

    @property
    def target(self) -> str | None:
        """The target path for connection refs, otherwise ``None``."""
        if not self.is_connection:
            return None
        bracket_open = self.path.rfind("[")
        bracket_close = self.path.find("]", bracket_open + 1)
        return self.path[bracket_close + 1 :]

    @property
    def connection_type(self) -> str | None:
        """The edge type for connection refs, otherwise ``None``."""
        if not self.is_connection:
            return None
        bracket_open = self.path.rfind("[")
        bracket_close = self.path.find("]", bracket_open + 1)
        return self.path[bracket_open + 1 : bracket_close]

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def for_chunk(cls, file_path: str, symbol: str) -> Ref:
        """Create a chunk Ref: ``file_path#symbol``."""
        return cls(path=f"{file_path}#{symbol}")

    @classmethod
    def for_version(cls, file_path: str, version: int) -> Ref:
        """Create a version Ref: ``file_path@version``."""
        return cls(path=f"{file_path}@{version}")

    @classmethod
    def for_connection(cls, source: str, target: str, connection_type: str) -> Ref:
        """Create a connection Ref: ``source[connection_type]target``."""
        return cls(path=f"{source}[{connection_type}]{target}")

    # ------------------------------------------------------------------
    # Transform — return the appropriate runtime type
    # ------------------------------------------------------------------

    def transform(self) -> File | Directory | FileConnection:
        """Return the appropriate runtime type for this path.

        - Connection paths → ``FileConnection``
        - Chunk paths → ``File`` with the chunk in ``chunks``
        - Version paths → ``File`` with the version in ``versions``
        - Text file paths → ``File``
        - Everything else → ``Directory``

        Uses ``is_text_file`` to distinguish files from directories
        for plain paths: extensionless names that aren't in
        ``TEXT_FILENAMES`` (Makefile, Dockerfile, etc.) are directories.
        """
        from grover.util.content import is_text_file
        from grover.util.paths import split_path

        if self.is_connection:
            return FileConnection(
                path=self.path,
                source_path=self.source or "",
                target_path=self.target or "",
                type=self.connection_type or "",
            )
        if self.is_chunk:
            return File(
                path=self.base_path,
                chunks=[FileChunk(path=self.path, name=self.chunk or "")],
            )
        if self.is_version:
            return File(
                path=self.base_path,
                versions=[FileVersion(path=self.path, number=self.version or 0)],
            )
        _, name = split_path(self.path)
        if name and is_text_file(name):
            return File(path=self.path)
        return Directory(path=self.path)


# =====================================================================
# Runtime types
# =====================================================================


@dataclass(slots=True)
class FileChunk:
    """A chunk (function, class, section) within a file."""

    path: str
    name: str = ""
    content: str = ""
    embedding: list[float] | None = None
    tokens: int = 0
    line_start: int = 0
    line_end: int = 0
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def details(self) -> list[Detail]:
        """Alias for ``evidence`` — migration bridge to Detail naming."""
        return self.evidence

    @details.setter
    def details(self, value: list[Detail]) -> None:
        self.evidence = value


@dataclass(slots=True)
class FileVersion:
    """A historical version of a file."""

    path: str
    number: int = 0
    embedding: list[float] | None = None
    evidence: list[Evidence] = field(default_factory=list)
    created_at: datetime | None = None

    @property
    def details(self) -> list[Detail]:
        """Alias for ``evidence`` — migration bridge to Detail naming."""
        return self.evidence

    @details.setter
    def details(self, value: list[Detail]) -> None:
        self.evidence = value


@dataclass(slots=True)
class File:
    """A file with optional hydrated content, chunks, and versions."""

    path: str
    content: str | None = None
    embedding: list[float] | None = None
    tokens: int = 0
    lines: int = 0
    size_bytes: int = 0
    mime_type: str = ""
    current_version: int = 0
    chunks: list[FileChunk] = field(default_factory=list)
    versions: list[FileVersion] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def details(self) -> list[Detail]:
        """Alias for ``evidence`` — migration bridge to Detail naming."""
        return self.evidence

    @details.setter
    def details(self, value: list[Detail]) -> None:
        self.evidence = value


@dataclass(slots=True)
class Directory:
    """A directory entry — distinct from ``File`` for type clarity.

    Directories cannot be distinguished from files by path alone.
    Use ``Directory`` when the caller knows the entity is a directory.
    """

    path: str
    details: list[Detail] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class FileConnection:
    """A directed edge between two entities."""

    path: str
    source_path: str
    target_path: str
    type: str
    weight: float = 1.0
    distance: float = 1.0
    evidence: list[Evidence] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def details(self) -> list[Detail]:
        """Alias for ``evidence`` — migration bridge to Detail naming."""
        return self.evidence

    @details.setter
    def details(self, value: list[Detail]) -> None:
        self.evidence = value

"""Internal types — Ref, File, Directory, FileChunk, FileVersion, FileConnection.

These are the runtime data types for Grover's internal API. They are
dataclasses and represent files hierarchically: chunks and versions
are attributes of files, not fake files.

DB models live in ``grover.models.database`` and use the ``Model`` suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from grover.models.internal.detail import Detail
    from grover.models.internal.evidence import Evidence


@dataclass(slots=True)
class Ref:
    """Identity for any addressable entity in Grover."""

    path: str


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

    Directories get their own type instead of being ``File(is_directory=True)``.
    During migration both representations coexist.
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

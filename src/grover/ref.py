"""Ref — immutable identity type for Grover entities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ref:
    """Immutable identity for any Grover entity.

    A thin wrapper around a single path string.  All four synthetic path
    formats are supported:

    - File:       ``/src/auth.py``
    - Chunk:      ``/src/auth.py#login``
    - Version:    ``/src/auth.py@3``
    - Connection: ``/src/auth.py[imports]/src/utils.py``

    Use the factory classmethods to build paths; use the properties to
    decompose them.
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

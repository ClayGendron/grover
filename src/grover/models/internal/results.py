"""New internal result types — FileOperationResult and FileSearchResult.

These are Pydantic BaseModels that collapse the 30+ old result subclasses
into two families.  Operations differentiate via Evidence types on Files,
not separate result classes.
"""

import copy
from collections.abc import Callable, Iterator
from typing import Self

from pydantic import BaseModel

from grover.models.internal.evidence import Evidence
from grover.models.internal.ref import File, FileConnection, Ref


class FileOperationResult(BaseModel):
    """Result of a single-file operation (read, write, delete, etc.)."""

    file: File = File(path="")
    message: str = ""
    success: bool = True


class FileSearchResult(BaseModel):
    """Result of a multi-file query (search, graph, glob, etc.).

    Supports set algebra:

    - ``&`` (intersection) — paths in both, merges evidence
    - ``|`` (union) — paths from either, merges evidence
    - ``-`` (difference) — paths in LHS not in RHS
    - ``>>`` (pipeline) — passes LHS paths as candidates to RHS
    """

    files: list[File] = []
    connections: list[FileConnection] = []
    message: str = ""
    success: bool = True

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file paths in this result."""
        return tuple(f.path for f in self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __bool__(self) -> bool:
        return self.success and len(self.files) > 0

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(f.path for f in self.files)

    def __contains__(self, path: object) -> bool:
        return any(f.path == path for f in self.files)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _as_dict(self) -> dict[str, File]:
        """Convert files to dict keyed by path."""
        return {f.path: f for f in self.files}

    @staticmethod
    def _merge_files(f1: File, f2: File) -> File:
        """Merge two files with the same path, combining evidence."""
        return File(
            path=f1.path,
            is_directory=f1.is_directory or f2.is_directory,
            content=f1.content if f1.content is not None else f2.content,
            embedding=f1.embedding if f1.embedding is not None else f2.embedding,
            tokens=max(f1.tokens, f2.tokens),
            lines=max(f1.lines, f2.lines),
            current_version=max(f1.current_version, f2.current_version),
            chunks=f1.chunks or f2.chunks,
            versions=f1.versions or f2.versions,
            evidence=list(f1.evidence) + list(f2.evidence),
            created_at=f1.created_at or f2.created_at,
            updated_at=f1.updated_at or f2.updated_at,
        )

    def _connections_as_dict(self) -> dict[str, FileConnection]:
        """Convert connections to dict keyed by source[type]target."""
        result: dict[str, FileConnection] = {}
        for c in self.connections:
            key = f"{c.source.path}[{c.type}]{c.target.path}"
            result[key] = c
        return result

    @staticmethod
    def _merge_connections(c1: FileConnection, c2: FileConnection) -> FileConnection:
        """Merge two connections, combining evidence."""
        return FileConnection(
            source=c1.source,
            target=c1.target,
            type=c1.type,
            weight=c1.weight,
            distance=c1.distance,
            evidence=list(c1.evidence) + list(c2.evidence),
            created_at=c1.created_at or c2.created_at,
            updated_at=c1.updated_at or c2.updated_at,
        )

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str], *, operation: str = "unknown") -> Self:
        """Create a result from a list of paths with default evidence."""
        files = [
            File(
                path=p,
                evidence=[Evidence(operation=operation)],
            )
            for p in paths
        ]
        return cls(
            success=True,
            message=f"{len(paths)} paths",
            files=files,
        )

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> Self:
        """Return a new result with all paths prefixed by *prefix*."""
        result = copy.copy(self)
        result.files = [
            File(
                path=(prefix + f.path if f.path != "/" else prefix),
                is_directory=f.is_directory,
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                current_version=f.current_version,
                chunks=f.chunks,
                versions=f.versions,
                evidence=list(f.evidence),
                created_at=f.created_at,
                updated_at=f.updated_at,
            )
            for f in self.files
        ]
        result.connections = [
            FileConnection(
                source=Ref(path=(prefix + c.source.path if c.source.path != "/" else prefix)),
                target=Ref(path=(prefix + c.target.path if c.target.path != "/" else prefix)),
                type=c.type,
                weight=c.weight,
                distance=c.distance,
                evidence=list(c.evidence),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in self.connections
        ]
        return result

    def remap_paths(self, fn: Callable[[str], str]) -> Self:
        """Return a new result with all paths transformed by *fn*.

        If two files map to the same new path, their evidence is merged.
        """
        result = copy.copy(self)
        merged: dict[str, File] = {}
        for f in self.files:
            new_path = fn(f.path)
            new_file = File(
                path=new_path,
                is_directory=f.is_directory,
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                current_version=f.current_version,
                chunks=f.chunks,
                versions=f.versions,
                evidence=list(f.evidence),
                created_at=f.created_at,
                updated_at=f.updated_at,
            )
            if new_path in merged:
                merged[new_path] = self._merge_files(merged[new_path], new_file)
            else:
                merged[new_path] = new_file
        result.files = list(merged.values())
        result.connections = [
            FileConnection(
                source=Ref(path=fn(c.source.path)),
                target=Ref(path=fn(c.target.path)),
                type=c.type,
                weight=c.weight,
                distance=c.distance,
                evidence=list(c.evidence),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in self.connections
        ]
        return result

    # -----------------------------------------------------------------
    # Set algebra
    # -----------------------------------------------------------------

    def __and__(self, other: object) -> Self:
        """Intersection — paths in both, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        common = set(d1) & set(d2)
        files = [self._merge_files(d1[p], d2[p]) for p in common]
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_common = set(cd1) & set(cd2)
        conns = [self._merge_connections(cd1[p], cd2[p]) for p in conn_common]
        return type(self)(
            success=self.success and other.success,
            message=f"{len(files)} paths",
            files=files,
            connections=conns,
        )

    def __or__(self, other: object) -> Self:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        all_paths = set(d1) | set(d2)
        files: list[File] = []
        for p in all_paths:
            if p in d1 and p in d2:
                files.append(self._merge_files(d1[p], d2[p]))
            elif p in d1:
                files.append(d1[p])
            else:
                files.append(d2[p])
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_all = set(cd1) | set(cd2)
        conns: list[FileConnection] = []
        for cp in conn_all:
            if cp in cd1 and cp in cd2:
                conns.append(self._merge_connections(cd1[cp], cd2[cp]))
            elif cp in cd1:
                conns.append(cd1[cp])
            else:
                conns.append(cd2[cp])
        return type(self)(
            success=self.success or other.success,
            message=f"{len(files)} paths",
            files=files,
            connections=conns,
        )

    def __sub__(self, other: object) -> Self:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        diff = set(d1) - set(d2)
        files = [d1[p] for p in diff]
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_diff = set(cd1) - set(cd2)
        conns = [cd1[p] for p in conn_diff]
        return type(self)(
            success=self.success,
            message=f"{len(files)} paths",
            files=files,
            connections=conns,
        )

    def __rshift__(self, other: object) -> Self:
        """Pipeline — passes LHS paths as candidates to RHS."""
        if not isinstance(other, FileSearchResult):
            return NotImplemented
        d1 = self._as_dict()
        d2 = other._as_dict()
        common = set(d1) & set(d2)
        files = [self._merge_files(d1[p], d2[p]) for p in common]
        # Connection algebra
        cd1 = self._connections_as_dict()
        cd2 = other._connections_as_dict()
        conn_common = set(cd1) & set(cd2)
        conns = [self._merge_connections(cd1[p], cd2[p]) for p in conn_common]
        return type(self)(
            success=self.success and other.success,
            message=f"{len(files)} paths",
            files=files,
            connections=conns,
        )

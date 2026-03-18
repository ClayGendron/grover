"""Internal result types — GroverResult, FileSearchSet, FileSearchResult, FileOperationResult.

``GroverResult`` is the unified result type that replaces the four older
types (``FileOperationResult``, ``BatchResult``, ``FileSearchSet``,
``FileSearchResult``).  It works as both output and candidate input, and
carries per-file outcome detail via ``Detail`` objects on each ``File``.

The older types remain for backward compatibility during incremental
migration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from grover.models.internal.detail import Detail
from grover.models.internal.evidence import Evidence
from grover.models.internal.ref import Directory, File, FileConnection, Ref

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(slots=True)
class FileOperationResult:
    """Result of a single-file operation (read, write, delete, etc.)."""

    file: File = field(default_factory=lambda: File(path=""))
    message: str = ""
    success: bool = True


@dataclass(slots=True)
class BatchResult:
    """Result of a batch operation (multiple file or chunk writes)."""

    results: list[FileOperationResult] = field(default_factory=list)
    message: str = ""
    success: bool = True
    succeeded: int = 0
    failed: int = 0


@dataclass(slots=True)
class FileSearchSet:
    """An unordered set of file candidates and connections.

    Supports set algebra (``&``, ``|``, ``-``, ``>>``), iteration over
    file paths, and path transformations (``rebase``, ``remap_paths``).

    Unlike ``FileSearchResult`` this type carries **no** ``success`` /
    ``message`` fields — it is a pure candidate container suitable for
    passing into search methods as input.
    """

    files: list[File] = field(default_factory=list)
    connections: list[FileConnection] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str]) -> Self:
        """Create a set from a list of paths (no evidence — added by consumers)."""
        return cls(files=[File(path=p) for p in paths])

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file paths in this set."""
        return tuple(f.path for f in self.files)

    @property
    def connection_paths(self) -> tuple[str, ...]:
        """All connection ref-format paths (``source[type]target``)."""
        return tuple(f"{c.source_path}[{c.type}]{c.target_path}" for c in self.connections)

    def __len__(self) -> int:
        return len(self.files)

    def __bool__(self) -> bool:
        return len(self.files) > 0

    def __iter__(self) -> Iterator[str]:
        return iter(f.path for f in self.files)

    def __contains__(self, path: object) -> bool:
        return any(f.path == path for f in self.files)

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------

    def explain(self, path: str) -> list[Evidence]:
        """Return the evidence chain for *path*, or ``[]`` if absent."""
        for f in self.files:
            if f.path == path:
                return list(f.evidence)
        return []

    def to_refs(self) -> list[Ref]:
        """Convert file paths to a list of ``Ref`` objects."""
        return [Ref(path=f.path) for f in self.files]

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
            content=f1.content if f1.content is not None else f2.content,
            embedding=f1.embedding if f1.embedding is not None else f2.embedding,
            tokens=max(f1.tokens, f2.tokens),
            lines=max(f1.lines, f2.lines),
            size_bytes=max(f1.size_bytes, f2.size_bytes),
            mime_type=f1.mime_type or f2.mime_type,
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
            key = f"{c.source_path}[{c.type}]{c.target_path}"
            result[key] = c
        return result

    @staticmethod
    def _merge_connections(c1: FileConnection, c2: FileConnection) -> FileConnection:
        """Merge two connections, combining evidence."""
        return FileConnection(
            path=c1.path,
            source_path=c1.source_path,
            target_path=c1.target_path,
            type=c1.type,
            weight=c1.weight,
            distance=c1.distance,
            evidence=list(c1.evidence) + list(c2.evidence),
            created_at=c1.created_at or c2.created_at,
            updated_at=c1.updated_at or c2.updated_at,
        )

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> Self:
        """Return a new set with all paths prefixed by *prefix*."""
        result = copy.copy(self)
        result.files = [
            File(
                path=(prefix + f.path if f.path != "/" else prefix),
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                size_bytes=f.size_bytes,
                mime_type=f.mime_type,
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
                path=c.path,
                source_path=(prefix + c.source_path if c.source_path != "/" else prefix),
                target_path=(prefix + c.target_path if c.target_path != "/" else prefix),
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
        """Return a new set with all paths transformed by *fn*.

        If two files map to the same new path, their evidence is merged.
        """
        result = copy.copy(self)
        merged: dict[str, File] = {}
        for f in self.files:
            new_path = fn(f.path)
            new_file = File(
                path=new_path,
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                size_bytes=f.size_bytes,
                mime_type=f.mime_type,
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
                path=c.path,
                source_path=fn(c.source_path),
                target_path=fn(c.target_path),
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
        if not isinstance(other, FileSearchSet):
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
            files=files,
            connections=conns,
        )

    def __or__(self, other: object) -> Self:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchSet):
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
            files=files,
            connections=conns,
        )

    def __sub__(self, other: object) -> Self:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchSet):
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
            files=files,
            connections=conns,
        )

    def __rshift__(self, other: object) -> Self:
        """Pipeline — passes LHS paths as candidates to RHS."""
        if not isinstance(other, FileSearchSet):
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
            files=files,
            connections=conns,
        )


@dataclass(slots=True)
class FileSearchResult(FileSearchSet):
    """Result of a multi-file query (search, graph, glob, etc.).

    Inherits candidate storage and set algebra from ``FileSearchSet``
    and adds ``success`` / ``message`` fields plus factory methods.
    """

    message: str = ""
    success: bool = True

    def __bool__(self) -> bool:
        return self.success and len(self.files) > 0

    # -----------------------------------------------------------------
    # Set algebra overrides (propagate success/message)
    # -----------------------------------------------------------------

    def __and__(self, other: object) -> Self:
        """Intersection — paths in both, evidence merged."""
        if not isinstance(other, FileSearchSet):
            return NotImplemented
        base = FileSearchSet.__and__(self, other)
        other_success = other.success if isinstance(other, FileSearchResult) else True
        return type(self)(
            success=self.success and other_success,
            message=f"{len(base.files)} paths",
            files=base.files,
            connections=base.connections,
        )

    def __or__(self, other: object) -> Self:
        """Union — paths from either, evidence merged."""
        if not isinstance(other, FileSearchSet):
            return NotImplemented
        base = FileSearchSet.__or__(self, other)
        other_success = other.success if isinstance(other, FileSearchResult) else True
        return type(self)(
            success=self.success or other_success,
            message=f"{len(base.files)} paths",
            files=base.files,
            connections=base.connections,
        )

    def __sub__(self, other: object) -> Self:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, FileSearchSet):
            return NotImplemented
        base = FileSearchSet.__sub__(self, other)
        return type(self)(
            success=self.success,
            message=f"{len(base.files)} paths",
            files=base.files,
            connections=base.connections,
        )

    def __rshift__(self, other: object) -> Self:
        """Pipeline — passes LHS paths as candidates to RHS."""
        if not isinstance(other, FileSearchSet):
            return NotImplemented
        base = FileSearchSet.__rshift__(self, other)
        other_success = other.success if isinstance(other, FileSearchResult) else True
        return type(self)(
            success=self.success and other_success,
            message=f"{len(base.files)} paths",
            files=base.files,
            connections=base.connections,
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

    @classmethod
    def from_refs(cls, refs: list[Ref], *, operation: str = "unknown") -> Self:
        """Create a result from a list of ``Ref`` objects."""
        files = [
            File(
                path=ref.path,
                evidence=[Evidence(operation=operation)],
            )
            for ref in refs
        ]
        return cls(
            success=True,
            message=f"{len(refs)} refs",
            files=files,
        )


# =====================================================================
# GroverResult — unified result type
# =====================================================================


@dataclass(slots=True)
class GroverResult:
    """Unified result type for all Grover operations.

    Replaces ``FileOperationResult``, ``BatchResult``, ``FileSearchSet``,
    and ``FileSearchResult``.  Works as both output **and** candidate input
    (enabling chaining like ``await g.glob("*.py") & await g.grep("import")``).

    Per-file outcomes live in ``file.details`` (a list of ``Detail`` objects).
    The top-level ``success`` / ``message`` summarize the overall operation.
    """

    files: list[File] = field(default_factory=list)
    directories: list[Directory] = field(default_factory=list)
    connections: list[FileConnection] = field(default_factory=list)
    message: str = ""
    success: bool = True

    # -----------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------

    @property
    def file(self) -> File:
        """First file — for single-file operations."""
        return self.files[0] if self.files else File(path="")

    @property
    def content(self) -> str | None:
        """First file's content."""
        return self.file.content

    @property
    def succeeded(self) -> int:
        """Count of entities (files, directories, connections) where all details report success."""
        count = sum(1 for f in self.files if all(d.success for d in f.details))
        count += sum(1 for d in self.directories if all(dt.success for dt in d.details))
        count += sum(1 for c in self.connections if all(d.success for d in c.details))
        return count

    @property
    def failed(self) -> int:
        """Count of entities where any detail reports failure."""
        total = len(self.files) + len(self.directories) + len(self.connections)
        return total - self.succeeded

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[str], *, operation: str = "unknown") -> GroverResult:
        """Create a result from a list of paths with default detail."""
        files = [File(path=p, evidence=[Detail(operation=operation)]) for p in paths]
        return cls(success=True, message=f"{len(paths)} paths", files=files)

    @classmethod
    def from_refs(cls, refs: list[Ref], *, operation: str = "unknown") -> GroverResult:
        """Create a result from a list of ``Ref`` objects."""
        files = [File(path=ref.path, evidence=[Detail(operation=operation)]) for ref in refs]
        return cls(success=True, message=f"{len(refs)} refs", files=files)

    # -----------------------------------------------------------------
    # Properties and iteration
    # -----------------------------------------------------------------

    @property
    def paths(self) -> tuple[str, ...]:
        """All file and directory paths."""
        return tuple(f.path for f in self.files) + tuple(d.path for d in self.directories)

    @property
    def connection_paths(self) -> tuple[str, ...]:
        """All connection ref-format paths (``source[type]target``)."""
        return tuple(f"{c.source_path}[{c.type}]{c.target_path}" for c in self.connections)

    def __len__(self) -> int:
        return len(self.files)

    def __bool__(self) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(f.path for f in self.files)

    def __contains__(self, path: object) -> bool:
        return any(f.path == path for f in self.files)

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------

    def explain(self, path: str) -> list[Detail]:
        """Return the detail chain for *path*, or ``[]`` if absent."""
        for f in self.files:
            if f.path == path:
                return list(f.details)
        return []

    def to_refs(self) -> list[Ref]:
        """Convert file paths to a list of ``Ref`` objects."""
        return [Ref(path=f.path) for f in self.files]

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _as_dict(self) -> dict[str, File]:
        return {f.path: f for f in self.files}

    def _dirs_as_dict(self) -> dict[str, Directory]:
        return {d.path: d for d in self.directories}

    def _connections_as_dict(self) -> dict[str, FileConnection]:
        result: dict[str, FileConnection] = {}
        for c in self.connections:
            key = f"{c.source_path}[{c.type}]{c.target_path}"
            result[key] = c
        return result

    @staticmethod
    def _merge_files(f1: File, f2: File) -> File:
        return File(
            path=f1.path,
            content=f1.content if f1.content is not None else f2.content,
            embedding=f1.embedding if f1.embedding is not None else f2.embedding,
            tokens=max(f1.tokens, f2.tokens),
            lines=max(f1.lines, f2.lines),
            size_bytes=max(f1.size_bytes, f2.size_bytes),
            mime_type=f1.mime_type or f2.mime_type,
            current_version=max(f1.current_version, f2.current_version),
            chunks=f1.chunks or f2.chunks,
            versions=f1.versions or f2.versions,
            evidence=list(f1.evidence) + list(f2.evidence),
            created_at=f1.created_at or f2.created_at,
            updated_at=f1.updated_at or f2.updated_at,
        )

    @staticmethod
    def _merge_dirs(d1: Directory, d2: Directory) -> Directory:
        return Directory(
            path=d1.path,
            details=list(d1.details) + list(d2.details),
            created_at=d1.created_at or d2.created_at,
            updated_at=d1.updated_at or d2.updated_at,
        )

    @staticmethod
    def _merge_connections(c1: FileConnection, c2: FileConnection) -> FileConnection:
        return FileConnection(
            path=c1.path,
            source_path=c1.source_path,
            target_path=c1.target_path,
            type=c1.type,
            weight=c1.weight,
            distance=c1.distance,
            evidence=list(c1.evidence) + list(c2.evidence),
            created_at=c1.created_at or c2.created_at,
            updated_at=c1.updated_at or c2.updated_at,
        )

    # -----------------------------------------------------------------
    # Path transformations
    # -----------------------------------------------------------------

    def rebase(self, prefix: str) -> GroverResult:
        """Return a new result with all paths prefixed by *prefix*."""

        def _rebase_path(p: str) -> str:
            return prefix + p if p != "/" else prefix

        files = [
            File(
                path=_rebase_path(f.path),
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                size_bytes=f.size_bytes,
                mime_type=f.mime_type,
                current_version=f.current_version,
                chunks=f.chunks,
                versions=f.versions,
                evidence=list(f.evidence),
                created_at=f.created_at,
                updated_at=f.updated_at,
            )
            for f in self.files
        ]
        directories = [
            Directory(
                path=_rebase_path(d.path),
                details=list(d.details),
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in self.directories
        ]
        connections = [
            FileConnection(
                path=c.path,
                source_path=_rebase_path(c.source_path),
                target_path=_rebase_path(c.target_path),
                type=c.type,
                weight=c.weight,
                distance=c.distance,
                evidence=list(c.evidence),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in self.connections
        ]
        return GroverResult(
            files=files,
            directories=directories,
            connections=connections,
            message=self.message,
            success=self.success,
        )

    def remap_paths(self, fn: Callable[[str], str]) -> GroverResult:
        """Return a new result with all paths transformed by *fn*."""
        merged: dict[str, File] = {}
        for f in self.files:
            new_path = fn(f.path)
            new_file = File(
                path=new_path,
                content=f.content,
                embedding=f.embedding,
                tokens=f.tokens,
                lines=f.lines,
                size_bytes=f.size_bytes,
                mime_type=f.mime_type,
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

        merged_dirs: dict[str, Directory] = {}
        for d in self.directories:
            new_path = fn(d.path)
            new_dir = Directory(
                path=new_path,
                details=list(d.details),
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            if new_path in merged_dirs:
                merged_dirs[new_path] = self._merge_dirs(merged_dirs[new_path], new_dir)
            else:
                merged_dirs[new_path] = new_dir

        connections = [
            FileConnection(
                path=c.path,
                source_path=fn(c.source_path),
                target_path=fn(c.target_path),
                type=c.type,
                weight=c.weight,
                distance=c.distance,
                evidence=list(c.evidence),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in self.connections
        ]
        return GroverResult(
            files=list(merged.values()),
            directories=list(merged_dirs.values()),
            connections=connections,
            message=self.message,
            success=self.success,
        )

    # -----------------------------------------------------------------
    # Set algebra
    # -----------------------------------------------------------------

    def _algebra(
        self,
        other: GroverResult,
        file_op: Callable[[dict[str, File], dict[str, File]], list[File]],
        dir_op: Callable[[dict[str, Directory], dict[str, Directory]], list[Directory]],
        conn_op: Callable[
            [dict[str, FileConnection], dict[str, FileConnection]],
            list[FileConnection],
        ],
        success_op: Callable[[bool, bool], bool],
    ) -> GroverResult:
        """Shared implementation for set algebra operators."""
        return GroverResult(
            files=file_op(self._as_dict(), other._as_dict()),
            directories=dir_op(self._dirs_as_dict(), other._dirs_as_dict()),
            connections=conn_op(self._connections_as_dict(), other._connections_as_dict()),
            success=success_op(self.success, other.success),
            message="",
        )

    def __and__(self, other: object) -> GroverResult:
        """Intersection — paths in both, details merged."""
        if not isinstance(other, GroverResult):
            return NotImplemented
        return self._algebra(
            other,
            lambda d1, d2: [self._merge_files(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda d1, d2: [self._merge_dirs(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda d1, d2: [self._merge_connections(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda a, b: a and b,
        )

    def __or__(self, other: object) -> GroverResult:
        """Union — paths from either, details merged where overlapping."""
        if not isinstance(other, GroverResult):
            return NotImplemented

        def _union_files(d1: dict[str, File], d2: dict[str, File]) -> list[File]:
            files: list[File] = []
            seen: set[str] = set()
            for p, f in d1.items():
                files.append(self._merge_files(f, d2[p]) if p in d2 else f)
                seen.add(p)
            for p, f in d2.items():
                if p not in seen:
                    files.append(f)
            return files

        def _union_dirs(d1: dict[str, Directory], d2: dict[str, Directory]) -> list[Directory]:
            dirs: list[Directory] = []
            seen: set[str] = set()
            for p, d in d1.items():
                dirs.append(self._merge_dirs(d, d2[p]) if p in d2 else d)
                seen.add(p)
            for p, d in d2.items():
                if p not in seen:
                    dirs.append(d)
            return dirs

        def _union_conns(d1: dict[str, FileConnection], d2: dict[str, FileConnection]) -> list[FileConnection]:
            conns: list[FileConnection] = []
            seen: set[str] = set()
            for p, c in d1.items():
                conns.append(self._merge_connections(c, d2[p]) if p in d2 else c)
                seen.add(p)
            for p, c in d2.items():
                if p not in seen:
                    conns.append(c)
            return conns

        return GroverResult(
            files=_union_files(self._as_dict(), other._as_dict()),
            directories=_union_dirs(self._dirs_as_dict(), other._dirs_as_dict()),
            connections=_union_conns(self._connections_as_dict(), other._connections_as_dict()),
            success=self.success or other.success,
            message="",
        )

    def __sub__(self, other: object) -> GroverResult:
        """Difference — paths in LHS not in RHS."""
        if not isinstance(other, GroverResult):
            return NotImplemented
        return self._algebra(
            other,
            lambda d1, d2: [d1[p] for p in set(d1) - set(d2)],
            lambda d1, d2: [d1[p] for p in set(d1) - set(d2)],
            lambda d1, d2: [d1[p] for p in set(d1) - set(d2)],
            lambda a, b: a,
        )

    def __rshift__(self, other: object) -> GroverResult:
        """Pipeline — passes LHS paths as candidates to RHS."""
        if not isinstance(other, GroverResult):
            return NotImplemented
        return self._algebra(
            other,
            lambda d1, d2: [self._merge_files(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda d1, d2: [self._merge_dirs(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda d1, d2: [self._merge_connections(d1[p], d2[p]) for p in set(d1) & set(d2)],
            lambda a, b: a and b,
        )

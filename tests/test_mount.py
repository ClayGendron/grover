"""Tests for the Mount class — construction, dispatch, backward compat."""

from __future__ import annotations

from typing import Any

import pytest

from grover.fs.permissions import Permission
from grover.mount import Mount, ProtocolConflictError, ProtocolNotAvailableError
from grover.mount.protocols import (
    SupportsEmbedding,
    SupportsGlob,
    SupportsGrep,
    SupportsLexicalSearch,
    SupportsListDir,
    SupportsTree,
    SupportsVectorSearch,
)

# ------------------------------------------------------------------
# Fake components for testing
# ------------------------------------------------------------------


class FakeFilesystem:
    """Minimal fake filesystem (no dispatch protocol methods)."""

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def read(self, path: str, **kw: Any) -> Any: ...

    async def write(self, path: str, content: str, **kw: Any) -> Any: ...


class GlobbableFilesystem(FakeFilesystem):
    """Filesystem that also satisfies SupportsGlob."""

    async def glob(self, pattern: str, path: str = "/", **kwargs: Any) -> Any:
        return []


class FullFilesystem(FakeFilesystem):
    """Filesystem that satisfies all four FS dispatch protocols."""

    async def glob(self, pattern: str, path: str = "/", **kwargs: Any) -> Any:
        return []

    async def grep(self, pattern: str, path: str = "/", **kwargs: Any) -> Any:
        return []

    async def tree(self, path: str = "/", **kwargs: Any) -> Any:
        return []

    async def list_dir(self, path: str = "/", **kwargs: Any) -> Any:
        return []


class FakeGraph:
    """Minimal fake graph."""

    def add_node(self, path: str, **attrs: object) -> None: ...

    def remove_node(self, path: str) -> None: ...

    def has_node(self, path: str) -> bool:
        return False


class FakeSearchEngine:
    """Fake SearchEngine that exposes supported_protocols()."""

    def __init__(self, protos: set[type] | None = None) -> None:
        self._protos = protos or set()

    def supported_protocols(self) -> set[type]:
        return self._protos


class ConflictingGlobSearch:
    """A search component that also satisfies SupportsGlob (conflict!)."""

    async def glob(self, pattern: str, path: str = "/", **kwargs: Any) -> Any:
        return []

    def supported_protocols(self) -> set[type]:
        return {SupportsGlob}


# ==================================================================
# Construction
# ==================================================================


class TestMountConstruction:
    def test_basic_construction(self):
        fs = FakeFilesystem()
        m = Mount(path="/project", filesystem=fs)
        assert m.path == "/project"
        assert m.filesystem is fs
        assert m.graph is None
        assert m.search is None

    def test_all_components(self):
        fs = FakeFilesystem()
        graph = FakeGraph()
        search = FakeSearchEngine()
        m = Mount(path="/app", filesystem=fs, graph=graph, search=search)
        assert m.filesystem is fs
        assert m.graph is graph
        assert m.search is search

    def test_path_normalized(self):
        m = Mount(path="project/src", filesystem=FakeFilesystem())
        assert m.path == "/project/src"

    def test_trailing_slash_stripped(self):
        m = Mount(path="/project/", filesystem=FakeFilesystem())
        assert m.path == "/project"

    def test_default_label_from_path(self):
        m = Mount(path="/my-project", filesystem=FakeFilesystem())
        assert m.label == "my-project"

    def test_custom_label(self):
        m = Mount(path="/project", filesystem=FakeFilesystem(), label="My App")
        assert m.label == "My App"

    def test_default_permission(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        assert m.permission == Permission.READ_WRITE

    def test_custom_permission(self):
        m = Mount(
            path="/project",
            filesystem=FakeFilesystem(),
            permission=Permission.READ_ONLY,
        )
        assert m.permission == Permission.READ_ONLY

    def test_hidden_default(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        assert m.hidden is False

    def test_hidden_true(self):
        m = Mount(path="/project", filesystem=FakeFilesystem(), hidden=True)
        assert m.hidden is True

    def test_read_only_paths_default(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        assert m.read_only_paths == set()

    def test_read_only_paths_custom(self):
        m = Mount(
            path="/project",
            filesystem=FakeFilesystem(),
            read_only_paths={"/project/locked"},
        )
        assert m.read_only_paths == {"/project/locked"}

    def test_mount_type_default(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        assert m.mount_type == "vfs"


# ==================================================================
# Protocol dispatch
# ==================================================================


class TestMountDispatch:
    def test_dispatch_filesystem_glob(self):
        fs = GlobbableFilesystem()
        m = Mount(path="/project", filesystem=fs)
        assert m.dispatch(SupportsGlob) is fs

    def test_dispatch_all_fs_protocols(self):
        fs = FullFilesystem()
        m = Mount(path="/project", filesystem=fs)
        assert m.dispatch(SupportsGlob) is fs
        assert m.dispatch(SupportsGrep) is fs
        assert m.dispatch(SupportsTree) is fs
        assert m.dispatch(SupportsListDir) is fs

    def test_dispatch_search_protocols(self):
        search = FakeSearchEngine({SupportsVectorSearch, SupportsEmbedding})
        m = Mount(path="/project", filesystem=FakeFilesystem(), search=search)
        assert m.dispatch(SupportsVectorSearch) is search
        assert m.dispatch(SupportsEmbedding) is search

    def test_dispatch_not_available(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        with pytest.raises(ProtocolNotAvailableError, match="SupportsVectorSearch"):
            m.dispatch(SupportsVectorSearch)

    def test_dispatch_conflict(self):
        fs = GlobbableFilesystem()
        conflicting_search = ConflictingGlobSearch()
        with pytest.raises(ProtocolConflictError, match="SupportsGlob"):
            Mount(path="/project", filesystem=fs, search=conflicting_search)

    def test_has_capability_true(self):
        fs = GlobbableFilesystem()
        m = Mount(path="/project", filesystem=fs)
        assert m.has_capability(SupportsGlob) is True

    def test_has_capability_false(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        assert m.has_capability(SupportsVectorSearch) is False

    def test_supported_protocols_filesystem_only(self):
        fs = FullFilesystem()
        m = Mount(path="/project", filesystem=fs)
        protos = m.supported_protocols()
        assert SupportsGlob in protos
        assert SupportsGrep in protos
        assert SupportsTree in protos
        assert SupportsListDir in protos
        assert SupportsVectorSearch not in protos

    def test_supported_protocols_with_search(self):
        search = FakeSearchEngine({SupportsVectorSearch, SupportsLexicalSearch})
        m = Mount(path="/project", filesystem=FakeFilesystem(), search=search)
        protos = m.supported_protocols()
        assert SupportsVectorSearch in protos
        assert SupportsLexicalSearch in protos

    def test_supported_protocols_empty(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        protos = m.supported_protocols()
        # FakeFilesystem doesn't satisfy any dispatch protocols
        assert len(protos) == 0

    def test_none_components_skipped(self):
        """None graph and None search should not cause errors."""
        m = Mount(path="/project", filesystem=FakeFilesystem(), graph=None, search=None)
        assert m.graph is None
        assert m.search is None


# ==================================================================
# Repr
# ==================================================================


class TestMountRepr:
    def test_repr_basic(self):
        m = Mount(path="/project", filesystem=FakeFilesystem())
        r = repr(m)
        assert "Mount(" in r
        assert "path='/project'" in r
        assert "FakeFilesystem" in r

    def test_repr_with_all_components(self):
        m = Mount(
            path="/app",
            filesystem=FakeFilesystem(),
            graph=FakeGraph(),
            search=FakeSearchEngine(),
        )
        r = repr(m)
        assert "FakeFilesystem" in r
        assert "FakeGraph" in r
        assert "FakeSearchEngine" in r

"""Contract tests for Grover/GroverAsync public behavior."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

import grover
from _helpers import FakeProvider
from grover.client import Grover, GroverAsync
from grover.models.config import SessionConfig
from grover.models.internal.ref import File
from grover.models.internal.results import FileOperationResult, FileSearchResult, FileSearchSet


class InMemoryBackend:
    """Simple backend that implements only core GroverFileSystem methods."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self.open_calls = 0
        self.close_calls = 0

    async def open(self) -> None:
        self.open_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        content = self._files.get(path)
        if content is None:
            return FileOperationResult(success=False, message=f"Not found: {path}")
        f = File(path=path, content=content)
        return FileOperationResult(success=True, message="OK", file=f)

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        return FileSearchResult(success=True, message="OK")

    async def exists(
        self,
        path: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        exists = path in self._files
        return FileOperationResult(
            success=True,
            message="exists" if exists else "not found",
            file=File(path=path),
        )

    async def get_info(
        self,
        path: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        if path not in self._files:
            return FileOperationResult(success=False, message="Not found", file=File(path=path))
        return FileOperationResult(success=True, message="OK", file=File(path=path, is_directory=False))

    async def write(
        self,
        path: str,
        content: str,
        created_by: str = "agent",
        *,
        overwrite: bool = True,
        session: object | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        self._files[path] = content
        return FileOperationResult(success=True, message="OK", file=File(path=path))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        created_by: str = "agent",
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        content = self._files.get(path)
        if content is None:
            return FileOperationResult(success=False, message=f"Not found: {path}")
        self._files[path] = content.replace(old_string, new_string, 1)
        return FileOperationResult(success=True, message="OK", file=File(path=path))

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        if path not in self._files:
            return FileOperationResult(success=False, message=f"Not found: {path}")
        del self._files[path]
        return FileOperationResult(success=True, message="OK", file=File(path=path))

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        f = File(path=path, is_directory=True)
        return FileOperationResult(success=True, message="OK", file=f)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        content = self._files.pop(src, None)
        if content is None:
            return FileOperationResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return FileOperationResult(success=True, message="OK", file=File(path=dest))

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileOperationResult:
        content = self._files.get(src)
        if content is None:
            return FileOperationResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return FileOperationResult(success=True, message="OK", file=File(path=dest))


class OpenFailBackend(InMemoryBackend):
    """Backend whose open() fails to test mount rollback behavior."""

    async def open(self) -> None:
        raise RuntimeError("open failed")


class BadCommitSession:
    """Session object whose commit fails."""

    async def commit(self) -> None:
        raise RuntimeError("commit failed")

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_async_unmount_requires_exact_mount_path(tmp_path: Path) -> None:
    g = GroverAsync()
    backend = InMemoryBackend()
    try:
        await g.add_mount("app", filesystem=backend, embedding_provider=FakeProvider())
        await g.unmount("/app/subpath")
        assert g._ctx.registry.has_mount("/app")
        assert backend.close_calls == 0
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_mount_open_failure_does_not_register_mount(tmp_path: Path) -> None:
    g = GroverAsync()
    try:
        with pytest.raises(RuntimeError, match="open failed"):
            await g.add_mount("bad", filesystem=OpenFailBackend(), embedding_provider=FakeProvider())
        assert not g._ctx.registry.has_mount("/bad")
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_write_edit_delete_return_result_types(tmp_path: Path) -> None:
    g = GroverAsync()
    try:
        await g.add_mount(
            "app",
            filesystem=InMemoryBackend(),
            embedding_provider=FakeProvider(),
            session_config=SessionConfig(session_factory=AsyncMock, dialect="sqlite"),
        )
        write_result = await g.write("/app/file.txt", "hello")
        edit_result = await g.edit("/app/file.txt", "hello", "world")
        delete_result = await g.delete("/app/file.txt", permanent=True)

        assert isinstance(write_result, FileOperationResult)
        assert isinstance(edit_result, FileOperationResult)
        assert isinstance(delete_result, FileOperationResult)
        assert write_result.success
        assert edit_result.success
        assert delete_result.success
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_write_commit_failure_returns_failed_result(tmp_path: Path) -> None:
    g = GroverAsync()
    try:
        await g.add_mount("app", filesystem=InMemoryBackend(), embedding_provider=FakeProvider())
        mount = next(m for m in g._ctx.registry.list_mounts() if m.path == "/app")
        mount.session_factory = BadCommitSession

        result = await g.write("/app/file.txt", "hello")
        assert isinstance(result, FileOperationResult)
        assert not result.success
        assert "commit" in result.message.lower()
    finally:
        await g.close()


def test_sync_write_edit_delete_return_result_types(tmp_path: Path) -> None:
    g = Grover()
    try:
        g.add_mount(
            "app",
            filesystem=InMemoryBackend(),
            embedding_provider=FakeProvider(),
            session_config=SessionConfig(session_factory=AsyncMock, dialect="sqlite"),
        )
        write_result = g.write("/app/file.txt", "hello")
        edit_result = g.edit("/app/file.txt", "hello", "world")
        delete_result = g.delete("/app/file.txt", permanent=True)

        assert isinstance(write_result, FileOperationResult)
        assert isinstance(edit_result, FileOperationResult)
        assert isinstance(delete_result, FileOperationResult)
        assert write_result.success
        assert edit_result.success
        assert delete_result.success
    finally:
        g.close()


def test_sync_write_commit_failure_returns_failed_result(tmp_path: Path) -> None:
    g = Grover()
    try:
        g.add_mount("app", filesystem=InMemoryBackend(), embedding_provider=FakeProvider())
        mount = next(m for m in g._async._ctx.registry.list_mounts() if m.path == "/app")
        mount.session_factory = BadCommitSession

        result = g.write("/app/file.txt", "hello")
        assert isinstance(result, FileOperationResult)
        assert not result.success
        assert "commit" in result.message.lower()
    finally:
        g.close()


def test_version_is_exported() -> None:
    assert hasattr(grover, "__version__")
    assert isinstance(grover.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+$", grover.__version__)


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', text, re.MULTILINE)
    assert match is not None, "Could not find version in pyproject.toml"
    assert grover.__version__ == match.group(1)


@pytest.mark.asyncio
async def test_grover_async_capability_check(tmp_path: Path) -> None:
    """Calling algorithm methods on a minimal graph raises AttributeError."""

    class MinimalGraph:
        """Graph with only CRUD methods — no algorithm methods."""

        nodes: ClassVar[set[str]] = set()

        def add_node(self, path: str, **attrs: object) -> None: ...
        def remove_node(self, path: str) -> None: ...
        def has_node(self, path: str) -> bool:
            return False

        def get_node(self, path: str) -> dict:
            return {}

        def add_edge(self, source: str, target: str, edge_type: str, **kw: object) -> None: ...
        def remove_edge(self, source: str, target: str) -> None: ...
        def has_edge(self, source: str, target: str) -> bool:
            return False

        def get_edge(self, source: str, target: str) -> dict:
            return {}

        @property
        def edges(self) -> list:
            return []

        def predecessors(self, path: str) -> list:
            return []

        def successors(self, path: str) -> list:
            return []

        def path_between(self, source: str, target: str) -> None:
            return None

        def is_dag(self) -> bool:
            return True

    ga = GroverAsync()
    try:
        await ga.add_mount("app", filesystem=InMemoryBackend(), embedding_provider=FakeProvider())
        # Inject MinimalGraph onto the mounted backend
        mount = next(m for m in ga._ctx.registry.list_visible_mounts() if m.path == "/app")
        mount.filesystem.graph_provider = MinimalGraph()
        with pytest.raises(AttributeError):
            await ga.pagerank(FileSearchSet.from_paths(["/app"]))
        with pytest.raises(AttributeError):
            await ga.betweenness_centrality(FileSearchSet.from_paths(["/app"]))
        with pytest.raises(AttributeError):
            await ga.hits(FileSearchSet.from_paths(["/app"]))
    finally:
        await ga.close()

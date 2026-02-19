"""Contract tests for Grover/GroverAsync public behavior."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import grover
from grover._grover import Grover
from grover._grover_async import GroverAsync
from grover.fs.types import (
    DeleteResult,
    EditResult,
    FileInfo,
    ListResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    WriteResult,
)


class FakeProvider:
    """Small deterministic embedding provider for tests."""

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class InMemoryBackend:
    """Simple backend that implements only core StorageBackend methods."""

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
    ) -> ReadResult:
        content = self._files.get(path)
        if content is None:
            return ReadResult(success=False, message=f"Not found: {path}")
        return ReadResult(success=True, message="OK", content=content, file_path=path)

    async def list_dir(
        self,
        path: str = "/",
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> ListResult:
        return ListResult(success=True, message="OK", entries=[], path=path)

    async def exists(
        self,
        path: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> bool:
        return path in self._files

    async def get_info(
        self,
        path: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> FileInfo | None:
        if path not in self._files:
            return None
        return FileInfo(path=path, name=path.rsplit("/", 1)[-1], is_directory=False)

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
    ) -> WriteResult:
        self._files[path] = content
        return WriteResult(success=True, message="OK", file_path=path)

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
    ) -> EditResult:
        content = self._files.get(path)
        if content is None:
            return EditResult(success=False, message=f"Not found: {path}")
        self._files[path] = content.replace(old_string, new_string, 1)
        return EditResult(success=True, message="OK", file_path=path)

    async def delete(
        self,
        path: str,
        permanent: bool = False,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> DeleteResult:
        if path not in self._files:
            return DeleteResult(success=False, message=f"Not found: {path}")
        del self._files[path]
        return DeleteResult(success=True, message="OK", file_path=path, permanent=permanent)

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> MkdirResult:
        return MkdirResult(success=True, message="OK", path=path)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> MoveResult:
        content = self._files.pop(src, None)
        if content is None:
            return MoveResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return MoveResult(success=True, message="OK", old_path=src, new_path=dest)

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: object | None = None,
        user_id: str | None = None,
    ) -> WriteResult:
        content = self._files.get(src)
        if content is None:
            return WriteResult(success=False, message=f"Not found: {src}")
        self._files[dest] = content
        return WriteResult(success=True, message="OK", file_path=dest)


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
    g = GroverAsync(data_dir=tmp_path / ".grover_data", embedding_provider=FakeProvider())
    backend = InMemoryBackend()
    try:
        await g.mount("/app", backend)
        await g.unmount("/app/subpath")
        assert g._registry.has_mount("/app")
        assert backend.close_calls == 0
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_mount_open_failure_does_not_register_mount(tmp_path: Path) -> None:
    g = GroverAsync(data_dir=tmp_path / ".grover_data", embedding_provider=FakeProvider())
    try:
        with pytest.raises(RuntimeError, match="open failed"):
            await g.mount("/bad", OpenFailBackend())
        assert not g._registry.has_mount("/bad")
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_write_edit_delete_return_result_types(tmp_path: Path) -> None:
    g = GroverAsync(data_dir=tmp_path / ".grover_data", embedding_provider=FakeProvider())
    try:
        await g.mount("/app", InMemoryBackend())
        write_result = await g.write("/app/file.txt", "hello")
        edit_result = await g.edit("/app/file.txt", "hello", "world")
        delete_result = await g.delete("/app/file.txt", permanent=True)

        assert isinstance(write_result, WriteResult)
        assert isinstance(edit_result, EditResult)
        assert isinstance(delete_result, DeleteResult)
        assert write_result.success
        assert edit_result.success
        assert delete_result.success
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_async_write_commit_failure_returns_failed_result(tmp_path: Path) -> None:
    g = GroverAsync(data_dir=tmp_path / ".grover_data", embedding_provider=FakeProvider())
    try:
        await g.mount("/app", InMemoryBackend())
        mount = next(m for m in g._registry.list_mounts() if m.mount_path == "/app")
        mount.session_factory = BadCommitSession

        result = await g.write("/app/file.txt", "hello")
        assert isinstance(result, WriteResult)
        assert not result.success
        assert "commit" in result.message.lower()
    finally:
        await g.close()


def test_sync_write_edit_delete_return_result_types(tmp_path: Path) -> None:
    g = Grover(data_dir=str(tmp_path / ".grover_data"), embedding_provider=FakeProvider())
    try:
        g.mount("/app", InMemoryBackend())
        write_result = g.write("/app/file.txt", "hello")
        edit_result = g.edit("/app/file.txt", "hello", "world")
        delete_result = g.delete("/app/file.txt", permanent=True)

        assert isinstance(write_result, WriteResult)
        assert isinstance(edit_result, EditResult)
        assert isinstance(delete_result, DeleteResult)
        assert write_result.success
        assert edit_result.success
        assert delete_result.success
    finally:
        g.close()


def test_sync_write_commit_failure_returns_failed_result(tmp_path: Path) -> None:
    g = Grover(data_dir=str(tmp_path / ".grover_data"), embedding_provider=FakeProvider())
    try:
        g.mount("/app", InMemoryBackend())
        mount = next(m for m in g._async._registry.list_mounts() if m.mount_path == "/app")
        mount.session_factory = BadCommitSession

        result = g.write("/app/file.txt", "hello")
        assert isinstance(result, WriteResult)
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


# ======================================================================
# Graph protocol and algorithm exports
# ======================================================================


def test_graph_store_exported() -> None:
    from grover import GraphStore

    assert GraphStore is not None


def test_subgraph_result_exported() -> None:
    from grover import SubgraphResult

    assert SubgraphResult is not None


def test_grover_has_pagerank() -> None:
    assert hasattr(Grover, "pagerank")


def test_grover_has_meeting_subgraph() -> None:
    assert hasattr(Grover, "meeting_subgraph")


def test_grover_has_neighborhood() -> None:
    assert hasattr(Grover, "neighborhood")


def test_grover_has_ancestors() -> None:
    assert hasattr(Grover, "ancestors")


def test_grover_has_descendants() -> None:
    assert hasattr(Grover, "descendants")


def test_grover_has_find_nodes() -> None:
    assert hasattr(Grover, "find_nodes")


def test_grover_get_graph_is_public() -> None:
    """GroverAsync.get_graph() is a public method (replaces old .graph property)."""
    ga = GroverAsync()
    try:
        assert hasattr(ga, "get_graph")
        assert callable(ga.get_graph)
    finally:
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(ga.close())


@pytest.mark.asyncio
async def test_grover_async_capability_check(tmp_path: Path) -> None:
    """Convenience wrappers raise CapabilityNotSupportedError for unsupported backends."""
    from grover.fs.exceptions import CapabilityNotSupportedError

    class MinimalGraph:
        """Graph that satisfies GraphStore but no capability protocols."""

        def add_node(self, path: str, **attrs: object) -> None: ...
        def remove_node(self, path: str) -> None: ...
        def has_node(self, path: str) -> bool:
            return False

        def get_node(self, path: str) -> dict:
            return {}

        def nodes(self) -> list[str]:
            return []

        def add_edge(self, source: str, target: str, edge_type: str, **kw: object) -> None: ...
        def remove_edge(self, source: str, target: str) -> None: ...
        def has_edge(self, source: str, target: str) -> bool:
            return False

        def get_edge(self, source: str, target: str) -> dict:
            return {}

        def edges(self) -> list:
            return []

        def dependents(self, path: str) -> list:
            return []

        def dependencies(self, path: str) -> list:
            return []

        def impacts(self, path: str, max_depth: int = 3) -> list:
            return []

        def path_between(self, source: str, target: str) -> None:
            return None

        def contains(self, path: str) -> list:
            return []

        def by_parent(self, parent_path: str) -> list:
            return []

        def remove_file_subgraph(self, path: str) -> list[str]:
            return []

        @property
        def node_count(self) -> int:
            return 0

        @property
        def edge_count(self) -> int:
            return 0

        def is_dag(self) -> bool:
            return True

    ga = GroverAsync(data_dir=tmp_path / ".grover_data", embedding_provider=FakeProvider())
    try:
        await ga.mount("/app", InMemoryBackend())
        # Inject MinimalGraph onto the mounted backend
        mount = next(m for m in ga._registry.list_visible_mounts() if m.mount_path == "/app")
        mount.backend._graph = MinimalGraph()  # type: ignore[union-attr]
        with pytest.raises(CapabilityNotSupportedError):
            ga.pagerank(path="/app")
        with pytest.raises(CapabilityNotSupportedError):
            ga.ancestors("/app/a.py")
        with pytest.raises(CapabilityNotSupportedError):
            ga.descendants("/app/a.py")
        with pytest.raises(CapabilityNotSupportedError):
            ga.meeting_subgraph(["/app/a.py"])
        with pytest.raises(CapabilityNotSupportedError):
            ga.neighborhood("/app/a.py")
        with pytest.raises(CapabilityNotSupportedError):
            ga.find_nodes(path="/app", lang="python")
    finally:
        await ga.close()

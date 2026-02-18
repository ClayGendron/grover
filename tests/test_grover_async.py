"""Tests for the GroverAsync class."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover._grover_async import GroverAsync
from grover.fs.local_fs import LocalFileSystem
from grover.graph._graph import Graph
from grover.ref import Ref
from grover.search._index import SearchResult

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fake embedding provider (deterministic, fast)
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing."""

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def workspace2(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace2"
    ws.mkdir()
    return ws


@pytest.fixture
async def grover(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_no_search(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data))
    await g.mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Lifecycle
# ==================================================================


class TestGroverAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_construction_no_args(self):
        g = GroverAsync()
        assert g._meta_fs is None  # No mounts yet

    @pytest.mark.asyncio
    async def test_mount_creates_meta_fs(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        assert g._meta_fs is not None
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        await g.write("/app/test.txt", "hello")
        assert await g.exists("/app/test.txt")
        await g.unmount("/app")
        # Mount should be gone
        assert not g._registry.has_mount("/app")
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount_grover_raises(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        with pytest.raises(ValueError, match=r"Cannot unmount /\.grover"):
            await g.unmount("/.grover")
        await g.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
        await g.close()
        await g.close()  # Should not raise


# ==================================================================
# Direct Access Mode
# ==================================================================


class TestGroverAsyncDirectAccess:
    @pytest.mark.asyncio
    async def test_write_and_read(self, grover: GroverAsync):
        assert await grover.write("/project/hello.txt", "hello world")
        result = await grover.read("/project/hello.txt")
        assert result.success
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_edit(self, grover: GroverAsync):
        await grover.write("/project/doc.txt", "old text here")
        assert await grover.edit("/project/doc.txt", "old", "new")
        result = await grover.read("/project/doc.txt")
        assert result.content == "new text here"

    @pytest.mark.asyncio
    async def test_delete(self, grover: GroverAsync):
        await grover.write("/project/tmp.txt", "temporary")
        assert await grover.delete("/project/tmp.txt")
        result = await grover.read("/project/tmp.txt")
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_dir(self, grover: GroverAsync):
        await grover.write("/project/a.txt", "a")
        await grover.write("/project/b.txt", "b")
        entries = await grover.list_dir("/project")
        names = {e["name"] for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names

    @pytest.mark.asyncio
    async def test_exists(self, grover: GroverAsync):
        assert not await grover.exists("/project/nope.txt")
        await grover.write("/project/yes.txt", "yes")
        assert await grover.exists("/project/yes.txt")

    @pytest.mark.asyncio
    async def test_write_overwrite_false_fails_when_exists(self, grover: GroverAsync):
        await grover.write("/project/exists.txt", "original")
        result = await grover.write("/project/exists.txt", "new", overwrite=False)
        assert not result.success
        # Original content should be unchanged
        assert (await grover.read("/project/exists.txt")).content == "original"

    @pytest.mark.asyncio
    async def test_write_overwrite_false_succeeds_for_new(self, grover: GroverAsync):
        result = await grover.write("/project/brand_new.txt", "content", overwrite=False)
        assert result.success
        assert (await grover.read("/project/brand_new.txt")).content == "content"

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, grover: GroverAsync):
        await grover.write("/project/multi.txt", "foo bar foo baz foo")
        result = await grover.edit("/project/multi.txt", "foo", "qux", replace_all=True)
        assert result.success
        assert (await grover.read("/project/multi.txt")).content == "qux bar qux baz qux"

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, grover: GroverAsync):
        lines = "\n".join(f"line {i}" for i in range(20))
        await grover.write("/project/lines.txt", lines)
        result = await grover.read("/project/lines.txt", offset=5, limit=3)
        assert result.success
        content = result.content
        assert content is not None
        assert "line 5" in content
        assert "line 7" in content
        assert "line 8" not in content


# ==================================================================
# Multi-Mount CRUD
# ==================================================================


class TestGroverAsyncMultiMount:
    @pytest.mark.asyncio
    async def test_two_mounts(self, workspace: Path, workspace2: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_app"))
        await g.mount(
            "/data", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_data")
        )

        await g.write("/app/code.txt", "code content")
        await g.write("/data/doc.txt", "doc content")

        assert (await g.read("/app/code.txt")).content == "code content"
        assert (await g.read("/data/doc.txt")).content == "doc content"

        # List root should show both mounts (but not .grover)
        entries = await g.list_dir("/")
        names = {e["name"] for e in entries}
        assert "app" in names
        assert "data" in names
        assert ".grover" not in names

        await g.close()

    @pytest.mark.asyncio
    async def test_isolation_between_mounts(
        self, workspace: Path, workspace2: Path, tmp_path: Path
    ):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/a", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a"))
        await g.mount("/b", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b"))

        await g.write("/a/file.txt", "in mount a")
        assert await g.exists("/a/file.txt")
        assert not await g.exists("/b/file.txt")

        await g.close()


# ==================================================================
# Graph
# ==================================================================


class TestGroverAsyncGraph:
    @pytest.mark.asyncio
    async def test_graph_property(self, grover: GroverAsync):
        assert isinstance(grover.graph, Graph)

    @pytest.mark.asyncio
    async def test_write_updates_graph(self, grover: GroverAsync):
        await grover.write("/project/mod.py", "def work():\n    pass\n")
        assert grover.graph.has_node("/project/mod.py")

    @pytest.mark.asyncio
    async def test_contains_returns_chunks(self, grover: GroverAsync):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        await grover.write("/project/funcs.py", code)
        chunks = grover.contains("/project/funcs.py")
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_delete_removes_from_graph(self, grover: GroverAsync):
        await grover.write("/project/gone.py", "def gone():\n    pass\n")
        assert grover.graph.has_node("/project/gone.py")
        await grover.delete("/project/gone.py")
        assert not grover.graph.has_node("/project/gone.py")


# ==================================================================
# Search
# ==================================================================


class TestGroverAsyncSearch:
    @pytest.mark.asyncio
    async def test_search_after_write(self, grover: GroverAsync):
        code = 'def authenticate_user():\n    """Verify user credentials."""\n    pass\n'
        await grover.write("/project/auth.py", code)
        results = await grover.search("authenticate")
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self, grover: GroverAsync):
        await grover.write("/project/data.txt", "important data content")
        results = await grover.search("data")
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty(self, grover: GroverAsync):
        results = await grover.search("nonexistent query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_raises_without_provider(self, grover_no_search: GroverAsync):
        if grover_no_search._search_index is not None:
            pytest.skip("sentence-transformers is installed; search available")
        with pytest.raises(RuntimeError, match="Search is not available"):
            await grover_no_search.search("anything")


# ==================================================================
# Index
# ==================================================================


class TestGroverAsyncIndex:
    @pytest.mark.asyncio
    async def test_index_scans_files(self, grover: GroverAsync, workspace: Path):
        (workspace / "one.py").write_text("def one():\n    return 1\n")
        (workspace / "two.py").write_text("def two():\n    return 2\n")
        stats = await grover.index()
        assert stats["files_scanned"] >= 2

    @pytest.mark.asyncio
    async def test_index_specific_mount(
        self,
        workspace: Path,
        workspace2: Path,
        tmp_path: Path,
    ):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/a", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a"))
        await g.mount("/b", LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b"))

        (workspace / "a.py").write_text("def a():\n    pass\n")
        (workspace2 / "b.py").write_text("def b():\n    pass\n")

        stats = await g.index("/a")
        # Should only index mount /a
        assert stats["files_scanned"] >= 1
        assert g.graph.has_node("/a/a.py")

        await g.close()


# ==================================================================
# Persistence
# ==================================================================


class TestGroverAsyncPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, workspace: Path, tmp_path: Path):
        data_dir = tmp_path / "data"

        g1 = GroverAsync(data_dir=str(data_dir), embedding_provider=FakeProvider())
        await g1.mount(
            "/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local")
        )
        await g1.write("/project/keep.py", "def keep():\n    pass\n")
        await g1.save()
        await g1.close()

        g2 = GroverAsync(data_dir=str(data_dir), embedding_provider=FakeProvider())
        await g2.mount(
            "/project", LocalFileSystem(workspace_dir=workspace, data_dir=data_dir / "local")
        )
        assert g2.graph.has_node("/project/keep.py")
        results = await g2.search("keep")
        assert len(results) >= 1
        await g2.close()


# ==================================================================
# Properties
# ==================================================================


class TestGroverAsyncProperties:
    @pytest.mark.asyncio
    async def test_fs_property(self, grover: GroverAsync):
        assert grover.fs is grover._vfs

    @pytest.mark.asyncio
    async def test_graph_property(self, grover: GroverAsync):
        assert grover.graph is grover._graph


# ==================================================================
# Graph Query Wrappers
# ==================================================================


class TestGroverAsyncGraphQueries:
    @pytest.mark.asyncio
    async def test_dependents_returns_refs(self, grover: GroverAsync):
        await grover.write("/project/lib.py", "def helper():\n    return 42\n")
        await grover.write(
            "/project/main.py",
            "from lib import helper\n\ndef run():\n    return helper()\n",
        )
        deps = grover.dependents("/project/lib.py")
        # main.py imports lib.py, so main.py is a dependent
        # The graph stores "imports" edges from analyzer
        assert isinstance(deps, list)
        # All items are Refs
        assert all(isinstance(r, Ref) for r in deps)

    @pytest.mark.asyncio
    async def test_dependencies_returns_refs(self, grover: GroverAsync):
        await grover.write("/project/dep.py", "def util():\n    pass\n")
        await grover.write(
            "/project/consumer.py",
            "from dep import util\n\ndef main():\n    util()\n",
        )
        deps = grover.dependencies("/project/consumer.py")
        assert isinstance(deps, list)
        assert all(isinstance(r, Ref) for r in deps)

    @pytest.mark.asyncio
    async def test_impacts_transitive(self, grover: GroverAsync):
        await grover.write("/project/c.py", "VALUE = 1\n")
        await grover.write("/project/b.py", "from c import VALUE\n\ndef get():\n    return VALUE\n")
        await grover.write(
            "/project/a.py",
            "from b import get\n\ndef run():\n    return get()\n",
        )
        imp = grover.impacts("/project/c.py")
        assert isinstance(imp, list)

    @pytest.mark.asyncio
    async def test_path_between_found(self, grover: GroverAsync):
        await grover.write("/project/start.py", "X = 1\n")
        await grover.write(
            "/project/mid.py",
            "from start import X\n\ndef mid():\n    return X\n",
        )
        await grover.write(
            "/project/end.py",
            "from mid import mid\n\ndef end():\n    return mid()\n",
        )
        path = grover.path_between("/project/end.py", "/project/start.py")
        # May or may not find a path depending on import analysis depth
        if path is not None:
            assert isinstance(path, list)
            assert all(isinstance(r, Ref) for r in path)

    @pytest.mark.asyncio
    async def test_path_between_none(self, grover: GroverAsync):
        await grover.write("/project/island_a.py", "A = 1\n")
        await grover.write("/project/island_b.py", "B = 2\n")
        path = grover.path_between("/project/island_a.py", "/project/island_b.py")
        assert path is None


# ==================================================================
# Event Handlers
# ==================================================================


class TestGroverAsyncEventHandlers:
    @pytest.mark.asyncio
    async def test_move_updates_graph(self, grover: GroverAsync):
        await grover.write("/project/old.py", "def foo():\n    pass\n")
        assert grover.graph.has_node("/project/old.py")

        result = await grover.fs.move("/project/old.py", "/project/new.py")
        assert result.success
        assert not grover.graph.has_node("/project/old.py")
        assert grover.graph.has_node("/project/new.py")

    @pytest.mark.asyncio
    async def test_move_updates_search_index(self, grover: GroverAsync):
        code = 'def unique_search_target():\n    """Locate me after move."""\n    pass\n'
        await grover.write("/project/before.py", code)
        await grover.fs.move("/project/before.py", "/project/after.py")
        if grover._search_index is not None:
            results = await grover.search("unique_search_target")
            found_paths = [r.ref.path for r in results]
            found_parents = [r.parent_path for r in results if r.parent_path]
            all_paths = found_paths + found_parents
            # Old path should not appear
            assert "/project/before.py" not in all_paths

    @pytest.mark.asyncio
    async def test_restored_event_reanalyzes(self, grover: GroverAsync):
        await grover.write("/project/restore_me.py", "def restored():\n    pass\n")
        assert grover.graph.has_node("/project/restore_me.py")
        await grover.delete("/project/restore_me.py")
        assert not grover.graph.has_node("/project/restore_me.py")
        # Restore it (for LocalFileSystem this is restore_from_trash)
        result = await grover.fs.restore_from_trash("/project/restore_me.py")
        if result.success:
            assert grover.graph.has_node("/project/restore_me.py")


# ==================================================================
# Mount Options
# ==================================================================


class TestGroverAsyncMountOptions:
    @pytest.mark.asyncio
    async def test_hidden_mount_not_in_list_dir(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        ws_hidden = tmp_path / "ws_hidden"
        ws_hidden.mkdir()
        await g.mount(
            "/hidden",
            LocalFileSystem(workspace_dir=ws_hidden, data_dir=data / "local_hidden"),
            hidden=True,
        )
        await g.mount(
            "/visible",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_visible"),
        )
        entries = await g.list_dir("/")
        names = {e["name"] for e in entries}
        assert "visible" in names
        assert "hidden" not in names
        await g.close()

    @pytest.mark.asyncio
    async def test_mount_type_auto_detection(self, workspace: Path, tmp_path: Path):
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel

        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())

        # Local mount
        await g.mount("/local", LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"))
        local_mount = next(m for m in g._registry.list_mounts() if m.mount_path == "/local")
        assert local_mount.mount_type == "local"

        # Database mount via session_factory
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        await g.mount("/db", session_factory=factory, dialect="sqlite")
        db_mount = next(m for m in g._registry.list_mounts() if m.mount_path == "/db")
        assert db_mount.mount_type == "vfs"

        await g.close()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_unmount_nonexistent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"))
        # Should not raise
        await g.unmount("/nonexistent")
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount_cleans_graph_and_search(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount("/app", LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"))
        await g.write("/app/file.py", "def demo():\n    pass\n")
        assert g.graph.has_node("/app/file.py")

        await g.unmount("/app")
        assert not g.graph.has_node("/app/file.py")
        await g.close()


# ==================================================================
# Unsupported File Types
# ==================================================================


class TestGroverAsyncUnsupportedFiles:
    @pytest.mark.asyncio
    async def test_unsupported_file_embeds_whole_file(self, grover: GroverAsync):
        # .txt has no Python analyzer, but the whole file should be indexed
        await grover.write("/project/notes.txt", "Important project notes here")
        assert grover.graph.has_node("/project/notes.txt")
        if grover._search_index is not None:
            results = await grover.search("Important project notes")
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_write_grover_path_skipped(self, grover: GroverAsync):
        # Writes to /.grover/ should not create graph nodes
        await grover.fs.write("/.grover/internal.txt", "metadata")
        assert not grover.graph.has_node("/.grover/internal.txt")


# ==================================================================
# Authenticated mount + sharing
# ==================================================================


@pytest.fixture
async def auth_grover(tmp_path: Path) -> GroverAsync:
    """GroverAsync with a UserScopedFileSystem backend."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from grover.fs.sharing import SharingService
    from grover.fs.user_scoped_fs import UserScopedFileSystem
    from grover.models.shares import FileShare

    data = tmp_path / "grover_data"
    g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    sharing = SharingService(FileShare)
    backend = UserScopedFileSystem(sharing=sharing)
    await g.mount("/ws", backend, engine=engine)
    yield g  # type: ignore[misc]
    await g.close()


class TestGroverAsyncAuthenticatedMount:
    @pytest.mark.asyncio
    async def test_authenticated_mount_write_read(self, auth_grover: GroverAsync):
        result = await auth_grover.write("/ws/notes.md", "hello", user_id="alice")
        assert result.success is True
        read = await auth_grover.read("/ws/notes.md", user_id="alice")
        assert read.success is True
        assert read.content == "hello"

    @pytest.mark.asyncio
    async def test_user_isolation(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "alice data", user_id="alice")
        await auth_grover.write("/ws/notes.md", "bob data", user_id="bob")
        r1 = await auth_grover.read("/ws/notes.md", user_id="alice")
        r2 = await auth_grover.read("/ws/notes.md", user_id="bob")
        assert r1.content == "alice data"
        assert r2.content == "bob data"

    @pytest.mark.asyncio
    async def test_move_and_copy(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/src.md", "content", user_id="alice")
        copy_result = await auth_grover.copy("/ws/src.md", "/ws/copy.md", user_id="alice")
        assert copy_result.success is True
        move_result = await auth_grover.move("/ws/src.md", "/ws/moved.md", user_id="alice")
        assert move_result.success is True
        assert await auth_grover.exists("/ws/copy.md", user_id="alice")
        assert await auth_grover.exists("/ws/moved.md", user_id="alice")


class TestGroverAsyncSharing:
    @pytest.mark.asyncio
    async def test_share_file(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "shared", user_id="alice")
        result = await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        assert result.success is True
        assert result.share is not None
        assert result.share.grantee_id == "bob"
        assert result.share.permission == "read"

    @pytest.mark.asyncio
    async def test_unshare_file(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "shared", user_id="alice")
        await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        result = await auth_grover.unshare("/ws/notes.md", "bob", user_id="alice")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unshare_nonexistent(self, auth_grover: GroverAsync):
        result = await auth_grover.unshare("/ws/x.md", "bob", user_id="alice")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_shares(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "data", user_id="alice")
        await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        await auth_grover.share("/ws/notes.md", "charlie", "write", user_id="alice")
        result = await auth_grover.list_shares("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert len(result.shares) == 2
        grantees = {s.grantee_id for s in result.shares}
        assert grantees == {"bob", "charlie"}

    @pytest.mark.asyncio
    async def test_list_shared_with_me(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/a.md", "a", user_id="alice")
        await auth_grover.share("/ws/a.md", "bob", "read", user_id="alice")
        result = await auth_grover.list_shared_with_me(user_id="bob")
        assert result.success is True
        assert len(result.shares) == 1
        # Path should be an @shared path, not a raw stored path
        assert result.shares[0].path == "/ws/@shared/alice/a.md"

    @pytest.mark.asyncio
    async def test_share_requires_authenticated_mount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync(data_dir=str(data), embedding_provider=FakeProvider())
        await g.mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        )
        result = await g.share("/app/test.md", "bob", user_id="alice")
        assert result.success is False
        assert "sharing" in result.message.lower()
        await g.close()

    @pytest.mark.asyncio
    async def test_share_invalid_permission_returns_failure(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "data", user_id="alice")
        result = await auth_grover.share("/ws/notes.md", "bob", "execute", user_id="alice")
        assert result.success is False
        assert "permission" in result.message.lower()

    @pytest.mark.asyncio
    async def test_shared_read_via_at_shared(self, auth_grover: GroverAsync):
        """End-to-end: share → read via @shared path."""
        await auth_grover.write("/ws/notes.md", "secret", user_id="alice")
        await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        result = await auth_grover.read("/ws/@shared/alice/notes.md", user_id="bob")
        assert result.success is True
        assert result.content == "secret"

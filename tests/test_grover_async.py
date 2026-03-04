"""Tests for the GroverAsync class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _helpers import FAKE_DIM, FakeProvider
from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.providers.graph import RustworkxGraph
from grover.providers.search.local import LocalVectorStore
from grover.results import GraphResult, VectorSearchResult

if TYPE_CHECKING:
    from pathlib import Path


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
    g = GroverAsync()
    await g.add_mount(
        "/project",
        LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


@pytest.fixture
async def grover_no_search(workspace: Path, tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    await g.add_mount("/project", LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"))
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Lifecycle
# ==================================================================


class TestGroverAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_construction_no_args(self):
        g = GroverAsync()
        assert not g._ctx.initialized  # No mounts yet

    @pytest.mark.asyncio
    async def test_mount_sets_initialized(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        assert g._ctx.initialized
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
        await g.write("/app/test.txt", "hello")
        assert (await g.exists("/app/test.txt")).exists
        await g.unmount("/app")
        # Mount should be gone
        assert not g._ctx.registry.has_mount("/app")
        await g.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
        )
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
        result = await grover.list_dir("/project")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "a.txt" in names
        assert "b.txt" in names

    @pytest.mark.asyncio
    async def test_exists(self, grover: GroverAsync):
        assert not (await grover.exists("/project/nope.txt")).exists
        await grover.write("/project/yes.txt", "yes")
        assert (await grover.exists("/project/yes.txt")).exists

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
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_app"),
            embedding_provider=FakeProvider(),
        )
        await g.add_mount(
            "/data",
            LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_data"),
            embedding_provider=FakeProvider(),
        )

        await g.write("/app/code.txt", "code content")
        await g.write("/data/doc.txt", "doc content")

        assert (await g.read("/app/code.txt")).content == "code content"
        assert (await g.read("/data/doc.txt")).content == "doc content"

        # List root should show both mounts (but not .grover)
        result = await g.list_dir("/")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
        assert "app" in names
        assert "data" in names
        assert ".grover" not in names

        await g.close()

    @pytest.mark.asyncio
    async def test_isolation_between_mounts(
        self, workspace: Path, workspace2: Path, tmp_path: Path
    ):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        lfs_a = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a")
        lfs_b = LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b")
        await g.add_mount("/a", lfs_a, embedding_provider=FakeProvider())
        await g.add_mount("/b", lfs_b, embedding_provider=FakeProvider())

        await g.write("/a/file.txt", "in mount a")
        assert (await g.exists("/a/file.txt")).exists
        assert not (await g.exists("/b/file.txt")).exists

        await g.close()


# ==================================================================
# Graph
# ==================================================================


class TestGroverAsyncGraph:
    @pytest.mark.asyncio
    async def test_get_graph(self, grover: GroverAsync):
        assert isinstance(grover.get_graph(), RustworkxGraph)

    @pytest.mark.asyncio
    async def test_write_updates_graph(self, grover: GroverAsync):
        await grover.write("/project/mod.py", "def work():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/mod.py")

    @pytest.mark.asyncio
    async def test_contains_returns_graph_result(self, grover: GroverAsync):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        await grover.write("/project/funcs.py", code)
        await grover.flush()
        result = grover.contains("/project/funcs.py")
        assert isinstance(result, GraphResult)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_delete_removes_from_graph(self, grover: GroverAsync):
        await grover.write("/project/gone.py", "def gone():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/gone.py")
        await grover.delete("/project/gone.py")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/gone.py")


# ==================================================================
# Search
# ==================================================================


class TestGroverAsyncSearch:
    @pytest.mark.asyncio
    async def test_vector_search_after_write(self, grover: GroverAsync):
        code = 'def authenticate_user():\n    """Verify user credentials."""\n    pass\n'
        await grover.write("/project/auth.py", code)
        await grover.flush()
        result = await grover.vector_search("authenticate")
        assert isinstance(result, VectorSearchResult)
        assert result.success is True
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_vector_search_returns_vector_search_result(self, grover: GroverAsync):
        await grover.write("/project/data.txt", "important data content")
        result = await grover.vector_search("data")
        assert isinstance(result, VectorSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_vector_search_empty(self, grover: GroverAsync):
        result = await grover.vector_search("nonexistent query")
        assert isinstance(result, VectorSearchResult)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_vector_search_returns_failure_without_provider(
        self, grover_no_search: GroverAsync
    ):
        has_search = any(
            getattr(m.filesystem, "search_provider", None) is not None
            for m in grover_no_search._ctx.registry.list_visible_mounts()
        )
        if has_search:
            pytest.skip("search provider is installed; search available")
        result = await grover_no_search.vector_search("anything")
        assert result.success is False
        assert "not available" in result.message


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
        g = GroverAsync()
        lfs_a = LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_a")
        lfs_b = LocalFileSystem(workspace_dir=workspace2, data_dir=data / "local_b")
        await g.add_mount("/a", lfs_a, embedding_provider=FakeProvider())
        await g.add_mount("/b", lfs_b, embedding_provider=FakeProvider())

        (workspace / "a.py").write_text("def a():\n    pass\n")
        (workspace2 / "b.py").write_text("def b():\n    pass\n")

        stats = await g.index("/a")
        # Should only index mount /a
        assert stats["files_scanned"] >= 1
        assert g.get_graph().has_node("/a/a.py")

        await g.close()


# ==================================================================
# Persistence
# ==================================================================


# ==================================================================
# Properties
# ==================================================================


class TestGroverAsyncProperties:
    @pytest.mark.asyncio
    async def test_get_graph(self, grover: GroverAsync):
        assert isinstance(grover.get_graph(), RustworkxGraph)


# ==================================================================
# Graph Query Wrappers
# ==================================================================


class TestGroverAsyncGraphQueries:
    @pytest.mark.asyncio
    async def test_predecessors_returns_graph_result(self, grover: GroverAsync):
        await grover.write("/project/lib.py", "def helper():\n    return 42\n")
        await grover.write(
            "/project/main.py",
            "from lib import helper\n\ndef run():\n    return helper()\n",
        )
        await grover.flush()
        result = grover.predecessors("/project/lib.py")
        # main.py imports lib.py, so main.py is a predecessor
        # The graph stores "imports" edges from analyzer
        assert isinstance(result, GraphResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_successors_returns_graph_result(self, grover: GroverAsync):
        await grover.write("/project/dep.py", "def util():\n    pass\n")
        await grover.write(
            "/project/consumer.py",
            "from dep import util\n\ndef main():\n    util()\n",
        )
        await grover.flush()
        result = grover.successors("/project/consumer.py")
        assert isinstance(result, GraphResult)
        assert result.success is True

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
        await grover.flush()
        result = grover.path_between("/project/end.py", "/project/start.py")
        assert isinstance(result, GraphResult)
        assert result.success is True
        # May or may not find a path depending on import analysis depth
        # If path found, result.paths will be non-empty
        if len(result) > 0:
            assert len(result.paths) > 0

    @pytest.mark.asyncio
    async def test_path_between_none(self, grover: GroverAsync):
        await grover.write("/project/island_a.py", "A = 1\n")
        await grover.write("/project/island_b.py", "B = 2\n")
        await grover.flush()
        result = grover.path_between("/project/island_a.py", "/project/island_b.py")
        assert isinstance(result, GraphResult)
        assert result.success is True
        assert len(result) == 0


# ==================================================================
# Event Handlers
# ==================================================================


class TestGroverAsyncEventHandlers:
    @pytest.mark.asyncio
    async def test_move_updates_graph(self, grover: GroverAsync):
        await grover.write("/project/old.py", "def foo():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/old.py")

        result = await grover.move("/project/old.py", "/project/new.py")
        assert result.success
        await grover.flush()
        assert not grover.get_graph().has_node("/project/old.py")
        assert grover.get_graph().has_node("/project/new.py")

    @pytest.mark.asyncio
    async def test_move_updates_search_engine(self, grover: GroverAsync):
        code = 'def unique_search_target():\n    """Locate me after move."""\n    pass\n'
        await grover.write("/project/before.py", code)
        await grover.move("/project/before.py", "/project/after.py")
        has_search = any(
            getattr(m.filesystem, "search_provider", None) is not None
            for m in grover._ctx.registry.list_visible_mounts()
        )
        if has_search:
            result = await grover.vector_search("unique_search_target")
            # Old path should not appear
            assert "/project/before.py" not in result.paths

    @pytest.mark.asyncio
    async def test_restored_event_reanalyzes(self, grover: GroverAsync):
        await grover.write("/project/restore_me.py", "def restored():\n    pass\n")
        await grover.flush()
        assert grover.get_graph().has_node("/project/restore_me.py")
        await grover.delete("/project/restore_me.py")
        await grover.flush()
        assert not grover.get_graph().has_node("/project/restore_me.py")
        # Restore it (for LocalFileSystem this is restore_from_trash)
        result = await grover.restore_from_trash("/project/restore_me.py")
        await grover.flush()
        if result.success:
            assert grover.get_graph().has_node("/project/restore_me.py")


# ==================================================================
# Mount Options
# ==================================================================


class TestGroverAsyncMountOptions:
    @pytest.mark.asyncio
    async def test_hidden_mount_not_in_list_dir(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        ws_hidden = tmp_path / "ws_hidden"
        ws_hidden.mkdir()
        await g.add_mount(
            "/hidden",
            LocalFileSystem(workspace_dir=ws_hidden, data_dir=data / "local_hidden"),
            hidden=True,
        )
        await g.add_mount(
            "/visible",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local_visible"),
            embedding_provider=FakeProvider(),
        )
        result = await g.list_dir("/")
        names = {p.rsplit("/", 1)[-1] for p in result.paths}
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
        g = GroverAsync()

        # Local mount
        await g.add_mount(
            "/local",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"),
            embedding_provider=FakeProvider(),
        )
        local_mount = next(m for m in g._ctx.registry.list_mounts() if m.path == "/local")
        assert local_mount.mount_type == "local"

        # Database mount via session_factory
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        await g.add_mount("/db", session_factory=factory, dialect="sqlite")
        db_mount = next(m for m in g._ctx.registry.list_mounts() if m.path == "/db")
        assert db_mount.mount_type == "vfs"

        await g.close()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_unmount_nonexistent(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"),
            embedding_provider=FakeProvider(),
        )
        # Should not raise
        await g.unmount("/nonexistent")
        await g.close()

    @pytest.mark.asyncio
    async def test_unmount_cleans_graph_and_search(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "l"),
            embedding_provider=FakeProvider(),
        )
        await g.write("/app/file.py", "def demo():\n    pass\n")
        await g.flush()
        assert g.get_graph().has_node("/app/file.py")

        await g.unmount("/app")
        # After unmount, the mount and its graph are gone
        with pytest.raises(RuntimeError, match="No graph available"):
            g.get_graph()
        await g.close()


# ==================================================================
# Unsupported File Types
# ==================================================================


class TestGroverAsyncUnsupportedFiles:
    @pytest.mark.asyncio
    async def test_unsupported_file_embeds_whole_file(self, grover: GroverAsync):
        # .txt has no Python analyzer, but the whole file should be indexed
        await grover.write("/project/notes.txt", "Important project notes here")
        await grover.flush()
        assert grover.get_graph().has_node("/project/notes.txt")
        has_search = any(
            getattr(m.filesystem, "search_provider", None) is not None
            for m in grover._ctx.registry.list_visible_mounts()
        )
        if has_search:
            result = await grover.vector_search("Important project notes")
            assert len(result) >= 1


# ==================================================================
# Authenticated mount + sharing
# ==================================================================


@pytest.fixture
async def auth_grover(tmp_path: Path) -> GroverAsync:
    """GroverAsync with a UserScopedFileSystem backend."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from grover.backends.user_scoped import UserScopedFileSystem
    from grover.models.share import FileShare

    g = GroverAsync()
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    backend = UserScopedFileSystem(share_model=FileShare)
    await g.add_mount("/ws", backend, engine=engine, embedding_provider=FakeProvider())
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
        assert (await auth_grover.exists("/ws/copy.md", user_id="alice")).exists
        assert (await auth_grover.exists("/ws/moved.md", user_id="alice")).exists


class TestGroverAsyncSharing:
    @pytest.mark.asyncio
    async def test_share_file(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/notes.md", "shared", user_id="alice")
        result = await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        assert result.success is True
        assert result.grantee_id == "bob"
        assert result.permission == "read"

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
        from grover.results import ShareEvidence

        await auth_grover.write("/ws/notes.md", "data", user_id="alice")
        await auth_grover.share("/ws/notes.md", "bob", "read", user_id="alice")
        await auth_grover.share("/ws/notes.md", "charlie", "write", user_id="alice")
        result = await auth_grover.list_shares("/ws/notes.md", user_id="alice")
        assert result.success is True
        assert len(result) == 2
        grantees = {
            e.grantee_id
            for c in result.file_candidates
            for e in c.evidence
            if isinstance(e, ShareEvidence)
        }
        assert grantees == {"bob", "charlie"}

    @pytest.mark.asyncio
    async def test_list_shared_with_me(self, auth_grover: GroverAsync):
        await auth_grover.write("/ws/a.md", "a", user_id="alice")
        await auth_grover.share("/ws/a.md", "bob", "read", user_id="alice")
        result = await auth_grover.list_shared_with_me(user_id="bob")
        assert result.success is True
        assert len(result) == 1
        # Path should be an @shared path, not a raw stored path
        assert result.file_candidates[0].path == "/ws/@shared/alice/a.md"

    @pytest.mark.asyncio
    async def test_share_requires_authenticated_mount(self, workspace: Path, tmp_path: Path):
        data = tmp_path / "grover_data"
        g = GroverAsync()
        await g.add_mount(
            "/app",
            LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
            embedding_provider=FakeProvider(),
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


# ==================================================================
# Version operations (file_ops)
# ==================================================================


class TestGroverAsyncVersionOps:
    @pytest.mark.asyncio
    async def test_read_version(self, grover: GroverAsync):
        """read_version returns the content of a specific version."""
        await grover.write("/project/doc.txt", "version one")
        await grover.write("/project/doc.txt", "version two")
        result = await grover.read_version("/project/doc.txt", 1)
        assert result.success is True
        assert result.content == "version one"

    @pytest.mark.asyncio
    async def test_diff_versions_basic(self, grover: GroverAsync):
        """diff_versions computes a unified diff between two versions."""
        await grover.write("/project/doc.txt", "hello\n")
        await grover.write("/project/doc.txt", "hello world\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 2)
        assert result.success is True
        assert result.version_a == 1
        assert result.version_b == 2
        assert result.content_a == "hello\n"
        assert result.content_b == "hello world\n"
        assert result.diff != ""
        assert "-hello" in result.diff or "+hello world" in result.diff

    @pytest.mark.asyncio
    async def test_diff_versions_same_version(self, grover: GroverAsync):
        """diff_versions with same version returns empty diff."""
        await grover.write("/project/doc.txt", "content\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 1)
        assert result.success is True
        assert result.diff == ""

    @pytest.mark.asyncio
    async def test_diff_versions_invalid_version(self, grover: GroverAsync):
        """diff_versions with nonexistent version returns failure."""
        await grover.write("/project/doc.txt", "content\n")
        result = await grover.diff_versions("/project/doc.txt", 1, 999)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_tree_still_works(self, grover: GroverAsync):
        """tree() still works after moving from search_ops to file_ops."""
        await grover.write("/project/a.py", "a\n")
        await grover.write("/project/sub/b.py", "b\n")
        result = await grover.tree("/project")
        assert result.success is True
        assert len(result) >= 2

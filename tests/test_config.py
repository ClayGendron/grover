"""Tests for EngineConfig, SessionConfig, and the add_mount config paths."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from grover.client import GroverAsync
from grover.models.config import EngineConfig, SessionConfig, create_async_engine_factory
from grover.worker import IndexingMode

# ---------------------------------------------------------------------------
# EngineConfig validation
# ---------------------------------------------------------------------------


class TestEngineConfig:
    def test_requires_url_or_factory(self):
        with pytest.raises(ValueError, match="requires either url or engine_factory"):
            EngineConfig()

    def test_rejects_both_url_and_factory(self):
        with pytest.raises(ValueError, match="not both"):
            EngineConfig(url="sqlite+aiosqlite://", engine_factory=lambda: None)  # type: ignore[arg-type]

    def test_url_creates_engine(self):
        config = EngineConfig(url="sqlite+aiosqlite://")
        engine = config.create_engine()
        assert engine.dialect.name == "sqlite"

    def test_factory_creates_engine(self):
        factory = create_async_engine_factory("sqlite+aiosqlite://")
        config = EngineConfig(engine_factory=factory)
        engine = config.create_engine()
        assert engine.dialect.name == "sqlite"

    def test_schema_and_create_tables_defaults(self):
        config = EngineConfig(url="sqlite+aiosqlite://")
        assert config.schema is None
        assert config.create_tables is True

    def test_frozen(self):
        config = EngineConfig(url="sqlite+aiosqlite://")
        with pytest.raises(AttributeError):
            config.url = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SessionConfig validation
# ---------------------------------------------------------------------------


class TestSessionConfig:
    def test_defaults(self):
        sf = lambda: None  # noqa: E731
        config = SessionConfig(session_factory=sf)  # type: ignore[arg-type]
        assert config.schema is None
        assert config.dialect is None

    def test_frozen(self):
        sf = lambda: None  # noqa: E731
        config = SessionConfig(session_factory=sf)  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            config.dialect = "pg"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# engine_factory called at mount time (not before)
# ---------------------------------------------------------------------------


class TestEngineFactoryDeferred:
    async def test_engine_factory_called_at_mount_time(self):
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return create_async_engine("sqlite+aiosqlite://", echo=False)

        config = EngineConfig(engine_factory=counting_factory)
        assert call_count == 0, "Factory should not be called at config creation"

        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)
        assert call_count == 1, "Factory should be called exactly once at mount time"
        await g.close()


# ---------------------------------------------------------------------------
# Engine disposed on unmount
# ---------------------------------------------------------------------------


class TestEngineDisposal:
    async def test_engine_set_on_mount_for_engine_config(self):
        """EngineConfig path stores engine on mount for disposal."""
        config = EngineConfig(url="sqlite+aiosqlite://")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        mount = g._ctx.registry.get_mount("/data")
        assert mount is not None
        assert mount.engine is not None
        assert mount.engine.dialect.name == "sqlite"
        await g.close()

    async def test_unmount_removes_mount(self):
        """Unmounting removes the mount from the registry."""
        config = EngineConfig(url="sqlite+aiosqlite://")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        assert g._ctx.registry.get_mount("/data") is not None
        await g.unmount("/data")
        assert g._ctx.registry.get_mount("/data") is None
        await g.close()

    async def test_session_config_no_engine_on_mount(self):
        """SessionConfig path should NOT store engine on mount."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        config = SessionConfig(session_factory=factory, dialect="sqlite")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", session_config=config)

        mount = g._ctx.registry.get_mount("/data")
        assert mount is not None
        assert mount.engine is None, "SessionConfig path should not store engine"
        await g.close()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Dialect inference from session factory bind
# ---------------------------------------------------------------------------


class TestDialectInference:
    async def test_dialect_inferred_from_bind(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        config = SessionConfig(session_factory=factory)
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", session_config=config)

        mount = g._ctx.registry.get_mount("/data")
        assert mount is not None
        assert mount.filesystem is not None
        assert mount.filesystem.dialect == "sqlite"  # type: ignore[union-attr]
        await g.close()
        await engine.dispose()

    async def test_dialect_explicit_override(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        config = SessionConfig(session_factory=factory, dialect="custom_dialect")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", session_config=config)

        mount = g._ctx.registry.get_mount("/data")
        assert mount.filesystem.dialect == "custom_dialect"  # type: ignore[union-attr]
        await g.close()
        await engine.dispose()

    async def test_dialect_inference_fails_without_bind(self):
        """If session factory has no bind and no explicit dialect, raise."""
        plain_factory = lambda: AsyncSession()  # noqa: E731
        config = SessionConfig(session_factory=plain_factory)
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        with pytest.raises(ValueError, match="Cannot infer dialect"):
            await g.add_mount("data", session_config=config)
        await g.close()


# ---------------------------------------------------------------------------
# create_tables=False skips DDL
# ---------------------------------------------------------------------------


class TestCreateTablesFalse:
    async def test_create_tables_false_skips_ddl(self):
        """With create_tables=False, tables are NOT created automatically."""
        config = EngineConfig(url="sqlite+aiosqlite://", create_tables=False)
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        # Writing should fail because tables don't exist
        result = await g.write("/data/test.txt", "hello")
        assert not result.success
        await g.close()


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schema / table creation log messages
# ---------------------------------------------------------------------------


class TestSchemaTableCreationMessages:
    async def test_tables_created_prints_message(self, capsys):
        """Fresh DB with create_tables=True prints table names."""
        config = EngineConfig(url="sqlite+aiosqlite://")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        captured = capsys.readouterr().out
        assert "Tables created:" in captured
        assert "grover_files" in captured
        assert "grover_file_versions" in captured
        assert "grover_file_chunks" in captured
        assert "grover_file_connections" in captured
        await g.close()

    async def test_tables_already_exist_no_print(self, capsys):
        """When tables already exist, nothing is printed."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Re-use the same engine so tables are already present
        config = EngineConfig(engine_factory=lambda: engine)  # type: ignore[arg-type]

        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        captured = capsys.readouterr().out
        assert "Tables created" not in captured
        await g.close()
        await engine.dispose()

    async def test_create_tables_false_no_print(self, capsys):
        """With create_tables=False, nothing is printed."""
        config = EngineConfig(url="sqlite+aiosqlite://", create_tables=False)
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        captured = capsys.readouterr().out
        assert "Tables created" not in captured
        assert "Schema created" not in captured
        await g.close()

    async def test_schema_none_no_schema_message(self, capsys):
        """Without schema set, no 'Schema created' message is printed."""
        config = EngineConfig(url="sqlite+aiosqlite://")
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        await g.add_mount("data", engine_config=config)

        captured = capsys.readouterr().out
        assert "Schema created" not in captured
        await g.close()


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    async def test_engine_and_session_config_rejected(self):
        g = GroverAsync(indexing_mode=IndexingMode.MANUAL)
        with pytest.raises(ValueError, match="not both"):
            await g.add_mount(
                "data",
                engine_config=EngineConfig(url="sqlite+aiosqlite://"),
                session_config=SessionConfig(session_factory=lambda: AsyncSession(), dialect="sqlite"),
            )
        await g.close()

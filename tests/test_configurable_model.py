"""Tests for configurable file models — custom table names via Model base classes."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.backends.database import DatabaseFileSystem
from grover.models.database.connection import FileConnectionModel, FileConnectionModelBase
from grover.models.database.file import FileModel, FileModelBase
from grover.models.database.version import FileVersionModelBase
from grover.providers.graph import RustworkxGraph
from grover.util.dialect import upsert_file

# ---------------------------------------------------------------------------
# Custom model definitions (what a developer would write)
# ---------------------------------------------------------------------------


class WikiFile(FileModelBase, table=True):
    """Custom file model with a different table name."""

    __tablename__ = "wiki_files"


class WikiFileVersion(FileVersionModelBase, table=True):
    """Custom file version model with a different table name."""

    __tablename__ = "wiki_file_versions"


class WikiFileConnection(FileConnectionModelBase, table=True):
    """Custom connection model with a different table name."""

    __tablename__ = "wiki_file_connections"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_custom_fs() -> tuple[DatabaseFileSystem, object, object]:
    """Create a DatabaseFileSystem backed by custom models."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem()
    # Configure custom models (normally done via EngineConfig/SessionConfig + _configure)
    fs.file_model = WikiFile
    fs.file_version_model = WikiFileVersion
    fs.file_connection_model = WikiFileConnection
    fs._init_default_providers()
    return fs, factory, engine


async def _make_default_fs() -> tuple[DatabaseFileSystem, object, object]:
    """Create a DatabaseFileSystem with default models."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem()
    return fs, factory, engine


# ---------------------------------------------------------------------------
# Model inheritance tests
# ---------------------------------------------------------------------------


class TestModelInheritance:
    def test_file_base_is_not_a_table(self):
        """FileModelBase should not be a concrete table (no __table__)."""
        assert not hasattr(FileModelBase, "__table__")

    def test_file_version_base_is_not_a_table(self):
        assert not hasattr(FileVersionModelBase, "__table__")

    def test_custom_model_has_table_name(self):
        assert WikiFile.__tablename__ == "wiki_files"
        assert WikiFileVersion.__tablename__ == "wiki_file_versions"

    def test_custom_model_inherits_fields(self):
        """WikiFile should have all fields from FileModelBase."""
        wiki = WikiFile(path="/test.md", parent_path="/")
        assert wiki.path == "/test.md"
        assert wiki.current_version == 1
        assert wiki.deleted_at is None
        assert wiki.mime_type == "text/markdown"  # Validator computes from .md extension

    def test_custom_version_inherits_fields(self):
        wv = WikiFileVersion(file_path="/test.txt", version=1, content="hello")
        assert wv.file_path == "/test.txt"
        assert wv.version == 1
        assert wv.content == "hello"

    def test_default_file_still_works(self):
        """Default FileModel model is unchanged."""
        assert FileModel.__tablename__ == "grover_files"
        f = FileModel(path="/hello.py", parent_path="/")
        assert f.path == "/hello.py"


# ---------------------------------------------------------------------------
# Filesystem with custom model
# ---------------------------------------------------------------------------


class TestCustomModelFilesystem:
    async def test_file_model_property(self):
        fs, _factory, engine = await _make_custom_fs()
        assert fs.file_model is WikiFile
        assert fs.file_version_model is WikiFileVersion
        await engine.dispose()

    async def test_default_model_property(self):
        fs, _factory, engine = await _make_default_fs()
        assert fs.file_model is FileModel
        await engine.dispose()

    async def test_write_and_read(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            result = await fs.write("/wiki/page.md", "# Hello World\n", session=session)
            assert result.success is True
            assert "Created" in result.message

            read = await fs.read("/wiki/page.md", session=session)
            assert read.success is True
            assert "Hello World" in read.file.content
        await engine.dispose()

    async def test_edit(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            await fs.write("/page.md", "old content\n", session=session)
            result = await fs.edit("/page.md", "old", "new", session=session)
            assert result.success is True
            assert result.file.current_version == 2

            read = await fs.read("/page.md", session=session)
            assert "new content" in read.file.content
        await engine.dispose()

    async def test_data_written_to_custom_table(self):
        """Verify data actually ends up in the wiki_files table, not grover_files."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem()
        fs.file_model = WikiFile
        fs.file_version_model = WikiFileVersion
        fs.file_connection_model = WikiFileConnection
        fs._init_default_providers()
        async with factory() as session:
            await fs.write("/page.md", "wiki content\n", session=session)
            await session.commit()

        # Query the custom table directly
        async with factory() as session:
            result = await session.execute(select(WikiFile).where(WikiFile.path == "/page.md"))
            wiki_file = result.scalar_one_or_none()
            assert wiki_file is not None
            assert wiki_file.path == "/page.md"

            # The default table should be empty
            result = await session.execute(select(FileModel))
            default_files = result.scalars().all()
            assert len(default_files) == 0

        await engine.dispose()


# ---------------------------------------------------------------------------
# Dialect upsert with custom model
# ---------------------------------------------------------------------------


class TestUpsertWithCustomModel:
    async def test_upsert_custom_model(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            rowcount = await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "wiki-1",
                    "path": "/wiki/page.md",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
                model=WikiFile,
            )
            await session.commit()
            assert rowcount >= 0

            # Verify it's in the wiki_files table
            result = await session.execute(select(WikiFile).where(WikiFile.path == "/wiki/page.md"))
            row = result.scalar_one_or_none()
            assert row is not None
            assert row.path == "/wiki/page.md"

            # Default table should be empty
            result = await session.execute(select(FileModel))
            assert len(result.scalars().all()) == 0

        await engine.dispose()

    async def test_upsert_default_model(self):
        """upsert_file without model param still uses default FileModel."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            await upsert_file(
                session,
                "sqlite",
                values={
                    "id": "default-1",
                    "path": "/hello.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()

            result = await session.execute(select(FileModel).where(FileModel.path == "/hello.txt"))
            assert result.scalar_one_or_none() is not None

        await engine.dispose()


# ---------------------------------------------------------------------------
# RustworkxGraph.from_sql with custom model
# ---------------------------------------------------------------------------


class TestGraphConnectionsOnly:
    """from_sql loads only connections — file_model param removed."""

    async def test_from_sql_loads_connections(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            session.add(
                FileConnectionModel(
                    source_path="/wiki/a.md",
                    target_path="/wiki/b.md",
                    type="links_to",
                    path="/wiki/a.md[links_to]/wiki/b.md",
                )
            )
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session)

            assert g.has_node("/wiki/a.md")
            assert g.has_node("/wiki/b.md")
            assert g.has_edge("/wiki/a.md", "/wiki/b.md")
            assert len(g.nodes) == 2

        await engine.dispose()

    async def test_from_sql_ignores_files_without_connections(self):
        """Files in DB with no connections should not appear in the graph."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            session.add(WikiFile(path="/wiki/active.md", parent_path="/wiki"))
            session.add(WikiFile(path="/wiki/lonely.md", parent_path="/wiki"))
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session)

            # No connections → empty graph
            assert len(g.nodes) == 0

        await engine.dispose()

    async def test_from_sql_empty_db(self):
        """from_sql on empty DB produces an empty graph."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            g = RustworkxGraph()
            await g.from_sql(session)
            assert len(g.nodes) == 0
            assert len(g.edges) == 0

        await engine.dispose()

    async def test_from_sql_uses_default_connection_model(self):
        """from_sql always uses FileConnectionModel (custom model param removed)."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Insert into default connection table
            session.add(
                FileConnectionModel(
                    source_path="/default/x.py",
                    target_path="/default/y.py",
                    type="imports",
                    path="/default/x.py[imports]/default/y.py",
                )
            )
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session)

            # Default table connections loaded
            assert g.has_node("/default/x.py")
            assert g.has_node("/default/y.py")
            assert g.has_edge("/default/x.py", "/default/y.py")
            assert len(g.nodes) == 2

        await engine.dispose()

    async def test_connections_with_custom_and_default_tables(self):
        """Connections are shared — from_sql sees all connections regardless of file model."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Add files to both tables
            session.add(FileModel(path="/default.py", parent_path="/"))
            session.add(WikiFile(path="/wiki/page.md", parent_path="/wiki"))
            # Connection references both
            session.add(
                FileConnectionModel(
                    source_path="/default.py",
                    target_path="/wiki/page.md",
                    type="links_to",
                    path="/default.py[links_to]/wiki/page.md",
                )
            )
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session)
            assert g.has_node("/default.py")
            assert g.has_node("/wiki/page.md")
            assert g.has_edge("/default.py", "/wiki/page.md")

        await engine.dispose()

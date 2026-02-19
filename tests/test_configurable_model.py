"""Tests for configurable file models — custom table names via FileBase/FileVersionBase."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

from grover.fs.database_fs import DatabaseFileSystem
from grover.fs.dialect import upsert_file
from grover.graph import RustworkxGraph
from grover.models.edges import GroverEdge
from grover.models.files import File, FileBase, FileVersionBase

# ---------------------------------------------------------------------------
# Custom model definitions (what a developer would write)
# ---------------------------------------------------------------------------


class WikiFile(FileBase, table=True):
    """Custom file model with a different table name."""

    __tablename__ = "wiki_files"


class WikiFileVersion(FileVersionBase, table=True):
    """Custom file version model with a different table name."""

    __tablename__ = "wiki_file_versions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_custom_fs() -> tuple[DatabaseFileSystem, object, object]:
    """Create a DatabaseFileSystem backed by custom models."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(
        dialect="sqlite",
        file_model=WikiFile,
        file_version_model=WikiFileVersion,
    )
    return fs, factory, engine


async def _make_default_fs() -> tuple[DatabaseFileSystem, object, object]:
    """Create a DatabaseFileSystem with default models."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fs = DatabaseFileSystem(dialect="sqlite")
    return fs, factory, engine


# ---------------------------------------------------------------------------
# Model inheritance tests
# ---------------------------------------------------------------------------


class TestModelInheritance:
    def test_file_base_is_not_a_table(self):
        """FileBase should not be a concrete table (no __table__)."""
        assert not hasattr(FileBase, "__table__")

    def test_file_version_base_is_not_a_table(self):
        assert not hasattr(FileVersionBase, "__table__")

    def test_custom_model_has_table_name(self):
        assert WikiFile.__tablename__ == "wiki_files"
        assert WikiFileVersion.__tablename__ == "wiki_file_versions"

    def test_custom_model_inherits_fields(self):
        """WikiFile should have all fields from FileBase."""
        wiki = WikiFile(path="/test.md", name="test.md", parent_path="/")
        assert wiki.path == "/test.md"
        assert wiki.name == "test.md"
        assert wiki.current_version == 1
        assert wiki.deleted_at is None
        assert wiki.mime_type == "text/plain"

    def test_custom_version_inherits_fields(self):
        wv = WikiFileVersion(file_id="abc", version=1, content="hello")
        assert wv.file_id == "abc"
        assert wv.version == 1
        assert wv.content == "hello"

    def test_default_file_still_works(self):
        """Default File model is unchanged."""
        assert File.__tablename__ == "grover_files"
        f = File(path="/hello.py", name="hello.py", parent_path="/")
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
        assert fs.file_model is File
        await engine.dispose()

    async def test_write_and_read(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            result = await fs.write("/wiki/page.md", "# Hello World\n", session=session)
            assert result.success is True
            assert result.created is True

            read = await fs.read("/wiki/page.md", session=session)
            assert read.success is True
            assert "Hello World" in read.content
        await engine.dispose()

    async def test_edit(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            await fs.write("/page.md", "old content\n", session=session)
            result = await fs.edit("/page.md", "old", "new", session=session)
            assert result.success is True
            assert result.version == 2

            read = await fs.read("/page.md", session=session)
            assert "new content" in read.content
        await engine.dispose()

    async def test_delete_and_trash(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            await fs.write("/page.md", "content\n", session=session)
            result = await fs.delete("/page.md", session=session)
            assert result.success is True

            read = await fs.read("/page.md", session=session)
            assert read.success is False

            trash = await fs.list_trash(session=session)
            assert len(trash.entries) == 1
        await engine.dispose()

    async def test_versioning(self):
        fs, factory, engine = await _make_custom_fs()
        async with factory() as session:
            await fs.write("/page.md", "v1\n", session=session)
            await fs.edit("/page.md", "v1", "v2", session=session)
            await fs.edit("/page.md", "v2", "v3", session=session)

            ver_result = await fs.list_versions("/page.md", session=session)
            assert len(ver_result.versions) >= 1

            vc_result = await fs.get_version_content("/page.md", 1, session=session)
            assert vc_result.success
            assert vc_result.content == "v1\n"
        await engine.dispose()

    async def test_data_written_to_custom_table(self):
        """Verify data actually ends up in the wiki_files table, not grover_files."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fs = DatabaseFileSystem(
            dialect="sqlite",
            file_model=WikiFile,
            file_version_model=WikiFileVersion,
        )
        async with factory() as session:
            await fs.write("/page.md", "wiki content\n", session=session)
            await session.commit()

        # Query the custom table directly
        async with factory() as session:
            result = await session.execute(select(WikiFile).where(WikiFile.path == "/page.md"))
            wiki_file = result.scalar_one_or_none()
            assert wiki_file is not None
            assert wiki_file.name == "page.md"

            # The default table should be empty
            result = await session.execute(select(File))
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
                    "name": "page.md",
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
            assert row.name == "page.md"

            # Default table should be empty
            result = await session.execute(select(File))
            assert len(result.scalars().all()) == 0

        await engine.dispose()

    async def test_upsert_default_model(self):
        """upsert_file without model param still uses default File."""
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
                    "name": "hello.txt",
                    "is_directory": False,
                    "current_version": 1,
                },
                conflict_keys=["path"],
            )
            await session.commit()

            result = await session.execute(select(File).where(File.path == "/hello.txt"))
            assert result.scalar_one_or_none() is not None

        await engine.dispose()


# ---------------------------------------------------------------------------
# RustworkxGraph.from_sql with custom model
# ---------------------------------------------------------------------------


class TestGraphWithCustomModel:
    async def test_from_sql_with_custom_model(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Insert into custom table
            session.add(WikiFile(path="/wiki/a.md", parent_path="/wiki", name="a.md"))
            session.add(WikiFile(path="/wiki/b.md", parent_path="/wiki", name="b.md"))
            session.add(
                GroverEdge(
                    source_path="/wiki/a.md",
                    target_path="/wiki/b.md",
                    type="links_to",
                )
            )
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session, file_model=WikiFile)

            assert g.has_node("/wiki/a.md")
            assert g.has_node("/wiki/b.md")
            assert g.has_edge("/wiki/a.md", "/wiki/b.md")
            assert g.node_count == 2

        await engine.dispose()

    async def test_from_sql_skips_deleted_custom(self):
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            session.add(WikiFile(path="/wiki/active.md", parent_path="/wiki", name="active.md"))
            session.add(
                WikiFile(
                    path="/wiki/deleted.md",
                    parent_path="/wiki",
                    name="deleted.md",
                    deleted_at=datetime.now(UTC),
                )
            )
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session, file_model=WikiFile)

            assert g.has_node("/wiki/active.md")
            assert not g.has_node("/wiki/deleted.md")

        await engine.dispose()

    async def test_from_sql_default_model(self):
        """from_sql without file_model param still uses default File."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            session.add(File(path="/a.py", parent_path="/", name="a.py"))
            await session.commit()

            g = RustworkxGraph()
            await g.from_sql(session)
            assert g.has_node("/a.py")

        await engine.dispose()

    async def test_custom_model_isolation(self):
        """Data in default File table should not appear when loading with WikiFile."""
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            # Add to default table
            session.add(File(path="/default.py", parent_path="/", name="default.py"))
            # Add to custom table
            session.add(WikiFile(path="/wiki/page.md", parent_path="/wiki", name="page.md"))
            await session.commit()

            # Load with custom model — should only see wiki_files
            g = RustworkxGraph()
            await g.from_sql(session, file_model=WikiFile)
            assert g.has_node("/wiki/page.md")
            assert not g.has_node("/default.py")

        await engine.dispose()

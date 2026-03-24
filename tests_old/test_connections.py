"""Tests for FileConnectionModel model — CRUD, defaults, table name, Base/Concrete subclassing."""

from __future__ import annotations

from sqlmodel import Session, SQLModel, select

from grover.models.database.connection import FileConnectionModel, FileConnectionModelBase


class TestFileConnectionModel:
    def test_table_exists(self, engine):
        """grover_file_connections table is created by create_all."""
        assert "grover_file_connections" in engine.dialect.get_table_names(engine.connect())

    def test_defaults(self, session: Session):
        conn = FileConnectionModel(
            source_path="/a.py",
            target_path="/b.py",
            path="/a.py[]/b.py",
        )
        session.add(conn)
        session.commit()
        session.refresh(conn)

        assert conn.id  # UUID string
        assert conn.source_path == "/a.py"
        assert conn.target_path == "/b.py"
        assert conn.path == "/a.py[]/b.py"
        assert conn.type == ""
        assert conn.weight == 1.0
        assert conn.created_at is not None

    def test_with_type_and_weight(self, session: Session):
        conn = FileConnectionModel(
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            weight=0.5,
            path="/a.py[imports]/b.py",
        )
        session.add(conn)
        session.commit()
        session.refresh(conn)

        assert conn.type == "imports"
        assert conn.weight == 0.5

    def test_round_trip(self, session: Session):
        conn = FileConnectionModel(
            source_path="/src/auth.py",
            target_path="/src/auth.py#login",
            type="contains",
            path="/src/auth.py[contains]/src/auth.py#login",
        )
        session.add(conn)
        session.commit()

        result = session.exec(
            select(FileConnectionModel).where(FileConnectionModel.source_path == "/src/auth.py")
        ).first()
        assert result is not None
        assert result.target_path == "/src/auth.py#login"
        assert result.type == "contains"

    def test_multiple_edges_same_source(self, session: Session):
        for target in ["/b.py", "/c.py", "/d.py"]:
            session.add(
                FileConnectionModel(
                    source_path="/a.py",
                    target_path=target,
                    type="imports",
                    path=f"/a.py[imports]{target}",
                )
            )
        session.commit()

        results = session.exec(select(FileConnectionModel).where(FileConnectionModel.source_path == "/a.py")).all()
        assert len(results) == 3

    def test_base_subclass_custom_table(self, engine):
        """Custom table name via subclassing FileConnectionModelBase."""

        class CustomConnection(FileConnectionModelBase, table=True):
            __tablename__ = "custom_connections"

        SQLModel.metadata.create_all(engine)
        tables = engine.dialect.get_table_names(engine.connect())
        assert "custom_connections" in tables

    def test_unique_ids(self, session: Session):
        conn1 = FileConnectionModel(source_path="/a.py", target_path="/b.py", path="/a.py[]/b.py")
        conn2 = FileConnectionModel(source_path="/a.py", target_path="/c.py", path="/a.py[]/c.py")
        session.add_all([conn1, conn2])
        session.commit()
        assert conn1.id != conn2.id

    def test_path_is_canonical_identity(self, session: Session):
        """path field stores the source[type]target canonical identity."""
        conn = FileConnectionModel(
            source_path="/x.py",
            target_path="/y.py",
            type="imports",
            path="/x.py[imports]/y.py",
        )
        session.add(conn)
        session.commit()
        session.refresh(conn)
        assert conn.path == "/x.py[imports]/y.py"

    def test_path_unique_constraint(self, session: Session):
        """Duplicate paths should be rejected."""
        import pytest
        from sqlalchemy.exc import IntegrityError

        conn1 = FileConnectionModel(source_path="/a.py", target_path="/b.py", path="/a.py[imports]/b.py")
        session.add(conn1)
        session.commit()

        conn2 = FileConnectionModel(source_path="/a.py", target_path="/b.py", path="/a.py[imports]/b.py")
        session.add(conn2)
        with pytest.raises(IntegrityError):
            session.commit()


class TestConnectionModelCreate:
    def test_create_with_mount(self):
        """mount= prepends /{mount}/ to both source and target paths."""
        conn = FileConnectionModelBase.create("/a.py", "/b.py", "imports", mount="project")
        assert conn.source_path == "/project/a.py"
        assert conn.target_path == "/project/b.py"
        assert conn.path == "/project/a.py[imports]/project/b.py"

    def test_create_with_mount_strips_slashes(self):
        """Leading/trailing slashes on mount are stripped."""
        conn = FileConnectionModelBase.create("/a.py", "/b.py", "imports", mount="/project/")
        assert conn.source_path == "/project/a.py"
        assert conn.target_path == "/project/b.py"

    def test_create_no_mount_unchanged(self):
        """Without mount, paths are unchanged (existing behavior)."""
        conn = FileConnectionModelBase.create("/a.py", "/b.py", "imports")
        assert conn.source_path == "/a.py"
        assert conn.target_path == "/b.py"
        assert conn.path == "/a.py[imports]/b.py"


class TestFileConnectionExports:
    def test_importable_from_models(self):
        from grover.models import FileConnectionModel, FileConnectionModelBase

        assert FileConnectionModel is not None
        assert FileConnectionModelBase is not None

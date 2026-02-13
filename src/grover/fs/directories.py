"""DirectoryService — parent dir creation and mkdir."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .dialect import upsert_file
from .utils import normalize_path, split_path, validate_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase

    GetFile = Callable[[AsyncSession, str, bool], Awaitable[FileBase | None]]


class DirectoryService:
    """Dialect-aware directory creation.

    Uses ``upsert_file`` from ``dialect.py`` to handle SQLite, PostgreSQL,
    and MSSQL upsert syntax differences.
    """

    def __init__(
        self,
        file_model: type[FileBase],
        dialect: str = "sqlite",
        schema: str | None = None,
    ) -> None:
        self._file_model = file_model
        self.dialect = dialect
        self.schema = schema

    async def ensure_parent_dirs(
        self,
        session: AsyncSession,
        path: str,
        owner_id: str | None = None,
    ) -> None:
        """Ensure all parent directories exist in the database."""
        parts = path.split("/")
        for i in range(2, len(parts)):
            dir_path = "/".join(parts[:i])
            if not dir_path:
                continue

            dir_name = parts[i - 1]
            parent = "/".join(parts[: i - 1]) or "/"
            values: dict[str, object] = {
                "id": str(uuid.uuid4()),
                "path": dir_path,
                "name": dir_name,
                "parent_path": parent,
                "is_directory": True,
                "current_version": 1,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            if owner_id is not None:
                values["owner_id"] = owner_id
            await upsert_file(
                session,
                self.dialect,
                values=values,
                conflict_keys=["path"],
                model=self._file_model,
                schema=self.schema,
                update_keys=["updated_at"],
            )

    async def mkdir(
        self,
        session: AsyncSession,
        path: str,
        parents: bool,
        get_file: GetFile,
        owner_id: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Create a directory using dialect-aware upsert.

        Returns ``(created_dirs, error_message)``.  On success,
        ``error_message`` is ``None``.
        """
        valid, error = validate_path(path)
        if not valid:
            return [], error

        path = normalize_path(path)

        existing = await get_file(session, path, False)
        if existing:
            if existing.is_directory:
                return [], None  # already exists, no error
            return [], f"Path exists as file: {path}"

        dirs_to_create: list[str] = []
        current = path

        while current != "/":
            existing = await get_file(session, current, False)
            if existing:
                if not existing.is_directory:
                    return [], f"Path exists as file: {current}"
                break
            dirs_to_create.insert(0, current)
            if not parents and len(dirs_to_create) > 1:
                return [], f"Parent directory does not exist: {split_path(current)[0]}"
            current = split_path(current)[0]

        created_dirs: list[str] = []
        for dir_path in dirs_to_create:
            parent, name = split_path(dir_path)
            values: dict[str, object] = {
                "id": str(uuid.uuid4()),
                "path": dir_path,
                "name": name,
                "parent_path": parent,
                "is_directory": True,
                "current_version": 1,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            if owner_id is not None:
                values["owner_id"] = owner_id
            rowcount = await upsert_file(
                session,
                self.dialect,
                values=values,
                conflict_keys=["path"],
                model=self._file_model,
                schema=self.schema,
            )
            if rowcount > 0:
                created_dirs.append(dir_path)

        await session.flush()
        return created_dirs, None

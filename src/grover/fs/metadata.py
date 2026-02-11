"""MetadataService — file lookup, info conversion, content hashing."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from sqlmodel import select

from .types import FileInfo, ReadResult
from .utils import normalize_path, validate_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.files import FileBase


def compute_content_hash(content: str) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for *content*."""
    encoded = content.encode()
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


class MetadataService:
    """Stateless helpers for file record lookup and conversion.

    Receives the concrete file model at construction so callers can
    use custom SQLModel subclasses.
    """

    def __init__(self, file_model: type[FileBase]) -> None:
        self._file_model = file_model

    async def get_file(
        self,
        session: AsyncSession,
        path: str,
        include_deleted: bool = False,
    ) -> FileBase | None:
        """Get a file record by path."""
        path = normalize_path(path)
        model = self._file_model
        query = select(model).where(
            model.path == path,
        )
        if not include_deleted:
            query = query.where(model.deleted_at.is_(None))  # type: ignore[unresolved-attribute]

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def exists(self, session: AsyncSession, path: str) -> bool:
        """Check if a file or directory exists."""
        valid, _ = validate_path(path)
        if not valid:
            return False

        path = normalize_path(path)
        if path == "/":
            return True

        file = await self.get_file(session, path)
        return file is not None

    async def get_info(self, session: AsyncSession, path: str) -> FileInfo | None:
        """Get metadata for a file or directory."""
        valid, _ = validate_path(path)
        if not valid:
            return None

        path = normalize_path(path)

        file = await self.get_file(session, path)
        if not file:
            return None
        return self.file_to_info(file)

    @staticmethod
    def file_to_info(f: FileBase) -> FileInfo:
        """Convert a file record to FileInfo."""
        return FileInfo(
            path=f.path,
            name=f.name,
            is_directory=f.is_directory,
            size_bytes=f.size_bytes,
            mime_type=f.mime_type,
            version=f.current_version,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )

    @staticmethod
    def paginate_content(
        content: str,
        path: str,
        offset: int,
        limit: int,
    ) -> ReadResult:
        """Paginate file content and return raw text."""
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines == 0 or (total_lines == 1 and lines[0] == ""):
            return ReadResult(
                success=True,
                message="File is empty.",
                content="",
                file_path=path,
                total_lines=0,
                lines_read=0,
                truncated=False,
                offset=offset,
            )

        end_line = min(len(lines), offset + limit)
        output_lines = lines[offset:end_line]

        last_read_line = offset + len(output_lines)
        has_more = total_lines > last_read_line

        return ReadResult(
            success=True,
            message=f"Read {len(output_lines)} lines from {path}",
            content="\n".join(output_lines),
            file_path=path,
            total_lines=total_lines,
            lines_read=len(output_lines),
            truncated=has_more,
            offset=offset,
        )

"""Compose/decompose helpers — convert between DB models and internal types.

These are pure mapping functions with no DB access or session management.
"""

from grover.models.database.chunk import FileChunkModel
from grover.models.database.connection import FileConnectionModel
from grover.models.database.file import FileModel, FileModelBase
from grover.models.database.vector import Vector
from grover.models.database.version import FileVersionModel
from grover.models.internal.ref import File, FileChunk, FileConnection, FileVersion

# =====================================================================
# DB model → internal type
# =====================================================================


def model_to_file(
    model: FileModel,
    *,
    chunks: list[FileChunkModel] | None = None,
    versions: list[FileVersionModel] | None = None,
) -> File:
    """Convert a FileModel (+ optional chunks/versions) to a File."""
    return File(
        path=model.path,
        content=model.content,
        embedding=list(model.embedding) if model.embedding is not None else None,
        tokens=model.tokens,
        lines=model.lines,
        current_version=model.current_version,
        chunks=[model_to_chunk(c) for c in (chunks or [])],
        versions=[model_to_version(v) for v in (versions or [])],
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def model_to_chunk(model: FileChunkModel) -> FileChunk:
    """Convert a FileChunkModel to a FileChunk."""
    return FileChunk(
        path=model.path or f"{model.file_path}#{model.id}",
        name=model.path.split("#", 1)[1] if "#" in (model.path or "") else "",
        content=model.content,
        embedding=list(model.embedding) if model.embedding is not None else None,
        tokens=model.tokens,
        line_start=model.line_start,
        line_end=model.line_end,
    )


def model_to_version(model: FileVersionModel) -> FileVersion:
    """Convert a FileVersionModel to a FileVersion."""
    return FileVersion(
        path=model.path,
        number=model.version,
        embedding=list(model.embedding) if model.embedding is not None else None,
        created_at=model.created_at,
    )


def model_to_connection(model: FileConnectionModel) -> FileConnection:
    """Convert a FileConnectionModel to a FileConnection."""
    return FileConnection(
        path=model.path,
        source_path=model.source_path,
        target_path=model.target_path,
        type=model.type,
        weight=model.weight,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


# =====================================================================
# Internal type → DB model
# =====================================================================


def file_to_model(file: File, model_cls: type[FileModelBase] | None = None) -> FileModelBase:
    """Convert a File to a FileModel.

    Sets basic fields only. Caller is responsible for setting ``id``,
    ``parent_path``, ``content_hash``, ``mime_type``, ``size_bytes``,
    ``owner_id``, and other DB-specific fields.
    """
    cls = model_cls or FileModel
    return cls(
        path=file.path,
        content=file.content,
        lines=file.lines,
        current_version=file.current_version,
        tokens=file.tokens,
        embedding=Vector(file.embedding) if file.embedding is not None else None,
    )


def chunk_to_model(chunk: FileChunk, file_path: str) -> FileChunkModel:
    """Convert a FileChunk to a FileChunkModel.

    ``file_path`` is the parent file's path (required by the DB schema).
    """
    return FileChunkModel(
        path=chunk.path,
        file_path=file_path,
        content=chunk.content,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        tokens=chunk.tokens,
        embedding=Vector(chunk.embedding) if chunk.embedding is not None else None,
    )


def connection_to_model(conn: FileConnection) -> FileConnectionModel:
    """Convert a FileConnection to a FileConnectionModel."""
    return FileConnectionModel(
        source_path=conn.source_path,
        target_path=conn.target_path,
        type=conn.type,
        weight=conn.weight,
        path=conn.path,
    )

"""FileOpsMixin — file CRUD operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.events import EventType, FileEvent
from grover.fs.exceptions import MountNotFoundError
from grover.fs.protocol import SupportsTrash
from grover.fs.utils import normalize_path
from grover.types import (
    DeleteResult,
    EditResult,
    FileInfoResult,
    FileSearchCandidate,
    ListDirEvidence,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    WriteResult,
)

if TYPE_CHECKING:
    from grover.facade.context import GroverContext


class FileOpsMixin:
    """File CRUD operations extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # FS Operations (absorbed from VFS)
    # ------------------------------------------------------------------

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        user_id: str | None = None,
    ) -> ReadResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.read(
                rel_path, offset, limit, session=sess, user_id=user_id
            )
        result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
        return result

    async def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> WriteResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return WriteResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.write(
                    rel_path,
                    content,
                    "agent",
                    overwrite=overwrite,
                    session=sess,
                    user_id=user_id,
                )
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(
                        event_type=EventType.FILE_WRITTEN,
                        path=path,
                        content=content,
                        user_id=user_id,
                    )
                )
            return result
        except Exception as e:
            return WriteResult(success=False, message=f"Write failed: {e}")

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> EditResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return EditResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.edit(
                    rel_path,
                    old,
                    new,
                    replace_all,
                    "agent",
                    session=sess,
                    user_id=user_id,
                )
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=path, user_id=user_id)
                )
            return result
        except Exception as e:
            return EditResult(success=False, message=f"Edit failed: {e}")

    async def delete(
        self, path: str, permanent: bool = False, *, user_id: str | None = None
    ) -> DeleteResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return DeleteResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None

            if not permanent and not self._ctx.get_capability(mount.filesystem, SupportsTrash):
                return DeleteResult(
                    success=False,
                    message="Trash not supported on this mount. "
                    "Use permanent=True to delete permanently.",
                )

            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.delete(
                    rel_path, permanent, session=sess, user_id=user_id
                )
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(event_type=EventType.FILE_DELETED, path=path, user_id=user_id)
                )
            return result
        except Exception as e:
            return DeleteResult(success=False, message=f"Delete failed: {e}")

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        user_id: str | None = None,
    ) -> MkdirResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return MkdirResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.mkdir(rel_path, parents, session=sess, user_id=user_id)
        result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
        result.created_dirs = [
            self._ctx.prefix_path(d, mount.path) or d for d in result.created_dirs
        ]
        return result

    async def list_dir(self, path: str = "/", *, user_id: str | None = None) -> ListDirResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.list_dir(rel_path, session=sess, user_id=user_id)
        return result.rebase(mount.path)

    def _list_root(self) -> ListDirResult:
        candidates = [
            FileSearchCandidate(
                path=mount.path,
                evidence=[
                    ListDirEvidence(
                        strategy="list_dir",
                        path=mount.path,
                        is_directory=True,
                    )
                ],
            )
            for mount in self._ctx.registry.list_visible_mounts()
        ]
        return ListDirResult(
            success=True,
            message=f"Found {len(candidates)} mount(s)",
            candidates=candidates,
        )

    async def exists(self, path: str, *, user_id: str | None = None) -> bool:
        path = normalize_path(path)

        if path == "/":
            return True

        if self._ctx.registry.has_mount(path):
            return True

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return False

        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileInfoResult | None:
        path = normalize_path(path)

        if self._ctx.registry.has_mount(path):
            for mount in self._ctx.registry.list_mounts():
                if mount.path == path:
                    name = mount.path.lstrip("/")
                    return FileInfoResult(
                        path=mount.path,
                        name=name,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return None

        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            info = await mount.filesystem.get_info(rel_path, session=sess, user_id=user_id)
        if info is not None:
            info = self._ctx.prefix_file_info(info, mount)
        return info

    def get_permission_info(self, path: str) -> tuple[str, bool]:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        permission = self._ctx.registry.get_permission(path)
        rel_normalized = normalize_path(rel_path)
        is_override = rel_normalized in mount.read_only_paths
        return permission.value, is_override

    async def move(
        self, src: str, dest: str, *, user_id: str | None = None, follow: bool = False
    ) -> MoveResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        if err := self._ctx.check_writable(src):
            return MoveResult(success=False, message=err)
        if err := self._ctx.check_writable(dest):
            return MoveResult(success=False, message=err)

        try:
            src_mount, src_rel = self._ctx.registry.resolve(src)
            dest_mount, dest_rel = self._ctx.registry.resolve(dest)

            assert src_mount.filesystem is not None
            assert dest_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._ctx.session_for(src_mount) as sess:
                    result = await src_mount.filesystem.move(
                        src_rel, dest_rel, session=sess, follow=follow, user_id=user_id
                    )
                result.old_path = (
                    self._ctx.prefix_path(result.old_path, src_mount.path) or result.old_path
                )
                result.new_path = (
                    self._ctx.prefix_path(result.new_path, dest_mount.path) or result.new_path
                )
                if result.success:
                    await self._ctx.emit(
                        FileEvent(
                            event_type=EventType.FILE_MOVED,
                            path=dest,
                            old_path=src,
                            user_id=user_id,
                        )
                    )
                return result

            # Cross-mount move: read → write → delete (non-atomic)
            async with self._ctx.session_for(src_mount) as src_sess:
                read_result = await src_mount.filesystem.read(
                    src_rel, session=src_sess, user_id=user_id
                )
            if not read_result.success:
                return MoveResult(
                    success=False,
                    message=f"Cannot read source for cross-mount move: {read_result.message}",
                )
            if read_result.content is None:
                return MoveResult(success=False, message=f"Source file has no content: {src}")

            async with self._ctx.session_for(dest_mount) as dest_sess:
                write_result = await dest_mount.filesystem.write(
                    dest_rel, read_result.content, session=dest_sess, user_id=user_id
                )
            if not write_result.success:
                return MoveResult(
                    success=False,
                    message=(
                        f"Cannot write to destination for cross-mount move: {write_result.message}"
                    ),
                )

            async with self._ctx.session_for(src_mount) as src_sess:
                delete_result = await src_mount.filesystem.delete(
                    src_rel, permanent=False, session=src_sess, user_id=user_id
                )
            if not delete_result.success:
                return MoveResult(
                    success=False,
                    message=f"Copied but failed to delete source: {delete_result.message}",
                )

            await self._ctx.emit(
                FileEvent(event_type=EventType.FILE_MOVED, path=dest, old_path=src, user_id=user_id)
            )
            return MoveResult(
                success=True,
                message=f"Moved {src} -> {dest} (cross-mount)",
                old_path=src,
                new_path=dest,
            )
        except Exception as e:
            return MoveResult(success=False, message=f"Move failed: {e}")

    async def copy(self, src: str, dest: str, *, user_id: str | None = None) -> WriteResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        if err := self._ctx.check_writable(dest):
            return WriteResult(success=False, message=err)

        try:
            src_mount, src_rel = self._ctx.registry.resolve(src)
            dest_mount, dest_rel = self._ctx.registry.resolve(dest)

            assert src_mount.filesystem is not None
            assert dest_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._ctx.session_for(src_mount) as sess:
                    result = await src_mount.filesystem.copy(
                        src_rel, dest_rel, session=sess, user_id=user_id
                    )
                result.path = self._ctx.prefix_path(result.path, dest_mount.path) or result.path
                if result.success:
                    await self._ctx.emit(
                        FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                    )
                return result

            # Cross-mount copy: read → write
            async with self._ctx.session_for(src_mount) as src_sess:
                read_result = await src_mount.filesystem.read(
                    src_rel, session=src_sess, user_id=user_id
                )
            if not read_result.success:
                return WriteResult(
                    success=False,
                    message=f"Cannot read source for cross-mount copy: {read_result.message}",
                )
            if not read_result.content:
                return WriteResult(success=False, message=f"Source file has no content: {src}")

            async with self._ctx.session_for(dest_mount) as dest_sess:
                result = await dest_mount.filesystem.write(
                    dest_rel, read_result.content, session=dest_sess, user_id=user_id
                )
            result.path = self._ctx.prefix_path(result.path, dest_mount.path) or result.path
            if result.success:
                await self._ctx.emit(
                    FileEvent(event_type=EventType.FILE_WRITTEN, path=dest, user_id=user_id)
                )
            return result
        except Exception as e:
            return WriteResult(success=False, message=f"Copy failed: {e}")

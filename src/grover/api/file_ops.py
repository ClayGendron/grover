"""FileOpsMixin — file CRUD, version, trash, and reconciliation operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.backends.protocol import SupportsReconcile
from grover.exceptions import MountNotFoundError
from grover.permissions import Permission
from grover.results import (
    DeleteResult,
    DiffVersionsResult,
    EditResult,
    ExistsResult,
    FileCandidate,
    FileInfoResult,
    GetVersionContentResult,
    ListDirEvidence,
    ListDirResult,
    MkdirResult,
    MoveResult,
    ReadResult,
    ReconcileResult,
    RestoreResult,
    TrashResult,
    TreeEvidence,
    TreeResult,
    VersionResult,
    WriteResult,
)
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext


class FileOpsMixin:
    """File CRUD, version, trash, and reconciliation operations extracted from GroverAsync."""

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
                self._ctx.worker.schedule(
                    path,
                    lambda p=path, c=content, u=user_id: self._process_write(p, c, u),  # type: ignore[attr-defined]
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
                self._ctx.worker.schedule(
                    path,
                    lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
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

            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.delete(
                    rel_path, permanent, session=sess, user_id=user_id
                )
            result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
            if result.success:
                self._ctx.worker.cancel(path)
                self._ctx.worker.schedule_immediate(self._process_delete(path, user_id))  # type: ignore[attr-defined]
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
            FileCandidate(
                path=mount.path,
                evidence=[
                    ListDirEvidence(
                        operation="list_dir",
                        is_directory=True,
                    )
                ],
            )
            for mount in self._ctx.registry.list_visible_mounts()
        ]
        return ListDirResult(
            success=True,
            message=f"Found {len(candidates)} mount(s)",
            file_candidates=candidates,
        )

    async def exists(self, path: str, *, user_id: str | None = None) -> ExistsResult:
        path = normalize_path(path)

        if path == "/":
            return ExistsResult(exists=True, path=path)

        if self._ctx.registry.has_mount(path):
            return ExistsResult(exists=True, path=path)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return ExistsResult(exists=False, path=path)

        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileInfoResult:
        path = normalize_path(path)

        if self._ctx.registry.has_mount(path):
            for mount in self._ctx.registry.list_mounts():
                if mount.path == path:
                    return FileInfoResult(
                        path=mount.path,
                        is_directory=True,
                        permission=mount.permission.value,
                        mount_type=mount.mount_type,
                    )

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return FileInfoResult(
                success=False, message=f"No mount found for path: {path}", path=path
            )

        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            info = await mount.filesystem.get_info(rel_path, session=sess, user_id=user_id)
        if info.success:
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
                    self._ctx.worker.cancel(src)
                    self._ctx.worker.schedule_immediate(self._process_move(src, dest, user_id))  # type: ignore[attr-defined]
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

            self._ctx.worker.cancel(src)
            self._ctx.worker.schedule_immediate(self._process_move(src, dest, user_id))  # type: ignore[attr-defined]
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
                    self._ctx.worker.schedule(
                        dest,
                        lambda p=dest, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
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
                self._ctx.worker.schedule(
                    dest,
                    lambda p=dest, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
                )
            return result
        except Exception as e:
            return WriteResult(success=False, message=f"Copy failed: {e}")

    # ------------------------------------------------------------------
    # Tree (moved from SearchOpsMixin)
    # ------------------------------------------------------------------

    async def tree(
        self, path: str = "/", *, max_depth: int | None = None, user_id: str | None = None
    ) -> TreeResult:
        path = normalize_path(path)
        try:
            if path == "/":
                root_candidates = [
                    FileCandidate(
                        path=mount.path,
                        evidence=[
                            TreeEvidence(
                                operation="tree",
                                depth=0,
                                is_directory=True,
                            )
                        ],
                    )
                    for mount in self._ctx.registry.list_visible_mounts()
                ]
                combined = TreeResult(success=True, message="", file_candidates=root_candidates)

                if max_depth is None or max_depth > 0:
                    for mount in self._ctx.registry.list_visible_mounts():
                        assert mount.filesystem is not None
                        async with self._ctx.session_for(mount) as sess:
                            result = await mount.filesystem.tree(
                                "/", max_depth=max_depth, session=sess, user_id=user_id
                            )
                        if result.success:
                            combined = combined | result.rebase(mount.path)

                combined.message = (
                    f"{combined.total_dirs} directories, {combined.total_files} files"
                )
                return combined

            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.tree(
                    rel_path, max_depth=max_depth, session=sess, user_id=user_id
                )
            return result.rebase(mount.path)
        except Exception as e:
            return TreeResult(success=False, message=f"Tree failed: {e}")

    # ------------------------------------------------------------------
    # Version operations (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> VersionResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.list_versions(rel_path, session=sess, user_id=user_id)

    async def read_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> GetVersionContentResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.get_version_content(
                rel_path, version, session=sess, user_id=user_id
            )

    async def diff_versions(
        self, path: str, version_a: int, version_b: int, *, user_id: str | None = None
    ) -> DiffVersionsResult:
        from grover.providers.versioning.diff import compute_diff

        path = normalize_path(path)
        result_a = await self.read_version(path, version_a, user_id=user_id)
        if not result_a.success:
            return DiffVersionsResult(
                success=False,
                message=f"Cannot read version {version_a}: {result_a.message}",
                path=path,
                version_a=version_a,
                version_b=version_b,
            )
        result_b = await self.read_version(path, version_b, user_id=user_id)
        if not result_b.success:
            return DiffVersionsResult(
                success=False,
                message=f"Cannot read version {version_b}: {result_b.message}",
                path=path,
                version_a=version_a,
                version_b=version_b,
            )
        diff = compute_diff(result_a.content, result_b.content)
        return DiffVersionsResult(
            success=True,
            message=f"Diff between v{version_a} and v{version_b}",
            path=path,
            version_a=version_a,
            version_b=version_b,
            diff=diff,
            content_a=result_a.content,
            content_b=result_b.content,
        )

    async def restore_version(
        self, path: str, version: int, *, user_id: str | None = None
    ) -> RestoreResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return RestoreResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.restore_version(
                rel_path, version, session=sess, user_id=user_id
            )
        result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
        if result.success:
            self._ctx.worker.schedule(
                path,
                lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
            )
        return result

    # ------------------------------------------------------------------
    # Trash operations (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> TrashResult:
        """List all items in trash across all mounts."""
        combined = TrashResult(success=True, message="")
        for mount in self._ctx.registry.list_mounts():
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.list_trash(session=sess, user_id=user_id)
            if result.success:
                rebased = result.rebase(mount.path)
                combined = combined | rebased
        combined.message = f"Found {len(combined)} item(s) in trash"
        return combined

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> RestoreResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return RestoreResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.restore_from_trash(
                rel_path, session=sess, user_id=user_id
            )
        result.path = self._ctx.prefix_path(result.path, mount.path) or result.path
        if result.success:
            self._ctx.worker.schedule(
                path,
                lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
            )
        return result

    async def empty_trash(self, *, user_id: str | None = None) -> DeleteResult:
        """Empty trash across all mounts.  Skips read-only mounts."""
        total_deleted = 0
        mounts_processed = 0
        for mount in self._ctx.registry.list_mounts():
            if mount.permission == Permission.READ_ONLY:
                continue
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.empty_trash(session=sess, user_id=user_id)
            if not result.success:
                return result
            total_deleted += result.total_deleted or 0
            mounts_processed += 1
        return DeleteResult(
            success=True,
            message=f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)",
            total_deleted=total_deleted,
            permanent=True,
        )

    # ------------------------------------------------------------------
    # Reconciliation (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> ReconcileResult:
        """Reconcile disk ↔ DB for capable mounts."""
        total = ReconcileResult()
        mounts = self._ctx.registry.list_mounts()
        if mount_path is not None:
            mount_path = normalize_path(mount_path).rstrip("/")
            mounts = [m for m in mounts if m.path == mount_path]

        for mount in mounts:
            if mount.permission == Permission.READ_ONLY:
                continue
            cap = self._ctx.get_capability(mount.filesystem, SupportsReconcile)
            if cap is None:
                continue
            async with self._ctx.session_for(mount) as sess:
                stats = await cap.reconcile(session=sess)
            total.created += stats.created
            total.updated += stats.updated
            total.deleted += stats.deleted
            total.chain_errors += stats.chain_errors

        return total

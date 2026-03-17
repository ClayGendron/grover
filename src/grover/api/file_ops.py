"""FileOpsMixin — file CRUD, version, trash, and reconciliation operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.backends.protocol import SupportsReconcile
from grover.exceptions import MountNotFoundError
from grover.models.internal.evidence import ListDirEvidence, TreeEvidence
from grover.models.internal.ref import File
from grover.models.internal.results import BatchResult, FileOperationResult, FileSearchResult, FileSearchSet
from grover.permissions import Permission
from grover.ref import Ref
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext
    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.file import FileModelBase


class FileOpsMixin:
    """File CRUD, version, trash, and reconciliation operations extracted from GroverAsync."""

    _ctx: GroverContext

    @staticmethod
    def _split_candidates(candidates: FileSearchSet | None, mount_path: str) -> FileSearchSet | None:
        """Strip mount prefix from candidate paths belonging to this mount."""
        if candidates is None:
            return None
        paths = [p.removeprefix(mount_path) or "/" for p in candidates.paths if p.startswith(mount_path)]
        return FileSearchSet.from_paths(paths) if paths else FileSearchSet()

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
    ) -> FileOperationResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.read(rel_path, offset, limit, session=sess, user_id=user_id)
        result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
        return result

    async def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

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
            result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
            if result.success:
                self._ctx.worker.schedule(
                    path,
                    lambda p=path, c=content, u=user_id: self._process_write(p, c, u),  # type: ignore[attr-defined]
                )
            return result
        except Exception as e:
            return FileOperationResult(success=False, message=f"Write failed: {e}")

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

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
            result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
            if result.success:
                self._ctx.worker.schedule(
                    path,
                    lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
                )
            return result
        except Exception as e:
            return FileOperationResult(success=False, message=f"Edit failed: {e}")

    async def delete(self, path: str, permanent: bool = False, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None

            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.delete(rel_path, permanent, session=sess, user_id=user_id)
            result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
            if result.success:
                self._ctx.worker.cancel(path)
                self._ctx.worker.schedule_immediate(self._process_delete(path, user_id))  # type: ignore[attr-defined]
            return result
        except Exception as e:
            return FileOperationResult(success=False, message=f"Delete failed: {e}")

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        user_id: str | None = None,
    ) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.mkdir(rel_path, parents, session=sess, user_id=user_id)
        result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
        return result

    async def list_dir(
        self, path: str = "/", *, candidates: FileSearchSet | None = None, user_id: str | None = None
    ) -> FileSearchResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return FileSearchResult(success=False, message=f"No mount found for path: {path}")

        assert mount.filesystem is not None
        mount_candidates = self._split_candidates(candidates, mount.path)
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.list_dir(
                rel_path, candidates=mount_candidates, session=sess, user_id=user_id
            )
        if result.success:
            result = result.rebase(mount.path)
        return result

    def _list_root(self) -> FileSearchResult:
        files = [
            File(
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
        return FileSearchResult(
            success=True,
            message=f"Found {len(files)} mount(s)",
            files=files,
        )

    async def exists(self, path: str, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)

        if path == "/":
            return FileOperationResult(file=File(path=path), success=True)

        if self._ctx.registry.has_mount(path):
            return FileOperationResult(file=File(path=path), success=True)

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return FileOperationResult(file=File(path=path), success=False)

        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.exists(rel_path, session=sess, user_id=user_id)

    async def get_info(self, path: str, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)

        if self._ctx.registry.has_mount(path):
            for mount in self._ctx.registry.list_mounts():
                if mount.path == path:
                    return FileOperationResult(
                        file=File(path=mount.path, is_directory=True),
                        success=True,
                    )

        try:
            mount, rel_path = self._ctx.registry.resolve(path)
        except MountNotFoundError:
            return FileOperationResult(
                success=False,
                message=f"No mount found for path: {path}",
                file=File(path=path),
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
    ) -> FileOperationResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        if err := self._ctx.check_writable(src):
            return FileOperationResult(success=False, message=err)
        if err := self._ctx.check_writable(dest):
            return FileOperationResult(success=False, message=err)

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
                if result.success:
                    self._ctx.worker.cancel(src)
                    self._ctx.worker.schedule_immediate(self._process_move(src, dest, user_id))  # type: ignore[attr-defined]
                return result

            # Cross-mount move: read → write → delete (non-atomic)
            async with self._ctx.session_for(src_mount) as src_sess:
                read_result = await src_mount.filesystem.read(src_rel, session=src_sess, user_id=user_id)
            if not read_result.success:
                return FileOperationResult(
                    success=False,
                    message=f"Cannot read source for cross-mount move: {read_result.message}",
                )
            if read_result.file.content is None:
                return FileOperationResult(success=False, message=f"Source file has no content: {src}")

            async with self._ctx.session_for(dest_mount) as dest_sess:
                write_result = await dest_mount.filesystem.write(
                    dest_rel, read_result.file.content, session=dest_sess, user_id=user_id
                )
            if not write_result.success:
                return FileOperationResult(
                    success=False,
                    message=(f"Cannot write to destination for cross-mount move: {write_result.message}"),
                )

            async with self._ctx.session_for(src_mount) as src_sess:
                delete_result = await src_mount.filesystem.delete(
                    src_rel, permanent=False, session=src_sess, user_id=user_id
                )
            if not delete_result.success:
                return FileOperationResult(
                    success=False,
                    message=f"Copied but failed to delete source: {delete_result.message}",
                )

            self._ctx.worker.cancel(src)
            self._ctx.worker.schedule_immediate(self._process_move(src, dest, user_id))  # type: ignore[attr-defined]
            return FileOperationResult(
                success=True,
                message=f"Moved {src} -> {dest} (cross-mount)",
            )
        except Exception as e:
            return FileOperationResult(success=False, message=f"Move failed: {e}")

    async def copy(self, src: str, dest: str, *, user_id: str | None = None) -> FileOperationResult:
        src = normalize_path(src)
        dest = normalize_path(dest)

        if err := self._ctx.check_writable(dest):
            return FileOperationResult(success=False, message=err)

        try:
            src_mount, src_rel = self._ctx.registry.resolve(src)
            dest_mount, dest_rel = self._ctx.registry.resolve(dest)

            assert src_mount.filesystem is not None
            assert dest_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._ctx.session_for(src_mount) as sess:
                    result = await src_mount.filesystem.copy(src_rel, dest_rel, session=sess, user_id=user_id)
                result.file.path = self._ctx.prefix_path(result.file.path, dest_mount.path) or result.file.path
                if result.success:
                    self._ctx.worker.schedule(
                        dest,
                        lambda p=dest, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
                    )
                return result

            # Cross-mount copy: read → write
            async with self._ctx.session_for(src_mount) as src_sess:
                read_result = await src_mount.filesystem.read(src_rel, session=src_sess, user_id=user_id)
            if not read_result.success:
                return FileOperationResult(
                    success=False,
                    message=f"Cannot read source for cross-mount copy: {read_result.message}",
                )
            if not read_result.file.content:
                return FileOperationResult(success=False, message=f"Source file has no content: {src}")

            async with self._ctx.session_for(dest_mount) as dest_sess:
                result = await dest_mount.filesystem.write(
                    dest_rel, read_result.file.content, session=dest_sess, user_id=user_id
                )
            result.file.path = self._ctx.prefix_path(result.file.path, dest_mount.path) or result.file.path
            if result.success:
                self._ctx.worker.schedule(
                    dest,
                    lambda p=dest, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
                )
            return result
        except Exception as e:
            return FileOperationResult(success=False, message=f"Copy failed: {e}")

    # ------------------------------------------------------------------
    # Tree (moved from SearchOpsMixin)
    # ------------------------------------------------------------------

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        path = normalize_path(path)
        try:
            if path == "/":
                root_files = [
                    File(
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
                combined = FileSearchResult(success=True, message="", files=root_files)

                if max_depth is None or max_depth > 0:
                    for mount in self._ctx.registry.list_visible_mounts():
                        assert mount.filesystem is not None
                        mount_candidates = self._split_candidates(candidates, mount.path)
                        async with self._ctx.session_for(mount) as sess:
                            result = await mount.filesystem.tree(
                                "/",
                                max_depth=max_depth,
                                candidates=mount_candidates,
                                session=sess,
                                user_id=user_id,
                            )
                        if result.success:
                            combined = combined | result.rebase(mount.path)

                total_dirs = sum(
                    1 for f in combined.files if any(isinstance(e, TreeEvidence) and e.is_directory for e in f.evidence)
                )
                total_files_count = sum(
                    1
                    for f in combined.files
                    if any(isinstance(e, TreeEvidence) and not e.is_directory for e in f.evidence)
                )
                combined.message = f"{total_dirs} directories, {total_files_count} files"
                return combined

            mount, rel_path = self._ctx.registry.resolve(path)
            assert mount.filesystem is not None
            mount_candidates = self._split_candidates(candidates, mount.path)
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.tree(
                    rel_path,
                    max_depth=max_depth,
                    candidates=mount_candidates,
                    session=sess,
                    user_id=user_id,
                )
            if result.success:
                result = result.rebase(mount.path)

            total_dirs = sum(
                1 for f in result.files if any(isinstance(e, TreeEvidence) and e.is_directory for e in f.evidence)
            )
            total_files_count = sum(
                1 for f in result.files if any(isinstance(e, TreeEvidence) and not e.is_directory for e in f.evidence)
            )
            result.message = f"{total_dirs} directories, {total_files_count} files"
            return result
        except Exception as e:
            return FileSearchResult(success=False, message=f"Tree failed: {e}")

    # ------------------------------------------------------------------
    # Version operations (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def list_versions(self, path: str, *, user_id: str | None = None) -> FileSearchResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.list_versions(rel_path, session=sess, user_id=user_id)
        return result

    async def read_version(self, path: str, version: int, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)
        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            return await mount.filesystem.get_version_content(rel_path, version, session=sess, user_id=user_id)

    async def diff_versions(
        self, path: str, version_a: int, version_b: int, *, user_id: str | None = None
    ) -> FileOperationResult:
        from grover.providers.versioning.diff import compute_diff

        path = normalize_path(path)
        result_a = await self.read_version(path, version_a, user_id=user_id)
        if not result_a.success:
            return FileOperationResult(
                success=False,
                message=f"Cannot read version {version_a}: {result_a.message}",
                file=File(path=path),
            )
        result_b = await self.read_version(path, version_b, user_id=user_id)
        if not result_b.success:
            return FileOperationResult(
                success=False,
                message=f"Cannot read version {version_b}: {result_b.message}",
                file=File(path=path),
            )
        diff = compute_diff(result_a.file.content or "", result_b.file.content or "")
        return FileOperationResult(
            success=True,
            message=f"Diff between v{version_a} and v{version_b}",
            file=File(path=path, content=diff),
        )

    async def restore_version(self, path: str, version: int, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.restore_version(rel_path, version, session=sess, user_id=user_id)
        result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
        if result.success:
            self._ctx.worker.schedule(
                path,
                lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
            )
        return result

    # ------------------------------------------------------------------
    # Trash operations (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def list_trash(self, *, user_id: str | None = None) -> FileSearchResult:
        """List all items in trash across all mounts."""
        combined = FileSearchResult(success=True, message="")
        for mount in self._ctx.registry.list_mounts():
            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                result = await mount.filesystem.list_trash(session=sess, user_id=user_id)
            if result.success:
                rebased = result.rebase(mount.path)
                combined = combined | rebased
        combined.message = f"Found {len(combined)} item(s) in trash"
        return combined

    async def restore_from_trash(self, path: str, *, user_id: str | None = None) -> FileOperationResult:
        path = normalize_path(path)
        if err := self._ctx.check_writable(path):
            return FileOperationResult(success=False, message=err)

        mount, rel_path = self._ctx.registry.resolve(path)
        assert mount.filesystem is not None
        async with self._ctx.session_for(mount) as sess:
            result = await mount.filesystem.restore_from_trash(rel_path, session=sess, user_id=user_id)
        result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
        if result.success:
            self._ctx.worker.schedule(
                path,
                lambda p=path, u=user_id: self._process_write(p, None, u),  # type: ignore[attr-defined]
            )
        return result

    async def empty_trash(self, *, user_id: str | None = None) -> FileOperationResult:
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
            mounts_processed += 1
        return FileOperationResult(
            success=True,
            message=(f"Permanently deleted {total_deleted} file(s) from {mounts_processed} mount(s)"),
        )

    # ------------------------------------------------------------------
    # Reconciliation (absorbed from VersionTrashMixin)
    # ------------------------------------------------------------------

    async def reconcile(self, mount_path: str | None = None) -> FileOperationResult:
        """Reconcile disk <-> DB for capable mounts."""
        total = FileOperationResult()
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
                await cap.reconcile(session=sess)

        return total

    # ------------------------------------------------------------------
    # Chunk write operations
    # ------------------------------------------------------------------

    async def write_chunk(
        self,
        chunk: FileChunkModelBase,
        *,
        user_id: str | None = None,
    ) -> FileOperationResult:
        """Write (upsert) a single chunk. Parent file must exist."""
        result = await self.write_chunks([chunk], user_id=user_id)
        if result.results:
            return result.results[0]
        return FileOperationResult(success=result.success, message=result.message, file=File(path=chunk.path))

    async def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        user_id: str | None = None,
    ) -> BatchResult:
        """Batch write (upsert) chunks. Parent files must exist."""
        if not chunks:
            return BatchResult(success=True, message="No chunks to write")

        # Validate all chunk refs upfront
        for chunk in chunks:
            ref = Ref(chunk.path)
            if not ref.is_chunk:
                return BatchResult(
                    success=False,
                    message=f"Invalid chunk ref (must contain '#'): {chunk.path}",
                )
            if chunk.file_path != ref.base_path:
                return BatchResult(
                    success=False,
                    message=(f"file_path mismatch: {chunk.file_path} != {ref.base_path} for chunk {chunk.path}"),
                )

        # Track results by original index to preserve input order
        results_by_idx: dict[int, FileOperationResult] = {}

        # Group chunks by mount, keeping original indices
        from collections import defaultdict

        mount_groups: dict[str, list[tuple[int, FileChunkModelBase]]] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            path = normalize_path(chunk.file_path)
            mount, _rel = self._ctx.registry.resolve(path)
            mount_groups[mount.path].append((idx, chunk))

        for mount_path, group in mount_groups.items():
            mount, _ = self._ctx.registry.resolve(mount_path + "/dummy")

            # Check writable
            if err := self._ctx.check_writable(mount_path + "/dummy"):
                for idx, c in group:
                    results_by_idx[idx] = FileOperationResult(success=False, message=err, file=File(path=c.path))
                continue

            assert mount.filesystem is not None
            async with self._ctx.session_for(mount) as sess:
                # Validate parent files exist (one exists() per unique parent)
                unique_parents = {normalize_path(c.file_path) for _, c in group}
                parent_exists: dict[str, bool] = {}
                for parent in unique_parents:
                    _, rel_parent = self._ctx.registry.resolve(parent)
                    ex = await mount.filesystem.exists(rel_parent, session=sess)
                    parent_exists[parent] = ex.message == "exists"

                valid_items: list[tuple[int, FileChunkModelBase]] = []
                for idx, c in group:
                    if not parent_exists.get(normalize_path(c.file_path), False):
                        results_by_idx[idx] = FileOperationResult(
                            success=False,
                            message=f"Parent file not found: {c.file_path}",
                            file=File(path=c.path),
                        )
                    else:
                        valid_items.append((idx, c))

                if valid_items:
                    # Strip mount prefix from paths for backend
                    backend_chunks = []
                    idx_order: list[int] = []
                    for idx, c in valid_items:
                        rel_file = normalize_path(c.file_path).removeprefix(mount.path) or "/"
                        rel_chunk_path = normalize_path(c.path).removeprefix(mount.path) or "/"
                        bc = type(c).model_validate(
                            {
                                "file_path": rel_file,
                                "path": rel_chunk_path,
                                "content": c.content,
                                "line_start": c.line_start,
                                "line_end": c.line_end,
                            }
                        )
                        backend_chunks.append(bc)
                        idx_order.append(idx)

                    batch_result = await mount.filesystem.write_chunks(backend_chunks, session=sess)
                    # Re-prefix paths and map back to original indices
                    batch_results: list[FileOperationResult] = getattr(batch_result, "results", [])
                    for i, r in enumerate(batch_results):
                        r.file.path = self._ctx.prefix_path(r.file.path, mount.path) or r.file.path
                        results_by_idx[idx_order[i]] = r

        # Build ordered results list
        all_results = [results_by_idx[i] for i in range(len(chunks))]

        # Schedule background processing for successful chunks
        for chunk, result in zip(chunks, all_results, strict=True):
            if result.success:
                self._ctx.worker.schedule_immediate(
                    self._process_chunk_write(chunk)  # type: ignore[attr-defined]
                )

        succeeded = sum(1 for r in all_results if r.success)
        failed = len(all_results) - succeeded
        return BatchResult(
            success=failed == 0,
            message=f"Wrote {succeeded} chunk(s)" + (f", {failed} failed" if failed else ""),
            results=all_results,
            succeeded=succeeded,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # File write from model (write_file / write_files)
    # ------------------------------------------------------------------

    async def write_file(
        self,
        file: FileModelBase,
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> FileOperationResult:
        """Write a single file from a model instance."""
        result = await self.write_files([file], overwrite=overwrite, user_id=user_id)
        if result.results:
            return result.results[0]
        return FileOperationResult(success=result.success, message=result.message, file=File(path=file.path))

    _BATCH_SIZE = 100

    async def write_files(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool = True,
        user_id: str | None = None,
    ) -> BatchResult:
        """Batch write files from model instances."""
        if not files:
            return BatchResult(success=True, message="No files to write")

        all_results: list[FileOperationResult] = []
        for start in range(0, len(files), self._BATCH_SIZE):
            batch = files[start : start + self._BATCH_SIZE]
            batch_result = await self._write_files_batch(batch, overwrite=overwrite, user_id=user_id)
            all_results.extend(batch_result.results)

        succeeded = sum(1 for r in all_results if r.success)
        failed = len(all_results) - succeeded
        return BatchResult(
            success=failed == 0,
            message=f"Wrote {succeeded} file(s)" + (f", {failed} failed" if failed else ""),
            results=all_results,
            succeeded=succeeded,
            failed=failed,
        )

    async def _write_files_batch(
        self,
        files: list[FileModelBase],
        *,
        overwrite: bool,
        user_id: str | None,
    ) -> BatchResult:
        """Process a single batch of <= _BATCH_SIZE files."""
        try:
            # Track results by original index
            results_by_idx: dict[int, FileOperationResult] = {}

            # Group files by mount, keeping original indices
            from collections import defaultdict

            mount_groups: dict[str, list[tuple[int, FileModelBase]]] = defaultdict(list)
            for idx, f in enumerate(files):
                path = normalize_path(f.path)
                try:
                    mount, _rel = self._ctx.registry.resolve(path)
                    mount_groups[mount.path].append((idx, f))
                except Exception as e:
                    results_by_idx[idx] = FileOperationResult(success=False, message=str(e), file=File(path=f.path))

            for mount_path, group in mount_groups.items():
                mount, _ = self._ctx.registry.resolve(mount_path + "/dummy")

                # Check writable
                if err := self._ctx.check_writable(mount_path + "/dummy"):
                    for idx, f in group:
                        results_by_idx[idx] = FileOperationResult(success=False, message=err, file=File(path=f.path))
                    continue

                assert mount.filesystem is not None
                async with self._ctx.session_for(mount) as sess:
                    # Strip mount prefix from paths for backend
                    backend_files = []
                    idx_order: list[int] = []
                    for idx, f in group:
                        rel_path = normalize_path(f.path).removeprefix(mount.path) or "/"
                        bf = type(f).model_validate(
                            {
                                "path": rel_path,
                                "content": f.content if f.content else "",
                            }
                        )
                        backend_files.append(bf)
                        idx_order.append(idx)

                    batch_result = await mount.filesystem.write_files(
                        backend_files,
                        overwrite=overwrite,
                        session=sess,
                        user_id=user_id,
                    )

                    # Re-prefix paths and map back to original indices
                    batch_results_list: list[FileOperationResult] = getattr(batch_result, "results", [])
                    for i, r in enumerate(batch_results_list):
                        r.file.path = self._ctx.prefix_path(r.file.path, mount.path) or r.file.path
                        results_by_idx[idx_order[i]] = r

            # Build ordered results list
            all_results = [results_by_idx[i] for i in range(len(files))]

            # Schedule background processing for successful writes
            for f, result in zip(files, all_results, strict=True):
                if result.success:
                    path = normalize_path(f.path)
                    content = f.content if f.content else ""
                    self._ctx.worker.schedule(
                        path,
                        lambda p=path, c=content, u=user_id: self._process_write(p, c, u),  # type: ignore[attr-defined]
                    )

            succeeded = sum(1 for r in all_results if r.success)
            failed = len(all_results) - succeeded
            return BatchResult(
                success=failed == 0,
                message=f"Wrote {succeeded} file(s)" + (f", {failed} failed" if failed else ""),
                results=all_results,
                succeeded=succeeded,
                failed=failed,
            )
        except Exception as e:
            return BatchResult(success=False, message=f"Batch write failed: {e}")

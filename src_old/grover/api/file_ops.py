"""FileOpsMixin — file CRUD and reconciliation operations for GroverAsync."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import ValidationError

from grover.backends.protocol import SupportsReconcile
from grover.models.internal.detail import WriteDetail
from grover.models.internal.ref import Directory, File
from grover.models.internal.results import (
    FileSearchSet,
    GroverResult,
)
from grover.permissions import Permission
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.api.context import GroverContext
    from grover.models.database.chunk import FileChunkModelBase
    from grover.models.database.file import FileModelBase
    from grover.mount import Mount


class FileOpsMixin:
    """File CRUD and reconciliation operations extracted from GroverAsync."""

    _ctx: GroverContext

    async def read(
        self,
        path: str,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        path = normalize_path(path)
        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.read(rel_path, session=session, user_id=user_id)
        return result.rebase(mount.path)

    async def read_files(
        self,
        paths: list[str],
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch read files, grouped by mount. Cross-mount reads run in parallel."""
        if not paths:
            return GroverResult(success=True, message="No files to read")

        normalized = [normalize_path(p) for p in paths]
        groups = self._ctx.group_by_mount(normalized, lambda p: p)

        async def _handler(mount: Mount, group: list[str], session: AsyncSession) -> GroverResult:
            rel_paths = [p.removeprefix(mount.path) or "/" for p in group]
            result = await mount.filesystem.read_files(rel_paths, session=session)
            return result.rebase(mount.path)

        result = await self._ctx.dispatch_to_mounts(groups, _handler)
        result.message = f"Read {result.succeeded} file(s)" + (f", {result.failed} failed" if result.failed else "")
        return result

    async def write(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        try:
            file_model = self._ctx.resolve_file_model(path)
            f = file_model(path=path, content=content)
        except (ValueError, ValidationError) as e:
            return GroverResult(
                success=False,
                message=str(e),
                files=[File(path=path, evidence=[WriteDetail(operation="write", success=False, message=str(e))])],
            )
        return await self.write_files([f], overwrite=overwrite, user_id=user_id)

    async def write_files(
        self,
        files: list[FileModelBase],
        overwrite: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch write files from model instances."""
        if not files:
            return GroverResult(success=True, message="No files to write")

        groups, err = self._ctx.group_by_mount_writable(files, lambda f: f.path)
        if err:
            return err

        async def _handler(mount: Mount, group: list[FileModelBase], session: AsyncSession) -> GroverResult:
            backend_files = [f.model_copy(update={"path": f.path.removeprefix(mount.path) or "/"}) for f in group]
            try:
                result = await mount.filesystem.write_files(
                    backend_files,
                    overwrite=overwrite,
                    session=session,
                )
                return result.rebase(mount.path)
            except Exception as e:
                return GroverResult(
                    success=False,
                    message=str(e),
                    files=[
                        File(
                            path=f.path,
                            evidence=[WriteDetail(operation="write", success=False, message=str(e))],
                        )
                        for f in group
                    ],
                )

        result = await self._ctx.dispatch_to_mounts(groups, _handler)
        result.message = f"Wrote {result.succeeded} file(s)" + (f", {result.failed} failed" if result.failed else "")
        return result

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        user_id: str | None = None,
    ) -> GroverResult:
        if error := self._ctx.check_writable(path):
            return error
        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.edit(
                rel_path,
                old,
                new,
                replace_all,
                "agent",
                session=session,
                user_id=user_id,
            )
        return result.rebase(mount.path)

    async def delete(self, path: str, permanent: bool = False, *, user_id: str | None = None) -> GroverResult:
        if error := self._ctx.check_writable(path):
            return error
        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.delete(rel_path, permanent, session=session, user_id=user_id)
        return result.rebase(mount.path)

    async def mkdir(
        self,
        path: str,
        parents: bool = True,
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        if error := self._ctx.check_writable(path):
            return error
        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.mkdir(rel_path, parents, session=session, user_id=user_id)
        return result.rebase(mount.path)

    async def exists(self, path: str, *, user_id: str | None = None) -> GroverResult:
        path = normalize_path(path)

        if path == "/":
            return GroverResult(success=True, directories=[Directory(path=path)])

        if self._ctx.registry.has_mount(path):
            return GroverResult(success=True, directories=[Directory(path=path)])

        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.exists(rel_path, session=session, user_id=user_id)
        return result.rebase(mount.path)

    async def move(
        self,
        src: str,
        dest: str,
        *,
        follow: bool = False,
        user_id: str | None = None,
    ) -> GroverResult:
        if follow:
            src, dest = normalize_path(src), normalize_path(dest)
            if error := self._ctx.check_writable(src):
                return error
            if error := self._ctx.check_writable(dest):
                return error
            async with self._ctx.mount_session(src) as (mount, src_rel, session):
                dest_rel = dest.removeprefix(mount.path) or "/"
                result = await mount.filesystem.move(
                    src_rel,
                    dest_rel,
                    session=session,
                    follow=True,
                    user_id=user_id,
                )
            result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
            return GroverResult(
                success=result.success,
                message=result.message,
                files=[result.file],
            )
        return await self.move_files([(src, dest)], user_id=user_id)

    async def move_files(
        self,
        pairs: list[tuple[str, str]],
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch move files. All pairs must be within the same mount.

        Directories are moved one at a time via the single ``move()`` method.
        Files are batched into a single ``move_files()`` call on the backend.
        """
        if not pairs:
            return GroverResult(success=True, message="No files to move")

        # --- Validation (fail-fast) ---
        normalized: list[tuple[str, str]] = []
        mount = None
        for src, dest in pairs:
            src, dest = normalize_path(src), normalize_path(dest)
            if error := self._ctx.check_writable(src):
                return error
            if error := self._ctx.check_writable(dest):
                return error
            src_mount, _ = self._ctx.registry.resolve(src)
            dest_mount, _ = self._ctx.registry.resolve(dest)
            if src_mount is not dest_mount:
                return GroverResult(success=False, message=f"Cannot move across mounts: {src} -> {dest}")
            if mount is None:
                mount = src_mount
            elif mount is not src_mount:
                return GroverResult(success=False, message=f"All moves must be on the same mount: {src}")
            normalized.append((src, dest))

        assert mount is not None
        assert mount.filesystem is not None

        # Check no duplicate destinations
        dests = [d for _, d in normalized]
        if len(set(dests)) != len(dests):
            return GroverResult(success=False, message="Duplicate destination paths in batch")

        # Classify sources as files vs directories via batch read
        src_paths = [s for s, _ in normalized]
        async with self._ctx.session_for(mount) as session:
            read_result = await mount.filesystem.read_files(
                [s.removeprefix(mount.path) or "/" for s in src_paths],
                session=session,
            )

        is_dir: dict[str, bool] = {}
        for src, f in zip(src_paths, read_result.files, strict=True):
            if all(d.success for d in f.details):
                is_dir[src] = False
            elif any("directory" in d.message.lower() for d in f.details):
                is_dir[src] = True
            else:
                return GroverResult(success=False, message=f"Source not found: {src}")

        # Conflict check: no source should be nested inside another source
        dir_srcs = {s for s, is_d in is_dir.items() if is_d}
        for src in src_paths:
            for ds in dir_srcs:
                if src != ds and src.startswith(ds + "/"):
                    return GroverResult(
                        success=False,
                        message=f"Cannot move {src} — its parent {ds} is also being moved",
                    )

        # --- Execution ---
        dir_pairs = [(s, d) for s, d in normalized if is_dir[s]]
        file_pairs = [(s, d) for s, d in normalized if not is_dir[s]]
        all_files: list[File] = []

        # Directories: one at a time via existing move()
        for src, dest in dir_pairs:
            src_rel = src.removeprefix(mount.path) or "/"
            dest_rel = dest.removeprefix(mount.path) or "/"
            async with self._ctx.session_for(mount) as session:
                result = await mount.filesystem.move(src_rel, dest_rel, session=session, user_id=user_id)
            if not result.success:
                return GroverResult(
                    success=False,
                    message=f"Move failed: {src} -> {dest}: {result.message}",
                    files=all_files,
                )
            result.file.path = self._ctx.prefix_path(result.file.path, mount.path) or result.file.path
            all_files.append(result.file)

        # Files: true batch
        if file_pairs:
            rel_pairs = [(s.removeprefix(mount.path) or "/", d.removeprefix(mount.path) or "/") for s, d in file_pairs]
            async with self._ctx.session_for(mount) as session:
                result = await mount.filesystem.move_files(rel_pairs, session=session)
            if not result.success:
                return GroverResult(success=False, message=result.message, files=all_files)
            rebased = result.rebase(mount.path)
            all_files.extend(rebased.files)

        return GroverResult(
            success=True,
            message=f"Moved {len(all_files)} file(s)",
            files=all_files,
        )

    async def copy(self, src: str, dest: str, *, user_id: str | None = None) -> GroverResult:
        return await self.copy_files([(src, dest)], user_id=user_id)

    async def copy_files(
        self,
        pairs: list[tuple[str, str]],
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch copy files. Cross-mount copies use read_files + write_files.

        Directories are copied one at a time via the single ``copy()`` method.
        Same-mount files use backend ``copy_files()``. Cross-mount files use
        batch ``read_files`` from source + ``write_files`` to dest.
        """
        if not pairs:
            return GroverResult(success=True, message="No files to copy")

        # --- Validation (fail-fast) ---
        normalized: list[tuple[str, str]] = []
        for src, dest in pairs:
            src, dest = normalize_path(src), normalize_path(dest)
            if error := self._ctx.check_writable(dest):
                return error
            normalized.append((src, dest))

        dests = [d for _, d in normalized]
        if len(set(dests)) != len(dests):
            return GroverResult(success=False, message="Duplicate destination paths in batch")

        # Group source paths by mount for batch classification
        src_paths = [s for s, _ in normalized]
        src_mount_groups = self._ctx.group_by_mount(src_paths, lambda p: p)

        # Batch-read all sources to classify files vs directories
        src_content: dict[str, str | None] = {}
        is_dir: dict[str, bool] = {}

        for mount_path, group_paths in src_mount_groups.items():
            mount = self._ctx.registry.mounts[mount_path]
            assert mount.filesystem is not None
            rel_paths = [p.removeprefix(mount.path) or "/" for p in group_paths]
            async with self._ctx.session_for(mount) as session:
                read_result = await mount.filesystem.read_files(rel_paths, session=session)

            for src, f in zip(group_paths, read_result.files, strict=True):
                if all(d.success for d in f.details):
                    is_dir[src] = False
                    src_content[src] = f.content
                elif any("directory" in d.message.lower() for d in f.details):
                    is_dir[src] = True
                else:
                    return GroverResult(success=False, message=f"Source not found: {src}")

        # Conflict check: no source should be nested inside another source
        dir_srcs = {s for s, is_d in is_dir.items() if is_d}
        for src in src_paths:
            for ds in dir_srcs:
                if src != ds and src.startswith(ds + "/"):
                    return GroverResult(
                        success=False,
                        message=f"Cannot copy {src} — its parent {ds} is also being copied",
                    )

        # --- Partition ---
        dir_pairs: list[tuple[str, str]] = []
        same_mount_file_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # dest_mount_path -> [(src, dest, content)]
        cross_mount_file_pairs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for src, dest in normalized:
            if is_dir[src]:
                dir_pairs.append((src, dest))
                continue
            src_mount, _ = self._ctx.registry.resolve(src)
            dest_mount, _ = self._ctx.registry.resolve(dest)
            if src_mount is dest_mount:
                same_mount_file_pairs[src_mount.path].append((src, dest))
            else:
                cross_mount_file_pairs[dest_mount.path].append((src, dest, src_content[src] or ""))

        all_files: list[File] = []

        # Directories: one at a time
        for src, dest in dir_pairs:
            src_mount, src_rel = self._ctx.registry.resolve(src)
            dest_mount, dest_rel = self._ctx.registry.resolve(dest)
            assert src_mount.filesystem is not None
            if src_mount is dest_mount:
                async with self._ctx.session_for(src_mount) as session:
                    result = await src_mount.filesystem.copy(src_rel, dest_rel, session=session, user_id=user_id)
                if not result.success:
                    return GroverResult(
                        success=False,
                        message=f"Copy failed: {src} -> {dest}: {result.message}",
                        files=all_files,
                    )
                result.file.path = self._ctx.prefix_path(result.file.path, dest_mount.path) or result.file.path
                all_files.append(result.file)
            else:
                return GroverResult(
                    success=False,
                    message=f"Cross-mount directory copy not supported: {src} -> {dest}",
                    files=all_files,
                )

        # Same-mount files: batch per mount
        for mount_path, file_pairs in same_mount_file_pairs.items():
            mount = self._ctx.registry.mounts[mount_path]
            assert mount.filesystem is not None
            rel_pairs = [(s.removeprefix(mount.path) or "/", d.removeprefix(mount.path) or "/") for s, d in file_pairs]
            async with self._ctx.session_for(mount) as session:
                result = await mount.filesystem.copy_files(rel_pairs, session=session)
            if not result.success:
                return GroverResult(success=False, message=result.message, files=all_files)
            all_files.extend(result.rebase(mount.path).files)

        # Cross-mount files: batch read already done, batch write to dest
        for dest_mount_path, triples in cross_mount_file_pairs.items():
            dest_mount = self._ctx.registry.mounts[dest_mount_path]
            assert dest_mount.filesystem is not None
            from grover.models.database.file import FileModel

            dest_model_cls = getattr(dest_mount.filesystem, "file_model", FileModel)
            try:
                dest_files: list[FileModelBase] = [
                    dest_model_cls(
                        path=dest.removeprefix(dest_mount.path) or "/",
                        content=content,
                    )
                    for _, dest, content in triples
                ]
            except (ValueError, ValidationError) as e:
                return GroverResult(success=False, message=str(e), files=all_files)
            async with self._ctx.session_for(dest_mount) as session:
                result = await dest_mount.filesystem.write_files(dest_files, overwrite=True, session=session)
            if not result.success:
                return GroverResult(success=False, message=result.message, files=all_files)
            all_files.extend(result.rebase(dest_mount.path).files)

        return GroverResult(
            success=True,
            message=f"Copied {len(all_files)} file(s)",
            files=all_files,
        )

    async def list_dir(
        self,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        path = normalize_path(path)

        if path == "/":
            return self._list_root()

        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.list_dir(rel_path, session=session, user_id=user_id)
        result = result.rebase(mount.path)
        if candidates is not None:
            allowed = set(candidates.paths)
            result.files = [f for f in result.files if f.path in allowed]
        return result

    def _list_root(self) -> GroverResult:
        dirs = [Directory(path=mount.path) for mount in self._ctx.registry.list_mounts()]
        return GroverResult(
            success=True,
            message=f"Found {len(dirs)} mount(s)",
            directories=dirs,
        )

    async def tree(
        self,
        path: str = "/",
        *,
        max_depth: int | None = None,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> GroverResult:
        path = normalize_path(path)

        if path == "/":
            # depth 0 = root (empty), depth 1 = mount roots, depth 2+ = mount contents
            mount_dirs = [Directory(path=mount.path) for mount in self._ctx.registry.list_mounts()]
            combined = GroverResult(
                success=True,
                directories=mount_dirs if (max_depth is None or max_depth >= 1) else [],
            )
            if max_depth is None or max_depth > 1:
                for mount in self._ctx.registry.list_mounts():
                    async with self._ctx.session_for(mount) as session:
                        mount_depth = max_depth - 1 if max_depth is not None else None
                        result = await mount.filesystem.tree(
                            "/",
                            max_depth=mount_depth,
                            session=session,
                            user_id=user_id,
                        )
                    if result.success:
                        combined = combined | result.rebase(mount.path)
            combined.message = f"{len(combined.directories)} directories, {len(combined.files)} files"
            if candidates is not None:
                allowed = set(candidates.paths)
                combined.files = [f for f in combined.files if f.path in allowed]
            return combined

        async with self._ctx.mount_session(path) as (mount, rel_path, session):
            result = await mount.filesystem.tree(
                rel_path,
                max_depth=max_depth,
                session=session,
                user_id=user_id,
            )
        result = result.rebase(mount.path)
        if candidates is not None:
            allowed = set(candidates.paths)
            result.files = [f for f in result.files if f.path in allowed]
        return result

    async def write_chunks(
        self,
        chunks: list[FileChunkModelBase],
        *,
        user_id: str | None = None,
    ) -> GroverResult:
        """Batch write (upsert) chunks. Parent files must exist."""
        if not chunks:
            return GroverResult(success=True, message="No chunks to write")

        groups, err = self._ctx.group_by_mount_writable(chunks, lambda c: c.file_path)
        if err:
            return err

        async def _handler(mount: Mount, group: list[FileChunkModelBase], session: AsyncSession) -> GroverResult:
            backend_chunks = [
                type(c).model_validate(
                    {
                        "file_path": normalize_path(c.file_path).removeprefix(mount.path) or "/",
                        "path": normalize_path(c.path).removeprefix(mount.path) or "/",
                        "content": c.content,
                        "line_start": c.line_start,
                        "line_end": c.line_end,
                    }
                )
                for c in group
            ]
            try:
                result = await mount.filesystem.write_chunks(backend_chunks, session=session)
                rebased = result.rebase(mount.path)
                # rebase only prefixes File.path — also rebase chunk paths
                for f in rebased.files:
                    for ch in f.chunks:
                        if not ch.path.startswith(mount.path):
                            ch.path = mount.path + ch.path
                return rebased
            except Exception as e:
                return GroverResult(
                    success=False,
                    message=str(e),
                    files=[
                        File(
                            path=c.path,
                            evidence=[WriteDetail(operation="write_chunk", success=False, message=str(e))],
                        )
                        for c in group
                    ],
                )

        result = await self._ctx.dispatch_to_mounts(groups, _handler)
        result.message = f"Wrote {result.succeeded} chunk(s)" + (f", {result.failed} failed" if result.failed else "")

        # Schedule background processing for successful chunks (keyed by parent file_path)
        success_paths = {f.path for f in result.files if all(d.success for d in f.details)}
        for chunk in chunks:
            if chunk.file_path in success_paths:
                self._ctx.worker.schedule_immediate(
                    self._process_chunk_write(chunk)  # type: ignore[attr-defined]
                )

        return result

    async def reconcile(self, mount_path: str | None = None) -> GroverResult:
        """Reconcile disk <-> DB for capable mounts."""
        total = GroverResult()
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
                result = await cap.reconcile(session=sess)
            total = total | result.rebase(mount.path)

        total.message = f"Reconcile: {len(total.files)} file(s) affected"
        return total

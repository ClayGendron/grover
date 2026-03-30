"""DatabaseFileSystem — SQL-backed implementation of GroverFileSystem.

All entities (files, directories, chunks, versions, connections) live in a
single ``grover_objects`` table.  Operations dispatch by kind — the path
determines the kind, and the kind determines the semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from sqlalchemy import select

from grover.base import GroverFileSystem
from grover.models import GroverObject, GroverObjectBase
from grover.paths import parent_path as compute_parent_path
from grover.paths import parse_kind, version_path
from grover.results import GroverResult
from grover.versioning import SNAPSHOT_INTERVAL

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from grover.results import Candidate


class DatabaseFileSystem(GroverFileSystem):
    """SQL-backed filesystem — portable baseline using SQLAlchemy.

    Stores everything in ``grover_objects``.  Glob, grep, and lexical search
    use SQL LIKE for pre-filtering and Python for authoritative matching/scoring.
    Graph operations delegate to an internal ``RustworkxGraph``.
    """

    DIALECT_PARAMETER_BUDGETS: ClassVar[dict[str, int]] = {
        "sqlite": 900,
        "mssql": 2000,
        "postgresql": 32700,
    }
    PARAMETER_BUDGET_FALLBACK: int = 900
    PARAMETER_RESERVE: int = 100

    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        model: type[GroverObjectBase] = GroverObject,
    ) -> None:
        super().__init__(engine=engine, session_factory=session_factory)
        self._model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_object(
        self,
        path: str,
        session: AsyncSession,
        include_deleted: bool = False,
    ) -> GroverObjectBase | None:
        """Fetch a single object by exact path."""
        stmt = select(self._model).where(self._model.path == path)
        if not include_deleted:
            stmt = stmt.where(self._model.deleted_at.is_(None))  # type: ignore[union-attr]
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    def _parameter_budget(self, session: AsyncSession) -> int:
        """Return a conservative parameter budget for the current SQL dialect."""
        bind = session.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""
        return self.DIALECT_PARAMETER_BUDGETS.get(dialect_name, self.PARAMETER_BUDGET_FALLBACK)

    def _query_chunk_size(
        self,
        session: AsyncSession,
        *,
        binds_per_item: int,
    ) -> int:
        """Compute a safe internal query chunk size for this session."""
        usable_budget = max(1, self._parameter_budget(session) - self.PARAMETER_RESERVE)
        per_item = max(1, binds_per_item)
        return max(1, usable_budget // per_item)

    def _chunk_paths(
        self,
        session: AsyncSession,
        paths: list[str],
        *,
        binds_per_item: int,
    ) -> list[list[str]]:
        """Chunk path lists for internal SQL queries without changing semantics."""
        if not paths:
            return []
        chunk_size = self._query_chunk_size(session, binds_per_item=binds_per_item)
        return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]

    async def _resolve_required_parents(
        self,
        paths: list[str],
        session: AsyncSession,
        *,
        required_kind: str,
        include_deleted: bool,
    ) -> dict[str, GroverObjectBase]:
        """Load required parent objects using a kind-specific policy."""
        resolved: dict[str, GroverObjectBase] = {}
        for batch in self._chunk_paths(session, paths, binds_per_item=1):
            stmt = select(self._model).where(
                self._model.path.in_(batch),  # type: ignore[union-attr]
                self._model.kind == required_kind,
            )
            if not include_deleted:
                stmt = stmt.where(self._model.deleted_at.is_(None))  # type: ignore[union-attr]
            result = await session.execute(stmt)
            resolved.update({obj.path: obj for obj in result.scalars().all()})
        return resolved

    async def _resolve_parent_dirs(
        self,
        paths: list[str],
        session: AsyncSession,
    ) -> tuple[list[GroverObjectBase], list[str]]:
        """Identify ancestor directories that need creation or revival.

        Returns ``(dirs, errors)`` where *dirs* are directory objects
        **without mutating or adding them to the session**.  Revived dirs
        still carry their original ``deleted_at`` value — the caller
        clears it in step 6 inside a savepoint so that a failed write
        batch does not leave revived dirs committed.

        New dirs are fresh model instances (``deleted_at is None``).
        The caller distinguishes the two via ``d.deleted_at is not None``.

        Queries ancestor paths **without** a kind filter so that
        non-directory ancestors (e.g. an existing file at ``/a.txt``)
        are detected and rejected rather than silently shadowed.
        """
        all_ancestors: set[str] = set()
        for path in paths:
            current = compute_parent_path(path)
            while current != "/":
                if current in all_ancestors:
                    break
                all_ancestors.add(current)
                current = compute_parent_path(current)

        if not all_ancestors:
            return [], []

        # Load ALL existing objects at ancestor paths (any kind, including
        # soft-deleted) so we can detect non-directory ancestors.
        existing: dict[str, GroverObjectBase] = {}
        for batch in self._chunk_paths(session, sorted(all_ancestors), binds_per_item=1):
            stmt = select(self._model).where(self._model.path.in_(batch))  # type: ignore[union-attr]
            result = await session.execute(stmt)
            existing.update({obj.path: obj for obj in result.scalars().all()})

        # Reject non-directory ancestors
        errors: list[str] = []
        for p, obj in existing.items():
            if obj.kind != "directory":
                errors.append(
                    f"Ancestor path exists as {obj.kind}, not directory: {p}"
                )

        if errors:
            return [], errors

        # Collect soft-deleted dirs for revival (not mutated yet)
        dirs: list[GroverObjectBase] = [
            existing[p]
            for p in sorted(existing, key=lambda p: p.count("/"))
            if existing[p].deleted_at is not None
        ]

        # Create missing directories (shallowest first)
        missing = sorted(all_ancestors - set(existing), key=lambda p: p.count("/"))
        dirs.extend(self._model(path=ancestor, kind="directory") for ancestor in missing)

        return dirs, []

    async def _validate_chunk_parents(
        self,
        write_map: dict[str, GroverObjectBase],
        session: AsyncSession,
    ) -> tuple[set[str], list[str]]:
        """Reject chunk writes whose companion file is absent from DB and batch."""
        chunk_writes = [
            obj
            for obj in write_map.values()
            if obj.kind == "chunk" and obj.parent_path not in write_map
        ]
        if not chunk_writes:
            return set(), []

        parent_paths = sorted({obj.parent_path for obj in chunk_writes})
        existing_parents = set(
            await self._resolve_required_parents(
                parent_paths,
                session,
                required_kind="file",
                include_deleted=False,
            )
        )

        invalid_paths: set[str] = set()
        errors: list[str] = []
        for obj in chunk_writes:
            if obj.parent_path not in existing_parents:
                invalid_paths.add(obj.path)
                errors.append(f"Chunk parent file not found: {obj.parent_path} (for {obj.path})")

        return invalid_paths, errors

    async def _cascade_delete(
        self,
        path: str,
        session: AsyncSession,
        *,
        timestamp: datetime | None = None,
    ) -> int:
        """Soft-delete children of *path*. Returns count of affected rows."""
        now = timestamp or datetime.now(UTC)
        kind = parse_kind(path)

        if kind == "directory":
            stmt = select(self._model).where(
                self._model.path.like(path + "/%"),  # type: ignore[union-attr]
                self._model.deleted_at.is_(None),  # type: ignore[union-attr]
            )
        else:
            stmt = select(self._model).where(
                self._model.parent_path == path,
                self._model.deleted_at.is_(None),  # type: ignore[union-attr]
            )

        result = await session.execute(stmt)
        children = list(result.scalars().all())

        for child in children:
            child.deleted_at = now

        return len(children)

    # ------------------------------------------------------------------
    # Per-item write helpers
    # ------------------------------------------------------------------

    async def _update_existing(
        self,
        existing: GroverObjectBase,
        incoming: GroverObjectBase,
        new_content: str,
        latest_version_hash: str | None,
        session: AsyncSession,
    ) -> Candidate:
        """Update an existing (or soft-deleted) object with new content.

        Fast path: when the file hash and latest version hash agree,
        ``plan_file_write`` skips reconstruction — no version rows needed.

        Slow path: when hashes disagree (external edit or broken chain),
        fetches the version chain from the DB for reconstruction.
        """
        if existing.deleted_at is not None:
            existing.deleted_at = None

        if incoming.kind == "file" and existing.content is not None:
            plan = existing.plan_file_write(
                new_content,
                latest_version_hash=latest_version_hash,
            )

            # Slow path: plan detected an integrity issue but had no
            # version rows to diagnose it. Fetch the chain and re-plan.
            if not plan.chain_verified and existing.version_number:
                version_rows = await self._fetch_version_chain(
                    existing.path, existing.version_number, session,
                )
                plan = existing.plan_file_write(
                    new_content,
                    version_rows=version_rows,
                    latest_version_hash=latest_version_hash,
                )

            existing.apply_write_plan(plan)
            for version_row in plan.version_rows:
                session.add(version_row)
        else:
            existing.update_content(new_content)

        return existing.to_candidate(operation="write", include_content=True)

    async def _fetch_version_chain(
        self,
        file_path: str,
        current_version: int,
        session: AsyncSession,
    ) -> list[GroverObjectBase]:
        """Fetch the version chain needed for reconstruction.

        Loads versions from the nearest snapshot boundary (within
        ``SNAPSHOT_INTERVAL``) up to *current_version*.
        """
        lower_bound = max(1, current_version - SNAPSHOT_INTERVAL + 1)
        version_paths = [
            version_path(file_path, v) for v in range(lower_bound, current_version + 1)
        ]
        rows: list[GroverObjectBase] = []
        for batch in self._chunk_paths(session, version_paths, binds_per_item=1):
            stmt = select(self._model).where(
                self._model.path.in_(batch),  # type: ignore[union-attr]
            )
            result = await session.execute(stmt)
            rows.extend(result.scalars().all())
        return rows

    def _insert_new(
        self,
        incoming: GroverObjectBase,
        new_content: str,
        session: AsyncSession,
    ) -> Candidate:
        """Insert a new file or chunk.

        New files get an initial v1 snapshot.  Chunks are added directly.
        """
        if incoming.kind == "file" and incoming.content is not None:
            version_obj = type(incoming).create_version_row(
                file_path=incoming.path,
                version_number=1,
                version_content=new_content,
                prev_content=None,
                created_by="auto",
                force_snapshot=True,
            )
            incoming.version_number = 1
            incoming.update_content(new_content)
            session.add(version_obj)
        session.add(incoming)
        return incoming.to_candidate(operation="write", include_content=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def _read_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Read content for one or more objects.

        Accepts either a single ``path`` or a ``GroverResult`` of candidates.

        - *Single path*: fetch the object by exact path, return with content.
        - *Candidates*: batch-fetch all candidate paths in one query,
          preserve prior details from the incoming candidates, and report
          errors for any paths not found.
        """
        if candidates is not None:
            incoming = {c.path: c for c in candidates.candidates}
            paths = list(incoming.keys())
            stmt = select(self._model).where(
                self._model.path.in_(paths),  # type: ignore[union-attr]
                self._model.deleted_at.is_(None),  # type: ignore[union-attr]
            )
            result = await session.execute(stmt)
            objs = {obj.path: obj for obj in result.scalars().all()}
            out: list[Candidate] = []
            errors: list[str] = []
            for p in paths:
                if p in objs:
                    out.append(objs[p].to_candidate(
                        operation="read",
                        include_content=True,
                        prior_details=incoming[p].details,
                    ))
                else:
                    errors.append(f"Not found: {p}")
            return GroverResult(
                candidates=out,
                errors=errors,
                success=len(errors) == 0,
            )

        if path is None:
            return self._error("read requires a path or candidates")

        obj = await self._get_object(path, session)
        if obj is None:
            return self._error(f"Not found: {path}")

        return GroverResult(
            candidates=[obj.to_candidate(operation="read", include_content=True)],
        )

    async def _stat_impl(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Return metadata for one or more objects.

        Delegates to ``_read_impl`` — returns the same result including
        content.  Callers that need metadata-only should strip content
        from the returned candidates.
        """
        return await self._read_impl(path=path, candidates=candidates, session=session)

    async def _write_impl(
        self,
        path: str | None = None,
        content: str | None = None,
        objects: list[GroverObjectBase] | None = None,
        overwrite: bool = True,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Write one or more file/chunk objects to the database.

        Accepts either a single ``path``/``content`` pair or a list of
        ``objects``.  Single writes are wrapped into a one-element list
        so all writes follow the same batch path.

        Process:

        1.  **Validate** — reject non-file/chunk kinds, reject duplicate paths,
            build a path→object map.
        2.  **Validate chunk parents** — chunk writes whose companion file is
            not already in the database must include that file in the same batch.
            Fail fast if not.
        3.  **Ensure parent dirs** — identify ancestor directories for all
            file paths, reviving any soft-deleted directories instead and
            creating new objects if they don't exist. These parent dir updates
            are not added to session, they are only created as objects.
        4.  **Fetch** — batch query retrieves existing objects (including
            soft-deleted) and the bounded version chains needed for file writes.
        5.  **Process each write**:
            - *Soft-deleted file*: clear ``deleted_at`` to undelete.
            - *Existing file, content unchanged*: refresh ``updated_at``.
            - *Existing file*: plan any external/repair snapshots and normal
              version rows via ``plan_file_write``.
            - *Existing chunk*: update content directly (no versioning).
            - *New file/chunk*: add to session.
            - *Flush Session*: session is flushed per batch.
        6.  **Create Parent Dirs** — if file creation was successful, parent
            dirs are added to session and created at this time.

        It is important that the session is managed properly to not overload
        the db passed its parameter threshold.
        """
        # ── Step 1: Validate ──────────────────────────────────────────
        if objects is None:
            if path is None:
                return self._error("Write requires a path or objects")
            objects = [self._model(path=path, content=content or "")]

        write_map: dict[str, GroverObjectBase] = {}
        errors: list[str] = []
        for obj in objects:
            if obj.kind not in ("file", "chunk"):
                errors.append(f"Cannot write to {obj.kind} path: {obj.path}")
                continue
            if obj.path in write_map:
                return self._error(f"Duplicate path in write batch: {obj.path}")
            write_map[obj.path] = obj

        if not write_map:
            return GroverResult(success=len(errors) == 0, errors=errors)

        # ── Step 2: Validate chunk parents ────────────────────────────
        invalid_chunk_paths, chunk_errors = await self._validate_chunk_parents(write_map, session)
        errors.extend(chunk_errors)
        if len(invalid_chunk_paths) == len(write_map):
            return self._error(errors)

        # ── Step 3: Resolve parent dirs (deferred) ────────────────────
        file_paths = [p for p, obj in write_map.items() if obj.kind == "file"]
        parent_dirs: list[GroverObjectBase] = []
        if file_paths:
            parent_dirs, dir_errors = await self._resolve_parent_dirs(file_paths, session)
            if dir_errors:
                errors.extend(dir_errors)
                return self._error(errors)

        # ── Step 4a: Fetch existing objects ──────────────────────────
        all_paths = list(write_map.keys())
        existing_map: dict[str, GroverObjectBase] = {}

        for batch in self._chunk_paths(session, all_paths, binds_per_item=1):
            stmt = select(self._model).where(
                self._model.path.in_(batch),  # type: ignore[union-attr]
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                existing_map[row.path] = row

        # ── Step 4b: Fetch latest version hash per file ───────────────
        # Construct the exact version path for each existing file and
        # fetch just those rows via the unique path index.
        latest_version_hash: dict[str, str | None] = {}
        version_path_to_file: dict[str, str] = {}
        for obj_path, existing in existing_map.items():
            if (
                existing.kind == "file"
                and existing.version_number is not None
                and existing.version_number > 0
            ):
                vp = version_path(obj_path, existing.version_number)
                version_path_to_file[vp] = obj_path

        if version_path_to_file:
            vp_list = list(version_path_to_file.keys())
            for batch in self._chunk_paths(session, vp_list, binds_per_item=1):
                stmt = select(self._model.path, self._model.content_hash).where(  # type: ignore[arg-type]
                    self._model.path.in_(batch),  # type: ignore[union-attr]
                )
                result = await session.execute(stmt)
                for vp, content_hash in result.all():
                    file_path = version_path_to_file[vp]
                    latest_version_hash[file_path] = content_hash

        # ── Step 5: Process each write ─────────────────────────────────
        out: list[Candidate] = []
        for obj_path, incoming in (
            (p, obj) for p, obj in write_map.items() if p not in invalid_chunk_paths
        ):
            new_content = incoming.content or ""
            existing = existing_map.get(obj_path)

            try:
                if existing is not None:
                    if existing.deleted_at is None and not overwrite:
                        errors.append(f"Already exists (overwrite=False): {obj_path}")
                        continue
                    candidate = await self._update_existing(
                        existing, incoming, new_content,
                        latest_version_hash.get(obj_path),
                        session,
                    )
                else:
                    candidate = self._insert_new(incoming, new_content, session)
                out.append(candidate)
            except Exception as exc:
                if existing is not None:
                    session.expire(existing)
                errors.append(f"Write failed for {obj_path}: {exc}")

        await session.flush()

        # ── Step 6: Commit parent dirs ────────────────────────────────
        if out:
            now = datetime.now(UTC)
            for d in parent_dirs:
                if d.deleted_at is not None:
                    d.deleted_at = None
                    d.updated_at = now
                else:
                    session.add(d)
            await session.flush()

        return GroverResult(candidates=out, errors=errors, success=len(errors) == 0)

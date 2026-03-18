"""SearchOpsMixin — search and query operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.models.internal.ref import File
from grover.models.internal.results import FileSearchResult, FileSearchSet
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext


class SearchOpsMixin:
    """Search and query operations extracted from GroverAsync."""

    _ctx: GroverContext

    @staticmethod
    def _split_candidates_for_mount(candidates: FileSearchSet | None, mount_path: str) -> FileSearchSet | None:
        """Strip mount prefix from candidate paths belonging to this mount."""
        if candidates is None:
            return None
        paths = [p.removeprefix(mount_path) or "/" for p in candidates.paths if p.startswith(mount_path)]
        return FileSearchSet.from_paths(paths) if paths else FileSearchSet()

    # ------------------------------------------------------------------
    # Search / Query operations (absorbed from VFS)
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        path = normalize_path(path)
        try:
            if path == "/":
                combined = FileSearchResult(success=True, message="")
                for mount in self._ctx.registry.list_mounts():
                    assert mount.filesystem is not None
                    mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                    async with self._ctx.session_for(mount) as sess:
                        result = await mount.filesystem.glob(
                            pattern, "/", candidates=mount_candidates, session=sess, user_id=user_id
                        )
                    if result.success:
                        combined = combined | result.rebase(mount.path)
                combined.message = f"Found {len(combined)} match(es)"
                return combined
            else:
                mount, rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                async with self._ctx.session_for(mount) as sess:
                    result = await mount.filesystem.glob(
                        pattern, rel_path, candidates=mount_candidates, session=sess, user_id=user_id
                    )
                return result.rebase(mount.path)
        except Exception as e:
            return FileSearchResult(success=False, message=f"Glob failed: {e}")

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        *,
        glob_filter: str | None = None,
        case_sensitive: bool = True,
        fixed_string: bool = False,
        invert: bool = False,
        word_match: bool = False,
        context_lines: int = 0,
        max_results: int = 1000,
        max_results_per_file: int = 0,
        count_only: bool = False,
        files_only: bool = False,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        path = normalize_path(path)
        try:
            if path == "/":
                combined_entries: dict[str, list] = {}
                total_matches = 0

                for mount in self._ctx.registry.list_mounts():
                    if max_results > 0 and total_matches >= max_results:
                        break
                    assert mount.filesystem is not None
                    remaining = max_results - total_matches if max_results > 0 else max_results
                    mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                    async with self._ctx.session_for(mount) as sess:
                        result = await mount.filesystem.grep(
                            pattern,
                            "/",
                            session=sess,
                            glob_filter=glob_filter,
                            case_sensitive=case_sensitive,
                            fixed_string=fixed_string,
                            invert=invert,
                            word_match=word_match,
                            context_lines=context_lines,
                            max_results=remaining,
                            max_results_per_file=max_results_per_file,
                            count_only=False,
                            files_only=files_only,
                            candidates=mount_candidates,
                            user_id=user_id,
                        )
                    if result.success:
                        rebased = result.rebase(mount.path)
                        for f in rebased.files:
                            combined_entries.setdefault(f.path, []).extend(f.evidence)
                            total_matches += sum(
                                len(e.line_matches)  # type: ignore[union-attr]
                                for e in f.evidence
                                if hasattr(e, "line_matches")
                            )

                total_matched = len(combined_entries)
                if count_only:
                    total = total_matched if files_only else total_matches
                    return FileSearchResult(
                        success=True,
                        message=f"Count: {total}",
                    )
                return FileSearchResult(
                    success=True,
                    message=f"Found {total_matches} match(es) in {total_matched} file(s)",
                    files=[File(path=p, evidence=evs) for p, evs in combined_entries.items()],
                )
            else:
                mount, rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                async with self._ctx.session_for(mount) as sess:
                    result = await mount.filesystem.grep(
                        pattern,
                        rel_path,
                        session=sess,
                        glob_filter=glob_filter,
                        case_sensitive=case_sensitive,
                        fixed_string=fixed_string,
                        invert=invert,
                        word_match=word_match,
                        context_lines=context_lines,
                        max_results=max_results,
                        max_results_per_file=max_results_per_file,
                        count_only=count_only,
                        files_only=files_only,
                        candidates=mount_candidates,
                        user_id=user_id,
                    )
                return result.rebase(mount.path)
        except Exception as e:
            return FileSearchResult(success=False, message=f"Grep failed: {e}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Semantic (vector) search, routed to per-mount filesystem providers."""
        path = normalize_path(path)

        # Check if any mount has search providers configured
        has_search = any(
            getattr(mount.filesystem, "search_provider", None) is not None
            and getattr(mount.filesystem, "embedding_provider", None) is not None
            for mount in self._ctx.registry.list_mounts()
        )
        if not has_search:
            return FileSearchResult(
                success=False,
                message=(
                    "Vector search is not available: no search_provider and/or "
                    "embedding_provider configured. Pass both to add_mount()."
                ),
            )

        try:
            if path == "/":
                final = FileSearchResult(success=True, message="")
                for mount in self._ctx.registry.list_mounts():
                    assert mount.filesystem is not None
                    if getattr(mount.filesystem, "search_provider", None) is None:
                        continue
                    if getattr(mount.filesystem, "embedding_provider", None) is None:
                        continue
                    mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                    result = await mount.filesystem.vector_search(query, k, candidates=mount_candidates)
                    if result.success:
                        final = final | result.rebase(mount.path)
                final.message = f"Found matches in {len(final)} file(s)"
            else:
                mount, _rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                if getattr(mount.filesystem, "search_provider", None) is None:
                    return FileSearchResult(success=False, message="No search_provider on mount")
                if getattr(mount.filesystem, "embedding_provider", None) is None:
                    return FileSearchResult(success=False, message="No embedding_provider on mount")
                mount_candidates = self._split_candidates_for_mount(candidates, mount.path)
                result = await mount.filesystem.vector_search(query, k, candidates=mount_candidates)
                final = result.rebase(mount.path)
        except Exception as e:
            return FileSearchResult(
                success=False,
                message=f"Vector search failed: {e}",
            )

        return final

    async def lexical_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """BM25/full-text search, routed to per-mount filesystem providers."""
        path = normalize_path(path)

        try:
            if path == "/":
                combined = FileSearchResult(success=True, message="")
                for mount in self._ctx.registry.list_mounts():
                    assert mount.filesystem is not None
                    async with self._ctx.session_for(mount) as sess:
                        result = await mount.filesystem.lexical_search(query, k=k, session=sess)
                    if result.success:
                        combined = combined | result.rebase(mount.path)
                combined.message = f"Found matches in {len(combined)} file(s)"
                final_lex = combined
            else:
                mount, _rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                async with self._ctx.session_for(mount) as sess:
                    result = await mount.filesystem.lexical_search(query, k=k, session=sess)
                final_lex = result.rebase(mount.path)
        except Exception as e:
            return FileSearchResult(
                success=False,
                message=f"Lexical search failed: {e}",
            )

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final_lex.files = [f for f in final_lex.files if f.path in candidate_paths]
            final_lex.message = f"Found {len(final_lex)} match(es) (filtered)"
        return final_lex

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        *,
        alpha: float = 0.5,
        path: str = "/",
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Hybrid search combining vector and lexical results.

        *alpha* controls the blend: 1.0 = pure vector, 0.0 = pure lexical.
        Falls back to whichever is available if only one is configured.
        """
        path = normalize_path(path)

        vec_result: FileSearchResult | None = None
        lex_result: FileSearchResult | None = None

        has_vector = any(
            getattr(mount.filesystem, "search_provider", None) is not None
            and getattr(mount.filesystem, "embedding_provider", None) is not None
            for mount in self._ctx.registry.list_mounts()
        )
        has_lexical = any(mount.filesystem is not None for mount in self._ctx.registry.list_mounts())

        if has_vector:
            vec_result = await self.vector_search(query, k=k, path=path, candidates=candidates, user_id=user_id)
        if has_lexical:
            lex_result = await self.lexical_search(query, k=k, path=path, candidates=candidates, user_id=user_id)

        if vec_result is not None and lex_result is not None:
            final_hybrid = vec_result | lex_result
        elif vec_result is not None:
            final_hybrid = vec_result
        elif lex_result is not None:
            final_hybrid = lex_result
        else:
            return FileSearchResult(
                success=False,
                message="Hybrid search not available: no vector or lexical search configured",
            )

        return final_hybrid

    async def search(
        self,
        query: str,
        *,
        path: str = "/",
        glob: str | None = None,
        grep: str | None = None,
        k: int = 10,
        candidates: FileSearchSet | None = None,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Composable search pipeline: optional glob/grep filters -> vector search.

        If *candidates* is provided, it seeds the pipeline as an initial filter.
        If *glob* is provided, files are filtered by glob pattern.
        If *grep* is provided, files are further filtered by content pattern.
        Then vector search is applied as the final stage.
        Results are chained using ``>>`` (intersection/pipeline).
        """
        # Start with candidates as the initial filter set
        filter_set: FileSearchSet | None = candidates

        if glob is not None:
            glob_r = await self.glob(glob, path=path, candidates=filter_set, user_id=user_id)
            if not glob_r.success:
                return glob_r
            filter_set = glob_r

        if grep is not None:
            grep_r = await self.grep(grep, path=path, candidates=filter_set, user_id=user_id)
            if not grep_r.success:
                return grep_r
            filter_set = grep_r

        vec_r = await self.vector_search(query, k=k, path=path, candidates=filter_set, user_id=user_id)
        return vec_r

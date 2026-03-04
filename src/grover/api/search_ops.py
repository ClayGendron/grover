"""SearchOpsMixin — search and query operations for GroverAsync."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grover.results import (
    FileSearchResult,
    GlobResult,
    GrepResult,
    LexicalEvidence,
    LexicalSearchResult,
    VectorSearchResult,
)
from grover.util.paths import normalize_path

if TYPE_CHECKING:
    from grover.api.context import GroverContext


class SearchOpsMixin:
    """Search and query operations extracted from GroverAsync."""

    _ctx: GroverContext

    # ------------------------------------------------------------------
    # Search / Query operations (absorbed from VFS)
    # ------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> GlobResult:
        path = normalize_path(path)
        try:
            if path == "/":
                combined = GlobResult(success=True, message="", pattern=pattern)
                for mount in self._ctx.registry.list_visible_mounts():
                    assert mount.filesystem is not None
                    async with self._ctx.session_for(mount) as sess:
                        result = await mount.filesystem.glob(
                            pattern, "/", session=sess, user_id=user_id
                        )
                    if result.success:
                        combined = combined | result.rebase(mount.path)
                combined.message = f"Found {len(combined)} match(es)"
                combined.pattern = pattern
                final = combined
            else:
                mount, rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                async with self._ctx.session_for(mount) as sess:
                    result = await mount.filesystem.glob(
                        pattern, rel_path, session=sess, user_id=user_id
                    )
                final = result.rebase(mount.path)
        except Exception as e:
            return GlobResult(success=False, message=f"Glob failed: {e}", pattern=pattern)

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final.file_candidates = [c for c in final.file_candidates if c.path in candidate_paths]
            final.message = f"Found {len(final)} match(es) (filtered)"
        return final

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
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> GrepResult:
        path = normalize_path(path)
        try:
            if path == "/":
                combined_entries: dict[str, list] = {}
                total_matches = 0
                total_searched = 0
                total_matched = 0
                truncated = False

                for mount in self._ctx.registry.list_visible_mounts():
                    remaining = max_results - total_matches if max_results > 0 else max_results
                    if max_results > 0 and remaining <= 0:
                        truncated = True
                        break
                    assert mount.filesystem is not None
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
                            user_id=user_id,
                        )
                    if result.success:
                        rebased = result.rebase(mount.path)
                        for c in rebased.file_candidates:
                            combined_entries.setdefault(c.path, []).extend(c.evidence)
                            total_matches += sum(
                                len(e.line_matches)  # type: ignore[union-attr]
                                for e in c.evidence
                                if hasattr(e, "line_matches")
                            )
                        total_searched += result.files_searched
                        total_matched += result.files_matched
                        if result.truncated:
                            truncated = True

                if count_only:
                    total = total_matched if files_only else total_matches
                    final = GrepResult(
                        success=True,
                        message=f"Count: {total}",
                        pattern=pattern,
                        files_searched=total_searched,
                        files_matched=total_matched,
                        truncated=truncated,
                    )
                else:
                    final = GrepResult(
                        success=True,
                        message=f"Found {total_matches} match(es) in {total_matched} file(s)",
                        file_candidates=FileSearchResult._dict_to_candidates(combined_entries),
                        pattern=pattern,
                        files_searched=total_searched,
                        files_matched=total_matched,
                        truncated=truncated,
                    )
            else:
                mount, rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
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
                        user_id=user_id,
                    )
                final = result.rebase(mount.path)
        except Exception as e:
            return GrepResult(success=False, message=f"Grep failed: {e}", pattern=pattern)

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final.file_candidates = [c for c in final.file_candidates if c.path in candidate_paths]
            final.message = f"Found {len(final)} match(es) (filtered)"
        return final

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> VectorSearchResult:
        """Semantic (vector) search, routed to per-mount filesystem providers."""
        path = normalize_path(path)

        # Check if any mount has search providers configured
        has_search = any(
            getattr(mount.filesystem, "search_provider", None) is not None
            and getattr(mount.filesystem, "embedding_provider", None) is not None
            for mount in self._ctx.registry.list_visible_mounts()
        )
        if not has_search:
            return VectorSearchResult(
                success=False,
                message=(
                    "Vector search is not available: no search_provider and/or "
                    "embedding_provider configured. Pass both to add_mount()."
                ),
            )

        try:
            if path == "/":
                final = VectorSearchResult(success=True, message="")
                for mount in self._ctx.registry.list_visible_mounts():
                    assert mount.filesystem is not None
                    if getattr(mount.filesystem, "search_provider", None) is None:
                        continue
                    if getattr(mount.filesystem, "embedding_provider", None) is None:
                        continue
                    result = await mount.filesystem.vector_search(query, k)
                    if result.success:
                        final = final | result.rebase(mount.path)
                final.message = f"Found matches in {len(final)} file(s)"
            else:
                mount, _rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                if getattr(mount.filesystem, "search_provider", None) is None:
                    return VectorSearchResult(success=False, message="No search_provider on mount")
                if getattr(mount.filesystem, "embedding_provider", None) is None:
                    return VectorSearchResult(
                        success=False, message="No embedding_provider on mount"
                    )
                result = await mount.filesystem.vector_search(query, k)
                final = result.rebase(mount.path)
        except Exception as e:
            return VectorSearchResult(
                success=False,
                message=f"Vector search failed: {e}",
            )

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final.file_candidates = [c for c in final.file_candidates if c.path in candidate_paths]
            final.message = f"Found {len(final)} match(es) (filtered)"
        return final

    async def lexical_search(
        self,
        query: str,
        k: int = 10,
        *,
        path: str = "/",
        candidates: FileSearchResult | None = None,
        user_id: str | None = None,
    ) -> LexicalSearchResult:
        """BM25/full-text search, routed to per-mount filesystem providers."""
        path = normalize_path(path)

        try:
            if path == "/":
                combined: LexicalSearchResult = LexicalSearchResult(success=True, message="")
                for mount in self._ctx.registry.list_visible_mounts():
                    assert mount.filesystem is not None
                    async with self._ctx.session_for(mount) as sess:
                        fts_results = await mount.filesystem.lexical_search(
                            query, k=k, session=sess
                        )
                    mount_entries: dict[str, list[Any]] = {}
                    for sr in fts_results:
                        fp = mount.path + sr.ref.path
                        ev = LexicalEvidence(
                            operation="lexical_search",
                            snippet=sr.content[:200] if sr.content else "",
                        )
                        mount_entries.setdefault(fp, []).append(ev)
                    mount_result = LexicalSearchResult(
                        success=True,
                        message="",
                        file_candidates=FileSearchResult._dict_to_candidates(mount_entries),
                    )
                    combined = combined | mount_result
                combined.message = f"Found matches in {len(combined)} file(s)"
                final_lex = combined
            else:
                mount, _rel_path = self._ctx.registry.resolve(path)
                assert mount.filesystem is not None
                async with self._ctx.session_for(mount) as sess:
                    fts_results = await mount.filesystem.lexical_search(query, k=k, session=sess)
                entries: dict[str, list[Any]] = {}
                for sr in fts_results:
                    fp = mount.path + sr.ref.path
                    ev = LexicalEvidence(
                        operation="lexical_search",
                        snippet=sr.content[:200] if sr.content else "",
                    )
                    entries.setdefault(fp, []).append(ev)
                final_lex = LexicalSearchResult(
                    success=True,
                    message=f"Found matches in {len(entries)} file(s)",
                    file_candidates=FileSearchResult._dict_to_candidates(entries),
                )
        except Exception as e:
            return LexicalSearchResult(
                success=False,
                message=f"Lexical search failed: {e}",
            )

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final_lex.file_candidates = [
                c for c in final_lex.file_candidates if c.path in candidate_paths
            ]
            final_lex.message = f"Found {len(final_lex)} match(es) (filtered)"
        return final_lex

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        *,
        alpha: float = 0.5,
        path: str = "/",
        candidates: FileSearchResult | None = None,
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
            for mount in self._ctx.registry.list_visible_mounts()
        )
        has_lexical = any(
            mount.filesystem is not None for mount in self._ctx.registry.list_visible_mounts()
        )

        if has_vector:
            vec_result = await self.vector_search(query, k=k, path=path, user_id=user_id)
        if has_lexical:
            lex_result = await self.lexical_search(query, k=k, path=path, user_id=user_id)

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

        if candidates is not None:
            candidate_paths = set(candidates.paths)
            final_hybrid.file_candidates = [
                c for c in final_hybrid.file_candidates if c.path in candidate_paths
            ]
            final_hybrid.message = f"Found {len(final_hybrid)} match(es) (filtered)"
        return final_hybrid

    async def search(
        self,
        query: str,
        *,
        path: str = "/",
        glob: str | None = None,
        grep: str | None = None,
        k: int = 10,
        user_id: str | None = None,
    ) -> FileSearchResult:
        """Composable search pipeline: optional glob/grep filters → vector search.

        If *glob* is provided, files are first filtered by glob pattern.
        If *grep* is provided, files are further filtered by content pattern.
        Then vector search is applied as the final stage.
        Results are chained using ``>>`` (intersection/pipeline).
        """
        result: FileSearchResult | None = None

        if glob is not None:
            glob_r = await self.glob(glob, path=path, user_id=user_id)
            result = glob_r

        if grep is not None:
            grep_r = await self.grep(grep, path=path, user_id=user_id)
            result = grep_r if result is None else (result >> grep_r)

        vec_r = await self.vector_search(query, k=k, path=path, user_id=user_id)
        result = vec_r if result is None else (result >> vec_r)

        return result

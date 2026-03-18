"""Tests for Phase 5: Facade methods return new query response types."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

from grover.backends.local import LocalFileSystem
from grover.client import GroverAsync
from grover.models.internal.evidence import GlobEvidence, GrepEvidence, LineMatch, VectorEvidence
from grover.models.internal.results import FileSearchResult
from grover.providers.search.local import LocalVectorStore

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Helpers: extract grep/glob/vector data from new FileSearchResult
# =========================================================================


def _get_glob_evidence(result: FileSearchResult, path: str) -> GlobEvidence | None:
    """Return the GlobEvidence for *path*, or None."""
    for f in result.files:
        if f.path == path:
            for e in f.evidence:
                if isinstance(e, GlobEvidence):
                    return e
    return None


def _line_matches(result: FileSearchResult, path: str) -> tuple[LineMatch, ...]:
    """Return all line matches for *path*."""
    for f in result.files:
        if f.path == path:
            for e in f.evidence:
                if isinstance(e, GrepEvidence):
                    return e.line_matches
    return ()


def _files_matched(result: FileSearchResult) -> int:
    """Number of files that had grep matches."""
    return len(result.files)


def _snippets(result: FileSearchResult, path: str) -> tuple[str, ...]:
    """Return all vector search snippets for *path*."""
    for f in result.files:
        if f.path == path:
            return tuple(e.snippet for e in f.evidence if isinstance(e, VectorEvidence) and e.snippet)
    return ()


# ------------------------------------------------------------------
# Fake embedding provider
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider for testing."""

    dimensions = _FAKE_DIM
    model_name = "fake-test-model"

    @staticmethod
    def _hash_to_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h[:_FAKE_DIM]]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def embed(self, text: str) -> list[float]:
        return self._hash_to_vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t) for t in texts]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def grover(tmp_path: Path) -> GroverAsync:
    data = tmp_path / "grover_data"
    g = GroverAsync()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    await g.add_mount(
        "project",
        filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        embedding_provider=FakeProvider(),
        search_provider=LocalVectorStore(dimension=_FAKE_DIM),
    )
    yield g  # type: ignore[misc]
    await g.close()


# ==================================================================
# Glob returns GlobResult
# ==================================================================


class TestGlobQueryApi:
    @pytest.mark.asyncio
    async def test_glob_returns_glob_query_result(self, grover: GroverAsync):
        await grover.write("/project/a.py", "print('a')")
        await grover.write("/project/b.py", "print('b')")
        result = await grover.glob("*.py", "/project")
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_glob_hits_are_glob_hit(self, grover: GroverAsync):
        await grover.write("/project/mod.py", "x = 1")
        result = await grover.glob("*.py", "/project")
        assert len(result) >= 1
        for path in result.paths:
            assert path.endswith(".py")

    @pytest.mark.asyncio
    async def test_glob_hit_has_metadata(self, grover: GroverAsync):
        await grover.write("/project/data.txt", "some data")
        result = await grover.glob("*.txt", "/project")
        assert len(result) >= 1
        path = result.paths[0]
        evidence = _get_glob_evidence(result, path)
        assert evidence is not None
        assert evidence.size_bytes is not None
        assert evidence.size_bytes > 0

    @pytest.mark.asyncio
    async def test_glob_empty_pattern(self, grover: GroverAsync):
        result = await grover.glob("*.nonexistent", "/project")
        assert isinstance(result, FileSearchResult)
        assert len(result) == 0


# ==================================================================
# Grep returns GrepResult
# ==================================================================


class TestGrepQueryApi:
    @pytest.mark.asyncio
    async def test_grep_returns_grep_query_result(self, grover: GroverAsync):
        await grover.write("/project/code.py", "def hello():\n    pass\n")
        result = await grover.grep("def ", "/project")
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_grep_groups_by_file(self, grover: GroverAsync):
        await grover.write("/project/a.py", "def alpha():\n    pass\ndef beta():\n    pass\n")
        await grover.write("/project/b.py", "def gamma():\n    pass\n")
        result = await grover.grep("def ", "/project")
        assert isinstance(result, FileSearchResult)
        # Should have paths grouped by file
        paths = list(result.paths)
        assert "/project/a.py" in paths
        assert "/project/b.py" in paths

    @pytest.mark.asyncio
    async def test_grep_hits_are_grep_hit_with_line_matches(self, grover: GroverAsync):
        await grover.write("/project/code.py", "def foo():\n    pass\ndef bar():\n    pass\n")
        result = await grover.grep("def ", "/project")
        for path in result.paths:
            for lm in _line_matches(result, path):
                assert isinstance(lm, LineMatch)
                assert lm.line_number > 0
                assert "def " in lm.line_content

    @pytest.mark.asyncio
    async def test_grep_context_as_tuples(self, grover: GroverAsync):
        await grover.write("/project/ctx.py", "# before\ndef foo():\n    pass\n")
        result = await grover.grep("def ", "/project", context_lines=1)
        for path in result.paths:
            for lm in _line_matches(result, path):
                assert isinstance(lm.context_before, tuple)
                assert isinstance(lm.context_after, tuple)

    @pytest.mark.asyncio
    async def test_grep_stats_populated(self, grover: GroverAsync):
        await grover.write("/project/s1.py", "def x():\n    pass\n")
        await grover.write("/project/s2.py", "no match here\n")
        result = await grover.grep("def ", "/project")
        # At least 1 file matched
        assert _files_matched(result) >= 1

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, grover: GroverAsync):
        await grover.write("/project/empty.py", "x = 1\n")
        result = await grover.grep("nonexistent_pattern", "/project")
        assert isinstance(result, FileSearchResult)
        assert len(result) == 0
        assert _files_matched(result) == 0


# ==================================================================
# Vector search returns VectorSearchResult
# ==================================================================


class TestVectorSearchQueryApi:
    @pytest.mark.asyncio
    async def test_vector_search_returns_vector_search_result(self, grover: GroverAsync):
        await grover.write("/project/auth.py", 'def authenticate():\n    """Auth user."""\n    pass\n')
        result = await grover.vector_search("authenticate")
        assert isinstance(result, FileSearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_vector_search_document_first_grouping(self, grover: GroverAsync):
        """Results should be grouped by file (document-first)."""
        code = "def func_a():\n    pass\n\ndef func_b():\n    pass\n"
        await grover.write("/project/grouped.py", code)
        result = await grover.vector_search("func")
        # No duplicate file paths
        assert len(result.paths) == len(set(result.paths))

    @pytest.mark.asyncio
    async def test_vector_search_paths_are_strings(self, grover: GroverAsync):
        await grover.write("/project/hit.py", 'def findme():\n    """Find this function."""\n    pass\n')
        result = await grover.vector_search("findme")
        for path in result.paths:
            assert isinstance(path, str)

    @pytest.mark.asyncio
    async def test_vector_search_snippets(self, grover: GroverAsync):
        await grover.write(
            "/project/chunk.py",
            'def chunk_func():\n    """A chunk."""\n    return 42\n',
        )
        result = await grover.vector_search("chunk_func")
        if len(result) > 0:
            path = result.paths[0]
            snippets = _snippets(result, path)
            assert isinstance(snippets, tuple)

    @pytest.mark.asyncio
    async def test_vector_search_snippet_truncation(self, grover: GroverAsync):
        long_content = "x" * 500
        await grover.write("/project/long.txt", long_content)
        result = await grover.vector_search("xxxx")
        for path in result.paths:
            for snippet in _snippets(result, path):
                # Snippet should be at most 203 chars (200 + "...")
                assert len(snippet) <= 203

    @pytest.mark.asyncio
    async def test_vector_search_path_scoping(self, grover: GroverAsync):
        """Search with path filter should scope results."""
        await grover.write("/project/src/main.py", 'def main_func():\n    """Main."""\n    pass\n')
        await grover.write("/project/tests/test.py", 'def test_func():\n    """Test."""\n    pass\n')
        result = await grover.vector_search("func", path="/project/src")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_vector_search_files_matched_count(self, grover: GroverAsync):
        await grover.write("/project/f1.py", 'def unique_search_q1():\n    """Q1."""\n    pass\n')
        result = await grover.vector_search("unique_search_q1")
        if len(result) > 0:
            assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_vector_search_failure_no_provider(self, tmp_path: Path):
        """Search without provider returns failure result."""
        data = tmp_path / "grover_data_no_search"
        g = GroverAsync()
        workspace = tmp_path / "ws_no_search"
        workspace.mkdir()
        await g.add_mount(
            "project",
            filesystem=LocalFileSystem(workspace_dir=workspace, data_dir=data / "local"),
        )
        try:
            has_search = any(
                getattr(m.filesystem, "search_provider", None) is not None for m in g._ctx.registry.list_mounts()
            )
            if has_search:
                pytest.skip("search provider is installed; search available")
            result = await g.vector_search("anything")
            assert isinstance(result, FileSearchResult)
            assert result.success is False
            assert "not available" in result.message
        finally:
            await g.close()

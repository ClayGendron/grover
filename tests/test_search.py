"""Tests for the vector search layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from _helpers import FakeProvider
from grover.analyzers.base import ChunkFile
from grover.providers.embedding.protocol import EmbeddingProvider
from grover.providers.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.providers.search.types import SearchResult
from grover.ref import Ref

# ==================================================================
# EmbeddableChunk
# ==================================================================


class TestEmbeddableChunk:
    def test_construction(self):
        ec = EmbeddableChunk(path="/a.py", content="hello", parent_path="/src/a.py")
        assert ec.path == "/a.py"
        assert ec.content == "hello"
        assert ec.parent_path == "/src/a.py"

    def test_default_parent_path(self):
        ec = EmbeddableChunk(path="/b.py", content="hi")
        assert ec.parent_path is None

    def test_frozen(self):
        ec = EmbeddableChunk(path="/a.py", content="hello")
        with pytest.raises(FrozenInstanceError):
            ec.path = "/other.py"  # type: ignore[misc]


# ==================================================================
# extract_from_chunks
# ==================================================================


class TestExtractFromChunks:
    def test_maps_chunk_files(self):
        chunks = [
            ChunkFile(
                path="/a.py#foo",
                parent_path="/a.py",
                content="def foo(): pass",
                line_start=1,
                line_end=1,
                name="foo",
            ),
            ChunkFile(
                path="/a.py#bar",
                parent_path="/a.py",
                content="def bar(): pass",
                line_start=3,
                line_end=3,
                name="bar",
            ),
        ]
        result = extract_from_chunks(chunks)
        assert len(result) == 2
        assert result[0].path == "/a.py#foo"
        assert result[0].content == "def foo(): pass"
        assert result[0].parent_path == "/a.py"
        assert result[1].path == "/a.py#bar"

    def test_filters_empty_content(self):
        chunks = [
            ChunkFile(
                path="/a.py#foo",
                parent_path="/a.py",
                content="def foo(): pass",
                line_start=1,
                line_end=1,
                name="foo",
            ),
            ChunkFile(
                path="/a.py#empty",
                parent_path="/a.py",
                content="   ",
                line_start=5,
                line_end=5,
                name="empty",
            ),
        ]
        result = extract_from_chunks(chunks)
        assert len(result) == 1
        assert result[0].path == "/a.py#foo"

    def test_preserves_parent_path(self):
        chunks = [
            ChunkFile(
                path="/src/b.py#B",
                parent_path="/src/b.py",
                content="class B: pass",
                line_start=1,
                line_end=1,
                name="B",
            ),
        ]
        result = extract_from_chunks(chunks)
        assert result[0].parent_path == "/src/b.py"

    def test_empty_list(self):
        assert extract_from_chunks([]) == []


# ==================================================================
# extract_from_file
# ==================================================================


class TestExtractFromFile:
    def test_single_entry(self):
        result = extract_from_file("/readme.md", "# Hello World")
        assert len(result) == 1
        assert result[0].path == "/readme.md"
        assert result[0].content == "# Hello World"
        assert result[0].parent_path is None

    def test_filters_empty_string(self):
        assert extract_from_file("/empty.txt", "") == []

    def test_filters_whitespace_only(self):
        assert extract_from_file("/blank.txt", "   \n  \t  ") == []

    def test_path_correct(self):
        result = extract_from_file("/src/lib/util.py", "x = 1")
        assert result[0].path == "/src/lib/util.py"


# ==================================================================
# SearchResult
# ==================================================================


class TestSearchResult:
    def test_construction(self):
        sr = SearchResult(
            ref=Ref(path="/a.py"),
            score=0.95,
            content="def foo(): pass",
            parent_path="/src/a.py",
        )
        assert sr.ref.path == "/a.py"
        assert sr.score == 0.95
        assert sr.content == "def foo(): pass"
        assert sr.parent_path == "/src/a.py"

    def test_default_parent_path(self):
        sr = SearchResult(ref=Ref(path="/a.py"), score=0.5, content="x")
        assert sr.parent_path is None

    def test_frozen(self):
        sr = SearchResult(ref=Ref(path="/a.py"), score=0.5, content="x")
        with pytest.raises(FrozenInstanceError):
            sr.score = 0.9  # type: ignore[misc]


# ==================================================================
# EmbeddingProvider protocol
# ==================================================================


class TestEmbeddingProviderProtocol:
    def test_fake_provider_satisfies_protocol(self):
        assert isinstance(FakeProvider(), EmbeddingProvider)

"""Tests for the vector search layer."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest

from grover.graph.analyzers._base import ChunkFile
from grover.ref import Ref
from grover.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.search.protocols import EmbeddingProvider
from grover.search.providers.sentence_transformers import SentenceTransformerEmbedding
from grover.search.types import SearchResult

# ------------------------------------------------------------------
# Fake provider for fast, deterministic unit tests
# ------------------------------------------------------------------

_FAKE_DIM = 32


class FakeProvider:
    """Deterministic embedding provider that hashes text into a vector."""

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return _FAKE_DIM

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) for b in h]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


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
                chunk_path="/.grover/chunks/a_py/foo.txt",
                parent_path="/a.py",
                content="def foo(): pass",
                line_start=1,
                line_end=1,
                name="foo",
            ),
            ChunkFile(
                chunk_path="/.grover/chunks/a_py/bar.txt",
                parent_path="/a.py",
                content="def bar(): pass",
                line_start=3,
                line_end=3,
                name="bar",
            ),
        ]
        result = extract_from_chunks(chunks)
        assert len(result) == 2
        assert result[0].path == "/.grover/chunks/a_py/foo.txt"
        assert result[0].content == "def foo(): pass"
        assert result[0].parent_path == "/a.py"
        assert result[1].path == "/.grover/chunks/a_py/bar.txt"

    def test_filters_empty_content(self):
        chunks = [
            ChunkFile(
                chunk_path="/.grover/chunks/a_py/foo.txt",
                parent_path="/a.py",
                content="def foo(): pass",
                line_start=1,
                line_end=1,
                name="foo",
            ),
            ChunkFile(
                chunk_path="/.grover/chunks/a_py/empty.txt",
                parent_path="/a.py",
                content="   ",
                line_start=5,
                line_end=5,
                name="empty",
            ),
        ]
        result = extract_from_chunks(chunks)
        assert len(result) == 1
        assert result[0].path == "/.grover/chunks/a_py/foo.txt"

    def test_preserves_parent_path(self):
        chunks = [
            ChunkFile(
                chunk_path="/.grover/chunks/b_py/init.txt",
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

    def test_sentence_transformer_satisfies_protocol(self):
        # Use isinstance on an uninitialised instance (bypass __init__)
        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "test"
        p._model = None
        assert isinstance(p, EmbeddingProvider)


# ==================================================================
# SentenceTransformerEmbedding (unit — no model loading)
# ==================================================================


class TestSentenceTransformerEmbedding:
    def test_model_name_default(self):
        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "all-MiniLM-L6-v2"
        p._model = None
        assert p.model_name == "all-MiniLM-L6-v2"

    def test_model_name_custom(self):
        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "custom-model"
        p._model = None
        assert p.model_name == "custom-model"


# ==================================================================
# Integration tests — real SentenceTransformerEmbedding (slow)
# ==================================================================


@pytest.mark.slow
class TestSentenceTransformerIntegration:
    def test_real_model_loads(self):
        provider = SentenceTransformerEmbedding()
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_embed_sync_returns_correct_dimensions(self):
        provider = SentenceTransformerEmbedding()
        vec = provider.embed_sync("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_sync_returns_correct_dimensions(self):
        provider = SentenceTransformerEmbedding()
        vecs = provider.embed_batch_sync(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_dimensions_property(self):
        provider = SentenceTransformerEmbedding()
        assert provider.dimensions == 384

    def test_protocol_satisfied_at_runtime(self):
        provider = SentenceTransformerEmbedding()
        assert isinstance(provider, EmbeddingProvider)

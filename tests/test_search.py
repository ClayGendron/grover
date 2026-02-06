"""Tests for the vector search layer."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from grover.graph.analyzers._base import ChunkFile
from grover.ref import Ref
from grover.search._index import SearchIndex, SearchResult, _content_hash
from grover.search.extractors import (
    EmbeddableChunk,
    extract_from_chunks,
    extract_from_file,
)
from grover.search.providers._protocol import EmbeddingProvider
from grover.search.providers.sentence_transformers import SentenceTransformerProvider

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
        p = SentenceTransformerProvider.__new__(SentenceTransformerProvider)
        p._model_name = "test"
        p._model = None
        assert isinstance(p, EmbeddingProvider)


# ==================================================================
# SentenceTransformerProvider (unit — no model loading)
# ==================================================================


class TestSentenceTransformerProvider:
    def test_model_name_default(self):
        p = SentenceTransformerProvider.__new__(SentenceTransformerProvider)
        p._model_name = "all-MiniLM-L6-v2"
        p._model = None
        assert p.model_name == "all-MiniLM-L6-v2"

    def test_model_name_custom(self):
        p = SentenceTransformerProvider.__new__(SentenceTransformerProvider)
        p._model_name = "custom-model"
        p._model = None
        assert p.model_name == "custom-model"


# ==================================================================
# SearchIndex — core operations
# ==================================================================


class TestSearchIndex:
    def test_add_single_entry(self):
        idx = SearchIndex(FakeProvider())
        key = idx.add("/a.py", "def foo(): pass")
        assert key == 0
        assert len(idx) == 1
        assert idx.has("/a.py")

    def test_add_with_parent_path(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/.grover/chunks/a_py/foo.txt", "def foo(): pass", parent_path="/a.py")
        results = idx.search("def foo(): pass")
        assert results[0].parent_path == "/a.py"

    def test_add_batch(self):
        idx = SearchIndex(FakeProvider())
        entries = [
            EmbeddableChunk(path="/a.py", content="def foo(): pass"),
            EmbeddableChunk(path="/b.py", content="class Bar: pass"),
            EmbeddableChunk(path="/c.py", content="x = 42", parent_path="/pkg.py"),
        ]
        keys = idx.add_batch(entries)
        assert len(keys) == 3
        assert len(idx) == 3
        assert idx.has("/a.py")
        assert idx.has("/b.py")
        assert idx.has("/c.py")

    def test_add_batch_empty(self):
        idx = SearchIndex(FakeProvider())
        keys = idx.add_batch([])
        assert keys == []
        assert len(idx) == 0

    def test_search_returns_results(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "def foo(): pass")
        idx.add("/b.py", "class Bar: pass")
        results = idx.search("def foo(): pass")
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_sorted_descending(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "alpha")
        idx.add("/b.py", "beta")
        idx.add("/c.py", "gamma")
        results = idx.search("alpha", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_best_match_is_exact(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "def foo(): pass")
        idx.add("/b.py", "class Bar: pass")
        results = idx.search("def foo(): pass")
        # Exact text should produce the highest score
        assert results[0].ref.path == "/a.py"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_remove_by_path(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "content a")
        idx.add("/b.py", "content b")
        assert len(idx) == 2
        idx.remove("/a.py")
        assert len(idx) == 1
        assert not idx.has("/a.py")
        assert idx.has("/b.py")

    def test_remove_file_removes_children(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "file content")
        idx.add("/.grover/chunks/a_py/foo.txt", "def foo(): pass", parent_path="/a.py")
        idx.add("/.grover/chunks/a_py/bar.txt", "def bar(): pass", parent_path="/a.py")
        idx.add("/b.py", "other file")
        assert len(idx) == 4

        idx.remove_file("/a.py")
        assert len(idx) == 1
        assert not idx.has("/a.py")
        assert not idx.has("/.grover/chunks/a_py/foo.txt")
        assert not idx.has("/.grover/chunks/a_py/bar.txt")
        assert idx.has("/b.py")

    def test_has_returns_false_for_missing(self):
        idx = SearchIndex(FakeProvider())
        assert not idx.has("/nonexistent.py")

    def test_content_hash(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert idx.content_hash("/a.py") == expected

    def test_content_hash_missing(self):
        idx = SearchIndex(FakeProvider())
        assert idx.content_hash("/missing.py") is None

    def test_len(self):
        idx = SearchIndex(FakeProvider())
        assert len(idx) == 0
        idx.add("/a.py", "content")
        assert len(idx) == 1
        idx.add("/b.py", "content")
        assert len(idx) == 2

    def test_search_with_k(self):
        idx = SearchIndex(FakeProvider())
        for i in range(10):
            idx.add(f"/file{i}.py", f"content number {i}")
        results = idx.search("content number 5", k=3)
        assert len(results) == 3

    def test_search_result_contains_ref(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "hello world")
        results = idx.search("hello world")
        assert isinstance(results[0].ref, Ref)
        assert results[0].ref.path == "/a.py"

    def test_search_result_contains_content(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "hello world")
        results = idx.search("hello world")
        assert results[0].content == "hello world"


# ==================================================================
# SearchIndex — edge cases
# ==================================================================


class TestSearchIndexEdgeCases:
    def test_search_empty_index(self):
        idx = SearchIndex(FakeProvider())
        results = idx.search("anything")
        assert results == []

    def test_remove_nonexistent_path_no_error(self):
        idx = SearchIndex(FakeProvider())
        idx.remove("/nonexistent.py")  # should not raise

    def test_add_duplicate_path_overwrites(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "version 1")
        assert len(idx) == 1
        assert idx.content_hash("/a.py") == _content_hash("version 1")

        idx.add("/a.py", "version 2")
        assert len(idx) == 1
        assert idx.content_hash("/a.py") == _content_hash("version 2")

    def test_add_batch_duplicate_path_overwrites(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "old content")
        entries = [
            EmbeddableChunk(path="/a.py", content="new content"),
            EmbeddableChunk(path="/b.py", content="b content"),
        ]
        idx.add_batch(entries)
        assert len(idx) == 2
        assert idx.content_hash("/a.py") == _content_hash("new content")

    def test_remove_file_without_children(self):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "content")
        idx.remove_file("/a.py")
        assert len(idx) == 0

    def test_remove_file_nonexistent(self):
        idx = SearchIndex(FakeProvider())
        idx.remove_file("/ghost.py")  # should not raise
        assert len(idx) == 0


# ==================================================================
# SearchIndex — persistence
# ==================================================================


class TestSearchIndexPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "hello world")
        idx.add("/b.py", "class Foo: pass", parent_path=None)
        idx.add("/.grover/chunks/c_py/bar.txt", "def bar():", parent_path="/c.py")

        save_dir = str(tmp_path / "index")
        idx.save(save_dir)

        idx2 = SearchIndex.from_directory(save_dir, FakeProvider())
        assert len(idx2) == 3
        assert idx2.has("/a.py")
        assert idx2.has("/b.py")
        assert idx2.has("/.grover/chunks/c_py/bar.txt")

    def test_search_works_after_load(self, tmp_path):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "machine learning model training")
        idx.add("/b.py", "database connection pooling")

        save_dir = str(tmp_path / "index")
        idx.save(save_dir)

        idx2 = SearchIndex.from_directory(save_dir, FakeProvider())
        results = idx2.search("machine learning model training")
        assert len(results) >= 1
        assert results[0].ref.path == "/a.py"

    def test_metadata_preserved(self, tmp_path):
        idx = SearchIndex(FakeProvider())
        idx.add("/.grover/chunks/a_py/foo.txt", "def foo(): pass", parent_path="/a.py")

        save_dir = str(tmp_path / "index")
        idx.save(save_dir)

        idx2 = SearchIndex.from_directory(save_dir, FakeProvider())
        assert idx2.content_hash("/.grover/chunks/a_py/foo.txt") == _content_hash("def foo(): pass")
        results = idx2.search("def foo(): pass")
        assert results[0].parent_path == "/a.py"

    def test_save_creates_two_files(self, tmp_path):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "content")

        save_dir = str(tmp_path / "index")
        idx.save(save_dir)

        assert (Path(save_dir) / "search.usearch").exists()
        assert (Path(save_dir) / "search_meta.json").exists()

    def test_next_key_preserved(self, tmp_path):
        idx = SearchIndex(FakeProvider())
        idx.add("/a.py", "first")
        idx.add("/b.py", "second")

        save_dir = str(tmp_path / "index")
        idx.save(save_dir)

        idx2 = SearchIndex.from_directory(save_dir, FakeProvider())
        # New key should continue from where we left off
        new_key = idx2.add("/c.py", "third")
        assert new_key == 2


# ==================================================================
# Integration tests — real SentenceTransformerProvider (slow)
# ==================================================================


@pytest.mark.slow
class TestSentenceTransformerIntegration:
    def test_real_model_loads(self):
        provider = SentenceTransformerProvider()
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_embed_returns_correct_dimensions(self):
        provider = SentenceTransformerProvider()
        vec = provider.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_returns_correct_dimensions(self):
        provider = SentenceTransformerProvider()
        vecs = provider.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_dimensions_property(self):
        provider = SentenceTransformerProvider()
        assert provider.dimensions == 384

    def test_search_finds_relevant_results(self):
        provider = SentenceTransformerProvider()
        idx = SearchIndex(provider)
        idx.add("/auth.py", "def login(username, password): authenticate user credentials")
        idx.add("/math.py", "def calculate_area(radius): return pi * radius * radius")
        idx.add("/db.py", "def connect_database(host, port): establish database connection")

        results = idx.search("user authentication login", k=3)
        assert results[0].ref.path == "/auth.py"

    def test_protocol_satisfied_at_runtime(self):
        provider = SentenceTransformerProvider()
        assert isinstance(provider, EmbeddingProvider)

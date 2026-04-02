"""Tests for bm25.py — BM25Scorer and BM25Index.

No external dependencies (rank_bm25 not required).
"""

from __future__ import annotations

import math

import pytest

from grover.bm25 import (
    QUERY_TERM_LIMIT,
    BM25Index,
    BM25Scorer,
    tokenize,
    tokenize_query,
)

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self) -> None:
        assert tokenize("foo-bar, baz!") == ["foo", "bar", "baz"]

    def test_empty(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only(self) -> None:
        assert tokenize("   ") == []

    def test_numbers(self) -> None:
        assert tokenize("item1 item2") == ["item1", "item2"]


class TestTokenizeQuery:
    def test_basic(self) -> None:
        assert tokenize_query("hello world") == ["hello", "world"]

    def test_capped(self) -> None:
        text = " ".join(f"term{i}" for i in range(100))
        result = tokenize_query(text)
        assert len(result) == QUERY_TERM_LIMIT


# ------------------------------------------------------------------
# BM25Scorer — IDF
# ------------------------------------------------------------------


class TestBM25ScorerIDF:
    def test_idf_formula(self) -> None:
        # IDF for a term appearing in 10 of 100 documents
        expected = math.log((100 - 10 + 0.5) / (10 + 0.5))
        assert BM25Scorer.idf(10, 100) == pytest.approx(expected)

    def test_idf_zero_docs(self) -> None:
        # Term not seen — max IDF
        result = BM25Scorer.idf(0, 100)
        assert result > 0

    def test_idf_all_docs(self) -> None:
        # Term in every document — negative IDF
        result = BM25Scorer.idf(100, 100)
        assert result < 0

    def test_set_idf_floors_negative(self) -> None:
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        # Term in 90 of 100 docs — raw IDF is negative
        scorer.set_idf({"common": 90, "rare": 2})
        assert scorer.get_idf("common") >= 0
        assert scorer.get_idf("rare") > scorer.get_idf("common")

    def test_set_idf_empty(self) -> None:
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({})
        assert scorer._average_idf == 0.0

    def test_set_idf_with_explicit_average(self) -> None:
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"term": 5}, average_idf=3.0)
        assert scorer._average_idf == 3.0

    def test_get_idf_unknown_term(self) -> None:
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"known": 10})
        unknown = scorer.get_idf("unknown")
        assert unknown > 0
        assert unknown == scorer._unknown_idf


# ------------------------------------------------------------------
# BM25Scorer — Scoring
# ------------------------------------------------------------------


class TestBM25ScorerScoring:
    @pytest.fixture()
    def scorer(self) -> BM25Scorer:
        s = BM25Scorer(corpus_size=5, avg_doc_length=10.0)
        s.set_idf({"hello": 2, "world": 1, "foo": 3})
        return s

    def test_score_document_basic(self, scorer: BM25Scorer) -> None:
        score = scorer.score_document(["hello"], ["hello", "world", "foo"])
        assert score > 0

    def test_score_document_empty_doc(self, scorer: BM25Scorer) -> None:
        assert scorer.score_document(["hello"], []) == 0.0

    def test_score_document_no_match(self, scorer: BM25Scorer) -> None:
        # Term not in document — score is 0 regardless of IDF
        assert scorer.score_document(["missing"], ["hello", "world"]) == 0.0

    def test_score_document_repeated_query_term(self, scorer: BM25Scorer) -> None:
        single = scorer.score_document(["hello"], ["hello", "world"])
        double = scorer.score_document(["hello", "hello"], ["hello", "world"])
        assert double > single

    def test_score_document_longer_doc_lower_score(self, scorer: BM25Scorer) -> None:
        short = scorer.score_document(["hello"], ["hello", "world"])
        long = scorer.score_document(["hello"], ["hello"] + ["padding"] * 50)
        assert short > long  # length normalization penalizes longer docs

    def test_score_batch(self, scorer: BM25Scorer) -> None:
        docs = [
            ["hello", "world"],
            ["foo", "bar"],
            ["hello", "foo"],
        ]
        scores = scorer.score_batch(["hello"], docs)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] == 0.0  # no match for "hello"
        assert scores[2] > 0

    def test_score_batch_empty_query(self, scorer: BM25Scorer) -> None:
        scores = scorer.score_batch([], [["hello"], ["world"]])
        assert scores == [0.0, 0.0]

    def test_score_batch_empty_doc_in_list(self, scorer: BM25Scorer) -> None:
        scores = scorer.score_batch(["hello"], [[], ["hello"]])
        assert scores[0] == 0.0
        assert scores[1] > 0

    def test_score_batch_term_frequencies(self, scorer: BM25Scorer) -> None:
        term_freqs = [{"hello": 2, "world": 1}, {"foo": 3}]
        doc_lengths = [10, 8]
        scores = scorer.score_batch_term_frequencies(["hello"], term_freqs, doc_lengths)
        assert len(scores) == 2
        assert scores[0] > 0
        assert scores[1] == 0.0

    def test_score_batch_term_frequencies_empty_query(self, scorer: BM25Scorer) -> None:
        scores = scorer.score_batch_term_frequencies([], [{"hello": 1}], [5])
        assert scores == [0.0]

    def test_score_batch_term_frequencies_length_mismatch(self, scorer: BM25Scorer) -> None:
        with pytest.raises(ValueError, match="same length"):
            scorer.score_batch_term_frequencies(["hello"], [{"hello": 1}], [5, 10])


# ------------------------------------------------------------------
# BM25Scorer — Edge cases
# ------------------------------------------------------------------


class TestBM25ScorerEdgeCases:
    def test_zero_avg_doc_length(self) -> None:
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=0.0)
        assert scorer.avg_doc_length == 1.0

    def test_collect_term_frequencies_few_terms(self) -> None:
        """When <= _COUNT_FREQUENCY_THRESHOLD terms, uses .count() path."""
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=5.0)
        freqs = scorer._collect_term_frequencies(
            ["a", "b", "a", "c"],
            ("a",),
            frozenset({"a"}),
        )
        assert freqs == {"a": 2}

    def test_collect_term_frequencies_many_terms(self) -> None:
        """When > _COUNT_FREQUENCY_THRESHOLD terms, uses scan path."""
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=5.0)
        freqs = scorer._collect_term_frequencies(
            ["a", "b", "c", "a", "d"],
            ("a", "b", "c"),
            frozenset({"a", "b", "c"}),
        )
        assert freqs == {"a": 2, "b": 1, "c": 1}

    def test_set_idf_negative_average_estimate(self) -> None:
        """When all terms are very common, estimated average IDF is negative."""
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=5.0)
        # All terms in > half the corpus
        scorer.set_idf({"a": 9, "b": 8, "c": 10})
        # Floor should still be non-negative
        for term in ("a", "b", "c"):
            assert scorer.get_idf(term) >= 0


# ------------------------------------------------------------------
# BM25Index
# ------------------------------------------------------------------


class TestBM25Index:
    @pytest.fixture()
    def index(self) -> BM25Index:
        docs = [
            tokenize("the cat sat on the mat"),
            tokenize("the dog played in the yard"),
            tokenize("a cat and a dog are friends"),
            tokenize("the quick brown fox"),
            tokenize("nothing relevant here at all"),
        ]
        return BM25Index(docs)

    def test_constructor(self, index: BM25Index) -> None:
        assert index.corpus_size == 5
        assert index.avg_doc_length > 0

    def test_score_sparse(self, index: BM25Index) -> None:
        scores = index.score_sparse(["cat"])
        assert len(scores) > 0
        # Doc 0 and 2 contain "cat"
        assert 0 in scores
        assert 2 in scores
        assert all(s > 0 for s in scores.values())

    def test_score_sparse_empty_query(self, index: BM25Index) -> None:
        assert index.score_sparse([]) == {}

    def test_score_batch(self, index: BM25Index) -> None:
        scores = index.score_batch(["cat"])
        assert len(scores) == 5
        assert scores[0] > 0  # "cat" in doc 0
        assert scores[4] == 0.0  # "cat" not in doc 4

    def test_topk(self, index: BM25Index) -> None:
        results = index.topk(["cat"], k=2)
        assert len(results) == 2
        # Results are (doc_idx, score) tuples sorted descending
        assert results[0][1] >= results[1][1]
        assert all(idx in (0, 2) for idx, _ in results)

    def test_topk_larger_than_matches(self, index: BM25Index) -> None:
        results = index.topk(["cat"], k=100)
        assert len(results) <= 5

    def test_empty_corpus(self) -> None:
        # Edge case: empty documents in the corpus
        idx = BM25Index([[], ["hello", "world"], []])
        scores = idx.score_sparse(["hello"])
        assert 1 in scores
        assert scores[1] > 0

    def test_multi_term_query(self, index: BM25Index) -> None:
        single = index.score_sparse(["cat"])
        multi = index.score_sparse(["cat", "dog"])
        # Doc 2 has both "cat" and "dog" — should score higher with multi
        assert multi.get(2, 0) > single.get(2, 0)

    def test_repeated_query_term(self, index: BM25Index) -> None:
        single = index.score_sparse(["cat"])
        repeated = index.score_sparse(["cat", "cat"])
        # Repeated term doubles the weight
        for idx in single:
            assert repeated[idx] == pytest.approx(single[idx] * 2)


# ------------------------------------------------------------------
# BM25Scorer — _score_term_frequencies edge cases
# ------------------------------------------------------------------


class TestScoreTermFrequenciesEdgeCases:
    def test_zero_doc_length_returns_zero(self) -> None:
        """Line 238: doc_length == 0 triggers early return of 0.0."""
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=50.0)
        scorer.set_idf({"hello": 5})
        prepared, _, _ = scorer._prepare_query_terms(["hello"])
        assert scorer._score_term_frequencies(prepared, {"hello": 1}, doc_length=0) == 0.0

    def test_empty_prepared_terms_returns_zero(self) -> None:
        """Line 238: empty prepared_terms triggers early return of 0.0."""
        scorer = BM25Scorer(corpus_size=10, avg_doc_length=50.0)
        assert scorer._score_term_frequencies([], {"hello": 1}, doc_length=10) == 0.0

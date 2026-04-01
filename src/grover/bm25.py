"""Hand-rolled BM25 scorer matching rank-bm25 BM25Okapi scoring.

Pure Python, no numpy.  Designed for use with SQL-backed corpora where
full-corpus IDF is computed via COUNT queries and document content is
pre-filtered before scoring.

For repeated in-memory queries over a stable corpus, ``BM25Index``
pre-computes postings and document norms so searches only touch
matching documents instead of rescanning every token every time.

Uses the standard BM25 IDF formula ``log((N - n + 0.5) / (n + 0.5))``
with an epsilon floor for common terms (matching BM25Okapi behavior).
Cached IDF values are always non-negative.

Usage::

    scorer = BM25Scorer(corpus_size=10_000, avg_doc_length=274.0)
    scorer.set_idf({"authentication": 47, "timeout": 312})
    results = scorer.score_batch(query_terms, documents)
"""

from __future__ import annotations

import heapq
import math
import re
from operator import itemgetter

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

QUERY_TERM_LIMIT: int = 50
_COUNT_FREQUENCY_THRESHOLD: int = 2

type PreparedQueryTerm = tuple[str, float, int]
type Posting = tuple[int, int]

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

_SPLIT_RE = re.compile(r"\W+")


def tokenize(text: str) -> list[str]:
    """Lowercase split on non-word characters, drop empties."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


def tokenize_query(text: str) -> list[str]:
    """Tokenize and cap at QUERY_TERM_LIMIT."""
    return tokenize(text)[:QUERY_TERM_LIMIT]


# ------------------------------------------------------------------
# BM25 Scorer
# ------------------------------------------------------------------


class BM25Scorer:
    """BM25 scorer matching rank-bm25 BM25Okapi IDF semantics.

    Corpus-level statistics (``corpus_size``, ``avg_doc_length``) and
    per-term document frequencies are provided externally — typically
    from SQL COUNT queries — so the scorer never needs the full corpus
    in memory.

    Uses the standard BM25 IDF: ``log((N - n + 0.5) / (n + 0.5))``.
    Terms appearing in more than half the corpus would get negative IDF;
    these are floored at ``epsilon * average_idf`` (matching BM25Okapi).

    Parameters
    ----------
    corpus_size:
        Total number of searchable documents in the corpus (N).
    avg_doc_length:
        Average document length in tokens across the corpus.
    k1:
        Term frequency saturation.  Default 1.5.
    b:
        Length normalization.  Default 0.75.
    epsilon:
        Floor factor for common-term IDF.  Default 0.25.
    """

    __slots__ = (
        "_average_idf",
        "_idf_cache",
        "_k1_plus_1",
        "_length_norm_base",
        "_length_norm_scale",
        "_unknown_idf",
        "avg_doc_length",
        "b",
        "corpus_size",
        "epsilon",
        "k1",
    )

    def __init__(
        self,
        corpus_size: int,
        avg_doc_length: float,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.corpus_size = corpus_size
        self.avg_doc_length = avg_doc_length if avg_doc_length > 0 else 1.0
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._idf_cache: dict[str, float] = {}
        self._average_idf: float = 0.0
        self._unknown_idf = max(self.idf(0, self.corpus_size), 0.0)
        self._k1_plus_1 = k1 + 1.0
        self._length_norm_base = k1 * (1.0 - b)
        self._length_norm_scale = k1 * b / self.avg_doc_length

    # -- IDF ------------------------------------------------------------

    @staticmethod
    def idf(n: int, corpus_size: int) -> float:
        """BM25 IDF: ``log((N - n + 0.5) / (n + 0.5))``.

        Can be negative for terms appearing in more than half the corpus.
        Callers should apply the epsilon floor via :meth:`set_idf`.
        """
        return math.log((corpus_size - n + 0.5) / (n + 0.5))

    def set_idf(
        self,
        doc_freqs: dict[str, int],
        *,
        average_idf: float | None = None,
    ) -> None:
        """Pre-compute IDF for each term from document-frequency counts.

        Mirrors rank-bm25 BM25Okapi: raw IDF values that are negative
        (terms in > N/2 documents) are replaced with
        ``epsilon * average_idf``, keeping all scores non-negative.

        Parameters
        ----------
        doc_freqs:
            ``{term: number_of_documents_containing_term}``
        average_idf:
            Pre-computed corpus-wide average IDF.  When provided, the
            epsilon floor uses this value (matching BM25Okapi exactly).
            When *None*, the average is estimated from *doc_freqs* and
            clamped to a positive fallback so tiny candidate sets do not
            collapse to all-zero scores.
        """
        n_docs = self.corpus_size

        # Compute raw IDF for each term
        raw: dict[str, float] = {}
        for term, df in doc_freqs.items():
            raw[term] = self.idf(df, n_docs)

        # Determine average_idf
        if average_idf is not None:
            self._average_idf = average_idf
        elif raw:
            estimated_average_idf = sum(raw.values()) / len(raw)
            self._average_idf = estimated_average_idf if estimated_average_idf > 0 else self._unknown_idf
        else:
            self._average_idf = 0.0

        # Floor negative IDFs at epsilon * average_idf
        eps_floor = self.epsilon * self._average_idf
        # Ensure floor itself is non-negative
        eps_floor = max(eps_floor, 0.0)

        floor_non_positive = average_idf is None
        self._idf_cache = {
            term: (eps_floor if (v < 0 or (floor_non_positive and v == 0)) else v) for term, v in raw.items()
        }

    def get_idf(self, term: str) -> float:
        """Return cached IDF for *term*, or compute from N if unknown."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        # Unknown term — treat as appearing in 0 documents (max IDF)
        return self._unknown_idf

    def _prepare_query_terms(
        self,
        query_terms: list[str],
    ) -> tuple[list[PreparedQueryTerm], tuple[str, ...], frozenset[str]]:
        """Collapse duplicate query terms while preserving order."""
        term_counts: dict[str, int] = {}
        ordered_terms: list[str] = []

        for term in query_terms:
            if term in term_counts:
                term_counts[term] += 1
            else:
                term_counts[term] = 1
                ordered_terms.append(term)

        prepared_terms = [(term, self.get_idf(term), term_counts[term]) for term in ordered_terms]
        term_names = tuple(ordered_terms)
        return prepared_terms, term_names, frozenset(term_names)

    def _collect_term_frequencies(
        self,
        doc_tokens: list[str],
        term_names: tuple[str, ...],
        term_set: frozenset[str],
    ) -> dict[str, int]:
        """Collect document term frequencies for the active query terms."""
        if len(term_names) <= _COUNT_FREQUENCY_THRESHOLD:
            freqs: dict[str, int] = {}
            for term in term_names:
                count = doc_tokens.count(term)
                if count:
                    freqs[term] = count
            return freqs

        freqs = {}
        get_freq = freqs.get
        for token in doc_tokens:
            if token in term_set:
                freqs[token] = get_freq(token, 0) + 1
        return freqs

    def _score_term_frequencies(
        self,
        prepared_terms: list[PreparedQueryTerm],
        term_frequencies: dict[str, int],
        doc_length: int,
    ) -> float:
        """Score pre-counted term frequencies for a single document."""
        if doc_length == 0 or not prepared_terms:
            return 0.0

        denominator_constant = self._length_norm_base + self._length_norm_scale * doc_length
        k1_plus_1 = self._k1_plus_1
        total = 0.0

        for term, idf_val, query_count in prepared_terms:
            freq = term_frequencies.get(term, 0)
            if freq == 0:
                continue

            numerator = freq * k1_plus_1
            total += query_count * idf_val * numerator / (freq + denominator_constant)

        return total

    # -- Scoring --------------------------------------------------------

    def score_document(
        self,
        query_terms: list[str],
        doc_tokens: list[str],
    ) -> float:
        """Score a single document against query terms.

        Parameters
        ----------
        query_terms:
            Tokenized query terms.
        doc_tokens:
            Tokenized document content.

        Returns
        -------
        BM25 score (>= 0).
        """
        if not doc_tokens:
            return 0.0

        prepared_terms, term_names, term_set = self._prepare_query_terms(
            query_terms,
        )
        term_frequencies = self._collect_term_frequencies(
            doc_tokens,
            term_names,
            term_set,
        )
        return self._score_term_frequencies(
            prepared_terms,
            term_frequencies,
            len(doc_tokens),
        )

    def score_batch(
        self,
        query_terms: list[str],
        documents: list[list[str]],
    ) -> list[float]:
        """Score multiple documents, return parallel list of scores.

        This keeps the no-index API efficient for ad hoc SQL pre-filter
        batches.  For repeated in-memory queries over the same corpus,
        prefer :class:`BM25Index`.
        """
        prepared_terms, term_names, term_set = self._prepare_query_terms(
            query_terms,
        )
        if not prepared_terms:
            return [0.0] * len(documents)

        scores: list[float] = []
        for doc_tokens in documents:
            if not doc_tokens:
                scores.append(0.0)
                continue

            term_frequencies = self._collect_term_frequencies(
                doc_tokens,
                term_names,
                term_set,
            )
            scores.append(
                self._score_term_frequencies(
                    prepared_terms,
                    term_frequencies,
                    len(doc_tokens),
                ),
            )

        return scores

    def score_batch_term_frequencies(
        self,
        query_terms: list[str],
        term_frequency_docs: list[dict[str, int]],
        doc_lengths: list[int],
    ) -> list[float]:
        """Score documents from precomputed term frequencies and lengths."""
        if len(term_frequency_docs) != len(doc_lengths):
            msg = "term_frequency_docs and doc_lengths must have the same length"
            raise ValueError(msg)

        prepared_terms, _term_names, _term_set = self._prepare_query_terms(
            query_terms,
        )
        if not prepared_terms:
            return [0.0] * len(term_frequency_docs)

        return [
            self._score_term_frequencies(prepared_terms, term_frequencies, doc_length)
            for term_frequencies, doc_length in zip(
                term_frequency_docs,
                doc_lengths,
                strict=True,
            )
        ]


class BM25Index:
    """Prepared BM25 index for repeated in-memory searches.

    Pre-computes postings, document norms, and full-corpus IDF once, so
    each query only visits matching documents.  This is intended for
    stable corpora already loaded in memory, such as benchmarks.
    """

    __slots__ = (
        "_doc_norms",
        "_postings",
        "avg_doc_length",
        "corpus_size",
        "scorer",
    )

    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.corpus_size = len(documents)
        total_tokens = sum(len(doc) for doc in documents)
        self.avg_doc_length = total_tokens / self.corpus_size if self.corpus_size else 1.0
        self.scorer = BM25Scorer(
            corpus_size=self.corpus_size,
            avg_doc_length=self.avg_doc_length,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

        postings: dict[str, list[Posting]] = {}
        doc_freqs: dict[str, int] = {}
        self._doc_norms = [
            self.scorer._length_norm_base + self.scorer._length_norm_scale * len(doc) for doc in documents
        ]

        for idx, doc_tokens in enumerate(documents):
            if not doc_tokens:
                continue

            term_frequencies: dict[str, int] = {}
            get_freq = term_frequencies.get
            for token in doc_tokens:
                term_frequencies[token] = get_freq(token, 0) + 1

            for term, freq in term_frequencies.items():
                postings.setdefault(term, []).append((idx, freq))
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        average_idf = (
            sum(self.scorer.idf(df, self.corpus_size) for df in doc_freqs.values()) / len(doc_freqs)
            if doc_freqs
            else 0.0
        )
        self.scorer.set_idf(doc_freqs, average_idf=average_idf)
        self._postings = postings

    def score_sparse(self, query_terms: list[str]) -> dict[int, float]:
        """Return scores for matching document indices only."""
        prepared_terms, _term_names, _term_set = self.scorer._prepare_query_terms(query_terms)
        if not prepared_terms:
            return {}

        scores: dict[int, float] = {}
        doc_norms = self._doc_norms
        postings = self._postings
        k1_plus_1 = self.scorer._k1_plus_1

        for term, idf_val, query_count in prepared_terms:
            weight = query_count * idf_val
            for doc_idx, freq in postings.get(term, ()):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + weight * (freq * k1_plus_1 / (freq + doc_norms[doc_idx]))

        return scores

    def score_batch(self, query_terms: list[str]) -> list[float]:
        """Return dense scores aligned to the indexed document order."""
        dense_scores = [0.0] * self.corpus_size
        for doc_idx, score in self.score_sparse(query_terms).items():
            dense_scores[doc_idx] = score
        return dense_scores

    def topk(
        self,
        query_terms: list[str],
        k: int,
    ) -> list[tuple[int, float]]:
        """Return the top-k matching document indices and scores."""
        return heapq.nlargest(
            k,
            ((doc_idx, score) for doc_idx, score in self.score_sparse(query_terms).items() if score > 0),
            key=itemgetter(1),
        )

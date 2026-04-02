"""Compare hand-rolled BM25 (Lucene IDF) against rank-bm25 using Flask repo.

Loads the Flask fixture into both:
  1. In-memory rank-bm25 (BM25Okapi as baseline)
  2. SQLite via DatabaseFileSystem with our hand-rolled BM25 scorer

Then runs 25 search queries and checks that the top-k ranking order is
close between the two implementations.  "Close" means the top-10 results
overlap significantly (>=50% overlap in paths) and the #1 result matches
most of the time.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from grover.backends.database import DatabaseFileSystem
from grover.bm25 import BM25Index, BM25Scorer, tokenize, tokenize_query

# ------------------------------------------------------------------
# Fixture paths
# ------------------------------------------------------------------

FLASK_ROOT = Path(__file__).parent.parent / "tests_old" / "fixtures" / "repos" / "flask"
TEXT_EXTENSIONS = {".py", ".rst", ".txt", ".html", ".md", ".toml", ".cfg", ".ini"}

# ------------------------------------------------------------------
# 25 search queries relevant to the Flask codebase
# ------------------------------------------------------------------

QUERIES = [
    "flask application factory",
    "blueprint registration",
    "request context",
    "WSGI middleware",
    "template rendering jinja",
    "session cookie secure",
    "error handler exception",
    "url routing rule",
    "configuration from environment",
    "test client response",
    "json serialize response",
    "signal request started",
    "command line interface click",
    "static files serve",
    "file upload request",
    "logging debug production",
    "database sqlite connection",
    "decorator route methods",
    "before request hook",
    "response headers content type",
    "secret key session",
    "abort raise HTTPException",
    "werkzeug development server",
    "async view function",
    "redirect url for",
]


# ------------------------------------------------------------------
# Load Flask corpus from disk
# ------------------------------------------------------------------


def load_flask_corpus() -> dict[str, str]:
    """Return {relative_path: content} for all text files in the fixture."""
    corpus: dict[str, str] = {}
    for root, _dirs, files in os.walk(FLASK_ROOT):
        # Skip .git and cache dirs
        root_path = Path(root)
        if any(part.startswith(".") for part in root_path.relative_to(FLASK_ROOT).parts):
            continue
        for fname in files:
            if Path(fname).suffix in TEXT_EXTENSIONS:
                fpath = root_path / fname
                rel = fpath.relative_to(FLASK_ROOT)
                try:
                    content = fpath.read_text(errors="replace")
                    if content.strip():
                        corpus[f"/{rel}"] = content
                except Exception:
                    pass
    return corpus


# ------------------------------------------------------------------
# rank-bm25 baseline (in-memory, full-corpus IDF)
# ------------------------------------------------------------------


def rank_bm25_search(
    corpus: dict[str, str],
    query: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Run BM25Okapi on the full in-memory corpus, return top-k (path, score)."""
    paths = list(corpus.keys())
    tokenized = [tokenize(corpus[p]) for p in paths]
    bm25 = BM25Okapi(tokenized)
    terms = tokenize_query(query)
    scores = bm25.get_scores(terms)
    ranked = sorted(
        ((paths[i], float(scores[i])) for i in range(len(paths))),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(p, s) for p, s in ranked[:k] if s > 0]


# ------------------------------------------------------------------
# Hand-rolled BM25 with full-corpus IDF (in-memory, no SQL)
# ------------------------------------------------------------------


def handrolled_bm25_search(
    corpus: dict[str, str],
    query: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Run our indexed BM25 search, return top-k (path, score)."""
    paths = list(corpus.keys())
    tokenized = [tokenize(corpus[p]) for p in paths]
    terms = tokenize_query(query)
    bm25_index = BM25Index(tokenized)
    return [(paths[i], score) for i, score in bm25_index.topk(terms, k)]


# ------------------------------------------------------------------
# SQL-backed search (DatabaseFileSystem + hand-rolled BM25)
# ------------------------------------------------------------------


async def sql_bm25_search(
    db: DatabaseFileSystem,
    query: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Run lexical_search through DatabaseFileSystem, return top-k (path, score)."""
    async with db._use_session() as s:
        r = await db._lexical_search_impl(query, k=k, session=s)
    return [(c.path, c.score) for c in r.candidates]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def flask_corpus() -> dict[str, str]:
    corpus = load_flask_corpus()
    assert len(corpus) > 50, f"Expected 50+ files, got {len(corpus)}"
    return corpus


@pytest.fixture(scope="module")
def flask_db(flask_corpus: dict[str, str]):
    """Load Flask corpus into SQLite and return a DatabaseFileSystem."""

    async def _setup() -> DatabaseFileSystem:
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        db = DatabaseFileSystem(engine=engine)
        # Batch write all files
        async with db._use_session() as s:
            for path, content in flask_corpus.items():
                await db._write_impl(path, content, session=s)
        return db

    return (
        asyncio.get_event_loop_policy()
        .new_event_loop()
        .run_until_complete(
            _setup(),
        )
    )


class TestHandrolledVsRankBM25:
    """Compare hand-rolled BM25 (Lucene IDF) against rank-bm25 (Okapi IDF).

    These use different IDF formulas, so scores will differ.  But the
    ranking order should be very similar — we check top-10 overlap.
    """

    @pytest.mark.parametrize("query", QUERIES)
    def test_top10_overlap(
        self,
        flask_corpus: dict[str, str],
        query: str,
    ):
        baseline = rank_bm25_search(flask_corpus, query, k=10)
        ours = handrolled_bm25_search(flask_corpus, query, k=10)

        baseline_paths = [p for p, _ in baseline]
        our_paths = [p for p, _ in ours]

        if not baseline_paths and not our_paths:
            return  # Both found nothing — fine

        overlap = set(baseline_paths) & set(our_paths)
        overlap_pct = len(overlap) / max(len(baseline_paths), len(our_paths))

        assert overlap_pct >= 0.5, (
            f"Query {query!r}: only {overlap_pct:.0%} overlap.\n  rank-bm25: {baseline_paths}\n  ours:      {our_paths}"
        )

    @pytest.mark.parametrize("query", QUERIES)
    def test_top1_usually_matches(
        self,
        flask_corpus: dict[str, str],
        query: str,
    ):
        baseline = rank_bm25_search(flask_corpus, query, k=5)
        ours = handrolled_bm25_search(flask_corpus, query, k=5)

        if not baseline or not ours:
            return

        # Top-1 from our scorer should be in baseline's top-5
        our_top = ours[0][0]
        baseline_top5 = {p for p, _ in baseline}
        assert our_top in baseline_top5, (
            f"Query {query!r}: our #1 ({our_top}) not in baseline top-5.\n"
            f"  baseline top-5: {[p for p, _ in baseline]}\n"
            f"  ours top-5:     {[p for p, _ in ours]}"
        )


class TestSQLBackedSearch:
    """Verify SQL-backed search produces similar rankings to in-memory."""

    @pytest.mark.parametrize("query", QUERIES[:10])  # subset for speed
    async def test_sql_vs_inmemory_overlap(
        self,
        flask_corpus: dict[str, str],
        flask_db: DatabaseFileSystem,
        query: str,
    ):
        inmemory = handrolled_bm25_search(flask_corpus, query, k=10)
        sql = await sql_bm25_search(flask_db, query, k=10)

        inmemory_paths = [p for p, _ in inmemory]
        sql_paths = [p for p, _ in sql]

        if not inmemory_paths and not sql_paths:
            return

        overlap = set(inmemory_paths) & set(sql_paths)
        n = max(len(inmemory_paths), len(sql_paths))
        overlap_pct = len(overlap) / n if n > 0 else 1.0

        assert overlap_pct >= 0.5, (
            f"Query {query!r}: only {overlap_pct:.0%} SQL vs in-memory overlap.\n"
            f"  in-memory: {inmemory_paths}\n"
            f"  sql:       {sql_paths}"
        )


class TestBM25ScorerUnit:
    """Unit tests for the BM25Scorer class."""

    def test_raw_idf_matches_bm25_formula(self):
        # Term in every doc is negative before epsilon flooring.
        assert BM25Scorer.idf(100, 100) < 0
        # Term in no docs is maximally rare.
        assert BM25Scorer.idf(0, 100) > 0
        # Rare terms should score higher than common terms.
        assert BM25Scorer.idf(50, 100) < BM25Scorer.idf(1, 100)

    def test_set_idf_floors_negative_values(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=10.0)
        scorer.set_idf({"common": 100, "rare": 1})
        assert scorer.get_idf("common") >= 0
        assert scorer.get_idf("rare") > scorer.get_idf("common")

    def test_idf_rare_term_higher(self):
        rare = BM25Scorer.idf(1, 1000)
        common = BM25Scorer.idf(500, 1000)
        assert rare > common

    def test_score_empty_doc_is_zero(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"test": 10})
        assert scorer.score_document(["test"], []) == 0.0

    def test_score_no_matching_terms_is_zero(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"test": 10})
        assert scorer.score_document(["test"], ["other", "words"]) == 0.0

    def test_score_positive_for_match(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"test": 10})
        score = scorer.score_document(["test"], ["this", "is", "a", "test"])
        assert score > 0

    def test_higher_tf_higher_score(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=10.0)
        scorer.set_idf({"test": 10})
        low_tf = scorer.score_document(["test"], ["test", "other"] * 1)
        high_tf = scorer.score_document(["test"], ["test", "test", "test"])
        assert high_tf > low_tf

    def test_shorter_doc_higher_score_at_same_tf(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=50.0)
        scorer.set_idf({"test": 10})
        short = scorer.score_document(["test"], ["test", "word"])
        long = scorer.score_document(
            ["test"],
            ["test"] + ["word"] * 200,
        )
        assert short > long

    def test_score_batch(self):
        scorer = BM25Scorer(corpus_size=100, avg_doc_length=10.0)
        scorer.set_idf({"hello": 5})
        docs = [["hello", "world"], ["no", "match"], ["hello", "hello"]]
        scores = scorer.score_batch(["hello"], docs)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] == 0.0
        assert scores[2] > scores[0]  # higher TF

    def test_query_term_limit(self):
        long_query = " ".join(f"term{i}" for i in range(100))
        terms = tokenize_query(long_query)
        assert len(terms) == 50

    def test_unknown_term_gets_max_idf(self):
        scorer = BM25Scorer(corpus_size=1000, avg_doc_length=50.0)
        # Never called set_idf — term is unknown
        idf_unknown = scorer.get_idf("xyzzy")
        idf_zero_df = BM25Scorer.idf(0, 1000)
        assert idf_unknown == idf_zero_df


class TestBM25Index:
    def test_index_matches_scorer_for_dense_scores(self):
        docs = [
            ["hello", "world"],
            ["hello", "hello"],
            ["other", "words"],
        ]
        scorer = BM25Scorer(corpus_size=3, avg_doc_length=2.0)
        scorer.set_idf(
            {"hello": 2, "other": 1, "words": 1, "world": 1},
            average_idf=(BM25Scorer.idf(2, 3) + BM25Scorer.idf(1, 3) + BM25Scorer.idf(1, 3) + BM25Scorer.idf(1, 3)) / 4,
        )
        expected = scorer.score_batch(["hello"], docs)

        index = BM25Index(docs)
        actual = index.score_batch(["hello"])

        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize("query", QUERIES[:10])
    def test_index_matches_rank_bm25_top10(
        self,
        flask_corpus: dict[str, str],
        query: str,
    ):
        baseline = rank_bm25_search(flask_corpus, query, k=10)
        indexed = handrolled_bm25_search(flask_corpus, query, k=10)
        assert [path for path, _ in indexed] == [path for path, _ in baseline]
        assert [score for _, score in indexed] == pytest.approx(
            [score for _, score in baseline],
        )


class TestCorpusStats:
    """Verify corpus loading and basic statistics."""

    def test_flask_corpus_loads(self, flask_corpus: dict[str, str]):
        assert len(flask_corpus) > 50
        # Should have Python source
        py_files = [p for p in flask_corpus if p.endswith(".py")]
        assert len(py_files) > 20
        # Should have docs
        rst_files = [p for p in flask_corpus if p.endswith(".rst")]
        assert len(rst_files) > 10

    def test_all_queries_find_results(self, flask_corpus: dict[str, str]):
        """Every query should find at least one result in the Flask corpus."""
        for query in QUERIES:
            results = handrolled_bm25_search(flask_corpus, query, k=5)
            assert len(results) > 0, f"No results for query: {query!r}"


# ===========================================================================
# BM25Scorer / BM25Index edge cases
# ===========================================================================


class TestBM25EdgeCases:
    def test_set_idf_empty_doc_freqs(self):
        scorer = BM25Scorer(corpus_size=5, avg_doc_length=10.0)
        scorer.set_idf({})
        assert scorer._average_idf == 0.0

    def test_score_batch_empty_query_terms(self):
        """Empty query_terms → prepared_terms is empty → all zeros (line 306)."""
        scorer = BM25Scorer(corpus_size=1, avg_doc_length=5.0)
        scorer.set_idf({"hello": 1})
        scores = scorer.score_batch([], [["hello", "world"]])
        assert scores == [0.0]

    def test_score_batch_empty_document(self):
        scorer = BM25Scorer(corpus_size=2, avg_doc_length=3.0)
        scorer.set_idf({"hello": 1})
        scores = scorer.score_batch(["hello"], [[], ["hello"]])
        assert scores[0] == 0.0
        assert scores[1] > 0.0

    def test_score_batch_term_frequencies_mismatched_lengths(self):
        scorer = BM25Scorer(corpus_size=1, avg_doc_length=5.0)
        scorer.set_idf({"a": 1})
        with pytest.raises(ValueError, match="same length"):
            scorer.score_batch_term_frequencies(["a"], [{"a": 1}], [])

    def test_score_batch_term_frequencies_empty_query(self):
        """Empty query → prepared_terms is empty → all zeros (line 344)."""
        scorer = BM25Scorer(corpus_size=1, avg_doc_length=5.0)
        scorer.set_idf({"hello": 1})
        scores = scorer.score_batch_term_frequencies([], [{"hello": 1}], [5])
        assert scores == [0.0]

    def test_score_term_frequencies_zero_doc_length(self):
        """doc_length=0 returns 0.0 (line 238)."""
        scorer = BM25Scorer(corpus_size=1, avg_doc_length=5.0)
        scorer.set_idf({"hello": 1})
        prepared, _, _ = scorer._prepare_query_terms(["hello"])
        score = scorer._score_term_frequencies(prepared, {"hello": 1}, 0)
        assert score == 0.0

    def test_index_with_empty_document(self):
        index = BM25Index([[], ["hello", "world"]])
        assert index.corpus_size == 2

    def test_score_sparse_empty_query(self):
        """Empty query → prepared_terms is empty → empty dict (line 422)."""
        index = BM25Index([["hello", "world"]])
        result = index.score_sparse([])
        assert result == {}

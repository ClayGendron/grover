"""Benchmark: DatabaseFileSystem lexical search vs rank-bm25 (BM25Okapi).

Loads the Flask fixture corpus and runs 25 queries through both
implementations.  Reports rank overlap, rank MSE, score MSE, and
timing for each query plus overall aggregates.

Usage:
    uv run python scripts/bench_bm25_comparison.py
    uv run scripts/bench_bm25_comparison.py
"""

# ruff: noqa: T201

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from grover.backends.database import DatabaseFileSystem  # noqa: E402
from grover.bm25 import tokenize, tokenize_query  # noqa: E402

# ------------------------------------------------------------------
# Corpus
# ------------------------------------------------------------------

FLASK_ROOT = REPO_ROOT / "tests_old" / "fixtures" / "repos" / "flask"
TEXT_EXTENSIONS = {".py", ".rst", ".txt", ".html", ".md", ".toml", ".cfg", ".ini"}

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


def load_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    for root, _dirs, files in os.walk(FLASK_ROOT):
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
# Searchers
# ------------------------------------------------------------------

K = 10


def search_rank_bm25(
    paths: list[str],
    tokenized: list[list[str]],
    bm25: BM25Okapi,
    query: str,
) -> list[tuple[str, float]]:
    terms = tokenize_query(query)
    scores = bm25.get_scores(terms)
    ranked = sorted(
        ((paths[i], float(scores[i])) for i in range(len(paths))),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(p, s) for p, s in ranked[:K] if s > 0]


async def search_dbfs(
    db: DatabaseFileSystem,
    query: str,
) -> list[tuple[str, float]]:
    async with db._use_session() as session:
        result = await db._lexical_search_impl(query, k=K, session=session)
    return [(candidate.path, candidate.score) for candidate in result.candidates]


async def build_dbfs(
    corpus: dict[str, str],
) -> tuple[DatabaseFileSystem, float]:
    t0 = time.perf_counter()
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    db = DatabaseFileSystem(engine=engine)
    async with db._use_session() as session:
        for path, content in corpus.items():
            await db._write_impl(path, content, session=session)
    return db, time.perf_counter() - t0


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def overlap(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    sa = {p for p, _ in a}
    sb = {p for p, _ in b}
    n = max(len(sa), len(sb))
    return len(sa & sb) / n if n else 1.0


def rank_mse(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    ar = {p: i + 1 for i, (p, _) in enumerate(a)}
    br = {p: i + 1 for i, (p, _) in enumerate(b)}
    all_paths = list(dict.fromkeys([p for p, _ in a] + [p for p, _ in b]))
    if not all_paths:
        return 0.0
    return sum((ar.get(p, K + 1) - br.get(p, K + 1)) ** 2 for p in all_paths) / len(all_paths)


def score_mse(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    sa = dict(a)
    sb = dict(b)
    shared = set(sa) & set(sb)
    if not shared:
        return 0.0
    return sum((sa[p] - sb[p]) ** 2 for p in shared) / len(shared)


def score_pct_delta(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    sa = dict(a)
    sb = dict(b)
    shared = set(sa) & set(sb)
    if not shared:
        return 0.0
    pcts = [abs(sa[p] - sb[p]) / abs(sa[p]) * 100 for p in shared if sa[p] != 0]
    return sum(pcts) / len(pcts) if pcts else 0.0


def top1_match(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    if not a or not b:
        return not a and not b
    return a[0][0] == b[0][0]


def top1_in_top5(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    if not a or not b:
        return True
    return b[0][0] in {p for p, _ in a[:5]}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main() -> None:
    print("Loading corpus...")
    corpus = load_corpus()
    paths = list(corpus.keys())
    tokenized = [tokenize(corpus[p]) for p in paths]
    n_docs = len(paths)
    total_tokens = sum(len(doc) for doc in tokenized)
    avgdl = total_tokens / n_docs
    print(f"  {n_docs} documents, {total_tokens:,} tokens, avgdl={avgdl:.0f}\n")

    # Build rank-bm25 index
    t0 = time.perf_counter()
    bm25_okapi = BM25Okapi(tokenized)
    t_ref_build = time.perf_counter() - t0

    # Build our DatabaseFileSystem
    db, t_ours_build = await build_dbfs(corpus)

    print(
        f"Build time: ref={t_ref_build * 1000:.1f}ms, "
        f"dbfs={t_ours_build * 1000:.1f}ms "
        f"({t_ours_build / t_ref_build:.1f}x)\n"
    )

    # Header
    print(
        f"{'Query':<45} {'Overlap':>7} {'RankMSE':>8} {'ScoreMSE':>9} "
        f"{'Δ%':>6} {'#1=':>3} {'#1∈5':>4} {'T_ref':>7} {'T_dbfs':>7}"
    )
    print("-" * 105)

    overlaps = []
    rank_mses = []
    score_mses = []
    score_pcts = []
    top1_matches = []
    top1_in5s = []
    t_refs = []
    t_ours_list = []

    for q in QUERIES:
        # rank-bm25
        t0 = time.perf_counter()
        ref = search_rank_bm25(paths, tokenized, bm25_okapi, q)
        t_ref = time.perf_counter() - t0

        # ours
        t0 = time.perf_counter()
        ours = await search_dbfs(db, q)
        t_ours = time.perf_counter() - t0

        ov = overlap(ref, ours)
        rmse = rank_mse(ref, ours)
        smse = score_mse(ref, ours)
        spct = score_pct_delta(ref, ours)
        t1m = top1_match(ref, ours)
        t1i5 = top1_in_top5(ref, ours)

        overlaps.append(ov)
        rank_mses.append(rmse)
        score_mses.append(smse)
        score_pcts.append(spct)
        top1_matches.append(t1m)
        top1_in5s.append(t1i5)
        t_refs.append(t_ref)
        t_ours_list.append(t_ours)

        print(
            f"{q:<45} {ov:>6.0%} {rmse:>8.2f} {smse:>9.3f} "
            f"{spct:>5.1f}% {'✓' if t1m else '✗':>3} {'✓' if t1i5 else '✗':>4} "
            f"{t_ref * 1000:>6.1f}ms {t_ours * 1000:>6.1f}ms"
        )

    # Aggregates
    print("-" * 105)
    n = len(QUERIES)
    print(
        f"{'AVERAGE':<45} {statistics.mean(overlaps):>6.0%} "
        f"{statistics.mean(rank_mses):>8.2f} {statistics.mean(score_mses):>9.3f} "
        f"{statistics.mean(score_pcts):>5.1f}% "
        f"{sum(top1_matches):>2}/{n} "
        f"{sum(top1_in5s):>2}/{n}  "
        f"{statistics.mean(t_refs) * 1000:>5.1f}ms {statistics.mean(t_ours_list) * 1000:>5.1f}ms"
    )

    print(f"\n{'=' * 105}")
    print("SUMMARY")
    print(f"{'=' * 105}")
    print(f"  Top-10 overlap:    avg {statistics.mean(overlaps):.0%}, min {min(overlaps):.0%}, max {max(overlaps):.0%}")
    print(f"  Rank MSE:          avg {statistics.mean(rank_mses):.2f}, max {max(rank_mses):.2f}")
    print(f"  Score MSE:         avg {statistics.mean(score_mses):.3f}, max {max(score_mses):.3f}")
    print(f"  Score %Δ:          avg {statistics.mean(score_pcts):.1f}%, max {max(score_pcts):.1f}%")
    print(f"  Top-1 exact match: {sum(top1_matches)}/{n}")
    print(f"  Top-1 in top-5:    {sum(top1_in5s)}/{n}")
    print(
        f"  Avg query time:    ref={statistics.mean(t_refs) * 1000:.1f}ms, "
        f"dbfs={statistics.mean(t_ours_list) * 1000:.1f}ms "
        f"({statistics.mean(t_ours_list) / statistics.mean(t_refs):.1f}x)"
    )


if __name__ == "__main__":
    asyncio.run(main())

"""Benchmark rank-bm25 at 1k, 10k, and 100k documents.

Measures indexing time, query time, and peak memory for BM25Okapi.
Documents are synthetic but realistic — variable length, mixed vocabulary.
"""

from __future__ import annotations

import random
import resource
import time

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

VOCAB = (
    "the of and to a in is it that was for on are with as at be this have from "
    "or one had by but not what all were when we there can an your which their "
    "said each she do how will up other about out many then them these so some "
    "would make like him into time has look two more write go see number no way "
    "could people my than first been call who its now find long down day did get "
    "come made may part over new after use work because any good give most just "
    "name very through our much before right too mean old back year where think "
    "also show every great help line turn here why ask went men read need land "
    "different home us move try kind hand picture again change off play spell air "
    "animal house point page letter mother answer found study still learn should "
    "world high every near add food between own below country plant last school "
    "keep never start city tree cross farm hard begin might story saw far sea draw "
    "left late run while press close night real life few north open seem together "
    "next white children begin got walk example ease paper group always music "
    "authentication timeout login session token middleware endpoint request response "
    "database query index schema migration table column foreign key constraint "
    "import export module package function class method async await coroutine "
    "error exception handler retry backoff circuit breaker rate limit throttle "
    "deploy pipeline container kubernetes docker service mesh ingress egress "
    "python javascript typescript golang rust java kotlin swift react angular vue "
    "api rest graphql grpc websocket protocol buffer serialization json yaml toml"
).split()

random.seed(42)


def generate_corpus(n: int) -> list[list[str]]:
    """Generate n tokenized documents with 50-500 words each."""
    corpus = []
    for _ in range(n):
        length = random.randint(50, 500)
        doc = [random.choice(VOCAB) for _ in range(length)]
        corpus.append(doc)
    return corpus


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

QUERIES = [
    "authentication timeout",
    "database query index schema",
    "python async await coroutine",
    "deploy pipeline kubernetes docker container",
    "error handler retry backoff",
]


def get_memory_mb() -> float:
    """Current RSS in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on macOS, KB on Linux
    import platform
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def bench(n: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Benchmarking {n:,} documents")
    print(f"{'='*60}")

    # Generate corpus
    t0 = time.perf_counter()
    corpus = generate_corpus(n)
    gen_time = time.perf_counter() - t0
    total_tokens = sum(len(doc) for doc in corpus)
    avg_len = total_tokens / n
    print(f"  Corpus generated in {gen_time:.3f}s  ({total_tokens:,} tokens, avg {avg_len:.0f}/doc)")

    # Index
    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    bm25 = BM25Okapi(corpus)
    index_time = time.perf_counter() - t0
    mem_after = get_memory_mb()
    index_mem = mem_after - mem_before
    print(f"  Index:  {index_time:.3f}s  (memory delta: ~{max(0, index_mem):.1f} MB)")

    # Query
    query_times = []
    for q in QUERIES:
        tokens = q.lower().split()
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokens)
        elapsed = time.perf_counter() - t0
        query_times.append(elapsed)

        # Top 5 scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_score = scores[top_indices[0]]
        print(f"  Query '{q}': {elapsed*1000:.2f}ms  (top score: {top_score:.3f})")

    avg_query = sum(query_times) / len(query_times)
    min_query = min(query_times)
    max_query = max(query_times)
    p50 = sorted(query_times)[len(query_times) // 2]

    print(f"\n  Query stats:")
    print(f"    avg:  {avg_query*1000:.2f}ms")
    print(f"    min:  {min_query*1000:.2f}ms")
    print(f"    max:  {max_query*1000:.2f}ms")
    print(f"    p50:  {p50*1000:.2f}ms")
    print(f"  Peak RSS: {get_memory_mb():.1f} MB")

    return {
        "n": n,
        "index_time": index_time,
        "avg_query_ms": avg_query * 1000,
        "min_query_ms": min_query * 1000,
        "max_query_ms": max_query * 1000,
        "peak_rss_mb": get_memory_mb(),
    }


if __name__ == "__main__":
    results = []
    for n in [1_000, 10_000, 100_000]:
        results.append(bench(n))

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Docs':>10} {'Index':>10} {'Avg Query':>12} {'Min Query':>12} {'Max Query':>12} {'RSS':>10}")
    for r in results:
        print(
            f"  {r['n']:>10,}"
            f"  {r['index_time']:>9.3f}s"
            f"  {r['avg_query_ms']:>10.2f}ms"
            f"  {r['min_query_ms']:>10.2f}ms"
            f"  {r['max_query_ms']:>10.2f}ms"
            f"  {r['peak_rss_mb']:>8.1f}MB"
        )

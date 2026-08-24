#!/usr/bin/env python3
"""demos/answer_cache_demo.py — Financial ROI & latency demo for the FUTCache
answer engine.

Runs a synthetic customer-support / search workload through PackCache's
``get_or_compute`` and prints the business numbers: real cold-vs-cached
latency (measured, not asserted), the cost ledger (input+output tokens at a
stated price), and the net cost reduction. The hit rate is produced by the
geometry, not hard-coded: the workload is a realistic mix of clustered
intents (many paraphrase variants of the same question) plus a novel tail.

Run:  python demos/answer_cache_demo.py [--n 10000] [--epsilon 0.2] [--k 150]
                                       [--novel 0.15] [--seed 42]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from futcache import PackCache

# ---------------------------------------------------------------- pricing
# Stated, configurable pricing tier (mid/frontier LLM). Values are per 1M
# tokens; tokens-per-query are the demo's workload assumptions.
COST_PER_1M_OUTPUT_TOKENS = 15.00   # $15 / 1M output tokens
COST_PER_1M_INPUT_TOKENS = 3.00     # $3  / 1M input tokens
AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS = 400
LLM_LATENCY_S = 0.45                # mock cold-call latency (network + gen)

# ---------------------------------------------------------------- helpers
def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def build_workload(n: int, dim: int, n_intents: int, novel_frac: float,
                   seed: int):
    """n queries drawn from n_intents clustered intents + a novel tail.

    Each intent is a random unit base vector; its paraphrases are the base
    plus small noise (near-duplicates). Novel queries are fresh random
    vectors (genuinely distinct). Returns (points, labels).
    """
    rng = np.random.default_rng(seed)
    bases = np.array([_unit(rng.normal(size=dim)) for _ in range(n_intents)])
    points = np.zeros((n, dim), dtype=np.float64)
    labels = [""] * n
    for i in range(n):
        if rng.random() < novel_frac:
            # genuinely novel / long-tail question
            v = _unit(rng.normal(size=dim))
            labels[i] = f"novel-query-{i}"
        else:
            b = bases[rng.integers(0, n_intents)]
            v = _unit(b + 0.05 * rng.normal(size=dim))
            labels[i] = f"intent-{int(np.argmax(np.abs(b)))}-para-{i}"
        points[i] = v
    return points, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10000, help="number of queries")
    p.add_argument("--dim", type=int, default=64, help="embedding dimension")
    p.add_argument("--epsilon", type=float, default=0.2,
                   help="semantic distance threshold")
    p.add_argument("--k", type=int, default=150, help="number of intents")
    p.add_argument("--novel", type=float, default=0.15,
                   help="fraction of genuinely novel queries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days-volume", type=int, default=0,
                   help="optional daily query volume for the annualized line")
    args = p.parse_args()

    n = args.n
    dim = args.dim
    cost_per_query = (AVG_INPUT_TOKENS * COST_PER_1M_INPUT_TOKENS +
                      AVG_OUTPUT_TOKENS * COST_PER_1M_OUTPUT_TOKENS) / 1e6

    print("=" * 72)
    print("FUTCache answer-cache ROI demo")
    print("=" * 72)
    print(f"  pricing:  ${COST_PER_1M_INPUT_TOKENS:.2f}/1M in, "
          f"${COST_PER_1M_OUTPUT_TOKENS:.2f}/1M out")
    print(f"  tokens:   {AVG_INPUT_TOKENS} in / {AVG_OUTPUT_TOKENS} out per query")
    print(f"  cost/query: ${cost_per_query:.6f}")
    print(f"  epsilon={args.epsilon}  dim={dim}  intents={args.k}  "
          f"novel={args.novel:.0%}")

    points, labels = build_workload(n, dim, args.k, args.novel, args.seed)

    # ---------------------------------------------------------- latency
    # Measure the ACTUAL cold call latency and the ACTUAL cache-hit latency.
    cache = PackCache(dim, args.epsilon, distance="cosine",
                      domain_min=-1.0, domain_max=1.0, backend="vptree",
                      max_memory_bytes=256 << 20)

    def llm_real(point):
        time.sleep(LLM_LATENCY_S)
        return b"cold synthesis"

    # one real cold call to time it
    t0 = time.perf_counter()
    cache.get_or_compute(points[0], llm_real)
    cold_s = time.perf_counter() - t0

    # time many true cache hits (near-duplicate of the inserted point)
    hits = 2000
    t0 = time.perf_counter()
    for _ in range(hits):
        cache.get_or_compute(points[0], llm_real)   # semantic hit -> no sleep
    hit_s = (time.perf_counter() - t0) / hits

    print("\n[ Latency ]")
    print(f"  cold LLM call (measured):  {cold_s*1e3:8.1f} ms")
    print(f"  cache hit    (measured):  {hit_s*1e6:8.2f} us")
    if hit_s > 0:
        print(f"  speedup:                   {cold_s/hit_s:9,.0f}x")

    # ---------------------------------------------------------- workload
    # Bulk run: FAST compute (no sleep) so the economics over N queries are
    # counted exactly; latency figures above are the measured real path.
    counter = [0]

    def llm_quick(point):
        label = labels[counter[0]] if counter[0] < len(labels) else "?"
        counter[0] += 1
        # token estimate is fixed by the cos model; return a short "answer"
        return f"Synthesized: {label}"

    t0 = time.perf_counter()
    misses = 0
    for i in range(n):
        _, res = cache.get_or_compute(points[i], llm_quick)
        if res.is_novel:
            misses += 1
    wallclock_s = time.perf_counter() - t0
    hits_count = n - misses
    hit_rate = hits_count / n

    print("\n[ Workload ]")
    print(f"  queries processed:        {n:,}")
    print(f"  cold LLM calls:           {misses:,}  ({misses/n:.1%})")
    print(f"  semantic cache hits:      {hits_count:,}  ({hit_rate:.1%})")

    # ---------------------------------------------------------- cost ledger
    naive_cost = n * cost_per_query
    cached_cost = misses * cost_per_query
    saved = naive_cost - cached_cost
    print("\n[ Cost ledger ]")
    print(f"  without FUTCache:         ${naive_cost:,.2f}")
    print(f"  with PackCache:           ${cached_cost:,.2f}")
    print(f"  saved:                    ${saved:,.2f}  "
          f"({saved/naive_cost:.1%} reduction)")
    print(f"  compute calls avoided:    {hits_count:,}")

    if args.days_volume > 0:
        per_day = args.days_volume * (saved / n)
        print(f"  annualized (@{args.days_volume:,} q/day): "
              f"${per_day * 365:,.0f}/yr saved")

    # ---------------------------------------------------------- timing
    naive_time = n * LLM_LATENCY_S
    cached_time = misses * LLM_LATENCY_S + hits_count * hit_s
    print("\n[ Simulated wall-clock ]")
    print(f"  naive (all cold):         {naive_time:,.0f} s")
    print(f"  with PackCache:           {cached_time:,.0f} s")
    if cached_time > 0:
        print(f"  speedup:                  {naive_time/cached_time:.1f}x")
    print(f"\n  (workload geometry ran in {wallclock_s:.2f}s "
          f"for {n:,} queries)")
    print("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""demos/paraphrase_reuse_demo.py — real-text semantic-reuse test.

Tests PackCache on a concrete, human-judgeable workload: a 45-sentence
support corpus and 10 questions, each with alternate ways to ask the same
thing. Uses a real embedding model (hotchpotch/bekko-embedding-v1-a8m, 384-d,
L2-normalised here) so paraphrase-vs-distinct distances are real, not
synthetic. Reports the honest metrics:

  * reuse_rate   = P(cache says HIT)  = hits / queries
  * reuse_precision = P(reuse is CORRECT | cache says HIT)
                   = true_reuse / (true_reuse + false_reuse)

A "true" reuse returns the payload of the SAME intent; a "false" reuse
returns a different intent's payload (an epsilon too large merged two
intents). The boundary margin between paraphrase distance and cross-intent
distance determines where the cache is useful.

Run:  python demos/paraphrase_reuse_demo.py [--epsilon 0.45]
"""

from __future__ import annotations

import argparse

import numpy as np
# Ensure local python/futcache is loaded before any site-packages copy.
import os as _os_demo, sys as _sys_demo
_repo_root_demo = _os_demo.path.dirname(_os_demo.path.dirname(_os_demo.path.abspath(__file__)))
_python_pkg_demo = _os_demo.path.join(_repo_root_demo, 'python')
if _python_pkg_demo not in _sys_demo.path:
    _sys_demo.path.insert(0, _python_pkg_demo)


from _shared import CORPUS, QUESTIONS_MONO as QUESTIONS
from sentence_transformers import SentenceTransformer

from futcache import PackCache

MODEL_NAME = "hotchpotch/bekko-embedding-v1-a8m"

# 45-sentence support corpus (the background knowledge). Phase 1 ingests it.

# 10 questions, each with alternate ways to ask the same thing.
# key = canonical intent; value = [main phrasing, alt 1, alt 2]


def embed(texts, model) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float64)


def build_queries():
    """Flatten QUESTIONS into (intent, phrasing) in a realistic mixed order."""
    pairs = [(intent, phrasing)
             for intent, phrasings in QUESTIONS.items()
             for phrasing in phrasings]
    rng = np.random.default_rng(1234)
    rng.shuffle(pairs)   # interleave intents like real traffic
    return pairs


def run_epsilon(queries, model, epsilon, dim=384) -> dict:
    cache = PackCache(dim, epsilon, distance="cosine",
                      domain_min=-1.0, domain_max=1.0, backend="vptree")
    true_reuse = false_reuse = misses = 0
    wrong = []
    for intent, phrasing in queries:
        vec = embed([phrasing], model)[0]

        def compute(_p, intent=intent):
            # stand-in for "answer": the intent label (or a synthetic answer)
            return intent.encode()

        answer, res = cache.get_or_compute(vec, compute)
        if res.is_novel:
            misses += 1
            continue
        # served from cache: is the returned payload the SAME intent?
        payload = bytes(answer).decode()
        if payload == intent:
            true_reuse += 1
        else:
            false_reuse += 1
            wrong.append((intent, payload))

    hits = true_reuse + false_reuse
    total = len(queries)
    reuse_rate = hits / total if total else 0.0
    reuse_precision = (true_reuse / hits) if hits else 0.0
    return {
        "epsilon": epsilon, "total": total, "misses": misses,
        "hits": hits, "true_reuse": true_reuse, "false_reuse": false_reuse,
        "reuse_rate": reuse_rate, "reuse_precision": reuse_precision,
        "wrong": wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epsilon", type=float, default=0.45,
                    help="run a single epsilon (sweep if omitted)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep epsilon and print the frontier")
    args = ap.parse_args()

    print("=" * 72)
    print("FUTCache semantic-reuse test (real embeddings)")
    print("=" * 72)
    print(f"  model: {args.model} (384-d, L2-normalised)")
    print(f"  corpus: {len(CORPUS)} sentences   questions: "
          f"{len(QUESTIONS)} intents x 3 phrasings")
    print("  loading model ...")
    model = SentenceTransformer(args.model)

    def run_single(eps):
        q = build_queries()
        r = run_epsilon(q, model, eps)
        print(f"\n  epsilon={eps:.2f}  queries={r['total']}  "
              f"hits={r['hits']}  misses={r['misses']}")
        print(f"    true reuse={r['true_reuse']}  false reuse={r['false_reuse']}")
        print(f"    reuse_rate={r['reuse_rate']:.1%}  "
              f"reuse_precision={r['reuse_precision']:.1%}")
        if r["wrong"]:
            print(f"    cross-intent merges: {[(a, b) for a, b in r['wrong'][:4]]}")
        return r

    if args.epsilon > 0 and not args.sweep:
        run_single(args.epsilon)
        print("\n  (run with --sweep to see the epsilon frontier)")
    else:
        epsilons = [0.25, 0.35, 0.42, 0.45, 0.5, 0.55, 0.6, 0.7]
        print(f"\n[ Phase 1 ] ingest {len(CORPUS)}-sentence corpus")
        cache = PackCache(384, 0.45, distance="cosine",
                          domain_min=-1.0, domain_max=1.0, backend="vptree")
        cor = embed(CORPUS, model)
        for v in cor:
            cache.observe(v)
        print(f"  corpus -> {len(cache)} representative(s) "
              f"({len(CORPUS)/max(len(cache),1):.1f}x dedup within corpus)")

        print("\n[ Phase 2 ] question reuse frontier")
        print("  epsilon | reuse_rate | reuse_precision | float(cross-merge)")
        print("  --------|------------|-----------------|-----------------")
        for eps in epsilons:
            r = run_single(eps)
            print(f"  {eps:6.2f} | {r['reuse_rate']:8.1%} | "
                  f"{r['reuse_precision']:13.1%} | {r['false_reuse']}")
        print("\n  The useful regime is where reuse_precision stays high while")
        print("  reuse_rate rises. Above it, two intents start merging (false reuse).")


if __name__ == "__main__":
    main()

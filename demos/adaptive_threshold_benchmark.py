#!/usr/bin/env python3
"""Adaptive FUTCache benchmark on human-judged BEIR queries.

Compares three exact cache decision systems:

  A. one fixed cosine radius;
  B. an Isolation-Forest-contracted cosine radius;
  C. Poincare radial resolution plus Isolation Forest contraction.

Prime-base Halton trials calibrate each parameter space. Human qrels score
reuse after the cache decision; they never alter VP-tree search. The default
dataset is CQAdupStack Stats so repeat opportunities are genuine human-marked
re-asks and the run stays small.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import os
import sys
import time

import numpy as np

# Ensure local python/futcache is loaded before any site-packages copy.
import os as _os_demo, sys as _sys_demo
_repo_root_demo = _os_demo.path.dirname(_os_demo.path.dirname(_os_demo.path.abspath(__file__)))
_python_pkg_demo = _os_demo.path.join(_repo_root_demo, 'python')
if _python_pkg_demo not in _sys_demo.path:
    _sys_demo.path.insert(0, _python_pkg_demo)



from beir_semantic_cache import (
    DATASETS,
    DIM,
    TOP_K,
    embed_texts,
    evidence_groups,
    load_beir,
    retrieve_topk,
)

try:
    from futcache import (
        AdaptiveRadiusPolicy,
        CompactIsolationForest,
        PackCache,
        halton_trials,
        poincare_embed,
    )
except ImportError as exc:
    sys.exit(f"futcache not importable: {exc}\nbuild it first: pip install .")


def _cached_embeddings(cache_dir: str, pattern: str, rows: int) -> np.ndarray:
    matches = glob.glob(os.path.join(cache_dir, pattern))
    if not matches:
        raise RuntimeError("no cached embeddings found; omit --skip-embed")
    best_path = min(
        matches,
        key=lambda path: abs(np.load(path, mmap_mode="r")["emb"].shape[0] - rows),
    )
    embeddings = np.load(best_path)["emb"]
    if embeddings.shape[0] < rows:
        raise RuntimeError(
            f"cached split has {embeddings.shape[0]} rows; expected {rows}")
    return embeddings[:rows].astype(np.float64, copy=False)


def local_specificity_proxy(
    embeddings: np.ndarray,
    *,
    neighbors: int = 8,
    block_size: int = 256,
) -> np.ndarray:
    """Rank local cosine sparsity with O(block_size*N), not O(N^2), memory.

    This is an explicitly unsupervised proxy for hierarchical specificity.
    For publishable experiments, replace it with radii from a learned
    hyperbolic encoder; qrels are intentionally not consulted here.
    """
    count = len(embeddings)
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    k = min(neighbors, count - 1)
    local_distance = np.empty(count, dtype=np.float64)
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        similarities = embeddings[start:stop] @ embeddings.T
        rows = np.arange(stop - start)
        similarities[rows, np.arange(start, stop)] = -np.inf
        top = np.partition(similarities, count - k, axis=1)[:, -k:]
        local_distance[start:stop] = 1.0 - np.mean(top, axis=1)
    order = np.argsort(local_distance, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(count, dtype=np.float64)
    ranks /= max(1, count - 1)
    return ranks


def build_evidence_index(qrels: dict[str, set[str]], qids: list[str]):
    groups = evidence_groups(qrels, qids)
    evidence_of: dict[int, set[str]] = defaultdict(set)
    for document_id, members in groups.items():
        for query_index in members:
            evidence_of[query_index].add(document_id)
    return groups, evidence_of


def evaluate(
    vectors: np.ndarray,
    radii: np.ndarray,
    *,
    metric: str,
    qids: list[str],
    qrels: dict[str, set[str]],
    payloads: dict[str, list[str]],
    evidence_of: dict[int, set[str]],
    max_memory_bytes: int,
) -> dict:
    cache = PackCache(
        dimension=vectors.shape[1],
        epsilon=0.0,
        distance=metric,
        backend="vptree",
        max_memory_bytes=max_memory_bytes,
    )
    hits = correct_hits = missed_reuse = repeat_opportunities = 0
    seen_groups: set[str] = set()
    start = time.perf_counter()
    for query_index, vector in enumerate(vectors):
        groups = evidence_of.get(query_index, set())
        repeat = bool(groups & seen_groups)
        if repeat:
            repeat_opportunities += 1
        query_id = qids[query_index]
        payload = "|".join(payloads[query_id]).encode()
        result = cache.observe(
            vector, payload=payload, radius=float(radii[query_index]))
        if result.is_novel:
            if repeat:
                missed_reuse += 1
        else:
            hits += 1
            cached = cache.get_payload(result.representative_id)
            documents = cached.decode().split("|") if cached else []
            if any(document_id in qrels[query_id]
                   for document_id in documents):
                correct_hits += 1
        seen_groups.update(groups)
    elapsed = time.perf_counter() - start

    replay_hits = replay_correct = 0
    for query_index, vector in enumerate(vectors):
        result = cache.query(vector)
        if result.is_novel:
            continue
        replay_hits += 1
        cached = cache.get_payload(result.representative_id)
        documents = cached.decode().split("|") if cached else []
        if any(document_id in qrels[qids[query_index]]
               for document_id in documents):
            replay_correct += 1

    return {
        "hits": hits,
        "reuse_rate": hits / len(vectors),
        "reuse_precision": correct_hits / hits if hits else None,
        "missed_reuse": missed_reuse,
        "repeat_opportunities": repeat_opportunities,
        "representatives": cache.representative_count,
        "evictions": cache.evictions(),
        "llm_calls_avoided": hits,
        "replay_hits": replay_hits,
        "replay_correctness": replay_correct / replay_hits
                              if replay_hits else None,
        "memory_bytes": cache.memory_bytes(),
        "peak_memory_bytes": cache.peak_memory_bytes(),
        "microseconds_per_observe": elapsed * 1e6 / len(vectors),
    }


def pareto_frontier(results: list[dict]) -> list[dict]:
    frontier = []
    for candidate in results:
        candidate_precision = candidate["reuse_precision"]
        if candidate_precision is None:
            continue
        dominated = any(
            other is not candidate
            and other["reuse_precision"] is not None
            and other["reuse_rate"] >= candidate["reuse_rate"]
            and other["reuse_precision"] >= candidate_precision
            and (other["reuse_rate"] > candidate["reuse_rate"]
                 or other["reuse_precision"] > candidate_precision)
            for other in results
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda result: result["reuse_rate"])


def select_result(results: list[dict], minimum_precision: float) -> dict:
    feasible = [
        result for result in results
        if result["reuse_precision"] is not None
        and result["reuse_precision"] >= minimum_precision
    ]
    if feasible:
        selected = max(feasible, key=lambda result: (
            result["reuse_rate"], result["reuse_precision"]))
        selected["precision_target_met"] = True
        return selected
    judged = [result for result in results
              if result["reuse_precision"] is not None]
    if judged:
        selected = max(judged, key=lambda result: (
            result["reuse_precision"], result["reuse_rate"]))
        selected["precision_target_met"] = False
        return selected
    results[0]["precision_target_met"] = False
    return results[0]


def tune(
    name: str,
    trials,
    radius_builder,
    *,
    vectors: np.ndarray,
    metric: str,
    evaluation_kwargs: dict,
    minimum_precision: float,
) -> tuple[dict, list[dict]]:
    results: list[dict] = []
    for parameters in trials:
        radii = np.asarray(radius_builder(parameters), dtype=np.float64)
        result = evaluate(
            vectors, radii, metric=metric, **evaluation_kwargs)
        result["system"] = name
        result["parameters"] = parameters
        results.append(result)
    best = select_result(results, minimum_precision)
    return best, pareto_frontier(results)


def _format_parameters(parameters: dict[str, float]) -> str:
    return ", ".join(
        f"{name}={value:.3g}" for name, value in parameters.items())


def _format_optional(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="cqadupstack-stats",
                        choices=sorted(DATASETS))
    parser.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache"))
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--minimum-precision", type=float, default=0.995)
    parser.add_argument("--max-memory-bytes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if not 0.0 <= args.minimum_precision <= 1.0:
        parser.error("--minimum-precision must lie in [0, 1]")
    if args.max_memory_bytes < 0:
        parser.error("--max-memory-bytes must be non-negative")

    dataset_key = args.dataset
    source, config, display = DATASETS[dataset_key]
    os.makedirs(args.cache_dir, exist_ok=True)
    print("=" * 78)
    print(f"Adaptive FUTCache — {display}")
    print("=" * 78)

    corpus, queries, qrels = load_beir(dataset_key, args.cache_dir)
    if args.max_queries:
        queries = queries[:args.max_queries]
    if not queries:
        raise RuntimeError("selected stream has no judged queries")
    qids = [query_id for query_id, _ in queries]
    query_texts = [text for _, text in queries]
    document_ids = list(corpus)
    document_texts = [f"{title}. {text}" for title, text in corpus.values()]
    if args.skip_embed:
        query_embeddings = _cached_embeddings(
            args.cache_dir, f"{dataset_key}_q.*.npz", len(query_texts))
        document_embeddings = _cached_embeddings(
            args.cache_dir, f"{dataset_key}_docs.*.npz", len(document_texts))
    else:
        query_embeddings = embed_texts(
            query_texts, os.path.join(args.cache_dir, f"{dataset_key}_q"))
        document_embeddings = embed_texts(
            document_texts,
            os.path.join(args.cache_dir, f"{dataset_key}_docs"),
        )
    print(f"data: {len(corpus):,} documents, {len(queries):,} judged queries; "
          f"Bekko {query_embeddings.shape[1]}-D")

    payloads: dict[str, list[str]] = {}
    retrieval_start = time.perf_counter()
    for query_index, query_id in enumerate(qids):
        ranked, _ = retrieve_topk(
            query_embeddings[query_index], document_embeddings, TOP_K)
        payloads[query_id] = [document_ids[index] for index in ranked]
    print(f"retrieval oracle built in {time.perf_counter() - retrieval_start:.2f}s")

    groups, evidence_of = build_evidence_index(qrels, qids)
    participating = len(evidence_of)
    print(f"human repeat structure: {len(groups)} direct groups, "
          f"{participating} participating queries")

    geometry_start = time.perf_counter()
    specificity = local_specificity_proxy(query_embeddings)
    poincare_vectors = poincare_embed(
        query_embeddings, specificity, min_norm=0.10, max_norm=0.85)
    cosine_forest = CompactIsolationForest(
        n_estimators=64, max_samples=256,
        random_state=args.seed).fit(query_embeddings)
    poincare_forest = CompactIsolationForest(
        n_estimators=64, max_samples=256,
        random_state=args.seed).fit(poincare_vectors)
    cosine_anomaly = cosine_forest.score_samples(query_embeddings)
    poincare_anomaly = poincare_forest.score_samples(poincare_vectors)
    print(f"geometry calibrated in {time.perf_counter() - geometry_start:.2f}s; "
          f"isolation trees use "
          f"{cosine_forest.memory_bytes + poincare_forest.memory_bytes:,} bytes")

    evaluation_kwargs = {
        "qids": qids,
        "qrels": qrels,
        "payloads": payloads,
        "evidence_of": evidence_of,
        "max_memory_bytes": args.max_memory_bytes,
    }
    fixed_trials = halton_trials(
        {"epsilon": (0.02, 0.80)}, max(8, args.trials // 2))
    fixed_best, fixed_frontier = tune(
        "A fixed cosine",
        fixed_trials,
        lambda parameters: np.full(
            len(query_embeddings), parameters["epsilon"]),
        vectors=query_embeddings,
        metric="cosine",
        evaluation_kwargs=evaluation_kwargs,
        minimum_precision=args.minimum_precision,
    )

    isolation_trials = halton_trials({
        "epsilon_0": (0.02, 1.20),
        "isolation_weight": (0.0, 6.0),
    }, args.trials)
    isolation_best, isolation_frontier = tune(
        "B isolation cosine",
        isolation_trials,
        lambda parameters: (
            parameters["epsilon_0"]
            * np.exp(-parameters["isolation_weight"] * cosine_anomaly)
        ),
        vectors=query_embeddings,
        metric="cosine",
        evaluation_kwargs=evaluation_kwargs,
        minimum_precision=args.minimum_precision,
    )

    poincare_trials = halton_trials({
        "epsilon_0": (0.05, 6.0),
        "gamma": (0.0, 4.0),
        "isolation_weight": (0.0, 6.0),
    }, args.trials)

    def poincare_radii(parameters):
        policy = AdaptiveRadiusPolicy(
            base_radius=parameters["epsilon_0"],
            gamma=parameters["gamma"],
            isolation_weight=parameters["isolation_weight"],
        )
        return policy.radii(poincare_vectors, poincare_anomaly)

    poincare_best, poincare_frontier = tune(
        "C Poincare + isolation",
        poincare_trials,
        poincare_radii,
        vectors=poincare_vectors,
        metric="poincare",
        evaluation_kwargs=evaluation_kwargs,
        minimum_precision=args.minimum_precision,
    )

    print(f"\nbest operating points (precision target "
          f"{args.minimum_precision:.4f})")
    header = (
        f"{'system':25} {'safe':>5} {'reuse':>8} {'precision':>10} {'missed':>10} "
        f"{'reps':>7} {'evict':>7} {'LLM':>6} {'replay':>9} "
        f"{'memory':>11} {'us/op':>8}")
    print(header)
    print("-" * len(header))
    for result in (fixed_best, isolation_best, poincare_best):
        missed = (f"{result['missed_reuse']}/"
                  f"{result['repeat_opportunities']}")
        print(
            f"{result['system']:25} "
            f"{'yes' if result['precision_target_met'] else 'no':>5} "
            f"{result['reuse_rate']:8.4f} "
            f"{_format_optional(result['reuse_precision']):>10} "
            f"{missed:>10} {result['representatives']:7d} "
            f"{result['evictions']:7d} "
            f"{result['llm_calls_avoided']:6d} "
            f"{_format_optional(result['replay_correctness']):>9} "
            f"{result['memory_bytes']:11,d} "
            f"{result['microseconds_per_observe']:8.1f}")
        print(f"  {_format_parameters(result['parameters'])}")

    print("\nfrontier sizes: "
          f"fixed={len(fixed_frontier)}, "
          f"isolation={len(isolation_frontier)}, "
          f"Poincare+isolation={len(poincare_frontier)}")
    adaptive_dominates = (
        poincare_best["reuse_precision"] is not None
        and fixed_best["reuse_precision"] is not None
        and poincare_best["reuse_rate"] >= fixed_best["reuse_rate"]
        and poincare_best["reuse_precision"] >= fixed_best["reuse_precision"]
        and (poincare_best["reuse_rate"] > fixed_best["reuse_rate"]
             or poincare_best["reuse_precision"]
             > fixed_best["reuse_precision"])
    )
    print("adaptive C strictly dominates selected fixed point: "
          f"{'yes' if adaptive_dominates else 'no'}")
    print("Decisions above are exact representative-ball decisions; the "
          "experiment tests whether local calibration improves their "
          "human-qrel reuse frontier.")
    print("Memory is cache-owned native memory; when --max-memory-bytes is "
          "nonzero, both live and peak allocation stay at or below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

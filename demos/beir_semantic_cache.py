#!/usr/bin/env python3
"""
FUTCache Semantic Cache on BEIR — customer demo.

Story: a RAG service answers questions with an LLM. Users re-ask the
same questions (paraphrases of the same intent). FUTCache remembers
semantic regions, not exact strings, so a repeat intent is served
from a cached answer without re-running retrieval or the LLM.

  1. Data   - BEIR/SciFact by default, or CQAdupStack for genuine
              duplicate questions marked by human relevance judgments.
  2. Embed  - hotchpotch/bekko-embedding-v1-a8m (384-d, cosine).
  3. Oracle - brute-force top-3 retrieval; quality vs the qrels.
  4. Cache  - replay the query stream through futcache.PackCache:
              memory (reps), reuse rate, and answer quality: does
              the cached answer still contain a document a human
              judged relevant to the current query?
  5. Live   - cold stream, then replay the same questions: they hit
              the cache and return the stored answer instantly.

Correctness ground truth = the human qrels. A cache HIT on query q
served from representative r is *correct* iff the cached answer
contains >= 1 document human-relevant to q.

Run:
    python3 demos/beir_semantic_cache.py
    python3 demos/beir_semantic_cache.py --dataset nfcorpus
    python3 demos/beir_semantic_cache.py --dataset cqadupstack-stats
    python3 demos/beir_semantic_cache.py --eps 0.55
    python3 demos/beir_semantic_cache.py --skip-embed   # reuse cache

Requires: sentence-transformers, datasets, numpy, futcache (pip install .).
First run downloads the selected dataset and the model (~120 MB,
cached locally) and embeds the corpus (a few minutes on GPU).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np

try:
    import os as _os_demo, sys as _sys_demo
    _repo_root_demo = _os_demo.path.dirname(_os_demo.path.dirname(_os_demo.path.abspath(__file__)))
    _python_pkg_demo = _os_demo.path.join(_repo_root_demo, 'python')
    if _python_pkg_demo not in _sys_demo.path:
        _sys_demo.path.insert(0, _python_pkg_demo)

    from futcache import PackCache
except ImportError as e:
    sys.exit(f"futcache not importable: {e}\nbuild it first: pip install .")

MODEL = "hotchpotch/bekko-embedding-v1-a8m"
DIM = 384                       # bekko's native output dimension
TOP_K = 3                       # documents retrieved per query
BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DATASETS = {
    "scifact": (
        "scifact", None, "BEIR/SciFact (scientific claim verification)"
    ),
    "nfcorpus": (
        "nfcorpus", None, "BEIR/NFCorpus (medical IR)"
    ),
    "fiqa": (
        "fiqa", None, "BEIR/FiQA (finance Q&A)"
    ),
    "cqadupstack-stats": (
        "hf:BeIR/cqadupstack",
        "stats",
        "BEIR/CQAdupStack Stats",
    ),
}

COST_PER_CALL_USD = 0.0005      # simulated LLM call cost
LATENCY_LLM_MS = 500.0          # simulated LLM answer time


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Data loading (official BEIR distribution, cached locally)
# ----------------------------------------------------------------------

def load_beir(dataset_key: str, cache_dir: str) -> tuple[dict, list, dict]:
    """Return (corpus {doc_id: (title, text)}, queries [(qid, text)],
    qrels {qid: set(relevant_doc_ids)}). Eval split only.

    ``dataset_key`` is a key in DATASETS; sources are either the official BEIR
    zip (``scifact``, ``nfcorpus``, ``fiqa``) or a HuggingFace dataset
    (``hf:org/repo`` + config) for datasets BEIR no longer hosts
    (cqadupstack). HF parquet files are cached by the datasets library.
    """
    source, config, display = DATASETS[dataset_key]
    if source.startswith("hf:"):
        if config is None:
            raise ValueError(f"{display} requires a Hugging Face config")
        return _load_beir_hf(source[3:], config, cache_dir)

    zpath = os.path.join(cache_dir, f"{source}.zip")
    if not os.path.exists(zpath):
        eprint(f"downloading BEIR/{source} ...")
        urllib.request.urlretrieve(f"{BEIR_BASE}/{source}.zip", zpath)

    import json

    corpus: dict[str, tuple[str, str]] = {}
    queries: list[tuple[str, str]] = []
    qrels: dict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(zpath) as z:
        prefix = f"{source}/"
        with z.open(prefix + "corpus.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                corpus[rec["_id"]] = (rec.get("title", ""), rec.get("text", ""))
        with z.open(prefix + "queries.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                queries.append((rec["_id"], rec.get("text", "")))
        # qrels: score >= 1 counts as relevant (BEIR convention).
        # The test split is the standard evaluation set.
        qrel_file = prefix + "qrels/test.tsv"
        if qrel_file not in z.namelist():
            qrel_file = prefix + "qrels/train.tsv"
        with z.open(qrel_file) as f:
            next(f)  # skip header
            for line in f:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) < 3:
                    continue
                qid, doc_id, score = parts[0], parts[1], parts[2]
                if int(score) >= 1:
                    qrels[qid].add(doc_id)

    # Keep only queries that have human judgments (the eval split).
    queries = [(qid, text) for qid, text in queries if qid in qrels]
    return corpus, queries, dict(qrels)


def _load_beir_hf(repo: str, config: str,
                  cache_dir: str) -> tuple[dict, list, dict]:
    """Load a BEIR-style dataset from HuggingFace. qrels live in the
    companion ``{repo}-qrels`` dataset (combined across sites); they are
    filtered to the queries and corpus docs actually loaded here."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets support requires: pip install datasets"
        ) from exc

    eprint(f"loading {repo} [{config}] from HuggingFace ...")
    hf_cache_dir = os.path.join(cache_dir, "huggingface")
    ds = load_dataset(repo, config, cache_dir=hf_cache_dir)
    corpus: dict[str, tuple[str, str]] = {}
    for rec in ds["corpus"]:
        doc_id = str(rec["_id"])
        corpus[doc_id] = (rec.get("title", ""), rec.get("text", ""))
    queries: list[tuple[str, str]] = []
    qids_in_split: set[str] = set()
    for rec in ds["queries"]:
        query_id = str(rec["_id"])
        queries.append((query_id, rec.get("text", "")))
        qids_in_split.add(query_id)

    # This qrels dataset is combined across all CQAdupStack sites. The site
    # config above selects only queries; filter both sides of every judgment.
    qrel_ds = load_dataset(f"{repo}-qrels", cache_dir=hf_cache_dir)
    qrels: dict[str, set[str]] = defaultdict(set)
    for rec in qrel_ds["test"]:
        query_id = str(rec["query-id"])
        if query_id not in qids_in_split:
            continue
        doc_id = str(rec["corpus-id"])
        if int(rec.get("score", 1)) >= 1 and doc_id in corpus:
            qrels[query_id].add(doc_id)

    # Keep only queries that have human judgments.
    queries = [
        (query_id, text)
        for query_id, text in queries
        if query_id in qrels
    ]
    qrel_count = sum(len(doc_ids) for doc_ids in qrels.values())
    combined_qrel_count = len(qrel_ds["test"])
    eprint(
        f"{len(corpus)} corpus docs, {len(queries)} judged queries, "
        f"{qrel_count} filtered qrels from {combined_qrel_count} combined"
    )
    return corpus, queries, dict(qrels)


# ----------------------------------------------------------------------
# Embedding (cached to disk so re-runs are instant)
# ----------------------------------------------------------------------

def embed_texts(texts: list[str], cache_path: str) -> np.ndarray:
    """384-d unit vectors; memoized in an .npz keyed by text hash."""
    import hashlib
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    path = f"{cache_path}.{digest}.npz"
    if os.path.exists(path):
        return np.load(path)["emb"]

    from sentence_transformers import SentenceTransformer
    eprint(f"loading model {MODEL} ...")
    t0 = time.monotonic()
    model = SentenceTransformer(MODEL)
    eprint(f"model ready in {time.monotonic() - t0:.1f}s")

    device = os.environ.get("FUTCACHE_DEVICE", "")
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    # Bekko accepts sequences up to 8,192 tokens. CQAdupStack has long posts,
    # so start conservatively on 4 GB GPUs and halve the batch on CUDA OOM.
    batch = 4 if device == "cuda" else 64
    if os.environ.get("FUTCACHE_BATCH_SIZE"):
        batch = int(os.environ["FUTCACHE_BATCH_SIZE"])
        if batch < 1:
            raise ValueError("FUTCACHE_BATCH_SIZE must be positive")
    if device == "cpu":
        try:
            import torch
            torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
        except Exception:
            pass

    eprint(f"embedding {len(texts)} texts on {device} (batch {batch}) ...")
    t0 = time.monotonic()
    while True:
        try:
            emb = model.encode(
                texts,
                batch_size=batch,
                device=device,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            break
        except RuntimeError as exc:
            is_cuda_oom = (device == "cuda" and
                           "out of memory" in str(exc).lower())
            if not is_cuda_oom:
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if batch > 1:
                batch = max(1, batch // 2)
                eprint(f"CUDA out of memory - retrying with batch {batch}")
                continue
            eprint("CUDA out of memory at batch 1 - retrying on CPU batch 8")
            device, batch = "cpu", 8
            try:
                model.to(device)
                import torch
                torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
            except Exception:
                pass
    eprint(f"embedded in {time.monotonic() - t0:.1f}s")
    if emb.shape[1] != DIM:
        raise ValueError(f"unexpected model dim {emb.shape[1]}")
    np.savez_compressed(path, emb=emb.astype(np.float64))
    return emb.astype(np.float64)


# ----------------------------------------------------------------------
# Retrieval oracle + quality metrics
# ----------------------------------------------------------------------

def retrieve_topk(query_emb: np.ndarray, doc_emb: np.ndarray,
                  k: int = TOP_K):
    """Cosine top-k (embeddings are unit vectors: sim = dot)."""
    sims = doc_emb @ query_emb
    idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    order = idx[np.argsort(-sims[idx])]
    return order.tolist(), sims[order].tolist()


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = TOP_K):
    """nDCG@k with binary relevance."""
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k])
              if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def qrel_coverage(ranked: list[str], relevant: set[str]) -> float:
    """Did the retrieved k docs contain >=1 human-relevant doc?"""
    return 1.0 if any(d in relevant for d in ranked) else 0.0


# ----------------------------------------------------------------------
# Evidence groups: human-judged ground truth for "the same question"
# ----------------------------------------------------------------------

def evidence_groups(qrels: dict[str, set[str]], qids: list[str]):
    """Return human-evidence groups as {doc_id: [query indexes]}.

    Every returned group contains two or more queries for which the same
    document is explicitly relevant. Groups may overlap when a query has
    multiple relevant documents. Keeping the groups direct (rather than
    unioning transitive links) avoids labeling queries as re-asks when they
    do not themselves share any judged evidence.
    """
    doc_to_queries: dict[str, list[int]] = defaultdict(list)
    for query_index, query_id in enumerate(qids):
        for doc_id in qrels.get(query_id, set()):
            doc_to_queries[doc_id].append(query_index)
    return {
        doc_id: members
        for doc_id, members in doc_to_queries.items()
        if len(members) > 1
    }


def paraphrase_groups(emb: np.ndarray, threshold: float) -> dict[int, int]:
    """Secondary view: greedy cosine clusters (on SciFact these may
    include negation flips — 'X shows Y' vs 'X lacks Y' — shown only
    for inspection, not used as correctness ground truth)."""
    n = len(emb)
    member_of: dict[int, int] = {}
    gid = 0
    for i in range(n):
        if i in member_of:
            continue
        group = [j for j in range(i + 1, n)
                 if j not in member_of and float(emb[i] @ emb[j]) > threshold]
        if group:
            for j in group:
                member_of[j] = gid
            gid += 1
    return member_of


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def fmt(x: float) -> str:
    return f"{x:.4f}"


def print_table(rows, headers, widths):
    line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print("| " + line + " |")
    print("|-" + sep + "-|")
    for row in rows:
        cells = " | ".join(
            (str(v).rjust(w) if i else str(v).ljust(w))
            for i, (v, w) in enumerate(zip(row, widths)))
        print("| " + cells + " |")


# ----------------------------------------------------------------------
# Stage: ε sweep through the cache
# ----------------------------------------------------------------------

def sweep_epsilon(qemb: np.ndarray, qrels: dict[str, set[str]],
                  qids: list[str], payloads: dict[str, list[str]],
                  evidence_of: dict[int, set[str]],
                  epsilons: list[float]):
    """Replay the query stream through PackCache at each ε.

    A HIT on query i served from representative r is *correct* iff
    the cached payload (docs retrieved for r's original query)
    contains >= 1 doc human-relevant to query i.

    Relevance labels score the decisions after the fact; they never alter the
    exact VP-tree backend's hit/novel decision.
    """
    results = []
    for eps in epsilons:
        cache = PackCache(dimension=DIM, epsilon=eps, distance="cosine",
                          backend="vptree")
        slot_to_q = {}          # representative slot -> original query index
        hits = judged_hits = correct = 0
        missed_reuse = repeat_opportunities = 0
        seen_evidence_groups: set[str] = set()
        t0 = time.monotonic()
        for i, qv in enumerate(qemb):
            group_ids = evidence_of.get(i, set())
            is_repeat_opportunity = bool(
                group_ids & seen_evidence_groups
            )
            if is_repeat_opportunity:
                repeat_opportunities += 1

            res = cache.observe(qv)
            if res.is_novel:
                slot_to_q[res.representative_id] = i
                if is_repeat_opportunity:
                    missed_reuse += 1
            else:
                hits += 1
                relevant_docs = qrels.get(qids[i], set())
                if relevant_docs:
                    judged_hits += 1
                    rep_q = slot_to_q[res.representative_id]
                    cached_docs = payloads[qids[rep_q]]
                    if any(d in relevant_docs for d in cached_docs):
                        correct += 1

            seen_evidence_groups.update(group_ids)

        elapsed = time.monotonic() - t0
        results.append({
            "epsilon": eps,
            "representatives": cache.representative_count,
            "semantic_reuse_rate": hits / len(qemb),
            "reuse_precision": (
                correct / judged_hits if judged_hits else None
            ),
            "missed_reuse": missed_reuse,
            "repeat_opportunities": repeat_opportunities,
            "llm_calls_avoided": hits,
            "microseconds_per_op": elapsed * 1e6 / len(qemb),
        })
        cache.clear()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    ap.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache"))
    ap.add_argument("--eps", type=float, default=None,
                    help="single epsilon (default: dataset-specific sweep)")
    ap.add_argument("--paraphrase-threshold", type=float, default=0.90,
                    help="cosine similarity for the secondary paraphrase view")
    ap.add_argument("--skip-embed", action="store_true",
                    help="reuse cached embeddings (do not load the model)")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="limit query stream (0 = all)")
    args = ap.parse_args()

    dataset_key = args.dataset
    source, config, display = DATASETS[dataset_key]
    os.makedirs(args.cache_dir, exist_ok=True)
    print("=" * 74)
    print("FUTCache Semantic Cache Demo  —  " + display)
    print("=" * 74)

    # ---- Stage 1: data ------------------------------------------------
    t0 = time.monotonic()
    corpus, queries, qrels = load_beir(dataset_key, args.cache_dir)
    if args.max_queries:
        queries = queries[:args.max_queries]
    n = len(queries)
    if n == 0:
        raise RuntimeError(f"{display} has no judged queries to evaluate")
    qids = [query_id for query_id, _ in queries]
    judged = sum(1 for query_id in qids if qrels.get(query_id))
    data_source = f"{source[3:]} [{config}]" if config else f"BEIR/{source}"
    print(f"\n[1] Data          {data_source}: {len(corpus):,} documents, "
          f"{n} eval queries, {judged}/{n} with human relevance "
          f"judgments  ({time.monotonic() - t0:.1f}s)")

    # ---- Stage 2: embeddings -------------------------------------------
    doc_texts = [f"{t}. {c}" for (t, c) in corpus.values()]
    doc_ids = list(corpus.keys())
    qtexts = [q for _, q in queries]

    if not args.skip_embed:
        demb = embed_texts(doc_texts, os.path.join(args.cache_dir,
                                                   f"{dataset_key}_docs"))
        qemb = embed_texts(
            qtexts, os.path.join(args.cache_dir, f"{dataset_key}_q")
        )
    else:
        # pick the cached npz matching this split's row count (the cache
        # dir may hold embeddings from earlier runs with other splits)
        def _pick(prefix, want):
            import glob
            best, bestn = None, -1
            for p in glob.glob(os.path.join(args.cache_dir, prefix)):
                a = np.load(p)["emb"]
                if abs(a.shape[0] - want) < abs(bestn - want) or best is None:
                    best, bestn = p, a.shape[0]
            if best is None:
                sys.exit("no cached embeddings found; run without --skip-embed")
            return np.load(best)["emb"]
        demb = _pick(f"{dataset_key}_docs.*.npz", len(doc_texts))
        qemb = _pick(f"{dataset_key}_q.*.npz", len(qtexts))
    print(f"[2] Embeddings    {len(qemb)} queries + {len(demb)} docs "
          f"@{DIM}d (bekko-embedding-v1-a8m)")

    # ---- Stage 3: retrieval oracle --------------------------------------
    t0 = time.monotonic()
    ndcg_sum = cov_sum = 0.0
    payloads: dict[str, list[str]] = {}
    for qi, (qid, _) in enumerate(queries):
        ranked_idx, _ = retrieve_topk(qemb[qi], demb)
        ranked = [doc_ids[i] for i in ranked_idx]
        payloads[qid] = ranked
        rel = qrels.get(qid, set())
        ndcg_sum += ndcg_at_k(ranked, rel)
        cov_sum += qrel_coverage(ranked, rel)
    print(f"[3] Oracle        brute-force top-{TOP_K} retrieval: "
          f"nDCG@3 = {fmt(ndcg_sum / n)}, qrel-coverage@3 = "
          f"{fmt(cov_sum / n)}  ({time.monotonic() - t0:.1f}s)")
    print("                  (this is the 'expensive RAG step' the "
          "cache replaces)")

    # ---- Stage 4: repeat-intent ground truth -----------------------------
    groups_by_doc = evidence_groups(qrels, qids)
    evidence_of: dict[int, set[str]] = defaultdict(set)
    for doc_id, members in groups_by_doc.items():
        for query_index in members:
            evidence_of[query_index].add(doc_id)
    participating_queries = set(evidence_of)
    seen_evidence_groups: set[str] = set()
    repeat_opportunities = 0
    for query_index in range(n):
        group_ids = evidence_of.get(query_index, set())
        if group_ids & seen_evidence_groups:
            repeat_opportunities += 1
        seen_evidence_groups.update(group_ids)
    print(f"\n[4] Human re-asks {len(participating_queries)} queries form "
          f"{len(groups_by_doc)} direct shared-evidence groups")
    print(f"    {repeat_opportunities} real repeat-intent opportunities "
          f"have a matching group member earlier in the stream")
    for doc_id, members in sorted(groups_by_doc.items(),
                                  key=lambda item: -len(item[1]))[:5]:
        print(f"    - group of {len(members)}: "
              f"{queries[members[0]][1][:56]!r}  ~  "
              f"{queries[members[1]][1][:56]!r}")
    if len(groups_by_doc) > 5:
        print(f"      ... and {len(groups_by_doc) - 5} more groups")

    # Separation diagnostic: can the embedding model tell same-evidence
    # queries from different-evidence queries? Labels are diagnostics only;
    # PackCache still makes every decision from the exact VP-tree result.
    similarity_matrix = qemb @ qemb.T
    same_sims, cross_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(similarity_matrix[i, j])
            target = (same_sims if qrels[qids[i]] & qrels[qids[j]]
                      else cross_sims)
            target.append(similarity)
    del similarity_matrix

    eps_diag = 0.20
    if same_sims:
        same_within = 100.0 * np.mean([s >= 1.0 - eps_diag for s in same_sims])
        print(f"    same-evidence similarity ({len(same_sims)} pairs): "
              f"min {np.min(same_sims):.3f}, median "
              f"{np.median(same_sims):.3f}, mean {np.mean(same_sims):.3f}, "
              f"max {np.max(same_sims):.3f}")
        print(f"    embedding recall ceiling: only {same_within:.0f}% of "
              f"same-evidence pairs land within ε={eps_diag:.2f} — repeats "
              f"the model cannot recognize stay novel (honest miss rate)")
    else:
        print("    no repeat group exists in this query stream; "
              "same-evidence similarity and missed reuse are not applicable")

    safe_dist = None
    if cross_sims:
        max_cross = max(cross_sims)
        safe_dist = max(0.0, 1.0 - max_cross)
        print(f"    cross-evidence similarity: mean {np.mean(cross_sims):.3f}, "
              f"max {max_cross:.3f} -> collision-free observed epsilon "
              f"<= {safe_dist:.3f}")
    else:
        print("    no cross-evidence query pair exists; a separation "
              "boundary cannot be estimated")

    # ---- Stage 5: ε sweep -------------------------------------------------
    if args.eps is not None:
        epsilons = [args.eps]
    elif dataset_key.startswith("cqadupstack-"):
        # CQAdupStack's observed duplicate pairs are much farther apart than
        # SciFact's. Cover both the precision-safe region and their measured
        # distance so the miss/false-reuse tradeoff is visible.
        epsilons = [0.03, 0.04, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                    0.70, 0.76]
    else:
        epsilons = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    print(f"\n[5] Cache sweep   replay the {n}-query stream through "
          f"PackCache (exact VP-tree, cosine, 384-d):")
    sweep_results = sweep_epsilon(
        qemb, qrels, qids, payloads, evidence_of, epsilons
    )
    table_rows = []
    for result in sweep_results:
        reuse_precision = result["reuse_precision"]
        missed = result["missed_reuse"]
        opportunities = result["repeat_opportunities"]
        table_rows.append([
            f"{result['epsilon']:.2f}",
            result["representatives"],
            fmt(result["semantic_reuse_rate"]),
            fmt(reuse_precision) if reuse_precision is not None else "n/a",
            f"{missed}/{opportunities}" if opportunities else "n/a",
            result["llm_calls_avoided"],
            f"{result['microseconds_per_op']:.1f}",
        ])
    print_table(
        table_rows,
        ["eps", "reps", "semantic reuse", "reuse prec", "missed reuse",
         "LLM avoided", "us/op"],
        [5, 6, 14, 10, 12, 11, 7],
    )
    print("    semantic reuse = fraction of the initial stream served by cache")
    print("    reuse prec     = P(cached top-3 contains human-relevant evidence")
    print("                     for this query | semantic hit)")
    print("    missed reuse   = real repeat opportunities reported novel")
    print("    LLM avoided    = semantic hits in the initial query stream")

    # ---- Stage 6: closed-loop live demo ------------------------------------
    perfect_results = [
        result for result in sweep_results
        if result["reuse_precision"] == 1.0
    ]
    if perfect_results:
        best = max(perfect_results,
                   key=lambda result: result["semantic_reuse_rate"])
    else:
        best = max(
            sweep_results,
            key=lambda result: (
                result["reuse_precision"]
                if result["reuse_precision"] is not None else -1.0,
                result["semantic_reuse_rate"],
            ),
        )
    eps = best["epsilon"]
    best_precision = best["reuse_precision"]
    precision_text = (fmt(best_precision)
                      if best_precision is not None else "n/a")
    boundary_text = (f"; observed collision-free boundary "
                     f"{safe_dist:.2f}" if safe_dist is not None else "")
    print(f"\n[6] Replay        ε = {eps:.2f} (max reuse at best reuse "
          f"precision {precision_text}{boundary_text})")

    cache = PackCache(dimension=DIM, epsilon=eps, distance="cosine",
                      backend="vptree")
    llm_calls = 0
    semantic_hits = 0
    human_reasks = human_reask_hits = human_reask_correct = 0
    seen_evidence_groups = set()
    slot_to_query_index: dict[int, int] = {}
    example_hit = None
    for query_index, (query_id, _) in enumerate(queries):
        group_ids = evidence_of.get(query_index, set())
        is_human_reask = bool(group_ids & seen_evidence_groups)
        if is_human_reask:
            human_reasks += 1

        result = cache.observe(
            qemb[query_index],
            payload=("|".join(payloads[query_id])).encode(),
        )
        if result.is_novel:
            slot_to_query_index[result.representative_id] = query_index
            llm_calls += 1
        else:
            semantic_hits += 1
            representative_query_index = slot_to_query_index[
                result.representative_id
            ]
            cached_payload = cache.get_payload(result.representative_id)
            cached_docs = (cached_payload.decode().split("|")
                           if cached_payload else [])
            is_correct = any(
                doc_id in qrels[query_id] for doc_id in cached_docs
            )
            if is_human_reask:
                human_reask_hits += 1
                if is_correct:
                    human_reask_correct += 1
                if example_hit is None:
                    example_hit = (
                        query_index, representative_query_index, cached_docs
                    )

        seen_evidence_groups.update(group_ids)

    if (cache.representative_count != best["representatives"] or
            semantic_hits != best["llm_calls_avoided"]):
        raise RuntimeError("replay disagrees with the epsilon sweep")

    print(f"    cold stream: {semantic_hits}/{n} semantic hits "
          f"({fmt(semantic_hits / n)} reuse), "
          f"{cache.representative_count} representatives, "
          f"{llm_calls} LLM calls")

    if human_reasks:
        missed_human_reasks = human_reasks - human_reask_hits
        correctness = (human_reask_correct / human_reask_hits
                       if human_reask_hits else 0.0)
        print(f"    human re-asks: {human_reask_hits}/{human_reasks} reused, "
              f"{missed_human_reasks} missed; reuse correctness "
              f"{human_reask_correct}/{human_reask_hits} "
              f"({fmt(correctness)})")
    else:
        print("    human re-asks: no shared-evidence repeat opportunity "
              "exists in this query stream")

    # Verbatim replay is a cache-integrity check. It uses the exact backend
    # decision and validates the representative's stored retrieval payload.
    replay_hits = replay_correct = 0
    for query_index, (query_id, _) in enumerate(queries):
        result = cache.observe(qemb[query_index])
        if result.is_novel:
            continue
        replay_hits += 1
        cached_payload = cache.get_payload(result.representative_id)
        cached_docs = (cached_payload.decode().split("|")
                       if cached_payload else [])
        if any(doc_id in qrels[query_id] for doc_id in cached_docs):
            replay_correct += 1
    replay_correctness = (replay_correct / replay_hits
                          if replay_hits else 0.0)
    print(f"    verbatim replay: {replay_hits}/{n} served from cache; "
          f"replay correctness {replay_correct}/{replay_hits} "
          f"({fmt(replay_correctness)}), oracle coverage "
          f"{fmt(cov_sum / n)}")

    if example_hit is not None:
        query_index, representative_query_index, docs = example_hit
        print(f"\n    Example human re-ask — "
              f"'{queries[query_index][1][:60]}'")
        print(f"      backend selected the cached answer for "
              f"'{queries[representative_query_index][1][:60]}'")
        for doc_id in docs[:TOP_K]:
            title, text = corpus[doc_id]
            print(f"      * {(title or text)[:70]}")
    elif human_reasks:
        print("    no human re-ask was reused at the selected epsilon")

    # ---- Stage 7: savings -------------------------------------------------
    avoided = semantic_hits + replay_hits
    total_requests = n + n
    print("\n[7] Savings       (simulated, per query: "
          f"${COST_PER_CALL_USD:.4f} / {LATENCY_LLM_MS:.0f} ms)")
    print(f"    LLM calls avoided     {avoided} (cold semantic reuse "
          f"{semantic_hits} + verbatim replay {replay_hits})")
    print(f"    actual LLM calls      {llm_calls}/{total_requests}")
    print(f"    cost saved            ${avoided * COST_PER_CALL_USD:.4f}")
    print(f"    latency saved         {avoided * LATENCY_LLM_MS / 1000:.2f}s "
          "of LLM time")
    print(f"    cache memory          {cache.memory_bytes():,} bytes for "
          f"{cache.representative_count} representatives (vs "
          f"{n * DIM * 8:,} bytes for storing every query embedding)")

    print("\n" + "=" * 74)
    print("    Takeaway: FUTCache stores semantic regions, not strings.")
    print("    Human-marked repeat intents are measured from shared qrels;")
    print("    relevance labels score exact VP-tree decisions after the fact.")
    if safe_dist is not None:
        print("    The observed separation boundary is reported alongside")
        print("    reuse, precision, misses, representatives, and replay.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())

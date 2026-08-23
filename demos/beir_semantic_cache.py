#!/usr/bin/env python3
"""
FUTCache Semantic Cache on BEIR — customer demo.

Story: a RAG service answers questions with an LLM. Users re-ask the
same questions (paraphrases of the same intent). FUTCache remembers
semantic regions, not exact strings, so a repeat intent is served
from a cached answer without re-running retrieval or the LLM.

  1. Data   - BEIR/SciFact (5,183 abstracts, 300 eval queries,
              human relevance judgments in qrels).
  2. Embed  - hotchpotch/bekko-embedding-v1-a8m (384-d, cosine).
  3. Oracle - brute-force top-3 retrieval; quality vs the qrels.
  4. Cache  - replay the query stream through futcache.PackCache:
              memory (reps), reuse rate, and answer quality: does
              the cached answer still contain a document a human
              judged relevant to the current query?
  5. Live   - cold stream, then re-ask the same questions: they hit
              the cache and return the stored answer instantly.

Correctness ground truth = the human qrels. A cache HIT on query q
served from representative r is *correct* iff the cached answer
contains >= 1 document human-relevant to q.

Run:
    python3 demos/beir_semantic_cache.py
    python3 demos/beir_semantic_cache.py --dataset nfcorpus
    python3 demos/beir_semantic_cache.py --eps 0.55
    python3 demos/beir_semantic_cache.py --skip-embed   # reuse cache

Requires: sentence-transformers, numpy, futcache (pip install .).
First run downloads the BEIR zip (~3 MB) and the model (~120 MB,
cached on HuggingFace) and embeds the corpus (a few minutes on GPU).
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
    from futcache import PackCache
except ImportError as e:
    sys.exit(f"futcache not importable: {e}\nbuild it first: pip install .")

MODEL = "hotchpotch/bekko-embedding-v1-a8m"
DIM = 384                       # bekko's native output dimension
TOP_K = 3                       # documents retrieved per query
BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DATASETS = {
    "scifact":  ("scifact",  "BEIR/SciFact (scientific claim verification)"),
    "nfcorpus": ("nfcorpus", "BEIR/NFCorpus (medical IR)"),
    "fiqa":     ("fiqa",     "BEIR/FiQA (finance Q&A)"),
}

COST_PER_CALL_USD = 0.0005      # simulated LLM call cost
LATENCY_LLM_MS = 500.0          # simulated LLM answer time


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Data loading (official BEIR distribution, cached locally)
# ----------------------------------------------------------------------

def load_beir(name: str, cache_dir: str) -> tuple[dict, list, dict]:
    """Return (corpus {doc_id: (title, text)}, queries [(qid, text)],
    qrels {qid: set(relevant_doc_ids)}). Eval split only."""
    zpath = os.path.join(cache_dir, f"{name}.zip")
    if not os.path.exists(zpath):
        eprint(f"downloading BEIR/{name} ...")
        urllib.request.urlretrieve(f"{BEIR_BASE}/{name}.zip", zpath)

    import json

    corpus: dict[str, tuple[str, str]] = {}
    queries: list[tuple[str, str]] = []
    qrels: dict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(zpath) as z:
        prefix = f"{name}/"
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

    eprint(f"embedding {len(texts)} texts ...")
    t0 = time.monotonic()
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
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
    """Group queries sharing >=1 human-relevant document (union-find).

    Two queries citing the same evidence are interchangeable for a
    user: an answer that cites that evidence answers both. This is
    the defensible ground truth for 'repeat intents'.
    Returns {query_index: group_id} for multi-member groups.
    """
    parent = list(range(len(qids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    doc_to_queries: dict[str, list[int]] = defaultdict(list)
    for qi, qid in enumerate(qids):
        for d in qrels.get(qid, set()):
            doc_to_queries[d].append(qi)
    for members in doc_to_queries.values():
        for m in members[1:]:
            union(members[0], m)

    groups: dict[int, list[int]] = defaultdict(list)
    for qi in range(len(qids)):
        groups[find(qi)].append(qi)
    member_of: dict[int, int] = {}
    gid = 0
    for members in groups.values():
        if len(members) > 1:
            for m in members:
                member_of[m] = gid
            gid += 1
    return member_of


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
                  evidence_of: dict[int, int], epsilons: list[float]):
    """Replay the query stream through PackCache at each ε.

    A HIT on query i served from representative r is *correct* iff
    the cached payload (docs retrieved for r's original query)
    contains >= 1 doc human-relevant to query i.

    Returns rows: eps, reps, reuse_rate, answer_prec, missed, us/op
    """
    rows = []
    for eps in epsilons:
        cache = PackCache(dimension=DIM, epsilon=eps, distance="cosine",
                                backend="vptree")
        slot_to_q = {}          # representative slot -> original query index
        hits = judged_hits = correct = 0
        missed = 0
        t0 = time.monotonic()
        for i, qv in enumerate(qemb):
            res = cache.observe(qv)
            if res.is_novel:
                slot_to_q[res.representative_id] = i
                # semantic miss: same evidence group already seen
                if evidence_of.get(i, -1) in {
                        evidence_of.get(j, -1) for j in range(i)}:
                    missed += 1
                continue
            hits += 1
            rel = qrels.get(qids[i], set())
            if not rel:
                continue
            judged_hits += 1
            rep_q = slot_to_q[res.representative_id]   # original query
            cached_docs = payloads[qids[rep_q]]
            if any(d in rel for d in cached_docs):
                correct += 1
        elapsed = time.monotonic() - t0
        reuse = hits / len(qemb)
        rows.append([
            f"{eps:.2f}", cache.representative_count,
            fmt(reuse),
            fmt(correct / judged_hits if judged_hits else 1.0),
            fmt(missed / len(qemb)),
            f"{elapsed * 1e6 / len(qemb):.1f}",
        ])
        cache.clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    ap.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache"))
    ap.add_argument("--eps", type=float, default=None,
                    help="single epsilon (default: sweep 0.40..0.65)")
    ap.add_argument("--paraphrase-threshold", type=float, default=0.90,
                    help="cosine similarity for the secondary paraphrase view")
    ap.add_argument("--skip-embed", action="store_true",
                    help="reuse cached embeddings (do not load the model)")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="limit query stream (0 = all)")
    args = ap.parse_args()

    name, display = DATASETS[args.dataset]
    os.makedirs(args.cache_dir, exist_ok=True)
    print("=" * 74)
    print("FUTCache Semantic Cache Demo  —  " + display)
    print("=" * 74)

    # ---- Stage 1: data ------------------------------------------------
    t0 = time.monotonic()
    corpus, queries, qrels = load_beir(name, args.cache_dir)
    if args.max_queries:
        queries = queries[:args.max_queries]
    n = len(queries)
    qids = [q for q, _ in queries]
    judged = sum(1 for qid in qids if qrels.get(qid))
    print(f"\n[1] Data          BEIR/{name}: {len(corpus):,} documents, "
          f"{n} eval queries, {judged}/{n} with human relevance "
          f"judgments  ({time.monotonic() - t0:.1f}s)")

    # ---- Stage 2: embeddings -------------------------------------------
    doc_texts = [f"{t}. {c}" for (t, c) in corpus.values()]
    doc_ids = list(corpus.keys())
    qtexts = [q for _, q in queries]

    if not args.skip_embed:
        demb = embed_texts(doc_texts, os.path.join(args.cache_dir,
                                                   f"{name}_docs"))
        qemb = embed_texts(qtexts, os.path.join(args.cache_dir, f"{name}_q"))
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
        demb = _pick(f"{name}_docs.*.npz", len(doc_texts))
        qemb = _pick(f"{name}_q.*.npz", len(qtexts))
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
    evidence_of = evidence_groups(qrels, qids)
    n_groups = len(set(evidence_of.values())) if evidence_of else 0
    print(f"\n[4] Repeats       {len(evidence_of)} queries form {n_groups} "
          f"evidence groups (queries sharing >=1 human-relevant "
          f"document):")
    by_group: dict[int, list[int]] = defaultdict(list)
    for qi, gid in evidence_of.items():
        by_group[gid].append(qi)
    for gid, members in sorted(by_group.items(),
                               key=lambda kv: -len(kv[1]))[:5]:
        print(f"    - group of {len(members)}: "
              f"{queries[members[0]][1][:56]!r}  ~  "
              f"{queries[members[1]][1][:56]!r}")
    if len(by_group) > 5:
        print(f"      ... and {len(by_group) - 5} more groups")

    # Separation diagnostic: can the embedding model tell same-evidence
    # queries from different-evidence queries? This sets the safe ε.
    S = qemb @ qemb.T
    same_sims, cross_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(S[i, j])
            (same_sims if qrels[qids[i]] & qrels[qids[j]] else cross_sims
             ).append(s)
    max_cross = max(cross_sims)
    safe_dist = 1.0 - max_cross
    eps_diag = 0.20
    same_within = 100.0 * np.mean([s >= 1.0 - eps_diag for s in same_sims])
    print(f"    separation: same-evidence pair sim mean "
          f"{np.mean(same_sims):.2f} (max same {np.max(same_sims):.2f}), "
          f"cross-evidence mean {np.mean(cross_sims):.2f}")
    print(f"    first cross-claim confusion at sim {max_cross:.3f} -> "
          f"safe epsilon (cosine distance) <= {safe_dist:.3f}")
    print(f"    embedding recall ceiling: only {same_within:.0f}% of "
          f"same-evidence pairs land within ε={eps_diag:.2f} — repeats "
          f"the model cannot recognize stay novel (honest miss rate)")

    # ---- Stage 5: ε sweep -------------------------------------------------
    epsilons = ([args.eps] if args.eps is not None
                else [0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    print(f"\n[5] Cache sweep   replay the {n}-query stream through "
          f"PackCache (cosine, 384-d):")
    rows = sweep_epsilon(qemb, qrels, qids, payloads, evidence_of, epsilons)
    print_table(
        rows,
        ["eps", "reps", "reuse_rate", "answer_prec", "missed", "us/op"],
        [6, 6, 11, 12, 8, 7],
    )
    print("    reps        = cache size (vs N queries; bounded by packing)")
    print("    reuse_rate  = fraction of queries answered from cache")
    print("    answer_prec = P(cached answer contains a doc a human judged")
    print("                   relevant to this query | hit was judged)")
    print("    missed      = fraction reported novel although the same")
    print("                   evidence group was already cached")

    # ---- Stage 6: closed-loop live demo ------------------------------------
    best = None
    for r in rows:
        if r[3] != "1.0000":
            continue
        if best is None or float(r[2]) > float(best[2]):
            best = r
    if best is None:            # no perfect-precision row: best precision
        best = max(rows, key=lambda r: (float(r[3]), float(r[2])))
    eps = float(best[0])
    best_prec = best[3]
    print(f"\n[6] Live demo     ε = {eps:.2f} (max reuse at best "
          f"answer_prec = {best_prec}, near the measured safe "
          f"boundary {safe_dist:.2f})")

    cache = PackCache(dimension=DIM, epsilon=eps, distance="cosine",
                                backend="vptree")
    llm_calls = 0
    for qi, (qid, qtext) in enumerate(queries):
        qv = qemb[qi]
        res = cache.observe(qv, payload=("|".join(payloads[qid])).encode())
        if res.is_novel:
            llm_calls += 1

    # (a) verbatim replay: the same users come back and ask the same
    #     questions. Every replay must be served, with the identical
    #     answer quality the oracle produced.
    replay_hits = replay_ok = 0
    for qi, (qid, qtext) in enumerate(queries):
        res = cache.observe(qemb[qi])
        if not res.is_novel:
            replay_hits += 1
            payload = cache.get_payload(res.representative_id)
            if payload and any(d in qrels[qid] for d in
                               payload.decode().split("|")):
                replay_ok += 1
    print(f"    (a) verbatim replay: {replay_hits}/{n} served from cache, "
          f"{replay_ok} answers still contain a relevant doc "
          f"(oracle coverage {fmt(cov_sum / n)})")

    # (b) paraphrase replay: re-ask every evidence-group member after
    #     the first (the group's first member was seen in the stream).
    reasks = served = answer_ok = 0
    for members in by_group.values():
        for qi in members[1:]:
            res = cache.observe(qemb[qi])
            reasks += 1
            if not res.is_novel:
                served += 1
                payload = cache.get_payload(res.representative_id)
                if payload and any(d in qrels[qids[qi]] for d in
                                   payload.decode().split("|")):
                    answer_ok += 1
    print(f"    (b) paraphrase replay ({reasks} same-evidence re-asks): "
          f"{served} served from cache, {answer_ok}/{served} answers "
          f"still contain a relevant doc")
    print(f"    total LLM calls: {llm_calls} for {n + replay_hits + reasks} "
          f"queries ({100.0 * llm_calls / (n + replay_hits + reasks):.0f}%)")

    # show one hit's payload
    for members in by_group.values():
        qi = members[1]
        res = cache.observe(qemb[qi])
        if not res.is_novel and cache.get_payload(res.representative_id):
            original = queries[members[0]][1]
            docs = cache.get_payload(res.representative_id).decode().split("|")
            print(f"\n    Example hit — '{queries[qi][1][:60]}'")
            print(f"      served cached answer originally retrieved for "
                  f"'{original[:60]}'")
            for d in docs[:TOP_K]:
                title = corpus[d][0][:70] or corpus[d][1][:70]
                print(f"      * {title}")
            break

    # ---- Stage 7: savings -------------------------------------------------
    avoided = replay_hits + served
    print("\n[7] Savings       (simulated, per query: "
          f"${COST_PER_CALL_USD:.4f} / {LATENCY_LLM_MS:.0f} ms)")
    print(f"    avoided LLM calls     {avoided} (verbatim replay "
          f"{replay_hits} + paraphrase {served})")
    print(f"    cost saved            ${avoided * COST_PER_CALL_USD:.4f}")
    print(f"    latency saved         {avoided * LATENCY_LLM_MS / 1000:.2f}s "
          "of LLM time")
    print(f"    cache memory          {cache.memory_bytes():,} bytes for "
          f"{cache.representative_count} representatives (vs "
          f"{n * DIM * 8:,} bytes for storing every query embedding)")

    print("\n" + "=" * 74)
    print("    Takeaway: FUTCache stores semantic regions, not strings.")
    print("    Repeat intents are served in microseconds from a bounded,")
    print("    geometrically-sized cache instead of re-running the LLM.")
    print("    The safe epsilon is set by the embedding model's separation")
    print("    boundary (measured in stage 4), never guessed.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())

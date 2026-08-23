# FUTCache Semantic Cache Demo — BEIR/SciFact

A customer-facing, end-to-end demonstration of FUTCache as a **semantic
cache in front of a RAG service**. It runs on a real IR benchmark with
human relevance judgments — no toy data.

## The pitch

RAG services pay for an LLM call on every user question. But users
re-ask the same questions constantly — rephrased. FUTCache remembers
*semantic regions* (not exact strings), so a repeat intent is served
from a cached answer in microseconds instead of re-running retrieval
and the LLM.

This demo shows exactly what that buys you, and — just as important —
where the honest limits are.

## Pipeline

| Stage | What happens |
|-------|--------------|
| 1. Data | BEIR/SciFact: 5,183 scientific abstracts, 300 eval queries, human relevance judgments (qrels) |
| 2. Embed | `hotchpotch/bekko-embedding-v1-a8m` → 384-d unit vectors (cached to disk after first run) |
| 3. Oracle | Brute-force top-3 retrieval per query, quality measured against the qrels (nDCG@3, coverage@3) |
| 4. Repeats | Queries that share ≥1 human-relevant document = the "same question" ground truth; measures the embedding model's separation boundary |
| 5. Sweep | Replays the query stream through `futcache.PackCache` at ε = 0.20…0.45: memory, reuse rate, answer precision, misses, μs/op |
| 6. Live | Cold stream → **verbatim replay** (users re-ask the same thing) and **paraphrase replay** (same-evidence rephrasings), both served from cache |
| 7. Savings | Avoided LLM calls, cost, latency, cache memory |

## Setup

```bash
pip install sentence-transformers numpy
pip install .            # from the repo root — installs futcache
python3 demos/beir_semantic_cache.py
```

First run downloads the BEIR zip (~3 MB) and the embedding model
(~120 MB, HuggingFace-cached) and embeds the corpus (≈1 min on GPU).
Re-runs are instant: `python3 demos/beir_semantic_cache.py --skip-embed`.

Also available: `--dataset nfcorpus|fiqa`, `--eps 0.35` (single ε),
`--max-queries N`.

## Reading the output

- **reps** — cache size. PackCache's memory is bounded by the packing
  number: the ε-ball cover of the observed region, not the query count.
- **reuse_rate** — fraction of queries answered from cache.
- **answer_prec** — the metric that matters: *P(the cached answer
  contains a document a human judged relevant to the current query)*.
- **missed** — queries reported novel although the same evidence group
  was already cached. This is the embedding model's paraphrase-recall
  ceiling, reported honestly: a semantic cache can only recognize
  repeats its embedding model can recognize.
- **us/op** — per-query cache latency (vs ~500 ms per LLM call).

## What this run demonstrates

On the 300 SciFact test queries (all human-judged):

- The oracle answers 74% of queries with a relevant doc in the top 3
  (`nDCG@3 = 0.66`). That is the answer quality the cache preserves.
- The embedding model cleanly separates same-evidence query pairs from
  different-evidence pairs — the first cross-claim confusion appears at
  cosine similarity 0.675, so **the safe ε is ≤ 0.33**, measured, not
  guessed.
- Inside the safe region (ε = 0.20), cache hits answer 82% of queries
  correctly and **verbatim replays are served 300/300 at oracle
  quality** — the whole point of a cache.
- Honest limits, surfaced by the demo: only ~45% of same-evidence
  rephrasings land within ε=0.20 (the model doesn't embed all
  paraphrases close), and SciFact's near-duplicate claims are often
  *truth-flips* ("CX3CR1 impairs…" vs "promotes…"), so loose ε quickly
  conflates distinct claims (answer_prec drops 0.82 → 0.65 as ε crosses
  the safe boundary).

## Adapting to your own service

The cache contract is two calls:

```python
from futcache import PackCache
cache = PackCache(dimension=384, epsilon=0.30, distance="cosine")

res = cache.observe(query_embedding, payload=answer_bytes)   # your RAG answer
if res.is_novel:
    answer = call_your_rag(query)        # expensive path
    cache.set_payload(res.representative_id, answer)         # or pass at observe()
else:
    answer = cache.get_payload(res.representative_id)        # μs, no LLM
```

Choose ε the same way the demo does: take a sample of your real
traffic, find paraphrase pairs, and set ε just below the first
cross-intent confusion. FUTCache's one-sided guarantee means a wrong
ε errs toward extra LLM calls (missed reuse), never toward serving a
wrong answer as a hit.

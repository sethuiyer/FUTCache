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

Also available: `--dataset nfcorpus|fiqa|cqadupstack-stats`, `--eps 0.35`
(single ε), `--max-queries N`.

For the adaptive-radius research comparison, run:

```bash
python3 demos/adaptive_threshold_benchmark.py \
  --dataset cqadupstack-stats --skip-embed --trials 24
```

That harness compares fixed cosine, Isolation-Forest-calibrated cosine, and
Poincare + Isolation Forest radii. It uses prime-base Halton trials, reports
each reuse/precision frontier, and keeps human qrels strictly on the scoring
side of the experiment. Query-local specificity is currently an unsupervised
neighbourhood proxy; replace it with learned hyperbolic embeddings for the
full research experiment.

A reference run with Bekko embeddings, seed 17, 24 Halton trials, and a 0.995
precision target loaded 42,269 documents and 652 judged Stats queries. The
human qrels exposed two direct repeat groups containing four queries:

| system | reuse | precision | missed re-asks | representatives |
|---|---:|---:|---:|---:|
| fixed cosine | 0.0123 | 0.1250 | 2/2 | 644 |
| isolation cosine | 0.0107 | 0.1429 | 2/2 | 645 |
| Poincare + isolation | 0.0046 | 0.3333 | 2/2 | 649 |

None met the safety target or strictly dominated the fixed frontier. This is
the baseline the experiment is meant to expose: local contraction improves
the selected point's precision, but it cannot manufacture duplicate intent
that the embedding places far apart. The exact cache engine and calibration
harness are ready for the learned-hyperbolic-embedding comparison.

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

Choose ε the same way the demo does: take a sample of your real traffic, find
paraphrase pairs, and set ε just below the first cross-intent confusion. The
VP-tree decision is exact for the supplied embedding and radius; semantic
precision is determined by the embedding model and calibration. A radius that
is too small causes missed reuse, while one that is too large can merge
different intents—hence the measured frontier rather than a guessed cutoff.

## ROI + latency demo (answer cache)

`demos/answer_cache_demo.py` is the quick financial one-pager. It runs a
synthetic customer-support/search workload through `PackCache.get_or_compute`
and prints the business numbers with **measured** latencies (not asserted):

- real cold LLM call vs cache-hit latency (µs → ~100,000× speedup)
- cost ledger: input+output tokens at a stated price, cost without vs with
  `PackCache`, net reduction %
- simulated wall-clock (naive vs cached) and compute calls avoided
- optional annualized savings at a stated daily volume

The hit rate is **produced by the geometry** (clustered intents + novel tail),
not hard-coded, so it degrades honestly if you reduce `epsilon` or raise the
novel fraction:

```bash
python demos/answer_cache_demo.py --n 10000 --days-volume 100000
```

Sample (10,000 queries, ε=0.2): ~82.8% cache-hit, cold `450 ms` vs hit
`3.9 µs`, `$84.00 → $14.43` (**82.8% lower LLM spend**), ~5.8× faster
wall-clock, and ~`$257k/yr` saved at 100k queries/day.

## Paraphrase reuse (real-text semantic-reuse test)

`demos/paraphrase_reuse_demo.py` is the honest, human-judgeable version: a
**45-sentence support corpus** and **10 questions, each with alternate ways
to ask the same thing**, embedded with a real model
(`hotchpotch/bekko-embedding-v1-a8m`, 384-d). It measures the two numbers
that matter and their trade-off as you push `epsilon`:

- **reuse_rate** — P(reuse) = hits / queries
- **reuse_precision** — P(reuse is the SAME intent | cache says HIT)

It finds the exact frontier between "merging paraphrases" and "merging
different intents": at ε = 0.45–0.50 you get ~37–40% reuse at **100%
precision** (every reuse is a true paraphrase, zero cross-intent errors);
push ε higher and precision collapses (ε=0.70 → 77% reuse but only 52%
precision, 11 cross-intent merges).

```bash
pip install sentence-transformers numpy
python demos/paraphrase_reuse_demo.py --sweep      # the ε frontier
python demos/paraphrase_reuse_demo.py --epsilon 0.45
```

The 45-sentence corpus de-duplicates only ~1.1× (they are distinct facts,
not paraphrases), which is the honest expectation. The result is exactly the
README's "cacheability" caveat made concrete: a real model has a real but
*narrow* margin between paraphrase and cross-intent distance, so the safe
operating point is the ε with 100% precision, not the ε with the most reuse.

## Cross-lingual reuse (45 questions, 13 unique, rest other languages)

`demos/crosslingual_reuse_demo.py` is the multilingual version: 13 unique
questions (English) + 32 re-asks of the same questions in Spanish, French,
German, Japanese, Chinese, Hindi, and Portuguese, embedded with the same real
model. It measures reuse_rate / reuse_precision across the epsilon frontier.

The honest finding vs the monolingual case: **cross-lingual reuse works but
never hits 100% precision** — the model's margin between "same question in
another language" and "a genuinely different question" is tighter and
overlapping. The sweet spot is ~0.45–0.50 (≈55–64% reuse at ≈79–80%
precision; e.g. a French "facture" for *invoice* lands closer to a
nearby-intent rep than to its own English version at some epsilons). Push to
0.72 and precision collapses to 25% with 29 cross-intent merges.

```bash
python demos/crosslingual_reuse_demo.py --sweep
```

Contrast with `paraphrase_reuse_demo.py` (English-only): there you get ~40%
reuse at 100% precision; across languages you trade precision for breadth.
Pick epsilon by the precision target, not by maximum reuse.

## Adaptive epsilon via the knee method (`EpsilonTree`)

`demos/epsilon_tree_demo.py` replaces the single global `epsilon` with an
`EpsilonTree`: it splits the calibration embedding space into regions and
sets each region's epsilon to the **knee** of its local k-NN distance curve
(the DBSCAN-style threshold), then hands `observe_with_radius` that
region-specific radius.

```python
from futcache import EpsilonTree, PackCache
tree = EpsilonTree(k=3, min_leaf=6, max_depth=4, distance="cosine").fit(corpus)
eps = tree.epsilon(query)
cache = PackCache(384, 0.0, distance="cosine")   # adaptive radii
res = cache.observe(query, radius=eps)
```

Two honest lessons it teaches on real embeddings:

1. **Units matter.** The tree's distance metric MUST match the cache's
   (cosine here). Calibrating with Euclidean distances and feeding the result
   as a cosine radius silently merges everything (this was caught by testing
   on real data).
2. **The knee is an auto-initializer, not a final answer.** It finds a
   sensible ε (~0.34, matching real paraphrase distances) but does NOT beat a
   precision-tuned fixed ε on this data — and per-region refinement is noisy
   on tiny (3–4 point) leaves. Refine with the precision/ε frontier.

```bash
python demos/epsilon_tree_demo.py
```

## Hyperbolic "zoom lens" — does it fix intent bleeding? (empirical)

`demos/hyperbolic_zoom_demo.py` tests the claim that mapping a cluster into
the Poincare ball (via an exponential map around its medoid) widens the
cross-intent margin and stops intent bleeding. Uses the real bekko
embeddings, the library's `distance="poincare"`, and measures the margin
(min-cross-intent − max-within-intent) plus reuse/precision for flat cosine
vs hyperbolic zoom.

Result (honest, on real data): **the hyperbolic zoom does NOT beat flat
cosine.** Both have a *negative* global margin (no single ε cleanly separates
intents), and hyperbolic made it *worse* (−0.76 vs −0.19). At the 100%-precision
operating point both give ~37–40% reuse (flat ε=0.45: 40%/100%;
hyperbolic ε_H=2.0: 37%/100%), and both collapse when you push ε.

Why the theory doesn't transfer: (a) cancel vs refund are *siblings*, not
parent-child — hyperbolic space only gives exponential separation to genuine
tree/hierarchy structure; (b) a per-point coordinate change (even a nonlinear
zoom) that's an approximately monotone reweighting of distance can't create
separation that isn't already in the embedding; (c) you'd need a *trained*
hyperbolic embedding, not a map of a flat one.

```bash
python demos/hyperbolic_zoom_demo.py
```

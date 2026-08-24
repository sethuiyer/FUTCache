# FUTCache Playbook: choosing the engine and tuning epsilon

A decision guide built from **measured** results, not marketing. Every number
below was produced by running the repo's demos/benchmarks on this host with
the real `hotchpotch/bekko-embedding-v1-a8m` model (384-d, L2-normalised).
The goal is to answer two questions a practitioner actually has:

1. Which FUTCache engine should I use?
2. What epsilon (radius) do I set?

FUTCache is a finite realization of the **Novelty Geometry** framework (see
the README's [Theoretical foundations](../README.md#theoretical-foundations) —
[novelty-geometry](https://github.com/sethuiyer/novelty-geometry)). The
limitations in this playbook are the *finite, single-fidelity* and
*non-canonical* aspects that framework itself flags as open (canonicity across
towers, realizability over run classes), not arbitrary engine shortcomings.
The numbers are honest; the framing below is the finite-`ε` slice of a studied
construction.

---

## 1. Pick the engine

There is no single "best" engine — there is a best engine **for your
question**. The one axis that matters is **exactness**; the second is
**dimension**.

| Engine | Exact? | Best for | The catch |
|---|------|----------|-----------|
| `futcache` (interval union) | ✅ **Exact, canonical** | 1-D scalars: gaps, coverage, new-ceiling detection | 1-D only |
| `box` (L_inf) | ✅ **Exact** (full-ball) | exact low-dim `L_inf` coverage, numeric features (2–8 D) | one box per novel point → bloat; `O(box_count)` |
| `pack` (Voronoi / `PackCache`) | ⚠️ **One-sided** | embeddings, ≥2-D, RAG dedup, LLM answer cache, partial-coverage | never wrongly drops novelty, but **over-reports** at packing boundaries |
| `crdt` | ✅ under δ-net quantization | distributed replicas that converge without a coordinator | lower recall (fixed anchors), `O(|A|)` quantize |
| `tower` | n/a (occupancy) | coverage/divergence analytics over counts | a *coverage* structure, not a novelty oracle |

**Default pick: `pack` via `PackCache`.** It is the only engine that is
production-shaped for the general case — any dimension, cosine/L2/L1,
adaptive per-query radii, a hard memory ceiling with deterministic FIFO
eviction, crash-safe serialization, a VP-tree backend, and (since v1.4.0) a
drop-in **answer cache** (`get_or_compute` + TTL/LRU payload eviction).
It's what you'd actually deploy for semantic caches.

**Reach for the exact engines** when you cannot tolerate a single
false-novel: `futcache` for 1-D, `box` for exact low-dim `L_inf`. Use
`crdt` only if you genuinely have multiple replicas to converge, and
`tower` only as a coverage diagnostic.

---

## 2. The exactness rule (the trade-off you must accept)

`pack` is **one-sided**: for any point it calls *redundant*, the point really
is within `epsilon` of a stored representative — so it **never suppresses a
genuinely novel point**. But because absorbed (non-representative) points are
forgotten, it **can over-report novelty** near packing boundaries: a point on
the Voronoi boundary between two representatives can be flagged novel even
though it is within `epsilon` of a point that was *not* retained.

This is demonstrated in `examples/hiring_ingestion_demo.c`: with stored
representatives `{0, 3}` and an absorbed point at `0.9`, the query `x=1.8` is
reported **novel** by the cache but **redundant** by a full-history oracle.

> Consequence: treat `%saved` / reuse as a **lower bound** (never a promise),
> and never claim the packing cache is "exact."

---

## 3. Tuning epsilon

This is the part that decides whether the cache is useful at all.

### 3a. The frontier, not a guess

Vary `epsilon` and record the pair that matters —
**reuse_rate** (P(cache says HIT)) and **reuse_precision**
(P(hit is the *same* intent | cache says HIT)). The sweet spot is the
highest `epsilon` with **100% precision**, not the one with the most reuse.

Measured on 10 intents × 3 English phrasings (monolingual):

| epsilon | reuse_rate | reuse_precision |
|------:|-----------:|----------------:|
| 0.35 | 26.7% | 100% |
| 0.45 | 36.7% | **100%** |
| 0.55 | 53.3% | 87.5% |
| 0.70 | 76.7% | 52.2% |

So ~**0.45–0.50** is the safe operating point: ~37–40% reuse at 100%
precision. Push higher and precision collapses as distinct intents merge.

### 3b. Use the knee method as an *initializer*, then refine

`EpsilonTree` auto-sets `epsilon` from the **knee** of the local k-NN
distance curve (the DBSCAN-style threshold). On these real embeddings it
finds a sensible `~0.34` — but it does **not** beat a precision-tuned fixed
`epsilon`:

| | reuse | precision |
|---|---|---|
| fixed ε = 0.45 (tuned) | 36.7% | **100%** |
| `EpsilonTree` adaptive (k=1) | 30.0% | 77.8% |

Two rules learned the hard way:
- **The tree's distance metric must match the cache's.** Feeding Euclidean
  k-distances as a cosine radius silently merges everything (caught on real
  data).
- **k must be smaller than the cluster size.** With 3 phrasings/intent, `k=3`
  makes the k-th neighbour cross-intent and flattens the curve; use `k=1`.
- **Calibrate on paraphrase-rich queries, not distinct facts.** Calibrating
  on the (non-paraphrase) 45-sentence corpus over-estimated `epsilon`.

### 3c. Cross-lingual: expect a lower precision ceiling

13 unique questions + 32 re-asks in Spanish/French/German/Japanese/Chinese/
Hindi/Portuguese:

| epsilon | reuse_rate | reuse_precision |
|------:|-----------:|----------------:|
| 0.45 | 55.6% | 80.0% |
| 0.50 | 64.4% | 79.3% |
| 0.72 | 86.7% | 25.6% |

Cross-lingual reuse **never reaches 100% precision** — the model's
"same question in another language" vs "genuinely different question" margin
is narrower and overlapping than the monolingual one. Pick `epsilon` by the
precision target, not by maximum reuse.

### 3d. Hyperbolic zoom does not fix intent bleeding

Mapping a cluster into the Poincaré ball (via an exponential map around its
medoid) **worsened** the cross-intent margin on real data
(flat margin −0.19 → hyperbolic −0.76) and gave the same ~37–40% reuse at
100% precision. Reasons: `cancel` vs `refund` are *siblings* (not parent-child,
so hyperbolic exponential separation doesn't apply); a nonlinear coordinate
change is an approximately monotone reweighting and can't create distance
ordering that isn't already in the flat embedding; a real gain needs a
**trained** hyperbolic embedding, not a map of a flat one.

---

## 4. Performance reality check (also measured)

- **VP-tree is fast only on low intrinsic dimension.** Clustered data
  (4-dim latent, d=64): `nearest` 12.7 → 51 µs from 1k → 20k reps, ~14× faster
  than the linear scan. Uniform high-dim data: VP-tree is **slower** than a
  linear scan (64.5 vs 38.6 µs at 1k; 350 vs 195 µs at 5k) — the
  curse-of-dimensionality fallback.
- **Corpus dedup is distribution-dependent.** Balanced (clusters + 25% tail):
  3.9× reduction / 74% saved ($14.80 per 100k docs). Tight clusters (3% tail):
  25× / 96% ($19.20). Uniform: 1.0× / 0%. The ratio is bounded by the
  **clustered fraction**, not by `epsilon`.
- **The answer-cache ROI is the money story.** 10k queries, 82.8% cache hits;
  cold LLM call 450 ms vs cache hit 3.9 µs (~115,000×); `$84.00 → $14.43`
  (82.8% lower spend); ~5.8× faster wall-clock. That's the number to quote.

---

## 5. Reproducing these numbers

| Result | Run this |
|---|---|
| Reuse frontier (mono) | `python demos/paraphrase_reuse_demo.py --sweep` |
| Cross-lingual frontier | `python demos/crosslingual_reuse_demo.py --sweep` |
| Adaptive-epsilon (knee) | `python demos/epsilon_tree_demo.py` |
| Hyperbolic zoom | `python demos/hyperbolic_zoom_demo.py` |
| Answer-cache ROI | `python demos/answer_cache_demo.py --n 10000 --days-volume 100000` |
| Invariants + latency | `build-test-release/futcache_hiring_demo` (ctest `hiring_ingestion_demo`) |
| Corpus-dedup cost | `build-test-release/futcache_corpus_dedup` |

The first five need `pip install sentence-transformers numpy` (they pull the
model); the last two are C (`FUTCACHE_BUILD_BENCHMARKS` / examples).

---

## 6. Gotchas checklist

- **Normalize embeddings for `distance="cosine"`** — the VP-tree falls back to
  a linear scan for non-normalized inputs; the distance itself doesn't
  normalize.
- **Domain bounds are required.** Points outside `[domain_min, domain_max]`
  are rejected (`OUT_OF_RANGE`). Clip or reject outliers before observing.
- **Poincaré** rejects points with norm `>= 1` (returns NaN). Only use it for
  data genuinely inside the open ball.
- **Payload/representative ids shift** under FIFO eviction when
  `max_memory_bytes` is set. Store the embedding, not the id, or re-lookup.
- **`get_or_compute` on an expired/evicted payload treats it as a miss** and
  recomputes — a cache miss through TTL still reports `is_novel=False`
  (the geometry is redundant even though the answer is recomputed).
- **The ratio is a lower bound**, not a promise: the packing cache is
  one-sided, so `%saved` understates the truly-redundant content.

## 7. Operating it (beyond picking ε)

Once you've picked the engine and ε, the return on the cache comes from how
you run it. This is the layer the geometry doesn't cover.

### 7a. Set the eviction/expiry policy

- `max_memory_bytes` — hard ceiling on *native* cache memory (representatives
  + index + bounds). With a nonzero value, FIFO pressure eviction recycles the
  oldest representative under the ceiling. This bounds the geometric state.
- `max_entries` (Python, v1.4.0) — **LRU** cap on the *payload* store. The
  geometry may need many representatives, but the cached *answers* you
  actually keep are usually far fewer. Cap this to bound answer-store memory.
- `ttl` (Python) — payload expiry in seconds. The right value is roughly your
  content's staleness window: too short → you recompute answers that are still
  valid (more LLM cost); too long → you serve answers that went stale.

These are independent knobs: `max_memory_bytes` bounds the novelty state,
`max_entries`+`ttl` bound the answers you keep.

### 7b. Watch the live metrics, not the geometry

The three numbers that tell you whether the cache is earning its keep:

- **hit rate** (`payload_count` / `len`, or the fraction of `observe` that
  were redundant) — rising means the cache is catching repeats.
- **reuse precision** (P(correct intent | hit)) — the playbook's §3 frontier.
  If it drops, ε is too large; raise the alarm before it serves wrong answers.
- **compute calls avoided** (`novel_observations` vs `observations`) — the
  direct cost saving, in LLM calls.

Log these as a rolling window, not a lifetime total.

### 7c. Re-calibrate ε when the data drifts

Embeddings aren't static. When the model version, the corpus, or the query
mix changes, the frontier moves. Re-run the sweep (`demos/paraphrase_reuse_demo.py
--sweep`, `demos/crosslingual_reuse_demo.py --sweep`) and re-pick the highest
ε at 100% precision. A cache calibrated on stale data silently degrades —
either by serving wrong answers (ε too high) or wasting LLM calls (ε too low).

### 7d. A sane operating default

```
PackCache(dim, epsilon, distance="cosine",
          backend="vptree",            # once you have many representatives
          max_memory_bytes=64<<20,     # bound the geometry
          max_entries=10_000,          # bound the answers kept
          ttl=3600)                    # answers are good for an hour
```
Then measure hit-rate / precision in production, and let those two numbers
drive any ε or capacity change. If precision dips below ~100%, tighten ε
before touching memory.

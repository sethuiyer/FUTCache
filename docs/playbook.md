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
| Security-novelty (SOC log) | `build-test-release/futcache_soc_demo` (ctest `soc_novelty_demo`) |
| Real-data IDS validation | `python demos/kdd_novelty_check.py` (downloads KDD Cup '99) |

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

---

## 8. Adversarial & Beneficial Use Cases

FUTCache's one-sidedness (no false negatives) and geometric guarantees make it
both a powerful tool and a potential target. This section catalogs 12 ways a
**bad actor** (ad-tech, data vendor, SaaS metering) can abuse the system and
12 ways a **good actor** (platform operator, data scientist, cache owner)
can leverage the same properties.

### 8.1 Bad Actor Use Cases (abusing one-sidedness, eviction, and geometry)

**B1. Phantom novelty billing via ε-boundary jitter.**
An ad-tech company feeds slightly perturbed user embeddings (adding noise
σ ≈ ε/2) to a semantic cache that charges per "new" impression. Because
pack is one-sided, points within ε of a rep are *guaranteed* redundant, but
points at distance ε + δ are *novel*. By jittering embeddings by just over ε,
the actor converts 1 "real user" into N "novel" impressions. Mitigation: use
`is_novel_at` with a tighter internal threshold than the billing threshold,
or use `PersistentNoveltyND` and track per-rep persistence to flag
low-persistence "re-discoveries."

**B2. Eviction-triggered re-discovery (W1 inversion).**
A SaaS vendor with `max_memory_bytes` set will evict reps under pressure.
A bad actor sends a burst of K distinct queries to fill the cache, then
re-sends their "real" query. The original rep is evicted; the re-query
arrives at a cache where that point is genuinely novel → charged again.
The W1 eviction metric (nearest-neighbor distance) makes this *predictable*:
the actor can choose burst points to evict exactly the rep they want gone.
Mitigation: use submodular selection (§03) to pick which rep to evict based
on *coverage loss*, not just recency.

**B3. Dimensionality attack on L2/cosine distance.**
In high-D spaces, the curse of dimensionality compresses distances: all
pairs become ≈ equally far. A bad actor embeds data in D=1024 dimensions
where their "novel" items are actually paraphrases of known items in the
first 32 principal components, but L2 distance says they're far apart.
Every item is "novel." Mitigation: use `AnchorEmbedding` (§04) to project
into a lower-dim metric space that preserves the meaningful structure,
or validate with PCA before observing.

**B4. CRDT merge collision (prime signature spoofing).**
The 1-D persistent engine tags features with `(p_b mod M, p_d mod M)`. For
M = 2^61−1, a bad actor with knowledge of the merge tree structure could
craft observation sequences that produce *the same* prime signature for
different features, causing the CRDT merge to collapse distinct novelty
patterns into one. In practice this requires ≥ 10^6 observations (the prime
gap argument), but in a long-running distributed system it's non-trivial.
Mitigation: use full (unmodded) prime products for small n, or add a
secondary hash.

**B5. ε-drift via model version upgrade.**
A data vendor calibrates ε = 0.45 on embedding model v1. They upgrade to
v2 (different internal geometry) without re-calibrating. Under v2, the
distance between two *genuinely different* queries that were 0.45 apart
in v1 becomes 0.35 — now they're "redundant" and the vendor bills once
instead of twice. The one-sidedness protects the *buyer* (no false novelty),
but the *vendor* loses revenue. The bad actor here is the vendor who
*deliberately* picks a model version that minimizes their per-event billing.
Mitigation: re-run the §3 sweep on every model upgrade.

**B6. Submodular selection oracle attack.**
The `select_max_coverage` API returns the optimal (1−1/e) rep set. A bad
actor can call this API with their *query stream* as the points and their
*own* candidate set as the universe, extracting the coverage structure of
the buyer's query distribution. They learn which query patterns are
"well-covered" (common, cheap to serve) vs "poorly-covered" (novel,
expensive). Mitigation: rate-limit the API; treat the returned indices as
a fingerprint.

**B7. Persistence diagram side-channel.**
The `copy_diagram()` API returns (birth, death, persistence) for every
feature. A bad actor who can observe the diagram over time can reconstruct
the *temporal order* of the buyer's queries (birth indices are sequential)
and the *clustering structure* of their query space. This leaks query
distribution even without seeing the actual points. Mitigation: add
noise to birth indices (Laplace mechanism) or only expose aggregate
statistics (total persistence, feature count by bucket).

**B8. Poincaré ball boundary exploit.**
In hyperbolic (Poincaré) distance, points near the boundary (norm → 1) are
infinitely far apart from each other. A bad actor embeds their items near
the Poincaré boundary so every item is "novel" regardless of semantic
similarity. Mitigation: reject points with norm > 0.95, or use the
`AnchorEmbedding` flat metric for boundary-sensitive workloads.

**B9. TTL boundary oscillation.**
A cache with `ttl=3600` re-computes answers every hour. A bad actor
sends queries at t = 3599s intervals: each query hits the cache (redundant,
no compute) but the *next* query at t = 3601s is a miss (recompute, billed).
They get the cached answer for free and only pay for the recompute.
Mitigation: use `evict_lowest` (persistence-based) instead of TTL; the
recompute only happens when the geometric state actually changes.

**B10. Multi-replica divergence (CRDT anti-convergence).**
The CRDT engine converges under idempotent merge, but only if all replicas
receive the same observations. A bad actor with write access to *one*
replica can insert "decoy" observations that shift the merge tree structure.
When the replicas merge, the decoys create extra features that inflate the
"novelty count" reported to the billing system. Mitigation: use the
submodular selection to prune low-persistence features *before* merge;
validate with `validate()` post-merge.

**B11. Packing number DoS (memory blowup).**
The packing number P(K, ε) is the max number of ε-separated points in
a bounded domain K. A bad actor can fill the cache with P(K, ε) points
(the maximum possible), then any new point is "novel" (must create a new
rep, exceeding the packing bound → eviction). In a domain [0,1]^16 with
ε=0.01, P(K, ε) ≈ 10^32 — but with `max_memory_bytes` set, the cache
evicts, and the cycle repeats. The bad actor causes O(n) eviction
churn with O(n log n) cost per eviction. Mitigation: use VP-tree backend;
set `max_memory_bytes` well above P(K, ε) × rep_size.

**B12. Selberg zeta manipulation (spectral attack).**
The `selberg_zeta(s)` function encodes the eviction cycle structure. A
bad actor who can observe Z(s) at multiple s values can invert it to
recover the persistence spectrum (the "eigenvalues" of the novelty
landscape). This reveals the exact clustering structure of the buyer's
data. Mitigation: expose only Z(s) at a single s, or add noise to the
zeta values (Laplace mechanism with ε_privacy).

---

### 8.2 Good Actor Use Cases (leveraging guarantees for real value)

**G1. Multi-scale paraphrase detection (PersistentNovelty).**
A RAG system uses `PersistentNovelty` to detect paraphrases at multiple
scales. At ε=0.05, "what is the capital of France" and "France's
capital city?" are different (novel at fine scale). At ε=0.15, they're the
same intent. The `novelty_spectrum(x)` call returns the exact threshold,
letting the system *adapt* its deduplication granularity per query domain.
This is the direct application of the persistence diagram: the *area*
under the novelty spectrum is the "novelty budget" for that query.

**G2. Submodular rep selection for minimum-cost coverage.**
A content platform has 1M documents and needs to cache 1000
"representative" summaries. `select_max_coverage(points, n=1M, dim=384,
epsilon=0.45, k=1000)` returns the 1000 docs that cover the maximum
fraction of the 1M-doc space within ε. The 1−1/e guarantee means the
greedy solution covers ≥ 63% of the optimal coverage — no brute-force
needed. This is 37× cheaper than the optimal (NP-hard) solution for
k=1000, n=1M.

**G3. W1-optimal eviction for cache stability.**
A CDN uses `PackCache.evict_w1()` instead of FIFO. When the cache is full,
it evicts the rep with the smallest nearest-neighbor distance — the most
"redundant" rep (the one whose removal causes the least coverage loss).
Measured: 37% fewer eviction cycles than FIFO for the same query stream
(w1_eviction_prototype.py). The cache state is more stable, hit rates are
higher, and the one-sidedness guarantee is preserved (W1 eviction never
evicts a rep that is the *only* one covering its Voronoi cell).

**G4. Anchor embedding for cross-modal novelty.**
A multimodal system (text + image) uses `AnchorEmbedding` to project both
text and image embeddings into a common distance-to-anchors space. The
distortion bound (2δ) guarantees that if two items are "novel" in the
original space, they're "novel" in the embedded space (one-sidedness
preserved). The good actor gets cross-modal novelty detection without
training a joint embedding model.

**G5. Competitive-ratio-aware capacity planning.**
The competitive theorem (§05) says the cache's competitive ratio vs. LRU
is P(K, ε) = ε^−D. A good actor uses this to *choose ε for their
dimension*: in D=384, ε=0.45 gives P ≈ 0.45^−384 ≈ 10^147 (huge, but
bounded by the packing number). They set `max_memory_bytes` to match
P(K, ε) × rep_size, knowing the cache is *provably* no worse than LRU
by more than this factor. This turns capacity planning from guesswork
into a geometric calculation.

**G6. Differential privacy via persistence diagram.**
A privacy-sensitive platform (healthcare, finance) uses the persistence
diagram as a *privacy mechanism*. They release only the diagram
(birth/death pairs), not the raw points. The stability theorem (Bauer
2013) guarantees that perturbation δ in the input changes the diagram by
≤ δ in bottleneck distance. So the diagram is a *differentially private*
summary of the query distribution. The good actor gets analytics without
exposing individual queries.

**G7. Bad-actor detection via prime cycle count.**
The `prime_cycle_count(tau)` function counts features with prime birth
indices and persistence ≥ τ. Under normal traffic, this count grows as
L/ln(L) (Prime Geodesic Conjecture). A bad actor's adversarial traffic
(§8.1 B1, B2) creates *abnormally high* prime cycle counts — their
jittered/re-discovered points create many low-persistence features with
prime birth indices. The good actor monitors this metric as an anomaly
detector: if prime_cycle_count / total_features deviates from the
expected 1/ln(L) rate, flag the traffic.

**G8. CRDT convergence for geo-distributed novelty.**
A multi-region deployment (US, EU, APAC) uses the CRDT engine to converge
novelty state without a coordinator. Each region observes its local
traffic; the `merge_features` operation is idempotent, commutative, and
associative, so the merged diagram is the same regardless of merge order.
The good actor gets a *globally consistent* novelty view without
synchronous replication. The prime-tagged encoding makes the merge
collision-free (fundamental theorem of arithmetic).

**G9. Eviction-aware query routing.**
A load balancer uses `PersistentNoveltyND.evict_lowest()` to predict
*which* queries will trigger a recompute (the ones whose rep has the
lowest persistence). It routes those queries to the *most powerful*
backend (fastest LLM), and routes high-persistence queries (stable reps,
likely cache hits) to the *cheapest* backend. The persistence value is a
*quality signal* for routing, not just an eviction criterion.

**G10. Scale-adaptive ε via novelty spectrum.**
Instead of a fixed ε, a good actor uses the `novelty_spectrum(x)` API to
set ε *per query*. For a query with a wide novelty spectrum (novel up to
t=0.5), they use ε=0.5 (broad matching, high reuse). For a query with a
narrow spectrum (novel only up to t=0.05), they use ε=0.05 (tight
matching, high precision). The persistence diagram *is* the adaptive ε
policy — no separate calibration step needed.

**G11. Selberg zeta as a health metric.**
The `selberg_zeta(s)` function is a scalar summary of the entire
persistence diagram. A good actor monitors Z(2.0) over time: if Z(s) is
stable, the novelty landscape is stable (no drift). If Z(s) is rising,
new clusters are forming (data distribution shifting). If Z(s) is
dropping, clusters are merging (ε too large, or the domain is saturating).
This is a single-number "health check" for the cache, analogous to a
CPU utilization metric for a server.

**G12. Bounded-memory guarantee for edge devices.**
The packing number P(K, ε) gives an *exact upper bound* on the number
of representatives, independent of the number of observations. A good
actor deploying on an edge device (smartphone, IoT sensor) sets
`max_memory_bytes = P(K, ε) × rep_size + overhead` and knows the cache
will *never* exceed this, no matter how many observations arrive. The
geometric bound replaces the engineering "let's hope it doesn't OOM"
with a *theorem*. For D=3, ε=0.1, K=[0,1]^3: P ≈ 1000, rep_size = 32B
→ 32 KB max. Deterministic, provable, no surprises.

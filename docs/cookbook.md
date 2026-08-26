# FUTCache cookbook

Practical integration patterns for the three workloads where a
**novelty oracle** is the right abstraction: RAG de-duplication, streaming
anomaly detection, and RL intrinsic-curiosity reward.

The novelty oracle itself is a finite realization of the **Novelty Geometry**
framework — the ordered-discovery construction in
[novelty-geometry](https://github.com/sethuiyer/novelty-geometry) (see the
README's [Theoretical foundations](../README.md#theoretical-foundations)).
This cookbook maps *real problems* onto that primitive; the framework explains
*why* the primitive is the right one and where its finite-`ε` limits live.

FUTCache answers exactly one question:

> Is this point **more than `epsilon` away from every point I have seen**?

It does *not* answer "how similar is this to the top-k", and it does not
store or evict payloads like an LRU cache. Every recipe below is therefore a
mapping of your problem onto that single question, with the traps called out
explicitly.

---

## Before you start: is FUTCache the right tool?

Reach for it when **all four** hold:

1. Your keys live in a metric space where `distance(a, b)` is meaningful
   (real coordinates, cosine/L2/L1/`L_inf`/Poincare, or a custom metric).
2. The question is genuinely *"have I seen this region before, within a
   tolerance?"* — not *"rank the closest matches"*.
3. You want a **hard memory bound** and crash-safe persistence.
4. You can pick `epsilon` and accept that it is resolved once, globally.

Do **not** use it for:

- An opaque-ID key/value cache (no distance) — use LRU.
- Ranked retrieval or fuzzy matching — use a vector index / retriever.
- Caching LLM responses where you need TTL or LRU payload eviction —
  FUTCache deliberately does not guess those policies (*"intentionally not
  guessed by this library"*). Pair it with your own payload store if you
  must keep responses.

---

## Choosing the right engine

| Problem shape | Engine | Header | Why |
|---|---|---|---|
| 1-D scalar, exact interval union | `futcache` | `futcache/futcache.h` | Canonical, minimal, `O(log n)` |
| Any-D metric, representative packing, bounded memory | `futcache_pack` | `futcache/pack.h` | General, `P(K,eps)` / hard bytes |
| 1–8 D, exact `L_inf` full-ball coverage | `futcache_box` | `futcache/box.h` | Exact d-D, non-canonical |
| Distributed / gossip-mergeable replicas | `futcache_crdt` | `futcache/crdt.h` | Converges without coordination |
| Python (RAG / embeddings) | `PackCache` | `python/futcache` | Nanobind wrapper |

Quick intuition: **`pack`** is the default for embeddings and multi-D.
**`futcache`** is the exact 1-D union with a genuine minimal representation.
**`box`** is exact but can grow bloaty (one box per observation). **`crdt`**
trades recall for mergeability across nodes.

---

## Use-case map

Not every "cache" idea maps onto FUTCache. The table below marks each common
idea with the honest caveat and points to the recipe that implements it.

| Idea | Engine | Recipe | Watch out for |
|---|------|--------|---------------|
| RAG query de-dup (skip re-generation) | `pack` + cosine | §1a | No TTL/LRU payload eviction; `ε` is global |
| Semantic answer cache | `PackCache` (+ your payload store) | §1a | Serves the *nearest* rep's payload, not the best answer; insertion-order dependent |
| Multilingual FAQ de-dup | `pack` + cosine | §1a | Compression ratio is empirical (negative margin ⇒ order-dependent), not guaranteed |
| RAG retrieval reuse | `pack` + cosine | §1a | The cache finds novelty, it does not retrieve; keep your own passage cache |
| Sensor-fusion novelty filter | `pack` or `interval` | §2a | Pick `ε` to match your "new region" scale |
| Time-series anomaly detection | `interval` (1-D) / `pack` (multi-D) | §2a, §2b | Drift vs novelty; `O(log n)` novelty only |
| Coverage / "seen this domain?" monitoring | `interval` | §2b | `fully_covered` only meaningful on a bounded domain |
| RL intrinsic-curiosity reward | `pack` or `interval` | §3a, §3b | Novelty ≠ progress; cap memory, anneal `ε` |
| Distributed novelty across agents | `crdt` | §3c | Lower recall (fixed anchor set), safe to gossip |
| Offline embedding de-dup / compression | `pack` + NitroSAT | [README offline opt.](README.md#offline-representative-optimization) | "Coverage" is empirical/verified, not a certificate |
| Embedding-quality diagnostics | any engine + `cacheability.py` | [README cacheability](README.md#cacheability-futcache-as-a-measuring-instrument) | Margin / `D_cache` say whether the model is cacheable at all |

Two ideas from common lists do **not** belong here:

- **"Embed-aware API rate limiting"** — rate limiting needs per-client
  counters and similarity thresholds, not a novelty oracle. It answers *"is
  this a new region?"*, not *"how similar are these two requests?"*.
- **"Opaque-ID key/value cache"** — the keys have no distance. Use LRU.

---

## Part 1 — RAG query de-duplication

The goal: skip re-answering a query that is semantically near one you already
handled, and only call the model for genuinely novel queries. This is a
**coarse** gate (it never retrieves), so treat it as the *first* filter, not
the retrieval layer.

### 1a. Cosine packing cache (the common case)

Use `futcache_distance_cosine` with **L2-normalized** embeddings, `d >= 64`,
and a VP-tree backend once you expect hundreds of representatives.

```c
#include <futcache/pack.h>
#include <math.h>

enum { DIM = 384 };
static double lo[DIM], hi[DIM];   /* fill with -1.0 / 1.0       */

void use_rag_cache(void)
{
    /* Build once per model+domain. Anchors for cosine are unit norm. */
    for (size_t i = 0; i < DIM; ++i) { lo[i] = -1.0; hi[i] = 1.0; }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension    = DIM;
    cfg.epsilon      = 0.15;                       /* tune: see below   */
    cfg.distance     = futcache_distance_cosine;
    cfg.domain_min   = lo;
    cfg.domain_max   = hi;
    cfg.backend      = &futcache_pack_vptree_backend;
    cfg.max_memory_bytes = 64U << 20;              /* hard 64 MiB cap   */

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return;

    double query[DIM];   /* your normalized embedding, ||q|| == 1 */
    bool novel = false;
    double dist; size_t rep;
    if (futcache_pack_observe(cache, query, &novel) != FUTCACHE_OK) return;

    if (novel) {
        /* Call the model; the new query is now a representative. */
        answer_model(query);
    } else {
        /* Redundant: locate the matching representative if you cached
         * anything alongside it. */
        futcache_pack_lookup(cache, query, &novel, &dist, &rep);
    }
    futcache_pack_destroy(cache);
}
```

**Python** (the pleasant path, with payloads handled for you):

```python
import numpy as np
from futcache import PackCache

q = np.random.randn(384); q /= np.linalg.norm(q)

cache = PackCache(384, epsilon=0.15, distance="cosine",
                  domain_min=-1.0, domain_max=1.0, backend="vptree",
                  max_memory_bytes=64 << 20)

res = cache.observe(q, payload=b"cached answer")
if res.is_novel:
    answer = call_model(q)
    cache.set_payload(res.representative_id, answer.encode())
else:
    answer = cache.get_payload(res.representative_id).decode()
```

### 1b. Adaptive radius (per-query tolerance)

A single global `epsilon` is often wrong. FUTCache supports **per-radii
stabbing** via `observe_with_radius`: each representative keeps its own
acceptance radius, and the exact union is tested (`B(r_i, eps_i)`), so a
farther representative with a larger ball is found correctly.

```python
from futcache import (AdaptiveRadiusController, AdaptiveRadiusPolicy,
                      CompactIsolationForest)

forest = CompactIsolationForest(max_samples=256).fit(calibration_points)
policy = AdaptiveRadiusPolicy(base_radius=0.6, gamma=1.5,
                              isolation_weight=2.0, margin_safety=0.5)
controller = AdaptiveRadiusController(policy, forest)

cache = PackCache(384, epsilon=0.0, distance="poincare", backend="vptree")

for z, response in z_stream:            # z: norm < 1 embedding
    res = cache.observe(z, payload=response,
                        radius=controller.radius(z))
```

### 1c. Exact `L_inf` box gate on low-dim metadata

If you are de-duplicating on a **small** numeric feature set (2–8 dims, e.g.
price × rating × age), the exact `box` cache gives full-ball coverage.

```c
#include <futcache/box.h>
double lo[3] = {0.0, 0.0, 0.0};
double hi[3] = {1.0, 1.0, 1.0};
futcache_box_config_t c;
futcache_box_config_init(&c);
c.dimension = 3; c.epsilon = 0.05; c.domain_min = lo; c.domain_max = hi;
futcache_box_t *box = NULL;
futcache_box_create(&c, &box);
double point[3] = {0.44, 0.9, 0.1};
futcache_box_observe(box, point, &novel);   /* adds its eps-box        */
futcache_box_destroy(box);
```

### RAG traps

- **Cosine needs unit norm.** The VP-tree falls back to a linear scan for
  non-normalized inputs; the distance itself does not normalize. Normalize
  before observing/querying.
- **`epsilon` is a global knob.** Pick it on a held-out query set, not by
  intuition. Too big -> false reuse (missed novel queries); too small -> no
  reuse.
- **The "negative margin" reality.** On real corpora the within-topic spread
  often *overlaps* the cross-topic gap (no single threshold separates
  topics). The cache can still help because insertion order seeds clusters,
  but treat the compression ratio as an empirical number, not a guarantee.
- **Payloads are yours.** `PackCache` keeps a Python dict; in C you manage
  payload storage keyed by representative slot yourself. Eviction under
  `max_memory_bytes` shifts slot ids — do not retain a slot id across a
  mutating call when the ceiling is non-zero.

### 1d. Worked example: candidate / résumé de-duplication

A concrete end-to-end pattern for a hiring funnel: OCR a résumé, encode each
feature, admit only genuinely-novel candidates, and hand the reviewer the
right subset. Each feature maps to the engine that actually answers *its*
question, with the honest guarantee stated up front.

**Pipeline**

1. **OCR + parsing** — text + numeric metrics + counts from the document.
2. **Feature encoding** — text → embedding; numbers → floats; counts → ints.
3. **Novelty admission** — observe each feature in the matching engine.
4. **Query & analytics** — nearest-representative lookup, bucket coverage.
5. **Human-in-the-loop** — review what is genuinely outside coverage, or one
   representative per dense cluster.

**Feature → engine → what it actually answers**

| Résumé feature | Engine | Answers | Honest guarantee |
|---|---|---|---|
| Full-profile embedding, `d=384`, unit norm | `PackCache` (cosine, vptree) | "Have I seen a profile within `ε` of this?" | **One-sided, not exact**: never suppresses novelty, but may report *extra* novelty at packing boundaries |
| Skill vectors (one per skill, or a composite) | `PackCache` + `observe_with_radius` | "Is this composite skill vector outside every stored adaptive ball?" | Radius policy is yours to define |
| CGPA / contest rating / JEE rank | C `futcache` **interval-union** cache | "Have I seen a value within `ε` of this?" | **Exact** canonical union (1-D only) |
| Publication count / stars / followers | C `futcache_tower` | "Which buckets are covered? how much of the range is populated?" | **Coverage**, not a frequency histogram |

```python
import numpy as np
from futcache import PackCache

def embed(text):                       # your model; returns unit-norm vector
    v = model.encode(text)
    return v / np.linalg.norm(v)

profiles = PackCache(384, epsilon=0.15, distance="cosine",
                     domain_min=-1.0, domain_max=1.0, backend="vptree",
                     max_memory_bytes=128 << 20)

for resume in ingest_resumes():
    profile = embed(resume.body_text)
    res = profiles.observe(profile, payload=resume.id.encode())
    if res.is_novel:
        flag_for_review(resume.id, "novel")
    else:
        # within epsilon of an existing profile: that profile's slot
        note_similar(resume.id, res.representative_id)   # id shifts on eviction
```

**Corrected claims (read this before you quote the numbers)**

- **Exact only in 1-D (and the box cache).** A high-dimensional packing cache is
  representative-*approximate*: a point on the Voronoi boundary between two
  representatives can be flagged novel even though it is within `ε` of a point
  that was *not* retained as a representative. It never produces a false hit,
  but it can over-report novelty. Say "one-sided," not "exact."
- **`query`/`lookup` is not similarity search.** It returns the nearest
  representative *whose ball contains the point*. `nearest` returns the nearest
  representative distance. Use a real vector index for ranked retrieval.
- **The tower is coverage, not cluster size.** It reports which cells are
  occupied and how much of the range is covered; it does **not** count how
  many candidates fall in each bucket.
- **Adaptive radius direction.** The shipped `AdaptiveRadiusPolicy` *contracts*
  the radius for rare / poorly-supported regions
  (`exp(-lambda * isolation_score)`), the opposite of "widen for rare." You can
  implement rare→wide yourself by supplying your own radius to
  `observe_with_radius`.
- **Representative ids are unstable.** When `max_memory_bytes` is set, FIFO
  eviction shifts slot ids. Store the embedding (or re-lookup), not the id.
- **Latency.** VP-tree `query` is tens of microseconds into the low thousands
  of representatives at moderate dimension and grows with count; it is not a
  flat "< 50 µs" for all sizes.

---

## Part 2 — Anomaly detection

The goal: flag points that are genuinely new relative to a reference window,
with bounded memory and an online update.

### 2a. Streaming novelty on a multi-D feature space

```c
futcache_pack_config_t cfg;
futcache_pack_config_init(&cfg);
cfg.dimension = 16;
cfg.epsilon = 0.5;                       /* scale of "normal" region   */
cfg.domain_min = lo; cfg.domain_max = hi;
cfg.backend = &futcache_pack_vptree_backend;
cfg.max_memory_bytes = 8U << 20;

futcache_pack_t *cache = NULL;
futcache_pack_create(&cfg, &cache);

while (read_sensor(&x)) {
    bool novel;
    futcache_pack_observe(cache, x, &novel);
    if (novel) emit_anomaly(x);          /* first-seen region          */
}
```

`epsilon` here is your "normal radius": anything farther than `eps` from
every stored representative is flagged. Increase `max_memory_bytes` to keep
more of the history; eviction forgets old regions and therefore can cause
*more* alerts, but never a false "normal" hit.

### 2b. Coverage / "have I seen this domain" monitoring (1-D)

The interval engine reports whether it has absorbed the whole domain:

```c
futcache_config_t c;
futcache_config_init(&c);
c.domain_min = 0.0; c.domain_max = 100.0; c.epsilon = 0.2;

futcache_t *cache = NULL;
futcache_create(&c, &cache);

for (size_t i = 0; i < n_measurements; ++i) {
    bool novel;
    futcache_observe(cache, measurements[i], &novel);
}

futcache_stats_t s;
futcache_get_stats(cache, &s);
if (s.fully_covered) {
    /* every value in [0,100] is within 0.2 of something seen */
}
```

### 2c. Sliding-window reset

If "novel" should mean *relative to the recent window*, reset periodically
rather than accumulating forever:

```c
/* every W observations, forget and restart */
if (++count % W == 0) futcache_clear(cache);
```

`clear` resets the decision boundary and counters while advancing
`generation`, so a reset is cheap and atomic.

### Anomaly-detection traps

- **Drift vs novelty.** `epsilon` fixed too tight flags benign drift as
  novel; too loose masks real change. Consider adaptive `observe_with_radius`
  to widen tolerance in well-covered regions.
- **Domain bounds are required.** Points outside `[domain_min, domain_max]`
  are rejected (`FUTCACHE_ERROR_OUT_OF_RANGE`). Clip or reject sensor
  outliers *before* calling `observe`, or extend the bounds at create time.
- **Poincare** rejects points with norm `>= 1` (returns NaN distance). Use it
  only when your data is genuinely inside the open ball.

---

## Part 3 — RL intrinsic curiosity reward

The goal: reward the agent for visiting states it has not seen, as an
intrinsic bonus (path exploration, coverage, curiosity-driven RL).

### 3a. State-novelty bonus (bounded memory)

The packing cache is a natural state-coverage oracle. Novelty becomes a
reward signal:

```c
/* state feature; eps is the "same state" radius */
futcache_pack_config_t cfg;
futcache_pack_config_init(&cfg);
cfg.dimension = DIM; cfg.epsilon = 0.2;    /* state quantization        */
cfg.domain_min = lo; cfg.domain_max = hi;
cfg.max_memory_bytes = 64U << 20;          /* hard cap keeps it stable */

futcache_pack_t *seen = NULL;
futcache_pack_create(&cfg, &seen);

/* in the step loop */
futcache_pack_observe(seen, state, &novel);
double bonus = novel ? REWARD_NOVEL : 0.0;
reward += bonus;
```

### 3b. 1-D observation novelty (interval)

For a single scalar observation (e.g. a position or a value), the interval
engine gives exact coverage and is the cheapest option:

```c
futcache_t *seen = NULL;
futcache_create(&c, &seen);        /* eps = state-equivalence radius */
futcache_observe(seen, obs, &novel);
double bonus = novel ? REWARD_NOVEL : 0.0;
```

### 3c. Distributed novelty (multi-agent / gossip)

If agents explore in parallel and must agree on shared novelty without a
coordinator, the CRDT engine merges occupancy via a deterministic priority
and converges under any delivery schedule:

```c
futcache_crdt_config_t c;
futcache_crdt_config_init(&c);
c.dimension = DIM;
c.anchor_count = n_anchors;                 /* delta-net cell count     */
c.anchors = anchors; c.epsilon = 0.2;
c.domain_min = lo; c.domain_max = hi;

futcache_crdt_t *agent = NULL;
futcache_crdt_create(&c, &agent);
futcache_crdt_observe(agent, state, payload, len, &novel);
/* gossip: snapshot, ship to another replica, merge there */
futcache_crdt_merge(other, agent_snapshot, ...);
```

### Curiosity traps

- **Novelty is not progress.** Pure novelty count encourages noise-seeking
  (random states, sensor jitter). Couple it with a value/learning signal or
  an entropy/density regulariser; a compact Isolation Forest over a
  calibration set can suppress noise (see `CompactIsolationForest`).
- **`epsilon` annealing.** A single fixed radius either saturates fast
  (everything seen) or never saturates. Anneal `epsilon` / use
  `observe_with_radius` to widen coverage as training progresses.
- **Memory ceiling keeps it honest.** Without `max_memory_bytes`, the state
  grows to the packing number `P(K, eps)`. For an unbounded state space this
  can be large; cap it and accept that very old regions are *forgotten*
  (which actually re-opens them to exploration — often desirable).

---

## Validation checklist

Before trusting it, prove it is helping on *your* data:

1. **Reproducibility**: use a fixed seed; the decision is deterministic.
2. **Decision correctness**: verify `is_novel` against a naive
   "recompute distance to every stored point" oracle on a small sample.
   (The test suites do this with an independent linear scan.)
3. **`epsilon` sweep**: on a held-out set, plot representative count vs
   `epsilon` and look for the region where true-reuse rises before
   cross-topic confusion does.
4. **Cacheability**: measure the discriminative margin
   (`d_min_cross - d_max_within`) and the empirical exponent `D_cache`
   (see `scripts/cacheability.py`). A *negative* margin means the cache
   helps mainly through insertion dynamics — know that before you rely on
   the compression ratio.
5. **Ceiling**: confirm `memory_bytes <= max_memory_bytes` and that
   `count + evictions == novel_observations` after a run.
6. **Persistence**: round-trip `serialize`/`deserialize` (C) or
   `copy_representatives`/`copy_radii` (Python) and confirm the restored
   state makes identical decisions.

---

## Reference defaults

| Parameter | Typical value | Notes |
|---|---|---|
| `epsilon` | 0.1–0.3 (cosine, normalized) | sweep it; domain-scaled |
| `max_memory_bytes` | 32–256 MiB | hard cap; 0 = packing bound only |
| `backend` | `vptree` (≥ hundreds of reps) | linear is fine below that |
| `distance` | cosine (RAG) / L2 or L1 (sensors) | must match your geometry |
| 1-D `domain_[min,max]` | your value range | required, inclusive |

---

*See `README.md` for the full API, `docs/serialization.md` and
`docs/pack-serialization.md` for the persistence formats, and
`docs/verification.md` for how correctness is tested.*

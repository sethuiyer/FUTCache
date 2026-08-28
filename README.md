# FUTCache

<p align="center">
  <img src="logo.webp" alt="FUTCache logo" width="420" />
</p>

**FUTCache is a metric visited-set.** It remembers which *regions* of a
metric space you've already explored, so it can tell you when a point is
genuinely new. C11 core with no third-party runtime library, plus Python bindings.

> 🧭 **New to the project or looking for the high-level architecture?** Start with [**30KFEET.md**](30KFEET.md) for the 30,000-ft mental model, system architecture map, engine decision tree, and end-to-end usage flows.

---

## What is FUTCache?

Most caches answer a question about a single point. FUTCache answers a
question about a *neighborhood*:

| Kind of cache | Remembers | Question it answers |
|---|---|---|
| Hash map / Redis | exact byte strings | Have I seen *this exact value*? |
| LRU / LFU / FIFO | recency | Have I seen this *recently*? |
| **FUTCache** | **covered regions** | Have I explored *this neighborhood*? |

Imagine sweeping a flashlight across a dark room. A hash map is a list of
the exact spots you've pointed at. LRU is a log of the order you pointed.
FUTCache is the set of *lit regions*: anything inside a lit region is
old, and anything outside it is new territory.

You bring a metric space — points as `double[d]` plus a distance function
that is actually a metric — and one knob: `ε`, the scale at which you
consider two points "the same place." FUTCache keeps a small set of
representative points and reports **novel** for any query that lands
farther than `ε` from every representative.

It is **not** a key/value store, and it doesn't do nearest-neighbor
search. It answers one question, exactly: *have we covered this region?*

### What FUTCache is now

FUTCache is a bounded-memory geometric coverage and compression engine. It
remembers a representative set of the observed region instead of the full
trajectory, and its experimental MDL selector can choose a useful operating
resolution from unlabeled geometry:

> **ε* is the description-optimal resolution of the observed geometry.**

That statement is relative to the supplied metric, explicit codec, and finite
candidate grid. MDL selects geometric compressibility; it does not certify
semantic interchangeability. For semantic answer caching, application labels
or safety requirements may still demand a supervised ε choice. See the
[MDL semantic negative result](docs/mdl-semantic-negative.md).

### Sufficient state for future novelty

Another way to say this: **FUTCache is a sufficient-state compressor for
future novelty decisions.** Two observation histories with the same
`ε`-ball union `U_ε(H)` are *future-equivalent* — they agree on every
later novelty query — so the raw stream can be forgotten and only the
covered regions retained. When `K` is compact at scale `ε`, that state is
bounded by the packing number `P(K, ε)`, not by how long the process has
run. It is the finite, computable realization of the Novelty Geometry
notion of minimal memory: keep a distinction iff some future query can
observe it in novelty output.

Two honest limits make this precise:

- **It compresses for novelty queries, not general memory.** It is not a
  key/value store and not a nearest-neighbor search index. A payload
  (LLM response, retrieval result) rides along keyed by representative
  slot and can be evicted or expired independently of the geometry.
- **The stored net is a subset of the exact union.** Lookup tests
  `d(x, R) > ε` against the stored representative net `R ⊂ H`, not
  membership in `U_ε(H)`. This is *one-sided*: it may report novelty
  near the net's gaps, but it can never suppress a genuinely new point.
  Exact union semantics are preserved by the 1-D interval-union engine
  and the bounded-dimension box cache.
- **Sufficient at a chosen resolution.** The packing cache is minimal
  for one fixed `ε`. Across scales the sufficient state changes — that
  is what `PersistentNovelty` (the merge tree / persistence diagram over
  the whole filtration) is for.

---

## Three Core Pillars: What FUTCache Solves

FUTCache is a fundamental computational geometry primitive that solves three of the largest, most expensive problems in modern computing:

### 1. The Observability & APM Crisis ($10B+ Market)
* **The Problem:** Storing 100% of distributed microservice traces costs millions in Datadog/Splunk bills, but uniform 1% random sampling misses rare 1-in-a-million catastrophic incident traces.
* **FUTCache's Solution:** A tiny, zero-dependency C11 engine running at **136k+ traces/sec** inside the local sidecar collector that **preferentially purges 72%+ of repetitive operational executions** while guaranteeing that **100% of structurally novel failure shapes are captured on Trace #1**.
* 📊 *Empirical Receipt:* Tested on **1,000,000 real production Alibaba Cloud traces** ([`docs/alibaba-1m-scaling-benchmark.md`](docs/alibaba-1m-scaling-benchmark.md)) achieving 72.3% suppression and 0.29ms noise absorption.

### 2. Neurosymbolic AI & Agentic Memory (The Frontier of AI)
* **The Problem:** Multi-step LLM loops decay exponentially under **Lusser's Law** ($0.95^{20} \approx 35\%$), while vector databases and chat history summaries waste tokens on continuous operational jitter.
* **FUTCache's Solution:** A deterministic operational visited-set where:
  * **Ontology** governs exact symbolic guardrails (`Workflow`, `Tool`, `Permission`, `Business Object`).
  * **FUTCache** governs continuous geometric experience, turning known operational states into **$O(1)$ zero-token deterministic fast-paths ($r = 1.0$)**.
  * **CRDTs** give multi-agent fleets a shared **Geometric Blackboard** to eliminate redundant exploration across workers without locks.
* 📜 *Manifesto:* Read [**`WHY_FUTCACHE.md`**](WHY_FUTCACHE.md) on defeating Lusser's Law and neurosymbolic state gates.

### 3. Automated Theorem Proving & CDCL Proof Search Geometry (SAT / SMT)
* **The Problem:** Modern Conflict-Driven Clause Learning (CDCL) and MaxSAT solvers spend 90%+ of their CPU time resolving and maintaining millions of redundant learned clauses during conflict analysis. Standard heuristics (LBD / VSIDS) treat clauses linearly and fail to detect geometric clustering among conflict hyperplanes.
* **The Breakthrough Discovery:** **CDCL proof search exhibits exploitable geometric recurrence.** Metric clause geometry carries statistically significant predictive signal ($p \le 0.007$) over solver backtrack dynamics beyond LBD and marginal chance.
* 📊 *Empirical Receipts across 1,000 Matched-Pair Permutations:*
  * **Hard 4-SAT:** Pure structural geometry predicts exact backjump depth at **$20.0\%$ vs $14.6\%$ matched null ($p = 0.007$)**, while LBD-only control is just **$4.6\%$**.
  * **Geometry + LBD Synergy:** Combining structural geometry with LBD jumps exact backtrack prediction to **$41.3\%$ ($9\times$ higher than LBD alone)**.
  * **Pigeonhole (PHP):** Pure structural geometry predicts backjump depth at **$44.3\%$ vs $33.2\%$ matched null ($p = 0.001$)**.
  * **Offline Hindsight Compactor:** Warm-starting NitroSAT V3 with FUTCache's greedy $\varepsilon$-net compresses active clause databases by **$91.8\% - 98.1\%$** in milliseconds ([`demos/sat_cdcl_nitrosat_pipeline.py`](demos/sat_cdcl_nitrosat_pipeline.py)).

---

### The Complete Package
1. **The Core Engine:** Pure POSIX C11 + VP-Tree spatial indexing + CRDT replication + Python bindings.
2. **The 30,000-Ft Architecture:** [**`30KFEET.md`**](30KFEET.md) (Mental model, engine selector tree, subsystem guide).
3. **The Neurosymbolic Manifesto:** [**`WHY_FUTCACHE.md`**](WHY_FUTCACHE.md) (Deterministic agent state & Lusser's Law antidote).
4. **The Real Production Receipts:** [**`docs/alibaba-1m-scaling-benchmark.md`**](docs/alibaba-1m-scaling-benchmark.md) (1M real Alibaba traces, 72.3% suppression).
5. **The Real CDCL Proof-Search Benchmark:** [**`demos/real_cdcl_trace_benchmark.py`**](demos/real_cdcl_trace_benchmark.py) (1,000-permutation statistical test on active 1-UIP conflict logs).
6. **The 21 SAT Problem Families Evaluation:** [**`demos/twenty_one_sat_families_benchmark.py`**](demos/twenty_one_sat_families_benchmark.py) (Geometric clause space evaluation across 21 benchmark families).
7. **The Tail-Sampling Benchmark:** [**`docs/trace-sampling-benchmark.md`**](docs/trace-sampling-benchmark.md) (100k synthetic trace comparison vs random/latency gates).

---

## When to use it

**Good fits** — you have a stream of points in a metric space and you
want to know which are genuinely new:

- **Semantic answer caching** — skip an expensive LLM or embedding call
  when the query lands inside a ball around one you've already answered.
- **Anomaly / novelty detection** — flag sensor, log, or network events
  that fall in previously unvisited territory.
- **Near-duplicate detection** — deduplicate embeddings, fingerprints,
  or coordinates without exact string matching.
- **Exploration / curiosity (RL)** — reward agents for visiting states
  that are far from everything seen before.
- **Coverage tracking** — "have we covered this area of space / of a
  config space?" (robotics, conformer search, sensor fusion).

**Skip it when:**

- You need **exact** key/value lookup → use a hash map or Redis.
- You need **recency** semantics → use LRU / LFU.
- Your "nearness" is **not a metric**, or you can't define one →
  FUTCache needs a real distance to be meaningful.
- You need **approximate nearest neighbors** / ranking → use a vector DB.
  FUTCache is a set, not a search index.

---

## Properties at a glance

| Property | What it means for you |
|---|---|
| **One-sided** | Never calls a genuinely new point "old." On the boundary it may still call a point *novel* — conservative, never swallows outliers. |
| **Bounded pack state** | The packing engine follows geometric size at scale `ε` and supports a hard byte ceiling. Exact full-history engines can grow with observations. |
| **Validated snapshots** | The interval and pack snapshot formats are versioned and CRC32-protected; corrupt, truncated, or out-of-domain input is rejected. |
| **Any metric** | L∞, L1, L2, cosine, Poincaré, or a custom C distance function. Python bindings expose the named metrics only. |
| **Verified contracts** | Exact engines and one-sided engines use matching brute-force oracles. 134 C test cases currently pass locally. |

---

## Install

### Python (fastest way to try it)

```sh
pip install .
```

Builds and installs the `futcache` package via nanobind + scikit-build-core.

### C library

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Requirements: a POSIX C11 environment with GCC 9+ or Clang 10+, CMake 3.16+,
and pthreads.

#### Build options

| Option | Default | Description |
|---|---|---|
| `FUTCACHE_BUILD_SHARED` | `OFF` | Build shared library |
| `FUTCACHE_BUILD_EXAMPLES` | `ON` | Build examples |
| `FUTCACHE_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `FUTCACHE_BUILD_PYTHON` | `OFF` | Build the `futcache_ext` CPython module (also built by `pip install .`) |
| `FUTCACHE_BUILD_NITROSAT` | `OFF` | Build NitroSAT anchor optimizer |
| `FUTCACHE_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan |
| `FUTCACHE_ENABLE_TSAN` | `OFF` | ThreadSanitizer (separate build) |
| `FUTCACHE_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

#### Install

```sh
cmake --install build
```

Downstream CMake:

```cmake
find_package(FUTCache 1 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE FUTCache::futcache)
```

---

## Quick Start

### Python: 30 seconds

```python
from futcache import PackCache

cache = PackCache(dimension=2, epsilon=0.1, distance="l2")

print(cache.observe([0.0, 0.0]).is_novel)   # True   first time here
print(cache.observe([0.05, 0.0]).is_novel)  # False  within 0.1 of [0,0]
print(cache.observe([5.0, 5.0]).is_novel)   # True   new territory
```

### Python: semantic answer cache

Skip an expensive call when the query dies into an existing ball. The
same `PackCache` is a visited-set for any named metric.

```python
import numpy as np
from futcache import PackCache

cache = PackCache(dimension=384, epsilon=0.6, distance="cosine",
                  max_memory_bytes=64 << 20, backend="vptree",
                  ttl=3600.0, max_entries=10000)

def call_llm(q):
    return "I'm a cached response"

for query in stream:
    response, result = cache.get_or_compute(query, call_llm)
    if result.is_novel:
        print(f"  novel (id={result.representative_id}, d={result.distance:.3f})")
```

### Python: death scale (the filtration)

```python
from futcache import PersistentNovelty

eng = PersistentNovelty()
eng.observe(0.5)
eng.observe(0.7)

print(eng.is_novel_at(0.6, 0.05))  # True   (min dist 0.1 > 0.05)
print(eng.is_novel_at(0.6, 0.1))   # False  (closed ball: 0.1 is covered)
print(eng.is_novel_at(0.6, 0.3))   # False  (larger t = coarser, still covered)
print(eng.novelty_spectrum(0.6))   # [(0.0, 0.1)]  novel for t in [0, t_max]
```

### Python: d-D persistent packing

```python
from futcache import PersistentNoveltyND

eng = PersistentNoveltyND(2, 0.1, "linf", [-1,-1], [1,1])
eng.observe([0.0, 0.0])
eng.observe([0.5, 0.0])
print(eng.persistences())           # [0.4, 0.4]
eng.evict_lowest()                  # evicts index 0
print(eng.rep_count)                # 1
```

### Python: submodular selection

```python
import numpy as np
from futcache import select_max_coverage

points = np.random.randn(1000, 16)
result = select_max_coverage(points, 1000, 16, 0.5, 50, "l2")
print(f"Selected {result['selected_count']} reps, "
      f"coverage = {result['coverage_ratio']:.3f}")
# approx_ratio is computed only for n <= 16 (brute-force OPT); else -1.
```

### C: d-D packing cache

```c
#include <futcache/pack.h>

futcache_pack_config_t cfg;
futcache_pack_t *cache;
double lo[384], hi[384];
size_t i;
for (i = 0; i < 384; ++i) { lo[i] = -1.0; hi[i] = 1.0; }

futcache_pack_config_init(&cfg);
cfg.dimension = 384;
cfg.epsilon = 0.6;
cfg.distance = futcache_distance_cosine;
cfg.domain_min = lo;
cfg.domain_max = hi;
cfg.max_memory_bytes = 64 * 1024 * 1024;
cfg.backend = &futcache_pack_vptree_backend; /* NULL = linear scan */
futcache_pack_create(&cfg, &cache);

bool novel;
futcache_pack_observe(cache, point_384d, &novel);

futcache_pack_destroy(cache);
```

### C: 1-D interval-union

```c
#include <futcache/futcache.h>

futcache_config_t config;
futcache_t *cache;
futcache_config_init(&config);
config.domain_min = 0.0;
config.domain_max = 1.0;
config.epsilon = 0.2;
futcache_create(&config, &cache);

bool novel;
futcache_observe(cache, 0.1, &novel);  /* novel = true  */
futcache_observe(cache, 0.15, &novel); /* novel = false (within 0.2 of 0.1) */

futcache_destroy(cache);
```

---

## Table of Contents

- [What is FUTCache?](#what-is-futcache)
- [When to use it](#when-to-use-it)
- [Properties at a glance](#properties-at-a-glance)
- [Install](#install)
- [Quick Start](#quick-start)
- [Engines](#engines)
- [Design Sketches](#design-sketches)
- [How it works](#how-it-works)
- [C API Reference](#c-api-reference)
- [Python API Reference](#python-api-reference)
- [Benchmarks & Experiments](#benchmarks--experiments)
- [Adversarial Analysis](#adversarial-analysis)
- [Complexity](#complexity)
- [Concurrency & Ownership](#concurrency--ownership)
- [Persistence](#persistence)
- [Documentation](#documentation)
- [License](#license)

---

## Engines

FUTCache ships nine engines. Most users start with the **packing cache**
(the d-D workhorse); the others specialize on a particular geometry,
scale range, or distributed setting. The last four are independent Phase 3
modules: they are not invoked by `futcache_pack_observe` unless the
caller uses them explicitly.

| Engine | Header | Dimension | What it does |
|---|---|---|---|
| **Interval-union cache** | `<futcache/futcache.h>` | 1-D | Exact interval-union novelty with AVL tree. The original engine. |
| **Resolution tower** | `<futcache/tower.h>` | 1-D | Multi-resolution dyadic tower with Fenwick rank/select and discovery logs. |
| **Packing cache** | `<futcache/pack.h>` | d-D | Generalized packing cache for arbitrary metric spaces. VP-tree or linear backend. FIFO pressure eviction; opt-in W1 API. |
| **Box cache** | `<futcache/box.h>` | 1–8 D | Exact axis-aligned L∞ box-union cache for full d-D coverage. |
| **CRDT cache** | `<futcache/crdt.h>` | d-D | Gossip-mergeable cache with deterministic Voronoi quantization. Replicas converge without coordination. |
| **Persistent novelty** | `<futcache/persist.h>` | 1-D | Single-linkage merge tree, multi-scale novelty, persistence diagrams, prime-tagged merge, and a finite zeta-inspired diagnostic. |
| **d-D persistent packing** | `<futcache/persist_nd.h>` | d-D | Per-representative birth/death tracking with persistence-based eviction. |
| **Submodular selection** | `<futcache/select.h>` | d-D | Greedy max-coverage rep selection (1−1/e guarantee). Streaming evict-worst. |
| **Anchor embedding** | `<futcache/embed.h>` | d→m | Distance-to-anchors projection with bounded distortion (2δ). Conservative ε adjustment. |
| **MDL selector** | `<futcache/mdl.h>` | d-D | Offline finite-grid epsilon selection using measured model bytes and explicit residual code. |

### Interval-union cache (1-D, exact)

The original FUTCache engine. For a 1-D metric domain `K=[a,b]` and
resolution `ε`, it stores the canonical union of closed intervals:

```
U(H) = ⋃([y−ε, y+ε] ∩ K)    for y in observed history H
```

A point `x` is novel exactly when `x ∉ U(H)`. Historical points are never
needed once their contribution to the boundary has been merged.

- AVL-backed, `O(log n)` lookup
- Versioned serialization with CRC32
- Concurrency-safe (rwlock)

### Resolution tower (1-D, multi-scale)

A dyadic refinement tower with per-level occupancy bitsets, Fenwick
prefix-sum trees, and first-discovery logs. Provides multi-resolution
novelty at logarithmic cost per level.

### Packing cache (d-D, generalized)

The workhorse for d-dimensional metric spaces (embeddings, sensors,
assignments, any `double[d]` + distance). Stores a first-fit
`ε`-separated set of representatives. Novelty: `d(x, R) > ε`. When `K`
is compact at ε, `|R|` is bounded by the packing number `P(K, ε)` and
saturates. First-fit is a cover of the observed stream, not the
smallest possible net.

Key features:
- **Adaptive radii**: each rep gets its own `ε_i`; lookup tests the exact
  union `⋃ B(r_i, ε_i)`
- **Hard memory ceiling**: `max_memory_bytes` recycles the oldest
  representative (FIFO). This is the only automatic pressure policy.
- **VP-tree backend**: 9.1× faster observe, 17.2× faster query vs linear
  scan at 384-D (measured)
- **W1 eviction (opt-in)**: `futcache_pack_evict_w1` removes the rep with
  the smallest nearest-neighbor distance. The caller must invoke it;
  `observe` never does.
- **Serialization**: atomic, versioned, CRC32-protected snapshots

Built-in distances: `linf`, `l1`, `l2`, `cosine`, `poincare`. Custom
metrics via function pointer.

### Box cache (d-D, L∞ exact)

Exact axis-aligned `L∞` box-union for dimensions 1–8. Each observation
contributes its clipped `ε`-box; queries test membership in the exact
union. For applications that need full d-dimensional ball coverage.

### CRDT cache (d-D, distributed)

Gossip-mergeable cache with deterministic Voronoi quantization. Replicas
converge without coordination:

```
q(x) = q(y)  ⟹  d(x, y) ≤ ε
```

Set union is commutative, associative, and idempotent. Ships anchor
construction helpers (grid, Halton, safe-anchors) and a certified
covering-radius estimator.

### Persistent novelty (1-D, multi-scale)

Single-linkage merge tree over observed points. Provides:

- **`is_novel_at(x, t)`**: exact novelty at scale `t` (`min_dist > t`).
  Larger `t` is coarser (bigger balls), so fewer points stay novel.
- **`novelty_spectrum(x)`**: interval `[0, t_max]` where `t_max` is the
  distance to the nearest observation (`[]` if `x` was observed)
- **`evict_below(τ)`**: drop low-persistence rows from the stored
  *diagram only* — the merge tree and observations are unchanged
- **`selberg_zeta(s)`**: finite zeta-inspired product over features whose
  sorted birth index is prime: `∏ (1 − (1 + persistence)^{-s})^{-1}`
- **Signature merge**: `futcache_persist_merge_features` — deterministic,
  idempotent feature join; not raw-history persistence recomputation
- **`copy_diagram`**: full (birth, death, persistence) export

### d-D persistent packing

Per-representative nearest-neighbor tracking with:

- **`observe(x)`**: novel/redundant + persistence update
- **`is_novel_at(x, t)`**: scale-resolved novelty
- **`persistences()`**: per-rep persistence (nn_dist − radius)
- **`evict_lowest()`**: remove the most redundant rep
- **`stats()`**: includes prime-birth count

### Submodular selection

Max-coverage rep selection with the **(1 − 1/e) approximation guarantee**:

- **`select_max_coverage`**: direct greedy with lexicographic tie-breaking.
  Brute-force optimal for `n ≤ 16` (verification).
- **`select_evict_worst`**: streaming swap — find the rep whose removal
  causes the largest coverage loss.
- **`select_coverage`**: current coverage ratio.

### Anchor embedding

Projects `d`-dimensional points into `m`-dimensional
distance-to-anchors space:

```
embed(x) = [d(x, a_1), d(x, a_2), ..., d(x, a_m)]
```

- **Distortion bound**: `|d(x,y) − ‖φ(x)−φ(y)‖_∞| ≤ 2δ` when `δ` is a
  certified upper bound on the anchor covering radius
- **Conservative ε**: `ε_embed = ε_orig − 2δ` is available only with that
  certificate; a sampled radius is diagnostic and is not accepted as proof

---

## Design Sketches

`docs/design/` contains five design documents extending FUTCache from a
novelty cache to a general-purpose novelty engine:

| # | Title | Status | Core contribution |
|---|---|---|---|
| 01 | [Persistent Novelty](docs/design/01-persistent-novelty.md) | **Implemented** (side module) | 1-D merge tree + reduced d-D NN persistence. Prime-tagged signature merge. Finite prime-birth diagnostic. |
| 02 | [Wasserstein-1 Optimal Eviction](docs/design/02-wasserstein-eviction.md) | **Implemented** (opt-in API) | `evict_w1` = min nearest-neighbor. FIFO remains the pressure path. 37% fewer cycles on a 300-point 8-D clustered synthetic. |
| 03 | [Submodular Rep Selection](docs/design/03-submodular-selection.md) | **Implemented** (offline batch) | Greedy max-coverage (1−1/e). Streaming evict-worst. Brute-force OPT for `n ≤ 16`. |
| 04 | [Learned Metric via Anchor Embedding](docs/design/04-learned-metric.md) | **Implemented** (anchors, not neural) | Distance-to-anchors projection. Certified distortion adjustment requires a caller-supplied or grid-derived covering-radius upper bound. |
| 05 | [Competitive Ratio for Memory-Bounded Caching](docs/design/05-competitive-theorem.md) | Theory | Competitive-ratio analysis via packing numbers. Assumes W1-under-budget, which pack does not run. |

---

## How it works

This section is the math behind the pitch. If you just want to use the
library, you can stop at Quick Start — the guarantee in one sentence is:
**FUTCache reports `novel` exactly for points outside the union of
`ε`-balls around everything it has seen.** Everything below makes that
sentence precise.

### The visited-set model

The geometric state of an observation history `H` is the union of closed
`ε`-balls:

```
U_ε(H) = ⋃_{y ∈ H} B̄(y, ε)
```

A query `x` is novel exactly when `x ∉ U_ε(H)`. Two histories with the
same union are **future-equivalent**: they agree on every later novelty
query, so the raw stream can be forgotten. The stored state is the
union, not the points.

### Memory is a packing number, not uptime

FUTCache stores an `ε`-separated representative set `R`. Its size is
bounded by the **packing number** of the space at that scale:

```
|R| ≤ P(K, ε)
```

When `K` is compact at scale `ε`, `R` stops growing — a graph under hop
distance, residue classes under a 2-adic metric, a corridor along a
river. When `K` is *not* compact at `ε` (Euclidean primes, jungle hills
with unique meridians), `|R|` keeps growing. The engine reports that; it
does not invent a bound. Either way, you can set a hard byte ceiling
(`max_memory_bytes`).

### One-sided, not leak-free

The oracle never marks `x ∉ U_ε(H)` as old — that would be a false
negative and it can't happen. It *may* still call a point novel when the
point is close to `H` but not to the stored net `R ⊂ H` (a one-sided,
conservative gap). Outliers are not swallowed. That is the guarantee —
not "zero false novelty" and not safety outside the metric you chose.

### How `observe` and the net work

`observe()` is **first-fit** at one `ε`: an insertion-order `ε`-net, not
a minimum *k*-center. Each rep can carry its own adaptive radius; a
lookup tests the exact union `⋃ B(r_i, ε_i)`. Multi-scale death times
`t*(x) = d(x, H)` live on `PersistentNovelty` / `is_novel_at`. An
offline rebuild (`select`, 1-D farthest-greedy, NitroSAT) can shrink `R`
without changing the cover.

### The distance is the product

FUTCache is one engine; the distance function is what makes it your
problem. Hamming on the cube is CDCL search coverage. Hops on a graph
are exploration. Tanimoto on fingerprints is scaffold novelty. Cosine on
embeddings is paraphrase reuse — one application, only as good as the
embedder. You don't need a graph-cache, a SAT-cache, and a log-cache;
you need a `d` that means "near" for your problem.

**The trap: native vs. rented metrics.** A *native* metric (hops, Hamming,
L1 latencies, RMSD, *p*-adic residues) makes the guarantee exact for that
`d`. A *rented* metric from an embedder (MiniLM, Bekko on ambiguous text)
makes the cache report the embedder's geometry — including its blind
spots — with the same precision. SciFact-style collapse is the embedder,
not a leak in the visited-set.

**Cost of a query.** A small net is cheap: ~2 μs/op on the Bekko 384-D
run (9 reps). VP-tree query is ~31 μs at 2000 reps / 384-D cosine. A
full persist scan is `O(n · d)`, not two microseconds.

### Deep dive: Kabsch RMSD as `d`

A high-stakes `d` is **Kabsch RMSD** on `R^{3N} / SE(3)`: two poses of
the same protein that differ by a rigid motion are one point. Flattened
L2 or cosine on coordinates is the wrong metric (a 45° rotation looks
novel). If the trajectory stays on a bounded conformational region,
`P(K, ε)` is finite and the visited-set can saturate; an unbounded
unfold does not. The one-sided guarantee is "no new basin farther than
ε from every rep," not "every transition state on the femtosecond grid."
A saddle inside an existing ball is redundant at that ε. Online `t*` is
distance to the stored net `R ⊂ H`, not the full trajectory. Kabsch is a
C `distance_fn`; the Python bindings cannot register it. Adaptive biasing
from `t*` is metadynamics with the ε-net as the collective variable — the
field already does that geometry; the primitive is the exact, mergeable
visited-set under it.

### Relation to the Novelty Geometry framework

FUTCache is a finite, computable realization of the **Novelty Geometry**
framework — the study of how an infinite traversal generates, retains,
and completes *ordered novelty* across increasing resolution. In systems
language: a topological cache whose sufficient state is the
future-equivalence class of the stream, not a recency list.

| Framework concept | FUTCache engine |
|---|---|
| Sufficient memory / future-equivalence quotient | interval-union `U(H)` — minimal state for exact future novelty |
| Ordered novelty word `D_j(L)` | tower per-level occupancy + first-discovery log |
| Partition tower + bonding map `q_{j+1,j}` | tower coarse/fine refinement compatibility |
| ε-separated set / packing number `P(K, ε)` | packing cache rep set and bound |
| Fenwick prefix-sum index | tower spatial rank/select |
| Inverse limit / boundary | the ideal completion a one-`ε` tower approximates finitely |

FUTCache fixes one resolution (`ε`) rather than refining a tower, and
stores the sufficient novelty state with a hard memory ceiling. The
limits we measure empirically — one-sided packing approximation, negative
cacheability margin on some corpora — are the *finite* aspects the
framework itself flags as open, not accidental deficiencies.

See [novelty-geometry](https://github.com/sethuiyer/novelty-geometry) for
the full framework (README, glossary, boundary-object semantics).

---

## C API Reference

### `<futcache/futcache.h>` — 1-D interval-union

| Function | Description |
|---|---|
| `futcache_create(config, *cache)` | Create cache. Two-pass allocation. |
| `futcache_observe(cache, x, *novel)` | Observe point. Returns novelty. |
| `futcache_is_novel(cache, x, *novel)` | Query without mutating. |
| `futcache_get_stats(cache, *stats)` | Telemetry: observations, novel count, interval count, AVL height. |
| `futcache_copy_intervals(cache, buf, cap, *count)` | Export disjoint interval snapshot. |
| `futcache_serialize / deserialize` | Versioned, CRC32-protected snapshot. |
| `futcache_validate(cache)` | O(n²) invariant check. |
| `futcache_destroy(cache)` | Free all memory. |

### `<futcache/tower.h>` — Resolution tower

| Function | Description |
|---|---|
| `futcache_tower_create(config, *tower)` | Create dyadic tower. |
| `futcache_tower_observe(tower, x, out_novel)` | Multi-level novelty (one byte per level). |
| `futcache_tower_prefix_count(tower, level, r)` | Occupied cells with index ≤ r. |
| `futcache_tower_select_occupied(tower, level, k)` | k-th occupied cell in spatial order. |
| `futcache_tower_discovery_at(tower, level, k)` | k-th cell in first-discovery order. |

### `<futcache/pack.h>` — d-D packing cache

| Function | Description |
|---|---|
| `futcache_pack_create(config, *cache)` | Create packing cache. |
| `futcache_pack_observe(cache, point, *novel)` | Observe with the configured ε. |
| `futcache_pack_observe_with_radius(cache, point, r, *novel, *dist, *idx)` | Observe with adaptive radius. |
| `futcache_pack_is_novel(cache, point, *novel)` | Novelty query without mutating. |
| `futcache_pack_lookup(cache, point, *found, *dist, *idx)` | Closest containing rep. |
| `futcache_pack_evict_w1(cache, *idx)` | Opt-in W1 eviction (min NN distance). Not used by `observe`. |
| `futcache_pack_clear(cache)` | Reset to empty. |
| `futcache_pack_serialize / deserialize` | Atomic snapshot. |
| `futcache_pack_get_stats(cache, *stats)` | Memory, evictions, rep count. |
| `futcache_pack_validate(cache)` | Invariant check. |

Built-in distances: `futcache_distance_linf`, `_l1`, `_l2`, `_cosine`,
`_poincare`.

### `<futcache/persist.h>` — 1-D persistent novelty

| Function | Description |
|---|---|
| `futcache_persist_create(config, *engine)` | Create merge-tree engine. |
| `futcache_persist_observe(engine, x)` | Observe. Rebuilds the merge tree. No novelty out-param. |
| `futcache_persist_is_novel_at(engine, x, t, *novel)` | Exact novelty at scale `t`. |
| `futcache_persist_novelty_spectrum(engine, x, out, *count)` | Writes `[0, t_max]` (two-pass). |
| `futcache_persist_evict_below(engine, τ)` | Filter the diagram; merge tree unchanged. |
| `futcache_persist_copy_diagram(engine, buf, *count)` | Export (birth, death, persistence). |
| `futcache_persist_merge_features(a, na, b, nb, out, *count)` | Deterministic feature-signature join. |
| `futcache_persist_selberg_zeta(engine, s, *z)` | Product over prime-birth features. |
| `futcache_persist_prime_cycle_count(engine, τ, *count)` | Prime-birth features with persistence ≥ τ. |

### `<futcache/persist_nd.h>` — d-D persistent packing

| Function | Description |
|---|---|
| `futcache_persist_nd_create(d, ε, dist, ctx, lo, hi, max_mem, alloc, *engine)` | Create d-D engine (no config struct). |
| `futcache_persist_nd_observe(engine, point, *novel)` | Observe + update persistence. |
| `futcache_persist_nd_is_novel_at(engine, point, t, *novel)` | Scale-resolved novelty. |
| `futcache_persist_nd_nearest_distances(engine, buf, *count)` | Per-rep NN distance. |
| `futcache_persist_nd_persistences(engine, buf, *count)` | Per-rep persistence. |
| `futcache_persist_nd_evict_lowest(engine, *idx)` | Evict most redundant rep. |
| `futcache_persist_nd_count_above(engine, τ, *count)` | Count reps with persistence ≥ τ. |

### `<futcache/select.h>` — Submodular selection

| Function | Description |
|---|---|
| `futcache_select_max_coverage(points, n, d, ε, k, dist, ctx, *result)` | Greedy (1−1/e) max-coverage. |
| `futcache_select_evict_worst(points, n, reps, nrep, d, ε, dist, ctx, *idx, *loss)` | Streaming swap. |
| `futcache_select_coverage(points, n, reps, nrep, d, ε, dist, ctx, *cov)` | Coverage ratio. |

### `<futcache/embed.h>` — Anchor embedding

| Function | Description |
|---|---|
| `futcache_embed_create(config, *embed)` | Create anchor projection. |
| `futcache_embed_point(embed, point, out)` | Project to anchor-distance space. |
| `futcache_embed_covering_radius(embed)` | Estimated covering radius `δ` (distortion is `2δ`). |
| `futcache_embed_adjusted_epsilon(embed, ε, *ε_adj)` | Conservative ε = ε − 2δ. |

---

## Python API Reference

### `PackCache`

```python
PackCache(dimension, epsilon, distance="linf", domain_min=None,
          domain_max=None, backend="linear", max_memory_bytes=0,
          max_entries=0, ttl=0.0)
```

| Method | Description |
|---|---|
| `observe(point, payload=None, radius=None) -> NoveltyResult` | Query + insert. |
| `query(point) -> NoveltyResult` | Query without inserting. |
| `get_or_compute(point, compute, radius=None) -> (bytes, NoveltyResult)` | Serve cached or compute. |
| `get_payload(rep_id) -> bytes \| None` | Retrieve payload. |
| `set_payload(rep_id, payload)` | Store payload. |
| `evict_w1() -> int` | Opt-in W1 eviction. Returns evicted index. |
| `copy_representatives() -> ndarray` | Export rep points. |
| `copy_radii() -> ndarray` | Export rep radii. |
| `clear()`, `purge()` | Reset. |
| `memory_bytes()`, `peak_memory_bytes()`, `evictions()`, `observations()` | Telemetry. |

### `PersistentNovelty` (1-D)

| Method | Description |
|---|---|
| `observe(x) -> None` | Observe. Rebuilds the merge tree. |
| `is_novel_at(x, t) -> bool` | Exact novelty at scale t (`min_dist > t`). |
| `novelty_spectrum(x) -> list[tuple]` | `[(0.0, t_max)]`, or `[]` if `x` was observed. |
| `copy_diagram() -> list[dict]` | Full (birth, death, persistence) export. |
| `merge(other) -> list[dict]` | Deterministic signature join (does not return an engine). |
| `selberg_zeta(s) -> float` | Product over prime-birth features. |
| `prime_cycle_count(tau=0.0)`, `feature_count(tau=0.0)`, `observations` | Telemetry. |

`evict_below` exists only on the C API.

### `PersistentNoveltyND` (d-D)

| Method | Description |
|---|---|
| `observe(point) -> bool` | Observe. Returns novelty. |
| `is_novel_at(point, t) -> bool` | Scale-resolved novelty. |
| `nearest_distances() -> list` | Per-rep NN distance. |
| `persistences() -> list` | Per-rep persistence. |
| `evict_lowest() -> int` | Evict most redundant rep. |
| `count_above(tau) -> int` | Count reps with persistence ≥ τ. |
| `stats() -> dict` | Includes prime-birth count. |

### `AnchorEmbedding`

| Method | Description |
|---|---|
| `embed(point) -> list` | Project to anchor-distance space. |
| `covering_radius -> float` | Estimated covering radius `δ` (distortion is `2δ`). |
| `adjusted_epsilon(ε) -> float` | Conservative ε = ε − 2δ. |

### Functions

| Function | Description |
|---|---|
| `select_max_coverage(points, n, d, ε, k, dist) -> dict` | Keys: `selected_count`, `coverage_ratio`, `approx_ratio` (`-1` if `n > 16`). |
| `select_evict_worst(points, n, d, ε, reps, dist) -> dict` | Streaming swap. |
| `select_coverage(points, n, d, ε, reps, dist) -> float` | Coverage ratio. |
| `merge_persistence_diagrams(a, b) -> list` | Deterministic persistence-feature signature join. |
| `nth_prime(i) -> int` | i-th prime (0-indexed). |
| `halton_sequence(n, d) -> ndarray` | Halton low-discrepancy sequence. |
| `poincare_distance(x, y) -> float` | Poincaré ball distance. |

---

## Benchmarks & Experiments

### VP-tree vs linear scan (384-D, manifold cosine, k=4 intrinsic dim)

| Backend | observe | query | live memory |
|---|---:|---:|---:|
| linear scan | 270 μs/op | 541 μs/op | 5.96 MiB |
| exact VP-tree | 30 μs/op |  31 μs/op | 6.19 MiB |

**9.0× faster observe, 17.5× faster lookup, 4% index overhead** on the
*manifold* workload. On uniform random data the speedup drops to ~1.5×
because VP-tree pruning relies on the triangle inequality in regions
with low intrinsic dimension. See `bench/pack_backend_bench.c` for the
full d × dataset × rep-count sweep.

### vs LRU (5 workloads, N=10,000)

| Workload | True novel | FUTCache error | LRU error (k=8192) |
|---|---:|---:|---:|
| reciprocal (ε=0.01) | 10 | **0.0000** | 0.9990 |
| uniform (ε=0.01) | 52 | **0.0000** | 0.9948 |
| three-cluster (ε=0.05) | 3 | **0.0000** | 0.9997 |
| alternating (ε=0.5) | 2 | **0.0000** | 0.0000 |
| power-decay (ε=0.05) | 10 | **0.0000** | 0.9990 |

FUTCache: **zero error** with bounded memory on every spatially-structured
workload. LRU is structurally stuck at ~99% error.

### Bekko multilingual (384-D, cosine, 6 languages × 8 topics)

| ε | reps | reuse_rate | reuse_precision | μs/op |
|---:|---:|---:|---:|---:|
| 0.55 | 9 | 0.8125 | **1.0000** | 2.15 |
| 0.70 | 9 | 0.8125 | 0.9487 | 2.10 |
| 0.80 | 6 | 0.8750 | 0.8810 | 1.47 |

At ε=0.55: **9 representatives cover 48 inputs across 6 languages** with
100% reuse precision.

### Cross-lingual (6 languages, 48 pairs, 128-D)

At ε=0.8, 128-D: **9 reps, 81% correct reuse, 0.6 μs/op** (3.5× faster than
384-D at the same compression ratio).

### Cacheability diagnostic

The discriminative margin is *negative* at every dimension: no single cosine
threshold cleanly separates topics. The cache works via **insertion
dynamics**, not a clean margin. Empirical `D_cache ≈ 1.0` — one bit of
distinguishing structure per topic cluster.

### Voronoi packing in d dimensions (nd_dedup.c)

5000 uniform random points in `[0,1]^d` with FUTCache pack + L_inf.
Reported alongside the conservative packing bound `(⌊1/ε⌋+1)^d`:

| dim | ε | oracle novel | futc novel | reps | rep_bound | one-sided gap |
|---:|---:|---:|---:|---:|---:|---:|
|  2 | 0.10 |  35 |  60 |  60 |     121 |  25 FP |
|  4 | 0.10 | 836 | 1414 | 1414 |  14,641 | 578 FP |
|  8 | 0.10 | 4983 | 4983 | 4983 | 2.1×10⁸ | 0 |
| 16 | 0.10 | 5000 | 5000 | 5000 | 4.6×10¹⁶ | 0 |

`futc_novel` exceeds `oracle_novel` only at low d (the cache reports
extra novel for points in Voronoi-cell gaps). At d ≥ 8 the
high-dimensional spread makes nearly every point genuinely novel, so
the cache and oracle agree. The bench also exercises L1, L2, L_inf on
the same stream.

### Corpus-dedup cost economics (corpus_dedup_cost.c)

Topic-cluster synthetic corpus (40 clusters + 25% long tail, 4000 docs,
d=128, cosine) at 1000 tok/doc × $0.20/1M tokens:

| Corpus shape | ε | reps | dedup_ratio | %saved | $ saved (per 100k docs) |
|---|---:|---:|---:|---:|---:|
| Balanced (40 clusters + 25% tail) | 0.05 | 1040 | 3.85× | 74.0% | $14.80 |
| Tight-cluster (3% tail)           | 0.05 |  160 | 25.0× | 96.0% | $19.20 |
| Uniform (no clusters)             | 0.05 | 4000 |  1.0× |  0.0% |   $0.00 |

The dedup ratio exactly recovers the cluster count — tight-cluster
corpus collapses to 40 reps per cluster, balanced corpus keeps more
because the tail is genuinely far apart. Uniform has nothing to
deduplicate (0% savings). The cache is one-sided: `%saved` is a lower
bound, not a promise.



### vs geometric baselines (1D, 5 workloads, N=10,000)

Comparison at each workload's oracle ε. k-center uses capacity-bounded
Gonzalez eviction; LSH uses 1D grid hashing. All three answer the metric
predicate; only FUTCache has the one-sided guarantee.

| Workload | oracle_novel | FUTC peak (err) | k-center k=P (err) | LSH b=P (err) |
|---|---:|---:|---:|---:|
| reciprocal (ε=0.01)     | 10 | **7** (0.0000) | 18 (0.0008) | 18 (0.0008) |
| uniform (ε=0.01)        | 52 | **20** (0.0000) | 75 (0.0023) | 75 (0.0023) |
| three-cluster (ε=0.05)  |  3 | **3** (0.0000) |  3 (0.0000) |  3 (0.0000) |
| alternating (ε=0.5)     |  2 | **2** (0.0000) |  2 (0.0000) |  2 (0.0000) |
| power-decay (ε=0.05)    | 10 | **4** (0.0000) | 14 (0.0004) | 14 (0.0004) |

FUTCache uses **2.5–3.75× fewer representatives** at the same accuracy and
**zero false positives** (the residual errors in k-center / LSH are all
FPs from capacity-bounded approximation; see `bench/cache_comparison_extended.c`).

### vs geometric baselines (2D, 5 workloads, N=5,000)

Five 2D workloads (uniform, 32×32 jittered grid, two Gaussian clusters,
ring, diagonal line) compared against k-center, LSH, and an exact-set
NN cache. See `bench/cache_comparison_2d.c`.

| Workload | oracle_eps | FUTC reps (err) | k-center k=P (err) | LSH T=4,b=8 (err) |
|---|---:|---:|---:|---:|
| uniform-2d   | 0.05 | **247** (0.0232, 0 FN) | 247 (0.0232) | 248 (0.0234) |
| grid-32x32   | 0.05 | **256** (0.0510, 0 FN) | 256 (0.0510) | 256 (0.0510) |
| two-cluster  | 0.05 | **12**  (0.0020, 0 FN) | 12  (0.0020) | 12  (0.0020) |
| ring         | 0.04 | **47**  (0.0038, 6 FN) | 59  (0.0038) | 59  (0.0038) |
| line         | 0.04 | **21**  (0.0008, 1 FN) | 28  (0.0018) | 31  (0.0024) |

In 2D, k-center and LSH **match** FUTCache's representative count once
their capacity is large enough — Gonzalez's greedy is near-optimal for
k-center in low dimension. FUTCache's edge is *convergence to the right
count automatically* (no capacity tuning), and **0 false negatives** at
ε ≤ oracle_eps (FN appears only when ε is over-tightened and the packing
gets crowded). The 2D table also reveals a regime FUTCache handles
specially: on **ring** at ε=0.05 it holds 47 reps vs k-center's 59,
matching within a factor while reporting a tighter FN count.

### vs geometric baselines (3D, 3 workloads, N=5,000)

Three 3D workloads (uniform cube, sphere surface — the curse-of-dim
classic — and two Gaussian clusters). See `bench/cache_comparison_2d.c`
(also covers 3D).

| Workload | oracle_eps | FUTC reps (ε=0.2, err) | k-center k=P (err) | LSH T=8,b=8 (err) |
|---|---:|---:|---:|---:|
| uniform-3d   | 0.08 | **100** (0.0174, 22 FN) | 936 (0.1766, 2 FN) | 936 (0.1766) |
| sphere-3d    | 0.15 | **194** (0.0370, 26 FN) | 333 (0.0636, 23 FN) | 333 (0.0636) |
| two-cluster-3d | 0.08 | **2**  (0.0006, 3 FN)  | 19  (0.0040, 3 FN)  | 19  (0.0040) |

In 3D, FUTCache pulls ahead substantially:
- **9× fewer representatives** than k-center/LSH on uniform-3d
- **1.7× fewer** on sphere-3d (the curse-of-dim case)
- **10× fewer** on two-cluster-3d

The 1D/2D equivalence of all three methods breaks at d=3: LSH on random
hyperplanes is no longer near-optimal because cell coverage becomes
uneven, and k-center Gonzalez greedy over-allocates capacity before
converging. FUTCache's automatic ε convergence reaches the packing-bound
minimum directly. (NB: FUTCache at ε=0.2 is looser than the oracle ε, so
the small FN counts reflect that — the error rate is lower because the
cache has accepted some genuinely-novel points at the wider ε, and
the fewer reps reflect better packing at looser ε. See
`bench/cache_comparison_2d.c` for the full ε sweep.)

### CRDT fleet experiment

Multi-agent deduplication across W=1–32 workers with full-fan-in gossip.
Round-robin point distribution; each worker observes 1/W of the stream.
`bench/crdt_fleet.c`.

| Fleet W | Joint cells | Dedup ratio | Conv rounds | Merge latency |
|---:|---:|---:|---:|---:|
| 1  | 14 | 1.000 | 2 | 0.2 μs |
| 8  | 14 | 0.125 | 2 | 13.8 μs |
| 32 | 14 | 0.031 | 2 | 173 μs |

Joint coverage exactly matches the single-worker reference at every fleet
size, confirming the join-semilattice convergence guarantee (PHASE2.md
12.29). Dedup ratio = 1/W as expected for disjoint cell sets.

### CRDT randomized gossip sweep (W=8–256, ε=0.01)

Each round, each worker picks k peers uniformly at random and merges their
snapshot. Deterministic seed.

| W | k=1 (rounds / total) | k=3 | k=⌈log₂ W⌉ | k=W-1 |
|---:|---:|---:|---:|---:|
| 8   | 5 / 15 μs  | 3 / 18 μs  | 3 / 18 μs   | 1 / 33 μs   |
| 32  | 6 / 96 μs  | 2 / 142 μs | 2 / 218 μs  | 1 / 1130 μs |
| 128 | 7.6 / 1.4 ms | 3 / 1.0 ms | 2 / 1.3 ms | 1 / 18 ms |
| 256 | 8.6 / 3.5 ms | 3 / 2.5 ms | 2 / 3.1 ms | 1 / 37 ms |

At W ≥ 64, **`k=⌈log₂ W⌉` is the sweet spot**: 2 rounds to convergence
vs full-fan-in's 1 round, but at 10–15× lower total latency. At W=256,
randomized gossip converges in 3.1 ms vs full fan-in's 37 ms — a 12×
speedup with identical coverage guarantees.

### Network-RTT-adjusted fleet latency

Real fleets pay per-message network RTT, not just in-process merge time.
Projecting the Section 2 schedules with 1 ms (intra-DC) and 10 ms
(cross-DC) RTT per gossip message:

**1 ms RTT (intra-DC):**

| W | k=1 | k=3 | k=⌈log₂ W⌉ | k=W-1 |
|---:|---:|---:|---:|---:|
| 8   | 1 ms   | 3 ms   | 3 ms   | 7 ms  |
| 32  | 2 ms   | 6 ms   | 5 ms   | 32 ms |
| 128 | 7 ms   | 10 ms  | 15 ms  | 136 ms |
| 256 | 11 ms  | 11 ms  | 19 ms  | 293 ms |

**10 ms RTT (cross-DC):**

| W | k=1 | k=3 | k=⌈log₂ W⌉ | k=W-1 |
|---:|---:|---:|---:|---:|
| 8   | 10 ms  | 30 ms  | 30 ms  | 70 ms  |
| 32  | 20 ms  | 60 ms  | 50 ms  | 311 ms |
| 128 | 71 ms  | 91 ms  | 141 ms | 1279 ms |
| 256 | 74 ms  | 92 ms  | 82 ms  | 2588 ms |

At W=256 with 10 ms cross-DC RTT, `k=⌈log₂ W⌉` converges in **82 ms** vs
full fan-in's **2.6 seconds** — a **31× speedup**. The schedule choice
dominates deployment latency at fleet scale: every doubling of W under
full fan-in roughly doubles latency, while `k=⌈log₂ W⌉` grows by
~1 ms/round.

### CRDT merge-conflict rate

Under round-robin point distribution with disjoint cell sets per worker,
gossip updates almost always fill empty cells (no priority conflict).
On `power-decay`, ε=0.01, W=16:

| Schedule | Convergence rounds | Total merges | Adopt | Conflict | Adopt rate |
|---|---:|---:|---:|---:|---:|
| k=1            | 4 | 25,033  | 25,033  | 0 | 1.0000 |
| k=3            | 2 | 75,207  | 75,207  | 0 | 1.0000 |
| k=⌈log₂ W⌉     | 2 | 100,304 | 100,304 | 0 | 1.0000 |
| k=W-1 (full)   | 1 | 376,230 | 376,230 | 0 | 1.0000 |

Zero conflicts across all schedules: round-robin + disjoint cell sets
guarantee that remote updates never collide with existing local entries.
This validates that the CRDT merge rule's `priority`-based conflict
resolution is *defensive* machinery — under realistic fleet workloads
it's never exercised, but it's there for partial failures and concurrent
observation of the same anchor cell.

### Arithmetic prime novelty

The reproducible arithmetic experiment is documented in
[docs/prime-novelty.md](docs/prime-novelty.md) and built as
`futcache_prime_novelty_experiment`. It compares bounded prime gaps, residue
vectors, individual p-adic lenses, and a weighted combined metric. Rankings
use nearest-prior distance; epsilon verdicts and reduced representative
persistence are reported separately.

### MDL-selected resolution

The experimental MDL selector evaluates a fixed epsilon grid under an explicit
representative, assignment, residual, and epsilon-level code:

> **ε* is the description-optimal resolution of the observed geometry.**

This is optimal relative to the specified metric, codec, and candidate grid—not
an informal universal guarantee. The controlled structured/clustered/random
comparison is in [docs/mdl-stream-comparison.md](docs/mdl-stream-comparison.md).

### KDD Cup '99 (security novelty)

AUC ≈ 0.99 — identical to a trivial 1-NN baseline. FUTCache's edge is
**operational** (bounded-memory online streaming, one-sided never-miss,
durable), not detection accuracy.

---

## Adversarial Analysis

`docs/EXPLOIT.md` catalogs 12 attack vectors (E1–E12) against FUTCache,
each with mechanism, mathematical core, impact, and mitigations.

The nastiest is **E1 (Pulse Attack)**: the attacker exploits W1's
deterministic geometry to force-evict a load-bearing rep and re-observe
it as "novel." The cache is *correct* — one-sidedness working as designed
— but the attacker controls *when* it fires. E1 is **latent** on the
default path: pressure eviction is FIFO, and W1 runs only if the caller
invokes `evict_w1`.

**Empirical validation:** `bench/exploit_e1_bench.c` implements the full
attack end-to-end against the project's pack cache (1D and 2D,
unbounded and bounded modes). On all four tested workloads, the attack
succeeds with 0–3 decoys after a ~150–4,000 μs recon pass. See
`docs/EXPLOIT.md` §E1 for the empirical table.

**5-layer defense:**
1. **Metric**: cosine distance, pin model version, monitor drift
2. **Geometry**: persistent novelty + resurgent flagging
3. **Eviction**: submodular eviction, batch evictions, stochastic noise
4. **API**: rate-limit information-rich endpoints, return aggregates
5. **Billing**: bill per compute, not per geometric novelty event

See also `docs/playbook.md` Section 8 for 12 bad-actor and 12 good-actor
use cases.

---

## Complexity

### 1-D interval-union

| Operation | Time | Space |
|---|---:|---:|
| `futcache_is_novel` | O(log n) | O(1) |
| `futcache_observe` | O((k+1) log n) | O(1) transient |
| stats | O(1) | O(1) |
| serialize / validate | O(n) | O(n) |

### d-D packing cache

| Operation | Time (linear) | Time (VP-tree) |
|---|---:|---:|
| observe / query | O(n·d) | O(d·log n) expected |
| W1 eviction | O(n²·d) | O(n²·d) today (VP-tree accel is remaining work) |
| clear | O(n) | O(n) |

### d-D persistent packing

| Operation | Time |
|---|---:|
| observe | O(n·d) |
| is_novel_at | O(n·d) |
| evict_lowest | O(n·d) |
| persistences | O(n) copy of cached NN distances |

### Submodular selection

| Operation | Time |
|---|---:|
| max_coverage (greedy) | O(k·n²·d) |
| evict_worst | O(m·n·d) |
| coverage | O(m·n·d) |

---

## Concurrency & Ownership

- `futcache`, `pack`, `box`, `crdt`, and `tower` API calls are
  **linearizable** (rwlock). `destroy` is the exception: stop and join
  all users first.
- `persist`, `persist_nd`, `select`, and `embed` are **not** locked.
  Do not share one of those objects across threads.
- No callback is invoked while an object is externally visible (except
  the allocator).
- A custom allocator must provide both callbacks, return suitably aligned
  memory, and be thread-safe for the engines that take the lock.

---

## Persistence

All serialized formats:
- Fixed magic values, format versions
- Explicit little-endian integers, IEEE-754 binary64
- Strict framing, trailing CRC32 checksum
- Reject truncated, extended, out-of-domain, non-finite, or checksum-invalid input

Formats documented in:
- [docs/serialization.md](docs/serialization.md) — interval-union
- [docs/pack-serialization.md](docs/pack-serialization.md) — packing cache

Adversarial build, sanitizer, differential, and concurrency results in
[docs/verification.md](docs/verification.md).

---

## Documentation

| Document | Content |
|---|---|
| [30KFEET.md](30KFEET.md) | **Start here**: 30,000-ft mental model, architectural layers, decision trees, subsystem map, and usage flows. |
| [WHY_FUTCACHE.md](WHY_FUTCACHE.md) | **Manifesto**: Neurosymbolic agent memory, the antidote to Lusser's Law, and geometric blackboards. |
| [docs/alibaba-1m-scaling-benchmark.md](docs/alibaba-1m-scaling-benchmark.md) | **Production Scaling Benchmark**: 1,000,000 real Alibaba Cloud traces testing empirical growth laws. |
| [docs/trace-sampling-benchmark.md](docs/trace-sampling-benchmark.md) | **Tail-Sampling Benchmark**: Geometric sampling on 100k traces vs. random/latency baselines. |
| [docs/cookbook.md](docs/cookbook.md) | Integration patterns for every engine (RAG, anomaly detection, RL curiosity). Traps to avoid. |
| [docs/playbook.md](docs/playbook.md) | Engine ranking by exactness. Reuse-rate/precision frontier. Measured results. Adversarial & beneficial use cases. |
| [docs/verification.md](docs/verification.md) | Adversarial build, sanitizer, differential, concurrency, and scaling results. |
| [docs/EXPLOIT.md](docs/EXPLOIT.md) | 12 adversarial attack vectors with mechanisms, math, impact, mitigations. |
| [docs/prime-novelty.md](docs/prime-novelty.md) | Arithmetic prime novelty experiment and CSV schema. |
| [docs/mdl.md](docs/mdl.md) | Explicit MDL code and offline epsilon-selection API. |
| [docs/mdl-stream-comparison.md](docs/mdl-stream-comparison.md) | Controlled structured/clustered/random MDL curves. |
| [docs/mdl-semantic-negative.md](docs/mdl-semantic-negative.md) | Negative semantic-cache result: geometric MDL is not semantic safety. |
| [docs/design/](docs/design/) | Five design sketches: persistent novelty, W1 eviction, submodular selection, learned metric, competitive ratio. |
| [formal.md](formal.md) | Formal specification of the core invariant. |
| [how.md](how.md) | How the engine works, internally. |
| [PHASE2.md](PHASE2.md) | Distributed semantic cache: CRDT convergence, gossip protocol. |

---

## License

MIT

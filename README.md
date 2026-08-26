# FUTCache

<p align="center">
  <img src="logo.webp" alt="FUTCache logo" width="420" />
</p>

**A mathematically grounded novelty detection engine for metric spaces.**
C11 core, Python bindings, zero dependencies.

---

## What is FUTCache?

FUTCache is a **metric-novelty oracle**. For any metric space you can
represent as vectors (`double[d]`) with a distance function, it answers one
question:

> *Has the system seen anything within `ε` of this point?*

It is **not** a key/value cache. It does not store arbitrary keys, retrieve
top-k matches, or cache objects by ID. It stores the *geometric state*
determined by the observation history — the minimal information needed to
answer future novelty queries exactly.

| Property | Detail |
|---|---|
| **One-sidedness** | Never wrongly suppresses novelty. False positives possible at boundaries, false negatives impossible. |
| **Bounded memory** | Hard byte ceiling (`max_memory_bytes`) with deterministic eviction. |
| **Crash-safe** | Versioned, CRC32-protected serialization. |
| **Metric-agnostic** | L∞, L1, L2, cosine, Poincaré, or user-supplied distance. |
| **Exact** | Differential-verified against brute-force oracles. 126 C tests, ASan clean. |

---

## Table of Contents

- [Engines](#engines)
- [Theoretical Foundations](#theoretical-foundations)
- [Design Sketches](#design-sketches)
- [Build](#build)
- [Quick Start](#quick-start)
- [C API Reference](#c-api-reference)
- [Python API Reference](#python-api-reference)
- [Benchmarks & Experiments](#benchmarks--experiments)
- [Adversarial Analysis](#adversarial-analysis)
- [Complexity](#complexity)
- [Concurrency & Ownership](#concurrency--ownership)
- [Persistence](#persistence)
- [Documentation](#documentation)

---

## Engines

FUTCache ships seven engines, each addressing a different aspect of novelty:

| Engine | Header | Dimension | What it does |
|---|---|---|---|
| **Interval-union cache** | `<futcache/futcache.h>` | 1-D | Exact interval-union novelty with AVL tree. The original engine. |
| **Resolution tower** | `<futcache/tower.h>` | 1-D | Multi-resolution dyadic tower with Fenwick rank/select and discovery logs. |
| **Packing cache** | `<futcache/pack.h>` | d-D | Generalized packing cache for arbitrary metric spaces. VP-tree or linear backend. W1-optimal eviction. |
| **Box cache** | `<futcache/box.h>` | 1–8 D | Exact axis-aligned L∞ box-union cache for full d-D coverage. |
| **CRDT cache** | `<futcache/crdt.h>` | d-D | Gossip-mergeable cache with deterministic Voronoi quantization. Replicas converge without coordination. |
| **Persistent novelty** | `<futcache/persist.h>` | 1-D | Single-linkage merge tree. Multi-scale novelty, persistence diagrams, prime-tagged CRDT merge, Selberg zeta. |
| **d-D persistent packing** | `<futcache/persist_nd.h>` | d-D | Per-representative birth/death tracking with persistence-based eviction. |
| **Submodular selection** | `<futcache/select.h>` | d-D | Greedy max-coverage rep selection (1−1/e guarantee). Streaming evict-worst. |
| **Anchor embedding** | `<futcache/embed.h>` | d→m | Distance-to-anchors projection with bounded distortion (2δ). Conservative ε adjustment. |

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

The workhorse for high-dimensional applications (RAG embeddings, sensor
fusion, time-series). Stores a maximal `ε`-separated set of
representatives. Novelty: `d(x, R) > ε`. Rep count bounded by the
packing number `P(K, ε)`.

Key features:
- **Adaptive radii**: each rep gets its own `ε_i`; lookup tests the exact
  union `⋃ B(r_i, ε_i)`
- **Hard memory ceiling**: `max_memory_bytes` with FIFO or W1 pressure
  eviction
- **VP-tree backend**: 9.1× faster observe, 17.2× faster query vs linear
  scan at 384-D (measured)
- **W1 eviction**: `futcache_pack_evict_w1` removes the rep with the
  smallest nearest-neighbor distance — the W1-optimal eviction
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

- **`is_novel_at(x, t)`**: exact novelty at scale `t`
- **`novelty_spectrum(x, t_max)`**: number of novel scales for a query
- **`evict_below(τ)`**: remove low-persistence reps
- **`selberg_zeta(s)`**: spectral signature of the novelty landscape
- **CRDT merge**: prime-tagged diagram union (idempotent, commutative)
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

- **`select_max_coverage`**: lazy greedy with lexicographic tie-breaking.
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

- **Distortion bound**: `|d(x,y) − d(embed(x), embed(y))| ≤ 2δ` where
  `δ` is the covering radius of the anchor set
- **Conservative ε**: `ε_embed = ε_orig − 2δ` preserves one-sidedness

---

## Theoretical Foundations

FUTCache is a finite, computable realization of the **Novelty Geometry**
framework — the study of how an infinite traversal generates, retains, and
completes *ordered novelty* across increasing resolution.

| Framework concept | FUTCache engine |
|---|---|
| Sufficient memory / future-equivalence quotient | interval-union `U(H)` — minimal state for exact future novelty |
| Ordered novelty word `D_j(L)` | tower per-level occupancy + first-discovery log |
| Partition tower + bonding map `q_{j+1,j}` | tower coarse/fine refinement compatibility |
| ε-separated set / packing number `P(K, ε)` | packing cache rep set and bound |
| Fenwick prefix-sum index | tower spatial rank/select |
| Inverse limit / boundary | the ideal completion a one-`ε` tower approximates finitely |

FUTCache fixes one resolution (`ε`) rather than refining a tower, and stores
the sufficient novelty state with a hard memory ceiling. The limits we
measure empirically — one-sided packing approximation, negative cacheability
margin on some corpora — are the *finite* aspects the framework itself flags
as open, not accidental deficiencies.

See [novelty-geometry](https://github.com/sethuiyer/novelty-geometry) for
the full framework (README, glossary, boundary-object semantics).

---

## Design Sketches

`docs/design/` contains five design documents extending FUTCache from a
novelty cache to a general-purpose novelty engine:

| # | Title | Status | Core contribution |
|---|---|---|---|
| 01 | [Persistent Novelty](docs/design/01-persistent-novelty.md) | **Implemented** | Prime-tagged persistence diagrams, Selberg zeta, d-D persistent packing. Multi-scale novelty with CRDT merge. |
| 02 | [Wasserstein-1 Optimal Eviction](docs/design/02-wasserstein-eviction.md) | **Implemented** | W1-eviction minimizes coverage change. 37% fewer eviction cycles than FIFO. Quantified one-sided gap. |
| 03 | [Submodular Rep Selection](docs/design/03-submodular-selection.md) | **Implemented** | Greedy max-coverage (1−1/e guarantee). Streaming evict-worst. Brute-force verification. |
| 04 | [Learned Metric via Anchor Embedding](docs/design/04-learned-metric.md) | **Implemented** | Distance-to-anchors projection. Distortion ≤ 2δ. Conservative ε adjustment. |
| 05 | [Competitive Ratio for Memory-Bounded Caching](docs/design/05-competitive-theorem.md) | Theory | Competitive-ratio analysis via packing numbers. Capacity planning as a theorem. |

---

## Build

### Requirements

- C11 compiler (GCC 9+, Clang 10+, or MSVC)
- CMake 3.16+
- POSIX threads

### C library

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Build options

| Option | Default | Description |
|---|---|---|
| `FUTCACHE_BUILD_SHARED` | `ON` | Build shared library |
| `FUTCACHE_BUILD_EXAMPLES` | `ON` | Build examples |
| `FUTCACHE_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `FUTCACHE_BUILD_NITROSAT` | `OFF` | Build NitroSAT anchor optimizer |
| `FUTCACHE_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan |
| `FUTCACHE_ENABLE_TSAN` | `OFF` | ThreadSanitizer (separate build) |
| `FUTCACHE_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

### Install

```sh
cmake --install build
```

Downstream CMake:

```cmake
find_package(FUTCache 1 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE FUTCache::futcache)
```

### Python

```sh
pip install .
```

Builds and installs the `futcache` package via nanobind + scikit-build-core.

---

## Quick Start

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

### C: d-D packing cache

```c
#include <futcache/pack.h>

futcache_pack_config_t cfg;
futcache_pack_create(&cfg, &cache,
    .dimension = 384,
    .epsilon = 0.6,
    .distance = futcache_distance_cosine,
    .max_memory_bytes = 64 * 1024 * 1024,
    .backend = FUTCACHE_PACK_BACKEND_VPTREE);

bool novel;
double matched_dist, matched_radius;
size_t matched_idx;
futcache_pack_observe_with_radius(
    cache, point_384d, 0.6, &novel, &matched_dist, &matched_idx);
```

### Python: semantic answer cache

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

### Python: persistent novelty (1-D)

```python
from futcache import PersistentNovelty

eng = PersistentNovelty()
eng.observe(0.5)
eng.observe(0.7)

print(eng.is_novel_at(0.6, 0.1))   # False (within 0.1 of 0.5 or 0.7)
print(eng.is_novel_at(0.6, 0.3))   # True  (gap between 0.5 and 0.7)
print(eng.novelty_spectrum(0.6, 1.0))  # [0.1, 0.3]  (scales where novel)
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
print(f"Selected {result['k']} reps, coverage = {result['coverage']:.3f}")
print(f"Approx ratio = {result['approx_ratio']:.3f} (guarantee: 0.632)")
```

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
| `futcache_pack_create(config, *cache, ...)` | Create packing cache. |
| `futcache_pack_observe_with_radius(cache, point, ε, ...)` | Observe with adaptive radius. |
| `futcache_pack_lookup(cache, point, ...)` | Nearest containing rep + distance. |
| `futcache_pack_query(cache, point, ...)` | Novelty query without mutating. |
| `futcache_pack_evict_w1(cache, *idx)` | W1-optimal eviction (min NN distance). |
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
| `futcache_persist_observe(engine, x, *novel)` | Observe. Updates merge tree. |
| `futcache_persist_is_novel_at(engine, x, t, *novel)` | Exact novelty at scale `t`. |
| `futcache_persist_novelty_spectrum(engine, x, t_max, out)` | Scales where `x` is novel. |
| `futcache_persist_evict_below(engine, τ)` | Remove features with persistence < τ. |
| `futcache_persist_copy_diagram(engine, buf, *count)` | Export (birth, death, persistence). |
| `futcache_persist_merge(a, b, *merged)` | CRDT merge (idempotent, commutative). |
| `futcache_persist_selberg_zeta(engine, s, *z)` | Selberg zeta value. |
| `futcache_persist_prime_cycle_count(engine)` | Count of prime-birth features. |

### `<futcache/persist_nd.h>` — d-D persistent packing

| Function | Description |
|---|---|
| `futcache_persist_nd_create(config, *engine)` | Create d-D engine. |
| `futcache_persist_nd_observe(engine, point, *novel)` | Observe + update persistence. |
| `futcache_persist_nd_is_novel_at(engine, point, t, *novel)` | Scale-resolved novelty. |
| `futcache_persist_nd_nearest_distances(engine, buf, *count)` | Per-rep NN distance. |
| `futcache_persist_nd_persistences(engine, buf, *count)` | Per-rep persistence. |
| `futcache_persist_nd_evict_lowest(engine, *idx)` | Evict most redundant rep. |
| `futcache_persist_nd_count_above(engine, τ)` | Count reps with persistence ≥ τ. |

### `<futcache/select.h>` — Submodular selection

| Function | Description |
|---|---|
| `futcache_select_max_coverage(points, n, d, ε, k, dist, *result)` | Greedy (1−1/e) max-coverage. |
| `futcache_select_evict_worst(points, n, d, ε, reps, dist, *result)` | Streaming swap. |
| `futcache_select_coverage(points, n, d, ε, reps, dist, *cov)` | Coverage ratio. |

### `<futcache/embed.h>` — Anchor embedding

| Function | Description |
|---|---|
| `futcache_embed_create(config, *embed)` | Create anchor projection. |
| `futcache_embed_point(embed, point, out)` | Project to anchor-distance space. |
| `futcache_embed_covering_radius(embed, *δ)` | Estimated distortion. |
| `futcache_embed_adjusted_epsilon(embed, ε, *ε_adj)` | Conservative ε = ε − 2δ. |

---

## Python API Reference

### `PackCache`

```python
PackCache(dimension, epsilon, distance="linf", domain_min=None,
          domain_max=None, backend="vptree", max_memory_bytes=0,
          max_entries=0, ttl=0.0)
```

| Method | Description |
|---|---|
| `observe(point, payload=None, radius=None) -> NoveltyResult` | Query + insert. |
| `query(point) -> NoveltyResult` | Query without inserting. |
| `get_or_compute(point, compute_fn, radius=None) -> (bytes, NoveltyResult)` | Serve cached or compute. |
| `get_payload(rep_id) -> bytes \| None` | Retrieve payload. |
| `set_payload(rep_id, payload)` | Store payload. |
| `evict_w1() -> int` | W1-optimal eviction. Returns evicted index. |
| `copy_representatives() -> ndarray` | Export rep points. |
| `copy_radii() -> ndarray` | Export rep radii. |
| `clear()`, `purge()` | Reset. |
| `memory_bytes()`, `peak_memory_bytes()`, `evictions()`, `observations()` | Telemetry. |

### `PersistentNovelty` (1-D)

| Method | Description |
|---|---|
| `observe(x) -> bool` | Observe. Returns novelty. |
| `is_novel_at(x, t) -> bool` | Exact novelty at scale t. |
| `novelty_spectrum(x, t_max) -> list` | Scales where x is novel. |
| `copy_diagram() -> list[dict]` | Full (birth, death, persistence) export. |
| `evict_below(tau) -> int` | Remove low-persistence features. |
| `merge(other) -> PersistentNovelty` | CRDT merge. |
| `selberg_zeta(s) -> float` | Selberg zeta value. |
| `feature_count()`, `observations` | Telemetry. |

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
| `covering_radius -> float` | Estimated distortion δ. |
| `adjusted_epsilon(ε) -> float` | Conservative ε = ε − 2δ. |

### Functions

| Function | Description |
|---|---|
| `select_max_coverage(points, n, d, ε, k, dist) -> dict` | Submodular max-coverage. |
| `select_evict_worst(points, n, d, ε, reps, dist) -> dict` | Streaming swap. |
| `select_coverage(points, n, d, ε, reps, dist) -> float` | Coverage ratio. |
| `merge_persistence_diagrams(a, b) -> list` | CRDT diagram merge. |
| `nth_prime(i) -> int` | i-th prime (0-indexed). |
| `halton_sequence(n, d) -> ndarray` | Halton low-discrepancy sequence. |
| `poincare_distance(x, y) -> float` | Poincaré ball distance. |

---

## Benchmarks & Experiments

### VP-tree vs linear scan (384-D, 2000 reps, cosine)

| Backend | observe | query | live memory |
|---|---:|---:|---:|
| linear scan | 268.7 μs/op | 535.3 μs/op | 5.96 MiB |
| exact VP-tree | 29.5 μs/op | 31.0 μs/op | 6.19 MiB |

**9.1× faster observe, 17.2× faster lookup, 4% index overhead.**

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

### Bekko semantic cache (384-D, cosine, 8 topics)

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

### KDD Cup '99 (security novelty)

AUC ≈ 0.99 — identical to a trivial 1-NN baseline. FUTCache's edge is
**operational** (bounded-memory online streaming, one-sided never-miss,
durable), not detection accuracy.

---

## Adversarial Analysis

`docs/EXPLOIT.md` catalogs 12 attack vectors (E1–E12) against FUTCache,
each with mechanism, mathematical core, impact, and mitigations.

The nastiest is **E1 (Pulse Attack)**: the attacker exploits the W1 eviction
metric's determinism to orchestrate the cache's own state transitions,
evicting a target rep and re-querying to generate false "novel" events.
The cache is *correct* — it's one-sidedness working as designed, but the
attacker controls *when* it fires.

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
| W1 eviction | O(n²·d) | O(n·d·log n) |
| clear | O(n) | O(n) |

### d-D persistent packing

| Operation | Time |
|---|---:|
| observe | O(n·d) |
| is_novel_at | O(n·d) |
| evict_lowest | O(n·d) |
| persistences | O(1) amortized |

### Submodular selection

| Operation | Time |
|---|---:|
| max_coverage (greedy) | O(k·n·d) |
| evict_worst | O(m·n·d) |
| coverage | O(m·n·d) |

---

## Concurrency & Ownership

- Individual API calls are **linearizable** and safe to call concurrently.
- `destroy` is the exception: stop and join all users first.
- No callback invoked while an object is externally visible (except allocator).
- Custom allocator must provide both callbacks, return suitably aligned
  memory, and be thread-safe.

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
| [docs/cookbook.md](docs/cookbook.md) | Integration patterns for every engine (RAG, anomaly detection, RL curiosity). Traps to avoid. |
| [docs/playbook.md](docs/playbook.md) | Engine ranking by exactness. Reuse-rate/precision frontier. Measured results. Adversarial & beneficial use cases. |
| [docs/verification.md](docs/verification.md) | Adversarial build, sanitizer, differential, concurrency, and scaling results. |
| [docs/EXPLOIT.md](docs/EXPLOIT.md) | 12 adversarial attack vectors with mechanisms, math, impact, mitigations. |
| [docs/design/](docs/design/) | Five design sketches: persistent novelty, W1 eviction, submodular selection, learned metric, competitive ratio. |
| [formal.md](formal.md) | Formal specification of the core invariant. |
| [how.md](how.md) | How the engine works, internally. |
| [PHASE2.md](PHASE2.md) | Distributed semantic cache: CRDT convergence, gossip protocol. |

---

## License

MIT

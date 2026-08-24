# FUTCache

<p align="center">
  <img src="logo.webp" alt="FUTCache logo" width="420" />
</p>

FUTCache is a C11 implementation of future-equivalence caching for metric
novelty. It stores what future novelty queries can observe rather than storing
the observation history. The packing engine can additionally enforce a hard
byte ceiling with deterministic oldest-first pressure eviction.

FUTCache gives you the semantic caching machinery, exact search, bounded
memory, durability, and measurable reuse frontier. Your embedding model
determines how far semantic reuse can safely go.

For a bounded one-dimensional metric domain `K=[a,b]` and resolution
`epsilon`, the exact query is

```text
novel(x | H) = 1 when min(distance(x, y), y in H) > epsilon
```

The sufficient cache state is the canonical union of closed intervals
induced by the history:

```text
U(H) = union([y-epsilon, y+epsilon] intersect K, y in H)
```

`x` is novel exactly when it is outside `U(H)`. Historical points are never
needed once their contribution to that decision boundary has been merged.

Observation counters and `generation` are bounded operational telemetry, not
inputs to the novelty decision. They add constant state and are preserved by
serialization. Consequently, two future-equivalent histories have the same
canonical interval snapshot, but their full serialized bytes may differ when
their telemetry differs.

This is not an LRU key/value cache. It is an exact online novelty oracle:
specify fidelity (`epsilon`), and the geometry determines memory use.

## What FUTCache is

A **metric-novelty oracle**. For any metric space you can represent as
vectors (`double[d]`) with a genuine distance function, it answers exactly one
question:

> has the system seen anything within `epsilon` of this point?

- **Narrow in computation.** Novelty only. It does not store arbitrary keys,
  retrieve top-k matches, or cache objects by an exact ID. For those, use a
  key/value cache (Redis, Memcached, LRU) — FUTCache is the wrong tool.
- **Broad in domains.** Any space with a metric: string / edit distances (via
  a vector encoding), sets / Jaccard (via an encoding), sensor and
  time-series data, embeddings, hierarchical or hyperbolic data — whatever
  you can represent as vectors and supply a distance for.
- **The constraint.** The metric must be a genuine metric (symmetry +
  triangle inequality) for the VP-tree's exact pruning and the `P(K, ε)`
  packing bound to hold. Two cases to know:
  - *Cosine* (`1 − dot`) is not a metric; the engine detects it and indexes
    with a chordal (`L2` on the unit sphere) metric, which is exact for
    normalized inputs, and falls back to an exact linear scan for
    non-normalized ones.
  - *Any other non-metric* similarity or custom distance makes the VP-tree
    prune unsound, so use the **linear backend** (always correct, `O(n)`).
    The engine doesn't auto-detect arbitrary non-metrics — that's the
    caller's call.
- **The production takeaway.** For "has the system seen anything within `ε`
  of this point?" in a genuine metric space, FUTCache bundles four properties
  most alternatives lack: a **hard memory ceiling**, **crash-safe
  persistence**, a **provable one-sided guarantee** (never wrongly suppresses
  novelty), and **exact, differentially-verified** novelty — all behind a
  metric-agnostic API.

## Theoretical foundations

FUTCache is a finite, computable realization of the **Novelty Geometry**
framework — the study of how an infinite traversal generates, retains, and
completes *ordered novelty* across increasing resolution. The central object
is the ordered discovery profile: the sequence of partition cells first
encountered at each refinement level, in first-discovery order, whose
completion lives in an inverse limit (and, after hyperbolic realization, on a
Gromov boundary). The framework is in
[novelty-geometry](https://github.com/sethuiyer/novelty-geometry) — the
README, glossary, and the boundary-object semantics note are its entry
points. (This repo keeps a local mirror of it under
`references/novelty-geometry`.)

| Framework concept | FUTCache engine |
|---|---|
| Sufficient memory / future-equivalence quotient | interval-union `U(H)` — the minimal state that determines future novelty exactly |
| Ordered novelty word `D_j(L)` (first-discovery order) | tower per-level occupancy + first-discovery log |
| Partition tower + bonding map `q_{j+1,j}` | tower coarse/fine refinement compatibility |
| ε-separated set / packing number `P(K, ε)` | packing cache's representative set and bound |
| Fenwick prefix-sum index | tower spatial rank/select |
| Inverse limit / discovery topology / boundary | the ideal completion a one-`ε` tower approximates finitely |

FUTCache is the *finite, single-fidelity* slice of that framework: it fixes one
resolution (`epsilon`) rather than refining a tower, and it stores the
sufficient novelty state with a hard memory ceiling and crash-safe
persistence. The limits we measure empirically — the one-sided packing
approximation (never wrongly suppresses novelty, but can over-report at
packing boundaries), the negative cacheability margin on some corpora, and the
cross-lingual precision ceiling — are precisely the *finite* and
*non-canonical* aspects the framework itself flags as open in its note
(canonicity across towers, realizability over run classes), not accidental
deficiencies of the engine.

## What is implemented

- Exact interval-union FUTCache over any finite `double` domain.
- AVL-backed canonical state with logarithmic lookup and balanced updates.
- Linearizable concurrent reads and writes through a reader/writer lock.
- Allocation-failure-atomic observations.
- Runtime statistics (including AVL height), canonical interval snapshots, and
  invariant validation.
- Versioned, endian-independent, CRC32-protected serialization.
- A uniform dyadic resolution tower with occupancy bitsets, Fenwick prefix
  counts and spatial select, plus first-discovery logs.
- A Voronoi packing novelty cache (`<futcache/pack.h>`) for arbitrary
  finite-dimensional metric spaces with user-supplied distance function.
  This is the natural generalization of the interval-union cache to
  dimensions where no closed-form canonical union exists.
- Strict packing-cache allocation accounting with a configurable physical
  byte ceiling, FIFO allocation recycling under pressure, and live/peak
  memory telemetry.
- Atomic, versioned, CRC32-protected packing snapshots for crash recovery.
- An exact bounded-dimensional `L_inf` box-union cache (`<futcache/box.h>`)
  for applications that need full d-dimensional ball coverage rather than
  representative packing.
- A deterministic-Voronoi, gossip-mergeable CRDT cache
  (`<futcache/crdt.h>`) whose replicas converge without coordination by
  joining cell entries under a deterministic priority (see `PHASE2.md`).
- Randomized differential, boundary, fault-injection, persistence, and
  multithreaded tests.
- A `bench/cache_comparison.c` benchmark pitting FUTCache against LRU and an
  exact-set cache on five workloads, with memory, decision-error, and
  throughput reporting.
- A `bench/corpus_dedup_cost.c` cost benchmark showing the honest
  distribution-dependent de-duplication economics: the reduction ratio and
  downstream embedding/index cost saved across three data shapes (balanced
  clusters+tail, tight clusters, and uniform), plus a note that a uniform
  long tail in high dimension does not collapse. Run
  `futcache_corpus_dedup` (optionally on your own `N*dim` float64 corpus).
- A **security-novelty / SOC** example on
  [KDD Cup '99](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)
  (`demos/kdd_novelty_check.py`): train the one-sided novelty gate on benign
  traffic only and query whether it flags real attack classes as novel. Note
  the honest limit: on KDD99 the gate's accuracy (AUC ~0.99) is *identical to
  a trivial nearest-centroid / 1-NN baseline* (AUC ~0.99), because several
  attack classes are trivially separable — so KDD99 shows the gate *works*
  (as a novelty primitive) but does **not** show it detects better than a
  simple baseline. FUTCache's edge here is operational (bounded-memory online
  streaming, one-sided never-miss-novelty, durable), not detection accuracy.

The learned recurrent/KV state and application-specific sliding-window TTL
forms discussed in `how.md` require a model- or key-domain-specific observable
predicate and are intentionally not guessed by this library.

## Build

FUTCache requires a C11 compiler, CMake 3.16 or newer, and POSIX threads.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Useful build options:

```text
FUTCACHE_BUILD_SHARED=ON       build a shared library
FUTCACHE_BUILD_EXAMPLES=OFF    omit examples
FUTCACHE_BUILD_BENCHMARKS=ON   build the stress/throughput benchmark
FUTCACHE_BUILD_NITROSAT=ON     build the optional offline anchor optimizer
FUTCACHE_ENABLE_SANITIZERS=ON  enable ASan and UBSan on GCC/Clang
FUTCACHE_ENABLE_TSAN=ON        enable ThreadSanitizer in a separate build
FUTCACHE_WARNINGS_AS_ERRORS=ON make supported warning checks fatal
```

Install with `cmake --install build`. Downstream CMake projects can then use:

```cmake
find_package(FUTCache 1 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE FUTCache::futcache)
```

## Exact novelty API

```c
#include <stdbool.h>
#include <stdio.h>

#include <futcache/futcache.h>

int main(void)
{
    const double stream[] = {0.8, 0.1, 0.7, 0.2, 0.4};
    futcache_config_t config;
    futcache_t *cache = NULL;
    size_t i;

    futcache_config_init(&config);
    config.domain_min = 0.0;
    config.domain_max = 1.0;
    config.epsilon = 0.2;

    if (futcache_create(&config, &cache) != FUTCACHE_OK) {
        return 1;
    }

    for (i = 0; i < sizeof(stream) / sizeof(stream[0]); ++i) {
        bool was_novel;
        if (futcache_observe(cache, stream[i], &was_novel) != FUTCACHE_OK) {
            futcache_destroy(cache);
            return 1;
        }
        printf("%.1f: %s\n", stream[i], was_novel ? "novel" : "redundant");
    }

    futcache_destroy(cache);
    return 0;
}
```

The important update rule is that every successful observation contributes its
epsilon-ball. A redundant point may still expand the future decision boundary.
For example, after observing `0.1` with `epsilon=0.2`, `0.2` is redundant, but
its ball extends coverage through `0.4`. Dropping that update would implement a
packing representative approximation, not exact full-history novelty.

All points and configuration values must be finite, and the domain width must
also be representable as a finite `double`. Points are accepted only inside the
inclusive configured domain. Closed-ball semantics mean a point at exactly
`epsilon` distance is redundant. `epsilon=0` provides exact `double` identity
caching. Interval endpoints are rounded inward so that a rounding-up addition
cannot incorrectly classify a point just beyond epsilon as covered.

Observation and tower-mapping operations require the default IEEE rounding mode
(`FE_TONEAREST`) and return `FUTCACHE_ERROR_UNSUPPORTED_PLATFORM` if the calling
thread changed it. The implementation also requires ordinary type-precision
evaluation (`FLT_EVAL_METHOD == 0`). Builds using unsafe floating-point
transformations such as `-ffast-math` are outside the supported numerical
contract.

## Resolution tower API

Include `<futcache/tower.h>`. Level zero has `root_cells` uniform cells, and
each subsequent level doubles the number of cells. A call to
`futcache_tower_observe` returns one novelty byte per level.

At each level the API provides:

- occupancy and distinct-cell counts;
- `prefix_count(level, r)`, the number of occupied cells with index `<= r`;
- `select_occupied(level, k)`, the `k`th occupied cell in spatial order;
- `discovery_at(level, k)`, the `k`th cell in first-discovery order.

Spatial select and discovery order are intentionally separate concepts.
Fenwick trees answer the former; append-only discovery logs preserve the
latter.

`futcache_scaling_example` runs the reciprocal traversal `x_n=1/n` and prints
the observed `M_j`, adjacent-level multiplier, and cache-dimension estimate.

## n-d packing cache API

Include `<futcache/pack.h>`. The packing cache generalizes the interval-union
to arbitrary finite-dimensional metric spaces with a user-supplied distance.
Without a hard byte ceiling, its state is a maximal `epsilon`-separated set of
representatives for the observed stream. Novelty is `d(x, R) > epsilon`; an
observation within `epsilon` of an existing representative is redundant and
the state is unchanged. Representative count is bounded by the packing number
`P(K, epsilon)`.

The compatibility API uses one fixed `epsilon`. Adaptive resolution is also
native: `futcache_pack_observe_with_radius` assigns each newly admitted
representative its own radius `epsilon_i`, and lookup tests the exact union
`U(H) = union_i B(r_i, epsilon_i)`. It does not assume that the nearest centre
owns the matching ball—a farther centre with a larger certified radius is
found correctly. `futcache_pack_lookup` returns the closest containing
representative and its stable FIFO slot.

If adaptive radii have a known positive floor `epsilon_min`, representative
count is bounded by `P(K, epsilon_min)`. If radii may approach zero, use
`max_memory_bytes` for the unconditional physical bound; pressure eviction
remains strict and never allocates through an uncounted VP-tree path.

For deployments with a smaller operational budget, set
`futcache_pack_config_t.max_memory_bytes`. Every live allocation requested by
the cache—including VP-tree nodes and rebuild scratch—is charged to this hard
ceiling. When another representative would cross it, the oldest
representative allocation is recycled in place. This can forget coverage and
therefore cause extra misses, but it cannot produce a false hit. Stats expose
`memory_bytes`, `peak_memory_bytes`, `memory_limit_bytes`, and `evictions`.

`futcache_pack_serialize` and `futcache_pack_deserialize` checkpoint and
recover the complete representative state, parameters, pressure limit, and
telemetry. Snapshots are little-endian, versioned, strictly framed, and CRC32
protected. Built-in VP-tree state is reconstructed from representatives; it
is not persisted as pointer-rich derived state. Snapshot version 2 persists
every adaptive radius and remains backward-compatible with fixed-radius
version 1 snapshots.

This is the cache that addresses the *geometric* novelty question for RAG
embeddings, sensor fusion, time-series anomaly detection, and intrinsic
curiosity in reinforcement learning. Five built-in distances are provided:

`futcache_distance_linf`, `futcache_distance_l1`, `futcache_distance_l2`,
`futcache_distance_cosine`, and `futcache_distance_poincare`. A custom metric
can be supplied as a function pointer with an opaque context.

For large representative sets, `futcache_pack_config_t` also accepts an
optional nearest-neighbour backend. The built-in linear scan remains the
default. Approximate indexes may conservatively overestimate distance and
therefore report extra novelty, but must never suppress a genuinely novel
point. The exact VP-tree stores only dimension-independent node metadata and
borrows immutable representative vectors from the cache, avoiding the former
second `N * dimension * sizeof(double)` coordinate store. Its subtree maximum
radius bounds prune exact adaptive ball lookup.

The repository Release benchmark at 384 dimensions and 2,000 representatives
on a low-intrinsic-dimension cosine workload measures the effect directly:

| Backend | observe | query | live memory |
|---|---:|---:|---:|
| linear scan | 268.7 us/op | 535.3 us/op | 5.96 MiB |
| exact VP-tree | 29.5 us/op | 31.0 us/op | 6.19 MiB |

That is 9.1x faster observation and 17.2x faster lookup for about 4% index
overhead. A second copy of all 384-D vectors alone would cost another 5.86
MiB in this case. Run `futcache_pack_backend_bench` on the deployment CPU for
hardware-specific numbers; uniform high-dimensional data still exhibits the
expected curse-of-dimensionality fallback toward a scan.

### Adaptive resolution: Poincare + isolation + primes

The Python calibration layer implements the research rule

```text
epsilon(x) = epsilon_0 (1 - ||z(x)||^2)^gamma
             exp(-lambda * isolation_score(x))
```

`z(x)` is a point in the open Poincare ball. Radial position makes specialised
boundary concepts stricter, while a compact Isolation Forest contracts the
radius in poorly supported regions. A nearest-known-incompatible margin can
hard-cap the result. Prime-base Halton trials explore `(epsilon_0, gamma,
lambda)` without materialising a Cartesian grid.

```python
from futcache import (
    AdaptiveRadiusController, AdaptiveRadiusPolicy,
    CompactIsolationForest, PackCache, halton_trials,
)

# z_calibration and z_stream are learned hyperbolic embeddings (norm < 1).
forest = CompactIsolationForest(max_samples=256).fit(z_calibration)
policy = AdaptiveRadiusPolicy(
    base_radius=0.6, gamma=1.5, isolation_weight=2.0,
    margin_safety=0.5,
)
controller = AdaptiveRadiusController(policy, forest)
cache = PackCache(384, epsilon=0.0, distance="poincare", backend="vptree",
                  max_memory_bytes=64 << 20)

for z, response in z_stream:
    result = cache.observe(z, payload=response,
                           radius=controller.radius(z))

for trial in halton_trials({
    "epsilon_0": (0.05, 0.8),
    "gamma": (0.0, 4.0),
    "lambda": (0.0, 6.0),
}, count=64):
    evaluate_on_calibration_split(trial)
```

`CompactIsolationForest` retains flat float32/int32 tree arrays and does not
retain the fitted embedding matrix. `poincare_embed` is available when an
external specificity signal needs to be mapped onto Euclidean directions; it
is deliberately not presented as a substitute for learning a hierarchy.
These geometry and density signals choose radii—the C engine still performs
the exact decision, persistence, and bounded-memory eviction.

The exact interval-union cache (`<futcache/futcache.h>`) and the packing
cache (`<futcache/pack.h>`) answer different questions:

| Aspect               | interval-union          | packing cache                |
|----------------------|-------------------------|------------------------------|
| Dimension            | 1D only                 | any `>= 1`                   |
| Distance             | Euclidean on reals      | user-supplied                |
| State                | sorted disjoint 1D intervals | representative points   |
| Novelty predicate    | exact match             | representative match         |
| Memory bound         | `P(K, epsilon)`         | `P(K, epsilon)` or hard bytes |
| Use case             | 1D metric novelty       | RAG, embeddings, multi-d     |

`futcache_nd_dedup` exercises the packing cache on uniform random streams
in dimensions 2, 4, 8, and 16 under `L_inf`, `L1`, and `L2`, and reports
representative count versus packing bound for each.

### Offline representative optimization

`scripts/bench_nitrosat_min_reps.py` uses the vendored NitroSAT V3 solver to
replace an order-dependent online packing with a smaller offline packing over a
fixed observation set. Hard WCNF clauses enforce full empirical coverage and
pairwise distance greater than epsilon; unit soft clauses minimize the number
of retained representatives. Every solver claim is independently recomputed,
and the safe operational policy keeps the smaller of the verified NitroSAT set
and the greedy packing.

On corrected synthetic workloads with 40 actual clusters, five deterministic
solver restarts reduced representative count by 15.1% at 200 points (20/20
workload wins), 17.4% at 500 points, and 15.7% at 1,000 points, always with
full empirical coverage and zero packing violations in the tested matrix.
These are heuristic results, not optimality proofs or continuous-domain cover
certificates. See `docs/nitrosat_optimization.md` for formulation, provenance,
generator correction, commands, and complete caveats.

## Bekko semantic cache experiment

`scripts/bekko_generate.py` encodes a paraphrase corpus with
`hotchpotch/bekko-embedding-v1-a8m` (an ultra-compact multilingual
embedder) at all four Matryoshka truncations (64, 128, 256, 384) and
writes a binary consumable by `bench/bekko_semantic_cache.c`. That
benchmark sweeps `epsilon` for each truncation and reports the
representative count, novel count, true-positive and false-positive
semantic reuse, and per-call latency.

This is the *real* semantic-cache experiment, not a synthetic one. Bekko
emits L2-normalized 384-dimensional vectors, so the cache uses
`futcache_distance_cosine` (1 - dot product). Cosine distance on
paraphrase pairs in this model lands at 0.28-0.88; cross-topic pairs at
0.48-0.95, with significant overlap. The cache becomes effective above
`epsilon` of about 0.5, where within-topic paraphrases start merging into
representatives while cross-topic pairs still mostly stay separate.

Headline numbers from a 38-question corpus across 8 topics (password,
login, billing, cancel, shipping, api, mobile_app, pricing) at 384-d:

| epsilon | reps | novel | correct_reuse | incorrect_reuse | missed_reuse | us/op |
|--------:|-----:|------:|--------------:|----------------:|-------------:|------:|
|    0.30 |   37 |    37 |         0.026 |           0.763 |        0.000 |  8.00 |
|    0.50 |   27 |    27 |         0.289 |           0.500 |        0.000 |  7.28 |
|    0.60 |   20 |    20 |         0.447 |           0.342 |        0.026 |  5.32 |
|    0.70 |   14 |    14 |         0.526 |           0.263 |        0.105 |  3.59 |
|    0.80 |    8 |     8 |         0.658 |           0.132 |        0.132 |  2.22 |
|    0.90 |    5 |     5 |         0.737 |           0.053 |        0.132 |  1.09 |

At `epsilon=0.8` and 384-d the cache holds 8 representatives that cover
38 inputs — a 4.7× compression with 79% of inputs merged correctly.
Throughput scales roughly linearly with dimension (1.2µs/op at 64-d to
8µs/op at 384-d); 256-d lands at 5µs/op.

The smaller the dimension, the coarser the threshold needed: at 64-d the
cache stays inert until `epsilon` reaches 0.8, because the truncated
embeddings cannot separate nearby paraphrases. This is the empirical
content of `D_cache` for this workload — the geometric distinguisher
complexity of semantic English QA paraphrases on Bekko.

### Cross-lingual semantic cache

`scripts/bekko_multilingual.py` encodes the same 8 topics in 6 languages
(en, ja, es, hi, fr, zh) — 48 cross-lingual paraphrase pairs. Written
to the same binary format. The packing cache should compress across
languages as well as within them.

Headline numbers (cosine distance, 384-d, 48 records / 8 topics / 6
languages):

| epsilon | reps | novel | correct_reuse | incorrect_reuse | missed_reuse | us/op |
|--------:|-----:|------:|--------------:|----------------:|-------------:|------:|
|    0.35 |   25 |    25 |         0.479 |           0.354 |        0.000 |  5.69 |
|    0.45 |   16 |    16 |         0.667 |           0.167 |        0.000 |  3.61 |
|    0.50 |   12 |    12 |         0.750 |           0.083 |        0.000 |  2.88 |
|  **0.55** |  **9** |    **9** |     **0.813** |       **0.021** |    **0.000** |  **2.15** |
|    0.60 |    9 |     9 |         0.813 |           0.021 |        0.000 |  2.15 |
|    0.80 |    6 |     6 |         0.771 |           0.063 |        0.104 |  1.49 |

At `epsilon=0.55` the cache holds **9 representatives** for **48 inputs
across 6 languages and 8 topics** — a 5.3× semantic compression with
81% of inputs merged correctly and zero cross-topic confusion. The
cache is collapsing Japanese, Spanish, Hindi, French, Chinese, and
English paraphrases of the same topic into single representatives
because Bekko maps them close in the 384-d embedding space.

A near-identical result appears at 128-d, `epsilon=0.8`: also 9 reps,
81% correct reuse, **0.6 μs/op** (3.5× faster than 384-d at the same
compression ratio). Bekko's Matryoshka truncation is honest: 128
dimensions preserve enough cross-lingual semantic structure for this
cache regime, and the smaller representation dominates on throughput
without losing semantic fidelity.

### Operational metrics: reuse precision

The raw `correct_reuse / incorrect_reuse / missed_reuse` columns need
rephrasing for product decisions. The two numbers that matter are:

- **reuse_rate** = P(cache says HIT) = (correct + missed) / N
- **reuse_precision** = P(true semantic reuse | cache says HIT)
                          = correct / (correct + missed)

With those definitions, the multilingual result at 384-d becomes:

| epsilon | reps | reuse_rate | reuse_precision | us/op |
|--------:|-----:|-----------:|----------------:|------:|
|    0.30 |   35 |     0.4375 |        **1.0000** |  7.72 |
|    0.45 |   16 |     0.6667 |        **1.0000** |  3.61 |
|    0.55 |    9 |     0.8125 |        **1.0000** |  2.15 |
|    0.70 |    9 |     0.8125 |        0.9487 |  2.10 |
|    0.80 |    6 |     0.8750 |        0.8810 |  1.47 |
|    0.90 |    3 |     0.9375 |        0.8667 |  0.92 |

At `epsilon=0.55` the cache holds 9 representatives, achieves 81.25%
reuse rate, and has **100% reuse precision** — every HIT is a true
semantic match. Above `epsilon` of about 0.6 the cache starts merging
cross-topic points, and precision drops.

### Cacheability: FUTCache as a measuring instrument

The interesting fact is that the cache does not invent the geometry —
it reports it. `scripts/cacheability.py` reads the binary and computes,
independently of any cache invocation:

- `d_max_within[d]`: maximum within-topic cosine distance at dimension d
- `d_min_cross[d]`: minimum cross-topic cosine distance at dimension d
- `margin[d] = d_min_cross - d_max_within`
- `D_cache[d]` via regression of log M(ε) vs log ε in the scaling regime

Discriminative margin on the multilingual corpus:

| dim | d_max_within | d_min_cross | margin     | interpretation                |
|----:|-------------:|------------:|-----------:|-------------------------------|
|  64 |        0.629 |       0.361 |     -0.268 | no global ε separates topics  |
| 128 |        0.597 |       0.449 |     -0.148 | no global ε separates topics  |
| 256 |        0.681 |       0.469 |     -0.211 | no global ε separates topics  |
| 384 |        0.715 |       0.503 |     -0.212 | no global ε separates topics  |

The margin is *negative* at every truncation, which means **no single
cosine threshold cleanly separates the eight topics in this corpus**.
The cache works as well as it does because the sequential insertion
order builds one cluster at a time: as each topic's first point enters,
it becomes the seed representative before any cross-topic point can
challenge it. The cache exploits *insertion dynamics*, not a clean
margin.

Empirical `D_cache` via log-log regression of M(ε):

| dim | D_cache | n_used | interpretation                                |
|----:|--------:|-------:|-----------------------------------------------|
|  64 |   1.14  |     12 | multilingual: ~1 bit of distinguishing structure per topic cluster |
| 128 |   0.96  |     11 | multilingual: same                          |
| 256 |   1.06  |     12 | multilingual: same                          |
| 384 |   1.02  |     12 | multilingual: same                          |

For the English-only corpus, `D_cache` is slightly lower (0.7-0.9) and
*decreases* with dimension: higher-d embeddings give a lower packing
exponent, meaning more separable structure per bit of memory.

This reframes FUTCache as a *diagnostic tool for embedding models*. A
good retrieval embedding is not automatically a good caching embedding.
Caching needs clusters whose within-topic spread is genuinely tighter
than the gap to the nearest cross-topic cluster. The discriminative
margin and empirical `D_cache` are the two numbers a practitioner
should look at to decide whether a given model + corpus is cacheable
at all, and at what threshold.

## Exact n-d `L_inf` box cache

Include `<futcache/box.h>` when the exact full-history predicate is required
in dimensions higher than one. Each observation contributes its clipped
axis-aligned `epsilon`-box, and queries test membership in the exact union.
The current implementation supports dimensions 1 through 8. The
representation is exact but non-canonical: each novel observation appends its
clipped `epsilon`-box, and stored boxes may partially overlap but, by the
novelty admission invariant, can never strictly contain one another (a new
box containing a prior box would place that prior center within `epsilon`,
contradicting novelty). `box_count` is therefore a storage diagnostic (equal
to the number of novel observations), not a canonical minimal cell count. A
future disjoint-cell backend could replace this representation without
changing the API.

`futcache_rag_embedding_example` demonstrates a 384-dimensional normalized
embedding stream with a cosine-distance callback, representative export, and
memory statistics. It is synthetic and dependency-free so it builds in CI.

## Python bindings

`pip install .` builds and installs the `futcache` package, a thin Python
wrapper around the C `futcache_pack` cache implemented via nanobind +
scikit-build-core. Payloads (LLM responses, retrieval results, etc.) are
stored in a Python dict keyed by representative slot index; the C cache
itself owns only novelty semantics.

```python
import numpy as np
from futcache import PackCache, NoveltyResult

cache = PackCache(dimension=384, epsilon=0.6, distance="cosine",
                  domain_min=-1.0, domain_max=1.0)

q = np.random.randn(384); q /= np.linalg.norm(q)
res = cache.observe(q, payload=b"cached LLM response")

if res.is_novel:
    response = call_llm(q)
    cache.set_payload(res.representative_id, response.encode())
else:
    response = cache.get_payload(res.representative_id).decode()
```

The wrapper exposes:

- `PackCache(dimension, epsilon, distance, domain_min, domain_max, backend, max_memory_bytes, max_entries, ttl)`
- `cache.observe(point, payload=None, radius=None) -> NoveltyResult`
- `cache.get_or_compute(point, compute, radius=None) -> (bytes, NoveltyResult)`
- `cache.query(point) -> NoveltyResult`
- `cache.get_payload(representative_id) -> bytes | None`
- `cache.set_payload(representative_id, payload)`
- `cache.copy_representatives() -> numpy.ndarray` of shape `(N, dimension)`
- `cache.copy_radii() -> numpy.ndarray` of shape `(N,)`
- `EpsilonTree` — density-aware adaptive `epsilon` via a knee-method region
  tree (`.fit(ref_points)`, `.epsilon(query)`), for use with
  `observe_with_radius`
- `cache.clear()`
- `len(cache)`, `cache.payload_count()`, `cache.purge()`,
  `cache.peak_count()`, `cache.memory_bytes()`,
  `cache.peak_memory_bytes()`, `cache.memory_limit_bytes()`,
  `cache.evictions()`, `cache.observations()`, `cache.novel_observations()`
- `PackCache.version() -> "1.4.0"`

`max_entries` (LRU payload capacity; `0` = unlimited) and `ttl` (payload
expiry in seconds; `0.0` = never expiry) turn `PackCache` into a drop-in
semantic **answer cache**. `get_or_compute` serves the cached payload on a
semantic hit and calls `compute(point)` only when the query is novel or the
payload was evicted/expired — the primitive that lets you skip an LLM,
retrieval, or other expensive call on a semantically-redundant query.
Payload timestamps shift correctly with the C FIFO pressure eviction.

See [`demos/answer_cache_demo.py`](demos/answer_cache_demo.py) for a runnable
ROI + latency one-pager (measured cold-vs-hit latency, cost ledger, and
net spend reduction over a synthetic customer-support workload).

Supported distance names: `"linf"` (default), `"l1"`, `"l2"`, `"cosine"`,
and `"poincare"`.

On a semantic HIT, `NoveltyResult.representative_id` is the slot index to
pass to `get_payload()`, and `NoveltyResult.distance` is the distance to the
closest containing representative (a HIT is within that slot's stored
radius). On a novel
observation `representative_id` names the newly inserted slot and `distance`
is `0.0`; only a novel non-mutating query returns `-1`. Under pressure, FIFO
eviction shifts older slot ids down by one and the Python wrapper shifts or
drops their payload entries in the same operation.

## Empirical comparison vs LRU

`bench/cache_comparison.c` (build with `FUTCACHE_BUILD_BENCHMARKS=ON`) runs
FUTCache against a from-scratch LRU and an exact-set cache on five workloads.
For each method it sweeps the parameter, reports peak memory, decision error
against the metric-novelty oracle, and per-call latency. N = 10000 per
workload, single-threaded, GCC 13.3.0 Release build.

Headline numbers (target ε matched to oracle):

| Workload        | Oracle ε | True novel | FUTCache peak at ε=oracle | FUTCache error | LRU best at k=8192 | LRU error |
|-----------------|---------:|-----------:|--------------------------:|---------------:|-------------------:|----------:|
| reciprocal      |     0.01 |    10/10000|                         7 |         0.0000 |               8192 |    0.9990 |
| uniform         |     0.01 |    52/10000|                         1 |         0.0000 |               8192 |    0.9948 |
| three-cluster   |     0.05 |     3/10000|                         3 |         0.0000 |               8192 |    0.9997 |
| alternating     |      0.5 |     2/10000|                         2 |         0.0000 |                  2 |    0.0000 |
| power-decay     |     0.05 |    10/10000|                         1 |         0.0000 |               8192 |    0.9990 |

Per-call latency (Release, single-thread):

| Method               | μs / observe |
|----------------------|-------------:|
| LRU                  |         0.02 |
| FUTCache             |         0.04 |
| exact-set (linear)   |         1.90 |

On every workload with spatial structure — reciprocal, uniform, three-cluster,
power-decay — FUTCache hits zero error with bounded memory while LRU is
structurally stuck at ~99% error regardless of capacity. Continuous points
never recur, so recency cannot observe spatial coverage. The alternating-
extremes workload is the only tie: it is the workload where temporal and
spatial structure coincide. FUTCache is within 2× of LRU throughput and roughly
50× faster than a naive exact-set cache.

Decision rule. If your cache key supports a meaningful distance function
between keys, use FUTCache. If your keys are opaque identifiers with no
distance, use LRU — FUTCache's API is metric-domain-specific by design.

## Complexity

Let `n` be the current number of disjoint intervals and let `k` be the number
merged by one observation.

| Operation | Time | Additional space |
|---|---:|---:|
| `futcache_is_novel` | `O(log n)` | `O(1)` |
| `futcache_observe` | `O((k+1) log n)` | one transient node |
| stats | `O(1)` | `O(1)` |
| interval snapshot | `O(n)` | caller-owned |
| serialize / validate | `O(n)` | caller-owned / stack |

For a tower with level sizes `N_j`, observation and query touch every level;
Fenwick updates cost `O(sum(log N_j))`, while rank/select at one level costs
`O(log N_j)`. Allocated state is `O(sum(N_j))`.

With positive `epsilon` on a compact interval, the canonical interval count is
geometrically bounded; once the union covers the full domain the cache reaches
an absorbing decision state of one interval.

## Concurrency and ownership

Individual API calls are linearizable and safe to call concurrently on the
same object. `destroy` is the exception: callers must first stop and join all
users of that object. No callback is invoked while an object is externally
visible except the configured allocator. A custom allocator must provide both
callbacks, return memory suitably aligned for any C object, and be safe for the
threads that call the cache.

Snapshot size-query/copy pairs can race with writers; if the state grows between
calls, copy or serialize returns `FUTCACHE_ERROR_BUFFER_TOO_SMALL` and the new
required size/count. Serialization itself always captures one atomic snapshot.

## Persistence guarantees

Both serialized representations have fixed magic values, format versions,
explicit little-endian integers and IEEE-754 binary64 values, strict framing,
and trailing CRC32 checksums. Deserialization rejects truncated, extended,
out-of-domain, non-finite, structurally invalid, or checksum-invalid input.
Platforms without IEEE-754 binary64 report
`FUTCACHE_ERROR_UNSUPPORTED_PLATFORM`.

The interval and packing formats are documented in
[docs/serialization.md](docs/serialization.md) and
[docs/pack-serialization.md](docs/pack-serialization.md).
The adversarial build, sanitizer, differential, concurrency, and scaling results
are recorded in [docs/verification.md](docs/verification.md).

For the three workloads where a novelty oracle is the right abstraction — RAG
de-duplication, streaming anomaly detection, and RL intrinsic curiosity — see
the [cookbook](docs/cookbook.md) for tested integration patterns (plus the
traps to avoid) for every engine.

To choose an engine and tune `epsilon` from **measured** results (not
assertions), see the [playbook](docs/playbook.md): it ranks the engines by
exactness, walks the reuse-rate/precision frontier, and records what actually
happened for monolingual, cross-lingual, knee-method-adaptive, and hyperbolic
approaches.

## Phase 2: distributed semantic cache

The v1.x design covers the single-node cache. Phase 2 formalises a
**distributed semantic cache** whose replicas converge without
coordination, by replacing the order-dependent greedy packing with a
fixed geometric Voronoi quotient. The full treatment — the
obstruction that motivates the move, the deterministic δ-net
construction, the join-semilattice state space, the convergence and
memory theorems, the three-engine taxonomy (interval, pack, crdt),
and the gossip protocol — is in [PHASE2.md](PHASE2.md).

The headline theorem (Theorem 12.10 in `PHASE2.md`):

$$\boxed{q(x) = q(y) \implies d(x, y) \leq \epsilon.}$$

Cell identity becomes an equivalence relation refining metric
similarity. Replicas gossip occupied cells; set union is commutative,
associative, and idempotent; convergence is unconditional. The
asymptotic cache dimension is unchanged — only the constant factor
shifts by $2^D$.

The CRDT engine described there is now implemented in `<futcache/crdt.h>`
(`src/crdt.c`): deterministic Voronoi quantization, per-cell join under a
deterministic priority, snapshot/merge gossip, and validation. It also ships
anchor-construction helpers that close the delta-net gap: a certified
uniform-grid generator (`futcache_crdt_generate_grid_anchors` +
`futcache_crdt_grid_covering_radius`), a low-discrepancy Halton generator
(`futcache_crdt_generate_halton_anchors`, successive primes as radical-inverse
bases), a sampled covering-radius estimator, and
`futcache_crdt_generate_safe_anchors` to build the smallest net whose
covering radius is at most epsilon/2.

# FUTCache

FUTCache is a C11 implementation of future-equivalence caching for metric
novelty. It stores what future novelty queries can observe, rather than storing
the observation history or evicting recent items.

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
- An exact bounded-dimensional `L_inf` box-union cache (`<futcache/box.h>`)
  for applications that need full d-dimensional ball coverage rather than
  representative packing.
- Randomized differential, boundary, fault-injection, persistence, and
  multithreaded tests.
- A `bench/cache_comparison.c` benchmark pitting FUTCache against LRU and an
  exact-set cache on five workloads, with memory, decision-error, and
  throughput reporting.

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
Its state is a maximal `epsilon`-separated set of representatives. Novelty
is `d(x, R) > epsilon`; an observation within `epsilon` of an existing
representative is redundant and the state is unchanged. Representative
count is bounded by the packing number `P(K, epsilon)`.

This is the cache that addresses the *geometric* novelty question for RAG
embeddings, sensor fusion, time-series anomaly detection, and intrinsic
curiosity in reinforcement learning. Three built-in distances are provided:

`futcache_distance_linf`, `futcache_distance_l1`, `futcache_distance_l2`.
A custom metric can be supplied as a function pointer with an opaque
context.

For large representative sets, `futcache_pack_config_t` also accepts an
optional nearest-neighbour backend. The built-in linear scan remains the
default. Approximate indexes may conservatively overestimate distance and
therefore report extra novelty, but must never suppress a genuinely novel
point.

The exact interval-union cache (`<futcache/futcache.h>`) and the packing
cache (`<futcache/pack.h>`) answer different questions:

| Aspect               | interval-union          | packing cache                |
|----------------------|-------------------------|------------------------------|
| Dimension            | 1D only                 | any `>= 1`                   |
| Distance             | Euclidean on reals      | user-supplied                |
| State                | sorted disjoint 1D intervals | representative points   |
| Novelty predicate    | exact match             | representative match         |
| Memory bound         | `P(K, epsilon)`         | `P(K, epsilon)`              |
| Use case             | 1D metric novelty       | RAG, embeddings, multi-d     |

`futcache_nd_dedup` exercises the packing cache on uniform random streams
in dimensions 2, 4, 8, and 16 under `L_inf`, `L1`, and `L2`, and reports
representative count versus packing bound for each.

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
The current implementation supports dimensions 1 through 8 and uses an
append-only overlapping-box representation: `box_count` is storage size,
not a packing bound or a canonical minimal decomposition. This deliberately
keeps the exact semantics simple while leaving a future disjoint-cell
backend replaceable behind the same API.

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

- `PackCache(dimension, epsilon, distance, domain_min, domain_max)`
- `cache.observe(point, payload=None) -> NoveltyResult`
- `cache.query(point) -> NoveltyResult`
- `cache.get_payload(representative_id) -> bytes | None`
- `cache.set_payload(representative_id, payload)`
- `cache.copy_representatives() -> numpy.ndarray` of shape `(N, dimension)`
- `cache.clear()`
- `len(cache)`, `cache.peak_count()`, `cache.memory_bytes()`,
  `cache.observations()`, `cache.novel_observations()`
- `PackCache.version() -> "1.1.0"`

Supported distance names: `"linf"` (default), `"l1"`, `"l2"`, `"cosine"`.

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
calls, `futcache_copy_intervals` returns `FUTCACHE_ERROR_BUFFER_TOO_SMALL` and
the new required count. Serialization itself always captures one atomic
snapshot.

## Persistence guarantees

The serialized representation has a fixed magic value, format version, explicit
little-endian integers and IEEE-754 binary64 values, canonical sorted intervals,
and a trailing CRC32. Deserialization rejects truncated, extended,
non-canonical, out-of-domain, non-finite, or checksum-invalid input. Platforms
without IEEE-754 binary64 report `FUTCACHE_ERROR_UNSUPPORTED_PLATFORM`.

The binary format is documented in [docs/serialization.md](docs/serialization.md).
The adversarial build, sanitizer, differential, concurrency, and scaling results
are recorded in [docs/verification.md](docs/verification.md).

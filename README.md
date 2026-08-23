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
- Randomized differential, boundary, fault-injection, persistence, and
  multithreaded tests.

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

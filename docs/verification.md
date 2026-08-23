# FUTCache verification report

Verification date: 2026-08-23. Host compiler: GCC 13.3.0 on Linux.

## A. Repository understanding

The exact novelty cache is an opaque, allocator-aware object containing a
reader/writer lock and an AVL tree of sorted, pairwise non-overlapping closed
intervals. An observation computes its clipped epsilon-ball, determines novelty
against the pre-update tree, and merges the ball into the canonical union. A
replacement node is allocated before any destructive merge, making allocation
failure atomic. Telemetry and covered measure are maintained beside the
decision state.

The tower is a separately opaque object. Each uniform level has an occupancy
bitset, a Fenwick tree, and a fixed-capacity first-discovery log. A point is
mapped once at the finest level and shifted to every ancestor, making refinement
compatibility structural rather than dependent on repeated floating-point
rounding.

Core snapshots use an explicit little-endian binary format with IEEE-754
binary64 fields and CRC32. The parser checks framing, overflow, checksum,
configuration, telemetry relations, interval bounds/order/canonicality, and AVL
invariants. Both objects support custom allocation and linearizable public
operations; destruction requires external quiescence.

## B. Build matrix

| Configuration | Result | Warnings/findings |
|---|---|---|
| GCC Debug, static, strongest project warnings, `-Werror` | 55 tests pass | none |
| GCC Release, shared, strongest project warnings, `-Werror` | 55 tests pass | none |
| GCC ASan + UBSan, Debug | 55 tests pass | none; leak detector disabled as noted below |
| GCC TSan, Debug, non-PIE plus ASLR-disabled environment workaround | 55 tests pass | no races or lock findings |
| Release examples and benchmarks, `-Werror` | build and run | RAG, n-d packing, LRU comparison clean |
| GCC `-fanalyzer` | library builds | no analyzer findings |
| Installed shared package + external `find_package` consumer | builds and runs | none |
| Clang | not run | compiler is not installed on this host |

LeakSanitizer cannot run in this managed container because its ptrace startup
fails. Valgrind is not installed. Library-owned cleanup is instead checked with
counting allocators across every tower-construction allocation site, interval
insertion failures, bridge failures, and every deserialization allocation site.
This is a verification limitation, not an observed leak.

## C. Invariants verified

- Novelty equals an independent full-history distance oracle.
- Every successful observation contributes its ball, including redundant ones.
- Distance exactly epsilon is redundant; one representable value beyond it is
  novel.
- Interval state is sorted, disjoint, non-mergeable, domain-clipped, and equal
  to an independently sorted/merged reference union.
- AVL heights and balance factors, node count, coverage measure, and telemetry
  relations remain valid after each adversarial mutation checkpoint. A global
  regression also bounds root height by `2 ceil(log2(n+1))` after 10,000 sorted
  exact-state insertions.
- Coverage is monotone; duplicate insertion is idempotent; final coverage is
  permutation-invariant. Once full coverage is reached, every sampled future
  query and insertion is redundant and the one-interval state never grows.
- Allocation failure leaves pre-operation semantics, structure, counters, and
  locks intact.
- Snapshot round trips preserve canonical state, parameters, telemetry, and
  future behavior.
- Every one-byte snapshot mutation and every truncation offset is rejected.
  Valid-checksum snapshots with NaN parameters, impossible telemetry, absurd
  counts, or touching intervals are also rejected.
- Tower occupancy, prefix sums, spatial select, discovery order, coarse/fine
  novelty implication, adjacent projection, and composed projection match
  independent arrays and arithmetic. A focused two-level test checks the exact
  discovery-word identity `q_(1,0)(D_1(L)) = D_0(L)` element by element.
- The packing cache has twelve tests covering configurable metrics, Voronoi
  representative separation, query purity, allocator failure, concurrency,
  the pluggable nearest-neighbour backend lifecycle, and the nearest-site
  id/distance lookup (`futcache_pack_nearest`). Approximate backend
  semantics are explicitly one-sided: over-estimation may report extra
  novelty, but must not suppress a genuinely novel point.
- The exact `L_inf` box cache has five tests covering bounded 2D/3D unions,
  clipping, domain errors, query purity, clear semantics, allocation
  failure atomicity, and lifecycle-telemetry/containment validation. Its
  boxes may partially overlap but, by the novelty admission invariant, can
  never strictly contain one another; the representation is exact but not a
  canonical minimal cell decomposition.
- The deterministic-Voronoi CRDT cache has thirteen tests covering
  configuration validation, hand-checked quantization (including tie
  breaking), cell test-and-set and semantic reuse, priority-based merge,
  idempotence, two-way gossip convergence, snapshot roundtrip, clear,
  allocation-failure atomicity, and the anchor-construction family
  (`futcache_crdt_generate_halton_anchors`,
  `futcache_crdt_generate_grid_anchors`,
  `futcache_crdt_grid_covering_radius`,
  `futcache_crdt_estimate_covering_radius`, and
  `futcache_crdt_generate_safe_anchors`). The grid construction carries a
  certified covering radius under L1/L2/L_inf (no sampling); the Halton net
  uses successive primes as radical-inverse bases and carries only an
  estimated (lower-bound) radius, which the tests assert never exceeds the
  certified grid radius for the same resolution.
- Concurrent observations, queries, serialization/restoration, tower reads,
  and tower writes remain valid under TSan.

## D. Bugs found and fixed

### P0: outward floating-point rounding could answer novelty incorrectly

Minimal reproducer before the fix:

```text
x       = 1.0
epsilon = nextafter(2^-52, 0)
query   = nextafter(1.0, +infinity)
```

The exact represented distance is `2^-52`, which is greater than epsilon, so
the query must be novel. Plain `x + epsilon` rounded upward to the query value,
causing the interval cache to report redundant. The cause was treating a
round-to-nearest sum as an exact closed-ball boundary.

The update now uses a TwoSum residual and `nextafter` to round the lower endpoint
upward and the upper endpoint downward. A 4,096-case long-double oracle plus
explicit ULP boundary tests covers both directions. Non-default rounding modes
are rejected.

No unresolved P0 or P1 defect was found.

## E. Differential-testing statistics

- Fixed randomized seeds: 4 major streams, all printed in source for replay.
- Full-history differential stream: 3,000 observations, 4,096-point dyadic
  coordinate grid, exact epsilon boundary.
- Independent canonical-union comparison: 2,048 observations in forward and
  reverse order.
- Directed-rounding oracle: 4,096 independently computed balls.
- Future-equivalence continuation: 5,000 observations applied to two distinct
  but initially equivalent histories.
- Exhaustive exact-novelty model: all `2^8 = 256` states of an eight-symbol
  alphabet and all eight distinguishing queries per state.
- Absorbing-state sweep: 1,001 domain-wide future queries and insertions after
  full coverage, with interval count checked after every insertion.
- AVL height regression: 10,000 sorted exact-state insertions followed by
  10,000 reverse-order duplicates under a global logarithmic height ceiling.
- Tower differential stream: 4,000 observations across six levels, with every
  prefix, select, discovery, and projection result checked.
- n-d packing benchmark: 5,000-point streams in dimensions 2, 4, 8, and 16,
  across `L_inf`, `L1`, and `L2`, plus a 384-dimensional cosine embedding
  demonstration over 256 clustered queries.

Distributions include uniform dyadic values, duplicates, sorted/reverse-sorted
exact states, redundant clusters, bridge merges, reciprocal traversal, domain
endpoints, subnormals, and ULP-neighbor boundary values.

## F. Concurrency statistics

- Exact cache: four writers x 2,000 operations.
- Snapshot stress: two writers x 3,000 operations concurrent with two snapshot
  readers x 500 serialize/restore/validate cycles.
- Tower: four writers and four readers x 1,000 operations.
- TSan result: clean with `halt_on_error=1`.
- Deadlocks: none observed.
- Linearizability argument: each public operation has one object-level read or
  write critical section; outputs are copied only from state held under that
  section. No lock-free multi-step mutation exists.

Destroy-during-use is intentionally outside the documented contract.

## G. Fuzzing statistics

Coverage-guided libFuzzer/AFL execution was not available because Clang and AFL
are absent. The deterministic parser attack mutates every byte, truncates at
every byte offset, appends trailing data, and supplies several malformed inputs
with recomputed valid checksums. Unique crashes: 0.

## H. Complexity results

One Release benchmark sample on this host:

```text
100,000 sorted epsilon=0 inserts: 0.021 s, 4.81 M operations/s
2,000,000 AVL queries:           0.405 s, 4.94 M operations/s
1,600,076-byte snapshot:         0.014 s, 106.9 MiB/s
250,000 x 12-level observations: 0.011 s, 22.36 M streams/s
```

The fragmented cache retained 100,000 intervals and passed full AVL validation.
The documented bounds are `O(log n)` query, `O((k+1) log n)` update for `k`
merged components, `O(n)` snapshot, and `O(sum(log N_j))` for a novel tower
update.

## I. Theory-validation results

The defining implication was tested executably:

```text
U_epsilon(H) = U_epsilon(H')
    => identical novelty output for a shared 5,000-item continuation
    => equal canonical interval state after every checkpoint
```

For `L=(1,1/2,...,1/65536)` and cells `N_j=2^j`, the final levels were:

| j | M_j | M_j / M_(j-1) | D_hat |
|---:|---:|---:|---:|
| 12 | 128 | 1.4222 | 0.5833 |
| 13 | 181 | 1.4141 | 0.5769 |
| 14 | 256 | 1.4144 | 0.5714 |
| 15 | 362 | 1.4141 | 0.5667 |
| 16 | 511 | 1.4116 | 0.5623 |

The adjacent multiplier approaches `sqrt(2)` and the finite-resolution
dimension estimate trends toward `1/2`, as predicted.

## J. Release verdict

**RELEASE READY WITH NON-BLOCKING ISSUES**

No correctness, memory-safety, race, persistence, or allocation-atomicity
failure remains in the tested scope. The non-blocking gaps are environmental:
no Clang/AFL coverage-guided fuzz run and no usable LeakSanitizer/Valgrind run on
this host. Custom allocator accounting covers the library's material ownership
failure paths, but those external tools should still be added to CI on a host
that supports them.

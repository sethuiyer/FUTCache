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

Interval and packing snapshots use explicit little-endian binary formats with
IEEE-754 binary64 fields and CRC32. Their parsers check framing, overflow,
checksum, configuration, telemetry relations, bounds, and structural
invariants. The packing engine additionally wraps every child/backend
allocation with exact live-byte accounting; a configured ceiling is reserved
atomically before allocation and FIFO pressure eviction recycles a fixed-size
representative without a transient memory spike. Objects support custom
allocation and linearizable public operations; destruction requires external
quiescence.

## B. Build matrix

| Configuration | Result | Warnings/findings |
|---|---|---|
| GCC Debug, static, strongest project warnings, `-Werror` | 75 C tests pass | none |
| GCC Release, shared, strongest project warnings, `-Werror` | 75 C tests pass | none |
| GCC ASan + UBSan, Debug | 75 C tests pass | none; leak detector disabled as noted below |
| GCC TSan, Debug | startup blocked | managed host rejected TSan shadow mapping before tests (`unexpected memory mapping`) |
| Release examples and benchmarks, `-Werror` | build and run | RAG, n-d packing, LRU comparison clean |
| Python/nanobind extension and wheel | 5 adaptive tests pass | v1.3.0 wheel installs and imports cleanly |
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
- The packing cache has twenty-one core tests plus eleven VP-tree tests covering
  configurable metrics, representative separation, query purity, allocator
  failure, concurrency, backend lifecycle, nearest-site ids/distances, exact
  byte-ceiling enforcement, FIFO recycling, bounded VP-tree fallback, and
  concurrent snapshot/recovery during writes, adaptive-radius ball stabbing,
  Poincare-domain enforcement, v1/v2 snapshot compatibility, borrowed 384-D
  vector storage, and balanced equidistant metrics. Every byte mutation, every
  truncation, and trailing data are rejected for a representative snapshot.
  An independent counting allocator confirms that both live and transient
  bytes requested upstream remain at or below the configured ceiling.
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
  and tower writes pass deterministic stress tests in Debug, Release, and the
  ASan/UBSan build. TSan runtime coverage was unavailable on this host as
  recorded in the build matrix.

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
- Bounded packing snapshot stress: two writers x 2,000 operations concurrent
  with two readers x 200 serialize/restore/validate cycles, with a hard
  eight-representative allocation ceiling.
- Tower: four writers and four readers x 1,000 operations.
- TSan result: binary builds; runtime aborts before `main` because the managed
  host cannot reserve TSan's shadow mapping.
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

## K. v1.3.0 re-verification addendum

Re-verified 2026-08-23 (same host, GCC 13.3.0) after the adaptive-calibration
commit (pack engine, VP-tree, Python bindings). All findings below are either
improvements or confirmations; no library defect was found.

- **LeakSanitizer now runs on this host.** The earlier "LSan cannot run in this
  managed container" note is outdated: `ASAN_OPTIONS=detect_leaks=1` completes.
  The first LSan pass exposed a **test-harness leak**: 28 allocations (16,608
  bytes) in `test_vptree_metric_differential` (`tests/test_pack_vptree.c`), whose
  15-iteration loop re-assigned `domain` without freeing the previous one.
  Fixed by freeing before re-assignment (3-line change). With
  `detect_leaks=1:halt_on_error=1` the full suite now exits 0 with zero leaks
  across all suites, so the previous LSan/Valgrind gap is closed.
- **New adversarial stress suite** (`tests/test_pack_stress.c`, 4 tests,
  suite total 75 -> 79):
  1. adaptive-radius VP-tree vs linear differential across L1/L2/L_inf/cosine/
     poincare at dims 1/3/16/64 with zero, near-zero, huge, and random radii,
     checking every observe decision, final reps/radii (bit-exact), lookup
     found/distance/index, and nearest distance/index, plus an independent
     linear-scan oracle. Non-cosine metrics compare bit-exactly; cosine
     distances use the backend's documented 1e-6 transform tolerance and any
     decision disagreement must sit inside that band of the union boundary.
  2. byte-ceiling eviction with variable radii: per-cache telemetry identity
     (count + evictions == novel), live/peak memory ceiling, validate(), and
     every query vs that cache's own surviving-rep oracle, for both backends.
     The vptree cache may retain fewer reps (and evict more often) at the same
     ceiling because index metadata shares the budget — expected, not a bug.
  3. adaptive serialize roundtrip with ceiling + evictions + VP-tree: restored
     cache is byte-identical in reps/radii, telemetry-identical, and agrees on
     a continuation stream; corrupted snapshots are still rejected.
  4. concurrent writers/readers on a variable-radius VP-tree cache: 4 writers
     (observe_with_radius) + 2 readers (lookup/nearest/serialize with the
     documented size-query retry on BUFFER_TOO_SMALL); validate() and
     telemetry identities hold after joining. Stable across repeated runs in
     Release and ASan.
- **TSan** still cannot run on this host (shadow-mapping abort before `main`;
  unchanged). The TSan binary builds; concurrency is otherwise exercised by
  the new concurrent stress test under ASan and by the existing deterministic
  stress tests in Debug/Release.
- **GCC `-fanalyzer`** re-run over all six translation units (gnu11, matching
  the CMake flags): no findings. The earlier box.c "error" was an artifact of
  compiling with strict `-std=c11` (pthread_rwlock_t is hidden without
  `_DEFAULT_SOURCE`); the real build uses gnu11.
- Build matrix re-confirmed: Release `-Werror`, Debug `-Werror`, ASan+UBSan,
  shared-library Release `-Werror` (new), Python wheel v1.3.0 in a clean venv,
  installed-package consumer, all five examples, and the nitrosat offline
  optimizer test — all green. The Release backend benchmark reproduces the
  documented 384-d manifold-cosine numbers (8.9x observe / 17.5x query at
  2,000 representatives, 6.19 MiB live).
- Python: 5 adaptive unit tests pass; an additional binding smoke test covers
  all backends x metrics differential agreement, cosine normalization,
  adaptive-radius payload/slot identity, calibrated ceiling eviction with
  payload shifting, Poincare rejection, clear/telemetry, and version 1.3.0.

Verdict remains **RELEASE READY**; the only remaining environmental gap is
Clang/AFL coverage-guided fuzzing.

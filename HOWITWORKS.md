# How FUTCache works

This document is the engineering onboarding reference for FUTCache. It
describes what the library computes, how each engine is implemented, where
the numerical and concurrency hazards are, and how the code is verified. It
assumes the reader can read C and is comfortable with basic metric-space and
data-structure vocabulary. It does not assume prior exposure to the codebase.

The intended reading order is the section order. Sections 2 through 6
establish the shared model. Sections 7 through 11 cover one engine each.
Sections 12 through 16 cover the cross-cutting machinery. Sections 17 through
20 are reference material.

---

## 1. Scope

FUTCache is a C11 library that answers one question: given a stream of points
in a metric space and a threshold epsilon, is the next point "new", in the
sense that nothing within distance epsilon of it has been seen before. The
library computes this without retaining the full stream. It keeps a compact
representation of everything seen — a union of epsilon-balls, a set of
representative points, a set of boxes, a set of dyadic cells, or a set of
Voronoi cells — and tests each new point against that representation.

The library ships five engines behind five headers:

- `futcache/futcache.h` — exact one-dimensional interval-union cache.
- `futcache/pack.h` — adaptive n-dimensional Voronoi packing cache.
- `futcache/box.h` — n-dimensional axis-aligned box cache (dimensions 1–8).
- `futcache/tower.h` — one-dimensional dyadic resolution tower.
- `futcache/crdt.h` — deterministic-Voronoi CRDT cache for gossip-based
  replication.

There is also a Python binding, `futcache`, that wraps the packing engine and
adds payload storage for semantic caching.

The library is version 1.1.0. The version is defined in
`include/futcache/futcache.h` and mirrored in `CMakeLists.txt` and
`pyproject.toml`.

---

## 2. The problem

A novelty cache is a data structure that answers, for a query point x, whether
x is within epsilon of any point previously observed. The naive solution keeps
every observation and scans the list on each query. That is correct but costs
O(n) per query and O(n) memory, where n is the number of observations. The
quantity n is unbounded: a long-running process can observe millions of
points.

The observation that makes a novelty cache useful is that the *decision
boundary* is what matters, not the history. Two histories that induce the same
set of "redundant" points are interchangeable for the purpose of answering
novelty queries. FUTCache stores the decision boundary in a canonical or
near-canonical form and lets the history fall away.

The decision boundary is formalized as the epsilon-ball union. Write B_e(h)
for the closed ball of radius e around a point h. After observing the
multiset H, a point x is redundant if and only if

    x ∈ U_e(H) = ⋃_{h ∈ H} B_e(h).

A point is novel if and only if it lies outside that union. The library's
term for this predicate is *metric novelty*. The set U_e(H) is the smallest
state that is sufficient to answer every future novelty query, and it is the
object the engines try to represent.

The cost of the naive approach is the reason the library exists. A stream of
n observations with a linear scan costs O(n^2) time in total and O(n) memory,
and neither number is acceptable in a long-running process. The interval and
box engines reduce memory to something bounded by the geometry of the domain;
the packing engine reduces memory to the packing number; the tower fixes the
memory entirely by discretizing. Time costs drop to logarithmic or linear in
the *state size* rather than the observation count. The trade that buys these
bounds is a small amount of per-observation arithmetic — an interval merge, a
distance computation, a box scan, a cell lookup — and, in the approximate
engines, the acceptance of one-sided error.

The three recurring workloads the library serves are worth naming, because
they shape the API. The first is semantic caching: each observation is an
embedding of a user query, and a hit lets the caller skip an expensive model
call and return a cached answer. The second is deduplication: each
observation is a sensor reading or an event, and a hit means the reading was
already recorded. The third is spatial indexing: each observation is a
coordinate, and a hit means the neighborhood was already explored. All three
are instances of the same predicate with different distances and
dimensionalities.

---

## 3. The core idea: future-equivalence

Two histories H and H' are future-equivalent when they answer every future
novelty query identically, that is, when U_e(H) = U_e(H'). The interval engine
stores U_e(H) exactly. The packing engine stores a representative set R whose
ball union approximates U_e(H) from below. The box engine stores a subset of
the ball union. The tower stores a dyadic over-approximation at multiple
scales. The CRDT engine stores a fixed Voronoi quotient of the space.

The phrase "future-equivalence" names the invariant that drives all five
engines: the cache may discard a point as soon as the point's ball is
contained in the ball union of what remains, because discarding it changes no
future answer.

The practical consequences of this framing are threefold. First, storage is
bounded by the geometry of the domain rather than by the number of
observations; in the interval and box engines the state size is bounded by
the packing number of the domain at scale epsilon. Second, redundancy is
idempotent: observing the same point twice changes nothing. Third, the
predicate is monotone: once the union covers the whole domain, every future
query is redundant and the state stops growing.

One convention is fixed across all engines and is stated here so the rest of
the document can rely on it. Redundancy is inclusive at the boundary: a point
whose distance to the nearest observed point is exactly epsilon is redundant,
and a point one representable binary64 step beyond epsilon is novel. The
convention is not arbitrary; it is the one the closed ball B_e defines, and
it is what makes the decision boundary a closed set. The interval engine's
inward rounding (section 12) is the mechanism that enforces the convention in
the presence of floating-point error.

A second consequence of the framing is a size bound. In any bounded domain
the number of pairwise epsilon-separated points is finite; its maximum is the
packing number of the domain. The packing engine's representative set is
epsilon-separated, so its size never exceeds the packing number no matter how
many observations arrive. The interval and box engines enjoy a similar bound
through the geometry of the union. The tower and CRDT engines replace the
geometric bound with a fixed discretization the caller chooses.

---

## 4. Repository layout

The tree is small and flat.

```
CMakeLists.txt                 build definition
LICENSE                        MIT license
README.md                      overview, build, API summary
HOWITWORKS.md                  this document
formal.md                      the metric-novelty theory
how.md                         design rationale and non-goals
PHASE2.md                      distributed/CRDT design and theorems
testplan.md                    test strategy
pyproject.toml                 Python package metadata
include/futcache/              public headers
    futcache.h                 interval engine + status codes + allocator
    pack.h                     packing engine
    box.h                      box engine
    tower.h                    tower engine
    crdt.h                     CRDT engine
    export.h                   visibility macros
src/                           one .c file per engine
    futcache.c                 interval union (AVL)
    pack.c                     Voronoi packing
    box.c                      box union
    tower.c                    dyadic tower
    crdt.c                     CRDT engine
tests/                         test harness and per-engine suites
    test.h                     harness macros
    test_main.c                driver and registration
    test_futcache.c            18 tests
    test_pack.c                12 tests
    test_tower.c               7 tests
    test_box.c                 5 tests
    test_crdt.c                7 tests
examples/                      C demonstration programs
    basic.c                    interval engine walkthrough
    rag_embedding.c            384-d cosine packing demo
    reciprocal_scaling.c       scaling demonstration
    box.c                      2-D box demo
    crdt.c                     two-replica CRDT demo
bench/                         benchmarks
    cache_comparison.c         FUTCache vs LRU vs exact set
    bekko_semantic_cache.c     multilingual semantic-cache benchmark
    nd_dedup.c                 n-d deduplication benchmark
scripts/
    cacheability.py            cacheability analysis
    bekko_generate.py          benchmark data generation
    bekko_multilingual.py      multilingual data generation
python/
    futcache/__init__.py       Python wrapper and docstrings
    futcache/_core.cpp         nanobind extension
    examples/rag_semantic_cache.py  end-to-end RAG cache demo
docs/
    serialization.md           interval snapshot format
    verification.md            build and test results
```

The split between `include/` and `src/` is one header per engine. Each engine
is independent; the only shared header is `futcache/futcache.h`, which
defines the status enum, the allocator struct, and the version macros. The
engines do not call each other.

---

## 5. Building

The library requires a C11 compiler, CMake 3.16 or newer, and POSIX threads
(pthreads). It has no other runtime dependencies. The core library builds as
a static library by default.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The build options are:

- `FUTCACHE_BUILD_SHARED=ON` — also build a shared library.
- `FUTCACHE_BUILD_EXAMPLES=OFF` — skip the example programs.
- `FUTCACHE_BUILD_BENCHMARKS=ON` — build the stress and throughput benchmarks.
- `FUTCACHE_ENABLE_SANITIZERS=ON` — enable ASan and UBSan (GCC/Clang).
- `FUTCACHE_ENABLE_TSAN=ON` — enable ThreadSanitizer in a separate build.
- `FUTCACHE_WARNINGS_AS_ERRORS=ON` — make supported warning flags fatal.

Installation is `cmake --install build`. A downstream CMake project consumes
the installed package with

```
find_package(FUTCache 1 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE FUTCache::futcache)
```

The install tree contains the public headers, the static (and optionally
shared) library, the CMake package configuration files, and the LICENSE file.
The exported target `FUTCache::futcache` carries the include directories and
link interface, so a consumer needs no manual include or link flags.

The tests are a single executable, `futcache_tests`, that runs every suite
and prints one line per test. The test count is deliberately recorded in
`docs/verification.md` so that a change in the count is visible in review.

---

## 6. Shared API conventions

### 6.1 Status codes

Every function that can fail returns a `futcache_status_t`. The values are:

- `FUTCACHE_OK` — success.
- `FUTCACHE_ERROR_INVALID_ARGUMENT` — a null pointer, a bad dimension, a
  non-finite coordinate, or an invalid configuration.
- `FUTCACHE_ERROR_OUT_OF_MEMORY` — an allocation failed.
- `FUTCACHE_ERROR_OUT_OF_RANGE` — a point lies outside the configured domain.
- `FUTCACHE_ERROR_BUFFER_TOO_SMALL` — an output buffer was too small (the
  required size is written back through an in/out count parameter).
- `FUTCACHE_ERROR_CORRUPT_DATA` — a deserialized or validated state violated
  an invariant.
- `FUTCACHE_ERROR_UNSUPPORTED_PLATFORM` — the floating-point environment does
  not satisfy the correctness preconditions (see section 12).
- `FUTCACHE_ERROR_SYSTEM` — a pthread operation failed.

`futcache_status_string()` maps a code to a stable name for diagnostics.

### 6.2 Allocators

Each config struct carries a `futcache_allocator_t` with three fields:
`allocate`, `deallocate`, and an opaque `context`. The callbacks must either
both be provided or both be null; null selects `malloc`/`free`. The allocator
is copied into the engine at create time. Engines route every allocation
through it, including the cache object itself, so a counting allocator can
verify that a destroyed cache frees everything it allocated.

### 6.3 Locking

Every engine owns a `pthread_rwlock_t`. Mutating operations — `observe`,
`merge`, `clear` — take the write lock. Read operations — `is_novel`,
`query`, `get_stats`, `copy_*`, `snapshot`, `validate`, `serialize` — take
the read lock. `observe` is linearizable: it computes novelty and applies the
mutation while holding the write lock, so two concurrent observations of the
same novel region cannot both be reported novel.

`destroy` takes no lock. The caller must ensure the cache is quiescent before
destroying it, which is the standard C lifetime rule and is stated on each
destroy function.

### 6.4 Telemetry

Each engine exposes counters through a `get_stats` function. The counters are
`uint64_t` and saturate at `UINT64_MAX` instead of wrapping. Saturation keeps
the telemetry invariants (for example, "novel observations never exceed total
observations") true even after overflow. The invariants are checked by each
engine's `validate` function and, for the interval engine, by the
deserializer.

The common counters are:

- `observations` — number of successful `observe` calls (or `observe` calls
  plus merges, in the CRDT).
- `novel_observations` — number of `observe` calls that changed state.
- `generation` — bumped on every state mutation, including `clear`.
- `memory_bytes` — an approximate accounting of engine-owned memory,
  recomputed from the live structure on `clear` and incrementally adjusted
  otherwise; it is a diagnostic, not a promise.

`validate` is an O(n) (or worse) internal consistency check intended for
diagnostics and tests. It returns `FUTCACHE_OK` or
`FUTCACHE_ERROR_CORRUPT_DATA`.

---

## 7. Engine 1: the interval-union cache

Header `futcache/futcache.h`, implementation `src/futcache.c`. This is the
reference engine: it is exact for the one-dimensional case on a closed domain
`[domain_min, domain_max]`, and it is the only engine with serialization.

### 7.1 Representation

An observation x induces the interval `[x - epsilon, x + epsilon]`, clipped to
the domain. The cache stores the union of all such intervals in canonical
form: sorted, disjoint, and non-mergeable. Adjacent or overlapping intervals
are merged eagerly, so the stored intervals never touch and never overlap.
Two closed intervals that share an endpoint are one interval; the test is
strict, `previous.upper < next.lower`, because closed intervals that meet are
a single connected region and must be represented once.

The intervals are held in a balanced AVL tree keyed by lower endpoint. The
tree supports O(log n) insertion and membership test. The root height is
exposed as `tree_height` in the stats and is bounded in practice by
`2 * ceil(log2(n + 1))`, a bound that the test suite checks after 10,000
sorted insertions.

### 7.2 Operations

`futcache_observe` computes novelty and, atomically with the novelty
decision, inserts the interval `[x - epsilon, x + epsilon]` even when x was
redundant. Inserting a redundant point's interval matters: the interval of a
redundant point is not necessarily contained in the existing union, and
omitting it would make the cache answer future queries incorrectly. This is
the behavior that makes the interval engine exact, and it is the behavior the
box engine does not replicate (see section 9).

The insertion is a standard AVL procedure followed by a union repair. The new
interval is located by lower endpoint; if it overlaps or touches a neighbor,
the two are merged and the merge is propagated until the union is again
disjoint. The repair runs under the write lock, so a reader never observes an
intermediate state. The allocation-before-commit rule (section 12.4) applies
to the repair: if a rebalancing step needs memory and the allocation fails,
the function returns `FUTCACHE_ERROR_OUT_OF_MEMORY` and the tree is
unchanged. The net effect is that `observe` is linearizable and
failure-atomic, and the stored union is always canonical.

`futcache_is_novel` is the non-mutating query. `futcache_copy_intervals`
exports the canonical union. `futcache_clear` empties the union and resets the
counters while advancing generation. `futcache_validate` walks the tree and
checks sorting, disjointness, non-mergeability, balance factors, node count,
coverage, and the telemetry relations.

### 7.3 Coverage

The cache maintains `covered_measure`, the total length of the stored union,
accumulated in `long double` to avoid the drift of summing many binary64
values. When the union covers the whole domain, `fully_covered` is set, and
every subsequent query and observation is redundant. The state then contains
a single interval and never grows again. This is the saturation behavior
described in section 3.

### 7.4 Serialization

`futcache_serialize` produces an atomic snapshot under the read lock;
`futcache_deserialize` restores it. The format is versioned, endian-independent
(little-endian), and CRC32-protected, and is documented in
`docs/serialization.md` and section 13. This is the only engine that persists
to bytes; the others are in-memory, and the CRDT's `snapshot` is a
zero-copy gossip structure rather than a disk format.

---

## 8. Engine 2: the Voronoi packing cache

Header `futcache/pack.h`, implementation `src/pack.c`. This engine
generalizes novelty to arbitrary finite-dimensional metric spaces, including
embedding spaces, where a closed-form canonical union does not exist.

### 8.1 Representation

The cache keeps a set R of representative points. A point x is redundant if
its distance to the nearest representative is at most epsilon; otherwise x is
novel and is added to R. This is a greedy farthest-point-style insertion: a
new representative is always at distance strictly greater than epsilon from
every existing representative, so R is epsilon-separated, and its size is
bounded by the packing number of the domain at scale epsilon.

The invariant follows directly from the insertion rule. A point is added only
when it is farther than epsilon from every current representative, so the new
representative is farther than epsilon from all of them, and separation is
preserved by induction. Conversely, a point is rejected only when it is
within epsilon of some representative, so every observed point is either a
representative or within epsilon of one. That second fact is the coverage
side of the trade: the representative set "covers" the history in the sense
that every history point lies inside some representative's ball, which is why
the cache can discard the history point itself.

### 8.2 One-sided error

Representatives are a subset of the observed history, so the ball union of R
is a subset of the true union U_e(H). A point inside a representative's ball
is therefore genuinely within epsilon of an observed point: the cache never
reports a false hit. The converse does not hold. A point can be within
epsilon of an observed non-representative point while being farther than
epsilon from every representative, in which case the cache reports a false
novelty. The packing cache is therefore one-sided and conservative: it may
over-report novelty, and it must not, and does not, suppress a genuinely
novel point. This is the correct error profile for a cache, because the worst
case of a false novelty is a redundant recomputation, whereas a false hit
would return stale data.

### 8.3 Distance functions

The config accepts a `futcache_distance_fn`. Four are provided:
`futcache_distance_l1`, `futcache_distance_l2`, `futcache_distance_linf`, and
`futcache_distance_cosine`. The default is L_inf. The cosine distance is
defined for vectors and is the natural choice for normalized embeddings,
which is how `examples/rag_embedding.c` and the Python binding use the engine.
Callers are responsible for normalization: the cosine distance assumes unit
vectors, and the examples normalize before observing.

### 8.4 Pluggable backend

The default nearest-neighbor search is a linear scan over R, which is O(|R|)
per observation. The config accepts an optional `futcache_pack_backend_ops_t`
that supplies an external nearest-neighbor index. The backend contract is
one-sided in the same direction: an approximate backend may return a
distance that is too small, which over-reports novelty, but it must never
return a distance that is too large for a point that is genuinely novel,
because that would suppress novelty. The backend does not expose per-index
identities, which is why `futcache_pack_nearest` (below) always uses the
linear representative array.

### 8.5 Nearest-site lookup

`futcache_pack_nearest` reports the distance to, and slot index of, the
nearest representative. On an empty cache it reports `+infinity` and
`SIZE_MAX`. The slot index matches the ordering used by
`futcache_pack_copy_representatives`. This function is what the Python
binding uses to populate a hit's `representative_id` and `distance`.

### 8.6 Validation

`futcache_pack_validate` checks the telemetry relations (`generation >=
observations`, `novel_observations <= observations`, `representative_count ==
novel_observations`, `representative_count <= peak_count`) and the
epsilon-separation invariant (every pair of representatives is strictly
farther than epsilon apart). The separation check is O(|R|^2), which is why
it is a diagnostic rather than a per-operation check.

---

## 9. Engine 3: the box cache

Header `futcache/box.h`, implementation `src/box.c`. This engine targets
exact L_inf novelty in a bounded domain of dimension 1 through 8.

### 9.1 Representation

Each novel observation contributes its clipped axis-aligned box
`[p_i - epsilon, p_i + epsilon]` in every coordinate. Queries test membership
in the union of stored boxes. Because the boxes all have the same half-width
epsilon, one box strictly contains another only if the two centers are
identical; distinct novel centers therefore produce boxes that may partially
overlap but never strictly contain one another. The test suite and the
example program assert this invariant by construction.

### 9.2 Important: the current implementation is one-sided, not exact

The header and the README describe the box cache as exact. That description
is inaccurate for the current implementation, and the reader should treat the
engine as one-sided until the code or the claim is changed.

The reason is a one-line asymmetry with the interval engine. The interval
engine inserts the ball of every observation, including redundant ones. The
box engine does not: `futcache_box_observe` returns early for a redundant
point and adds no box. A redundant point's box is not necessarily contained
in the existing union — two points at distance exactly epsilon have boxes
that overlap only along a boundary — so omitting it shrinks the stored union.

The consequence is a false novelty. In one dimension with epsilon 0.1:

- observe 0.5: novel, box `[0.4, 0.6]`.
- observe 0.6: redundant (inside `[0.4, 0.6]`), no box added.
- query 0.7: the full history contains 0.6, and `|0.7 - 0.6| = 0.1 <= 0.1`,
  so 0.7 is redundant. The stored union `[0.4, 0.6]` does not contain 0.7,
  so the box cache reports novel.

The error profile is therefore the same as the packing cache: never a false
hit, possibly a false miss. The fix, if exactness is required, is to union in
every observation's box (at the cost of memory proportional to observations
rather than novel observations) or to canonicalize the union (expensive for
axis-aligned boxes in more than one dimension). If exactness is not required,
the header and README should be corrected to describe the engine as a
one-sided box packing.

`box_count` is equal to `novel_observations`, not `observations`, which is
exactly the symptom of the omission described above.

### 9.3 Validation

`futcache_box_validate` checks the telemetry relations
(`generation >= observations`, `novel_observations <= observations`,
`box_count == novel_observations`, `box_count <= peak_box_count`). It does
not re-derive the containment invariant (that property holds by construction
of equal-half-width boxes). The engine has no serializer.

---

## 10. Engine 4: the dyadic tower

Header `futcache/tower.h`, implementation `src/tower.c`. This is a
one-dimensional multi-resolution structure used for coarse-to-fine novelty
testing, prefix counts, and first-discovery logs.

### 10.1 Representation

The tower is a stack of levels. Level 0 is the coarsest and has
`root_cells` equal-width cells; each finer level doubles the cell count. The
config specifies the domain, the number of levels, and the root cell count.
The default is two levels over `[0, 1]` with 2 then 4 cells.

Each level owns a bitset of cell occupancy and a Fenwick tree over the
occupancy for prefix counts. Observing a point marks its cell occupied at
every level and appends the cell to a per-level first-discovery log. The
Fenwick tree supports O(log cells) prefix count and O(log cells) select of
the k-th occupied cell in spatial order; the discovery log supports O(1)
lookup of the j-th discovery in time order.

### 10.2 Coarse-to-fine implication

A point that is novel at a coarse level is necessarily novel at every finer
level, because a coarse cell is a superset of the finer cells inside it. The
converse does not hold: a point can be redundant at a coarse level and novel
at a fine level. The tower therefore answers novelty as a per-level byte
vector, and a caller that wants an early-out test checks the coarse level
first. The implication is checked element by element by a focused two-level
test that verifies the discovery-word identity.

### 10.3 Operations

`futcache_tower_query` writes one byte per level (1 for unseen, 0 otherwise).
`futcache_tower_observe` is the mutating form. The accessors are
`futcache_tower_cell_index` (the cell containing x at a level),
`futcache_tower_level_info` (cell count and discovered count),
`futcache_tower_prefix_count` (occupied cells up to a spatial index),
`futcache_tower_select_occupied` (ordinal to spatial cell), and
`futcache_tower_discovery_at` (time-ordered discovery). `futcache_tower_clear`
and `futcache_tower_validate` round out the surface.

The tower does not take an epsilon. Its resolution is fixed by the level
count and domain width, which makes it a discretization rather than a metric
cache. It is useful as a cheap prefilter in front of a metric engine.

---

## 11. Engine 5: the CRDT cache

Header `futcache/crdt.h`, implementation `src/crdt.c`. This engine turns the
novelty cache into a convergent replicated object: multiple replicas gossip
state and reach agreement without coordination. It implements the design of
`PHASE2.md` section 12.

### 11.1 Representation

The domain is partitioned by a fixed set of anchors into Voronoi cells. The
cache is an array of `anchor_count` cells; each cell is empty or holds one
entry: a representative point, a payload, and a `uint64_t` priority. The
anchors are copied at create time and never change, which is what makes the
state safe to merge.

A point is mapped to its cell by `futcache_crdt_quantize`, a linear scan over
the anchors under the configured distance function. The epsilon field is
stored for bookkeeping and validation only; quantization uses the anchors
alone. The documented contract is that the anchors form a delta-net with
delta at most epsilon/2, so that any two points in one cell are within
2 * delta at most epsilon of each other. The engine does not verify that the
supplied anchors satisfy the contract; it is the caller's responsibility.

### 11.2 Priority and merge

The priority is an FNV-1a 64-bit hash over the point coordinates (canonical
little-endian bit patterns) followed by the payload bytes. It is a
deterministic total order over entries. Merge adopts an update into an empty
cell; into an occupied cell it keeps the higher-priority entry, and on an
exact priority tie it keeps the local entry. Ties are deterministic because
an equal priority implies identical bytes, in which case either choice is the
same state.

`futcache_crdt_merge` validates the entire batch before mutating anything, so
a malformed update cannot leave the cache half-merged. The merge law is
idempotent, commutative, and associative, which is the join-semilattice
property that makes replicas converge under any delivery schedule.

The priority is a hash, not a wall-clock timestamp, for a deliberate reason:
convergence must not depend on any replica's clock. A timestamp tie or a
skewed clock would break the total order the join requires. A
content-addressed hash is a pure function of the data, so any two replicas
computing a priority for the same bytes compute the same number, with no
clock to disagree.

### 11.3 Snapshot and lifetime

`futcache_crdt_snapshot` returns occupied cells as an array of
`futcache_crdt_update_t`. The `point` and `payload` pointers alias
cache-owned storage and remain valid only until the next mutation (observe,
merge, clear, or destroy). This is a zero-copy design for gossip: a replica
can snapshot, transmit, and then re-snapshot after any merge that may have
mutated its own storage. `examples/crdt.c` documents and exercises this
lifetime hazard, including the re-snapshot-after-merge workaround.

### 11.4 Telemetry

The CRDT stats track local `observe` calls (`observations`), local observes
that filled an empty cell (`novel_observations`), and the number of occupied
cells (`occupied_cells`). After merges, `occupied_cells` may exceed
`novel_observations`, because remote updates can fill cells the local replica
never observed. `generation` is bumped on observe, merge, and clear.

---

## 12. Numerical correctness

The interval engine makes exact novelty decisions with binary64 arithmetic.
The hazards are outward rounding and excess precision, and the engine handles
both explicitly.

### 12.1 Outward rounding

Computing the endpoint of an interval as `x + epsilon` or `x - epsilon` in
binary64 can round outward: the stored interval may be slightly larger than
the true ball. An outward-rounded interval can absorb a point that is in fact
just outside the true ball, producing a false hit, which is the one error a
novelty cache must never make.

The engine computes each endpoint by tracking the residual of the
floating-point addition. For an upper endpoint it computes the rounded sum and
then recovers the error term of that addition (the TwoSum residual), which
measures how far the rounded sum drifted from the exact value. It then nudges
the stored endpoint one `nextafter` step toward the interior — for an upper
endpoint that is one step downward, for a lower endpoint one step upward. The
stored interval is therefore a subset of the true ball.

The direction of the nudge is the point of the exercise. A false hit
(reporting redundant a point that is outside the true ball) is the error that
must never occur, because it returns stale data as if it were fresh. A false
miss (reporting novel a point that is inside the true ball) is safe, because
the caller recomputes. Rounding inward trades a possible false miss at the
boundary for the elimination of false hits: a point reported redundant is
genuinely within the mathematical ball, and a point reported novel may sit
exactly at the boundary.

### 12.2 Excess precision

On platforms where the compiler evaluates floating-point expressions in
extended precision and spills to binary64 later (`FLT_EVAL_METHOD != 0`), the
TwoSum residual is not reliable. The engine gates on
`FLT_EVAL_METHOD == 0` and a rounding mode of `FE_TONEAREST`; if either does
not hold, `futcache_create` returns `FUTCACHE_ERROR_UNSUPPORTED_PLATFORM`.
This is the "unsupported platform" status described in section 6.1.

### 12.3 Coverage accumulation

The covered measure is accumulated in `long double` and only the final value
is exposed, avoiding the accumulation drift that a running binary64 sum would
introduce over many merged intervals.

### 12.4 Allocation-before-commit atomicity

Every mutating operation that can fail allocates everything it needs before
freeing or overwriting existing state. If an allocation fails, the function
returns `FUTCACHE_ERROR_OUT_OF_MEMORY` with the cache unchanged. This is
verified with counting allocators that fail at each allocation site and
assert that pre-operation state, counters, and locks survive the failure.

---

## 13. Serialization (interval engine)

`docs/serialization.md` specifies the version-1 format. The layout is a
little-endian sequence: an 8-byte ASCII magic `FUTCACHE`, a 2-byte format
version, a 2-byte header size (72), a 4-byte flags word (zero in version 1),
the parameters (epsilon, domain min, domain max), the telemetry
(observations, novel observations, generation), the interval count, the
`(lower, upper)` pairs as IEEE-754 binary64 bit patterns, and a trailing
CRC32 over every preceding byte.

Intervals must be finite, in-domain, sorted, and strictly disjoint. The
deserializer rejects unknown versions and flags, trailing bytes, NaN
parameters, impossible telemetry, and touching intervals. The checksum and
the no-trailing-bytes rule make the format safe to transport without an outer
length prefix, because any truncation or single-byte corruption fails either
the length check or the checksum.

Each rejection rule maps to a concrete corruption vector. The version and
flags checks reject a snapshot produced by a future or incompatible build.
The no-trailing-bytes rule rejects a snapshot concatenated with stray data
and, together with the length check, every truncation. The telemetry rules
reject a snapshot whose counters were edited to an impossible state, for
example a novel count larger than the observation count. The interval rules
reject a snapshot whose union was edited to overlap or to escape the domain.
The checksum is the last line of defense for anything the structural rules do
not catch.

The serializer snapshots under the read lock, so the bytes are a consistent
state even when writers are active. This is the only persistence surface in
the library.

---

## 14. Telemetry and invariants

Each engine maintains counters and checks them in `validate`. The relations
are engine-specific but share a shape.

- Interval: `generation >= observations`, `novel_observations <=
  observations`, `interval_count <= novel_observations`.
- Pack: `generation >= observations`, `novel_observations <= observations`,
  `representative_count == novel_observations`, `representative_count <=
  peak_count`, plus pairwise epsilon-separation.
- Box: `generation >= observations`, `novel_observations <= observations`,
  `box_count == novel_observations`, `box_count <= peak_box_count`.
- Tower: occupancy, prefix sums, and discovery logs are checked against
  independent recomputation.
- CRDT: `generation >= observations`, `novel_observations <= observations`,
  `occupied_cells <= anchor_count`, `occupied_cells >= novel_observations`,
  plus per-cell payload/point sanity.

The `generation >= observations` relation is subtle: generation must be
bumped on every observation, including redundant ones, or the relation fails
as soon as a redundant observation arrives. This exact bug existed in the box
engine, where the redundant path bumped `observations` but not `generation`;
it was fixed so the redundant path bumps both. The invariant check is what
exposed it.

---

## 15. Python bindings

The `futcache` package is a thin wrapper over the packing engine, built with
nanobind and scikit-build-core. It exists to serve the semantic-cache use
case: the C layer owns novelty decisions, and a Python dict owns payloads
keyed by representative slot index.

The main class is `PackCache(dimension, epsilon, distance, domain_min,
domain_max)`, where `distance` is one of `"linf"` (default), `"l1"`, `"l2"`,
or `"cosine"`, and the domain bounds are scalars broadcast to every
dimension. Points are `numpy.float64` one-dimensional arrays.

`observe(point, payload=None)` and `query(point)` return a `NoveltyResult`
with four fields:

- `representative_id` — the slot index of the matched or new representative,
  or `-1` when the point is novel (there is no representative to name).
- `is_novel` — whether the point is farther than epsilon from every existing
  representative.
- `distance` — distance to the nearest representative; `0.0` for a novel
  observation and at most epsilon for a hit.
- `inserted` — true when `observe` added a new representative.

The canonical pattern, demonstrated by
`python/examples/rag_semantic_cache.py`, is: observe the query embedding with
a placeholder payload; if novel, call the model and `set_payload` the result;
if not novel, `get_payload` the cached result. The slot index from a hit is
exactly the key to `get_payload`.

The class also exposes `copy_representatives()` (an `(N, dimension)` array),
`clear()`, `len()`, `peak_count()`, `memory_bytes()`, `observations()`,
`novel_observations()`, and a class method `PackCache.version()` returning
`"1.1.0"`.

---

## 16. Testing and verification

The test suite is the primary specification of behavior. It runs as a single
executable with one line per test, 49 tests across five suites at the time of
writing: 18 for the interval engine, 12 for packing, 7 for the tower, 5 for
the box cache, and 7 for the CRDT.

The verification strategy, recorded in `docs/verification.md`, is:

- **Differential testing** against a full-history oracle: novelty decisions
  must equal an independent recomputation that scans the whole stream.
- **Directed-rounding oracles**: thousands of independently computed balls
  verify the inward-rounded endpoints.
- **Fault injection**: counting allocators that fail at each allocation site
  verify allocation-before-commit atomicity and leak-freedom.
- **Concurrency**: multithreaded observers, readers, and serializers under
  ThreadSanitizer.
- **Adversarial snapshots**: every one-byte mutation and truncation of a
  serialized snapshot must be rejected.
- **Static analysis**: `-fanalyzer` and the strongest project warnings with
  `-Werror`.
- **Sanitizers**: ASan and UBSan on a debug build.

The differential streams are not toy sizes. The interval engine is checked
against a full-history oracle on 3,000 observations over a 4,096-point dyadic
grid, and against an independently computed canonical union on 2,048
observations in both forward and reverse order. The tower is checked over
4,000 observations across six levels. The packing engine is checked over
5,000-point streams in dimensions 2, 4, 8, and 16 under L_inf, L1, and L2,
plus a 384-dimensional cosine embedding stream. The randomized streams use
fixed seeds that are printed in the source, so a failing case can be replayed
exactly.

The gates are: a `-Werror` debug build, a `-Werror` release build, ASan +
UBSan, TSan, and `-fanalyzer`, all clean. The verification document records
these results so a regression is visible.

---

## 17. Choosing an engine

The decision is governed by dimensionality, exactness requirements, and
distribution.

- **Exact one-dimensional novelty, or persistence needed** — use the
  interval engine (`futcache.h`). It is the only exact engine and the only
  one that serializes.
- **Exact multi-dimensional L_inf novelty, dimensions 1–8** — the box
  engine (`box.h`) is the intended tool, but see the one-sided caveat in
  section 9.2. Until the code or the docs are fixed, treat it as conservative.
- **Arbitrary-dimension embeddings, approximate novelty, payload caching** —
  use the packing engine (`pack.h`) with the cosine or L2 distance. This is
  the semantic-cache workhorse and the engine behind the Python binding.
- **One-dimensional coarse prefilter, prefix counts, discovery order** — use
  the tower (`tower.h`).
- **Multiple replicas that must converge without coordination** — use the
  CRDT engine (`crdt.h`).

A concrete walkthrough makes the decision mechanical. A support chatbot
embeds each user query to 384 dimensions and wants to skip the model when a
near-duplicate has already been answered: packing engine, cosine distance,
epsilon tuned from the cacheability analysis in `scripts/cacheability.py`.
A telemetry pipeline deduplicates scalar sensor readings where an exact
one-dimensional answer and a restorable snapshot are required: interval
engine. A game server deduplicates 2-D spatial events with exact L_inf
semantics: box engine, with the section 9.2 caveat. An edge fleet serves
semantic cache replicas that must agree without a coordinator: CRDT engine
with a fixed anchor net dense enough for the chosen epsilon.

The non-goals are worth stating: none of the engines implements TTL or
sliding windows, none implements LRU or other bounded-memory eviction, and
none trains or adapts a model. State size is bounded by domain geometry
(packing number or cell count), not by an eviction policy. Applications that
need recency or bounded memory must layer those on top.

---

## 18. Performance notes

The engines differ in per-operation cost.

- Interval: O(log n) per observe and query (AVL), O(n) copy and validate,
  O(1) coverage. This is the fastest exact engine.
- Pack: O(|R|) per observe and query with the linear scan; a backend index
  can lower this. Copy is O(|R| * dimension); validate is O(|R|^2).
- Box: O(box_count * dimension) per observe and query (a scan over stored
  boxes); memory is O(box_count * dimension).
- Tower: O(levels) per observe and query; prefix count and select are
  O(log cells) via Fenwick; discovery lookup is O(1).
- CRDT: O(anchor_count * dimension) per observe (quantization scan); merge is
  O(update_count); snapshot is O(anchor_count).

The benchmark `bench/cache_comparison.c` measures throughput, memory, and
decision error against an LRU and an exact-set baseline across five
workloads. The scaling example `examples/reciprocal_scaling.c` demonstrates
the sub-linear state growth that the epsilon-ball union buys on a dense
stream.

The headline result of the comparison benchmark is that on continuous
one-dimensional workloads the interval engine achieves zero decision error
against the metric oracle at its target epsilon while holding a small
fraction of the stream in memory; the LRU and exact-set baselines match the
oracle only until their capacity is exhausted, after which their error rises.
The benchmark emits a Pareto frontier of peak memory against decision error
per workload, so the trade is visible rather than asserted.

---

## 19. Known limitations

- The box engine's "exact" documentation does not match its one-sided
  implementation (section 9.2). This is the most important open item.
- The CRDT engine does not verify the caller-supplied delta-net contract; a
  caller that supplies anchors too sparse for epsilon silently loses the
  epsilon guarantee.
- Serialization exists only for the interval engine.
- The Python binding exposes only the packing engine; the interval, box,
  tower, and CRDT engines are C-only.
- LeakSanitizer cannot run in the authoring environment (a ptrace limitation
  of the managed container); leak-freedom is instead established with
  counting allocators. Valgrind is not installed.
- ThreadSanitizer runs under a documented non-PIE, ASLR-disabled workaround.
- Clang is not installed in the authoring environment, so the compiler matrix
  covers GCC only.

---

## 20. Glossary

- **Metric novelty** — the predicate "x is not within epsilon of any
  previously observed point".
- **Ball union** — U_e(H), the union of closed epsilon-balls around every
  observed point.
- **Future-equivalence** — two histories are future-equivalent when they
  induce the same ball union and therefore answer every future query
  identically.
- **Representative** — a point the packing cache retained as a proxy for the
  observations near it.
- **One-sided** — an error profile that may over-report novelty but never
  under-reports it.
- **Packing number** — the maximum number of pairwise epsilon-separated
  points a bounded domain can contain; it bounds representative-set size.
- **Canonical union** — the sorted, disjoint, non-mergeable interval
  representation of a one-dimensional ball union.
- **delta-net** — a set of anchors whose maximum covering radius (the
  farthest any point is from its nearest anchor) is at most delta.
- **Join-semilattice** — a set with an associative, commutative, idempotent
  merge, which is the property that makes CRDT replicas converge.
- **Fenwick tree** — a binary-indexed tree supporting prefix sums and k-th
  selection in logarithmic time.

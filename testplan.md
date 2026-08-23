You are acting as a hostile systems verifier for **FUTCache**, a C implementation of future-equivalence caching / exact metric-novelty state compression.

Your job is **not** to make the library look good.

Your job is to break it.

Assume the implementation contains subtle bugs in interval canonicalization, AVL balancing, memory ownership, snapshot serialization, tower refinement, concurrency, boundary handling, and allocation-failure recovery until you have strong evidence otherwise.

Do not stop when the existing tests pass.

Do not merely run the repository test suite.

Inspect the implementation, derive the invariants from the code and mathematics, construct independent reference models, generate adversarial histories, fuzz the public API, force rare failure paths, and report every discrepancy with a minimal reproducer.

The core correctness principle is:

[
H\sim_{\mathrm{fut}}H'
\iff
\forall w,\quad Out(H;w)=Out(H';w).
]

For exact metric novelty at radius (\epsilon),

[
U_\epsilon(H)
=============

\bigcup_{x\in V(H)}
\overline B(x,\epsilon),
]

and

[
Nov_\epsilon(y\mid H)
=====================

\mathbf 1[y\notin U_\epsilon(H)].
]

Therefore two histories inducing the same canonical coverage set must be behaviorally indistinguishable for every future novelty query.

The implementation is expected to maintain the **canonical union of covered intervals**, rather than merely remembering historical points.

The multiresolution tower uses resolutions such as

[
\epsilon_j=2^{-j},
]

with finer levels projecting consistently onto coarser levels.

The intended systems inversion is:

[
\boxed{\text{LRU: choose memory, lose information accordingly.}}
]

[
\boxed{\text{FUTCache: choose information resolution, memory follows geometrically.}}
]

Your validation should determine whether the implementation actually satisfies that semantics.

# 1. First inspect the repository

Before changing code:

1. Inspect all public headers.
2. Inspect all implementation files.
3. Inspect current tests.
4. Inspect CMake options.
5. Identify:

   * public API;
   * cache object ownership model;
   * interval representation;
   * AVL invariants;
   * locking strategy;
   * tower representation;
   * snapshot format;
   * allocation abstraction, if any;
   * invariant/debug validation functions;
   * numerical type used for coordinates and epsilon;
   * documented thread-safety guarantees.

Do not invent API names.

Use the actual API exposed by the repository.

Produce a short architecture map before testing.

# 2. Establish clean builds

Build from clean directories.

At minimum test:

```bash
cmake -S . -B build-debug \
  -DFUTCACHE_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build-debug --parallel
```

If tests are separately controlled, enable them using the actual repository option.

Also build optimized:

```bash
cmake -S . -B build-release \
  -DFUTCACHE_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-release --parallel
```

Compile with the strongest warnings reasonably supported by GCC/Clang:

```text
-Wall
-Wextra
-Wpedantic
-Wconversion
-Wshadow
-Wformat=2
-Wundef
-Wcast-align
-Wstrict-prototypes
-Wmissing-prototypes
-Wdouble-promotion
-Wnull-dereference
-Werror=return-type
```

Do not blindly force a warning flag unsupported by the compiler. Record any warnings instead of suppressing them.

Test both GCC and Clang if available.

# 3. Sanitizer matrix

Create sanitizer builds.

At minimum:

### ASan + UBSan

```text
-fsanitize=address,undefined
-fno-omit-frame-pointer
```

Run the entire deterministic suite and fuzz/property suite.

### ThreadSanitizer

```text
-fsanitize=thread
```

Run all concurrency tests separately under TSan.

Do not combine TSan with ASan.

### Leak detection

Use ASan leak detection or Valgrind if available.

Every create/destroy, failed insertion, failed snapshot restore, partially initialized tower, and deliberately induced allocation failure must be checked for leaks.

Any sanitizer finding is a release blocker.

# 4. Build an independent mathematical oracle

Do NOT test FUTCache against FUTCache.

Implement a tiny, obviously-correct reference model independently of the production data structure.

For 1D exact metric novelty, represent history naively as a vector of inserted points.

Reference query:

[
Nov_\epsilon(y\mid H)
=====================

\mathbf1[
\forall x\in H,\ |y-x|>\epsilon
].
]

Reference coverage:

[
U_\epsilon(H)
=============

\bigcup_{x\in H}
[x-\epsilon,x+\epsilon]
]

clipped to the configured domain if the production API uses a bounded domain.

For every operation sequence compare:

* novelty result;
* membership / covered result;
* canonical interval union;
* interval count;
* total covered measure if exposed;
* snapshot round-trip behavior;
* tower query results;
* all documented statistics.

The oracle should favor simplicity over speed.

# 5. Deterministic interval-union adversarial suite

Construct explicit tests for all canonicalization cases.

For intervals (A) and (B), cover:

* disjoint, (A<B);
* disjoint, (B<A);
* overlap from left;
* overlap from right;
* one fully contains another;
* exact duplicate;
* same left endpoint;
* same right endpoint;
* touching endpoints;
* epsilon-distance exactly equal to the novelty threshold;
* bridging insertion joining two components;
* one insertion joining three or more components;
* repeated bridge insertion;
* insertion inside existing coverage;
* insertion exactly on coverage boundary;
* insertion just inside boundary;
* insertion just outside boundary.

Example bridge pattern:

[
[0.0,0.2]\cup[0.4,0.6]
]

followed by an insertion whose induced interval causes the canonical state to become

[
[0.0,0.6].
]

Verify that obsolete AVL nodes disappear and the canonical representation contains exactly one component.

Test long merge cascades, e.g. thousands of initially disjoint components joined by one strategic insertion.

# 6. Boundary semantics

This is critical.

The mathematical predicate currently uses:

[
Nov_\epsilon(y\mid H)
=====================

\mathbf1[d(y,V(H))>\epsilon].
]

Therefore distance exactly equal to (\epsilon) is **not novel**.

Verify the implementation agrees.

Explicitly test:

[
d=\epsilon,
]

[
d=\epsilon-\delta,
]

[
d=\epsilon+\delta,
]

for tiny representable (\delta).

Test:

* (0);
* (1), if domain is ([0,1]);
* negative coordinates if API permits them;
* coordinates greater than one if API permits them;
* (\epsilon=0);
* smallest accepted epsilon;
* very large epsilon;
* epsilon covering entire domain.

If floating-point coordinates are used, test `nextafter()` on both sides of every important boundary.

Do not use approximate assertions where exact API semantics require exact decisions.

# 7. IEEE-754 hostile inputs

Determine the documented contract first.

Then test according to that contract:

* `+0.0`;
* `-0.0`;
* `DBL_MIN`;
* subnormals;
* `DBL_MAX`;
* values near overflow when adding/subtracting epsilon;
* `INFINITY`;
* `-INFINITY`;
* quiet NaN;
* multiple NaN payloads if practical.

If invalid values are prohibited, verify deterministic rejection without corruption.

If infinities are supported, verify semantics explicitly.

No undefined behavior, hangs, tree corruption, or silent canonicalization failure is acceptable.

# 8. Algebraic / metamorphic properties

Test properties that should hold independently of implementation details.

## Idempotence

Inserting the same point twice must not change future behavior after the first insertion.

[
U_\epsilon(Hxx)=U_\epsilon(Hx).
]

## Permutation invariance of final coverage

For any finite multiset of inserted points,

[
U_\epsilon(x_{\pi(1)},\ldots,x_{\pi(n)})
========================================

U_\epsilon(x_1,\ldots,x_n)
]

for every permutation (\pi).

Intermediate novelty outputs can differ because arrival order differs, but the final coverage state must be identical.

## Monotonicity

Without expiry/deletion,

[
U_\epsilon(H_t)
\subseteq
U_\epsilon(H_{t+1}).
]

Coverage must never shrink.

## Absorption

Once the entire domain is covered,

[
U_\epsilon(H)=K,
]

every future query must return non-novel and subsequent insertions must not grow structural state unnecessarily.

## Translation invariance

Where the domain/API permits:

[
H' = H+c,\qquad y'=y+c
]

must preserve novelty.

## Scale relation

Where representable and domain-independent:

[
H' = aH,\quad
y'=ay,\quad
\epsilon'=|a|\epsilon
]

must preserve novelty for (a\neq0).

## History quotient invariance

Generate distinct histories (H,H') with identical canonical coverage:

[
U_\epsilon(H)=U_\epsilon(H').
]

Then append the same randomly generated continuation (w).

The complete sequence of future novelty outputs must match:

[
Out(H;w)=Out(H';w).
]

This is the single most important semantic test in the project.

# 9. Explicit future-equivalence attack

Construct many pairs of histories that look very different but induce identical coverage.

Example families:

* one point versus many redundant nearby points;
* increasing sequence versus shuffled sequence;
* dense cluster versus its extremal sufficient points;
* histories containing thousands of redundant points;
* histories produced via different merge orders.

After verifying canonical equality, feed each pair a long adversarial continuation.

Require exact equality of every future observable.

If any pair with equal canonical coverage produces different future outputs, FUTCache violates its defining theorem.

# 10. Compare against representative packing

Implement a separate naive epsilon-packing cache and deliberately find histories where packing-cache novelty differs from full-history metric novelty.

Example structure:

```text
representative history sees A
redundant point B is skipped as representative
later point C is within epsilon of B
but farther than epsilon from retained representatives
```

Confirm that FUTCache agrees with the full-history oracle, not merely with the packing representative set.

This guards against accidentally implementing an approximate packing cache rather than exact future-equivalence state.

# 11. Random differential testing

Generate millions of operation sequences if runtime permits.

Vary:

* history length;
* epsilon;
* coordinate distribution;
* repeated-point probability;
* clustering;
* sorted data;
* reverse-sorted data;
* alternating extremes;
* pathological merge cascades.

Use several generators:

### Uniform

[
x\sim U(K).
]

### Gaussian clusters

Several tight semantic clusters.

### Near-boundary

Most samples within a few ULPs of existing interval boundaries.

### Duplicate-heavy

90–99.9% repeated or redundant insertions.

### Adversarial alternating

Alternate leftmost and rightmost uncovered regions.

### Merge storm

Create many components, then repeatedly bridge them.

### Reciprocal

[
x_n=\frac1n.
]

### Dyadic

[
x=\frac{k}{2^j}.
]

### Cantor-like samples

Useful for nontrivial geometric occupancy.

Seed every random test and print the seed on failure.

On mismatch, automatically delta-debug / shrink the operation sequence to a minimal reproducer.

# 12. AVL tree structural verification

After every mutation in debug/property tests, validate:

* strict key ordering;
* parent/child consistency if parents exist;
* stored heights;
* balance factor in allowed range;
* no cycles;
* no duplicate canonical intervals;
* no overlapping canonical intervals;
* no mergeable adjacent intervals left separate;
* node count matches reported interval count;
* root metadata is valid;
* empty-state invariants;
* all heap objects reachable exactly once.

Stress sorted insertion orders that normally destroy unbalanced BSTs.

Measure height after large runs and verify AVL height remains logarithmic.

For (n) nodes, detect any suspicious degeneration toward (O(n)).

# 13. Allocation-failure atomicity

This is mandatory.

If the implementation has allocation-failure injection, use it.

Otherwise, add a test-only allocator shim if feasible without changing release semantics.

For every operation that may allocate, fail allocation at:

* first allocation;
* second allocation;
* ...
* every allocation site reachable by that operation.

After failure verify:

1. function returns documented error;
2. cache remains internally valid;
3. pre-operation state is preserved unless API explicitly documents another atomicity contract;
4. no locks remain held;
5. no memory leaks occur;
6. subsequent valid operations still succeed.

Pay special attention to an insertion that requires:

* new node allocation;
* deleting/relinking multiple merged nodes;
* AVL rotations;
* tower updates across multiple levels.

The operation must not leave half-committed semantic state.

# 14. Snapshot / serialization attacks

For every supported snapshot format test:

## Round trip

[
S
\xrightarrow{\text{serialize}}
B
\xrightarrow{\text{deserialize}}
S'
]

Require semantic equality and canonical equality.

## Determinism

If format promises deterministic snapshots, identical logical state built through different histories must serialize to identical bytes.

## Corruption

Mutate:

* magic;
* version;
* length;
* checksum;
* interval count;
* node count;
* epsilon;
* level count;
* payload bytes;
* trailing bytes;
* truncation at every byte offset.

Deserializer must reject malformed data safely.

Never:

* overread;
* allocate absurd memory from untrusted length fields;
* integer-overflow size calculations;
* create partially valid state;
* leak on failure.

## Fuzz deserializer

Use libFuzzer/AFL++ if practical.

The snapshot parser should be considered an untrusted-input boundary.

# 15. Tower correctness

For every tower level (j), derive the expected level semantics independently.

Verify:

[
\epsilon_j=2^{-j}
]

or the actual repository's configured scale.

For every inserted point verify novelty independently at every level.

Test the fundamental monotonic relation:

If a point is novel at a **coarser** radius, determine the mathematically required implications for finer radii and verify them.

Be careful about direction:

larger epsilon means stricter novelty because more prior points fall inside the covered radius.

Verify tower results against independent per-level naive caches.

Do not test one tower level against another production tower level.

# 16. Tower projection / compatibility

If the implementation exposes discovery words, cells, occupancy, or projection operations, test:

[
q_{j+1,j}(D_{j+1}(L))=D_j(L)
]

according to the repository's exact parent/dedup definition.

Generate random traversals and test this identity at every adjacent level.

Also test multi-level composition:

[
q_{k,j}
=

q_{j+1,j}
\circ
q_{j+2,j+1}
\circ\cdots\circ
q_{k,k-1}.
]

Direct coarse projection and repeated adjacent projection must agree.

# 17. Reciprocal scaling regression

Use

[
L_N=
\left(
1,\frac12,\frac13,\ldots,\frac1N
\right).
]

At dyadic resolutions (2^{-j}), measure discovered-state complexity.

The expected law from the theory is

[
M_j=\Theta(2^{j/2})
]

for the appropriate dyadic-cell discovery construction.

Estimate

[
\widehat D_{\mathrm{cache}}
===========================

\frac{\log M_j}
{j\log2}
]

over increasing (j).

It should trend toward

[
\frac12
]

within finite-size effects.

Do not turn this into a pass/fail assertion with an unrealistically tight constant; use it as a regression / theory-validation benchmark.

Print the observed table:

```text
j
epsilon
M_j
M_{j+1}/M_j
D_hat
```

Expected asymptotic multiplier:

[
2^{1/2}=\sqrt2.
]

# 18. Dense geometric scaling

Generate dense traversals over:

[
[0,1],
]

and if the implementation supports higher-dimensional metric spaces, also:

[
[0,1]^d.
]

For sufficiently dense coverage, verify empirical growth consistent with:

[
M_j=2^{jD+o(j)}.
]

For 1D dense data:

[
D_{\mathrm{cache}}\approx1.
]

If only the 1D interval implementation exists, do not fabricate higher-dimensional support.

# 19. Concurrency torture

Read the documented thread-safety contract and attack exactly that.

Create many threads performing legal concurrent combinations of:

* query/query;
* query/insert;
* insert/insert;
* snapshot/query;
* snapshot/insert;
* tower queries;
* tower updates;
* create/destroy only if explicitly legal.

Use synchronized barriers to force races.

Construct cases where multiple threads simultaneously:

* insert the exact same point;
* insert overlapping intervals;
* bridge the same two components;
* bridge different components;
* query a boundary while another thread merges it;
* trigger AVL rotations;
* snapshot during sustained writes.

Run under ThreadSanitizer.

Look for:

* data races;
* deadlocks;
* lock-order inversion;
* livelock;
* missed wakeups;
* double free;
* use-after-free;
* inconsistent snapshots;
* non-linearizable novelty results if linearizability is promised.

# 20. Linearizability testing

If operations are intended to be linearizable, record concurrent histories with invocation/response timestamps.

For small histories, brute-force all legal sequential linearizations respecting happens-before order.

Verify at least one sequential execution reproduces the observed results.

Focus on 2–4 threads and very short histories; exhaustive search is feasible there.

Particularly test:

```text
T1: query(x)
T2: insert(x)
```

and simultaneous insertions whose intervals merge.

If the contract is weaker than linearizability, state the actual guarantee and test that instead.

# 21. Lock-failure and cleanup behavior

Inspect every lock/unlock path.

Force or simulate error paths where possible.

Verify:

* every successful lock is eventually unlocked;
* unlock failures are propagated according to contract;
* primary failure is not accidentally overwritten by cleanup success;
* cleanup failure does not hide a more important earlier error unless API specifies that behavior;
* no early return leaks a lock.

Review code paths manually in addition to dynamic tests.

# 22. Destruction under stress

Repeatedly:

1. create;
2. perform random operations;
3. snapshot;
4. restore;
5. query;
6. destroy.

Do this millions of times in smaller configurations.

Also test:

* destroy empty cache;
* destroy after failed operation;
* destroy fully-covered cache;
* destroy huge fragmented cache;
* destroy restored snapshot;
* destroy partially constructed object after injected failure if supported.

No leak, crash, or invalid free.

# 23. Integer overflow audit

Audit all expressions involving:

* number of levels;
* expected cell counts;
* `size_t`;
* multiplication by 2;
* serialized lengths;
* allocation sizes;
* node counts;
* bitset sizes;
* Fenwick indexes if present.

For code equivalent to:

```c
expected_cells *= 2U;
```

test near the type's maximum and verify overflow is rejected before it occurs.

Use UBSan where useful, but do not rely on sanitizers alone.

Manually inspect for wraparound.

# 24. Large-state stress

Push the implementation until memory/time becomes meaningful.

Suggested scales, adapting to machine resources:

```text
10^5 inserts
10^6 inserts
10^7 queries
```

Use workloads with:

* almost no merges;
* almost all merges;
* one huge interval;
* maximum fragmentation;
* duplicate storm.

Record:

* insertion throughput;
* query throughput;
* memory use;
* number of canonical intervals;
* tree height;
* lock contention;
* snapshot size;
* snapshot time.

Check for accidental (O(n)) operations where (O(\log n)) is intended.

# 25. Complexity adversary

Construct inputs intended to expose hidden quadratic behavior.

Examples:

* sorted points creating many components;
* reverse-sorted points;
* insertion repeatedly bridging one additional neighbor;
* insertion bridging a very long run of nodes;
* alternating construction/destruction patterns if deletion exists;
* snapshots after every insertion.

Plot or tabulate runtime versus (n).

Do not claim asymptotic complexity from one timing. Look for clear scaling regressions.

# 26. Exact novelty special case

Set:

[
\epsilon=0.
]

Then FUTCache should reduce to exact seen/not-seen semantics, subject to floating-point equality rules documented by the API.

Compare against an independent hash set / sorted set oracle.

For a finite alphabet of (m) discrete values, test all histories for small (m), e.g.

[
m\le8.
]

Enumerate every reachable subset.

Verify all (2^m) future-equivalence classes are distinguishable by an appropriate future query.

This directly exercises the lower-bound intuition:

[
b_{\min}=m.
]

# 27. Exhaustive model checking for tiny domains

For a tiny discrete domain such as:

[
K={0,1,2,3},
]

enumerate every history up to a chosen length.

For each pair (H,H'):

1. compute canonical FUT state;
2. compute whether they are mathematically future-equivalent;
3. verify:

[
State(H)=State(H')
\iff
H\sim_{\mathrm{fut}}H'.
]

Then enumerate continuations up to sufficient length.

For exact novelty, a one-symbol distinguishing continuation should suffice whenever visited sets differ.

For metric novelty on finite (K), similarly find separating queries.

This is small enough to be exhaustive, not probabilistic.

# 28. Differential state reconstruction

At random checkpoints:

1. extract/serialize production state;
2. independently rebuild a new cache from the full historical sequence;
3. compare semantic state;
4. independently rebuild another cache from only a minimal history producing the same coverage;
5. compare again.

All three must behave identically on future continuations.

This specifically validates the quotient property.

# 29. Fuzz the public API

If practical, create libFuzzer harnesses for:

* operation sequences;
* snapshot parser;
* tower creation/configuration;
* serialization/deserialization;
* public query/update API.

Interpret bytes as structured operations.

Never merely fuzz random pointers into APIs requiring valid objects.

Maintain API-valid state while mutating operation sequences.

Run sanitizer-backed fuzzing for as long as budget permits.

Keep the corpus and every crashing input.

# 30. Misuse resistance

For every documented invalid usage, verify controlled failure:

* null pointers;
* null output parameters;
* invalid level count;
* impossible epsilon;
* zero-size configuration where prohibited;
* double initialization where applicable;
* malformed snapshot;
* operation on invalid/uninitialized handles if API can detect it.

Do not require the library to survive behavior explicitly documented as undefined C behavior, but identify such boundaries clearly.

# 31. Compare Debug and Release

Run deterministic reference tests in both builds.

Behavior must match.

Look especially for bugs that disappear in Debug due to:

* assertions;
* zero-initialized memory;
* different optimization;
* timing;
* integer overflow;
* race windows.

# 32. Determinism

For single-threaded execution, run identical operation sequences repeatedly.

Require identical semantic results.

If serialization is promised canonical/deterministic, require byte-identical snapshots.

For concurrent execution, distinguish allowed scheduling nondeterminism from semantic incorrectness.

# 33. Do not silently fix failures

When you discover a bug:

1. preserve the failing test;
2. minimize the reproducer;
3. explain the violated invariant;
4. identify likely root cause;
5. only then propose a patch.

Do not weaken assertions to make tests pass.

Do not alter the mathematical oracle to match implementation behavior.

If implementation semantics differ intentionally from the mathematics, identify that as a specification discrepancy.

# 34. Severity classification

Classify findings:

### P0 — correctness/security

* wrong novelty answer;
* corrupted canonical state;
* snapshot accepts dangerous malformed input;
* data race producing invalid result;
* UAF/double free;
* deadlock;
* memory corruption.

### P1 — major robustness

* leak;
* allocation-failure corruption;
* snapshot nondeterminism contrary to contract;
* broken tower compatibility;
* serious integer-overflow risk;
* linearizability violation if promised.

### P2 — performance/spec

* hidden quadratic behavior;
* unnecessary state growth;
* poor error propagation;
* ambiguous boundary semantics;
* portability warning.

### P3 — cleanup

* weak test coverage;
* documentation mismatch;
* minor diagnostics.

# 35. Required final report

Do not return “tests pass.”

Return a structured verification report containing:

## A. Repository understanding

Describe the actual implementation architecture discovered.

## B. Build matrix

For each compiler/build/sanitizer combination:

```text
configuration
build result
tests run
pass/fail
warnings
```

## C. Invariants verified

Explicitly list each mathematical and structural invariant tested.

## D. Bugs found

For every bug:

```text
severity
minimal reproducer
expected result
actual result
root cause
affected code
proposed fix
```

## E. Differential-testing statistics

Report:

```text
number of seeds
number of operation sequences
total operations
epsilon range
coordinate distributions
maximum history size
```

## F. Concurrency statistics

Report:

```text
thread counts
operation counts
TSan result
deadlock result
linearizability checks
```

## G. Fuzzing statistics

Report:

```text
fuzzer
runtime
executions
coverage if available
unique crashes
corpus size
```

## H. Complexity results

Give measured scaling for:

* insert;
* query;
* merge storms;
* snapshot;
* tower operations.

## I. Theory-validation results

Include:

[
H\sim_{\mathrm{fut}}H'
\Longrightarrow
Out(H;w)=Out(H';w),
]

the reciprocal workload estimate

[
\widehat D_{\mathrm{cache}},
]

and any tower-projection tests.

## J. Release verdict

Choose exactly one:

```text
RELEASE READY
RELEASE READY WITH NON-BLOCKING ISSUES
NOT RELEASE READY
```

Justify it.

# 36. Standard of evidence

A test is not convincing merely because it executes code.

Prefer:

[
\text{production implementation}
\quad\text{vs}\quad
\text{independent oracle}.
]

Prefer exhaustive enumeration where the state space is small.

Prefer metamorphic properties where exact expected outputs are difficult to enumerate.

Prefer sanitizer/fuzzer evidence for memory and concurrency properties.

Prefer minimal reproducers over giant logs.

Assume every untested edge case contains a bug.

The goal is to reach the point where an adversarial reviewer can ask:

> “How do you know FUTCache actually implements the quotient (H/\sim_{\mathrm{fut}})?”

and we can answer with executable evidence rather than intuition.

The most important invariant in the entire campaign is:

[
\boxed{
U_\epsilon(H)=U_\epsilon(H')
\implies
\forall w,;
Out(H;w)=Out(H';w).
}
]

Try as hard as possible to falsify it.

If you cannot falsify it after exhaustive tiny-state checking, differential randomized testing, sanitizer runs, fuzzing, allocation-failure injection, snapshot corruption, and concurrency torture, report exactly what was tested and the remaining limits of confidence.

**Do not protect FUTCache. Attack it.**

# Changelog

All notable changes to FUTCache are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version numbers track `CMakeLists.txt` / `pyproject.toml`; releases before
v1.3.0 were development milestones that were never tagged.

## [Unreleased]

### Added

- **Payload TTL + LRU answer-cache layer** (`PackCache`, v1.4.0):
  - `PackCache(..., max_entries=N, ttl=seconds)` adds an LRU payload-cap and
    a time-to-live expiry on the Python payload store. Expired payloads are
    lazily dropped (and cleared by `purge()`); the LRU cap evicts the
    least-recently-used payload when `max_entries` is exceeded.
  - `get_or_compute(point, compute)` is the drop-in answer-cache primitive:
    it serves the cached payload on a semantic hit, or calls `compute(point)`
    and stores the result when the query is novel or the payload was
    evicted/expired. This is what lets a deployment skip an LLM/expensive
    call on a semantically-redundant query.
  - New accessors `payload_count()` and `purge()` for cache-health metrics.
  - Payload timestamps shift with the C FIFO pressure eviction, so LRU/TTL
    stay correct under `max_memory_bytes`.
  - Tests in `tests/test_answer_cache.py` (TTL expiry, lazy purge, LRU
    capacity + recency, `get_or_compute` compute-once/recompute-after-expiry).

- **Density-aware adaptive epsilon via a knee-method region tree**
  (`EpsilonTree`): replaces the single global `epsilon` with one that varies
  by region. A binary space-partition tree is built over a calibration set,
  and each leaf's `epsilon` is set to the **knee** of its local
  k-th-nearest-neighbour distance curve (the DBSCAN-style threshold). A query
  traverses to its leaf and the radius is passed to `observe_with_radius`.
  Supports `l2`/`l1`/`linf`/`cosine` distances (the metric must match the
  cache). Tests in `tests/test_epsilon_tree.py`; interactive demo in
  `demos/epsilon_tree_demo.py`.
  - Honest result on real data: the knee gives a sensible auto-ε (~0.34,
    matching real paraphrase distances) but does NOT beat a precision-tuned
    fixed ε — use it as an auto-initializer, then refine with the
    precision/ε frontier. Per-region refinement needs larger clusters for a
    stable knee. (Also surfaced a units bug: the tree must use the same
    distance as the cache, not Euclidean when the cache uses cosine.)

## [1.3.0] - 2026-08-24

First tagged release.

### Added

- **Adaptive semantic cache calibration** (`src/pack.c`, `python/futcache/adaptive.py`):
  - `futcache_pack_observe_with_radius` — exact variable-ball stabbing over a
    union of per-representative balls (a farther centre with a larger
    certified radius matches correctly; not a nearest-centre heuristic).
  - `AdaptiveRadiusPolicy` / `AdaptiveRadiusController` with Poincare radial
    resolution and Isolation-Forest contraction
    (`epsilon(x) = epsilon_0 * (1 - ||z(x)||^2)^gamma * exp(-lambda * i(x))`).
  - `CompactIsolationForest` — flat float32/int32 tree arrays, no retained
    embedding matrix.
  - Prime-base Halton trials for `(epsilon_0, gamma, lambda)` calibration.
- **Exact variable-radius VP-tree covering**: subtree `max_acceptance_radius`
  bounds prune adaptive ball lookup while staying exact.
- Python bindings for the full calibration stack (`PackCache(..., radius=...)`,
  `AdaptiveRadiusController`, `CompactIsolationForest`, `halton_trials`,
  `poincare_embed`).
- Adversarial stress test suite (`tests/test_pack_stress.c`, 4 tests):
  adaptive-radius VP-tree vs linear differential across all metrics and
  dimensions, byte-ceiling eviction correctness, adaptive serialize
  roundtrip, and concurrent writers/readers on a variable-radius VP-tree
  cache.

### Fixed

- Test-harness leak in `test_vptree_metric_differential`
  (`tests/test_pack_vptree.c`): the 15-iteration loop re-assigned the domain
  bounds without freeing the previous allocation (16,608 bytes across 28
  allocations, reported by LeakSanitizer). The full suite is now
  leak-clean under `detect_leaks=1`.

### Changed

- `tests/` suite grows from 75 to 79 tests; `docs/verification.md` gains a
  v1.3.0 re-verification addendum (LSan now usable on the build host).

## [1.2.0] - 2026-08-23

Development milestone (untagged).

### Added

- **Crash-safe packing persistence**: version-2 snapshots persist every
  adaptive radius, strictly framed, little-endian, CRC32-protected,
  backward-compatible with v1 snapshots.
- **Hard byte ceiling** for the packing cache (`max_memory_bytes`):
  deterministic FIFO pressure eviction that recycles the oldest
  representative allocation in place, plus exact live/peak memory telemetry
  and allocation-failure-atomic updates.
- **Exact scapegoat VP-tree backend** (`futcache_pack_vptree_backend`):
  O(n) index metadata (borrows cache-owned vectors), logarithmic inserts,
  exact nearest/lookup; chordal-transform cosine with automatic linear
  fallback for non-normalized inputs.
- **NitroSAT V3 offline representative optimization** (`FUTCACHE_BUILD_NITROSAT`):
  hard WCNF coverage/separations, verified solver claims, safe smaller-of
  (solver, greedy) policy.
- BEIR/SciFact semantic-cache customer demo and human-CQAdupStack benchmark.

## [1.1.0] - 2026-08-23

Development milestone (untagged).

### Added

- **Deterministic-Voronoi CRDT cache** (`<futcache/crdt.h>`): per-cell join
  under a deterministic priority, gossip-mergeable snapshots, and
  anchor-construction helpers (certified uniform grid, low-discrepancy
  Halton nets, sampled covering-radius estimator, safe δ-net builder).
- **Exact bounded-dimension `L_inf` box cache** (`<futcache/box.h>`),
  dimensions 1-8.
- **Python bindings** for the packing cache via nanobind + scikit-build-core
  (`PackCache`, payload dict keyed by representative slot, memory telemetry).
- Bekko embedding semantic-cache experiments (English + multilingual) with
  cacheability analysis (`D_cache`, discriminative margin) and reuse-precision
  reporting.
- Headline result plots (`plots/`) and `scripts/plot_results.py`.

## [1.0.0] - 2026-08-23

Initial production-ready release (untagged).

### Added

- Exact interval-union novelty cache (`<futcache/futcache.h>`): AVL-backed
  canonical state, linearizable reader/writer locking, allocation-failure-
  atomic observations, runtime stats + invariant validation, and versioned
  CRC32-protected serialization.
- Uniform dyadic resolution tower (`<futcache/tower.h>`): occupancy bitsets,
  Fenwick prefix counts, spatial select, and first-discovery logs.
- n-d Voronoi packing cache (`<futcache/pack.h>`): five built-in distances
  (L1/L2/L_inf/cosine/Poincare) plus custom metrics, exact bounded memory
  telemetry, and versioned serialization.
- Empirical LRU comparison benchmark (`bench/cache_comparison.c`) and
  throughput benchmark (`bench/benchmark.c`).
- Formal treatment in `how.md`, `formal.md`, `testplan.md`, and
  `HOWITWORKS.md`.

[Unreleased]: https://github.com/sethuiyer/FUTCache/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/sethuiyer/FUTCache/releases/tag/v1.3.0
[1.2.0]: https://github.com/sethuiyer/FUTCache/compare/v1.1.0...v1.3.0
[1.1.0]: https://github.com/sethuiyer/FUTCache/compare/v1.0.0...v1.3.0
[1.0.0]: https://github.com/sethuiyer/FUTCache/commit/30a757d

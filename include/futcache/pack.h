#ifndef FUTCACHE_PACK_H
#define FUTCACHE_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_pack: Voronoi-backed metric coverage and representative selection.
 *
 * Mathematical framing.
 *
 *   Let H be the observation history and rho_H(x) = min_{h in H} d(x, h)
 *   the distance transform. Metric novelty is the event { rho_H(x) > eps }.
 *
 *   In unbounded mode, let R be a maximal eps-separated subset of H. The
 *   Voronoi cells of R tile the domain: every point lies in exactly one cell,
 *   whose site is the nearest representative. Novelty of a query x therefore
 *   reduces to
 *   a single nearest-site lookup, d(x, R) > eps. Representative count
 *   is bounded by the packing number P(K, eps), independent of stream
 *   length. The cache state *is* the Voronoi seed set R.
 *
 *   Without an explicit memory ceiling, distance comparisons are monotone in
 *   the stream: a representative is added once and never removed. With
 *   `max_memory_bytes`, the same geometric packing is layered over a strict
 *   physical allocation bound; memory pressure recycles the oldest
 *   representative. Eviction may report extra novelty for a forgotten
 *   region, but cannot create a false cache hit.
 *
 * Trade-off vs the exact union-of-balls cache.
 *
 *   - Exact cache (futcache.h): preserves the metric-novelty oracle
 *     exactly. Available in 1D as the interval-union, and in dD under
 *     L_inf as a union of d-boxes. General exact representations exist
 *     in fixed-dimensional Euclidean space via Delaunay triangulations,
 *     alpha-complexes, and Cech complexes, but their combinatorial size
 *     grows with the complexity of the union and they do not have a
 *     clean canonical form for arbitrary metric or high-d embedding
 *     workloads.
 *
 *   - Packing cache (this header): preserves the *representative* novelty
 *     oracle exactly. Bounded by P(K, eps). Practical for arbitrary
 *     metric, including high-dimensional embeddings under cosine or L2.
 *     This is the regime where the exact representations above become
 *     unwieldy or inapplicable.
 *
 * Storage and lookup.
 *
 *   The current implementation keeps representatives in FIFO order and
 *   performs an O(|R|) linear scan per observe or query. Each representative
 *   is one allocation containing its d-dimensional point and list header;
 *   this lets bounded mode recycle the oldest allocation without a transient
 *   memory spike. For large |R|, the scan can be replaced by the built-in
 *   VP-tree or another nearest-neighbour backend without changing the public
 *   query API.
 */

/*
 * Distance function. Given two points of length `dimension`, return the
 * non-negative distance between them. Must satisfy the triangle inequality.
 */
typedef double (*futcache_distance_fn)(const double *a, const double *b,
                                        size_t dimension, void *context);

/*
 * Optional nearest-neighbour backend.  The cache remains the owner of the
 * representative coordinates; a backend owns only its index/state.  A
 * backend must not retain `points` from nearest() or insert().
 * All backend state and scratch memory must be obtained from the allocator
 * supplied to create(); this is required for max_memory_bytes enforcement.
 *
 * nearest() returns the distance to the closest indexed representative.  It
 * may return a conservative (over-estimated) distance when approximate: that
 * can cause false novelty, but must never suppress a genuinely novel point.
 * Return FUTCACHE_OK with `*out_distance = INFINITY` when count is zero.
 */
typedef struct futcache_pack_backend_ops {
    futcache_status_t (*create)(void **out_state, size_t dimension,
                                futcache_distance_fn distance,
                                void *distance_context,
                                const futcache_allocator_t *allocator,
                                void *context);
    void (*destroy)(void *state, const futcache_allocator_t *allocator,
                    void *context);
    futcache_status_t (*clear)(void *state, void *context);
    futcache_status_t (*insert)(void *state, const double *point,
                                size_t dimension, void *context);
    futcache_status_t (*nearest)(void *state, const double *point,
                                 size_t dimension, double *out_distance,
                                 void *context);
} futcache_pack_backend_ops_t;

typedef struct futcache_pack_config {
    /* Number of coordinates per point. Must be >= 1. */
    size_t dimension;
    /* Novelty resolution; same units as the distance function. */
    double epsilon;
    /* Distance function. NULL selects L_inf (Chebyshev). */
    futcache_distance_fn distance;
    /* Opaque pointer passed through to the distance function. */
    void *distance_context;
    /* Inclusive lower domain bound per coordinate. Required. */
    const double *domain_min;
    /* Inclusive upper domain bound per coordinate. Required. */
    const double *domain_max;
    /* Allocator. NULL selects malloc/free. */
    futcache_allocator_t allocator;
    /* Optional index backend. NULL selects the built-in linear scan. */
    const futcache_pack_backend_ops_t *backend;
    /* Opaque backend context, passed to every backend callback. */
    void *backend_context;
    /*
     * Hard ceiling for cache-owned live allocation requests, in bytes.
     * Zero leaves the geometric packing bound as the only limit.  A non-zero
     * value includes the cache object, domain bounds, representatives, backend
     * state, and backend scratch allocations made through the supplied
     * allocator.  When a novel point cannot be added without crossing the
     * ceiling, the oldest representative is evicted and its allocation is
     * recycled in place (FIFO pressure eviction).
     * The value must be large enough for the cache, two bound vectors, and
     * at least one representative; smaller non-zero limits are invalid.
     *
     * Backends must allocate exclusively through the allocator passed to
     * create() for this ceiling to cover their memory.
     */
    size_t max_memory_bytes;
} futcache_pack_config_t;

/*
 * Built-in exact nearest-neighbour backend: a scapegoat VP-tree.
 *
 * A VP-tree partitions the space by distance to a vantage point and
 * prunes with the triangle inequality, so it works in any metric space
 * (L1, L2, L_inf, or a custom metric) and its query cost tracks the
 * intrinsic dimensionality (the doubling constant), not the ambient
 * dimension: on low-intrinsic-dimension data (real embeddings) it is
 * 10-100x faster than the linear scan even at 384 dimensions, while
 * on adversarial uniform data in high dimensions it degrades toward
 * the linear scan (the curse of dimensionality; see the backend bench).
 *
 * Inserts are amortized O(log n): the tree is a scapegoat tree with
 * local rebuilds. nearest() is exact: the returned distance equals the
 * linear-scan result, so observe()/is_novel() semantics are identical
 * to the built-in backend (differential-tested).
 *
 * Cosine: 1 - dot(a, b) is not a metric, but for L2-normalized inputs
 * the chordal distance d = |a - b| satisfies 1 - dot = d^2 / 2, and the
 * chordal distance is a metric (Euclidean on the unit sphere). The
 * backend therefore indexes with the chordal metric and converts the
 * result back, which is exact for normalized inputs. To preserve the
 * one-sided guarantee when the normalization contract is violated, the
 * backend detects non-unit-norm inputs (at insert and per query) and
 * falls back to an exact linear scan over the engine's cosine distance
 * for those points — a wrong epsilon can never turn a genuinely novel
 * point into a hit.
 *
 * Memory is allocated through the cache's allocator (counting-
 * allocator / fault-injection compatible), and every insert is
 * all-or-nothing: an allocation failure leaves the index exactly as
 * it was before the call.
 */
extern FUTCACHE_API const futcache_pack_backend_ops_t futcache_pack_vptree_backend;

typedef struct futcache_pack_stats {
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    /* FIFO pressure evictions since create or the most recent clear. */
    uint64_t evictions;
    size_t representative_count;
    /* High-water representative count since create or the most recent clear. */
    size_t peak_count;
    /* Exact live bytes requested from the configured allocator by the cache. */
    size_t memory_bytes;
    /*
     * High-water mark since create or clear, including transient backend
     * scratch.
     */
    size_t peak_memory_bytes;
    /* Configured hard limit; zero means unlimited. */
    size_t memory_limit_bytes;
} futcache_pack_stats_t;

typedef struct futcache_pack futcache_pack_t;

/*
 * Fills `config` with sensible defaults: dimension=1, epsilon=0.0, L_inf,
 * and max_memory_bytes=0 (unlimited).
 */
FUTCACHE_API void futcache_pack_config_init(futcache_pack_config_t *config);

/* Built-in distance functions. `context` is ignored. */
FUTCACHE_API double futcache_distance_l1(const double *a, const double *b,
                                          size_t dimension, void *context);
FUTCACHE_API double futcache_distance_l2(const double *a, const double *b,
                                          size_t dimension, void *context);
FUTCACHE_API double futcache_distance_linf(const double *a, const double *b,
                                            size_t dimension, void *context);
/*
 * Cosine distance, 1 - dot(a, b). Assumes the inputs are already
 * L2-normalized; the function does not normalize internally. Use this
 * with sentence-transformers output and `normalize_embeddings=True`.
 */
FUTCACHE_API double futcache_distance_cosine(const double *a, const double *b,
                                              size_t dimension, void *context);

/*
 * Creates a thread-safe packing novelty cache. `config` is consumed for
 * setup; the cache does not retain the `domain_min`/`domain_max` arrays,
 * so those may be caller-stack scratch.
 */
FUTCACHE_API futcache_status_t futcache_pack_create(
    const futcache_pack_config_t *config,
    futcache_pack_t **out_cache
);

/* Destroys a cache. No other thread may use it during destruction. */
FUTCACHE_API void futcache_pack_destroy(futcache_pack_t *cache);

/*
 * Reports whether `point` is farther than `epsilon` from every existing
 * representative. Pure query, no state change.
 */
FUTCACHE_API futcache_status_t futcache_pack_is_novel(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_is_novel
);

/*
 * Atomically tests novelty of `point` and, if novel, adds it as a new
 * representative. A point within epsilon of an existing representative is
 * reported as redundant and the state is unchanged. Under a non-zero memory
 * ceiling, adding a novel point may FIFO-evict the oldest representative;
 * `out_was_novel` remains true and stats.evictions advances.
 */
FUTCACHE_API futcache_status_t futcache_pack_observe(
    futcache_pack_t *cache,
    const double *point,
    bool *out_was_novel
);

/* Returns a consistent stats snapshot. */
FUTCACHE_API futcache_status_t futcache_pack_get_stats(
    const futcache_pack_t *cache,
    futcache_pack_stats_t *out_stats
);

/*
 * Copies up to `*inout_count` representatives into `out_points` (length
 * `dimension` each). On input, `*inout_count` is the destination capacity
 * in *representatives*; on return it is the number of representatives
 * actually written (also the required capacity). Pass `out_points=NULL`
 * and `*inout_count=0` to query the required capacity.
 *
 * On return, allocate `out_points` as
 *   (*inout_count) * dimension * sizeof(double)
 * bytes, then call again to copy.
 */
FUTCACHE_API futcache_status_t futcache_pack_copy_representatives(
    const futcache_pack_t *cache,
    double *out_points,
    size_t *inout_count
);

/*
 * Reports the distance to, and slot index of, the nearest representative.
 * On an empty cache, `*out_distance` is +infinity and `*out_index` is
 * SIZE_MAX. The index matches the ordering used by
 * futcache_pack_copy_representatives(). This always scans the representative
 * list (a custom nearest-neighbour backend does not expose identities), so it
 * is O(|R|) and never mutates state. Indices remain stable across appends but
 * may shift after a pressure eviction; callers must not retain them across a
 * mutating call when max_memory_bytes is non-zero.
 */
FUTCACHE_API futcache_status_t futcache_pack_nearest(
    const futcache_pack_t *cache,
    const double *point,
    double *out_distance,
    size_t *out_index
);

/*
 * Empties the representative set; resets observation, eviction, and peak
 * telemetry while advancing generation.
 */
FUTCACHE_API futcache_status_t futcache_pack_clear(futcache_pack_t *cache);

/*
 * Validates the epsilon-separation invariant (every pair of representatives
 * is strictly farther than epsilon apart), the telemetry lifecycle
 * invariants (generation >= observations, novel <= observations, count +
 * evictions == novel before counter saturation, count <= peak), and the hard
 * live/peak memory ceiling. O(n^2) in the representative count.
 */
FUTCACHE_API futcache_status_t futcache_pack_validate(const futcache_pack_t *cache);

/*
 * Serializes an atomic packing-state snapshot into a versioned,
 * endian-independent, CRC32-protected format. Pass buffer=NULL to query the
 * required size. The snapshot contains representatives, bounds, counters,
 * eviction telemetry, the memory ceiling, and the selected built-in metric.
 * Custom distance functions cannot be serialized and return
 * FUTCACHE_ERROR_UNSUPPORTED_PLATFORM.
 *
 * The output buffer is caller-owned and is not charged to max_memory_bytes.
 */
FUTCACHE_API futcache_status_t futcache_pack_serialize(
    const futcache_pack_t *cache,
    void *buffer,
    size_t buffer_size,
    size_t *out_size
);

/*
 * Restores a packing snapshot. Derived nearest-neighbour state is rebuilt;
 * snapshots made with the built-in VP-tree restore that backend, while a
 * custom backend restores to the exact linear scan. `allocator` may be NULL
 * to use malloc/free. On failure, *out_cache is NULL.
 */
FUTCACHE_API futcache_status_t futcache_pack_deserialize(
    const void *data,
    size_t data_size,
    const futcache_allocator_t *allocator,
    futcache_pack_t **out_cache
);

#ifdef __cplusplus
}
#endif

#endif

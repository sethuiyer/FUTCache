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
 *   Let R be a maximal eps-separated subset of H. The Voronoi cells of R
 *   tile the domain: every point lies in exactly one cell, whose site is
 *   the nearest representative. Novelty of a query x therefore reduces to
 *   a single nearest-site lookup, d(x, R) > eps. Representative count
 *   is bounded by the packing number P(K, eps), independent of stream
 *   length. The cache state *is* the Voronoi seed set R.
 *
 *   Distance comparisons are monotone in the stream: a representative is
 *   added once and never removed. The state grows monotonically and
 *   saturates when no further eps-separated point exists in the domain.
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
 *   The current implementation uses a dynamic array of representatives and
 *   performs an O(|R|) linear scan per observe or query. For moderate |R|
 *   this is fast and branch-free, and the cache stays compact because each
 *   representative is a single d-dimensional point plus header. For large
 *   |R| in high-dimensional embedding workloads, the linear scan can be
 *   swapped for a k-d tree or an approximate nearest-neighbor structure
 *   (HNSW, IVF-PQ, etc.) without changing the public API.
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
} futcache_pack_config_t;

typedef struct futcache_pack_stats {
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    size_t representative_count;
    size_t peak_count;
    size_t memory_bytes;
} futcache_pack_stats_t;

typedef struct futcache_pack futcache_pack_t;

/* Fills `config` with sensible defaults: dimension=1, epsilon=0.0, L_inf. */
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
 * reported as redundant and the state is unchanged.
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

/* Empties the representative set; bumps generation. */
FUTCACHE_API futcache_status_t futcache_pack_clear(futcache_pack_t *cache);

/*
 * Validates the epsilon-separation invariant: every pair of representatives
 * is strictly farther than epsilon apart. O(n^2) in the representative count.
 */
FUTCACHE_API futcache_status_t futcache_pack_validate(const futcache_pack_t *cache);

#ifdef __cplusplus
}
#endif

#endif

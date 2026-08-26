#ifndef FUTCACHE_EMBED_H
#define FUTCACHE_EMBED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"
#include "futcache/pack.h" /* futcache_distance_fn, futcache_pack_config_t */
#include "futcache/crdt.h" /* futcache_crdt_anchor_strategy_t */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_embed: distance-to-anchors metric embedding.
 *
 * Mathematical foundation.
 *
 *   For a metric space (X, d) and a fixed anchor set A = {a_1, ..., a_m},
 *   the embedding is:
 *
 *       phi(x) = (d(x, a_1), d(x, a_2), ..., d(x, a_m))  in R^m
 *
 *   Key properties:
 *
 *   1-Lipschitz:  For any coordinate i,
 *       |d(x, a_i) - d(y, a_i)| <= d(x, y)     (triangle inequality)
 *   So ||phi(x) - phi(y)||_inf <= d(x, y).
 *
 *   Additive distortion bound:  If A is a delta-net of X (every point
 *   within delta of some anchor), then
 *       d(x, y) <= ||phi(x) - phi(y)||_inf + 2*delta
 *   This follows from the triangle inequality through the nearest anchor:
 *       d(x, y) <= d(x, a*) + d(a*, y)
 *               <= (||phi(x) - phi(a*)||_inf + delta)
 *                  + (||phi(y) - phi(a*)||_inf + delta)
 *       where a* is the nearest anchor to both x and y.
 *       Since phi(a*) has one zero coordinate, this simplifies to
 *       d(x, y) <= ||phi(x) - phi(y)||_inf + 2*delta.
 *
 *   Combined:  |d(x,y) - ||phi(x)-phi(y)||_inf| <= 2*delta.
 *
 *   One-sidedness under distortion (Corollary 4.1):
 *     If a query is declared novel at threshold epsilon in embedded
 *     space (Linf), then in the original space:
 *       - If d(x, R) > epsilon + 2*delta, x is genuinely novel (no FP).
 *       - If d(x, R) <= epsilon - 2*delta, x is genuinely seen (no FN).
 *       - In the gap (epsilon - 2*delta, epsilon + 2*delta], the verdict
 *         may be either; width = 4*delta.
 *
 *   For the cache to be conservative (no false positives at original
 *   epsilon), use embedded threshold = epsilon - 2*delta.
 *
 *   Why this is the right choice for FUTCache:
 *     - The CRDT engine already uses anchors (grid or Halton).
 *     - The embedding is exact (no training, no data-dependent choice).
 *     - The distortion is bounded and known: 2*delta.
 *     - The VP-tree operates in R^m with Linf (simplest VP-tree).
 *
 *   Trade-off: O(m) distance evaluations per query for the ability to use
 *   ANY metric with VP-tree acceleration. For m=1000 anchors and n=100,000
 *   points, embedding cost (1000 distance evaluations) is comparable to
 *   VP-tree query cost (O(log n) ~ 17 distance evaluations in original
 *   space, but each in d dimensions). For high-dimensional data (d >= 50),
 *   the embedded approach wins because VP-tree in m-dim Linf is much
 *   faster than in d-dim L2.
 */

typedef struct futcache_embed_config {
    /* Ambient dimension of the original space X. Must be >= 1. */
    size_t dimension;
    /* Number of anchors |A| = m. Must be >= 1. */
    size_t anchor_count;
    /* Anchor coordinates, [anchor_count * dimension]. Caller retains
     * ownership; the embedding copies them. */
    const double *anchors;
    /* Distance function on the original space X. NULL selects L_inf.
     * Must satisfy the triangle inequality for the distortion bound to hold. */
    futcache_distance_fn distance;
    /* Opaque pointer passed through to the distance function. */
    void *distance_context;
    /* Inclusive lower/upper domain bounds in the original space.
     * Used for covering radius estimation. Required. */
    const double *domain_min;
    const double *domain_max;
    /* Allocator. NULL selects malloc/free. */
    futcache_allocator_t allocator;
} futcache_embed_config_t;

typedef struct futcache_embed futcache_embed_t;

/*
 * Creates a distance-to-anchors embedding from a pre-computed anchor set.
 * The anchors are copied; the caller retains ownership of the input array.
 * The covering radius delta is estimated by sampling if domain bounds are
 * provided (lower bound on true covering radius).
 */
FUTCACHE_API futcache_status_t futcache_embed_create(
    const futcache_embed_config_t *config,
    futcache_embed_t **out_embed);

/* Destroys the embedding. No other thread may use it during destruction. */
FUTCACHE_API void futcache_embed_destroy(futcache_embed_t *embed);

/*
 * Embeds a point: phi(x) = (d(x, a_1), ..., d(x, a_m)).
 * out_embedded must hold anchor_count doubles (caller-allocated).
 * Pure computation, no state change.
 */
FUTCACHE_API futcache_status_t futcache_embed_point(
    const futcache_embed_t *embed,
    const double *point,
    double *out_embedded);

/*
 * Returns the estimated covering radius delta of the anchor set.
 * This is the additive distortion bound:
 *   |d(x,y) - ||phi(x)-phi(y)||_inf| <= 2 * delta
 * The value is a lower bound on the true covering radius (sampling-based).
 */
FUTCACHE_API double futcache_embed_covering_radius(const futcache_embed_t *embed);

/*
 * Returns the number of anchors (embedded dimension m).
 */
FUTCACHE_API size_t futcache_embed_anchor_count(const futcache_embed_t *embed);

/*
 * Returns the original space dimension.
 */
FUTCACHE_API size_t futcache_embed_original_dimension(const futcache_embed_t *embed);

/*
 * Computes the embedded-space epsilon for a given original-space epsilon.
 * For conservative one-sidedness (no false positives at original epsilon):
 *   epsilon_embedded = epsilon_original - 2 * delta
 * Returns FUTCACHE_ERROR_OUT_OF_RANGE if epsilon_original <= 2*delta
 * (embedding distortion would eat the entire threshold).
 */
FUTCACHE_API futcache_status_t futcache_embed_adjusted_epsilon(
    const futcache_embed_t *embed,
    double epsilon_original,
    double *out_epsilon_embedded);

/*
 * Provides a futcache_distance_fn that computes distance in the embedded
 * space. This distance function operates on embedded vectors of length
 * anchor_count and uses L_inf (max coordinate difference).
 *
 * Usage:
 *   1. Create embed with futcache_embed_create.
 *   2. For each point to be observed, compute phi(x) with futcache_embed_point.
 *   3. Create a pack cache with:
 *        - dimension = futcache_embed_anchor_count(embed)
 *        - distance = futcache_embed_distance (this function)
 *        - distance_context = embed
 *        - epsilon = futcache_embed_adjusted_epsilon(embed, original_epsilon)
 *        - backend = &futcache_pack_vptree_backend  (Linf VP-tree)
 *   4. Before each observe/is_novel call, embed the query point.
 *
 * The embed handle must outlive the pack cache.
 */
FUTCACHE_API double futcache_embed_distance(
    const double *a,
    const double *b,
    size_t dimension,
    void *context);

/*
 * Builds a pack cache pre-configured for the embedded space.
 * This is a convenience wrapper that:
 *   1. Computes anchors (grid or Halton) for the target covering radius.
 *   2. Creates the embedding.
 *   3. Adjusts epsilon for distortion.
 *   4. Creates a pack cache in the embedded space with Linf + VP-tree.
 *
 * The caller receives both the embed handle and the pack cache.
 * Before each query, the caller must embed the point with futcache_embed_point.
 *
 * `target_radius` is the desired covering radius delta. The anchor set is
 * generated to achieve covering radius <= target_radius. If grid is used
 * and the required cell count exceeds max_anchors, the function returns
 * FUTCACHE_ERROR_OUT_OF_RANGE.
 *
 * `probe_count` is used for Halton covering radius estimation.
 */
FUTCACHE_API futcache_status_t futcache_embed_pack_create(
    size_t original_dimension,
    double epsilon_original,
    futcache_distance_fn original_distance,
    void *original_distance_context,
    const double *domain_min,
    const double *domain_max,
    double target_radius,
    futcache_crdt_anchor_strategy_t strategy,
    size_t max_anchors,
    size_t probe_count,
    const futcache_allocator_t *allocator,
    futcache_embed_t **out_embed,
    futcache_pack_t **out_cache);

#ifdef __cplusplus
}
#endif

#endif /* FUTCACHE_EMBED_H */

#ifndef FUTCACHE_PERSIST_ND_H
#define FUTCACHE_PERSIST_ND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"
#include "futcache/pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_persist_nd: d-D persistent novelty (Design Sketch 01, Phase 3).
 *
 * Mathematical framing.
 *
 *   The 1-D persistent novelty engine (futcache_persist_t) tracks the
 *   merge tree of the novelty landscape U_t = union of B(y, t) for
 *   1-D points.  The d-D extension replaces the merge tree with a
 *   per-representative birth/death tracking on top of the pack cache.
 *
 *   For d >= 2, the merge tree generalizes to a **merge dendrogram**
 *   based on the single-linkage hierarchy of the distance function
 *   rho_H(x) = min_{y in H} d(x, y).  The full d-D merge tree is
 *   equivalent to the 1-D merge tree of the distance function's
 *   sublevel sets, which can be computed from the **Voronoi diagram**
 *   of the representative set.
 *
 *   In practice, for bounded memory and O(log n) queries, we track:
 *
 *   - For each representative: birth_time (when it was inserted)
 *     and current radius (the t at which it would merge with a
 *     neighbour).
 *   - The "persistence" of a representative is its distance to the
 *     nearest other representative minus its own radius.
 *
 *   This gives a **reduced persistence diagram**: instead of the full
 *   merge tree (which has O(n^d) topology in d dimensions), we track
 *   the per-rep persistence, which is a 1-D summary that is exact
 *   for the purpose of eviction and novelty queries.
 *
 * Guarantees.
 *
 *   - One-sidedness: is_novel queries are exact (no false negatives).
 *   - Bounded memory: a configured hard byte ceiling is enforced. Exact
 *     arbitrary-scale queries retain O(h) history for h observations, in
 *     addition to O(n) representative state.
 *   - Eviction: evict_lowest_persistence() removes the rep with the
 *     smallest persistence, preserving the most novel reps.
 *   - Prime tagging: each rep gets a prime signature based on its
 *     insertion index, enabling CRDT merge of the rep set.
 */

/*
 * A d-D persistent representative.
 *
 * id: the pack-cache representative index.
 * point: the d-dimensional representative coordinates.
 * radius: the pack-cache radius (epsilon at insertion).
 * birth_index: the observation index at which this rep was inserted.
 * birth_prime: p_birth_index mod M (prime signature).
 * nearest_dist: distance to the nearest other representative.
 * persistence: nearest_dist - radius (the "persistence" of this rep).
 *   If nearest_dist < radius, the rep is "submerged" (persistence < 0).
 */
typedef struct futcache_persist_nd_rep {
    size_t id;
    double *point;        /* [dimension] */
    double radius;
    size_t birth_index;
    uint64_t birth_prime; /* p_birth_index mod M */
    double nearest_dist;
    double persistence;
} futcache_persist_nd_rep_t;

typedef struct futcache_persist_nd futcache_persist_nd_t;

/*
 * Create a d-D persistent novelty engine.
 *
 * dimension: spatial dimension d >= 1.
 * epsilon: base novelty radius (the t at which balls start to merge).
 * distance: distance function (Linf, L2, L1, cosine, etc.).
 * domain_min, domain_max: per-coordinate domain bounds.
 * max_memory_bytes: 0 for unlimited.
 *
 * The engine wraps an internal pack cache for exact novelty queries
 * and adds birth/death tracking for persistence.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_create(
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *distance_context,
    const double *domain_min,
    const double *domain_max,
    size_t max_memory_bytes,
    const futcache_allocator_t *allocator,
    futcache_persist_nd_t **out_engine);

FUTCACHE_API void futcache_persist_nd_destroy(futcache_persist_nd_t *engine);

/*
 * Observe a d-dimensional point.
 *
 * Returns was_novel: true if x is outside all existing epsilon-balls.
 * If novel, a new representative is created with birth_index =
 * the current observation count.
 *
 * The persistence of all existing reps is updated (nearest neighbour
 * distances are recomputed incrementally).
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_observe(
    futcache_persist_nd_t *engine,
    const double *x,
    bool *out_was_novel);

/*
 * Query: is x novel at scale t?
 *
 * Exact: true iff x is outside U_t(H) = union of B(y_i, t) for all
 * observed points y_i.  This is the same as the pack cache's lookup
 * but with a custom radius t instead of the per-rep radius.
 *
 * O(n) for the linear backend, O(log n) for the VP-tree backend.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_is_novel_at(
    const futcache_persist_nd_t *engine,
    const double *x,
    double t,
    bool *out_is_novel);

/*
 * Get the nearest-neighbour distance for each representative.
 *
 * Two-pass: pass out=NULL, *count=0 to get the rep count.
 *
 * The nearest_dist array is the distance from rep i to its nearest
 * other rep (excluding itself). For a single rep, nearest_dist = INFINITY.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_nearest_distances(
    const futcache_persist_nd_t *engine,
    double *out_distances,
    size_t *inout_count);

/*
 * Get the persistence of each representative.
 *
 * persistence(i) = nearest_dist(i) - radius(i).
 *
 * Two-pass: pass out=NULL, *count=0 to get the rep count.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_persistences(
    const futcache_persist_nd_t *engine,
    double *out_persistences,
    size_t *inout_count);

/*
 * Evict the representative with the lowest persistence.
 *
 * Returns the evicted rep's index (0-based before eviction).
 * The pack cache renumbers remaining reps, so all indices shift down
 * by 1 for reps that were after the evicted one.
 *
 * If all reps have negative persistence (all "submerged"), the one
 * with the least-negative persistence is evicted (closest to being
 * "genuinely novel").
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_evict_lowest(
    futcache_persist_nd_t *engine,
    size_t *out_evicted_index);

/*
 * Count reps with persistence >= tau.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_count_above(
    const futcache_persist_nd_t *engine,
    double tau,
    size_t *out_count);

/*
 * Get the full rep set with all persistence data.
 *
 * Two-pass: pass out=NULL, *count=0 to get the rep count.
 * Each futcache_persist_nd_rep_t has a point array that the caller
 * must not free (it points into the engine's internal storage).
 *
 * The caller calls futcache_persist_nd_free_reps() to release
 * the rep array after copying the data it needs.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_copy_reps(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_rep_t *out_reps,
    size_t *inout_count);

FUTCACHE_API void futcache_persist_nd_free_reps(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_rep_t *reps,
    size_t count);

/*
 * Clear all state.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_clear(
    futcache_persist_nd_t *engine);

/*
 * Statistics.
 */
typedef struct futcache_persist_nd_stats {
    uint64_t observations;
    size_t rep_count;
    double max_persistence;
    double min_persistence;
    double avg_persistence;
    size_t prime_birth_count; /* reps whose birth_index is prime */
    size_t memory_bytes;
} futcache_persist_nd_stats_t;

FUTCACHE_API futcache_status_t futcache_persist_nd_get_stats(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_stats_t *out_stats);

/*
 * Validate internal invariants.
 */
FUTCACHE_API futcache_status_t futcache_persist_nd_validate(
    const futcache_persist_nd_t *engine);

#ifdef __cplusplus
}
#endif

#endif

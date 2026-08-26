#define _POSIX_C_SOURCE 200809L

#include "futcache/persist_nd.h"
#include "futcache/persist.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Internal state
 * ============================================================ */

struct futcache_persist_nd {
    size_t dimension;
    double epsilon;
    futcache_distance_fn distance_fn;
    void *distance_context;

    /* Domain bounds (owned copies). */
    double *domain_min;
    double *domain_max;

    /* Observed points (unsorted). */
    double *points;           /* [capacity * dimension] */
    size_t point_count;
    size_t point_capacity;

    /* Per-observation birth tracking: observation index -> rep index,
     * or SIZE_MAX if the observation was non-novel (redundant). */
    size_t *obs_to_rep;       /* [point_count] */

    /* Rep-level data, indexed by rep index (0..rep_count-1). */
    double *radii;            /* [rep_capacity] */
    double *nearest_dist;     /* [rep_capacity] */
    size_t *birth_index;      /* [rep_capacity] */
    uint64_t *birth_prime;    /* [rep_capacity] */

    size_t rep_count;
    size_t rep_capacity;

    uint64_t observations;

    futcache_allocator_t allocator;
    size_t max_memory_bytes;
};

/* ============================================================
 * Default allocator
 * ============================================================ */

static void *default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

static bool normalize_allocator(
    const futcache_allocator_t *requested, futcache_allocator_t *normalized)
{
    if (normalized == NULL) return false;
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = default_allocate;
        normalized->deallocate = default_deallocate;
        normalized->context = NULL;
        return true;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL)
        return false;
    normalized->allocate = requested->allocate;
    normalized->deallocate = requested->deallocate;
    normalized->context = requested->context;
    return true;
}

/* ============================================================
 * Is n prime? (for stats)
 * ============================================================ */

static bool is_prime_uint(uint64_t n)
{
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
    uint64_t i = 5;
    while (i * i <= n) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
        i += 6;
    }
    return true;
}

/* ============================================================
 * Nearest-neighbour distance computation
 *
 * For each rep i, compute the minimum distance to any other rep j != i.
 * O(n^2 * d) — fine for moderate n; for large n, a VP-tree would help.
 * ============================================================ */

static void recompute_nearest(
    const futcache_persist_nd_t *engine)
{
    size_t n = engine->rep_count;
    size_t dim = engine->dimension;
    if (n == 0) return;

    for (size_t i = 0; i < n; ++i) {
        double best = INFINITY;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double d = engine->distance_fn(
                engine->points + i * dim,
                engine->points + j * dim,
                dim, engine->distance_context);
            if (d < best) best = d;
        }
        engine->nearest_dist[i] = best;
    }
}

/* ============================================================
 * futcache_persist_nd_create / destroy
 * ============================================================ */

futcache_status_t futcache_persist_nd_create(
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *distance_context,
    const double *domain_min,
    const double *domain_max,
    size_t max_memory_bytes,
    const futcache_allocator_t *allocator,
    futcache_persist_nd_t **out_engine)
{
    if (dimension == 0 || out_engine == NULL || distance == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (!isfinite(epsilon) || epsilon < 0.0)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (domain_min == NULL || domain_max == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *out_engine = NULL;

    for (size_t i = 0; i < dimension; ++i) {
        if (!isfinite(domain_min[i]) || !isfinite(domain_max[i]) ||
            domain_max[i] <= domain_min[i])
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_allocator_t alloc;
    if (!normalize_allocator(allocator, &alloc))
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    futcache_persist_nd_t *engine =
        (futcache_persist_nd_t *)alloc.allocate(alloc.context,
                                                sizeof(futcache_persist_nd_t));
    if (engine == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;

    memset(engine, 0, sizeof(*engine));
    engine->allocator = alloc;
    engine->dimension = dimension;
    engine->epsilon = epsilon;
    engine->distance_fn = distance;
    engine->distance_context = distance_context;
    engine->max_memory_bytes = max_memory_bytes;

    /* Copy domain bounds. */
    engine->domain_min = (double *)alloc.allocate(
        alloc.context, dimension * sizeof(double));
    engine->domain_max = (double *)alloc.allocate(
        alloc.context, dimension * sizeof(double));
    if (engine->domain_min == NULL || engine->domain_max == NULL) {
        alloc.deallocate(alloc.context, engine->domain_min);
        alloc.deallocate(alloc.context, engine->domain_max);
        alloc.deallocate(alloc.context, engine);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(engine->domain_min, domain_min, dimension * sizeof(double));
    memcpy(engine->domain_max, domain_max, dimension * sizeof(double));

    *out_engine = engine;
    return FUTCACHE_OK;
}

void futcache_persist_nd_destroy(futcache_persist_nd_t *engine)
{
    if (engine == NULL) return;
    futcache_allocator_t *a = &engine->allocator;
    a->deallocate(a->context, engine->points);
    a->deallocate(a->context, engine->obs_to_rep);
    a->deallocate(a->context, engine->radii);
    a->deallocate(a->context, engine->nearest_dist);
    a->deallocate(a->context, engine->birth_index);
    a->deallocate(a->context, engine->birth_prime);
    a->deallocate(a->context, engine->domain_min);
    a->deallocate(a->context, engine->domain_max);
    a->deallocate(a->context, engine);
}

/* ============================================================
 * futcache_persist_nd_observe
 * ============================================================ */

static bool point_in_domain(
    const futcache_persist_nd_t *engine, const double *x)
{
    for (size_t i = 0; i < engine->dimension; ++i) {
        if (!isfinite(x[i]) ||
            x[i] < engine->domain_min[i] - 1e-12 ||
            x[i] > engine->domain_max[i] + 1e-12)
            return false;
    }
    return true;
}

/* Check if x is within epsilon of any existing rep. O(n*d). */
static bool is_within_any_rep(
    const futcache_persist_nd_t *engine, const double *x,
    double radius, size_t *out_nearest_rep, double *out_nearest_dist)
{
    size_t n = engine->rep_count;
    size_t dim = engine->dimension;

    if (n == 0) {
        if (out_nearest_rep) *out_nearest_rep = SIZE_MAX;
        if (out_nearest_dist) *out_nearest_dist = INFINITY;
        return false;
    }

    double best_dist = INFINITY;
    size_t best_idx = SIZE_MAX;
    for (size_t i = 0; i < n; ++i) {
        double d = engine->distance_fn(
            x, engine->points + i * dim, dim, engine->distance_context);
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }

    if (out_nearest_rep) *out_nearest_rep = best_idx;
    if (out_nearest_dist) *out_nearest_dist = best_dist;

    return best_dist <= radius + 1e-12;
}

futcache_status_t futcache_persist_nd_observe(
    futcache_persist_nd_t *engine,
    const double *x,
    bool *out_was_novel)
{
    if (engine == NULL || x == NULL || out_was_novel == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (!point_in_domain(engine, x))
        return FUTCACHE_ERROR_OUT_OF_RANGE;

    /* Check novelty. */
    size_t nearest_rep = SIZE_MAX;
    double nearest_dist = INFINITY;
    bool known = is_within_any_rep(
        engine, x, engine->epsilon, &nearest_rep, &nearest_dist);
    *out_was_novel = !known;

    uint64_t obs_index = engine->observations;

    if (!known) {
        /* Novel: add a new representative. */

        /* Grow points array. */
        if (engine->rep_count >= engine->rep_capacity) {
            size_t new_cap = engine->rep_capacity == 0 ? 16
                                                       : engine->rep_capacity * 2;
            double *new_pts = (double *)engine->allocator.allocate(
                engine->allocator.context,
                new_cap * engine->dimension * sizeof(double));
            if (new_pts == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
            if (engine->rep_count > 0) {
                memcpy(new_pts, engine->points,
                       engine->rep_count * engine->dimension * sizeof(double));
            }
            engine->allocator.deallocate(
                engine->allocator.context, engine->points);
            engine->points = new_pts;
            engine->rep_capacity = new_cap;

            /* Grow per-rep arrays. */
            double *new_radii = (double *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(double));
            double *new_nd = (double *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(double));
            size_t *new_bi = (size_t *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(size_t));
            uint64_t *new_bp = (uint64_t *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(uint64_t));
            if (new_radii == NULL || new_nd == NULL ||
                new_bi == NULL || new_bp == NULL) {
                /* Roll back on OOM: free partial allocations. */
                engine->allocator.deallocate(
                    engine->allocator.context, new_radii);
                engine->allocator.deallocate(
                    engine->allocator.context, new_nd);
                engine->allocator.deallocate(
                    engine->allocator.context, new_bi);
                engine->allocator.deallocate(
                    engine->allocator.context, new_bp);
                return FUTCACHE_ERROR_OUT_OF_MEMORY;
            }
            size_t old_cap = engine->rep_count;
            if (old_cap > 0) {
                memcpy(new_radii, engine->radii,
                       old_cap * sizeof(double));
                memcpy(new_nd, engine->nearest_dist,
                       old_cap * sizeof(double));
                memcpy(new_bi, engine->birth_index,
                       old_cap * sizeof(size_t));
                memcpy(new_bp, engine->birth_prime,
                       old_cap * sizeof(uint64_t));
            }
            engine->allocator.deallocate(
                engine->allocator.context, engine->radii);
            engine->allocator.deallocate(
                engine->allocator.context, engine->nearest_dist);
            engine->allocator.deallocate(
                engine->allocator.context, engine->birth_index);
            engine->allocator.deallocate(
                engine->allocator.context, engine->birth_prime);
            engine->radii = (double *)new_radii;
            engine->nearest_dist = (double *)new_nd;
            engine->birth_index = new_bi;
            engine->birth_prime = new_bp;
        }

        /* Store the new rep. */
        size_t rep_idx = engine->rep_count;
        memcpy(engine->points + rep_idx * engine->dimension, x,
               engine->dimension * sizeof(double));
        engine->radii[rep_idx] = engine->epsilon;
        engine->birth_index[rep_idx] = obs_index;
        engine->birth_prime[rep_idx] = futcache_persist_prime_mod(obs_index);
        engine->nearest_dist[rep_idx] = INFINITY;
        engine->rep_count++;

        /* Track observation -> rep mapping. */
        if (engine->point_count >= engine->point_capacity) {
            size_t new_cap = engine->point_capacity == 0 ? 16
                                                         : engine->point_capacity * 2;
            size_t *new_map = (size_t *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(size_t));
            if (new_map == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
            if (engine->point_count > 0) {
                memcpy(new_map, engine->obs_to_rep,
                       engine->point_count * sizeof(size_t));
            }
            engine->allocator.deallocate(
                engine->allocator.context, engine->obs_to_rep);
            engine->obs_to_rep = new_map;
            engine->point_capacity = new_cap;
        }
        engine->obs_to_rep[engine->point_count++] = rep_idx;

        /* Recompute all nearest distances. O(n^2 * d). */
        recompute_nearest(engine);
    } else {
        /* Redundant observation: track the mapping. */
        if (engine->point_count >= engine->point_capacity) {
            size_t new_cap = engine->point_capacity == 0 ? 16
                                                         : engine->point_capacity * 2;
            size_t *new_map = (size_t *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(size_t));
            if (new_map == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
            if (engine->point_count > 0) {
                memcpy(new_map, engine->obs_to_rep,
                       engine->point_count * sizeof(size_t));
            }
            engine->allocator.deallocate(
                engine->allocator.context, engine->obs_to_rep);
            engine->obs_to_rep = new_map;
            engine->point_capacity = new_cap;
        }
        engine->obs_to_rep[engine->point_count++] = nearest_rep;
    }

    engine->observations++;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_is_novel_at
 * ============================================================ */

futcache_status_t futcache_persist_nd_is_novel_at(
    const futcache_persist_nd_t *engine,
    const double *x,
    double t,
    bool *out_is_novel)
{
    if (engine == NULL || x == NULL || out_is_novel == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (!isfinite(t) || t < 0.0)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    if (engine->rep_count == 0) {
        *out_is_novel = true;
        return FUTCACHE_OK;
    }

    /* Linear scan: check if x is within distance t of any rep. */
    size_t dim = engine->dimension;
    for (size_t i = 0; i < engine->rep_count; ++i) {
        double d = engine->distance_fn(
            x, engine->points + i * dim, dim, engine->distance_context);
        if (d <= t + 1e-12) {
            *out_is_novel = false;
            return FUTCACHE_OK;
        }
    }

    *out_is_novel = true;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_nearest_distances
 * ============================================================ */

futcache_status_t futcache_persist_nd_nearest_distances(
    const futcache_persist_nd_t *engine,
    double *out_distances,
    size_t *inout_count)
{
    if (engine == NULL || inout_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    if (*inout_count < engine->rep_count) {
        *inout_count = engine->rep_count;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out_distances != NULL && engine->rep_count > 0) {
        memcpy(out_distances, engine->nearest_dist,
               engine->rep_count * sizeof(double));
    }
    *inout_count = engine->rep_count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_persistences
 * ============================================================ */

futcache_status_t futcache_persist_nd_persistences(
    const futcache_persist_nd_t *engine,
    double *out_persistences,
    size_t *inout_count)
{
    if (engine == NULL || inout_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    if (*inout_count < engine->rep_count) {
        *inout_count = engine->rep_count;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out_persistences != NULL) {
        for (size_t i = 0; i < engine->rep_count; ++i) {
            out_persistences[i] = engine->nearest_dist[i] - engine->radii[i];
        }
    }
    *inout_count = engine->rep_count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_evict_lowest
 * ============================================================ */

static void shift_reps_down(futcache_persist_nd_t *engine, size_t evicted)
{
    size_t n = engine->rep_count - 1;
    if (evicted >= n) return;

    size_t dim = engine->dimension;
    /* Shift points. */
    memmove(engine->points + evicted * dim,
            engine->points + (evicted + 1) * dim,
            (n - evicted) * dim * sizeof(double));
    /* Shift per-rep arrays. */
    memmove(engine->radii + evicted, engine->radii + evicted + 1,
            (n - evicted) * sizeof(double));
    memmove(engine->nearest_dist + evicted, engine->nearest_dist + evicted + 1,
            (n - evicted) * sizeof(double));
    memmove(engine->birth_index + evicted, engine->birth_index + evicted + 1,
            (n - evicted) * sizeof(size_t));
    memmove(engine->birth_prime + evicted, engine->birth_prime + evicted + 1,
            (n - evicted) * sizeof(uint64_t));

    /* Update obs_to_rep mapping. */
    for (size_t i = 0; i < engine->point_count; ++i) {
        if (engine->obs_to_rep[i] > evicted) {
            engine->obs_to_rep[i]--;
        }
    }
}

futcache_status_t futcache_persist_nd_evict_lowest(
    futcache_persist_nd_t *engine,
    size_t *out_evicted_index)
{
    if (engine == NULL || out_evicted_index == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (engine->rep_count == 0)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    /* Find rep with lowest persistence. */
    size_t worst = 0;
    double worst_persistence = INFINITY;
    for (size_t i = 0; i < engine->rep_count; ++i) {
        double p = engine->nearest_dist[i] - engine->radii[i];
        if (p < worst_persistence) {
            worst_persistence = p;
            worst = i;
        }
    }

    shift_reps_down(engine, worst);
    engine->rep_count--;

    *out_evicted_index = worst;

    /* Recompute nearest distances after eviction. */
    if (engine->rep_count > 0) {
        recompute_nearest(engine);
    }

    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_count_above
 * ============================================================ */

futcache_status_t futcache_persist_nd_count_above(
    const futcache_persist_nd_t *engine,
    double tau,
    size_t *out_count)
{
    if (engine == NULL || out_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t count = 0;
    for (size_t i = 0; i < engine->rep_count; ++i) {
        double p = engine->nearest_dist[i] - engine->radii[i];
        if (p >= tau) count++;
    }
    *out_count = count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_copy_reps
 * ============================================================ */

futcache_status_t futcache_persist_nd_copy_reps(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_rep_t *out_reps,
    size_t *inout_count)
{
    if (engine == NULL || inout_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    if (*inout_count < engine->rep_count) {
        *inout_count = engine->rep_count;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out_reps == NULL) {
        *inout_count = 0;
        return FUTCACHE_OK;
    }

    size_t n = engine->rep_count;
    size_t dim = engine->dimension;

    for (size_t i = 0; i < n; ++i) {
        out_reps[i].id = i;
        out_reps[i].point = (double *)(engine->points + i * dim);
        out_reps[i].radius = engine->radii[i];
        out_reps[i].birth_index = engine->birth_index[i];
        out_reps[i].birth_prime = engine->birth_prime[i];
        out_reps[i].nearest_dist = engine->nearest_dist[i];
        out_reps[i].persistence = engine->nearest_dist[i] - engine->radii[i];
    }

    *inout_count = n;
    return FUTCACHE_OK;
}

void futcache_persist_nd_free_reps(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_rep_t *reps,
    size_t count)
{
    (void)engine;
    (void)reps;
    (void)count;
    /* No-op: the rep structs point into engine memory. The caller's
     * array is stack-allocated. This exists for API symmetry. */
}

/* ============================================================
 * futcache_persist_nd_clear
 * ============================================================ */

futcache_status_t futcache_persist_nd_clear(futcache_persist_nd_t *engine)
{
    if (engine == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    engine->rep_count = 0;
    engine->point_count = 0;
    engine->observations = 0;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_get_stats
 * ============================================================ */

futcache_status_t futcache_persist_nd_get_stats(
    const futcache_persist_nd_t *engine,
    futcache_persist_nd_stats_t *out_stats)
{
    if (engine == NULL || out_stats == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    out_stats->observations = engine->observations;
    out_stats->rep_count = engine->rep_count;
    out_stats->max_persistence = 0.0;
    out_stats->min_persistence = INFINITY;
    out_stats->avg_persistence = 0.0;
    out_stats->prime_birth_count = 0;

    if (engine->rep_count > 0) {
        double max_p = -INFINITY, min_p = INFINITY, sum_p = 0.0;
        size_t prime_count = 0;
        for (size_t i = 0; i < engine->rep_count; ++i) {
            double p = engine->nearest_dist[i] - engine->radii[i];
            if (p > max_p) max_p = p;
            if (p < min_p) min_p = p;
            sum_p += p;
            if (is_prime_uint(engine->birth_index[i])) prime_count++;
        }
        out_stats->max_persistence = max_p;
        out_stats->min_persistence = min_p;
        out_stats->avg_persistence = sum_p / (double)engine->rep_count;
        out_stats->prime_birth_count = prime_count;
    }

    out_stats->memory_bytes = sizeof(futcache_persist_nd_t) +
        engine->rep_count * engine->dimension * sizeof(double) +
        engine->point_count * sizeof(size_t) +
        engine->rep_count * (sizeof(double) + sizeof(double) +
                             sizeof(size_t) + sizeof(uint64_t));
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_nd_validate
 * ============================================================ */

futcache_status_t futcache_persist_nd_validate(
    const futcache_persist_nd_t *engine)
{
    if (engine == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    /* All domain bounds must be valid. */
    for (size_t i = 0; i < engine->dimension; ++i) {
        if (engine->domain_max[i] <= engine->domain_min[i])
            return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* All rep points must be in domain. */
    for (size_t i = 0; i < engine->rep_count; ++i) {
        if (!point_in_domain(engine, engine->points + i * engine->dimension))
            return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* Nearest distances must be non-negative or INFINITY. */
    for (size_t i = 0; i < engine->rep_count; ++i) {
        if (engine->nearest_dist[i] < 0.0 &&
            !isinf(engine->nearest_dist[i]))
            return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* obs_to_rep entries must be valid rep indices or SIZE_MAX. */
    for (size_t i = 0; i < engine->point_count; ++i) {
        size_t r = engine->obs_to_rep[i];
        if (r != SIZE_MAX && r >= engine->rep_count)
            return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    return FUTCACHE_OK;
}

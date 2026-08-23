#define _POSIX_C_SOURCE 200809L

#include "futcache/pack.h"
#include "futcache/futcache.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Default allocator (matches futcache.c pattern)
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

static futcache_status_t normalize_allocator(
    const futcache_allocator_t *requested,
    futcache_allocator_t *normalized)
{
    if (normalized == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = default_allocate;
        normalized->deallocate = default_deallocate;
        normalized->context = NULL;
        return FUTCACHE_OK;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    normalized->allocate = requested->allocate;
    normalized->deallocate = requested->deallocate;
    normalized->context = requested->context;
    return FUTCACHE_OK;
}

/* ============================================================
 * Built-in distance functions
 * ============================================================ */

double futcache_distance_l1(const double *a, const double *b,
                             size_t dimension, void *context)
{
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        double diff = a[i] - b[i];
        if (diff < 0.0) diff = -diff;
        sum += diff;
    }
    return sum;
}

double futcache_distance_l2(const double *a, const double *b,
                             size_t dimension, void *context)
{
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double futcache_distance_linf(const double *a, const double *b,
                               size_t dimension, void *context)
{
    (void)context;
    double max_d = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        double diff = a[i] - b[i];
        if (diff < 0.0) diff = -diff;
        if (diff > max_d) max_d = diff;
    }
    return max_d;
}

/*
 * Cosine distance: 1 - dot(a, b).
 *
 * The inputs are assumed L2-normalized so that dot(a, b) equals cosine
 * similarity and the result lies in [0, 2]. The function does not
 * normalize internally; Bekko and most sentence-transformers output
 * pre-normalized vectors when called with normalize_embeddings=True.
 */
double futcache_distance_cosine(const double *a, const double *b,
                                 size_t dimension, void *context)
{
    (void)context;
    double dot = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0 - dot;
}

/* ============================================================
 * Representative: variable-size struct with inline coordinates.
 * One allocation per representative regardless of dimension.
 * ============================================================ */

typedef struct pack_representative {
    size_t dimension;
    double coordinates[];  /* [dimension] */
} pack_representative_t;

/* Saturating counter increment. Mirrors the helper in futcache.c so the
 * telemetry invariants (generation >= observations, novel <= observations)
 * survive overflow rather than wrap into a state the serializer would
 * reject. */
static uint64_t increment_saturating(uint64_t value)
{
    return value < UINT64_MAX ? value + 1U : value;
}

static pack_representative_t *make_representative(
    const futcache_allocator_t *allocator,
    const double *point,
    size_t dimension)
{
    size_t bytes = sizeof(pack_representative_t) +
                   dimension * sizeof(double);
    pack_representative_t *rep =
        (pack_representative_t *)allocator->allocate(allocator->context, bytes);
    if (rep == NULL) return NULL;
    rep->dimension = dimension;
    memcpy(rep->coordinates, point, dimension * sizeof(double));
    return rep;
}

static void free_representative(
    const futcache_allocator_t *allocator,
    pack_representative_t *rep)
{
    if (rep == NULL) return;
    allocator->deallocate(allocator->context, rep);
}

/* ============================================================
 * Cache object
 * ============================================================ */

struct futcache_pack {
    size_t dimension;
    double epsilon;
    futcache_distance_fn distance;
    void *distance_context;
    futcache_allocator_t allocator;
    const futcache_pack_backend_ops_t *backend;
    void *backend_context;
    void *backend_state;

    /* Domain bounds. Copied at create time, freed at destroy. */
    double *domain_min;
    double *domain_max;

    /* Representative storage: array of pointers to variable-size structs. */
    pack_representative_t **representatives;
    size_t count;
    size_t capacity;
    size_t peak_count;

    pthread_rwlock_t lock;

    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
};

/* ============================================================
 * Public API
 * ============================================================ */

void futcache_pack_config_init(futcache_pack_config_t *config)
{
    if (config == NULL) return;
    memset(config, 0, sizeof(*config));
    config->dimension = 1U;
    config->epsilon = 0.0;
    config->distance = NULL;
    config->distance_context = NULL;
    config->domain_min = NULL;
    config->domain_max = NULL;
    config->backend = NULL;
    config->backend_context = NULL;
}

static bool valid_config(const futcache_pack_config_t *config)
{
    if (config->dimension == 0) return false;
    if (!(config->epsilon >= 0.0)) return false;  /* reject NaN and negative */
    if (config->domain_min == NULL) return false;
    if (config->domain_max == NULL) return false;
    for (size_t i = 0; i < config->dimension; ++i) {
        if (!(config->domain_max[i] > config->domain_min[i])) return false;
        if (!isfinite(config->domain_min[i])) return false;
        if (!isfinite(config->domain_max[i])) return false;
    }
    return true;
}

futcache_status_t futcache_pack_create(
    const futcache_pack_config_t *config,
    futcache_pack_t **out_cache)
{
    if (config == NULL || out_cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!valid_config(config)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (config->backend != NULL &&
        (config->backend->create == NULL || config->backend->destroy == NULL ||
         config->backend->clear == NULL || config->backend->insert == NULL ||
         config->backend->nearest == NULL)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_allocator_t allocator;
    futcache_status_t st = normalize_allocator(&config->allocator, &allocator);
    if (st != FUTCACHE_OK) return st;

    futcache_pack_t *cache = (futcache_pack_t *)allocator.allocate(
        allocator.context, sizeof(*cache));
    if (cache == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;

    memset(cache, 0, sizeof(*cache));
    cache->dimension = config->dimension;
    cache->epsilon = config->epsilon;
    cache->distance = config->distance != NULL
                          ? config->distance
                          : futcache_distance_linf;
    cache->distance_context = config->distance_context;
    cache->allocator = allocator;
    cache->backend = config->backend;
    cache->backend_context = config->backend_context;

    /* Copy domain bounds so the caller may free the source arrays. */
    size_t bounds_bytes = config->dimension * sizeof(double);
    cache->domain_min = (double *)allocator.allocate(allocator.context, bounds_bytes);
    cache->domain_max = (double *)allocator.allocate(allocator.context, bounds_bytes);
    if (cache->domain_min == NULL || cache->domain_max == NULL) {
        if (cache->domain_min != NULL)
            allocator.deallocate(allocator.context, cache->domain_min);
        if (cache->domain_max != NULL)
            allocator.deallocate(allocator.context, cache->domain_max);
        allocator.deallocate(allocator.context, cache);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(cache->domain_min, config->domain_min, bounds_bytes);
    memcpy(cache->domain_max, config->domain_max, bounds_bytes);

    cache->representatives = NULL;
    cache->count = 0;
    cache->capacity = 0;
    cache->peak_count = 0;

    if (pthread_rwlock_init(&cache->lock, NULL) != 0) {
        allocator.deallocate(allocator.context, cache->domain_max);
        allocator.deallocate(allocator.context, cache->domain_min);
        allocator.deallocate(allocator.context, cache);
        return FUTCACHE_ERROR_SYSTEM;
    }

    if (cache->backend != NULL) {
        st = cache->backend->create(&cache->backend_state, cache->dimension,
                                    cache->distance, cache->distance_context,
                                    &cache->allocator, cache->backend_context);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_destroy(&cache->lock);
            allocator.deallocate(allocator.context, cache->domain_max);
            allocator.deallocate(allocator.context, cache->domain_min);
            allocator.deallocate(allocator.context, cache);
            return st;
        }
    }

    *out_cache = cache;
    return FUTCACHE_OK;
}

static void free_all_representatives(futcache_pack_t *cache)
{
    for (size_t i = 0; i < cache->count; ++i) {
        free_representative(&cache->allocator, cache->representatives[i]);
    }
    cache->count = 0;
}

void futcache_pack_destroy(futcache_pack_t *cache)
{
    if (cache == NULL) return;
    /* Caller guarantees quiescence per header. */
    free_all_representatives(cache);
    if (cache->backend != NULL) {
        cache->backend->destroy(cache->backend_state, &cache->allocator,
                                cache->backend_context);
    }
    if (cache->representatives != NULL) {
        cache->allocator.deallocate(cache->allocator.context,
                                     cache->representatives);
    }
    if (cache->domain_min != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_min);
    }
    if (cache->domain_max != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_max);
    }
    pthread_rwlock_destroy(&cache->lock);
    cache->allocator.deallocate(cache->allocator.context, cache);
}

static bool point_in_domain(const futcache_pack_t *cache, const double *point)
{
    for (size_t i = 0; i < cache->dimension; ++i) {
        if (point[i] < cache->domain_min[i]) return false;
        if (point[i] > cache->domain_max[i]) return false;
    }
    return true;
}

static double min_distance_to_set(const futcache_pack_t *cache,
                                   const double *point)
{
    double min_d = INFINITY;
    pack_representative_t *const *reps = cache->representatives;
    futcache_distance_fn distance = cache->distance;
    void *context = cache->distance_context;
    size_t dim = cache->dimension;
    for (size_t i = 0; i < cache->count; ++i) {
        double d = distance(point, reps[i]->coordinates, dim, context);
        if (d < min_d) min_d = d;
    }
    return min_d;
}

static futcache_status_t backend_nearest(const futcache_pack_t *cache,
                                         const double *point,
                                         double *out_distance)
{
    if (cache->backend != NULL) {
        return cache->backend->nearest(cache->backend_state, point,
                                       cache->dimension, out_distance,
                                       cache->backend_context);
    }
    *out_distance = min_distance_to_set(cache, point);
    return FUTCACHE_OK;
}

static futcache_status_t grow_capacity(futcache_pack_t *cache, size_t needed)
{
    if (needed <= cache->capacity) return FUTCACHE_OK;
    size_t new_cap = cache->capacity == 0 ? 16U : cache->capacity;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2U) {
            new_cap = needed;
            break;
        }
        new_cap *= 2U;
    }
    size_t new_bytes = new_cap * sizeof(pack_representative_t *);
    pack_representative_t **new_arr =
        (pack_representative_t **)cache->allocator.allocate(
            cache->allocator.context, new_bytes);
    if (new_arr == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    if (cache->representatives != NULL) {
        if (cache->count > 0) {
            memcpy(new_arr, cache->representatives,
                   cache->count * sizeof(pack_representative_t *));
        }
        cache->allocator.deallocate(cache->allocator.context,
                                     cache->representatives);
    }
    cache->representatives = new_arr;
    cache->capacity = new_cap;
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_is_novel(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_is_novel)
{
    if (cache == NULL || point == NULL || out_is_novel == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    bool novel = true;
    if (cache->count > 0) {
        double d = INFINITY;
        futcache_status_t st = backend_nearest(cache, point, &d);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(lock);
            return st;
        }
        novel = (d > cache->epsilon);
    }

    pthread_rwlock_unlock(lock);
    *out_is_novel = novel;
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_observe(
    futcache_pack_t *cache,
    const double *point,
    bool *out_was_novel)
{
    if (cache == NULL || point == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_wrlock(lock);

    bool novel = true;
    if (cache->count > 0) {
        double d = INFINITY;
        futcache_status_t st = backend_nearest(cache, point, &d);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(lock);
            return st;
        }
        novel = (d > cache->epsilon);
    }

    if (novel) {
        futcache_status_t st = grow_capacity(cache, cache->count + 1U);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(lock);
            return st;
        }
        pack_representative_t *rep = make_representative(
            &cache->allocator, point, cache->dimension);
        if (rep == NULL) {
            pthread_rwlock_unlock(lock);
            return FUTCACHE_ERROR_OUT_OF_MEMORY;
        }
        if (cache->backend != NULL) {
            futcache_status_t backend_st = cache->backend->insert(
                cache->backend_state, point, cache->dimension,
                cache->backend_context);
            if (backend_st != FUTCACHE_OK) {
                free_representative(&cache->allocator, rep);
                pthread_rwlock_unlock(lock);
                return backend_st;
            }
        }
        cache->representatives[cache->count] = rep;
        cache->count++;
        if (cache->count > cache->peak_count) {
            cache->peak_count = cache->count;
        }
    }

    cache->observations = increment_saturating(cache->observations);
    if (novel) cache->novel_observations =
        increment_saturating(cache->novel_observations);
    cache->generation = increment_saturating(cache->generation);

    if (out_was_novel != NULL) *out_was_novel = novel;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_get_stats(
    const futcache_pack_t *cache,
    futcache_pack_stats_t *out_stats)
{
    if (cache == NULL || out_stats == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    out_stats->observations = cache->observations;
    out_stats->novel_observations = cache->novel_observations;
    out_stats->generation = cache->generation;
    out_stats->representative_count = cache->count;
    out_stats->peak_count = cache->peak_count;
    /* Approximate memory: struct, bounds arrays, rep slots, rep coords.
     * Excludes any allocator-internal overhead. */
    out_stats->memory_bytes =
        sizeof(*cache) +
        2U * cache->dimension * sizeof(double) +
        cache->capacity * sizeof(pack_representative_t *) +
        cache->count * (sizeof(pack_representative_t) +
                        cache->dimension * sizeof(double));

    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_copy_representatives(
    const futcache_pack_t *cache,
    double *out_points,
    size_t *inout_count)
{
    if (cache == NULL || inout_count == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    /* inout_count is now uniform: it counts *representatives*, both as
     * the destination capacity and as the required count on query. */
    if (out_points == NULL) {
        *inout_count = cache->count;
        pthread_rwlock_unlock(lock);
        return FUTCACHE_OK;
    }

    if (*inout_count < cache->count) {
        *inout_count = cache->count;
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }

    for (size_t i = 0; i < cache->count; ++i) {
        memcpy(out_points + i * cache->dimension,
               cache->representatives[i]->coordinates,
               cache->dimension * sizeof(double));
    }

    *inout_count = cache->count;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_clear(futcache_pack_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    pthread_rwlock_wrlock(&cache->lock);
    if (cache->backend != NULL) {
        futcache_status_t st = cache->backend->clear(
            cache->backend_state, cache->backend_context);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(&cache->lock);
            return st;
        }
    }
    free_all_representatives(cache);
    cache->peak_count = 0U;
    cache->observations = 0U;
    cache->novel_observations = 0U;
    cache->generation = increment_saturating(cache->generation);
    pthread_rwlock_unlock(&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_validate(const futcache_pack_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    /* Pairwise epsilon-separation. O(n^2) — diagnostic only. */
    futcache_distance_fn distance = cache->distance;
    void *context = cache->distance_context;
    size_t dim = cache->dimension;
    double eps = cache->epsilon;

    for (size_t i = 0; i < cache->count; ++i) {
        if (cache->representatives[i]->dimension != dim) {
            pthread_rwlock_unlock(lock);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        for (size_t j = i + 1U; j < cache->count; ++j) {
            double d = distance(cache->representatives[i]->coordinates,
                                 cache->representatives[j]->coordinates,
                                 dim, context);
            if (!(d > eps)) {  /* reject d <= eps, including NaN */
                pthread_rwlock_unlock(lock);
                return FUTCACHE_ERROR_CORRUPT_DATA;
            }
        }
    }

    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

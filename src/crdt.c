/*
 * crdt.c — deterministic-Voronoi, gossip-mergeable novelty cache.
 *
 * Implements the CRDT engine of PHASE2.md §12.5. State is a fixed-size
 * array of anchor cells forming a join-semilattice: an empty cell is the
 * bottom element, an occupied cell holds (representative point, payload,
 * deterministic priority). Merge adopts into empty cells and keeps the
 * higher priority on conflict, which makes the merge idempotent,
 * commutative, and associative.
 *
 * Thread-safety: observe/merge/clear take the write lock; quantize reads
 * only immutable anchors; snapshot/get_payload/get_stats/validate take the
 * read lock. Pointers returned by snapshot/get_payload alias cache storage
 * and remain valid until the next mutation.
 */

#define _POSIX_C_SOURCE 200809L

#include "futcache/crdt.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Default allocator (matches futcache.c pattern)
 * ============================================================ */

static void *crdt_default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void crdt_default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

static futcache_status_t crdt_normalize_allocator(
    const futcache_allocator_t *requested,
    futcache_allocator_t *normalized)
{
    if (normalized == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = crdt_default_allocate;
        normalized->deallocate = crdt_default_deallocate;
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
 * Deterministic priority (PHASE2.md Algorithm 12.27: pi = H(r || p))
 * ============================================================ */

static uint64_t crdt_hash_bytes(uint64_t hash, const uint8_t *data, size_t n)
{
    /* FNV-1a 64-bit. */
    for (size_t i = 0; i < n; ++i) {
        hash ^= (uint64_t)data[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static uint64_t crdt_priority(const double *point, size_t dimension,
                              const void *payload, size_t payload_length)
{
    uint64_t hash = UINT64_C(0xcbf29ce484222325); /* FNV offset basis */

    /* Encode each coordinate as a canonical little-endian bit pattern so
     * the priority is stable across hosts of either endianness. */
    for (size_t i = 0; i < dimension; ++i) {
        uint64_t bits;
        uint8_t bytes[8];
        memcpy(&bits, &point[i], sizeof(bits));
        for (unsigned b = 0; b < 8U; ++b) {
            bytes[b] = (uint8_t)(bits >> (8U * b));
        }
        hash = crdt_hash_bytes(hash, bytes, 8U);
    }
    if (payload != NULL && payload_length > 0U) {
        hash = crdt_hash_bytes(hash, (const uint8_t *)payload, payload_length);
    }
    return hash;
}

/* ============================================================
 * Cache object
 * ============================================================ */

typedef struct crdt_entry {
    bool occupied;
    double *point;         /* [dimension], owned */
    unsigned char *payload;/* [payload_length], owned */
    size_t payload_length;
    uint64_t priority;
} crdt_entry_t;

struct futcache_crdt {
    size_t dimension;
    size_t anchor_count;
    double *anchors;        /* [anchor_count * dimension], owned */
    double epsilon;
    futcache_distance_fn distance;
    void *distance_context;
    double *domain_min;     /* owned */
    double *domain_max;     /* owned */
    futcache_allocator_t allocator;
    crdt_entry_t *cells;    /* [anchor_count] */
    size_t occupied_cells;
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    /* Approximate bytes owned by the cache; recomputed on clear. */
    size_t memory_bytes;
    pthread_rwlock_t lock;
};

/* Saturating increment so telemetry invariants survive overflow. */
static uint64_t crdt_saturating(uint64_t value)
{
    return value < UINT64_MAX ? value + 1U : value;
}

static bool crdt_checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0U && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool crdt_point_valid(const double *point, size_t dimension)
{
    if (point == NULL) return false;
    for (size_t i = 0; i < dimension; ++i) {
        if (!isfinite(point[i])) return false;
    }
    return true;
}

static bool crdt_point_in_domain(const futcache_crdt_t *cache,
                                 const double *point)
{
    for (size_t i = 0; i < cache->dimension; ++i) {
        if (point[i] < cache->domain_min[i] ||
            point[i] > cache->domain_max[i]) {
            return false;
        }
    }
    return true;
}

/* Quantizes against immutable anchors; no lock needed. */
static size_t crdt_quantize_impl(const futcache_crdt_t *cache,
                                 const double *point)
{
    futcache_distance_fn distance = cache->distance;
    void *context = cache->distance_context;
    size_t dim = cache->dimension;
    size_t best = 0U;
    double best_d = distance(point, cache->anchors, dim, context);
    for (size_t a = 1U; a < cache->anchor_count; ++a) {
        double d = distance(point, cache->anchors + a * dim, dim, context);
        if (d < best_d) {  /* strict: ties keep the smallest index */
            best_d = d;
            best = a;
        }
    }
    return best;
}

static bool crdt_config_is_valid(const futcache_crdt_config_t *config)
{
    if (config->dimension == 0U || config->anchor_count == 0U) return false;
    if (config->anchors == NULL) return false;
    if (!(config->epsilon >= 0.0)) return false;      /* NaN or negative */
    if (!isfinite(config->epsilon)) return false;
    if (config->domain_min == NULL || config->domain_max == NULL) return false;
    for (size_t i = 0; i < config->dimension; ++i) {
        if (!isfinite(config->domain_min[i]) ||
            !isfinite(config->domain_max[i])) {
            return false;
        }
        if (config->domain_max[i] <= config->domain_min[i]) return false;
    }
    /* Anchors are a delta-net of the domain: finite and inside bounds. */
    for (size_t a = 0; a < config->anchor_count; ++a) {
        for (size_t i = 0; i < config->dimension; ++i) {
            double v = config->anchors[a * config->dimension + i];
            if (!isfinite(v)) return false;
            if (v < config->domain_min[i] || v > config->domain_max[i]) {
                return false;
            }
        }
    }
    return true;
}

/* Frees every occupied cell entry; leaves `cells` allocated. */
static void crdt_free_entries(futcache_crdt_t *cache)
{
    if (cache->cells == NULL) return;
    for (size_t i = 0; i < cache->anchor_count; ++i) {
        crdt_entry_t *e = &cache->cells[i];
        if (e->occupied) {
            if (e->point != NULL) {
                cache->allocator.deallocate(cache->allocator.context, e->point);
            }
            if (e->payload != NULL) {
                cache->allocator.deallocate(cache->allocator.context,
                                            e->payload);
            }
            e->occupied = false;
            e->point = NULL;
            e->payload = NULL;
            e->payload_length = 0U;
            e->priority = 0U;
        }
    }
}

/* Frees storage excluding the lock; used on create-failure paths before
 * the lock exists and by destroy after it has been released. */
static void crdt_free_storage(futcache_crdt_t *cache)
{
    crdt_free_entries(cache);
    if (cache->anchors != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->anchors);
    }
    if (cache->domain_min != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_min);
    }
    if (cache->domain_max != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_max);
    }
    if (cache->cells != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->cells);
    }
    cache->allocator.deallocate(cache->allocator.context, cache);
}

/* ============================================================
 * Public API
 * ============================================================ */

void futcache_crdt_config_init(futcache_crdt_config_t *config)
{
    if (config == NULL) return;
    memset(config, 0, sizeof(*config));
    config->dimension = 1U;
    config->anchor_count = 1U;
    config->anchors = NULL;
    config->epsilon = 0.0;
    config->distance = NULL;
    config->distance_context = NULL;
    config->domain_min = NULL;
    config->domain_max = NULL;
}

futcache_status_t futcache_crdt_create(const futcache_crdt_config_t *config,
                                       futcache_crdt_t **out_cache)
{
    if (config == NULL || out_cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_cache = NULL;
    if (!crdt_config_is_valid(config)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_allocator_t allocator;
    futcache_status_t st = crdt_normalize_allocator(&config->allocator,
                                                    &allocator);
    if (st != FUTCACHE_OK) return st;

    size_t anchor_elems;
    size_t anchors_bytes;
    size_t cells_bytes;
    if (!crdt_checked_mul(config->anchor_count, config->dimension,
                          &anchor_elems) ||
        !crdt_checked_mul(anchor_elems, sizeof(double), &anchors_bytes) ||
        !crdt_checked_mul(config->anchor_count, sizeof(crdt_entry_t),
                          &cells_bytes)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    size_t bounds_bytes = config->dimension * sizeof(double);

    futcache_crdt_t *cache = allocator.allocate(allocator.context,
                                                sizeof(*cache));
    if (cache == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    memset(cache, 0, sizeof(*cache));
    cache->dimension = config->dimension;
    cache->anchor_count = config->anchor_count;
    cache->epsilon = config->epsilon;
    cache->distance = config->distance != NULL
                          ? config->distance
                          : futcache_distance_linf;
    cache->distance_context = config->distance_context;
    cache->allocator = allocator;
    cache->anchors = NULL;
    cache->domain_min = NULL;
    cache->domain_max = NULL;
    cache->cells = NULL;

    cache->anchors = allocator.allocate(allocator.context, anchors_bytes);
    cache->domain_min = allocator.allocate(allocator.context, bounds_bytes);
    cache->domain_max = allocator.allocate(allocator.context, bounds_bytes);
    cache->cells = allocator.allocate(allocator.context, cells_bytes);
    if (cache->anchors == NULL || cache->domain_min == NULL ||
        cache->domain_max == NULL || cache->cells == NULL) {
        crdt_free_storage(cache);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(cache->anchors, config->anchors, anchors_bytes);
    memcpy(cache->domain_min, config->domain_min, bounds_bytes);
    memcpy(cache->domain_max, config->domain_max, bounds_bytes);
    memset(cache->cells, 0, cells_bytes);

    cache->memory_bytes = sizeof(*cache) + anchors_bytes +
                          2U * bounds_bytes + cells_bytes;

    if (pthread_rwlock_init(&cache->lock, NULL) != 0) {
        crdt_free_storage(cache);
        return FUTCACHE_ERROR_SYSTEM;
    }

    *out_cache = cache;
    return FUTCACHE_OK;
}

void futcache_crdt_destroy(futcache_crdt_t *cache)
{
    if (cache == NULL) return;
    futcache_allocator_t allocator = cache->allocator;
    crdt_free_entries(cache);
    if (cache->anchors != NULL) {
        allocator.deallocate(allocator.context, cache->anchors);
    }
    if (cache->domain_min != NULL) {
        allocator.deallocate(allocator.context, cache->domain_min);
    }
    if (cache->domain_max != NULL) {
        allocator.deallocate(allocator.context, cache->domain_max);
    }
    if (cache->cells != NULL) {
        allocator.deallocate(allocator.context, cache->cells);
    }
    (void)pthread_rwlock_destroy(&cache->lock);
    allocator.deallocate(allocator.context, cache);
}

futcache_status_t futcache_crdt_quantize(const futcache_crdt_t *cache,
                                         const double *point,
                                         size_t *out_cell)
{
    if (cache == NULL || out_cell == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!crdt_point_valid(point, cache->dimension)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!crdt_point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    *out_cell = crdt_quantize_impl(cache, point);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_observe(futcache_crdt_t *cache,
                                        const double *point,
                                        const void *payload,
                                        size_t payload_length,
                                        bool *out_was_novel,
                                        size_t *out_cell)
{
    if (cache == NULL || point == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (payload_length > 0U && payload == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!crdt_point_valid(point, cache->dimension)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!crdt_point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    size_t cell = crdt_quantize_impl(cache, point);

    if (pthread_rwlock_wrlock(&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }
    if (out_cell != NULL) *out_cell = cell;

    crdt_entry_t *entry = &cache->cells[cell];
    bool was_novel;
    if (entry->occupied) {
        was_novel = false;
    } else {
        /* Allocate before committing so failure is atomic. */
        double *pt = cache->allocator.allocate(cache->allocator.context,
                                               cache->dimension * sizeof(double));
        if (pt == NULL) {
            pthread_rwlock_unlock(&cache->lock);
            return FUTCACHE_ERROR_OUT_OF_MEMORY;
        }
        memcpy(pt, point, cache->dimension * sizeof(double));
        unsigned char *pl = NULL;
        if (payload_length > 0U) {
            pl = cache->allocator.allocate(cache->allocator.context,
                                           payload_length);
            if (pl == NULL) {
                cache->allocator.deallocate(cache->allocator.context, pt);
                pthread_rwlock_unlock(&cache->lock);
                return FUTCACHE_ERROR_OUT_OF_MEMORY;
            }
            memcpy(pl, payload, payload_length);
        }
        entry->point = pt;
        entry->payload = pl;
        entry->payload_length = payload_length;
        entry->priority = crdt_priority(point, cache->dimension,
                                        payload, payload_length);
        entry->occupied = true;
        cache->occupied_cells++;
        cache->novel_observations =
            crdt_saturating(cache->novel_observations);
        cache->memory_bytes += cache->dimension * sizeof(double) +
                               payload_length;
        was_novel = true;
    }
    cache->observations = crdt_saturating(cache->observations);
    cache->generation = crdt_saturating(cache->generation);

    if (out_was_novel != NULL) *out_was_novel = was_novel;
    pthread_rwlock_unlock(&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_merge(futcache_crdt_t *cache,
                                      const futcache_crdt_update_t *updates,
                                      size_t update_count)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (update_count > 0U && updates == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    /* Validate the whole batch before mutating anything. */
    for (size_t u = 0; u < update_count; ++u) {
        const futcache_crdt_update_t *upd = &updates[u];
        if (upd->cell >= cache->anchor_count || upd->point == NULL ||
            (upd->payload_length > 0U && upd->payload == NULL)) {
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
        }
        if (!crdt_point_valid(upd->point, cache->dimension)) {
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
        }
        if (!crdt_point_in_domain(cache, upd->point)) {
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
    }

    if (pthread_rwlock_wrlock(&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }

    for (size_t u = 0; u < update_count; ++u) {
        const futcache_crdt_update_t *upd = &updates[u];
        crdt_entry_t *entry = &cache->cells[upd->cell];

        if (entry->occupied && upd->priority <= entry->priority) {
            continue;  /* keep local entry (equal or higher priority) */
        }

        /* Allocate before committing so a failed adopt/replace does not
         * corrupt the cell. */
        double *pt = cache->allocator.allocate(cache->allocator.context,
                                               cache->dimension * sizeof(double));
        if (pt == NULL) {
            pthread_rwlock_unlock(&cache->lock);
            return FUTCACHE_ERROR_OUT_OF_MEMORY;
        }
        memcpy(pt, upd->point, cache->dimension * sizeof(double));
        unsigned char *pl = NULL;
        if (upd->payload_length > 0U) {
            pl = cache->allocator.allocate(cache->allocator.context,
                                           upd->payload_length);
            if (pl == NULL) {
                cache->allocator.deallocate(cache->allocator.context, pt);
                pthread_rwlock_unlock(&cache->lock);
                return FUTCACHE_ERROR_OUT_OF_MEMORY;
            }
            memcpy(pl, upd->payload, upd->payload_length);
        }

        if (entry->occupied) {
            cache->memory_bytes -= cache->dimension * sizeof(double) +
                                   entry->payload_length;
            cache->allocator.deallocate(cache->allocator.context, entry->point);
            if (entry->payload != NULL) {
                cache->allocator.deallocate(cache->allocator.context,
                                            entry->payload);
            }
        } else {
            cache->occupied_cells++;
        }
        entry->point = pt;
        entry->payload = pl;
        entry->payload_length = upd->payload_length;
        entry->priority = upd->priority;
        entry->occupied = true;
        cache->memory_bytes += cache->dimension * sizeof(double) +
                               upd->payload_length;
    }

    cache->generation = crdt_saturating(cache->generation);
    pthread_rwlock_unlock(&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_snapshot(const futcache_crdt_t *cache,
                                         futcache_crdt_update_t *out_updates,
                                         size_t *inout_count)
{
    if (cache == NULL || inout_count == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    size_t capacity = *inout_count;
    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    if (pthread_rwlock_rdlock(lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }
    size_t required = cache->occupied_cells;
    *inout_count = required;

    if (out_updates == NULL) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_OK;
    }

    if (capacity < required) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }

    size_t written = 0U;
    for (size_t i = 0; i < cache->anchor_count; ++i) {
        crdt_entry_t *e = &cache->cells[i];
        if (e->occupied) {
            out_updates[written].cell = i;
            out_updates[written].point = e->point;
            out_updates[written].payload = e->payload;
            out_updates[written].payload_length = e->payload_length;
            out_updates[written].priority = e->priority;
            written++;
        }
    }

    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_get_payload(const futcache_crdt_t *cache,
                                            size_t cell,
                                            const void **out_payload,
                                            size_t *out_payload_length)
{
    if (cache == NULL || out_payload == NULL || out_payload_length == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (cell >= cache->anchor_count) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    if (pthread_rwlock_rdlock((pthread_rwlock_t *)&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }
    const crdt_entry_t *e = &cache->cells[cell];
    if (e->occupied) {
        *out_payload = e->payload;
        *out_payload_length = e->payload_length;
    } else {
        *out_payload = NULL;
        *out_payload_length = 0U;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_get_stats(const futcache_crdt_t *cache,
                                          futcache_crdt_stats_t *out_stats)
{
    if (cache == NULL || out_stats == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (pthread_rwlock_rdlock((pthread_rwlock_t *)&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }
    out_stats->observations = cache->observations;
    out_stats->novel_observations = cache->novel_observations;
    out_stats->generation = cache->generation;
    out_stats->occupied_cells = cache->occupied_cells;
    out_stats->memory_bytes = cache->memory_bytes;
    pthread_rwlock_unlock((pthread_rwlock_t *)&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_clear(futcache_crdt_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (pthread_rwlock_wrlock(&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }
    crdt_free_entries(cache);
    cache->occupied_cells = 0U;
    cache->observations = 0U;
    cache->novel_observations = 0U;
    cache->generation = crdt_saturating(cache->generation);
    cache->memory_bytes = sizeof(*cache) +
                          cache->anchor_count * cache->dimension *
                              sizeof(double) +
                          2U * cache->dimension * sizeof(double) +
                          cache->anchor_count * sizeof(crdt_entry_t);
    pthread_rwlock_unlock(&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_crdt_validate(const futcache_crdt_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (pthread_rwlock_rdlock((pthread_rwlock_t *)&cache->lock) != 0) {
        return FUTCACHE_ERROR_SYSTEM;
    }

    futcache_status_t status = FUTCACHE_OK;

    /* Telemetry lifecycle invariants. */
    if (cache->generation < cache->observations ||
        cache->novel_observations > cache->observations ||
        cache->occupied_cells > cache->anchor_count ||
        cache->occupied_cells < cache->novel_observations) {
        status = FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* Structural invariants of occupied cells. */
    size_t occupied = 0U;
    for (size_t i = 0; i < cache->anchor_count && status == FUTCACHE_OK; ++i) {
        const crdt_entry_t *e = &cache->cells[i];
        if (!e->occupied) continue;
        occupied++;
        if (e->point == NULL ||
            (e->payload_length > 0U && e->payload == NULL) ||
            !crdt_point_valid(e->point, cache->dimension) ||
            !crdt_point_in_domain(cache, e->point)) {
            status = FUTCACHE_ERROR_CORRUPT_DATA;
        }
    }
    if (status == FUTCACHE_OK && occupied != cache->occupied_cells) {
        status = FUTCACHE_ERROR_CORRUPT_DATA;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&cache->lock);
    return status;
}

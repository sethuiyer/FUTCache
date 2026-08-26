#define _POSIX_C_SOURCE 200809L

#include "futcache/pack.h"
#include "futcache/futcache.h"
#include "pack_vptree_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

enum {
    FUTCACHE_PACK_SERIAL_HEADER_SIZE = 104,
    FUTCACHE_PACK_SERIAL_CRC_SIZE = 4,
    FUTCACHE_PACK_SERIAL_VERSION_LEGACY = 1,
    FUTCACHE_PACK_SERIAL_VERSION = 2,
    FUTCACHE_PACK_SERIAL_FLAG_RADII = 1,
    FUTCACHE_PACK_DISTANCE_LINF = 1,
    FUTCACHE_PACK_DISTANCE_L1 = 2,
    FUTCACHE_PACK_DISTANCE_L2 = 3,
    FUTCACHE_PACK_DISTANCE_COSINE = 4,
    FUTCACHE_PACK_DISTANCE_POINCARE = 5,
    FUTCACHE_PACK_BACKEND_LINEAR = 0,
    FUTCACHE_PACK_BACKEND_VPTREE = 1
};

static const uint8_t futcache_pack_serial_magic[8] = {
    (uint8_t)'F', (uint8_t)'U', (uint8_t)'T', (uint8_t)'P',
    (uint8_t)'A', (uint8_t)'C', (uint8_t)'K', UINT8_C(0)
};

/* Prefix every child allocation so exact live-byte accounting is possible
 * even with a caller-supplied allocator.  The union preserves max_align_t
 * alignment for the returned payload. */
typedef union pack_allocation_header {
    max_align_t alignment;
    size_t total_bytes;
} pack_allocation_header_t;

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

double futcache_distance_poincare(const double *a, const double *b,
                                   size_t dimension, void *context)
{
    (void)context;
    double norm_a_squared = 0.0;
    double norm_b_squared = 0.0;
    double difference_squared = 0.0;
    for (size_t i = 0U; i < dimension; ++i) {
        double difference = a[i] - b[i];
        norm_a_squared += a[i] * a[i];
        norm_b_squared += b[i] * b[i];
        difference_squared += difference * difference;
    }
    if (!(norm_a_squared < 1.0) || !(norm_b_squared < 1.0) ||
        !isfinite(difference_squared)) {
        return NAN;
    }
    double denominator = sqrt((1.0 - norm_a_squared) *
                              (1.0 - norm_b_squared));
    if (!(denominator > 0.0)) return NAN;
    return 2.0 * asinh(sqrt(difference_squared) / denominator);
}

/* ============================================================
 * Representative: variable-size struct with inline coordinates.
 * One allocation per representative regardless of dimension.
 * ============================================================ */

typedef struct pack_representative {
    struct pack_representative *next;
    double radius;
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
    size_t dimension,
    double radius)
{
    if (dimension > (SIZE_MAX - sizeof(pack_representative_t)) /
                        sizeof(double)) {
        return NULL;
    }
    size_t bytes = sizeof(pack_representative_t) +
                   dimension * sizeof(double);
    pack_representative_t *rep =
        (pack_representative_t *)allocator->allocate(allocator->context, bytes);
    if (rep == NULL) return NULL;
    rep->next = NULL;
    rep->radius = radius;
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
    /* owner_allocator owns this object; allocator is the accounting wrapper
     * used for every child/backend allocation. */
    futcache_allocator_t owner_allocator;
    futcache_allocator_t allocator;
    const futcache_pack_backend_ops_t *backend;
    void *backend_context;
    void *backend_state;
    bool backend_active;
    bool variable_radii;

    /* Domain bounds. Copied at create time, freed at destroy. */
    double *domain_min;
    double *domain_max;

    /* FIFO-linked representatives.  Recycling head under pressure avoids a
     * transient allocation above the configured hard ceiling. */
    pack_representative_t *representatives;
    pack_representative_t *representatives_tail;
    size_t count;
    size_t peak_count;

    pthread_rwlock_t lock;

    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    uint64_t evictions;

    size_t memory_limit_bytes;
    atomic_size_t memory_bytes;
    atomic_size_t peak_memory_bytes;
};

/* ============================================================
 * Hard-limit allocation accounting
 * ============================================================ */

static bool allocation_total_size(size_t payload_bytes, size_t *out_total)
{
    if (out_total == NULL ||
        payload_bytes > SIZE_MAX - sizeof(pack_allocation_header_t)) {
        return false;
    }
    *out_total = sizeof(pack_allocation_header_t) + payload_bytes;
    return true;
}

static bool reserve_allocation_bytes(futcache_pack_t *cache, size_t bytes)
{
    size_t current = atomic_load_explicit(&cache->memory_bytes,
                                          memory_order_relaxed);
    for (;;) {
        if (current > SIZE_MAX - bytes) return false;
        size_t next = current + bytes;
        if (cache->memory_limit_bytes != 0U &&
            next > cache->memory_limit_bytes) {
            return false;
        }
        if (atomic_compare_exchange_weak_explicit(
                &cache->memory_bytes, &current, next,
                memory_order_acq_rel, memory_order_relaxed)) {
            return true;
        }
    }
}

static void update_peak_memory(futcache_pack_t *cache)
{
    size_t current = atomic_load_explicit(&cache->memory_bytes,
                                          memory_order_relaxed);
    size_t peak = atomic_load_explicit(&cache->peak_memory_bytes,
                                       memory_order_relaxed);
    while (peak < current &&
           !atomic_compare_exchange_weak_explicit(
               &cache->peak_memory_bytes, &peak, current,
               memory_order_relaxed, memory_order_relaxed)) {
        /* compare_exchange refreshes peak */
    }
}

static void *tracked_allocate(void *context, size_t size)
{
    futcache_pack_t *cache = (futcache_pack_t *)context;
    pack_allocation_header_t *header;
    size_t total;

    if (cache == NULL || !allocation_total_size(size, &total) ||
        !reserve_allocation_bytes(cache, total)) {
        return NULL;
    }
    header = (pack_allocation_header_t *)cache->owner_allocator.allocate(
        cache->owner_allocator.context, total);
    if (header == NULL) {
        (void)atomic_fetch_sub_explicit(&cache->memory_bytes, total,
                                        memory_order_acq_rel);
        return NULL;
    }
    header->total_bytes = total;
    update_peak_memory(cache);
    return (void *)(header + 1);
}

static void tracked_deallocate(void *context, void *pointer)
{
    futcache_pack_t *cache = (futcache_pack_t *)context;
    pack_allocation_header_t *header;
    size_t total;

    if (cache == NULL || pointer == NULL) return;
    header = ((pack_allocation_header_t *)pointer) - 1;
    total = header->total_bytes;
    cache->owner_allocator.deallocate(cache->owner_allocator.context, header);
    (void)atomic_fetch_sub_explicit(&cache->memory_bytes, total,
                                    memory_order_acq_rel);
}

static bool tracked_allocation_fits(const futcache_pack_t *cache,
                                    size_t payload_bytes)
{
    size_t total;
    size_t current;
    if (!allocation_total_size(payload_bytes, &total)) return false;
    current = atomic_load_explicit(&cache->memory_bytes, memory_order_relaxed);
    if (current > SIZE_MAX - total) return false;
    return cache->memory_limit_bytes == 0U ||
           current + total <= cache->memory_limit_bytes;
}

/* ============================================================
 * Snapshot encoding helpers
 * ============================================================ */

static bool pack_double_serialization_supported(void)
{
    return CHAR_BIT == 8 && sizeof(double) == sizeof(uint64_t) &&
        FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024;
}

static void pack_write_u16_le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)(value & UINT16_C(0xff));
    destination[1] = (uint8_t)((value >> 8U) & UINT16_C(0xff));
}

static void pack_write_u32_le(uint8_t *destination, uint32_t value)
{
    for (size_t i = 0U; i < 4U; ++i) {
        destination[i] = (uint8_t)((value >> (i * 8U)) & UINT32_C(0xff));
    }
}

static void pack_write_u64_le(uint8_t *destination, uint64_t value)
{
    for (size_t i = 0U; i < 8U; ++i) {
        destination[i] = (uint8_t)((value >> (i * 8U)) & UINT64_C(0xff));
    }
}

static uint16_t pack_read_u16_le(const uint8_t *source)
{
    return (uint16_t)((uint16_t)source[0] |
                      ((uint16_t)source[1] << 8U));
}

static uint32_t pack_read_u32_le(const uint8_t *source)
{
    uint32_t value = UINT32_C(0);
    for (size_t i = 0U; i < 4U; ++i) {
        value |= (uint32_t)source[i] << (i * 8U);
    }
    return value;
}

static uint64_t pack_read_u64_le(const uint8_t *source)
{
    uint64_t value = UINT64_C(0);
    for (size_t i = 0U; i < 8U; ++i) {
        value |= (uint64_t)source[i] << (i * 8U);
    }
    return value;
}

static void pack_write_double_le(uint8_t *destination, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    pack_write_u64_le(destination, bits);
}

static double pack_read_double_le(const uint8_t *source)
{
    uint64_t bits = pack_read_u64_le(source);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t pack_crc32_bytes(const uint8_t *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    for (size_t i = 0U; i < size; ++i) {
        crc ^= (uint32_t)data[i];
        for (unsigned int bit = 0U; bit < 8U; ++bit) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & UINT32_C(1)));
            crc = (crc >> 1U) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return ~crc;
}

static uint32_t pack_distance_identifier(futcache_distance_fn distance)
{
    if (distance == futcache_distance_linf) return FUTCACHE_PACK_DISTANCE_LINF;
    if (distance == futcache_distance_l1) return FUTCACHE_PACK_DISTANCE_L1;
    if (distance == futcache_distance_l2) return FUTCACHE_PACK_DISTANCE_L2;
    if (distance == futcache_distance_cosine) {
        return FUTCACHE_PACK_DISTANCE_COSINE;
    }
    if (distance == futcache_distance_poincare) {
        return FUTCACHE_PACK_DISTANCE_POINCARE;
    }
    return UINT32_C(0);
}

static futcache_distance_fn pack_distance_from_identifier(uint32_t identifier)
{
    switch (identifier) {
    case FUTCACHE_PACK_DISTANCE_LINF:
        return futcache_distance_linf;
    case FUTCACHE_PACK_DISTANCE_L1:
        return futcache_distance_l1;
    case FUTCACHE_PACK_DISTANCE_L2:
        return futcache_distance_l2;
    case FUTCACHE_PACK_DISTANCE_COSINE:
        return futcache_distance_cosine;
    case FUTCACHE_PACK_DISTANCE_POINCARE:
        return futcache_distance_poincare;
    default:
        return NULL;
    }
}

static bool pack_snapshot_size(size_t dimension, size_t count,
                               bool has_radii, size_t *out_size)
{
    size_t vector_bytes;
    size_t vector_count;
    size_t radii_bytes = 0U;
    size_t fixed = FUTCACHE_PACK_SERIAL_HEADER_SIZE +
                   FUTCACHE_PACK_SERIAL_CRC_SIZE;
    if (out_size == NULL || dimension > SIZE_MAX / sizeof(double) ||
        count > SIZE_MAX - 2U) {
        return false;
    }
    vector_bytes = dimension * sizeof(double);
    vector_count = count + 2U;
    if (vector_bytes != 0U &&
        vector_count > (SIZE_MAX - fixed) / vector_bytes) {
        return false;
    }
    if (has_radii) {
        if (count > (SIZE_MAX - fixed - vector_count * vector_bytes) /
                        sizeof(double)) {
            return false;
        }
        radii_bytes = count * sizeof(double);
    }
    *out_size = fixed + vector_count * vector_bytes + radii_bytes;
    return true;
}

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
    config->max_memory_bytes = 0U;
}

static bool valid_config(const futcache_pack_config_t *config)
{
    if (config->dimension == 0) return false;
    if (!isfinite(config->epsilon) || config->epsilon < 0.0) return false;
    if (config->dimension >
        (SIZE_MAX - sizeof(pack_representative_t)) / sizeof(double)) {
        return false;
    }
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
    *out_cache = NULL;
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

    size_t bounds_bytes = config->dimension * sizeof(double);
    size_t domain_allocation_bytes;
    size_t representative_allocation_bytes;
    if (!allocation_total_size(bounds_bytes, &domain_allocation_bytes) ||
        !allocation_total_size(sizeof(pack_representative_t) + bounds_bytes,
                               &representative_allocation_bytes) ||
        domain_allocation_bytes >
            (SIZE_MAX - sizeof(futcache_pack_t)) / 2U ||
        sizeof(futcache_pack_t) + 2U * domain_allocation_bytes >
            SIZE_MAX - representative_allocation_bytes) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    size_t minimum_useful_bytes = sizeof(futcache_pack_t) +
        2U * domain_allocation_bytes + representative_allocation_bytes;
    if (config->max_memory_bytes != 0U &&
        config->max_memory_bytes < minimum_useful_bytes) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

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
    cache->owner_allocator = allocator;
    cache->allocator.allocate = tracked_allocate;
    cache->allocator.deallocate = tracked_deallocate;
    cache->allocator.context = cache;
    cache->backend = config->backend;
    cache->backend_context = config->backend_context;
    cache->backend_active = false;
    cache->variable_radii = false;
    cache->memory_limit_bytes = config->max_memory_bytes;
    atomic_init(&cache->memory_bytes, sizeof(*cache));
    atomic_init(&cache->peak_memory_bytes, sizeof(*cache));

    /* Copy domain bounds so the caller may free the source arrays. */
    cache->domain_min = (double *)cache->allocator.allocate(
        cache->allocator.context, bounds_bytes);
    cache->domain_max = (double *)cache->allocator.allocate(
        cache->allocator.context, bounds_bytes);
    if (cache->domain_min == NULL || cache->domain_max == NULL) {
        if (cache->domain_min != NULL)
            cache->allocator.deallocate(cache->allocator.context,
                                        cache->domain_min);
        if (cache->domain_max != NULL)
            cache->allocator.deallocate(cache->allocator.context,
                                        cache->domain_max);
        allocator.deallocate(allocator.context, cache);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(cache->domain_min, config->domain_min, bounds_bytes);
    memcpy(cache->domain_max, config->domain_max, bounds_bytes);

    cache->representatives = NULL;
    cache->representatives_tail = NULL;
    cache->count = 0;
    cache->peak_count = 0;

    if (pthread_rwlock_init(&cache->lock, NULL) != 0) {
        cache->allocator.deallocate(cache->allocator.context,
                                    cache->domain_max);
        cache->allocator.deallocate(cache->allocator.context,
                                    cache->domain_min);
        allocator.deallocate(allocator.context, cache);
        return FUTCACHE_ERROR_SYSTEM;
    }

    if (cache->backend != NULL) {
        st = cache->backend->create(&cache->backend_state, cache->dimension,
                                    cache->distance, cache->distance_context,
                                    &cache->allocator, cache->backend_context);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_destroy(&cache->lock);
            cache->allocator.deallocate(cache->allocator.context,
                                        cache->domain_max);
            cache->allocator.deallocate(cache->allocator.context,
                                        cache->domain_min);
            allocator.deallocate(allocator.context, cache);
            return st;
        }
        cache->backend_active = true;
    }

    *out_cache = cache;
    return FUTCACHE_OK;
}

static void free_all_representatives(futcache_pack_t *cache)
{
    pack_representative_t *representative = cache->representatives;
    while (representative != NULL) {
        pack_representative_t *next = representative->next;
        free_representative(&cache->allocator, representative);
        representative = next;
    }
    cache->representatives = NULL;
    cache->representatives_tail = NULL;
    cache->count = 0;
}

void futcache_pack_destroy(futcache_pack_t *cache)
{
    if (cache == NULL) return;
    /* Caller guarantees quiescence per header. */
    if (cache->backend_active) {
        cache->backend->destroy(cache->backend_state, &cache->allocator,
                                cache->backend_context);
    }
    free_all_representatives(cache);
    cache->variable_radii = false;
    if (cache->domain_min != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_min);
    }
    if (cache->domain_max != NULL) {
        cache->allocator.deallocate(cache->allocator.context, cache->domain_max);
    }
    pthread_rwlock_destroy(&cache->lock);
    cache->owner_allocator.deallocate(cache->owner_allocator.context, cache);
}

static bool point_in_domain(const futcache_pack_t *cache, const double *point)
{
    double norm_squared = 0.0;
    for (size_t i = 0; i < cache->dimension; ++i) {
        if (!isfinite(point[i])) return false;
        if (point[i] < cache->domain_min[i]) return false;
        if (point[i] > cache->domain_max[i]) return false;
        if (cache->distance == futcache_distance_poincare) {
            norm_squared += point[i] * point[i];
        }
    }
    if (cache->distance == futcache_distance_poincare &&
        !(norm_squared < 1.0)) return false;
    return true;
}

static void min_covering_representative(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_found,
    double *out_distance,
    size_t *out_index)
{
    bool found = false;
    double best_distance = INFINITY;
    size_t best_index = SIZE_MAX;
    const pack_representative_t *representative = cache->representatives;
    size_t index = 0U;
    while (representative != NULL) {
        double distance = cache->distance(
            point, representative->coordinates, cache->dimension,
            cache->distance_context);
        if (distance <= representative->radius &&
            (!found || distance < best_distance ||
             (distance == best_distance && index < best_index))) {
            found = true;
            best_distance = distance;
            best_index = index;
        }
        representative = representative->next;
        ++index;
    }
    *out_found = found;
    *out_distance = best_distance;
    *out_index = best_index;
}

static futcache_status_t backend_covering(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_found,
    double *out_distance,
    size_t *out_index)
{
    if (cache->backend_active &&
        cache->backend == &futcache_pack_vptree_backend) {
        return futcache_pack_vptree_covering_internal(
            cache->backend_state, point, cache->dimension, out_found,
            out_distance, out_index, cache->backend_context);
    }
    if (cache->backend_active && !cache->variable_radii) {
        double nearest = INFINITY;
        futcache_status_t status = cache->backend->nearest(
            cache->backend_state, point, cache->dimension, &nearest,
            cache->backend_context);
        if (status != FUTCACHE_OK) return status;
        if (nearest > cache->epsilon) {
            *out_found = false;
            *out_distance = INFINITY;
            *out_index = SIZE_MAX;
            return FUTCACHE_OK;
        }
    }
    min_covering_representative(cache, point, out_found, out_distance,
                                out_index);
    return FUTCACHE_OK;
}

static void disable_backend_locked(futcache_pack_t *cache)
{
    if (!cache->backend_active) return;
    cache->backend->destroy(cache->backend_state, &cache->allocator,
                            cache->backend_context);
    cache->backend_state = NULL;
    cache->backend_active = false;
}

static void link_representative_tail(futcache_pack_t *cache,
                                     pack_representative_t *representative)
{
    representative->next = NULL;
    if (cache->representatives_tail != NULL) {
        cache->representatives_tail->next = representative;
    } else {
        cache->representatives = representative;
    }
    cache->representatives_tail = representative;
}

static futcache_status_t append_representative_locked(
    futcache_pack_t *cache, const double *point, double radius)
{
    pack_representative_t *representative = make_representative(
        &cache->allocator, point, cache->dimension, radius);
    if (representative == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;

    if (cache->backend_active) {
        futcache_status_t status;
        if (cache->backend == &futcache_pack_vptree_backend) {
            status = futcache_pack_vptree_insert_ball_internal(
                cache->backend_state, representative->coordinates,
                cache->dimension, radius, cache->count,
                cache->backend_context);
        } else {
            status = cache->backend->insert(
                cache->backend_state, representative->coordinates,
                cache->dimension, cache->backend_context);
        }
        if (status != FUTCACHE_OK) {
            if (cache->memory_limit_bytes == 0U ||
                status != FUTCACHE_ERROR_OUT_OF_MEMORY) {
                free_representative(&cache->allocator, representative);
                return status;
            }
            /* A bounded backend can run out of scratch space before the
             * representative store does. The exact linear scan is a safe,
             * allocation-free fallback. */
            disable_backend_locked(cache);
        }
    }

    link_representative_tail(cache, representative);
    cache->count++;
    if (cache->count > cache->peak_count) cache->peak_count = cache->count;
    return FUTCACHE_OK;
}

/* Rebuild the derived backend after FIFO replacement.  Once clear succeeds,
 * a later insert failure degrades to the exact linear scan rather than
 * exposing a partially indexed set. */
static futcache_status_t prepare_backend_for_eviction_locked(
    futcache_pack_t *cache)
{
    if (!cache->backend_active) return FUTCACHE_OK;
    return cache->backend->clear(cache->backend_state,
                                 cache->backend_context);
}

static void rebuild_backend_after_eviction_locked(futcache_pack_t *cache)
{
    pack_representative_t *representative;
    size_t index = 0U;
    if (!cache->backend_active) return;

    representative = cache->representatives;
    while (representative != NULL) {
        futcache_status_t status;
        if (cache->backend == &futcache_pack_vptree_backend) {
            status = futcache_pack_vptree_insert_ball_internal(
                cache->backend_state, representative->coordinates,
                cache->dimension, representative->radius, index,
                cache->backend_context);
        } else {
            status = cache->backend->insert(
                cache->backend_state, representative->coordinates,
                cache->dimension, cache->backend_context);
        }
        if (status != FUTCACHE_OK) {
            disable_backend_locked(cache);
            return;
        }
        representative = representative->next;
        ++index;
    }
}

static futcache_status_t evict_oldest_and_replace_locked(
    futcache_pack_t *cache, const double *point, double radius)
{
    pack_representative_t *oldest;
    futcache_status_t status;

    if (cache->count == 0U || cache->representatives == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    status = prepare_backend_for_eviction_locked(cache);
    if (status != FUTCACHE_OK) return status;

    oldest = cache->representatives;
    if (cache->count > 1U) {
        cache->representatives = oldest->next;
        cache->representatives_tail->next = oldest;
        cache->representatives_tail = oldest;
        oldest->next = NULL;
    }
    memcpy(oldest->coordinates, point, cache->dimension * sizeof(double));
    oldest->radius = radius;
    cache->evictions = increment_saturating(cache->evictions);
    rebuild_backend_after_eviction_locked(cache);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_lookup(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_found,
    double *out_distance,
    size_t *out_index)
{
    if (cache == NULL || point == NULL || out_found == NULL ||
        out_distance == NULL || out_index == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    bool found = false;
    double distance = INFINITY;
    size_t index = SIZE_MAX;
    futcache_status_t status = backend_covering(
        cache, point, &found, &distance, &index);
    if (status != FUTCACHE_OK) {
        pthread_rwlock_unlock(lock);
        return status;
    }

    pthread_rwlock_unlock(lock);
    *out_found = found;
    *out_distance = distance;
    *out_index = index;
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_is_novel(
    const futcache_pack_t *cache,
    const double *point,
    bool *out_is_novel)
{
    bool found;
    double distance;
    size_t index;
    if (out_is_novel == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    futcache_status_t status = futcache_pack_lookup(
        cache, point, &found, &distance, &index);
    if (status == FUTCACHE_OK) *out_is_novel = !found;
    return status;
}

futcache_status_t futcache_pack_observe_with_radius(
    futcache_pack_t *cache,
    const double *point,
    double radius,
    bool *out_was_novel,
    double *out_distance,
    size_t *out_index)
{
    if (cache == NULL || point == NULL || !isfinite(radius) || radius < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_wrlock(lock);

    bool found = false;
    double matched_distance = INFINITY;
    size_t matched_index = SIZE_MAX;
    futcache_status_t lookup_status = backend_covering(
        cache, point, &found, &matched_distance, &matched_index);
    if (lookup_status != FUTCACHE_OK) {
        pthread_rwlock_unlock(lock);
        return lookup_status;
    }
    bool novel = !found;

    if (novel) {
        size_t representative_bytes = sizeof(pack_representative_t) +
            cache->dimension * sizeof(double);
        futcache_status_t st;

        if (cache->memory_limit_bytes != 0U &&
            !tracked_allocation_fits(cache, representative_bytes)) {
            /* An empty bounded cache may be crowded only by optional backend
             * state. Drop that derived state before declaring the configured
             * limit unusable. */
            if (cache->count == 0U && cache->backend_active) {
                disable_backend_locked(cache);
            }
            if (tracked_allocation_fits(cache, representative_bytes)) {
                st = append_representative_locked(cache, point, radius);
            } else {
                st = evict_oldest_and_replace_locked(cache, point, radius);
            }
        } else {
            st = append_representative_locked(cache, point, radius);
        }
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(lock);
            return st;
        }
        matched_distance = 0.0;
        matched_index = cache->count - 1U;
        if (radius != cache->epsilon) cache->variable_radii = true;
    }

    cache->observations = increment_saturating(cache->observations);
    if (novel) cache->novel_observations =
        increment_saturating(cache->novel_observations);
    cache->generation = increment_saturating(cache->generation);

    if (out_was_novel != NULL) *out_was_novel = novel;
    if (out_distance != NULL) *out_distance = matched_distance;
    if (out_index != NULL) *out_index = matched_index;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_observe(
    futcache_pack_t *cache,
    const double *point,
    bool *out_was_novel)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    return futcache_pack_observe_with_radius(
        cache, point, cache->epsilon, out_was_novel, NULL, NULL);
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
    out_stats->evictions = cache->evictions;
    out_stats->representative_count = cache->count;
    out_stats->peak_count = cache->peak_count;
    out_stats->memory_bytes = atomic_load_explicit(
        &cache->memory_bytes, memory_order_relaxed);
    out_stats->peak_memory_bytes = atomic_load_explicit(
        &cache->peak_memory_bytes, memory_order_relaxed);
    out_stats->memory_limit_bytes = cache->memory_limit_bytes;

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

    const pack_representative_t *representative = cache->representatives;
    size_t i = 0U;
    while (representative != NULL) {
        memcpy(out_points + i * cache->dimension,
               representative->coordinates,
               cache->dimension * sizeof(double));
        representative = representative->next;
        ++i;
    }

    *inout_count = cache->count;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_copy_radii(
    const futcache_pack_t *cache,
    double *out_radii,
    size_t *inout_count)
{
    if (cache == NULL || inout_count == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);
    if (out_radii == NULL) {
        *inout_count = cache->count;
        pthread_rwlock_unlock(lock);
        return FUTCACHE_OK;
    }
    if (*inout_count < cache->count) {
        *inout_count = cache->count;
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }

    const pack_representative_t *representative = cache->representatives;
    size_t index = 0U;
    while (representative != NULL) {
        out_radii[index++] = representative->radius;
        representative = representative->next;
    }
    *inout_count = cache->count;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_nearest(
    const futcache_pack_t *cache,
    const double *point,
    double *out_distance,
    size_t *out_index)
{
    if (cache == NULL || point == NULL || out_distance == NULL ||
        out_index == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_in_domain(cache, point)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    if (cache->backend_active &&
        cache->backend == &futcache_pack_vptree_backend) {
        futcache_status_t status =
            futcache_pack_vptree_nearest_index_internal(
            cache->backend_state, point, cache->dimension, out_distance,
            out_index, cache->backend_context);
        pthread_rwlock_unlock(lock);
        return status;
    }

    double min_d = INFINITY;
    size_t min_i = SIZE_MAX;
    const pack_representative_t *representative = cache->representatives;
    futcache_distance_fn distance = cache->distance;
    void *context = cache->distance_context;
    size_t dim = cache->dimension;
    size_t i = 0U;
    while (representative != NULL) {
        double d = distance(point, representative->coordinates, dim, context);
        if (d < min_d) {
            min_d = d;
            min_i = i;
        }
        representative = representative->next;
        ++i;
    }

    pthread_rwlock_unlock(lock);
    *out_distance = min_d;
    *out_index = min_i;
    return FUTCACHE_OK;
}

/*
 * W1-optimal eviction: remove the representative with the smallest
 * distance to its nearest neighbour.
 *
 * Rationale (Design Sketch 02): treating the representative set R as an
 * empirical measure mu_R = (1/|R|) sum delta_r, the W1 cost of removing r
 * is the distance from r to the nearest surviving representative. Removing
 * the "most crowded" representative therefore minimises the redistribution
 * of coverage mass. This is the nearest-neighbor-minimum heuristic.
 *
 * The evicted representative is spliced out of the FIFO list. Its
 * allocation is freed (not recycled, unlike FIFO pressure eviction where
 * the recycled allocation is immediately refilled). The backend index is
 * rebuilt from scratch, matching the existing eviction pattern.
 */
static futcache_status_t evict_w1_locked(
    futcache_pack_t *cache,
    size_t *out_evicted_index)
{
    if (cache->count == 0U) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    /* Phase 1: find the representative with the smallest nearest-neighbour
     * distance. O(n^2 * d) brute force. For n > ~5000 the VP-tree backend
     * can accelerate this to O(n log n) but the brute force is sufficient
     * for the memory-ceiling regime where n is bounded by P(K, epsilon). */
    double best_nn_distance = INFINITY;
    size_t best_index = 0U;
    const pack_representative_t *left = cache->representatives;
    size_t left_index = 0U;

    while (left != NULL) {
        const pack_representative_t *right = left->next;
        size_t right_index = left_index + 1U;
        while (right != NULL) {
            double d = cache->distance(
                left->coordinates, right->coordinates,
                cache->dimension, cache->distance_context);
            if (d < best_nn_distance) {
                best_nn_distance = d;
                /* Pick the lower index on ties for determinism. */
                if (left_index < right_index) {
                    best_index = left_index;
                } else {
                    best_index = right_index;
                }
            }
            right = right->next;
            ++right_index;
        }
        left = left->next;
        ++left_index;
    }

    /* If count == 1, best_index stays 0 (no pairs to compare). */
    if (out_evicted_index != NULL) {
        *out_evicted_index = best_index;
    }

    /* Phase 2: prepare backend for eviction. */
    futcache_status_t status = prepare_backend_for_eviction_locked(cache);
    if (status != FUTCACHE_OK) return status;

    /* Phase 3: splice out the representative at best_index. */
    pack_representative_t *to_free;
    pack_representative_t **prev_ptr;
    size_t i = 0U;
    if (best_index == 0U) {
        to_free = cache->representatives;
        if (cache->count > 1U) {
            cache->representatives = to_free->next;
        } else {
            cache->representatives = NULL;
            cache->representatives_tail = NULL;
        }
        prev_ptr = NULL;
    } else {
        pack_representative_t *prev = cache->representatives;
        for (i = 1U; i < best_index; ++i) {
            prev = prev->next;
        }
        to_free = prev->next;
        prev->next = to_free->next;
        if (to_free == cache->representatives_tail) {
            cache->representatives_tail = prev;
        }
        prev_ptr = &prev->next;
    }
    (void)prev_ptr;
    to_free->next = NULL;
    free_representative(&cache->allocator, to_free);
    cache->count--;
    cache->evictions = increment_saturating(cache->evictions);

    /* Phase 4: rebuild backend. */
    rebuild_backend_after_eviction_locked(cache);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_evict_w1(
    futcache_pack_t *cache,
    size_t *out_evicted_index)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    pthread_rwlock_t *lock = &cache->lock;
    pthread_rwlock_wrlock(lock);

    if (cache->count == 0U) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    futcache_status_t status = evict_w1_locked(cache, out_evicted_index);
    cache->generation = increment_saturating(cache->generation);
    pthread_rwlock_unlock(lock);
    return status;
}

futcache_status_t futcache_pack_clear(futcache_pack_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    pthread_rwlock_wrlock(&cache->lock);
    if (cache->backend_active) {
        futcache_status_t st = cache->backend->clear(
            cache->backend_state, cache->backend_context);
        if (st != FUTCACHE_OK) {
            pthread_rwlock_unlock(&cache->lock);
            return st;
        }
    }
    free_all_representatives(cache);
    cache->peak_count = 0U;
    cache->variable_radii = false;
    cache->observations = 0U;
    cache->novel_observations = 0U;
    cache->evictions = 0U;
    cache->generation = increment_saturating(cache->generation);
    atomic_store_explicit(&cache->peak_memory_bytes,
        atomic_load_explicit(&cache->memory_bytes, memory_order_relaxed),
        memory_order_relaxed);
    pthread_rwlock_unlock(&cache->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_validate(const futcache_pack_t *cache)
{
    if (cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    /* Telemetry lifecycle invariants (mirror the interval engine). */
    size_t live_memory = atomic_load_explicit(&cache->memory_bytes,
                                               memory_order_relaxed);
    size_t peak_memory = atomic_load_explicit(&cache->peak_memory_bytes,
                                               memory_order_relaxed);
    if (cache->generation < cache->observations ||
        cache->novel_observations > cache->observations ||
        cache->count > (size_t)UINT64_MAX ||
        (uint64_t)cache->count > cache->novel_observations ||
        cache->evictions > cache->novel_observations ||
        cache->count > cache->peak_count ||
        live_memory > peak_memory ||
        (cache->memory_limit_bytes != 0U &&
         (live_memory > cache->memory_limit_bytes ||
          peak_memory > cache->memory_limit_bytes))) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    if (cache->novel_observations != UINT64_MAX &&
        cache->evictions != UINT64_MAX &&
        ((uint64_t)cache->count > UINT64_MAX - cache->evictions ||
         (uint64_t)cache->count + cache->evictions !=
             cache->novel_observations)) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* Insertion-order variable-radius separation. O(n^2), diagnostic only. */
    futcache_distance_fn distance = cache->distance;
    void *context = cache->distance_context;
    size_t dim = cache->dimension;

    const pack_representative_t *left = cache->representatives;
    size_t actual_count = 0U;
    while (left != NULL) {
        if (!isfinite(left->radius) || left->radius < 0.0 ||
            !point_in_domain(cache, left->coordinates)) {
            pthread_rwlock_unlock(lock);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        const pack_representative_t *right = left->next;
        while (right != NULL) {
            double d = distance(left->coordinates, right->coordinates,
                                dim, context);
            if (!(d > left->radius)) {
                pthread_rwlock_unlock(lock);
                return FUTCACHE_ERROR_CORRUPT_DATA;
            }
            right = right->next;
        }
        actual_count++;
        if (actual_count > cache->count) {
            pthread_rwlock_unlock(lock);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        if (left->next == NULL && left != cache->representatives_tail) {
            pthread_rwlock_unlock(lock);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        left = left->next;
    }
    if (actual_count != cache->count ||
        ((cache->count == 0U) != (cache->representatives_tail == NULL))) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_serialize(
    const futcache_pack_t *cache,
    void *buffer,
    size_t buffer_size,
    size_t *out_size)
{
    uint8_t *bytes = (uint8_t *)buffer;
    uint32_t distance_id;
    uint32_t backend_id;
    size_t required;
    size_t offset;
    size_t serialized_representatives = 0U;
    const pack_representative_t *representative;

    if (cache == NULL || out_size == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!pack_double_serialization_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&cache->lock;
    pthread_rwlock_rdlock(lock);

    distance_id = pack_distance_identifier(cache->distance);
    if (distance_id == UINT32_C(0)) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (cache->dimension > (size_t)UINT64_MAX ||
        cache->count > (size_t)UINT64_MAX ||
        cache->peak_count > (size_t)UINT64_MAX ||
        cache->memory_limit_bytes > (size_t)UINT64_MAX ||
        atomic_load_explicit(&cache->peak_memory_bytes,
                             memory_order_relaxed) > (size_t)UINT64_MAX ||
        !pack_snapshot_size(cache->dimension, cache->count, true,
                            &required)) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    *out_size = required;
    if (buffer == NULL) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_OK;
    }
    if (buffer_size < required) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }

    backend_id = cache->backend_active &&
                 cache->backend == &futcache_pack_vptree_backend
        ? FUTCACHE_PACK_BACKEND_VPTREE
        : FUTCACHE_PACK_BACKEND_LINEAR;
    memcpy(bytes, futcache_pack_serial_magic,
           sizeof(futcache_pack_serial_magic));
    pack_write_u16_le(bytes + 8U,
                      (uint16_t)FUTCACHE_PACK_SERIAL_VERSION);
    pack_write_u16_le(bytes + 10U,
                      (uint16_t)FUTCACHE_PACK_SERIAL_HEADER_SIZE);
    pack_write_u32_le(bytes + 12U,
                      (uint32_t)FUTCACHE_PACK_SERIAL_FLAG_RADII);
    pack_write_double_le(bytes + 16U, cache->epsilon);
    pack_write_u64_le(bytes + 24U, (uint64_t)cache->dimension);
    pack_write_u64_le(bytes + 32U, cache->observations);
    pack_write_u64_le(bytes + 40U, cache->novel_observations);
    pack_write_u64_le(bytes + 48U, cache->generation);
    pack_write_u64_le(bytes + 56U, (uint64_t)cache->count);
    pack_write_u64_le(bytes + 64U, (uint64_t)cache->peak_count);
    pack_write_u64_le(bytes + 72U, cache->evictions);
    pack_write_u64_le(bytes + 80U, (uint64_t)cache->memory_limit_bytes);
    pack_write_u32_le(bytes + 88U, distance_id);
    pack_write_u32_le(bytes + 92U, backend_id);
    pack_write_u64_le(bytes + 96U, (uint64_t)atomic_load_explicit(
        &cache->peak_memory_bytes, memory_order_relaxed));

    offset = FUTCACHE_PACK_SERIAL_HEADER_SIZE;
    for (size_t coordinate = 0U; coordinate < cache->dimension;
         ++coordinate) {
        pack_write_double_le(bytes + offset, cache->domain_min[coordinate]);
        offset += sizeof(double);
    }
    for (size_t coordinate = 0U; coordinate < cache->dimension;
         ++coordinate) {
        pack_write_double_le(bytes + offset, cache->domain_max[coordinate]);
        offset += sizeof(double);
    }
    representative = cache->representatives;
    while (representative != NULL) {
        pack_write_double_le(bytes + offset, representative->radius);
        offset += sizeof(double);
        for (size_t coordinate = 0U; coordinate < cache->dimension;
             ++coordinate) {
            pack_write_double_le(bytes + offset,
                representative->coordinates[coordinate]);
            offset += sizeof(double);
        }
        serialized_representatives++;
        representative = representative->next;
    }
    if (serialized_representatives != cache->count ||
        offset != required - FUTCACHE_PACK_SERIAL_CRC_SIZE) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    pack_write_u32_le(bytes + offset, pack_crc32_bytes(bytes, offset));
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_deserialize(
    const void *data,
    size_t data_size,
    const futcache_allocator_t *allocator,
    futcache_pack_t **out_cache)
{
    const uint8_t *bytes = (const uint8_t *)data;
    futcache_allocator_t normalized;
    futcache_pack_config_t config;
    futcache_pack_t *cache = NULL;
    double *bounds = NULL;
    double *point = NULL;
    futcache_distance_fn distance;
    futcache_status_t status;
    uint64_t serialized_dimension;
    uint64_t serialized_count;
    uint64_t serialized_peak_count;
    uint64_t serialized_limit;
    uint64_t serialized_peak_memory;
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    uint64_t evictions;
    uint16_t version;
    uint32_t flags;
    uint32_t backend_id;
    bool has_radii;
    size_t dimension;
    size_t count;
    size_t expected_size;
    size_t bounds_bytes;
    size_t offset;

    if (out_cache == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *out_cache = NULL;
    if (data == NULL || data_size < FUTCACHE_PACK_SERIAL_HEADER_SIZE +
                                      FUTCACHE_PACK_SERIAL_CRC_SIZE) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!pack_double_serialization_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (memcmp(bytes, futcache_pack_serial_magic,
               sizeof(futcache_pack_serial_magic)) != 0 ||
        pack_read_u16_le(bytes + 10U) !=
            (uint16_t)FUTCACHE_PACK_SERIAL_HEADER_SIZE) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    version = pack_read_u16_le(bytes + 8U);
    flags = pack_read_u32_le(bytes + 12U);
    if ((version == (uint16_t)FUTCACHE_PACK_SERIAL_VERSION_LEGACY &&
         flags == UINT32_C(0))) {
        has_radii = false;
    } else if (version == (uint16_t)FUTCACHE_PACK_SERIAL_VERSION &&
               flags == (uint32_t)FUTCACHE_PACK_SERIAL_FLAG_RADII) {
        has_radii = true;
    } else {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    serialized_dimension = pack_read_u64_le(bytes + 24U);
    serialized_count = pack_read_u64_le(bytes + 56U);
    serialized_peak_count = pack_read_u64_le(bytes + 64U);
    serialized_limit = pack_read_u64_le(bytes + 80U);
    serialized_peak_memory = pack_read_u64_le(bytes + 96U);
    if (serialized_dimension == UINT64_C(0) ||
        serialized_dimension > (uint64_t)SIZE_MAX ||
        serialized_count > (uint64_t)SIZE_MAX ||
        serialized_peak_count > (uint64_t)SIZE_MAX ||
        serialized_limit > (uint64_t)SIZE_MAX ||
        serialized_peak_memory > (uint64_t)SIZE_MAX) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    dimension = (size_t)serialized_dimension;
    count = (size_t)serialized_count;
    if (!pack_snapshot_size(dimension, count, has_radii, &expected_size) ||
        data_size != expected_size) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    if (pack_read_u32_le(bytes + data_size -
                         FUTCACHE_PACK_SERIAL_CRC_SIZE) !=
        pack_crc32_bytes(bytes, data_size - FUTCACHE_PACK_SERIAL_CRC_SIZE)) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    distance = pack_distance_from_identifier(pack_read_u32_le(bytes + 88U));
    backend_id = pack_read_u32_le(bytes + 92U);
    if (distance == NULL ||
        (backend_id != FUTCACHE_PACK_BACKEND_LINEAR &&
         backend_id != FUTCACHE_PACK_BACKEND_VPTREE)) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    observations = pack_read_u64_le(bytes + 32U);
    novel_observations = pack_read_u64_le(bytes + 40U);
    generation = pack_read_u64_le(bytes + 48U);
    evictions = pack_read_u64_le(bytes + 72U);
    if (novel_observations > observations || generation < observations ||
        serialized_count > novel_observations ||
        serialized_peak_count < serialized_count ||
        evictions > novel_observations ||
        (serialized_limit != UINT64_C(0) &&
         serialized_peak_memory > serialized_limit) ||
        (novel_observations != UINT64_MAX && evictions != UINT64_MAX &&
         (serialized_count > UINT64_MAX - evictions ||
          serialized_count + evictions != novel_observations))) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    status = normalize_allocator(allocator, &normalized);
    if (status != FUTCACHE_OK) return status;
    if (dimension > SIZE_MAX / sizeof(double)) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    bounds_bytes = dimension * sizeof(double);
    if (bounds_bytes > SIZE_MAX / 2U) return FUTCACHE_ERROR_CORRUPT_DATA;
    bounds = (double *)normalized.allocate(normalized.context,
                                            2U * bounds_bytes);
    point = (double *)normalized.allocate(normalized.context, bounds_bytes);
    if (bounds == NULL || point == NULL) {
        if (bounds != NULL) normalized.deallocate(normalized.context, bounds);
        if (point != NULL) normalized.deallocate(normalized.context, point);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    offset = FUTCACHE_PACK_SERIAL_HEADER_SIZE;
    for (size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
        bounds[coordinate] = pack_read_double_le(bytes + offset);
        offset += sizeof(double);
    }
    for (size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
        bounds[dimension + coordinate] = pack_read_double_le(bytes + offset);
        offset += sizeof(double);
    }

    futcache_pack_config_init(&config);
    config.dimension = dimension;
    config.epsilon = pack_read_double_le(bytes + 16U);
    config.distance = distance;
    config.domain_min = bounds;
    config.domain_max = bounds + dimension;
    config.allocator = normalized;
    config.backend = backend_id == FUTCACHE_PACK_BACKEND_VPTREE
        ? &futcache_pack_vptree_backend
        : NULL;
    config.max_memory_bytes = (size_t)serialized_limit;
    if (!valid_config(&config)) {
        normalized.deallocate(normalized.context, bounds);
        normalized.deallocate(normalized.context, point);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    status = futcache_pack_create(&config, &cache);
    normalized.deallocate(normalized.context, bounds);
    bounds = NULL;
    if (status != FUTCACHE_OK) {
        normalized.deallocate(normalized.context, point);
        return status == FUTCACHE_ERROR_INVALID_ARGUMENT
            ? FUTCACHE_ERROR_CORRUPT_DATA
            : status;
    }

    for (size_t representative_index = 0U;
         representative_index < count; ++representative_index) {
        bool was_novel = false;
        double radius = config.epsilon;
        if (has_radii) {
            radius = pack_read_double_le(bytes + offset);
            offset += sizeof(double);
        }
        for (size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
            point[coordinate] = pack_read_double_le(bytes + offset);
            offset += sizeof(double);
        }
        if (!isfinite(radius) || radius < 0.0 ||
            !point_in_domain(cache, point)) {
            normalized.deallocate(normalized.context, point);
            futcache_pack_destroy(cache);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        status = futcache_pack_observe_with_radius(
            cache, point, radius, &was_novel, NULL, NULL);
        if (status != FUTCACHE_OK || !was_novel) {
            normalized.deallocate(normalized.context, point);
            futcache_pack_destroy(cache);
            return status != FUTCACHE_OK ? status
                                         : FUTCACHE_ERROR_CORRUPT_DATA;
        }
    }
    normalized.deallocate(normalized.context, point);
    point = NULL;

    if (cache->count != count ||
        offset != data_size - FUTCACHE_PACK_SERIAL_CRC_SIZE) {
        futcache_pack_destroy(cache);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    cache->observations = observations;
    cache->novel_observations = novel_observations;
    cache->generation = generation;
    cache->peak_count = (size_t)serialized_peak_count;
    cache->evictions = evictions;
    {
        size_t restored_peak = (size_t)serialized_peak_memory;
        size_t local_peak = atomic_load_explicit(&cache->peak_memory_bytes,
                                                  memory_order_relaxed);
        if (local_peak > restored_peak) restored_peak = local_peak;
        if (cache->memory_limit_bytes != 0U &&
            restored_peak > cache->memory_limit_bytes) {
            futcache_pack_destroy(cache);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        atomic_store_explicit(&cache->peak_memory_bytes, restored_peak,
                              memory_order_relaxed);
    }

    status = futcache_pack_validate(cache);
    if (status != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return status;
    }
    *out_cache = cache;
    return FUTCACHE_OK;
}

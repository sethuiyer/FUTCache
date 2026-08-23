#ifndef FUTCACHE_FUTCACHE_H
#define FUTCACHE_FUTCACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FUTCACHE_VERSION_MAJOR 1
#define FUTCACHE_VERSION_MINOR 1
#define FUTCACHE_VERSION_PATCH 0

typedef enum futcache_status {
    FUTCACHE_OK = 0,
    FUTCACHE_ERROR_INVALID_ARGUMENT,
    FUTCACHE_ERROR_OUT_OF_MEMORY,
    FUTCACHE_ERROR_OUT_OF_RANGE,
    FUTCACHE_ERROR_BUFFER_TOO_SMALL,
    FUTCACHE_ERROR_CORRUPT_DATA,
    FUTCACHE_ERROR_UNSUPPORTED_PLATFORM,
    FUTCACHE_ERROR_SYSTEM
} futcache_status_t;

typedef void *(*futcache_allocate_fn)(void *context, size_t size);
typedef void (*futcache_deallocate_fn)(void *context, void *pointer);

/* Both callbacks must be supplied, or both left NULL to use malloc/free. */
typedef struct futcache_allocator {
    futcache_allocate_fn allocate;
    futcache_deallocate_fn deallocate;
    void *context;
} futcache_allocator_t;

typedef struct futcache_config {
    double domain_min;
    double domain_max;
    double epsilon;
    futcache_allocator_t allocator;
} futcache_config_t;

typedef struct futcache_interval {
    double lower;
    double upper;
} futcache_interval_t;

typedef struct futcache_parameters {
    double domain_min;
    double domain_max;
    double epsilon;
} futcache_parameters_t;

typedef struct futcache_stats {
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    size_t interval_count;
    /* Height of the interval AVL root, or zero when the cache is empty. */
    size_t tree_height;
    double covered_measure;
    size_t memory_bytes;
    bool fully_covered;
} futcache_stats_t;

typedef struct futcache futcache_t;

/* Initializes config for K=[0,1] and epsilon=0.1. */
FUTCACHE_API void futcache_config_init(futcache_config_t *config);

/* Creates a thread-safe exact metric-novelty cache. */
FUTCACHE_API futcache_status_t futcache_create(
    const futcache_config_t *config,
    futcache_t **out_cache
);

/* The caller must ensure no other thread is using cache during destruction. */
FUTCACHE_API void futcache_destroy(futcache_t *cache);

/*
 * Reports whether x is outside the union of epsilon-balls induced by all
 * successful observations. Querying does not modify the cache.
 */
FUTCACHE_API futcache_status_t futcache_is_novel(
    const futcache_t *cache,
    double x,
    bool *out_is_novel
);

/*
 * Atomically reports novelty and incorporates x into the decision boundary.
 * The epsilon-ball is incorporated even when x was not novel.
 */
FUTCACHE_API futcache_status_t futcache_observe(
    futcache_t *cache,
    double x,
    bool *out_was_novel
);

FUTCACHE_API futcache_status_t futcache_get_stats(
    const futcache_t *cache,
    futcache_stats_t *out_stats
);

FUTCACHE_API futcache_status_t futcache_get_parameters(
    const futcache_t *cache,
    futcache_parameters_t *out_parameters
);

/*
 * Copies the canonical sorted, disjoint interval union. On input,
 * *inout_count is the output capacity; on return it is the required count.
 * Pass intervals=NULL and *inout_count=0 to query the required size.
 */
FUTCACHE_API futcache_status_t futcache_copy_intervals(
    const futcache_t *cache,
    futcache_interval_t *intervals,
    size_t *inout_count
);

/* Clears the decision boundary and counters while advancing generation. */
FUTCACHE_API futcache_status_t futcache_clear(futcache_t *cache);

/* Performs an O(n) internal invariant check intended for diagnostics. */
FUTCACHE_API futcache_status_t futcache_validate(const futcache_t *cache);

/*
 * Serializes an atomic snapshot into a checksummed, endian-independent binary
 * format. Pass buffer=NULL to obtain the required size in out_size.
 */
FUTCACHE_API futcache_status_t futcache_serialize(
    const futcache_t *cache,
    void *buffer,
    size_t buffer_size,
    size_t *out_size
);

/* Restores a serialized cache. allocator may be NULL to use malloc/free. */
FUTCACHE_API futcache_status_t futcache_deserialize(
    const void *data,
    size_t data_size,
    const futcache_allocator_t *allocator,
    futcache_t **out_cache
);

FUTCACHE_API const char *futcache_status_string(futcache_status_t status);

#ifdef __cplusplus
}
#endif

#endif

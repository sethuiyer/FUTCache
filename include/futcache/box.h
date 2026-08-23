#ifndef FUTCACHE_BOX_H
#define FUTCACHE_BOX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "futcache/export.h"
#include "futcache/futcache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact L_inf novelty coverage for a bounded, fixed-dimensional domain.
 * The current implementation supports dimensions 1..8 and stores the union
 * as an append-only list of closed epsilon boxes.  Overlap is intentional:
 * the representation is exact but not canonical/minimal; box_count is a
 * storage diagnostic, not a packing bound.  A future canonical cell backend
 * can replace the representation without changing this API. */
typedef struct futcache_box_config {
    size_t dimension;
    double epsilon;
    const double *domain_min;
    const double *domain_max;
    futcache_allocator_t allocator;
} futcache_box_config_t;

typedef struct futcache_box_stats {
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
    size_t box_count;
    size_t peak_box_count;
    size_t memory_bytes;
} futcache_box_stats_t;

typedef struct futcache_box futcache_box_t;
FUTCACHE_API void futcache_box_config_init(futcache_box_config_t *config);
FUTCACHE_API futcache_status_t futcache_box_create(const futcache_box_config_t *config, futcache_box_t **out_cache);
FUTCACHE_API void futcache_box_destroy(futcache_box_t *cache);
FUTCACHE_API futcache_status_t futcache_box_is_novel(const futcache_box_t *cache, const double *point, bool *out_is_novel);
FUTCACHE_API futcache_status_t futcache_box_observe(futcache_box_t *cache, const double *point, bool *out_was_novel);
FUTCACHE_API futcache_status_t futcache_box_get_stats(const futcache_box_t *cache, futcache_box_stats_t *out_stats);
FUTCACHE_API futcache_status_t futcache_box_clear(futcache_box_t *cache);
FUTCACHE_API futcache_status_t futcache_box_validate(const futcache_box_t *cache);

#ifdef __cplusplus
}
#endif
#endif

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
 * Supports dimensions 1..8 and stores the union as a list of closed
 * epsilon boxes (one per novel observation).
 *
 * The representation is exact but non-canonical (not a minimal box
 * union). Under the novelty admission invariant a newly admitted box
 * can never strictly contain, be contained in, or duplicate a previously
 * admitted box — doing so would place the prior center within epsilon of
 * the new center, contradicting novelty — so stored boxes are pairwise
 * non-redundant under containment and only partially overlap. box_count
 * is therefore a storage diagnostic (equal to the number of novel
 * observations), bounded above by the packing number P(K, eps) but not a
 * canonical minimal cell count. A future disjoint-cell backend could
 * replace this representation without changing this API. */
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

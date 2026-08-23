#ifndef FUTCACHE_TOWER_H
#define FUTCACHE_TOWER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/futcache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct futcache_tower_config {
    double domain_min;
    double domain_max;
    size_t level_count;
    size_t root_cells;
    futcache_allocator_t allocator;
} futcache_tower_config_t;

typedef struct futcache_tower_level_info {
    size_t cell_count;
    size_t discovered_count;
} futcache_tower_level_info_t;

typedef struct futcache_tower_stats {
    uint64_t observations;
    uint64_t generation;
    size_t level_count;
    size_t total_cells;
    size_t total_discoveries;
    size_t memory_bytes;
} futcache_tower_stats_t;

typedef struct futcache_tower futcache_tower_t;

/* Initializes a two-level dyadic tower over [0,1], with 2 then 4 cells. */
FUTCACHE_API void futcache_tower_config_init(futcache_tower_config_t *config);

FUTCACHE_API futcache_status_t futcache_tower_create(
    const futcache_tower_config_t *config,
    futcache_tower_t **out_tower
);

FUTCACHE_API void futcache_tower_destroy(futcache_tower_t *tower);

/* Writes one byte per level: 1 when x's cell is unseen, 0 otherwise. */
FUTCACHE_API futcache_status_t futcache_tower_query(
    const futcache_tower_t *tower,
    double x,
    uint8_t *out_novel,
    size_t output_capacity
);

/* out_was_novel may be NULL when the per-level result is not needed. */
FUTCACHE_API futcache_status_t futcache_tower_observe(
    futcache_tower_t *tower,
    double x,
    uint8_t *out_was_novel,
    size_t output_capacity
);

FUTCACHE_API futcache_status_t futcache_tower_cell_index(
    const futcache_tower_t *tower,
    size_t level,
    double x,
    size_t *out_cell
);

FUTCACHE_API futcache_status_t futcache_tower_level_info(
    const futcache_tower_t *tower,
    size_t level,
    futcache_tower_level_info_t *out_info
);

/* Number of occupied cells whose spatial index is <= cell_inclusive. */
FUTCACHE_API futcache_status_t futcache_tower_prefix_count(
    const futcache_tower_t *tower,
    size_t level,
    size_t cell_inclusive,
    size_t *out_count
);

/* Returns the zero-based ordinal-th occupied cell in spatial order. */
FUTCACHE_API futcache_status_t futcache_tower_select_occupied(
    const futcache_tower_t *tower,
    size_t level,
    size_t ordinal,
    size_t *out_cell
);

/* Returns the cell recorded at a zero-based position in first-discovery order. */
FUTCACHE_API futcache_status_t futcache_tower_discovery_at(
    const futcache_tower_t *tower,
    size_t level,
    size_t discovery_index,
    size_t *out_cell
);

FUTCACHE_API futcache_status_t futcache_tower_get_stats(
    const futcache_tower_t *tower,
    futcache_tower_stats_t *out_stats
);

FUTCACHE_API futcache_status_t futcache_tower_clear(futcache_tower_t *tower);
FUTCACHE_API futcache_status_t futcache_tower_validate(const futcache_tower_t *tower);

#ifdef __cplusplus
}
#endif

#endif

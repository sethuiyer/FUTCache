#ifndef FUTCACHE_PACK_VPTREE_INTERNAL_H
#define FUTCACHE_PACK_VPTREE_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "futcache/pack.h"

/* Private fast paths used only when pack.c owns the built-in VP-tree state.
 * Keeping these out of futcache_pack_backend_ops_t preserves the public custom
 * backend ABI. */
futcache_status_t futcache_pack_vptree_insert_ball_internal(
    void *state,
    const double *point,
    size_t dimension,
    double radius,
    size_t index,
    void *context
);

futcache_status_t futcache_pack_vptree_covering_internal(
    void *state,
    const double *point,
    size_t dimension,
    bool *out_found,
    double *out_distance,
    size_t *out_index,
    void *context
);

futcache_status_t futcache_pack_vptree_nearest_index_internal(
    void *state,
    const double *point,
    size_t dimension,
    double *out_distance,
    size_t *out_index,
    void *context
);

#endif

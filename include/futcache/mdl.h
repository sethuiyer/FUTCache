#ifndef FUTCACHE_MDL_H
#define FUTCACHE_MDL_H

#include <stdbool.h>
#include <stddef.h>

#include "futcache/export.h"
#include "futcache/pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Objective family used by the offline epsilon selector. */
typedef enum futcache_mdl_mode {
    /* Lossless two-part code at a caller-declared coordinate precision. */
    FUTCACHE_MDL_LOSSLESS = 0,
    /* Lossy rate-distortion Lagrangian with squared metric residuals. */
    FUTCACHE_MDL_LOSSY = 1
} futcache_mdl_mode_t;

typedef struct futcache_mdl_config {
    /* Candidate epsilon values. The selector evaluates every entry. */
    const double *epsilon_grid;
    size_t epsilon_count;
    /* Optional code lengths for grid entries; NULL uses log2(index + 2). */
    const double *epsilon_code_bits;
    futcache_mdl_mode_t mode;
    /* Smallest exactly representable coordinate increment for lossless mode. */
    double precision;
    /* Bits charged per unit total squared residual in lossy mode. */
    double distortion_weight;
    /* Optional pack memory ceiling used while evaluating each candidate. */
    size_t max_memory_bytes;
    /* Optional backend; NULL selects the exact linear backend. */
    const futcache_pack_backend_ops_t *backend;
} futcache_mdl_config_t;

typedef struct futcache_mdl_result {
    double epsilon;
    double objective_bits;
    double model_bits;
    double epsilon_bits;
    double data_bits;
    double total_squared_error;
    size_t representative_count;
    size_t memory_bytes;
} futcache_mdl_result_t;

FUTCACHE_API void futcache_mdl_config_init(futcache_mdl_config_t *config);

/*
 * Select the shortest two-part code over the supplied finite epsilon grid.
 * The model term is the exact live pack allocation reported by FUTCache,
 * converted to bits. The lossless data term is
 *
 *   n log2(|R|) + n*d*log2(epsilon / precision),
 *
 * assuming coordinates are quantized at `precision` and epsilon >= precision.
 * The lossy data term is
 *
 *   n log2(|R|) + distortion_weight * sum_i nearest_distance_i^2.
 *
 * The epsilon-code term is `epsilon_code_bits[i]` when supplied, otherwise
 * `log2(i + 2)`. The optimum is guaranteed only relative to this codec and
 * candidate grid.
 * `inout_result_count` is a capacity/count parameter: pass out_results=NULL
 * and *inout_result_count=0 to query the required result count.
 */
FUTCACHE_API futcache_status_t futcache_mdl_select_epsilon(
    const double *points,
    size_t point_count,
    size_t dimension,
    futcache_distance_fn distance,
    void *distance_context,
    const double *domain_min,
    const double *domain_max,
    const futcache_mdl_config_t *config,
    futcache_mdl_result_t *out_results,
    size_t *inout_result_count,
    size_t *out_best_index);

#ifdef __cplusplus
}
#endif

#endif

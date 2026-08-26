#include "futcache/mdl.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void futcache_mdl_config_init(futcache_mdl_config_t *config)
{
    if (config == NULL) return;
    config->epsilon_grid = NULL;
    config->epsilon_count = 0U;
    config->epsilon_code_bits = NULL;
    config->mode = FUTCACHE_MDL_LOSSLESS;
    config->precision = 0.0;
    config->distortion_weight = 1.0;
    config->max_memory_bytes = 0U;
    config->backend = NULL;
}

static bool valid_config(const double *points, size_t point_count,
                         size_t dimension, const double *domain_min,
                         const double *domain_max,
                         const futcache_mdl_config_t *config,
                         size_t *out_grid_count)
{
    if (config == NULL || out_grid_count == NULL || point_count == 0U ||
        dimension == 0U ||
        domain_min == NULL || domain_max == NULL ||
        (point_count != 0U && points == NULL) ||
        config->epsilon_grid == NULL || config->epsilon_count == 0U) {
        return false;
    }
    if (config->mode != FUTCACHE_MDL_LOSSLESS &&
        config->mode != FUTCACHE_MDL_LOSSY) return false;
    if (config->mode == FUTCACHE_MDL_LOSSLESS &&
        (!isfinite(config->precision) || config->precision <= 0.0)) {
        return false;
    }
    if (config->mode == FUTCACHE_MDL_LOSSY &&
        (!isfinite(config->distortion_weight) ||
         config->distortion_weight < 0.0)) return false;
    for (size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
        if (!isfinite(domain_min[coordinate]) ||
            !isfinite(domain_max[coordinate]) ||
            domain_max[coordinate] <= domain_min[coordinate]) return false;
    }
    for (size_t i = 0U; i < config->epsilon_count; ++i) {
        if (!isfinite(config->epsilon_grid[i]) ||
            config->epsilon_grid[i] < 0.0) return false;
        if (config->mode == FUTCACHE_MDL_LOSSLESS &&
            config->epsilon_grid[i] < config->precision) return false;
        if (config->epsilon_code_bits != NULL &&
            (!isfinite(config->epsilon_code_bits[i]) ||
             config->epsilon_code_bits[i] < 0.0)) return false;
    }
    *out_grid_count = config->epsilon_count;
    return true;
}

static futcache_status_t evaluate_candidate(
    const double *points, size_t point_count, size_t dimension,
    futcache_distance_fn distance, void *distance_context,
    const double *domain_min, const double *domain_max,
    const futcache_mdl_config_t *config, size_t grid_index, double epsilon,
    futcache_mdl_result_t *out)
{
    futcache_pack_config_t pack_config;
    futcache_pack_config_init(&pack_config);
    pack_config.dimension = dimension;
    pack_config.epsilon = epsilon;
    pack_config.distance = distance;
    pack_config.distance_context = distance_context;
    pack_config.domain_min = domain_min;
    pack_config.domain_max = domain_max;
    pack_config.backend = config->backend;
    pack_config.max_memory_bytes = config->max_memory_bytes;

    futcache_pack_t *cache = NULL;
    futcache_status_t status = futcache_pack_create(&pack_config, &cache);
    if (status != FUTCACHE_OK) return status;
    for (size_t i = 0U; i < point_count; ++i) {
        bool unused_novel;
        status = futcache_pack_observe(cache,
            points + i * dimension, &unused_novel);
        if (status != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return status;
        }
    }

    futcache_pack_stats_t stats;
    status = futcache_pack_get_stats(cache, &stats);
    if (status != FUTCACHE_OK || stats.representative_count == 0U) {
        futcache_pack_destroy(cache);
        return status == FUTCACHE_OK ? FUTCACHE_ERROR_CORRUPT_DATA : status;
    }

    double squared_error = 0.0;
    for (size_t i = 0U; i < point_count; ++i) {
        double nearest_distance;
        size_t unused_index;
        status = futcache_pack_nearest(cache, points + i * dimension,
                                       &nearest_distance, &unused_index);
        if (status != FUTCACHE_OK || !isfinite(nearest_distance)) {
            futcache_pack_destroy(cache);
            return status == FUTCACHE_OK ? FUTCACHE_ERROR_CORRUPT_DATA : status;
        }
        if (nearest_distance > sqrt(DBL_MAX) ||
            squared_error > DBL_MAX - nearest_distance * nearest_distance) {
            futcache_pack_destroy(cache);
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
        squared_error += nearest_distance * nearest_distance;
    }

    double model_bits = 8.0 * (double)stats.memory_bytes;
    double epsilon_bits = config->epsilon_code_bits != NULL
        ? config->epsilon_code_bits[grid_index]
        : log2((double)grid_index + 2.0);
    double assignment_bits = point_count == 0U ? 0.0 :
        (double)point_count * log2((double)stats.representative_count);
    double data_bits = assignment_bits;
    if (config->mode == FUTCACHE_MDL_LOSSLESS) {
        double ratio = epsilon / config->precision;
        data_bits += (double)point_count * (double)dimension * log2(ratio);
    } else {
        data_bits += config->distortion_weight * squared_error;
    }
    if (!isfinite(model_bits) || !isfinite(data_bits)) {
        futcache_pack_destroy(cache);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    out->epsilon = epsilon;
    out->model_bits = model_bits;
    out->epsilon_bits = epsilon_bits;
    out->data_bits = data_bits;
    out->objective_bits = model_bits + epsilon_bits + data_bits;
    out->total_squared_error = squared_error;
    out->representative_count = stats.representative_count;
    out->memory_bytes = stats.memory_bytes;
    futcache_pack_destroy(cache);
    return FUTCACHE_OK;
}

futcache_status_t futcache_mdl_select_epsilon(
    const double *points, size_t point_count, size_t dimension,
    futcache_distance_fn distance, void *distance_context,
    const double *domain_min, const double *domain_max,
    const futcache_mdl_config_t *config,
    futcache_mdl_result_t *out_results, size_t *inout_result_count,
    size_t *out_best_index)
{
    if (inout_result_count == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (out_best_index != NULL) *out_best_index = SIZE_MAX;
    size_t grid_count = 0U;
    if (!valid_config(points, point_count, dimension, domain_min, domain_max,
                      config, &grid_count)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (out_results == NULL) {
        if (*inout_result_count != 0U) return FUTCACHE_ERROR_INVALID_ARGUMENT;
        *inout_result_count = grid_count;
        return FUTCACHE_OK;
    }
    if (*inout_result_count < grid_count) {
        *inout_result_count = grid_count;
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    size_t best = SIZE_MAX;
    double best_objective = HUGE_VAL;
    for (size_t i = 0U; i < grid_count; ++i) {
        futcache_status_t status = evaluate_candidate(
            points, point_count, dimension, distance, distance_context,
            domain_min, domain_max, config, i, config->epsilon_grid[i],
            &out_results[i]);
        if (status != FUTCACHE_OK) {
            *inout_result_count = 0U;
            return status;
        }
        if (out_results[i].objective_bits < best_objective) {
            best_objective = out_results[i].objective_bits;
            best = i;
        }
    }
    *inout_result_count = grid_count;
    if (out_best_index != NULL) *out_best_index = best;
    return FUTCACHE_OK;
}

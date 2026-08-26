#include "test.h"

#include "futcache/mdl.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool test_mdl_lossless_grid(void)
{
    const double points[] = {0.0, 0.1, 0.2, 1.0};
    const double lo[] = {0.0};
    const double hi[] = {1.0};
    const double grid[] = {0.05, 0.2, 1.0};
    futcache_mdl_config_t config;
    futcache_mdl_config_init(&config);
    config.epsilon_grid = grid;
    config.epsilon_count = 3U;
    config.precision = 0.01;

    futcache_mdl_result_t results[3];
    size_t result_count = 0U;
    size_t best = SIZE_MAX;
    TEST_STATUS(futcache_mdl_select_epsilon(
        points, 4U, 1U, futcache_distance_l2, NULL, lo, hi, &config,
        NULL, &result_count, &best), FUTCACHE_OK);
    TEST_ASSERT(result_count == 3U);
    TEST_ASSERT(best == SIZE_MAX);
    result_count = 3U;
    TEST_STATUS(futcache_mdl_select_epsilon(
        points, 4U, 1U, futcache_distance_l2, NULL, lo, hi, &config,
        results, &result_count, &best), FUTCACHE_OK);
    TEST_ASSERT(result_count == 3U && best < 3U);

    double minimum = results[0].objective_bits;
    for (size_t i = 0U; i < result_count; ++i) {
        TEST_NEAR(results[i].model_bits,
                  8.0 * (double)results[i].memory_bytes, 1e-9);
        TEST_ASSERT(results[i].epsilon_bits >= 0.0);
        TEST_ASSERT(isfinite(results[i].objective_bits));
        TEST_ASSERT(results[i].data_bits >= 0.0);
        if (results[i].objective_bits < minimum) minimum =
            results[i].objective_bits;
    }
    TEST_NEAR(results[best].objective_bits, minimum, 1e-9);
    TEST_ASSERT(results[0].representative_count >=
                results[1].representative_count);
    TEST_ASSERT(results[1].representative_count >=
                results[2].representative_count);
    return true;
}

static bool test_mdl_lossy_and_validation(void)
{
    const double points[] = {0.0, 0.1, 0.2, 1.0};
    const double lo[] = {0.0};
    const double hi[] = {1.0};
    const double grid[] = {0.0, 0.2, 1.0};
    futcache_mdl_config_t config;
    futcache_mdl_config_init(&config);
    config.epsilon_grid = grid;
    config.epsilon_count = 3U;
    config.mode = FUTCACHE_MDL_LOSSY;
    config.distortion_weight = 10.0;

    futcache_mdl_result_t results[3];
    size_t result_count = 3U;
    size_t best = SIZE_MAX;
    TEST_STATUS(futcache_mdl_select_epsilon(
        points, 4U, 1U, futcache_distance_l2, NULL, lo, hi, &config,
        results, &result_count, &best), FUTCACHE_OK);
    TEST_ASSERT(best < 3U);
    TEST_ASSERT(results[0].representative_count == 4U);
    TEST_ASSERT(results[2].representative_count == 1U);
    TEST_ASSERT(results[0].total_squared_error == 0.0);

    config.mode = FUTCACHE_MDL_LOSSLESS;
    config.precision = 0.25;
    result_count = 3U;
    TEST_STATUS(futcache_mdl_select_epsilon(
        points, 4U, 1U, futcache_distance_l2, NULL, lo, hi, &config,
        results, &result_count, &best), FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static size_t semantic_false_merges(const double *points, const char *labels,
                                    size_t point_count, double epsilon)
{
    const double lo[] = {0.0};
    const double hi[] = {1.0};
    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = 1U;
    config.epsilon = epsilon;
    config.distance = futcache_distance_l2;
    config.domain_min = lo;
    config.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&config, &cache) != FUTCACHE_OK) return SIZE_MAX;

    char representative_labels[8] = {0};
    size_t false_merges = 0U;
    for (size_t i = 0U; i < point_count; ++i) {
        bool found = false;
        double distance = 0.0;
        size_t index = SIZE_MAX;
        if (futcache_pack_lookup(cache, points + i, &found, &distance,
                                 &index) != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return SIZE_MAX;
        }
        if (found && representative_labels[index] != labels[i]) {
            false_merges++;
        }
        bool novel = false;
        if (futcache_pack_observe(cache, points + i, &novel) != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return SIZE_MAX;
        }
        if (novel) {
            futcache_pack_stats_t stats;
            if (futcache_pack_get_stats(cache, &stats) != FUTCACHE_OK ||
                stats.representative_count > sizeof(representative_labels)) {
                futcache_pack_destroy(cache);
                return SIZE_MAX;
            }
            representative_labels[stats.representative_count - 1U] = labels[i];
        }
    }
    futcache_pack_destroy(cache);
    return false_merges;
}

static bool test_mdl_is_not_semantic_safety(void)
{
    /* Three duplicated intents; coarse geometric compression merges them. */
    const double points[] = {0.10, 0.10, 0.45, 0.45, 0.80, 0.80};
    const char labels[] = {'A', 'A', 'B', 'B', 'C', 'C'};
    const double lo[] = {0.0};
    const double hi[] = {1.0};
    const double grid[] = {0.01, 0.50, 1.00};
    futcache_mdl_config_t config;
    futcache_mdl_config_init(&config);
    config.epsilon_grid = grid;
    config.epsilon_count = 3U;
    config.mode = FUTCACHE_MDL_LOSSY;
    config.distortion_weight = 1.0;

    futcache_mdl_result_t curve[3];
    size_t count = 3U;
    size_t best = SIZE_MAX;
    TEST_STATUS(futcache_mdl_select_epsilon(
        points, 6U, 1U, futcache_distance_l2, NULL, lo, hi, &config,
        curve, &count, &best), FUTCACHE_OK);
    TEST_ASSERT(best == 2U);
    TEST_ASSERT(curve[best].objective_bits < curve[0].objective_bits);
    TEST_ASSERT(semantic_false_merges(points, labels, 6U, grid[0]) == 0U);
    TEST_ASSERT(semantic_false_merges(points, labels, 6U, curve[best].epsilon) > 0U);
    return true;
}

int mdl_test_suite(void)
{
    static const test_case_t tests[] = {
        {"lossless finite-grid MDL selection", test_mdl_lossless_grid},
        {"lossy MDL and validation", test_mdl_lossy_and_validation},
        {"geometric MDL is not semantic safety", test_mdl_is_not_semantic_safety},
    };
    return run_test_cases("mdl", tests, sizeof(tests) / sizeof(tests[0]));
}

#include "test.h"

#include "futcache/mdl.h"

#include <math.h>
#include <stdint.h>
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

int mdl_test_suite(void)
{
    static const test_case_t tests[] = {
        {"lossless finite-grid MDL selection", test_mdl_lossless_grid},
        {"lossy MDL and validation", test_mdl_lossy_and_validation},
    };
    return run_test_cases("mdl", tests, sizeof(tests) / sizeof(tests[0]));
}

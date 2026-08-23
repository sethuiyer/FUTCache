#include "test.h"

#include <fenv.h>
#include <float.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct failing_allocator_context {
    size_t allocations_remaining;
    size_t active_allocations;
} failing_allocator_context_t;

typedef struct cache_thread_context {
    futcache_t *cache;
    size_t iterations;
    futcache_status_t status;
} cache_thread_context_t;

typedef struct snapshot_thread_context {
    futcache_t *cache;
    size_t iterations;
    size_t maximum_intervals;
    futcache_status_t status;
} snapshot_thread_context_t;

static void *failing_allocate(void *opaque, size_t size)
{
    failing_allocator_context_t *context = opaque;
    void *memory;

    if (context->allocations_remaining == 0U) {
        return NULL;
    }
    context->allocations_remaining--;
    memory = malloc(size);
    if (memory != NULL) {
        context->active_allocations++;
    }
    return memory;
}

static void failing_deallocate(void *opaque, void *pointer)
{
    failing_allocator_context_t *context = opaque;
    if (pointer != NULL) {
        if (context->active_allocations == 0U) {
            abort();
        }
        context->active_allocations--;
        free(pointer);
    }
}

static uint64_t random_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static void test_write_u64_le(uint8_t *destination, uint64_t value)
{
    size_t index;
    for (index = 0U; index < 8U; ++index) {
        destination[index] = (uint8_t)((value >> (index * 8U)) & UINT64_C(0xff));
    }
}

static uint32_t test_crc32(const uint8_t *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    size_t index;
    for (index = 0U; index < size; ++index) {
        unsigned int bit;
        crc ^= (uint32_t)data[index];
        for (bit = 0U; bit < 8U; ++bit) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & UINT32_C(1)));
            crc = (crc >> 1U) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return ~crc;
}

static void test_write_crc(uint8_t *data, size_t size)
{
    uint32_t crc = test_crc32(data, size - 4U);
    data[size - 4U] = (uint8_t)(crc & UINT32_C(0xff));
    data[size - 3U] = (uint8_t)((crc >> 8U) & UINT32_C(0xff));
    data[size - 2U] = (uint8_t)((crc >> 16U) & UINT32_C(0xff));
    data[size - 1U] = (uint8_t)((crc >> 24U) & UINT32_C(0xff));
}

static bool snapshot_is_rejected(const uint8_t *data, size_t size)
{
    futcache_t *cache = (futcache_t *)(uintptr_t)1U;
    futcache_status_t status = futcache_deserialize(data, size, NULL, &cache);
    if (status == FUTCACHE_OK) {
        futcache_destroy(cache);
        return false;
    }
    return cache == NULL;
}

static int interval_compare(const void *left, const void *right)
{
    const futcache_interval_t *left_interval = left;
    const futcache_interval_t *right_interval = right;
    if (left_interval->lower < right_interval->lower) {
        return -1;
    }
    if (left_interval->lower > right_interval->lower) {
        return 1;
    }
    return 0;
}

static size_t reference_interval_union(
    const double *points,
    size_t point_count,
    double epsilon,
    double domain_min,
    double domain_max,
    futcache_interval_t *intervals
)
{
    size_t index;
    size_t merged_count = 0U;

    for (index = 0U; index < point_count; ++index) {
        double lower = points[index] - epsilon;
        double upper = points[index] + epsilon;
        intervals[index].lower = lower < domain_min ? domain_min : lower;
        intervals[index].upper = upper > domain_max ? domain_max : upper;
    }
    qsort(intervals, point_count, sizeof(*intervals), interval_compare);
    for (index = 0U; index < point_count; ++index) {
        if (merged_count == 0U ||
            intervals[merged_count - 1U].upper < intervals[index].lower) {
            intervals[merged_count++] = intervals[index];
        } else if (intervals[index].upper > intervals[merged_count - 1U].upper) {
            intervals[merged_count - 1U].upper = intervals[index].upper;
        }
    }
    return merged_count;
}

static bool test_config_validation(void)
{
    futcache_config_t config;
    futcache_t *cache = (futcache_t *)(uintptr_t)1U;
    futcache_parameters_t parameters;
    bool novel = false;

    futcache_config_init(&config);
    TEST_NEAR(config.domain_min, 0.0, 0.0);
    TEST_NEAR(config.domain_max, 1.0, 0.0);
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_ASSERT(cache != NULL);
    TEST_STATUS(futcache_is_novel(cache, 0.5, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_get_parameters(cache, &parameters), FUTCACHE_OK);
    TEST_NEAR(parameters.domain_min, 0.0, 0.0);
    TEST_NEAR(parameters.domain_max, 1.0, 0.0);
    TEST_NEAR(parameters.epsilon, 0.1, 0.0);
    TEST_STATUS(futcache_is_novel(cache, -0.01, &novel), FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_is_novel(cache, 0.5, NULL), FUTCACHE_ERROR_INVALID_ARGUMENT);
    futcache_destroy(cache);

    config.epsilon = -1.0;
    cache = (futcache_t *)(uintptr_t)1U;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_ASSERT(cache == NULL);
    config.epsilon = 0.1;
    config.domain_min = 2.0;
    config.domain_max = 1.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    futcache_config_init(&config);
    config.allocator.allocate = failing_allocate;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_create(NULL, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_create(&config, NULL), FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_hand_checked_traversal(void)
{
    static const double traversal[] = {0.8, 0.1, 0.7, 0.2, 0.4, 0.4, 0.1, 0.9};
    static const bool expected[] = {true, true, false, false, false, false, false, false};
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    futcache_interval_t interval;
    size_t count = 1U;
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 0.2;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    for (index = 0U; index < sizeof(traversal) / sizeof(traversal[0]); ++index) {
        bool was_novel = !expected[index];
        TEST_STATUS(futcache_observe(cache, traversal[index], &was_novel), FUTCACHE_OK);
        TEST_ASSERT(was_novel == expected[index]);
        TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    }

    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == UINT64_C(8));
    TEST_ASSERT(stats.novel_observations == UINT64_C(2));
    TEST_ASSERT(stats.interval_count == 1U);
    TEST_ASSERT(stats.fully_covered);
    TEST_NEAR(stats.covered_measure, 1.0, 1e-15);
    TEST_STATUS(futcache_copy_intervals(cache, &interval, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 1U);
    TEST_NEAR(interval.lower, 0.0, 0.0);
    TEST_NEAR(interval.upper, 1.0, 0.0);
    futcache_destroy(cache);
    return true;
}

static bool test_absorption(void)
{
    static const double covering_points[] = {0.8, 0.1, 0.7, 0.2, 0.4};
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    size_t absorbed_interval_count;
    size_t index = 0U;

    futcache_config_init(&config);
    config.epsilon = 0.2;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    do {
        TEST_ASSERT(index < sizeof(covering_points) / sizeof(covering_points[0]));
        TEST_STATUS(futcache_observe(cache, covering_points[index], NULL), FUTCACHE_OK);
        TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
        index++;
    } while (!stats.fully_covered);

    absorbed_interval_count = stats.interval_count;
    TEST_ASSERT(absorbed_interval_count == 1U);
    for (index = 0U; index <= 1000U; ++index) {
        double x = (double)index / 1000.0;
        bool novel = true;
        TEST_STATUS(futcache_is_novel(cache, x, &novel), FUTCACHE_OK);
        TEST_ASSERT(!novel);
        TEST_STATUS(futcache_observe(cache, x, &novel), FUTCACHE_OK);
        TEST_ASSERT(!novel);
        TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.fully_covered);
        TEST_ASSERT(stats.interval_count == absorbed_interval_count);
    }
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    return true;
}

static bool test_redundant_observation_extends_boundary(void)
{
    futcache_config_t config;
    futcache_t *cache = NULL;
    bool novel;

    futcache_config_init(&config);
    config.epsilon = 0.2;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_observe(cache, 0.2, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_is_novel(cache, 0.4, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_is_novel(cache, 0.400000000000001, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_destroy(cache);
    return true;
}

static bool test_ieee_boundaries_and_hostile_values(void)
{
    futcache_config_t config;
    futcache_t *cache = NULL;
    bool novel;
    double one_ulp_above_one = nextafter(1.0, INFINITY) - 1.0;
    double one_ulp_below_one = 1.0 - nextafter(1.0, -INFINITY);

    futcache_config_init(&config);
    config.domain_min = -1.0;
    config.domain_max = 2.0;
    config.epsilon = nextafter(one_ulp_above_one, 0.0);
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 1.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, nextafter(1.0, INFINITY), &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_destroy(cache);

    config.epsilon = one_ulp_above_one;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 1.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, nextafter(1.0, INFINITY), &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    futcache_destroy(cache);

    config.epsilon = nextafter(one_ulp_below_one, 0.0);
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 1.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, nextafter(1.0, -INFINITY), &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_destroy(cache);

    config.epsilon = one_ulp_below_one;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 1.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, nextafter(1.0, -INFINITY), &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    futcache_destroy(cache);

    config.domain_min = 0.0;
    config.domain_max = 1.0;
    config.epsilon = 0.25;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.5, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, 0.75, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_is_novel(cache, nextafter(0.75, 0.0), &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_is_novel(cache, nextafter(0.75, INFINITY), &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_observe(cache, NAN, NULL), FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_observe(cache, INFINITY, NULL), FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_observe(cache, -INFINITY, NULL), FUTCACHE_ERROR_OUT_OF_RANGE);
    futcache_destroy(cache);

    config.domain_min = -1.0;
    config.domain_max = 1.0;
    config.epsilon = 0.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, -0.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, +0.0, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    {
        futcache_interval_t interval;
        size_t count = 1U;
        TEST_STATUS(futcache_copy_intervals(cache, &interval, &count), FUTCACHE_OK);
        TEST_ASSERT(!signbit(interval.lower) && !signbit(interval.upper));
    }
    futcache_destroy(cache);

    config.domain_min = -DBL_MIN;
    config.domain_max = DBL_MIN;
    config.epsilon = nextafter(0.0, 1.0);
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.0, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_is_novel(cache, nextafter(0.0, 1.0), &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_is_novel(cache,
        nextafter(nextafter(0.0, 1.0), 1.0), &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_destroy(cache);

    config.domain_min = 0.0;
    config.domain_max = DBL_MAX;
    config.epsilon = DBL_MAX;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, DBL_MAX, NULL), FUTCACHE_OK);
    {
        futcache_stats_t stats;
        TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.fully_covered);
    }
    futcache_destroy(cache);

    config.domain_max = 1.0;
    config.epsilon = INFINITY;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    config.epsilon = 0.1;
    config.domain_min = NAN;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    config.domain_min = -DBL_MAX;
    config.domain_max = DBL_MAX;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);

    futcache_config_init(&config);
    {
        int original_rounding = fegetround();
        if (original_rounding != -1 && fesetround(FE_DOWNWARD) == 0) {
            futcache_status_t changed_rounding_status = futcache_create(&config, &cache);
            TEST_ASSERT(fesetround(original_rounding) == 0);
            TEST_STATUS(changed_rounding_status, FUTCACHE_ERROR_UNSUPPORTED_PLATFORM);
            TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
            TEST_ASSERT(fesetround(FE_UPWARD) == 0);
            changed_rounding_status = futcache_observe(cache, 0.5, NULL);
            TEST_ASSERT(fesetround(original_rounding) == 0);
            TEST_STATUS(changed_rounding_status, FUTCACHE_ERROR_UNSUPPORTED_PLATFORM);
            TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
            futcache_destroy(cache);
        }
    }
    return true;
}

static bool test_directed_interval_rounding_against_long_double(void)
{
    uint64_t random_state = UINT64_C(0x243f6a8885a308d3);
    size_t iteration;

    if (LDBL_MANT_DIG <= DBL_MANT_DIG) {
        return true;
    }
    for (iteration = 0U; iteration < 4096U; ++iteration) {
        futcache_config_t config;
        futcache_t *cache = NULL;
        futcache_interval_t interval;
        size_t count = 1U;
        double x = (double)(random_next(&random_state) >> 11U) /
            9007199254740992.0;
        double epsilon = (double)(random_next(&random_state) >> 12U) /
            18014398509481984.0;
        long double exact_lower = (long double)x - (long double)epsilon;
        long double exact_upper = (long double)x + (long double)epsilon;
        double expected_lower = (double)exact_lower;
        double expected_upper = (double)exact_upper;

        if ((long double)expected_lower < exact_lower) {
            expected_lower = nextafter(expected_lower, INFINITY);
        }
        if ((long double)expected_upper > exact_upper) {
            expected_upper = nextafter(expected_upper, -INFINITY);
        }
        if (expected_lower <= 0.0) {
            expected_lower = 0.0;
        }
        if (expected_upper >= 1.0) {
            expected_upper = 1.0;
        }

        futcache_config_init(&config);
        config.epsilon = epsilon;
        TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
        TEST_STATUS(futcache_observe(cache, x, NULL), FUTCACHE_OK);
        TEST_STATUS(futcache_copy_intervals(cache, &interval, &count), FUTCACHE_OK);
        TEST_ASSERT(count == 1U);
        TEST_ASSERT(interval.lower == expected_lower);
        TEST_ASSERT(interval.upper == expected_upper);
        futcache_destroy(cache);
    }
    return true;
}

static bool intervals_are_equal(const futcache_t *left, const futcache_t *right)
{
    futcache_interval_t *left_intervals;
    futcache_interval_t *right_intervals;
    size_t left_count = 0U;
    size_t right_count = 0U;
    bool equal;

    if (futcache_copy_intervals(left, NULL, &left_count) != FUTCACHE_OK ||
        futcache_copy_intervals(right, NULL, &right_count) != FUTCACHE_OK ||
        left_count != right_count) {
        return false;
    }
    if (left_count == 0U) {
        return true;
    }
    left_intervals = malloc(left_count * sizeof(*left_intervals));
    right_intervals = malloc(right_count * sizeof(*right_intervals));
    if (left_intervals == NULL || right_intervals == NULL) {
        free(left_intervals);
        free(right_intervals);
        return false;
    }
    if (futcache_copy_intervals(left, left_intervals, &left_count) != FUTCACHE_OK ||
        futcache_copy_intervals(right, right_intervals, &right_count) != FUTCACHE_OK) {
        free(left_intervals);
        free(right_intervals);
        return false;
    }
    equal = memcmp(left_intervals, right_intervals,
        left_count * sizeof(*left_intervals)) == 0;
    free(left_intervals);
    free(right_intervals);
    return equal;
}

static bool test_future_equivalence_and_permutation_invariance(void)
{
    enum { CONTINUATION_COUNT = 5000 };
    static const double minimal_history[] = {0.125, 0.375};
    static const double redundant_history[] = {0.25, 0.375, 0.1875, 0.125, 0.3125};
    futcache_config_t config;
    futcache_t *minimal = NULL;
    futcache_t *redundant = NULL;
    uint64_t random_state = UINT64_C(0xc6a4a7935bd1e995);
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 0.125;
    TEST_STATUS(futcache_create(&config, &minimal), FUTCACHE_OK);
    TEST_STATUS(futcache_create(&config, &redundant), FUTCACHE_OK);
    for (index = 0U; index < sizeof(minimal_history) / sizeof(minimal_history[0]); ++index) {
        TEST_STATUS(futcache_observe(minimal, minimal_history[index], NULL), FUTCACHE_OK);
    }
    for (index = 0U; index < sizeof(redundant_history) / sizeof(redundant_history[0]); ++index) {
        TEST_STATUS(futcache_observe(redundant, redundant_history[index], NULL), FUTCACHE_OK);
    }
    TEST_ASSERT(intervals_are_equal(minimal, redundant));

    for (index = 0U; index < CONTINUATION_COUNT; ++index) {
        double x = (double)(random_next(&random_state) % 8193U) / 8192.0;
        bool minimal_novel;
        bool redundant_novel;
        TEST_STATUS(futcache_observe(minimal, x, &minimal_novel), FUTCACHE_OK);
        TEST_STATUS(futcache_observe(redundant, x, &redundant_novel), FUTCACHE_OK);
        TEST_ASSERT(minimal_novel == redundant_novel);
        if (index % 101U == 0U) {
            TEST_ASSERT(intervals_are_equal(minimal, redundant));
            TEST_STATUS(futcache_validate(minimal), FUTCACHE_OK);
            TEST_STATUS(futcache_validate(redundant), FUTCACHE_OK);
        }
    }
    TEST_ASSERT(intervals_are_equal(minimal, redundant));
    futcache_destroy(redundant);
    futcache_destroy(minimal);
    return true;
}

static bool test_exhaustive_exact_novelty_states(void)
{
    enum { ALPHABET_SIZE = 8, STATE_COUNT = 1 << ALPHABET_SIZE };
    unsigned int mask;

    for (mask = 0U; mask < STATE_COUNT; ++mask) {
        futcache_config_t config;
        futcache_t *cache = NULL;
        futcache_stats_t stats;
        unsigned int symbol;
        size_t expected_count = 0U;

        futcache_config_init(&config);
        config.domain_max = (double)(ALPHABET_SIZE - 1);
        config.epsilon = 0.0;
        TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
        for (symbol = 0U; symbol < ALPHABET_SIZE; ++symbol) {
            if ((mask & (1U << symbol)) != 0U) {
                TEST_STATUS(futcache_observe(cache, (double)symbol, NULL), FUTCACHE_OK);
                expected_count++;
            }
        }
        for (symbol = 0U; symbol < ALPHABET_SIZE; ++symbol) {
            bool novel;
            TEST_STATUS(futcache_is_novel(cache, (double)symbol, &novel), FUTCACHE_OK);
            TEST_ASSERT(novel == ((mask & (1U << symbol)) == 0U));
        }
        TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.interval_count == expected_count);
        TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
        futcache_destroy(cache);
    }
    return true;
}

static bool test_exact_cache_and_avl_balance(void)
{
    enum { INSERT_COUNT = 10000, AVL_HEIGHT_FACTOR = 2 };
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    size_t logarithmic_height;
    size_t index;

    futcache_config_init(&config);
    config.domain_max = (double)INSERT_COUNT;
    config.epsilon = 0.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);

    for (index = 0U; index < INSERT_COUNT; ++index) {
        double x = (double)index;
        bool novel = false;
        TEST_STATUS(futcache_observe(cache, x, &novel), FUTCACHE_OK);
        TEST_ASSERT(novel);
    }
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.interval_count == INSERT_COUNT);
    TEST_ASSERT(stats.novel_observations == UINT64_C(10000));
    logarithmic_height = (size_t)ceil(log2((double)INSERT_COUNT + 1.0));
    TEST_ASSERT(stats.tree_height <= AVL_HEIGHT_FACTOR * logarithmic_height);

    for (index = INSERT_COUNT; index > 0U; --index) {
        bool novel = true;
        TEST_STATUS(futcache_observe(cache, (double)(index - 1U), &novel), FUTCACHE_OK);
        TEST_ASSERT(!novel);
    }
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.interval_count == INSERT_COUNT);
    TEST_ASSERT(stats.observations == UINT64_C(20000));
    TEST_ASSERT(stats.tree_height <= AVL_HEIGHT_FACTOR * logarithmic_height);
    futcache_destroy(cache);
    return true;
}

static bool test_randomized_differential(void)
{
    enum { SAMPLE_COUNT = 3000, GRID_SIZE = 4096 };
    futcache_config_t config;
    futcache_t *cache = NULL;
    double observations[SAMPLE_COUNT];
    uint64_t random_state = UINT64_C(0x5a17c9e3d42b806f);
    uint64_t expected_novel_count = UINT64_C(0);
    size_t count = 0U;

    futcache_config_init(&config);
    config.epsilon = 1.0 / 32.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);

    while (count < SAMPLE_COUNT) {
        double x = (double)(random_next(&random_state) % (GRID_SIZE + 1U)) /
            (double)GRID_SIZE;
        bool expected_novel = true;
        bool actual_novel = false;
        size_t previous;

        for (previous = 0U; previous < count; ++previous) {
            if (fabs(x - observations[previous]) <= config.epsilon) {
                expected_novel = false;
                break;
            }
        }
        TEST_STATUS(futcache_is_novel(cache, x, &actual_novel), FUTCACHE_OK);
        TEST_ASSERT(actual_novel == expected_novel);
        TEST_STATUS(futcache_observe(cache, x, &actual_novel), FUTCACHE_OK);
        TEST_ASSERT(actual_novel == expected_novel);
        if (expected_novel) {
            expected_novel_count++;
        }
        observations[count++] = x;
        if (count % 97U == 0U) {
            TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
        }
    }

    {
        futcache_stats_t stats;
        TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.observations == SAMPLE_COUNT);
        TEST_ASSERT(stats.novel_observations == expected_novel_count);
        TEST_ASSERT(stats.fully_covered);
    }
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    return true;
}

static bool test_canonical_state_against_independent_union(void)
{
    enum { POINT_COUNT = 2048, GRID_SIZE = 4096 };
    futcache_config_t config;
    futcache_t *forward = NULL;
    futcache_t *reverse = NULL;
    double points[POINT_COUNT];
    futcache_interval_t reference[POINT_COUNT];
    futcache_interval_t *actual;
    uint64_t random_state = UINT64_C(0x9e3779b97f4a7c15);
    double previous_measure = 0.0;
    size_t reference_count;
    size_t actual_count = 0U;
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 1.0 / 8192.0;
    TEST_STATUS(futcache_create(&config, &forward), FUTCACHE_OK);
    TEST_STATUS(futcache_create(&config, &reverse), FUTCACHE_OK);
    for (index = 0U; index < POINT_COUNT; ++index) {
        futcache_stats_t stats;
        points[index] = (double)(random_next(&random_state) % (GRID_SIZE + 1U)) /
            (double)GRID_SIZE;
        TEST_STATUS(futcache_observe(forward, points[index], NULL), FUTCACHE_OK);
        TEST_STATUS(futcache_get_stats(forward, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.covered_measure >= previous_measure);
        previous_measure = stats.covered_measure;
    }
    for (index = POINT_COUNT; index > 0U; --index) {
        TEST_STATUS(futcache_observe(reverse, points[index - 1U], NULL), FUTCACHE_OK);
    }

    reference_count = reference_interval_union(points, POINT_COUNT, config.epsilon,
        config.domain_min, config.domain_max, reference);
    TEST_STATUS(futcache_copy_intervals(forward, NULL, &actual_count), FUTCACHE_OK);
    TEST_ASSERT(actual_count == reference_count);
    actual = malloc(actual_count * sizeof(*actual));
    TEST_ASSERT(actual != NULL);
    TEST_STATUS(futcache_copy_intervals(forward, actual, &actual_count), FUTCACHE_OK);
    for (index = 0U; index < actual_count; ++index) {
        TEST_ASSERT(actual[index].lower == reference[index].lower);
        TEST_ASSERT(actual[index].upper == reference[index].upper);
    }
    free(actual);
    TEST_ASSERT(intervals_are_equal(forward, reverse));
    TEST_STATUS(futcache_validate(forward), FUTCACHE_OK);
    TEST_STATUS(futcache_validate(reverse), FUTCACHE_OK);
    futcache_destroy(reverse);
    futcache_destroy(forward);
    return true;
}

static bool test_serialization_round_trip(void)
{
    static const double points[] = {0.05, 0.8, 0.4, 0.45, 0.95};
    futcache_config_t config;
    futcache_t *original = NULL;
    futcache_t *restored = NULL;
    futcache_stats_t original_stats;
    futcache_stats_t restored_stats;
    futcache_parameters_t restored_parameters;
    uint8_t *serialized;
    size_t serialized_size = 0U;
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 0.05;
    TEST_STATUS(futcache_create(&config, &original), FUTCACHE_OK);
    for (index = 0U; index < sizeof(points) / sizeof(points[0]); ++index) {
        TEST_STATUS(futcache_observe(original, points[index], NULL), FUTCACHE_OK);
    }
    TEST_STATUS(futcache_serialize(original, NULL, 0U, &serialized_size), FUTCACHE_OK);
    TEST_ASSERT(serialized_size > 72U);
    serialized = malloc(serialized_size);
    TEST_ASSERT(serialized != NULL);
    TEST_STATUS(futcache_serialize(original, serialized, serialized_size - 1U,
        &serialized_size), FUTCACHE_ERROR_BUFFER_TOO_SMALL);
    TEST_STATUS(futcache_serialize(original, serialized, serialized_size,
        &serialized_size), FUTCACHE_OK);
    TEST_STATUS(futcache_deserialize(serialized, serialized_size, NULL, &restored),
        FUTCACHE_OK);

    TEST_STATUS(futcache_get_stats(original, &original_stats), FUTCACHE_OK);
    TEST_STATUS(futcache_get_stats(restored, &restored_stats), FUTCACHE_OK);
    TEST_STATUS(futcache_get_parameters(restored, &restored_parameters), FUTCACHE_OK);
    TEST_ASSERT(restored_parameters.domain_min == config.domain_min);
    TEST_ASSERT(restored_parameters.domain_max == config.domain_max);
    TEST_ASSERT(restored_parameters.epsilon == config.epsilon);
    TEST_ASSERT(original_stats.observations == restored_stats.observations);
    TEST_ASSERT(original_stats.novel_observations == restored_stats.novel_observations);
    TEST_ASSERT(original_stats.generation == restored_stats.generation);
    TEST_ASSERT(original_stats.interval_count == restored_stats.interval_count);
    TEST_NEAR(original_stats.covered_measure, restored_stats.covered_measure, 0.0);

    for (index = 0U; index <= 1000U; ++index) {
        double x = (double)index / 1000.0;
        bool left;
        bool right;
        TEST_STATUS(futcache_is_novel(original, x, &left), FUTCACHE_OK);
        TEST_STATUS(futcache_is_novel(restored, x, &right), FUTCACHE_OK);
        TEST_ASSERT(left == right);
    }

    serialized[20] ^= UINT8_C(0x40);
    {
        futcache_t *corrupt = NULL;
        TEST_STATUS(futcache_deserialize(serialized, serialized_size, NULL, &corrupt),
            FUTCACHE_ERROR_CORRUPT_DATA);
        TEST_ASSERT(corrupt == NULL);
    }
    free(serialized);
    futcache_destroy(restored);
    futcache_destroy(original);
    return true;
}

static bool test_serialization_rejects_corruption_and_allocation_failures(void)
{
    static const double points[] = {0.0625, 0.25, 0.5, 0.75, 0.9375};
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    uint8_t *serialized;
    uint8_t *second;
    uint8_t *extended;
    size_t serialized_size = 0U;
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 0.03125;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    for (index = 0U; index < sizeof(points) / sizeof(points[0]); ++index) {
        TEST_STATUS(futcache_observe(cache, points[index], NULL), FUTCACHE_OK);
    }
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_STATUS(futcache_serialize(cache, NULL, 0U, &serialized_size), FUTCACHE_OK);
    serialized = malloc(serialized_size);
    second = malloc(serialized_size);
    extended = malloc(serialized_size + 1U);
    TEST_ASSERT(serialized != NULL && second != NULL && extended != NULL);
    TEST_STATUS(futcache_serialize(cache, serialized, serialized_size,
        &serialized_size), FUTCACHE_OK);
    {
        size_t second_size = serialized_size;
        TEST_STATUS(futcache_serialize(cache, second, second_size, &second_size),
            FUTCACHE_OK);
        TEST_ASSERT(second_size == serialized_size);
        TEST_ASSERT(memcmp(serialized, second, serialized_size) == 0);
    }

    for (index = 0U; index < serialized_size; ++index) {
        futcache_t *corrupt = (futcache_t *)(uintptr_t)1U;
        serialized[index] ^= UINT8_C(1);
        TEST_ASSERT(futcache_deserialize(serialized, serialized_size, NULL, &corrupt) !=
            FUTCACHE_OK);
        TEST_ASSERT(corrupt == NULL);
        serialized[index] ^= UINT8_C(1);
    }
    for (index = 0U; index < serialized_size; ++index) {
        futcache_t *truncated = (futcache_t *)(uintptr_t)1U;
        TEST_ASSERT(futcache_deserialize(serialized, index, NULL, &truncated) !=
            FUTCACHE_OK);
        TEST_ASSERT(truncated == NULL);
    }
    memcpy(extended, serialized, serialized_size);
    extended[serialized_size] = UINT8_C(0xa5);
    {
        futcache_t *with_trailing_data = NULL;
        TEST_STATUS(futcache_deserialize(extended, serialized_size + 1U, NULL,
            &with_trailing_data), FUTCACHE_ERROR_CORRUPT_DATA);
        TEST_ASSERT(with_trailing_data == NULL);
    }

    /* Valid checksums do not make structurally invalid input acceptable. */
    memcpy(second, serialized, serialized_size);
    test_write_u64_le(second + 16U, UINT64_C(0x7ff8000000000000));
    test_write_crc(second, serialized_size);
    TEST_ASSERT(snapshot_is_rejected(second, serialized_size));

    memcpy(second, serialized, serialized_size);
    test_write_u64_le(second + 40U, UINT64_C(0));
    test_write_u64_le(second + 48U, UINT64_C(1));
    test_write_crc(second, serialized_size);
    TEST_ASSERT(snapshot_is_rejected(second, serialized_size));

    memcpy(second, serialized, serialized_size);
    test_write_u64_le(second + 48U, UINT64_C(0));
    test_write_crc(second, serialized_size);
    TEST_ASSERT(snapshot_is_rejected(second, serialized_size));

    memcpy(second, serialized, serialized_size);
    test_write_u64_le(second + 56U, UINT64_C(0));
    test_write_crc(second, serialized_size);
    TEST_ASSERT(snapshot_is_rejected(second, serialized_size));

    memcpy(second, serialized, serialized_size);
    test_write_u64_le(second + 64U, UINT64_MAX);
    test_write_crc(second, serialized_size);
    TEST_ASSERT(snapshot_is_rejected(second, serialized_size));

    if (stats.interval_count >= 2U) {
        memcpy(second, serialized, serialized_size);
        memcpy(second + 72U + 16U, second + 72U + 8U, 8U);
        test_write_crc(second, serialized_size);
        TEST_ASSERT(snapshot_is_rejected(second, serialized_size));
    }

    for (index = 0U; index <= stats.interval_count; ++index) {
        failing_allocator_context_t context = {index, 0U};
        futcache_allocator_t allocator = {
            failing_allocate, failing_deallocate, &context
        };
        futcache_t *failed = NULL;
        TEST_STATUS(futcache_deserialize(serialized, serialized_size, &allocator,
            &failed), FUTCACHE_ERROR_OUT_OF_MEMORY);
        TEST_ASSERT(failed == NULL);
        TEST_ASSERT(context.active_allocations == 0U);
    }
    {
        failing_allocator_context_t context = {stats.interval_count + 1U, 0U};
        futcache_allocator_t allocator = {
            failing_allocate, failing_deallocate, &context
        };
        futcache_t *restored = NULL;
        TEST_STATUS(futcache_deserialize(serialized, serialized_size, &allocator,
            &restored), FUTCACHE_OK);
        futcache_destroy(restored);
        TEST_ASSERT(context.active_allocations == 0U);
    }

    free(extended);
    free(second);
    free(serialized);
    futcache_destroy(cache);
    return true;
}

static bool test_allocation_failure_is_atomic(void)
{
    failing_allocator_context_t allocator_context = {1U, 0U};
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    bool novel = false;

    futcache_config_init(&config);
    config.epsilon = 0.1;
    config.allocator.allocate = failing_allocate;
    config.allocator.deallocate = failing_deallocate;
    config.allocator.context = &allocator_context;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_ASSERT(allocator_context.active_allocations == 1U);

    TEST_STATUS(futcache_observe(cache, 0.5, &novel), FUTCACHE_ERROR_OUT_OF_MEMORY);
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == UINT64_C(0));
    TEST_ASSERT(stats.interval_count == 0U);
    TEST_STATUS(futcache_is_novel(cache, 0.5, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);

    allocator_context.allocations_remaining = 1U;
    TEST_STATUS(futcache_observe(cache, 0.5, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_observe(cache, 0.5, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    TEST_ASSERT(allocator_context.active_allocations == 0U);
    return true;
}

static bool test_bridge_allocation_failure_is_atomic(void)
{
    failing_allocator_context_t allocator_context = {3U, 0U};
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_interval_t before[2];
    futcache_interval_t after[2];
    size_t before_count = 2U;
    size_t after_count = 2U;
    futcache_stats_t stats;
    bool novel = false;

    futcache_config_init(&config);
    config.epsilon = 0.125;
    config.allocator.allocate = failing_allocate;
    config.allocator.deallocate = failing_deallocate;
    config.allocator.context = &allocator_context;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.125, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.5, NULL), FUTCACHE_OK);
    TEST_ASSERT(allocator_context.allocations_remaining == 0U);
    TEST_STATUS(futcache_copy_intervals(cache, before, &before_count), FUTCACHE_OK);
    TEST_ASSERT(before_count == 2U);

    TEST_STATUS(futcache_observe(cache, 0.3125, &novel),
        FUTCACHE_ERROR_OUT_OF_MEMORY);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_copy_intervals(cache, after, &after_count), FUTCACHE_OK);
    TEST_ASSERT(after_count == before_count);
    TEST_ASSERT(memcmp(before, after, sizeof(before)) == 0);
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == UINT64_C(2));
    TEST_ASSERT(stats.interval_count == 2U);

    allocator_context.allocations_remaining = 1U;
    TEST_STATUS(futcache_observe(cache, 0.3125, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.interval_count == 1U);
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    TEST_ASSERT(allocator_context.active_allocations == 0U);
    return true;
}

static void *cache_thread_main(void *opaque)
{
    cache_thread_context_t *context = opaque;
    size_t index;

    context->status = FUTCACHE_OK;
    for (index = 0U; index < context->iterations; ++index) {
        double x = (double)(index % 256U) / 255.0;
        context->status = futcache_observe(context->cache, x, NULL);
        if (context->status != FUTCACHE_OK) {
            break;
        }
    }
    return NULL;
}

static bool test_concurrent_observers(void)
{
    enum { THREAD_COUNT = 4, ITERATIONS = 2000 };
    futcache_config_t config;
    futcache_t *cache = NULL;
    pthread_t threads[THREAD_COUNT];
    cache_thread_context_t contexts[THREAD_COUNT];
    futcache_stats_t stats;
    size_t thread_index;

    futcache_config_init(&config);
    config.epsilon = 0.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    for (thread_index = 0U; thread_index < THREAD_COUNT; ++thread_index) {
        contexts[thread_index].cache = cache;
        contexts[thread_index].iterations = ITERATIONS;
        contexts[thread_index].status = FUTCACHE_ERROR_SYSTEM;
        TEST_ASSERT(pthread_create(&threads[thread_index], NULL,
            cache_thread_main, &contexts[thread_index]) == 0);
    }
    for (thread_index = 0U; thread_index < THREAD_COUNT; ++thread_index) {
        TEST_ASSERT(pthread_join(threads[thread_index], NULL) == 0);
        TEST_STATUS(contexts[thread_index].status, FUTCACHE_OK);
    }

    TEST_STATUS(futcache_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == (uint64_t)(THREAD_COUNT * ITERATIONS));
    TEST_ASSERT(stats.interval_count == 256U);
    TEST_ASSERT(stats.novel_observations == UINT64_C(256));
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    return true;
}

static void *snapshot_thread_main(void *opaque)
{
    snapshot_thread_context_t *context = opaque;
    size_t capacity = 72U + context->maximum_intervals * 16U + 4U;
    uint8_t *buffer = malloc(capacity);
    size_t iteration;

    if (buffer == NULL) {
        context->status = FUTCACHE_ERROR_OUT_OF_MEMORY;
        return NULL;
    }
    context->status = FUTCACHE_OK;
    for (iteration = 0U; iteration < context->iterations; ++iteration) {
        size_t written = 0U;
        futcache_t *snapshot = NULL;
        bool novel;

        context->status = futcache_serialize(context->cache, buffer, capacity, &written);
        if (context->status != FUTCACHE_OK) {
            break;
        }
        context->status = futcache_deserialize(buffer, written, NULL, &snapshot);
        if (context->status != FUTCACHE_OK) {
            break;
        }
        context->status = futcache_validate(snapshot);
        if (context->status == FUTCACHE_OK) {
            context->status = futcache_is_novel(snapshot,
                (double)(iteration % context->maximum_intervals) /
                    (double)(context->maximum_intervals - 1U),
                &novel);
        }
        futcache_destroy(snapshot);
        if (context->status != FUTCACHE_OK) {
            break;
        }
    }
    free(buffer);
    return NULL;
}

static bool test_concurrent_snapshots_and_writes(void)
{
    enum { WRITER_COUNT = 2, SNAPSHOT_COUNT = 2, POINTS = 512, ITERATIONS = 3000 };
    futcache_config_t config;
    futcache_t *cache = NULL;
    pthread_t writers[WRITER_COUNT];
    pthread_t snapshots[SNAPSHOT_COUNT];
    cache_thread_context_t writer_contexts[WRITER_COUNT];
    snapshot_thread_context_t snapshot_contexts[SNAPSHOT_COUNT];
    size_t index;

    futcache_config_init(&config);
    config.epsilon = 0.0;
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);

    for (index = 0U; index < WRITER_COUNT; ++index) {
        writer_contexts[index].cache = cache;
        writer_contexts[index].iterations = ITERATIONS;
        writer_contexts[index].status = FUTCACHE_ERROR_SYSTEM;
        TEST_ASSERT(pthread_create(&writers[index], NULL, cache_thread_main,
            &writer_contexts[index]) == 0);
    }
    for (index = 0U; index < SNAPSHOT_COUNT; ++index) {
        snapshot_contexts[index].cache = cache;
        snapshot_contexts[index].iterations = 500U;
        snapshot_contexts[index].maximum_intervals = POINTS;
        snapshot_contexts[index].status = FUTCACHE_ERROR_SYSTEM;
        TEST_ASSERT(pthread_create(&snapshots[index], NULL, snapshot_thread_main,
            &snapshot_contexts[index]) == 0);
    }
    for (index = 0U; index < WRITER_COUNT; ++index) {
        TEST_ASSERT(pthread_join(writers[index], NULL) == 0);
        TEST_STATUS(writer_contexts[index].status, FUTCACHE_OK);
    }
    for (index = 0U; index < SNAPSHOT_COUNT; ++index) {
        TEST_ASSERT(pthread_join(snapshots[index], NULL) == 0);
        TEST_STATUS(snapshot_contexts[index].status, FUTCACHE_OK);
    }
    TEST_STATUS(futcache_validate(cache), FUTCACHE_OK);
    futcache_destroy(cache);
    return true;
}

static bool test_clear(void)
{
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t before;
    futcache_stats_t after;
    bool novel;

    futcache_config_init(&config);
    TEST_STATUS(futcache_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_observe(cache, 0.5, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_get_stats(cache, &before), FUTCACHE_OK);
    TEST_STATUS(futcache_clear(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_get_stats(cache, &after), FUTCACHE_OK);
    TEST_ASSERT(after.observations == UINT64_C(0));
    TEST_ASSERT(after.novel_observations == UINT64_C(0));
    TEST_ASSERT(after.interval_count == 0U);
    TEST_ASSERT(after.generation == before.generation + UINT64_C(1));
    TEST_STATUS(futcache_is_novel(cache, 0.5, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_destroy(cache);
    return true;
}

int futcache_test_suite(void)
{
    static const test_case_t tests[] = {
        {"configuration and argument validation", test_config_validation},
        {"hand-checked traversal", test_hand_checked_traversal},
        {"absorbing full-coverage state", test_absorption},
        {"redundant observations extend coverage", test_redundant_observation_extends_boundary},
        {"IEEE-754 boundaries and hostile values", test_ieee_boundaries_and_hostile_values},
        {"directed rounding against long-double oracle", test_directed_interval_rounding_against_long_double},
        {"future equivalence and permutation invariance", test_future_equivalence_and_permutation_invariance},
        {"exhaustive exact-novelty states", test_exhaustive_exact_novelty_states},
        {"exact cache and AVL balance", test_exact_cache_and_avl_balance},
        {"randomized differential oracle", test_randomized_differential},
        {"canonical state against independent union", test_canonical_state_against_independent_union},
        {"serialization round trip and checksum", test_serialization_round_trip},
        {"serialization corruption and allocation failures", test_serialization_rejects_corruption_and_allocation_failures},
        {"allocation failure atomicity", test_allocation_failure_is_atomic},
        {"bridge allocation failure atomicity", test_bridge_allocation_failure_is_atomic},
        {"concurrent observers", test_concurrent_observers},
        {"concurrent snapshots and writes", test_concurrent_snapshots_and_writes},
        {"clear semantics", test_clear}
    };
    return run_test_cases("interval", tests, sizeof(tests) / sizeof(tests[0]));
}

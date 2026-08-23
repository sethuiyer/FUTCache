#include "test.h"

#include "futcache/box.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct box_allocator_context {
    size_t remaining;
    size_t active;
} box_allocator_context_t;

static void *box_allocate(void *opaque, size_t size)
{
    box_allocator_context_t *context = opaque;
    void *memory;

    if (context->remaining == 0U) return NULL;
    context->remaining--;
    memory = malloc(size);
    if (memory != NULL) context->active++;
    return memory;
}

static void box_deallocate(void *opaque, void *pointer)
{
    box_allocator_context_t *context = opaque;
    if (pointer != NULL) {
        if (context->active == 0U) abort();
        context->active--;
        free(pointer);
    }
}

static bool test_box_configuration(void)
{
    double lower[] = {0.0, 0.0};
    double upper[] = {1.0, 1.0};
    futcache_box_config_t config;
    futcache_box_t *cache = NULL;

    futcache_box_config_init(&config);
    TEST_ASSERT(config.dimension == 1U);
    config.dimension = 2U;
    config.epsilon = 0.1;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_validate(cache), FUTCACHE_OK);
    futcache_box_destroy(cache);

    config.dimension = 9U;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    config.dimension = 2U;
    upper[1] = -1.0;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_box_hand_checked_union(void)
{
    double lower[] = {0.0, 0.0};
    double upper[] = {1.0, 1.0};
    double first[] = {0.1, 0.1};
    double bridge[] = {0.35, 0.35};
    double covered[] = {0.25, 0.25};
    double distant[] = {0.9, 0.9};
    futcache_box_config_t config;
    futcache_box_stats_t stats;
    futcache_box_t *cache = NULL;
    bool novel = false;

    futcache_box_config_init(&config);
    config.dimension = 2U;
    config.epsilon = 0.2;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_observe(cache, first, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_box_is_novel(cache, covered, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_box_observe(cache, bridge, NULL), FUTCACHE_OK);
    TEST_STATUS(futcache_box_is_novel(cache, distant, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_box_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == UINT64_C(2));
    TEST_ASSERT(stats.novel_observations == UINT64_C(2));
    TEST_ASSERT(stats.box_count == 2U && stats.peak_box_count == 2U);
    TEST_STATUS(futcache_box_validate(cache), FUTCACHE_OK);
    futcache_box_destroy(cache);
    return true;
}

static bool test_box_domain_and_query_purity(void)
{
    double lower[] = {-1.0, -1.0, -1.0};
    double upper[] = {1.0, 1.0, 1.0};
    double point[] = {0.0, 0.0, 0.0};
    double outside[] = {2.0, 0.0, 0.0};
    double nan_point[] = {NAN, 0.0, 0.0};
    futcache_box_config_t config;
    futcache_box_stats_t before;
    futcache_box_stats_t after;
    futcache_box_t *cache = NULL;
    bool novel = false;

    futcache_box_config_init(&config);
    config.dimension = 3U;
    config.epsilon = 0.1;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_get_stats(cache, &before), FUTCACHE_OK);
    TEST_STATUS(futcache_box_is_novel(cache, point, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_box_get_stats(cache, &after), FUTCACHE_OK);
    TEST_ASSERT(memcmp(&before, &after, sizeof(before)) == 0);
    TEST_STATUS(futcache_box_observe(cache, outside, &novel), FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_box_is_novel(cache, nan_point, &novel), FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_box_observe(cache, point, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_box_clear(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_get_stats(cache, &after), FUTCACHE_OK);
    TEST_ASSERT(after.observations == UINT64_C(0));
    TEST_ASSERT(after.novel_observations == UINT64_C(0));
    TEST_ASSERT(after.box_count == 0U && after.peak_box_count == 0U);
    TEST_STATUS(futcache_box_is_novel(cache, point, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_box_destroy(cache);
    return true;
}

static bool test_box_allocation_failure_atomicity(void)
{
    double lower[] = {0.0, 0.0};
    double upper[] = {1.0, 1.0};
    double point[] = {0.5, 0.5};
    box_allocator_context_t allocator_context = {8U, 0U};
    futcache_box_config_t config;
    futcache_box_stats_t before;
    futcache_box_stats_t after;
    futcache_box_t *cache = NULL;
    bool novel = false;

    futcache_box_config_init(&config);
    config.dimension = 2U;
    config.epsilon = 0.1;
    config.domain_min = lower;
    config.domain_max = upper;
    config.allocator.allocate = box_allocate;
    config.allocator.deallocate = box_deallocate;
    config.allocator.context = &allocator_context;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_get_stats(cache, &before), FUTCACHE_OK);
    allocator_context.remaining = 0U;
    TEST_STATUS(futcache_box_observe(cache, point, &novel), FUTCACHE_ERROR_OUT_OF_MEMORY);
    TEST_STATUS(futcache_box_get_stats(cache, &after), FUTCACHE_OK);
    TEST_ASSERT(memcmp(&before, &after, sizeof(before)) == 0);
    allocator_context.remaining = 8U;
    TEST_STATUS(futcache_box_observe(cache, point, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    futcache_box_destroy(cache);
    TEST_ASSERT(allocator_context.active == 0U);
    return true;
}

static bool test_box_validate_telemetry(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double first[] = {0.2};
    double second[] = {0.8};
    double covered[] = {0.25};
    futcache_box_config_t config;
    futcache_box_stats_t stats;
    futcache_box_t *cache = NULL;
    bool novel = false;

    futcache_box_config_init(&config);
    config.dimension = 1U;
    config.epsilon = 0.1;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_box_create(&config, &cache), FUTCACHE_OK);

    TEST_STATUS(futcache_box_validate(cache), FUTCACHE_OK);

    TEST_STATUS(futcache_box_observe(cache, first, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_box_observe(cache, second, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    /* covered lies inside the first box; redundant, no new box. */
    TEST_STATUS(futcache_box_observe(cache, covered, &novel), FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_box_validate(cache), FUTCACHE_OK);

    TEST_STATUS(futcache_box_get_stats(cache, &stats), FUTCACHE_OK);
    uint64_t generation_before = stats.generation;
    TEST_STATUS(futcache_box_clear(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_box_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.generation == generation_before + UINT64_C(1));
    TEST_ASSERT(stats.observations == UINT64_C(0));
    TEST_ASSERT(stats.novel_observations == UINT64_C(0));
    TEST_ASSERT(stats.box_count == 0U && stats.peak_box_count == 0U);
    TEST_STATUS(futcache_box_validate(cache), FUTCACHE_OK);

    futcache_box_destroy(cache);
    return true;
}

int box_test_suite(void)
{
    static const test_case_t tests[] = {
        {"box configuration", test_box_configuration},
        {"exact hand-checked L_inf union", test_box_hand_checked_union},
        {"domain validation and query purity", test_box_domain_and_query_purity},
        {"allocation failure atomicity", test_box_allocation_failure_atomicity},
        {"telemetry and containment validation", test_box_validate_telemetry}
    };
    return run_test_cases("box", tests, sizeof(tests) / sizeof(tests[0]));
}

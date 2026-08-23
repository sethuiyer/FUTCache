#include "test.h"

#include "futcache/crdt.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct crdt_allocator_context {
    size_t remaining;
    size_t active;
} crdt_allocator_context_t;

static void *crdt_test_allocate(void *opaque, size_t size)
{
    crdt_allocator_context_t *context = opaque;
    if (context->remaining == 0U) return NULL;
    context->remaining--;
    void *memory = malloc(size);
    if (memory != NULL) context->active++;
    return memory;
}

static void crdt_test_deallocate(void *opaque, void *pointer)
{
    crdt_allocator_context_t *context = opaque;
    if (pointer != NULL) {
        if (context->active == 0U) abort();
        context->active--;
        free(pointer);
    }
}

static bool test_crdt_configuration(void)
{
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.5};

    futcache_crdt_config_init(&config);
    TEST_ASSERT(config.dimension == 1U);
    TEST_ASSERT(config.anchor_count == 1U);

    config.dimension = 1U;
    config.anchor_count = 1U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_validate(cache), FUTCACHE_OK);
    futcache_crdt_destroy(cache);

    futcache_crdt_config_init(&config);
    config.dimension = 0U;
    config.anchor_count = 1U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache),
                FUTCACHE_ERROR_INVALID_ARGUMENT);

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 0U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache),
                FUTCACHE_ERROR_INVALID_ARGUMENT);

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 1U;
    config.anchors = NULL;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache),
                FUTCACHE_ERROR_INVALID_ARGUMENT);

    /* Anchor outside the domain is rejected. */
    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 1U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    anchors[0] = 2.0; /* > domain_max 1.0 */
    TEST_STATUS(futcache_crdt_create(&config, &cache),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    anchors[0] = 0.5;
    return true;
}

static bool test_crdt_quantize_hand_checked(void)
{
    double lower[] = {0.0};
    double upper[] = {2.0};
    double anchors[] = {0.2, 1.8};
    double point[] = {0.0};
    double outside[] = {3.0};
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    size_t cell = 99U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);

    point[0] = 0.1;
    TEST_STATUS(futcache_crdt_quantize(cache, point, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 0U);

    point[0] = 1.0; /* equidistant 0.8 from both; tie -> smallest index */
    TEST_STATUS(futcache_crdt_quantize(cache, point, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 0U);

    point[0] = 1.5;
    TEST_STATUS(futcache_crdt_quantize(cache, point, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 1U);

    point[0] = 2.0;
    TEST_STATUS(futcache_crdt_quantize(cache, point, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 1U);

    TEST_STATUS(futcache_crdt_quantize(cache, outside, &cell),
                FUTCACHE_ERROR_OUT_OF_RANGE);
    futcache_crdt_destroy(cache);
    return true;
}

static bool test_crdt_observe_fill_and_reuse(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.0, 1.0};
    double point[] = {0.0};
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    futcache_crdt_stats_t stats;
    bool novel = false;
    size_t cell = 99U;
    const void *payload = NULL;
    size_t payload_length = 0U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);

    point[0] = 0.1;
    TEST_STATUS(futcache_crdt_observe(cache, point, "a", 1U, &novel, &cell),
                FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_ASSERT(cell == 0U);

    point[0] = 0.2; /* same cell; semantic HIT */
    TEST_STATUS(futcache_crdt_observe(cache, point, "b", 1U, &novel, &cell),
                FUTCACHE_OK);
    TEST_ASSERT(!novel);
    TEST_ASSERT(cell == 0U);

    TEST_STATUS(futcache_crdt_get_payload(cache, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 1U);
    TEST_ASSERT(payload != NULL && memcmp(payload, "a", 1U) == 0);

    TEST_STATUS(futcache_crdt_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == UINT64_C(2));
    TEST_ASSERT(stats.novel_observations == UINT64_C(1));
    TEST_ASSERT(stats.occupied_cells == 1U);

    point[0] = 0.9;
    TEST_STATUS(futcache_crdt_observe(cache, point, "c", 1U, &novel, &cell),
                FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_ASSERT(cell == 1U);

    TEST_STATUS(futcache_crdt_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.occupied_cells == 2U);
    TEST_STATUS(futcache_crdt_validate(cache), FUTCACHE_OK);
    futcache_crdt_destroy(cache);
    return true;
}

static bool test_crdt_merge_priority_and_lattice(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.0, 1.0};
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    futcache_crdt_update_t update;
    futcache_crdt_stats_t stats;
    double point_a[] = {0.1};
    double point_b[] = {0.15};
    const void *payload = NULL;
    size_t payload_length = 0U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);

    /* Adopt into empty cell. */
    update.cell = 0U;
    update.point = point_a;
    update.payload = "low";
    update.payload_length = 3U;
    update.priority = 10U;
    TEST_STATUS(futcache_crdt_merge(cache, &update, 1U), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_get_payload(cache, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 3U && memcmp(payload, "low", 3U) == 0);

    /* Higher priority replaces. */
    update.point = point_b;
    update.payload = "high";
    update.payload_length = 4U;
    update.priority = 20U;
    TEST_STATUS(futcache_crdt_merge(cache, &update, 1U), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_get_payload(cache, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 4U && memcmp(payload, "high", 4U) == 0);

    /* Lower priority loses. */
    update.point = point_a;
    update.payload = "low2";
    update.payload_length = 4U;
    update.priority = 5U;
    TEST_STATUS(futcache_crdt_merge(cache, &update, 1U), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_get_payload(cache, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 4U && memcmp(payload, "high", 4U) == 0);

    /* Idempotence: re-merging the winning update is a no-op. */
    update.point = point_b;
    update.payload = "high";
    update.payload_length = 4U;
    update.priority = 20U;
    TEST_STATUS(futcache_crdt_merge(cache, &update, 1U), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.occupied_cells == 1U);

    TEST_STATUS(futcache_crdt_validate(cache), FUTCACHE_OK);
    futcache_crdt_destroy(cache);
    return true;
}

static bool test_crdt_merge_convergence(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.0, 1.0};
    futcache_crdt_config_t config;
    futcache_crdt_t *a = NULL;
    futcache_crdt_t *b = NULL;
    futcache_crdt_update_t updates[2];
    futcache_crdt_stats_t stats;
    double point[] = {0.0};
    bool novel = false;
    size_t cell = 0U;
    size_t count = 0U;
    const void *payload = NULL;
    size_t payload_length = 0U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &a), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_create(&config, &b), FUTCACHE_OK);

    /* a discovers cell 0, b discovers cell 1. */
    point[0] = 0.1;
    TEST_STATUS(futcache_crdt_observe(a, point, "a", 1U, &novel, &cell),
                FUTCACHE_OK);
    point[0] = 0.9;
    TEST_STATUS(futcache_crdt_observe(b, point, "b", 1U, &novel, &cell),
                FUTCACHE_OK);

    /* Exchange snapshots in both directions. */
    count = 2U;
    TEST_STATUS(futcache_crdt_snapshot(a, updates, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 1U);
    TEST_STATUS(futcache_crdt_merge(b, updates, count), FUTCACHE_OK);

    count = 2U;
    TEST_STATUS(futcache_crdt_snapshot(b, updates, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 2U);
    TEST_STATUS(futcache_crdt_merge(a, updates, count), FUTCACHE_OK);

    TEST_STATUS(futcache_crdt_get_stats(a, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.occupied_cells == 2U);
    TEST_STATUS(futcache_crdt_get_stats(b, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.occupied_cells == 2U);

    TEST_STATUS(futcache_crdt_get_payload(a, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 1U && memcmp(payload, "a", 1U) == 0);
    TEST_STATUS(futcache_crdt_get_payload(a, 1U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 1U && memcmp(payload, "b", 1U) == 0);
    TEST_STATUS(futcache_crdt_get_payload(b, 0U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 1U && memcmp(payload, "a", 1U) == 0);
    TEST_STATUS(futcache_crdt_get_payload(b, 1U, &payload, &payload_length),
                FUTCACHE_OK);
    TEST_ASSERT(payload_length == 1U && memcmp(payload, "b", 1U) == 0);

    TEST_STATUS(futcache_crdt_validate(a), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_validate(b), FUTCACHE_OK);
    futcache_crdt_destroy(a);
    futcache_crdt_destroy(b);
    return true;
}

static bool test_crdt_snapshot_and_clear(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.0, 1.0};
    double point[] = {0.0};
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    futcache_crdt_update_t updates[2];
    futcache_crdt_stats_t stats;
    bool novel = false;
    size_t cell = 0U;
    size_t count = 0U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);

    point[0] = 0.1;
    TEST_STATUS(futcache_crdt_observe(cache, point, "a", 1U, &novel, &cell),
                FUTCACHE_OK);
    point[0] = 0.9;
    TEST_STATUS(futcache_crdt_observe(cache, point, "b", 1U, &novel, &cell),
                FUTCACHE_OK);

    /* Query required capacity. */
    count = 0U;
    TEST_STATUS(futcache_crdt_snapshot(cache, NULL, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 2U);

    /* Insufficient capacity. */
    count = 1U;
    TEST_STATUS(futcache_crdt_snapshot(cache, updates, &count),
                FUTCACHE_ERROR_BUFFER_TOO_SMALL);
    TEST_ASSERT(count == 2U);

    /* Full snapshot. */
    count = 2U;
    TEST_STATUS(futcache_crdt_snapshot(cache, updates, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 2U);
    TEST_ASSERT((updates[0].cell == 0U && updates[1].cell == 1U) ||
                (updates[0].cell == 1U && updates[1].cell == 0U));
    TEST_ASSERT(updates[0].point != NULL && updates[0].payload != NULL);
    TEST_ASSERT(updates[1].point != NULL && updates[1].payload != NULL);

    /* Clear resets observations and cells but bumps generation. */
    TEST_STATUS(futcache_crdt_get_stats(cache, &stats), FUTCACHE_OK);
    uint64_t generation_before = stats.generation;
    TEST_STATUS(futcache_crdt_clear(cache), FUTCACHE_OK);
    TEST_STATUS(futcache_crdt_get_stats(cache, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.occupied_cells == 0U);
    TEST_ASSERT(stats.observations == UINT64_C(0));
    TEST_ASSERT(stats.novel_observations == UINT64_C(0));
    TEST_ASSERT(stats.generation == generation_before + UINT64_C(1));
    TEST_STATUS(futcache_crdt_validate(cache), FUTCACHE_OK);
    futcache_crdt_destroy(cache);
    return true;
}

static bool test_crdt_allocation_failure_atomicity(void)
{
    double lower[] = {0.0};
    double upper[] = {1.0};
    double anchors[] = {0.0, 1.0};
    double point[] = {0.1};
    crdt_allocator_context_t allocator_context = {8U, 0U};
    futcache_crdt_config_t config;
    futcache_crdt_t *cache = NULL;
    futcache_crdt_stats_t before;
    futcache_crdt_stats_t after;
    bool novel = false;
    size_t cell = 99U;

    futcache_crdt_config_init(&config);
    config.dimension = 1U;
    config.anchor_count = 2U;
    config.anchors = anchors;
    config.epsilon = 0.5;
    config.domain_min = lower;
    config.domain_max = upper;
    config.allocator.allocate = crdt_test_allocate;
    config.allocator.deallocate = crdt_test_deallocate;
    config.allocator.context = &allocator_context;
    TEST_STATUS(futcache_crdt_create(&config, &cache), FUTCACHE_OK);

    TEST_STATUS(futcache_crdt_get_stats(cache, &before), FUTCACHE_OK);
    allocator_context.remaining = 0U;
    TEST_STATUS(futcache_crdt_observe(cache, point, "a", 1U, &novel, &cell),
                FUTCACHE_ERROR_OUT_OF_MEMORY);
    TEST_STATUS(futcache_crdt_get_stats(cache, &after), FUTCACHE_OK);
    TEST_ASSERT(memcmp(&before, &after, sizeof(before)) == 0);

    allocator_context.remaining = 8U;
    TEST_STATUS(futcache_crdt_observe(cache, point, "a", 1U, &novel, &cell),
                FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_ASSERT(cell == 0U);

    futcache_crdt_destroy(cache);
    TEST_ASSERT(allocator_context.active == 0U);
    return true;
}

int crdt_test_suite(void)
{
    static const test_case_t tests[] = {
        {"crdt configuration", test_crdt_configuration},
        {"hand-checked Voronoi quantization", test_crdt_quantize_hand_checked},
        {"observe fill and semantic reuse", test_crdt_observe_fill_and_reuse},
        {"merge priority and idempotence", test_crdt_merge_priority_and_lattice},
        {"two-way gossip convergence", test_crdt_merge_convergence},
        {"snapshot roundtrip and clear", test_crdt_snapshot_and_clear},
        {"allocation failure atomicity", test_crdt_allocation_failure_atomicity},
    };
    return run_test_cases("crdt", tests, sizeof(tests) / sizeof(tests[0]));
}

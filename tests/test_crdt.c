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

/* ---- Anchor construction: halton / grid / coverage ---- */

static bool test_crdt_halton_anchors(void)
{
    double lo1[] = {0.0};
    double hi1[] = {1.0};
    double a1[4];
    double lo2[] = {0.0, 0.0};
    double hi2[] = {1.0, 1.0};
    double a2[8];

    /* 1-D, base 2 van der Corput: {0.0, 0.5, 0.25, 0.75}. */
    TEST_STATUS(futcache_crdt_generate_halton_anchors(1U, lo1, hi1, 4U, a1),
                FUTCACHE_OK);
    TEST_NEAR(a1[0], 0.00, 1e-12);
    TEST_NEAR(a1[1], 0.50, 1e-12);
    TEST_NEAR(a1[2], 0.25, 1e-12);
    TEST_NEAR(a1[3], 0.75, 1e-12);

    /* 2-D: coordinate 0 base 2, coordinate 1 base 3. */
    TEST_STATUS(futcache_crdt_generate_halton_anchors(2U, lo2, hi2, 4U, a2),
                FUTCACHE_OK);
    TEST_NEAR(a2[0], 0.0, 1e-12); /* phi_2(0) */
    TEST_NEAR(a2[1], 0.0, 1e-12); /* phi_3(0) */
    TEST_NEAR(a2[2], 0.5, 1e-12); /* phi_2(1) */
    TEST_NEAR(a2[3], 1.0 / 3.0, 1e-12); /* phi_3(1) */
    TEST_NEAR(a2[4], 0.25, 1e-12); /* phi_2(2) */
    TEST_NEAR(a2[5], 2.0 / 3.0, 1e-12); /* phi_3(2) */

    /* In-domain for a shifted box. */
    double lo3[] = {10.0, -5.0};
    double hi3[] = {20.0, 5.0};
    double a3[16];
    TEST_STATUS(futcache_crdt_generate_halton_anchors(2U, lo3, hi3, 8U, a3),
                FUTCACHE_OK);
    for (size_t k = 0U; k < 8U; ++k) {
        TEST_ASSERT(a3[2U * k] >= 10.0 && a3[2U * k] <= 20.0);
        TEST_ASSERT(a3[2U * k + 1U] >= -5.0 && a3[2U * k + 1U] <= 5.0);
    }

    /* Invalid arguments. */
    TEST_STATUS(futcache_crdt_generate_halton_anchors(0U, lo1, hi1, 4U, a1),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_crdt_generate_halton_anchors(1U, lo1, hi1, 0U, a1),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_crdt_generate_halton_anchors(1U, lo1, hi1, 4U, NULL),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_crdt_grid_anchors(void)
{
    double lo1[] = {0.0};
    double hi1[] = {1.0};
    double g1[4];
    size_t count = 0U;

    TEST_STATUS(futcache_crdt_generate_grid_anchors(1U, lo1, hi1, 3U, g1,
                &count), FUTCACHE_OK);
    TEST_ASSERT(count == 3U);
    TEST_NEAR(g1[0], 1.0 / 6.0, 1e-12);
    TEST_NEAR(g1[1], 0.5, 1e-12);
    TEST_NEAR(g1[2], 5.0 / 6.0, 1e-12);

    /* 2-D, 2 cells per axis: 4 anchors at quadrant centers. */
    double lo2[] = {0.0, 0.0};
    double hi2[] = {1.0, 1.0};
    double g2[8];
    TEST_STATUS(futcache_crdt_generate_grid_anchors(2U, lo2, hi2, 2U, g2,
                &count), FUTCACHE_OK);
    TEST_ASSERT(count == 4U);
    TEST_NEAR(g2[0], 0.25, 1e-12);
    TEST_NEAR(g2[1], 0.25, 1e-12);
    TEST_NEAR(g2[2], 0.75, 1e-12);
    TEST_NEAR(g2[3], 0.25, 1e-12);
    TEST_NEAR(g2[4], 0.25, 1e-12);
    TEST_NEAR(g2[5], 0.75, 1e-12);
    TEST_NEAR(g2[6], 0.75, 1e-12);
    TEST_NEAR(g2[7], 0.75, 1e-12);

    TEST_STATUS(futcache_crdt_generate_grid_anchors(1U, lo1, hi1, 0U, g1,
                &count), FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_crdt_grid_covering_radius(void)
{
    double lo1[] = {0.0};
    double hi1[] = {1.0};
    double lo2[] = {0.0, 0.0};
    double hi2[] = {1.0, 1.0};
    double lo3[] = {0.0};
    double hi3[] = {2.0};
    double radius = 0.0;

    /* L_inf, 1-D [0,1], n=2: rho = 1/(2*2) = 0.25. */
    TEST_STATUS(futcache_crdt_grid_covering_radius(1U, lo1, hi1, 2U, NULL,
                NULL, &radius), FUTCACHE_OK);
    TEST_NEAR(radius, 0.25, 1e-12);

    /* L2, 2-D [0,1]^2, n=2: rho = sqrt(2)/(2*2). */
    TEST_STATUS(futcache_crdt_grid_covering_radius(2U, lo2, hi2, 2U,
                futcache_distance_l2, NULL, &radius), FUTCACHE_OK);
    TEST_NEAR(radius, sqrt(2.0) / 4.0, 1e-12);

    /* L1, 1-D [0,2], n=4: rho = 2/(2*4) = 0.25. */
    TEST_STATUS(futcache_crdt_grid_covering_radius(1U, lo3, hi3, 4U,
                futcache_distance_l1, NULL, &radius), FUTCACHE_OK);
    TEST_NEAR(radius, 0.25, 1e-12);

    /* Cosine has no grid bound. */
    TEST_STATUS(futcache_crdt_grid_covering_radius(1U, lo1, hi1, 2U,
                futcache_distance_cosine, NULL, &radius),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_crdt_estimate_covering_radius(void)
{
    double lo[] = {0.0};
    double hi[] = {1.0};
    double one[] = {0.5};
    double grid[8];
    size_t count = 0U;
    double radius = 0.0;

    /* Single anchor at 0.5: true radius 0.5; estimate approaches it. */
    TEST_STATUS(futcache_crdt_estimate_covering_radius(one, 1U, 1U, lo, hi,
                NULL, NULL, 65536U, &radius), FUTCACHE_OK);
    TEST_NEAR(radius, 0.5, 0.01);
    TEST_ASSERT(radius <= 0.5 + 1e-12); /* lower bound never exceeds true */

    /* A 4-cell grid has certified radius 0.125; estimate is a lower bound. */
    TEST_STATUS(futcache_crdt_generate_grid_anchors(1U, lo, hi, 4U, grid,
                &count), FUTCACHE_OK);
    TEST_ASSERT(count == 4U);
    TEST_STATUS(futcache_crdt_estimate_covering_radius(grid, 4U, 1U, lo, hi,
                NULL, NULL, 65536U, &radius), FUTCACHE_OK);
    TEST_ASSERT(radius <= 0.125 + 1e-12);
    TEST_ASSERT(radius >= 0.12);

    /* Invalid arguments. */
    TEST_STATUS(futcache_crdt_estimate_covering_radius(one, 0U, 1U, lo, hi,
                NULL, NULL, 8U, &radius), FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_crdt_estimate_covering_radius(one, 1U, 1U, lo, hi,
                NULL, NULL, 0U, &radius), FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_crdt_safe_anchors_grid(void)
{
    double lo[] = {0.0};
    double hi[] = {1.0};
    double anchors[16];
    size_t count = 0U;
    double radius = 0.0;

    /* eps=0.5 -> target 0.25; needs n=2 (2 anchors), certified 0.25. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.5, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_GRID,
                16U, 0U, anchors, &count, &radius), FUTCACHE_OK);
    TEST_ASSERT(count == 2U);
    TEST_NEAR(radius, 0.25, 1e-12);
    TEST_ASSERT(radius <= 0.25 + 1e-12); /* contract: rho <= eps/2 */
    TEST_NEAR(anchors[0], 0.25, 1e-12);
    TEST_NEAR(anchors[1], 0.75, 1e-12);

    /* Budget too small: needs 2 anchors but only 1 allowed. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.5, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_GRID,
                1U, 0U, anchors, &count, &radius), FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_ASSERT(count == 2U);
    TEST_NEAR(radius, 0.25, 1e-12);

    /* Cosine cannot be grid-certified. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.5, lo, hi, futcache_distance_cosine, NULL,
                FUTCACHE_CRDT_ANCHOR_GRID, 16U, 0U, anchors, &count, &radius),
                FUTCACHE_ERROR_INVALID_ARGUMENT);

    /* eps <= 0 is invalid. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.0, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_GRID,
                16U, 0U, anchors, &count, &radius),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    return true;
}

static bool test_crdt_safe_anchors_halton(void)
{
    double lo[] = {0.0};
    double hi[] = {1.0};
    double anchors[16];
    size_t count = 0U;
    double radius = 0.0;

    /* eps=0.5 -> target 0.25; van der Corput {0,.5,.25,.75} covers at 0.25. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.5, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_HALTON,
                16U, 65536U, anchors, &count, &radius), FUTCACHE_OK);
    TEST_ASSERT(count == 4U);
    TEST_ASSERT(radius <= 0.25 + 1e-9);
    TEST_ASSERT(radius >= 0.24);
    for (size_t k = 0U; k < count; ++k) {
        TEST_ASSERT(anchors[k] >= 0.0 && anchors[k] <= 1.0);
    }

    /* Too small a budget: eps=0.01 (target 0.005) needs far more anchors. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.01, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_HALTON,
                4U, 65536U, anchors, &count, &radius),
                FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_ASSERT(count == 0U);
    TEST_ASSERT(radius > 0.005);

    /* probe_count must be >= 1 for HALTON. */
    TEST_STATUS(futcache_crdt_generate_safe_anchors(
                1U, 0.5, lo, hi, NULL, NULL, FUTCACHE_CRDT_ANCHOR_HALTON,
                16U, 0U, anchors, &count, &radius),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
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
        {"halton anchor generation", test_crdt_halton_anchors},
        {"grid anchor generation", test_crdt_grid_anchors},
        {"certified grid covering radius", test_crdt_grid_covering_radius},
        {"estimated covering radius", test_crdt_estimate_covering_radius},
        {"safe anchors grid", test_crdt_safe_anchors_grid},
        {"safe anchors halton", test_crdt_safe_anchors_halton},
    };
    return run_test_cases("crdt", tests, sizeof(tests) / sizeof(tests[0]));
}

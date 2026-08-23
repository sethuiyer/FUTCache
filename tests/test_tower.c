#include "test.h"

#include "futcache/tower.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct tower_thread_context {
    futcache_tower_t *tower;
    size_t iterations;
    futcache_status_t status;
} tower_thread_context_t;

typedef struct tower_allocator_context {
    size_t allocations_remaining;
    size_t active_allocations;
} tower_allocator_context_t;

static void *tower_test_allocate(void *opaque, size_t size)
{
    tower_allocator_context_t *context = opaque;
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

static void tower_test_deallocate(void *opaque, void *pointer)
{
    tower_allocator_context_t *context = opaque;
    if (pointer != NULL) {
        if (context->active_allocations == 0U) {
            abort();
        }
        context->active_allocations--;
        free(pointer);
    }
}

static uint64_t tower_random_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static bool test_tower_hand_checked_traversal(void)
{
    static const double traversal[] = {0.8, 0.1, 0.7, 0.2, 0.4, 0.4, 0.1, 0.9};
    static const uint8_t expected[][2] = {
        {1U, 1U}, {1U, 1U}, {0U, 1U}, {0U, 0U},
        {0U, 1U}, {0U, 0U}, {0U, 0U}, {0U, 0U}
    };
    static const size_t discovery_level_zero[] = {1U, 0U};
    static const size_t discovery_level_one[] = {3U, 0U, 2U, 1U};
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    size_t index;
    size_t count;

    futcache_tower_config_init(&config);
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
    for (index = 0U; index < sizeof(traversal) / sizeof(traversal[0]); ++index) {
        uint8_t novelty[2] = {9U, 9U};
        TEST_STATUS(futcache_tower_observe(tower, traversal[index], novelty, 2U),
            FUTCACHE_OK);
        TEST_ASSERT(memcmp(novelty, expected[index], sizeof(novelty)) == 0);
    }
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);

    for (index = 0U; index < 2U; ++index) {
        size_t cell;
        TEST_STATUS(futcache_tower_discovery_at(tower, 0U, index, &cell), FUTCACHE_OK);
        TEST_ASSERT(cell == discovery_level_zero[index]);
    }
    for (index = 0U; index < 4U; ++index) {
        size_t cell;
        TEST_STATUS(futcache_tower_discovery_at(tower, 1U, index, &cell), FUTCACHE_OK);
        TEST_ASSERT(cell == discovery_level_one[index]);
        TEST_STATUS(futcache_tower_select_occupied(tower, 1U, index, &cell), FUTCACHE_OK);
        TEST_ASSERT(cell == index);
    }
    TEST_STATUS(futcache_tower_prefix_count(tower, 1U, 1U, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 2U);
    TEST_STATUS(futcache_tower_prefix_count(tower, 1U, 3U, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 4U);

    {
        futcache_tower_level_info_t coarse;
        futcache_tower_level_info_t fine;
        futcache_tower_stats_t stats;
        TEST_STATUS(futcache_tower_level_info(tower, 0U, &coarse), FUTCACHE_OK);
        TEST_STATUS(futcache_tower_level_info(tower, 1U, &fine), FUTCACHE_OK);
        TEST_ASSERT(coarse.cell_count == 2U && coarse.discovered_count == 2U);
        TEST_ASSERT(fine.cell_count == 4U && fine.discovered_count == 4U);
        TEST_STATUS(futcache_tower_get_stats(tower, &stats), FUTCACHE_OK);
        TEST_ASSERT(stats.observations == UINT64_C(8));
        TEST_ASSERT(stats.total_cells == 6U);
        TEST_ASSERT(stats.total_discoveries == 6U);
    }

    futcache_tower_destroy(tower);
    return true;
}

static bool test_tower_projection_compatibility(void)
{
    static const double traversal[] = {0.8, 0.1, 0.7, 0.2, 0.4};
    bool parent_seen[2] = {false, false};
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    futcache_tower_level_info_t coarse;
    futcache_tower_level_info_t fine;
    size_t projected_count = 0U;
    size_t index;

    futcache_tower_config_init(&config);
    TEST_ASSERT(config.level_count == 2U && config.root_cells == 2U);
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
    for (index = 0U; index < sizeof(traversal) / sizeof(traversal[0]); ++index) {
        size_t coarse_cell;
        size_t fine_cell;
        TEST_STATUS(futcache_tower_cell_index(tower, 0U, traversal[index],
            &coarse_cell), FUTCACHE_OK);
        TEST_STATUS(futcache_tower_cell_index(tower, 1U, traversal[index],
            &fine_cell), FUTCACHE_OK);
        TEST_ASSERT(coarse_cell == fine_cell / 2U);
        TEST_STATUS(futcache_tower_observe(tower, traversal[index], NULL, 0U),
            FUTCACHE_OK);
    }

    TEST_STATUS(futcache_tower_level_info(tower, 0U, &coarse), FUTCACHE_OK);
    TEST_STATUS(futcache_tower_level_info(tower, 1U, &fine), FUTCACHE_OK);
    for (index = 0U; index < fine.discovered_count; ++index) {
        size_t fine_cell;
        size_t parent;
        TEST_STATUS(futcache_tower_discovery_at(tower, 1U, index, &fine_cell),
            FUTCACHE_OK);
        parent = fine_cell / 2U;
        TEST_ASSERT(parent < sizeof(parent_seen) / sizeof(parent_seen[0]));
        if (!parent_seen[parent]) {
            size_t coarse_cell;
            TEST_ASSERT(projected_count < coarse.discovered_count);
            TEST_STATUS(futcache_tower_discovery_at(tower, 0U, projected_count,
                &coarse_cell), FUTCACHE_OK);
            TEST_ASSERT(coarse_cell == parent);
            parent_seen[parent] = true;
            projected_count++;
        }
    }
    TEST_ASSERT(projected_count == coarse.discovered_count);
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);
    futcache_tower_destroy(tower);
    return true;
}

static bool test_tower_boundaries_and_validation(void)
{
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    size_t cell;
    uint8_t novelty[2];

    futcache_tower_config_init(&config);
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
    TEST_STATUS(futcache_tower_cell_index(tower, 1U, 0.0, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 0U);
    TEST_STATUS(futcache_tower_cell_index(tower, 1U, 0.25, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 1U);
    TEST_STATUS(futcache_tower_cell_index(tower, 1U, 0.5, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 2U);
    TEST_STATUS(futcache_tower_cell_index(tower, 1U, 1.0, &cell), FUTCACHE_OK);
    TEST_ASSERT(cell == 3U);
    TEST_STATUS(futcache_tower_query(tower, 0.2, novelty, 1U),
        FUTCACHE_ERROR_BUFFER_TOO_SMALL);
    TEST_STATUS(futcache_tower_query(tower, 0.2, NULL, 0U),
        FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_tower_query(tower, -0.1, novelty, 2U),
        FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_tower_discovery_at(tower, 0U, 0U, &cell),
        FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);
    futcache_tower_destroy(tower);

    config.level_count = 0U;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_ERROR_INVALID_ARGUMENT);
    config.level_count = 2U;
    config.root_cells = 0U;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_ERROR_INVALID_ARGUMENT);
    config.level_count = 1U;
    config.root_cells = SIZE_MAX;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_ERROR_OUT_OF_RANGE);
    return true;
}

static bool test_tower_allocation_failure_cleanup(void)
{
    size_t allocation_limit;

    /* Default two-level construction performs eight owned allocations. */
    for (allocation_limit = 0U; allocation_limit < 8U; ++allocation_limit) {
        tower_allocator_context_t context = {allocation_limit, 0U};
        futcache_tower_config_t config;
        futcache_tower_t *tower = NULL;

        futcache_tower_config_init(&config);
        config.allocator.allocate = tower_test_allocate;
        config.allocator.deallocate = tower_test_deallocate;
        config.allocator.context = &context;
        TEST_STATUS(futcache_tower_create(&config, &tower),
            FUTCACHE_ERROR_OUT_OF_MEMORY);
        TEST_ASSERT(tower == NULL);
        TEST_ASSERT(context.active_allocations == 0U);
    }

    {
        tower_allocator_context_t context = {8U, 0U};
        futcache_tower_config_t config;
        futcache_tower_t *tower = NULL;
        futcache_tower_config_init(&config);
        config.allocator.allocate = tower_test_allocate;
        config.allocator.deallocate = tower_test_deallocate;
        config.allocator.context = &context;
        TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
        TEST_ASSERT(context.active_allocations == 8U);
        futcache_tower_destroy(tower);
        TEST_ASSERT(context.active_allocations == 0U);
    }
    return true;
}

static bool test_tower_randomized_differential(void)
{
    enum { LEVELS = 6, ROOT_CELLS = 3, SAMPLES = 4000, MAX_CELLS = 96 };
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    uint8_t occupancy[LEVELS][MAX_CELLS];
    size_t discovery[LEVELS][MAX_CELLS];
    size_t discovery_count[LEVELS] = {0U};
    uint64_t random_state = UINT64_C(0x4d595df4d0f33173);
    size_t sample;
    size_t level;

    memset(occupancy, 0, sizeof(occupancy));
    futcache_tower_config_init(&config);
    config.level_count = LEVELS;
    config.root_cells = ROOT_CELLS;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);

    for (sample = 0U; sample < SAMPLES; ++sample) {
        double x = (double)(tower_random_next(&random_state) % 65537U) / 65536.0;
        uint8_t novelty[LEVELS];
        TEST_STATUS(futcache_tower_observe(tower, x, novelty, LEVELS), FUTCACHE_OK);
        for (level = 0U; level < LEVELS; ++level) {
            size_t cells = ROOT_CELLS << level;
            size_t cell;
            size_t expected_cell = x == 1.0
                ? cells - 1U
                : (size_t)(x * (double)cells);
            TEST_STATUS(futcache_tower_cell_index(tower, level, x, &cell), FUTCACHE_OK);
            TEST_ASSERT(cell < cells);
            TEST_ASSERT(cell == expected_cell);
            TEST_ASSERT(novelty[level] == (occupancy[level][cell] == 0U ? 1U : 0U));
            if (occupancy[level][cell] == 0U) {
                occupancy[level][cell] = 1U;
                discovery[level][discovery_count[level]++] = cell;
            }
            if (level > 0U && novelty[level - 1U] != 0U) {
                TEST_ASSERT(novelty[level] != 0U);
            }
        }
    }

    for (level = 0U; level < LEVELS; ++level) {
        size_t cells = ROOT_CELLS << level;
        size_t running = 0U;
        size_t cell;
        size_t ordinal = 0U;
        futcache_tower_level_info_t info;

        TEST_STATUS(futcache_tower_level_info(tower, level, &info), FUTCACHE_OK);
        TEST_ASSERT(info.cell_count == cells);
        TEST_ASSERT(info.discovered_count == discovery_count[level]);
        for (cell = 0U; cell < cells; ++cell) {
            size_t actual;
            running += occupancy[level][cell] != 0U ? 1U : 0U;
            TEST_STATUS(futcache_tower_prefix_count(tower, level, cell, &actual),
                FUTCACHE_OK);
            TEST_ASSERT(actual == running);
            if (occupancy[level][cell] != 0U) {
                TEST_STATUS(futcache_tower_select_occupied(tower, level, ordinal,
                    &actual), FUTCACHE_OK);
                TEST_ASSERT(actual == cell);
                ordinal++;
            }
        }
        for (cell = 0U; cell < discovery_count[level]; ++cell) {
            size_t actual;
            TEST_STATUS(futcache_tower_discovery_at(tower, level, cell, &actual),
                FUTCACHE_OK);
            TEST_ASSERT(actual == discovery[level][cell]);
        }
    }
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);

    /* Project fine discovery order to parents and remove repeated parents. */
    for (level = 1U; level < LEVELS; ++level) {
        uint8_t parent_seen[MAX_CELLS] = {0U};
        size_t projected_count = 0U;
        size_t index;
        for (index = 0U; index < discovery_count[level]; ++index) {
            size_t parent = discovery[level][index] / 2U;
            if (parent_seen[parent] == 0U) {
                TEST_ASSERT(projected_count < discovery_count[level - 1U]);
                TEST_ASSERT(discovery[level - 1U][projected_count] == parent);
                parent_seen[parent] = 1U;
                projected_count++;
            }
        }
        TEST_ASSERT(projected_count == discovery_count[level - 1U]);
    }

    futcache_tower_destroy(tower);
    return true;
}

static bool test_reciprocal_discovery_scaling(void)
{
    enum { LEVELS = 17, SAMPLE_COUNT = 65536 };
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    size_t sample;
    size_t level;
    double finest_dimension = 0.0;

    futcache_tower_config_init(&config);
    config.level_count = LEVELS;
    config.root_cells = 1U;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
    for (sample = 1U; sample <= SAMPLE_COUNT; ++sample) {
        TEST_STATUS(futcache_tower_observe(tower, 1.0 / (double)sample, NULL, 0U),
            FUTCACHE_OK);
    }

    for (level = 4U; level < LEVELS; ++level) {
        futcache_tower_level_info_t info;
        size_t cells = (size_t)1U << level;
        double root_cells = sqrt((double)cells);
        double dimension;

        TEST_STATUS(futcache_tower_level_info(tower, level, &info), FUTCACHE_OK);
        TEST_ASSERT(info.discovered_count >= (size_t)(root_cells / 2.0));
        TEST_ASSERT(info.discovered_count <= (size_t)(2.0 * root_cells + 4.0));
        dimension = log((double)info.discovered_count) /
            ((double)level * log(2.0));
        if (level + 1U == LEVELS) {
            finest_dimension = dimension;
        }
    }
    TEST_ASSERT(finest_dimension > 0.45 && finest_dimension < 0.70);
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);
    futcache_tower_destroy(tower);
    return true;
}

static void *tower_thread_main(void *opaque)
{
    tower_thread_context_t *context = opaque;
    size_t index;

    context->status = FUTCACHE_OK;
    for (index = 0U; index < context->iterations; ++index) {
        double x = ((double)(index % 64U) + 0.5) / 64.0;
        context->status = futcache_tower_observe(context->tower, x, NULL, 0U);
        if (context->status != FUTCACHE_OK) {
            break;
        }
    }
    return NULL;
}

static void *tower_query_thread_main(void *opaque)
{
    tower_thread_context_t *context = opaque;
    size_t index;

    context->status = FUTCACHE_OK;
    for (index = 0U; index < context->iterations; ++index) {
        double x = ((double)(index % 64U) + 0.5) / 64.0;
        uint8_t novelty[6];
        context->status = futcache_tower_query(context->tower, x, novelty,
            sizeof(novelty) / sizeof(novelty[0]));
        if (context->status != FUTCACHE_OK) {
            break;
        }
    }
    return NULL;
}

static bool test_tower_concurrency_and_clear(void)
{
    enum { THREAD_COUNT = 4, ITERATIONS = 1000 };
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    pthread_t threads[THREAD_COUNT];
    pthread_t readers[THREAD_COUNT];
    tower_thread_context_t contexts[THREAD_COUNT];
    tower_thread_context_t reader_contexts[THREAD_COUNT];
    futcache_tower_stats_t before;
    futcache_tower_stats_t after;
    size_t thread_index;

    futcache_tower_config_init(&config);
    config.level_count = 6U;
    TEST_STATUS(futcache_tower_create(&config, &tower), FUTCACHE_OK);
    for (thread_index = 0U; thread_index < THREAD_COUNT; ++thread_index) {
        contexts[thread_index].tower = tower;
        contexts[thread_index].iterations = ITERATIONS;
        TEST_ASSERT(pthread_create(&threads[thread_index], NULL,
            tower_thread_main, &contexts[thread_index]) == 0);
        reader_contexts[thread_index].tower = tower;
        reader_contexts[thread_index].iterations = ITERATIONS;
        reader_contexts[thread_index].status = FUTCACHE_ERROR_SYSTEM;
        TEST_ASSERT(pthread_create(&readers[thread_index], NULL,
            tower_query_thread_main, &reader_contexts[thread_index]) == 0);
    }
    for (thread_index = 0U; thread_index < THREAD_COUNT; ++thread_index) {
        TEST_ASSERT(pthread_join(threads[thread_index], NULL) == 0);
        TEST_STATUS(contexts[thread_index].status, FUTCACHE_OK);
        TEST_ASSERT(pthread_join(readers[thread_index], NULL) == 0);
        TEST_STATUS(reader_contexts[thread_index].status, FUTCACHE_OK);
    }
    TEST_STATUS(futcache_tower_get_stats(tower, &before), FUTCACHE_OK);
    TEST_ASSERT(before.observations == (uint64_t)(THREAD_COUNT * ITERATIONS));
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);
    TEST_STATUS(futcache_tower_clear(tower), FUTCACHE_OK);
    TEST_STATUS(futcache_tower_get_stats(tower, &after), FUTCACHE_OK);
    TEST_ASSERT(after.observations == UINT64_C(0));
    TEST_ASSERT(after.total_discoveries == 0U);
    TEST_ASSERT(after.generation == before.generation + UINT64_C(1));
    TEST_STATUS(futcache_tower_validate(tower), FUTCACHE_OK);
    futcache_tower_destroy(tower);
    return true;
}

int tower_test_suite(void)
{
    static const test_case_t tests[] = {
        {"hand-checked resolution tower", test_tower_hand_checked_traversal},
        {"discovery projection compatibility", test_tower_projection_compatibility},
        {"partition boundaries and validation", test_tower_boundaries_and_validation},
        {"allocation failure cleanup", test_tower_allocation_failure_cleanup},
        {"randomized Fenwick and discovery oracle", test_tower_randomized_differential},
        {"reciprocal discovery scaling", test_reciprocal_discovery_scaling},
        {"concurrency and clear", test_tower_concurrency_and_clear}
    };
    return run_test_cases("tower", tests, sizeof(tests) / sizeof(tests[0]));
}

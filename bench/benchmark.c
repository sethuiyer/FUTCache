#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "futcache/futcache.h"
#include "futcache/tower.h"

static double monotonic_seconds(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void require_success(futcache_status_t status, const char *operation)
{
    if (status != FUTCACHE_OK) {
        fprintf(stderr, "%s: %s\n", operation, futcache_status_string(status));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    enum {
        FRAGMENTED_INSERTS = 100000,
        QUERY_COUNT = 2000000,
        TOWER_OBSERVATIONS = 250000
    };
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_stats_t stats;
    uint32_t random_state = UINT32_C(0x6d2b79f5);
    size_t index;
    size_t redundant_queries = 0U;
    double start;
    double elapsed;

    futcache_config_init(&config);
    config.domain_max = 2.0 * (double)FRAGMENTED_INSERTS;
    config.epsilon = 0.0;
    require_success(futcache_create(&config, &cache), "create fragmented cache");

    start = monotonic_seconds();
    for (index = 0U; index < FRAGMENTED_INSERTS; ++index) {
        require_success(futcache_observe(cache, 2.0 * (double)index, NULL),
            "fragmented observe");
    }
    elapsed = monotonic_seconds() - start;
    require_success(futcache_validate(cache), "validate fragmented cache");
    require_success(futcache_get_stats(cache, &stats), "fragmented stats");
    printf("fragmented insert: %u operations, %.3f s, %.0f ops/s, %zu intervals, %zu bytes\n",
        (unsigned int)FRAGMENTED_INSERTS, elapsed,
        elapsed > 0.0 ? (double)FRAGMENTED_INSERTS / elapsed : 0.0,
        stats.interval_count, stats.memory_bytes);

    start = monotonic_seconds();
    for (index = 0U; index < QUERY_COUNT; ++index) {
        bool novel;
        size_t point_index;
        random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
        point_index = (size_t)(random_state % FRAGMENTED_INSERTS);
        require_success(futcache_is_novel(cache, 2.0 * (double)point_index, &novel),
            "fragmented query");
        if (!novel) {
            redundant_queries++;
        }
    }
    elapsed = monotonic_seconds() - start;
    printf("fragmented query:  %u operations, %.3f s, %.0f ops/s, %zu hits\n",
        (unsigned int)QUERY_COUNT, elapsed,
        elapsed > 0.0 ? (double)QUERY_COUNT / elapsed : 0.0,
        redundant_queries);

    {
        size_t snapshot_size = 0U;
        void *snapshot;
        require_success(futcache_serialize(cache, NULL, 0U, &snapshot_size),
            "snapshot size");
        snapshot = malloc(snapshot_size);
        if (snapshot == NULL) {
            fputs("snapshot allocation failed\n", stderr);
            return EXIT_FAILURE;
        }
        start = monotonic_seconds();
        require_success(futcache_serialize(cache, snapshot, snapshot_size,
            &snapshot_size), "snapshot");
        elapsed = monotonic_seconds() - start;
        printf("snapshot:          %zu bytes, %.3f s, %.1f MiB/s\n",
            snapshot_size, elapsed,
            elapsed > 0.0 ? ((double)snapshot_size / (1024.0 * 1024.0)) / elapsed : 0.0);
        free(snapshot);
    }
    futcache_destroy(cache);

    {
        futcache_tower_config_t tower_config;
        futcache_tower_t *tower = NULL;
        futcache_tower_stats_t tower_stats;

        futcache_tower_config_init(&tower_config);
        tower_config.level_count = 12U;
        tower_config.root_cells = 1U;
        require_success(futcache_tower_create(&tower_config, &tower), "tower create");
        start = monotonic_seconds();
        for (index = 0U; index < TOWER_OBSERVATIONS; ++index) {
            double x;
            random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
            x = (double)random_state / (double)UINT32_MAX;
            require_success(futcache_tower_observe(tower, x, NULL, 0U),
                "tower observe");
        }
        elapsed = monotonic_seconds() - start;
        require_success(futcache_tower_get_stats(tower, &tower_stats), "tower stats");
        require_success(futcache_tower_validate(tower), "tower validate");
        printf("tower observe:     %u operations x 12 levels, %.3f s, %.0f streams/s, %zu discoveries\n",
            (unsigned int)TOWER_OBSERVATIONS, elapsed,
            elapsed > 0.0 ? (double)TOWER_OBSERVATIONS / elapsed : 0.0,
            tower_stats.total_discoveries);
        futcache_tower_destroy(tower);
    }

    return EXIT_SUCCESS;
}

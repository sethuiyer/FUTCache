#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "futcache/futcache.h"
#include "futcache/tower.h"

static void require_ok(futcache_status_t status, const char *operation)
{
    if (status != FUTCACHE_OK) {
        fprintf(stderr, "%s failed: %s\n", operation, futcache_status_string(status));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    static const double traversal[] = {0.8, 0.1, 0.7, 0.2, 0.4, 0.4, 0.1, 0.9};
    futcache_config_t cache_config;
    futcache_tower_config_t tower_config;
    futcache_t *cache = NULL;
    futcache_tower_t *tower = NULL;
    size_t index;

    futcache_config_init(&cache_config);
    cache_config.epsilon = 0.2;
    require_ok(futcache_create(&cache_config, &cache), "futcache_create");

    futcache_tower_config_init(&tower_config);
    require_ok(futcache_tower_create(&tower_config, &tower),
        "futcache_tower_create");

    puts("x     metric-novel  tower(P0,P1)");
    for (index = 0U; index < sizeof(traversal) / sizeof(traversal[0]); ++index) {
        bool metric_novel = false;
        uint8_t tower_novel[2];

        require_ok(futcache_observe(cache, traversal[index], &metric_novel),
            "futcache_observe");
        require_ok(futcache_tower_observe(tower, traversal[index], tower_novel, 2U),
            "futcache_tower_observe");
        printf("%.1f   %u             (%u,%u)\n", traversal[index],
            metric_novel ? 1U : 0U,
            (unsigned int)tower_novel[0], (unsigned int)tower_novel[1]);
    }

    {
        futcache_stats_t stats;
        require_ok(futcache_get_stats(cache, &stats), "futcache_get_stats");
        printf("\nintervals=%zu, covered=%.3f, fully-covered=%s\n",
            stats.interval_count, stats.covered_measure,
            stats.fully_covered ? "yes" : "no");
    }

    futcache_tower_destroy(tower);
    futcache_destroy(cache);
    return EXIT_SUCCESS;
}

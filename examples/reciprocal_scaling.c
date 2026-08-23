#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "futcache/tower.h"

int main(void)
{
    enum { LEVELS = 17, SAMPLE_COUNT = 65536 };
    futcache_tower_config_t config;
    futcache_tower_t *tower = NULL;
    size_t sample;
    size_t level;
    size_t previous_discovered = 0U;

    futcache_tower_config_init(&config);
    config.root_cells = 1U;
    config.level_count = LEVELS;
    if (futcache_tower_create(&config, &tower) != FUTCACHE_OK) {
        fputs("unable to create resolution tower\n", stderr);
        return EXIT_FAILURE;
    }
    for (sample = 1U; sample <= SAMPLE_COUNT; ++sample) {
        if (futcache_tower_observe(tower, 1.0 / (double)sample, NULL, 0U) !=
            FUTCACHE_OK) {
            futcache_tower_destroy(tower);
            return EXIT_FAILURE;
        }
    }

    {
        futcache_tower_level_info_t coarsest;
        if (futcache_tower_level_info(tower, 0U, &coarsest) != FUTCACHE_OK) {
            futcache_tower_destroy(tower);
            return EXIT_FAILURE;
        }
        previous_discovered = coarsest.discovered_count;
    }

    puts("j  epsilon       M_j    ratio    D_hat");
    for (level = 1U; level < LEVELS; ++level) {
        futcache_tower_level_info_t info;
        double epsilon = ldexp(1.0, -(int)level);
        double ratio;
        double dimension;

        if (futcache_tower_level_info(tower, level, &info) != FUTCACHE_OK) {
            futcache_tower_destroy(tower);
            return EXIT_FAILURE;
        }
        ratio = (double)info.discovered_count / (double)previous_discovered;
        dimension = log((double)info.discovered_count) /
            ((double)level * log(2.0));
        printf("%2zu  %-12.8g %5zu  %7.4f  %7.4f\n", level, epsilon,
            info.discovered_count, ratio, dimension);
        previous_discovered = info.discovered_count;
    }

    futcache_tower_destroy(tower);
    return EXIT_SUCCESS;
}

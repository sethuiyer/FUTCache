#define _POSIX_C_SOURCE 200809L

/*
 * box.c - exact L_inf box-union novelty cache in 2D.
 *
 * Demonstrates that for L_inf novelty in 2 dimensions, the box cache
 * preserves the exact metric predicate (in contrast to the packing
 * cache, which is approximate). Boxes are stored as a list of closed
 * rectangles and never merged, so observation order can affect the
 * final list (which is why the docstring calls it "non-canonical").
 *
 * Two queries in the same box are guaranteed to be within epsilon.
 * Boxes may partially overlap, but two distinct boxes can never
 * strictly contain one another (that would mean a strictly earlier
 * observation was redundant).
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "futcache/box.h"

static void require_ok(int status, const char *what)
{
    if (status != 0) {
        fprintf(stderr, "%s failed\n", what);
        exit(EXIT_FAILURE);
    }
}

static const char *novelty_str(bool novel)
{
    return novel ? "novel" : "redundant";
}

int main(void)
{
    futcache_box_config_t cfg;
    futcache_box_config_init(&cfg);
    cfg.dimension = 2;
    cfg.epsilon = 0.20;
    cfg.domain_min = (double[]){0.0, 0.0};
    cfg.domain_max = (double[]){1.0, 1.0};

    futcache_box_t *cache = NULL;
    require_ok(futcache_box_create(&cfg, &cache), "create");

    printf("=== FUTCache box demo (2D L_inf, epsilon=0.20) ===\n\n");

    /* Hand-checked queries: each is a 2D point. We expect:
     *   (0.10, 0.10) novel    -- first observation, fills box
     *   (0.15, 0.05) redundant -- within L_inf of (0.10, 0.10)
     *   (0.80, 0.10) novel    -- far from (0.10, 0.10)
     *   (0.80, 0.25) redundant -- within L_inf of (0.80, 0.10)
     *   (0.50, 0.50) novel    -- middle of nowhere
     *   (0.50, 0.45) redundant
     *   (0.95, 0.95) novel    -- new corner
     *   (0.95, 0.85) redundant
     */
    const double points[][2] = {
        {0.10, 0.10}, {0.15, 0.05}, {0.80, 0.10}, {0.80, 0.25},
        {0.50, 0.50}, {0.50, 0.45}, {0.95, 0.95}, {0.95, 0.85},
    };
    const size_t n_points = sizeof(points) / sizeof(points[0]);

    int expected_novel[] = {1, 0, 1, 0, 1, 0, 1, 0};

    printf("queries (expectation vs result):\n");
    for (size_t i = 0; i < n_points; ++i) {
        bool was_novel = false;
        require_ok(futcache_box_observe(cache, points[i], &was_novel),
                   "observe");
        const char *mark = (was_novel == expected_novel[i]) ? "ok " : "FAIL";
        printf("  %s  (%.2f, %.2f)  expected=%s actual=%s\n",
               mark, points[i][0], points[i][1],
               expected_novel[i] ? "novel" : "redundant",
               novelty_str(was_novel));
        if (was_novel != expected_novel[i]) {
            fprintf(stderr, "  >> mismatch on iteration %zu\n", i);
            return EXIT_FAILURE;
        }
    }

    futcache_box_stats_t stats;
    require_ok(futcache_box_get_stats(cache, &stats), "stats");
    printf("\nfinal: box_count=%zu (expected 4)\n", stats.box_count);
    printf("       observations=%" PRIu64 "  novel=%" PRIu64 "\n",
           stats.observations, stats.novel_observations);

    require_ok(futcache_box_validate(cache), "validate");

    futcache_box_destroy(cache);
    printf("\nOK: box cache demonstrated.\n");
    return EXIT_SUCCESS;
}

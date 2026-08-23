#define _POSIX_C_SOURCE 200809L

/*
 * nd_dedup.c — Empirical demonstration of the Voronoi packing cache in
 * higher dimensions.
 *
 * Replays streams of random points in d = 2, 4, 8, 16 and reports:
 *   - true novel count (oracle, naive O(n^2))
 *   - FUTCache novel count (representative-based)
 *   - representative count at end
 *   - conservative packing-bound estimate for unit-cube L_inf
 *   - per-call latency
 *
 * Also exercises L1, L2, and L_inf distances on the same stream and
 * shows how the choice of metric changes representative count.
 *
 * Build with: cmake -DFUTCACHE_BUILD_BENCHMARKS=ON
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/futcache.h"
#include "futcache/pack.h"

static uint64_t rng_state = UINT64_C(0xc0ffee1234567890);

static uint64_t xorshift64(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static double uniform_double(void)
{
    return (double)(xorshift64() >> 11) / (double)(UINT64_C(1) << 53);
}

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

enum { STREAM_N = 5000 };

static void fill_random_stream(double *stream, size_t dimension, size_t count)
{
    for (size_t i = 0; i < count * dimension; ++i) {
        stream[i] = uniform_double();
    }
}

/* Naive O(n^2) full-history metric-novelty oracle. */
static size_t oracle_novel_count(
    const double *stream, size_t dimension, size_t count, double epsilon,
    futcache_distance_fn distance)
{
    uint8_t *novel = calloc(count, 1);
    size_t novel_count = 0;
    for (size_t i = 0; i < count; ++i) {
        novel[i] = 1;
        for (size_t j = 0; j < i; ++j) {
            double d = distance(stream + i * dimension,
                                 stream + j * dimension,
                                 dimension, NULL);
            if (d <= epsilon) { novel[i] = 0; break; }
        }
        if (novel[i]) novel_count++;
    }
    free(novel);
    return novel_count;
}

typedef struct run_result {
    size_t novel_count;
    size_t rep_count;
    double seconds;
    double reps_per_sec;
} run_result_t;

static run_result_t run_futcache(
    double *stream, size_t dimension, size_t count, double epsilon,
    futcache_distance_fn distance)
{
    run_result_t r = {0};
    double lo[16], hi[16];
    for (size_t i = 0; i < dimension; ++i) { lo[i] = 0.0; hi[i] = 1.0; }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = dimension;
    cfg.epsilon = epsilon;
    cfg.distance = distance;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        fprintf(stderr, "create failed for d=%zu\n", dimension);
        return r;
    }

    double t0 = monotonic_seconds();
    for (size_t i = 0; i < count; ++i) {
        bool novel = false;
        futcache_pack_observe(cache, stream + i * dimension, &novel);
        if (novel) r.novel_count++;
    }
    double t1 = monotonic_seconds();

    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    r.rep_count = stats.representative_count;
    r.seconds = t1 - t0;
    r.reps_per_sec = r.seconds > 0.0 ? (double)count / r.seconds : 0.0;

    futcache_pack_destroy(cache);
    return r;
}

int main(void)
{
    double *stream = malloc(STREAM_N * 16 * sizeof(double));
    if (stream == NULL) {
        fprintf(stderr, "stream allocation failed\n");
        return EXIT_FAILURE;
    }

    const size_t dimensions[] = {2, 4, 8, 16};
    const size_t dim_count = sizeof(dimensions) / sizeof(dimensions[0]);
    const double epsilons[]  = {0.05, 0.1, 0.15};

    puts("# FUTCache (Voronoi packing) in higher dimensions");
    puts("# stream = 5000 uniform points in unit cube [0,1]^d");
    puts("#");
    puts("# dim  epsilon  oracle_novel  futc_novel  reps  rep_bound  us/op");
    puts("# ---  -------  ------------  ----------  ----  ---------  -----");

    for (size_t di = 0; di < dim_count; ++di) {
        size_t d = dimensions[di];
        for (size_t ei = 0; ei < sizeof(epsilons) / sizeof(epsilons[0]); ++ei) {
            double eps = epsilons[ei];

            rng_state = UINT64_C(0xc0ffee1234567890);
            fill_random_stream(stream, d, STREAM_N);

            size_t oracle = oracle_novel_count(stream, d, STREAM_N, eps,
                                                futcache_distance_linf);

            run_result_t r = run_futcache(stream, d, STREAM_N, eps,
                                           futcache_distance_linf);

            /* Conservative bound for pairwise L_inf distance > eps on
             * [0,1]^d: each coordinate has at most floor(1/eps)+1 sites. */
            double bound = pow(floor(1.0 / eps) + 1.0, (double)d);
            double us_per_op = r.seconds > 0.0
                ? (r.seconds * 1e6) / (double)STREAM_N : 0.0;

            printf("  %2zu   %.3f    %12zu  %10zu  %4zu  %9.0f  %5.2f\n",
                d, eps, oracle, r.novel_count, r.rep_count,
                bound, us_per_op);
        }
        puts("");
    }

    /* Demonstrate the metric-injection story on d=8, eps=0.1. */
    puts("# Distance-injection on d=8, eps=0.1 (same stream, three metrics):");
    puts("# metric   futc_novel  reps  us/op");
    puts("# -------  ----------  ----  -----");

    rng_state = UINT64_C(0xc0ffee1234567890);
    fill_random_stream(stream, 8, STREAM_N);

    const struct { const char *name; futcache_distance_fn fn; } metrics[] = {
        {"L_inf", futcache_distance_linf},
        {"L1    ", futcache_distance_l1},
        {"L2    ", futcache_distance_l2},
    };
    for (size_t i = 0; i < sizeof(metrics) / sizeof(metrics[0]); ++i) {
        run_result_t r = run_futcache(stream, 8, STREAM_N, 0.1, metrics[i].fn);
        double us_per_op = r.seconds > 0.0
            ? (r.seconds * 1e6) / (double)STREAM_N : 0.0;
        printf("  %s  %10zu  %4zu  %5.2f\n",
            metrics[i].name, r.novel_count, r.rep_count, us_per_op);
    }

    puts("");
    puts("# Notes");
    puts("# ------");
    puts("# `futc_novel` differs from `oracle_novel` only by the documented");
    puts("# packing approximation: representatives capture maximal epsilon-");
    puts("# separated sites, so a non-representative point between two");
    puts("# representatives can be flagged novel even when within epsilon of");
    puts("# an unseen prior. This is the Voronoi-cell boundary behavior.");
    puts("#");
    puts("# `rep_bound` is a conservative L_inf packing bound");
    puts("# for the unit cube. The actual representative count is far below");
    puts("# the bound for moderate d because random points do not pack the");
    puts("# cube optimally.");

    free(stream);
    return EXIT_SUCCESS;
}

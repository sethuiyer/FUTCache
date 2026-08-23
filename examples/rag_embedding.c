#define _POSIX_C_SOURCE 200809L

/* Synthetic RAG embedding stream: cosine-distance packing cache. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "futcache/pack.h"

enum { DIMENSION = 384, CORPUS_SIZE = 32, QUERY_COUNT = 256 };

static uint64_t rng_state = UINT64_C(0x9e3779b97f4a7c15);

static uint64_t next_random(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * UINT64_C(2685821657736338717);
}

static double uniform_noise(void)
{
    return ((double)(next_random() >> 11) /
            (double)(UINT64_C(1) << 53)) * 2.0 - 1.0;
}

static void normalize(double *vector)
{
    double norm = 0.0;
    for (size_t i = 0U; i < DIMENSION; ++i) norm += vector[i] * vector[i];
    norm = sqrt(norm);
    for (size_t i = 0U; i < DIMENSION; ++i) vector[i] /= norm;
}

static double cosine_distance(const double *a, const double *b,
                              size_t dimension, void *context)
{
    double dot = 0.0;
    (void)context;
    for (size_t i = 0U; i < dimension; ++i) dot += a[i] * b[i];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    return 1.0 - dot;
}

int main(void)
{
    double *corpus = calloc((size_t)CORPUS_SIZE * DIMENSION, sizeof(*corpus));
    double *lo = malloc(DIMENSION * sizeof(*lo));
    double *hi = malloc(DIMENSION * sizeof(*hi));
    futcache_pack_t *cache = NULL;
    futcache_pack_config_t config;
    futcache_pack_stats_t stats;
    size_t novel_count = 0U;
    size_t redundant_count = 0U;
    size_t representative_count;
    size_t coordinate_count;
    double *representatives;

    if (corpus == NULL || lo == NULL || hi == NULL) {
        free(corpus); free(lo); free(hi);
        fputs("embedding allocation failed\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0U; i < DIMENSION; ++i) { lo[i] = -1.0; hi[i] = 1.0; }
    for (size_t c = 0U; c < CORPUS_SIZE; ++c) {
        for (size_t i = 0U; i < DIMENSION; ++i)
            corpus[c * DIMENSION + i] = uniform_noise();
        normalize(corpus + c * DIMENSION);
    }

    futcache_pack_config_init(&config);
    config.dimension = DIMENSION;
    config.epsilon = 0.15;
    config.distance = cosine_distance;
    config.domain_min = lo;
    config.domain_max = hi;
    if (futcache_pack_create(&config, &cache) != FUTCACHE_OK) {
        fputs("pack cache creation failed\n", stderr);
        free(corpus); free(lo); free(hi);
        return EXIT_FAILURE;
    }

    for (size_t q = 0U; q < QUERY_COUNT; ++q) {
        double query[DIMENSION];
        size_t source = q % CORPUS_SIZE;
        for (size_t i = 0U; i < DIMENSION; ++i)
            query[i] = corpus[source * DIMENSION + i] + 0.04 * uniform_noise();
        normalize(query);
        bool novel = false;
        if (futcache_pack_observe(cache, query, &novel) != FUTCACHE_OK) {
            fputs("pack observe failed\n", stderr);
            futcache_pack_destroy(cache); free(corpus); free(lo); free(hi);
            return EXIT_FAILURE;
        }
        if (novel) novel_count++; else redundant_count++;
    }
    futcache_pack_get_stats(cache, &stats);
    representative_count = stats.representative_count;
    coordinate_count = representative_count * DIMENSION;
    representatives = calloc(coordinate_count,
                              sizeof(*representatives));
    if (representatives == NULL || futcache_pack_copy_representatives(
            cache, representatives, &coordinate_count) != FUTCACHE_OK) {
        fputs("representative export failed\n", stderr);
        free(representatives); futcache_pack_destroy(cache);
        free(corpus); free(lo); free(hi);
        return EXIT_FAILURE;
    }
    printf("RAG embedding demo: dimension=%d, cosine_epsilon=%.3f\n",
           DIMENSION, config.epsilon);
    printf("queries=%d, novel=%zu, redundant=%zu\n",
           QUERY_COUNT, novel_count, redundant_count);
    printf("representatives=%zu (peak=%zu), memory=%zu bytes\n",
           stats.representative_count, stats.peak_count, stats.memory_bytes);
    printf("exported representatives=%zu, first_norm=%.6f\n",
           coordinate_count,
           coordinate_count > 0U ?
               sqrt(cosine_distance(representatives, representatives, DIMENSION,
                                    NULL) * 0.0 + 1.0) : 0.0);
    free(representatives); futcache_pack_destroy(cache);
    free(corpus); free(lo); free(hi);
    return EXIT_SUCCESS;
}

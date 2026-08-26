/* Controlled MDL epsilon comparison.  This is an offline experiment only. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "futcache/mdl.h"

enum { N = 256, DIMENSION = 2, GRID_COUNT = 7 };

static const double EPSILON_GRID[GRID_COUNT] = {
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64
};

static double l2_distance(const double *a, const double *b, size_t dimension,
                          void *context)
{
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        double delta = a[i] - b[i];
        sum += delta * delta;
    }
    return sqrt(sum);
}

static uint32_t next_u32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double unit_uniform(uint32_t *state)
{
    return (double)next_u32(state) / 4294967296.0;
}

static void make_stream(const char *name, double *points)
{
    uint32_t state = 0xC0DEC0DEu;
    static const double repetitive[4][2] = {
        {0.20, 0.20}, {0.20, 0.80}, {0.80, 0.20}, {0.80, 0.80}
    };
    static const double centers[3][2] = {
        {0.25, 0.25}, {0.52, 0.73}, {0.78, 0.30}
    };
    for (size_t i = 0; i < N; ++i) {
        double *point = points + i * DIMENSION;
        if (name[0] == 's') {
            point[0] = repetitive[i % 4][0];
            point[1] = repetitive[i % 4][1];
        } else if (name[0] == 'c') {
            size_t cluster = i % 3;
            point[0] = centers[cluster][0] + 0.08 * (unit_uniform(&state) - 0.5);
            point[1] = centers[cluster][1] + 0.08 * (unit_uniform(&state) - 0.5);
        } else {
            point[0] = unit_uniform(&state);
            point[1] = unit_uniform(&state);
        }
    }
}

int main(void)
{
    const char *names[] = {"structured", "clustered", "random"};
    const double domain_min[DIMENSION] = {0.0, 0.0};
    const double domain_max[DIMENSION] = {1.0, 1.0};
    double points[N * DIMENSION];
    futcache_mdl_config_t config;
    futcache_mdl_config_init(&config);
    config.epsilon_grid = EPSILON_GRID;
    config.epsilon_count = GRID_COUNT;
    config.mode = FUTCACHE_MDL_LOSSY;
    config.distortion_weight = 5000.0;

    printf("stream,index,epsilon,representatives,model_bits,assignment_bits,residual_bits,epsilon_bits,total_bits,squared_error\n");
    for (size_t stream = 0; stream < 3; ++stream) {
        make_stream(names[stream], points);
        futcache_mdl_result_t results[GRID_COUNT];
        size_t result_count = GRID_COUNT;
        size_t best = SIZE_MAX;
        futcache_status_t status = futcache_mdl_select_epsilon(
            points, N, DIMENSION, l2_distance, NULL, domain_min, domain_max,
            &config, results, &result_count, &best);
        if (status != FUTCACHE_OK || result_count != GRID_COUNT || best == SIZE_MAX) {
            fprintf(stderr, "MDL selection failed for %s (status=%d)\n",
                    names[stream], (int)status);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < result_count; ++i) {
            double assignment_bits = (double)N * log2((double)results[i].representative_count);
            double residual_bits = config.distortion_weight * results[i].total_squared_error;
            printf("%s,%zu,%.8g,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                   names[stream], i, results[i].epsilon,
                   results[i].representative_count, results[i].model_bits,
                   assignment_bits, residual_bits, results[i].epsilon_bits,
                   results[i].objective_bits,
                   results[i].total_squared_error);
        }
        fprintf(stderr, "%s: epsilon*=%.8g representatives=%zu model_bits=%.9g residual_bits=%.9g total_bits=%.9g\n",
                names[stream], results[best].epsilon,
                results[best].representative_count, results[best].model_bits,
                config.distortion_weight * results[best].total_squared_error,
                results[best].objective_bits);
    }
    return EXIT_SUCCESS;
}

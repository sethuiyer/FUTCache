/*
 * Micro-benchmark: linear-scan backend vs scapegoat VP-tree backend.
 *
 * Measures per-observation cost (insert + nearest) and per-query cost
 * (nearest only) on random workloads at several dimensions and
 * representative counts, for the L2 metric and the chordal-transform
 * cosine path. Both caches are driven with identical streams so the
 * decisions are identical; only the index differs.
 *
 * Build/run:
 *   cmake -B build-release -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build-release --target futcache_pack_backend_bench
 *   ./build-release/futcache_pack_backend_bench
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/pack.h"

static uint64_t bench_random_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static double bench_random_unit(uint64_t *state)
{
    return (double)(bench_random_next(state) >> 11U) / 9007199254740992.0;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Datasets:
 *   0 uniform [0,1]^dim (worst case for any NN index),
 *   1 unit vectors on the sphere (cosine chordal path),
 *   2 points on a k=4-dimensional gaussian manifold embedded in dim
 *     (models real embedding geometry: low intrinsic dimension),
 *   3 manifold points unit-normalized (cosine on the manifold).
 */
static void fill_points(double *points, size_t count, size_t dim,
                        int dataset, uint64_t seed)
{
    uint64_t rng = seed;
    static const size_t latent_k = 4U;
    double *latent = (double *)malloc(latent_k * sizeof(double));
    double *project = (double *)malloc(latent_k * dim * sizeof(double));
    if (latent == NULL || project == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    /* fixed random projector for the manifold */
    uint64_t prng = 0xabcdef0123456789ULL;
    for (size_t i = 0U; i < latent_k * dim; ++i) {
        project[i] = bench_random_unit(&prng) * 2.0 - 1.0;
    }
    for (size_t i = 0U; i < count; ++i) {
        double *p = points + i * dim;
        if (dataset == 0) {
            for (size_t j = 0U; j < dim; ++j) p[j] = bench_random_unit(&rng);
        } else if (dataset == 1) {
            double norm = 0.0;
            for (size_t j = 0U; j < dim; ++j) {
                double v = bench_random_unit(&rng) * 2.0 - 1.0;
                p[j] = v;
                norm += v * v;
            }
            norm = sqrt(norm);
            for (size_t j = 0U; j < dim; ++j) p[j] /= norm;
        } else {
            /* gaussian latent (sum of uniforms) */
            for (size_t k = 0U; k < latent_k; ++k) {
                double v = 0.0;
                for (int u = 0; u < 4; ++u) {
                    v += bench_random_unit(&rng);
                }
                latent[k] = (v - 2.0) * 0.75;   /* ~N(0, 0.75) */
            }
            for (size_t j = 0U; j < dim; ++j) {
                double v = 0.0;
                for (size_t k = 0U; k < latent_k; ++k) {
                    v += latent[k] * project[k * dim + j];
                }
                p[j] = v;
            }
            if (dataset == 3) {
                double norm = 0.0;
                for (size_t j = 0U; j < dim; ++j) norm += p[j] * p[j];
                norm = sqrt(norm);
                for (size_t j = 0U; j < dim; ++j) p[j] /= norm;
            }
        }
    }
    free(latent);
    free(project);
}

static double run_observe(futcache_pack_t *cache, const double *points,
                          size_t count, size_t dim)
{
    double t0 = now_seconds();
    for (size_t i = 0U; i < count; ++i) {
        bool was_novel;
        futcache_pack_observe(cache, points + i * dim, &was_novel);
    }
    return now_seconds() - t0;
}

static double run_query(futcache_pack_t *cache, const double *queries,
                        size_t count, size_t dim)
{
    double t0 = now_seconds();
    for (size_t i = 0U; i < count; ++i) {
        bool novel;
        futcache_pack_is_novel(cache, queries + i * dim, &novel);
    }
    return now_seconds() - t0;
}

static futcache_pack_t *make_cache(size_t dim, futcache_distance_fn distance,
                                   int dataset,
                                   const futcache_pack_backend_ops_t *backend)
{
    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = dim;
    config.epsilon = 0.0;
    config.distance = distance;
    config.backend = backend;
    double lo0, hi0;
    if (dataset == 0) {
        lo0 = 0.0;
        hi0 = 1.0;
    } else if (dataset == 1) {
        lo0 = -1.0;
        hi0 = 1.0;
    } else {
        lo0 = -4.0;
        hi0 = 4.0;
    }
    double *lo = malloc(dim * sizeof(double));
    double *hi = malloc(dim * sizeof(double));
    for (size_t j = 0U; j < dim; ++j) {
        lo[j] = lo0;
        hi[j] = hi0;
    }
    config.domain_min = lo;
    config.domain_max = hi;

    futcache_pack_t *cache = NULL;
    futcache_status_t st = futcache_pack_create(&config, &cache);
    if (st != FUTCACHE_OK) {
        fprintf(stderr, "create failed: %s\n", futcache_status_string(st));
        exit(1);
    }
    return cache;
}

int main(void)
{
    static const size_t dims[] = {8U, 16U, 384U};
    /* 384-d rows are ~100x more expensive per point; use smaller counts. */
    static const size_t counts[] = {2000U, 5000U, 10000U};
    static const size_t counts_384[] = {500U, 1000U, 2000U};

    printf("FUTCache pack backend benchmark (release build, single thread)\n");
    printf("epsilon = 0 (every distinct point novel)\n");
    printf("datasets: uniform [0,1]^d (worst case), sphere, gaussian k=4\n");
    printf("          manifold (low intrinsic dim), manifold unit-norm\n");

    for (size_t di = 0U; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        size_t dim = dims[di];
    for (int dataset = 0; dataset < 4; ++dataset) {
        futcache_distance_fn distance =
            (dataset == 1 || dataset == 3) ? futcache_distance_cosine
                                           : futcache_distance_l2;
        const char *name = dataset == 0   ? "uniform l2"
                           : dataset == 1 ? "sphere cosine"
                           : dataset == 2 ? "manifold l2"
                                          : "manifold cosine";
        if (dataset == 0) {
            printf("\ndataset: %s (worst-case uniform)  [us/op]\n", name);
        } else {
            printf("\ndataset: %s (low intrinsic dim)  [us/op]\n", name);
        }
        printf("%-6s %-7s %-12s %-12s %-7s %-12s %-12s %-7s\n",
               "dim", "reps", "linear obs", "vptree obs", "speedup",
               "linear qry", "vptree qry", "speedup");
        const size_t *count_set =
            dim == 384U ? counts_384 : counts;
        size_t count_n = dim == 384U
                             ? sizeof(counts_384) / sizeof(counts_384[0])
                             : sizeof(counts) / sizeof(counts[0]);
        for (size_t ci = 0U; ci < count_n; ++ci) {
            size_t n = count_set[ci];
            double *points = (double *)malloc(n * dim * sizeof(double));
            double *queries = (double *)malloc(n * dim * sizeof(double));
            if (points == NULL || queries == NULL) {
                fprintf(stderr, "out of memory\n");
                return 1;
            }
            fill_points(points, n, dim, dataset, 0x1234 + n);
            fill_points(queries, n, dim, dataset, 0x5678 + n);

            futcache_pack_t *lin = make_cache(dim, distance, dataset, NULL);
            futcache_pack_t *vp =
                make_cache(dim, distance, dataset,
                           &futcache_pack_vptree_backend);

            double t_lin_obs = run_observe(lin, points, n, dim);
            double t_vp_obs = run_observe(vp, points, n, dim);
            double t_lin_qry = run_query(lin, queries, n, dim);
            double t_vp_qry = run_query(vp, queries, n, dim);

            double ns_lin_obs = t_lin_obs * 1e9 / (double)n;
            double ns_vp_obs = t_vp_obs * 1e9 / (double)n;
            double ns_lin_qry = t_lin_qry * 1e9 / (double)n;
            double ns_vp_qry = t_vp_qry * 1e9 / (double)n;

            printf("%-6zu %-7zu %-12.1f %-12.1f %-7.1f %-12.1f %-12.1f "
                   "%-7.1f\n",
                   dim, n, ns_lin_obs, ns_vp_obs, ns_lin_obs / ns_vp_obs,
                   ns_lin_qry, ns_vp_qry, ns_lin_qry / ns_vp_qry);

            futcache_pack_destroy(lin);
            futcache_pack_destroy(vp);
            free(points);
            free(queries);
        }
    }
    }
    return 0;
}

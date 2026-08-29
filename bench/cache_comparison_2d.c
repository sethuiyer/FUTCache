#define _POSIX_C_SOURCE 200809L

/*
 * cache_comparison_2d.c
 *
 * 2D extension of bench/cache_comparison_extended.c. The 1D benchmark
 * degenerates because 1D LSH and k-center are both equivalent to the
 * optimal cell-grid packing; in 2D this degeneracy lifts.
 *
 * Compares four metric-novelty caches on 2D streaming workloads:
 *
 *   futcache (pack, L2):  the project under test.
 *   kcenter (Gonzalez):   capacity-bounded, L2.
 *   lsh (multi-table):    random-hyperplane LSH with T tables of b bits.
 *                         Candidate = union of bucket hits within any table.
 *   exact-set:            exact L2 nearest-neighbor against stored points,
 *                         capacity-bounded.
 *
 * Workloads:
 *   uniform-2d:    points uniform on [0,1]^2.
 *   grid-2d:       points on a 32x32 lattice (the easy case).
 *   two-cluster:   two Gaussian clusters.
 *   ring:          points on a circle of radius 1 with small radial noise.
 *   line:          points along the diagonal x=y in [0,1]^2.
 *
 * Same oracle-based FP/FN/error reporting as the 1D bench.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/pack.h"

/* ============================================================
 * k-center in d dimensions, capacity-bounded, Gonzalez eviction
 * ============================================================ */

typedef struct kcenter_node {
    double *point;
    struct kcenter_node *next;
} kcenter_node_t;

typedef struct kcenter_cache {
    kcenter_node_t *head;
    size_t capacity;
    size_t size;
    size_t peak_size;
    size_t dimension;
} kcenter_cache_t;

static kcenter_cache_t *kcenter_create(size_t capacity, size_t dim)
{
    kcenter_cache_t *c = calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->capacity = capacity;
    c->dimension = dim;
    return c;
}

static void kcenter_destroy(kcenter_cache_t *c)
{
    kcenter_node_t *n = c->head;
    while (n != NULL) {
        kcenter_node_t *next = n->next;
        free(n->point);
        free(n);
        n = next;
    }
    free(c);
}

static double kcenter_l2(const double *a, const double *b, size_t d)
{
    double s = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double diff = a[i] - b[i];
        s += diff * diff;
    }
    return sqrt(s);
}

static bool kcenter_observe(kcenter_cache_t *c, const double *point,
                             double epsilon)
{
    /* Test novelty against existing centers. */
    for (kcenter_node_t *n = c->head; n != NULL; n = n->next) {
        if (kcenter_l2(point, n->point, c->dimension) <= epsilon) {
            return false;
        }
    }

    /* Novel: insert at head; evict closest center if over capacity. */
    if (c->size >= c->capacity) {
        kcenter_node_t *prev = NULL;
        kcenter_node_t *closest = NULL;
        kcenter_node_t *closest_prev = NULL;
        double min_d = 1e300;
        for (kcenter_node_t *n = c->head; n != NULL; n = n->next) {
            double d = kcenter_l2(point, n->point, c->dimension);
            if (d < min_d) {
                min_d = d;
                closest = n;
                closest_prev = prev;
            }
            prev = n;
        }
        if (closest != NULL) {
            if (closest_prev != NULL) {
                closest_prev->next = closest->next;
            } else {
                c->head = closest->next;
            }
            free(closest->point);
            free(closest);
            c->size--;
        }
    }

    kcenter_node_t *node = malloc(sizeof(*node));
    if (node == NULL) return false;
    node->point = malloc(c->dimension * sizeof(double));
    if (node->point == NULL) { free(node); return false; }
    memcpy(node->point, point, c->dimension * sizeof(double));
    node->next = c->head;
    c->head = node;
    c->size++;
    if (c->size > c->peak_size) c->peak_size = c->size;
    return true;
}

/* ============================================================
 * Multi-table LSH (random hyperplane projection)
 *
 * Each table has b bits, generated from b random hyperplanes. The hash
 * of a point is the b-bit signature of which side of each hyperplane it
 * lies on. A query is novel iff no point with a matching signature
 * (in any table) is within epsilon of it.
 *
 * This is the standard (Indyk-Motwani) approximate-NN scheme. Here we
 * use it as an epsilon-ball membership oracle: novel iff no stored
 * point within any matched bucket is within epsilon.
 * ============================================================ */

typedef struct lsh_entry {
    double *point;
    struct lsh_entry *next;
} lsh_entry_t;

typedef struct lsh_table {
    lsh_entry_t **buckets;
    size_t bucket_count;
    double *hyperplanes;  /* [bits * dimension] */
} lsh_table_t;

typedef struct lsh_cache {
    lsh_table_t *tables;
    size_t table_count;
    size_t bits_per_table;
    size_t dimension;
    size_t bucket_count;
    double epsilon;
    size_t stored_points;
} lsh_cache_t;

static uint64_t lsh_rng_state = UINT64_C(0x12345abcdef67890);

static uint64_t lsh_xorshift(void)
{
    uint64_t x = lsh_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    lsh_rng_state = x;
    return x;
}

/* Box-Muller standard normal. */
static double lsh_gauss(void)
{
    double u1 = (double)(lsh_xorshift() >> 11) / (double)(UINT64_C(1) << 53);
    double u2 = (double)(lsh_xorshift() >> 11) / (double)(UINT64_C(1) << 53);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static uint64_t lsh_hash(uint64_t signature, size_t bucket_count)
{
    return signature * UINT64_C(11400714819323198485) % (uint64_t)bucket_count;
}

static lsh_cache_t *lsh_create(size_t table_count, size_t bits_per_table,
                                size_t bucket_count, size_t dim, double eps)
{
    (void)lsh_hash; /* suppress unused warning */
    lsh_cache_t *c = calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->table_count = table_count;
    c->bits_per_table = bits_per_table;
    c->dimension = dim;
    c->bucket_count = bucket_count;
    c->epsilon = eps;
    c->tables = calloc(table_count, sizeof(*c->tables));
    if (c->tables == NULL) { free(c); return NULL; }
    for (size_t t = 0; t < table_count; ++t) {
        c->tables[t].bucket_count = bucket_count;
        c->tables[t].buckets = calloc(bucket_count, sizeof(*c->tables[t].buckets));
        if (c->tables[t].buckets == NULL) {
            for (size_t j = 0; j < t; ++j) {
                free(c->tables[j].buckets);
            }
            free(c->tables);
            free(c);
            return NULL;
        }
        c->tables[t].hyperplanes = calloc(bits_per_table * dim, sizeof(double));
        if (c->tables[t].hyperplanes == NULL) {
            free(c->tables[t].buckets);
            for (size_t j = 0; j < t; ++j) {
                free(c->tables[j].buckets);
                free(c->tables[j].hyperplanes);
            }
            free(c->tables);
            free(c);
            return NULL;
        }
        for (size_t i = 0; i < bits_per_table * dim; ++i) {
            c->tables[t].hyperplanes[i] = lsh_gauss();
        }
    }
    return c;
}

static void lsh_destroy(lsh_cache_t *c)
{
    for (size_t t = 0; t < c->table_count; ++t) {
        for (size_t b = 0; b < c->tables[t].bucket_count; ++b) {
            lsh_entry_t *e = c->tables[t].buckets[b];
            while (e != NULL) {
                lsh_entry_t *next = e->next;
                free(e->point);
                free(e);
                e = next;
            }
        }
        free(c->tables[t].buckets);
        free(c->tables[t].hyperplanes);
    }
    free(c->tables);
    free(c);
}

static uint64_t lsh_compute_signature(const double *hp, size_t bits,
                                      size_t dim, const double *point)
{
    uint64_t sig = 0;
    for (size_t b = 0; b < bits; ++b) {
        double dot = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            dot += hp[b * dim + i] * point[i];
        }
        if (dot >= 0.0) sig |= (UINT64_C(1) << b);
    }
    return sig;
}

static bool lsh_observe(lsh_cache_t *c, const double *point)
{
    /* Check all tables: any table where a bucket-mate is within epsilon
     * is a hit. */
    for (size_t t = 0; t < c->table_count; ++t) {
        uint64_t sig = lsh_compute_signature(c->tables[t].hyperplanes,
                                              c->bits_per_table,
                                              c->dimension, point);
        size_t h = (size_t)(sig % (uint64_t)c->tables[t].bucket_count);
        for (lsh_entry_t *e = c->tables[t].buckets[h]; e != NULL; e = e->next) {
            double s = 0.0;
            for (size_t i = 0; i < c->dimension; ++i) {
                double d = point[i] - e->point[i];
                s += d * d;
            }
            if (sqrt(s) <= c->epsilon) return false;
        }
    }

    /* Insert into every table. */
    for (size_t t = 0; t < c->table_count; ++t) {
        uint64_t sig = lsh_compute_signature(c->tables[t].hyperplanes,
                                              c->bits_per_table,
                                              c->dimension, point);
        size_t h = (size_t)(sig % (uint64_t)c->tables[t].bucket_count);
        lsh_entry_t *e = malloc(sizeof(*e));
        if (e == NULL) return true;  /* degenerate: already passed novelty */
        e->point = malloc(c->dimension * sizeof(double));
        if (e->point == NULL) { free(e); return true; }
        memcpy(e->point, point, c->dimension * sizeof(double));
        e->next = c->tables[t].buckets[h];
        c->tables[t].buckets[h] = e;
    }
    c->stored_points++;
    return true;
}

/* ============================================================
 * Exact-set cache (capacity-bounded, L2)
 * Stores up to k points; novelty iff no stored point within epsilon.
 * Saturates: when full, treats subsequent queries as hits (false negative
 * mode), since we cannot insert new points without evicting geometry.
 * ============================================================ */

typedef struct exact_cache {
    double *points;     /* [capacity * dimension] */
    size_t count;
    size_t capacity;
    size_t peak_count;
    size_t dimension;
    double epsilon;
} exact_cache_t;

static exact_cache_t *exact_create(size_t capacity, size_t dim, double eps)
{
    exact_cache_t *c = calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->capacity = capacity;
    c->dimension = dim;
    c->epsilon = eps;
    c->points = calloc(capacity * dim, sizeof(double));
    if (c->points == NULL) { free(c); return NULL; }
    return c;
}

static void exact_destroy(exact_cache_t *c)
{
    free(c->points);
    free(c);
}

static bool exact_observe(exact_cache_t *c, const double *point)
{
    for (size_t i = 0; i < c->count; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < c->dimension; ++k) {
            double d = point[k] - c->points[i * c->dimension + k];
            s += d * d;
        }
        if (sqrt(s) <= c->epsilon) return false;
    }
    if (c->count >= c->capacity) return false;  /* saturated */
    memcpy(c->points + c->count * c->dimension, point,
           c->dimension * sizeof(double));
    c->count++;
    if (c->count > c->peak_count) c->peak_count = c->count;
    return true;
}

/* ============================================================
 * 2D Workloads
 * ============================================================ */

typedef struct workload {
    const char *name;
    double *points;   /* [n * dim] */
    size_t count;
    size_t dim;
    double oracle_epsilon;
} workload_t;

static uint64_t w_rng_state = UINT64_C(0xfeedface12345678);

static uint64_t w_xorshift(void)
{
    uint64_t x = w_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    w_rng_state = x;
    return x;
}

static double w_uniform(void)
{
    return (double)(w_xorshift() >> 11) / (double)(UINT64_C(1) << 53);
}

static double w_gauss(void)
{
    double u1 = w_uniform();
    double u2 = w_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static workload_t *workload_uniform_2d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "uniform-2d";
    w->count = n;
    w->dim = 2;
    w->oracle_epsilon = 0.05;
    w->points = calloc(n * 2, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i * 2] = w_uniform();
        w->points[i * 2 + 1] = w_uniform();
    }
    return w;
}

static workload_t *workload_grid_2d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "grid-32x32";
    w->count = n;
    w->dim = 2;
    w->oracle_epsilon = 0.05;
    w->points = calloc(n * 2, sizeof(double));
    /* Sample 32x32 = 1024 lattice points, with small jitter, n rounds. */
    size_t p = 0;
    for (size_t round = 0; round < (n + 1023) / 1024; ++round) {
        for (int i = 0; i < 32 && p < n; ++i) {
            for (int j = 0; j < 32 && p < n; ++j) {
                w->points[p * 2]     = (i + 0.5) / 32.0 + (w_uniform() - 0.5) * 0.005;
                w->points[p * 2 + 1] = (j + 0.5) / 32.0 + (w_uniform() - 0.5) * 0.005;
                p++;
            }
        }
    }
    return w;
}

static workload_t *workload_two_cluster_2d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "two-cluster";
    w->count = n;
    w->dim = 2;
    w->oracle_epsilon = 0.05;
    w->points = calloc(n * 2, sizeof(double));
    double centers[2][2] = {{0.3, 0.3}, {0.7, 0.7}};
    double sigma = 0.02;
    for (size_t i = 0; i < n; ++i) {
        size_t cidx = i & 1U;
        w->points[i * 2]     = centers[cidx][0] + sigma * w_gauss();
        w->points[i * 2 + 1] = centers[cidx][1] + sigma * w_gauss();
    }
    return w;
}

static workload_t *workload_ring(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "ring";
    w->count = n;
    w->dim = 2;
    w->oracle_epsilon = 0.04;
    w->points = calloc(n * 2, sizeof(double));
    double r = 0.5;
    for (size_t i = 0; i < n; ++i) {
        double theta = 2.0 * 3.14159265358979323846 * w_uniform();
        double radial = r + 0.005 * w_gauss();
        w->points[i * 2]     = 0.5 + radial * cos(theta);
        w->points[i * 2 + 1] = 0.5 + radial * sin(theta);
    }
    return w;
}

static workload_t *workload_line(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "line";
    w->count = n;
    w->dim = 2;
    w->oracle_epsilon = 0.04;
    w->points = calloc(n * 2, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double t = w_uniform();
        double jitter = 0.005 * w_gauss();
        w->points[i * 2]     = t;
        w->points[i * 2 + 1] = t + jitter;
    }
    return w;
}

/* 3D workloads */

static workload_t *workload_uniform_3d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "uniform-3d";
    w->count = n;
    w->dim = 3;
    w->oracle_epsilon = 0.08;  /* larger ε to admit any non-trivially-novel point */
    w->points = calloc(n * 3, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i * 3]     = w_uniform();
        w->points[i * 3 + 1] = w_uniform();
        w->points[i * 3 + 2] = w_uniform();
    }
    return w;
}

/* Points on a 2-sphere in R^3, surface-only (the curse-of-dimensionality
 * classic). Each point is uniformly distributed on the sphere by
 * normalizing a 3D Gaussian vector. The "novelty" question on the
 * sphere is genuinely 3D-curved; random-hyperplane LSH is the standard
 * competitor and degrades as the surface area grows. */
static workload_t *workload_sphere_3d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "sphere-3d";
    w->count = n;
    w->dim = 3;
    w->oracle_epsilon = 0.15;
    w->points = calloc(n * 3, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        /* Sample 3D Gaussian and normalize. */
        double x = w_gauss(), y = w_gauss(), z = w_gauss();
        double norm = sqrt(x*x + y*y + z*z);
        if (norm > 0.0) {
            w->points[i * 3]     = x / norm;
            w->points[i * 3 + 1] = y / norm;
            w->points[i * 3 + 2] = z / norm;
        }
    }
    return w;
}

/* 3D version of two-cluster: two 3D Gaussian clusters in [0,1]^3. */
static workload_t *workload_two_cluster_3d(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "two-cluster-3d";
    w->count = n;
    w->dim = 3;
    w->oracle_epsilon = 0.08;
    w->points = calloc(n * 3, sizeof(double));
    double centers[2][3] = {{0.3, 0.3, 0.3}, {0.7, 0.7, 0.7}};
    double sigma = 0.03;
    for (size_t i = 0; i < n; ++i) {
        size_t cidx = i & 1U;
        w->points[i * 3]     = centers[cidx][0] + sigma * w_gauss();
        w->points[i * 3 + 1] = centers[cidx][1] + sigma * w_gauss();
        w->points[i * 3 + 2] = centers[cidx][2] + sigma * w_gauss();
    }
    return w;
}

static void workload_free(workload_t *w)
{
    free(w->points);
    free(w);
}

/* ============================================================
 * Oracle (O(n^2) L2 metric novelty)
 * ============================================================ */

static void naive_metric_novelty_2d(const double *points, size_t n,
                                     double epsilon, uint8_t *novel)
{
    for (size_t i = 0; i < n; ++i) {
        novel[i] = 1U;
        for (size_t j = 0; j < i; ++j) {
            double dx = points[i * 2]     - points[j * 2];
            double dy = points[i * 2 + 1] - points[j * 2 + 1];
            if (sqrt(dx * dx + dy * dy) <= epsilon) {
                novel[i] = 0U;
                break;
            }
        }
    }
}

typedef struct result {
    const char *method;
    char param_str[40];
    size_t peak_units;
    size_t errors;
    size_t fp;
    size_t fn;
    double seconds;
} result_t;

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static result_t run_kcenter(const double *points, size_t n, size_t dim,
                             size_t capacity, double epsilon,
                             const uint8_t *oracle)
{
    result_t r = {0};
    r.method = "kcenter";
    snprintf(r.param_str, sizeof(r.param_str), "k=%zu,e=%.4g",
             capacity, epsilon);
    kcenter_cache_t *c = kcenter_create(capacity, dim);
    if (c == NULL) { r.errors = n; return r; }
    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool novel = kcenter_observe(c, points + i * dim, epsilon);
        if (novel != (oracle[i] != 0U)) {
            r.errors++;
            if (novel && !oracle[i]) r.fp++;
            else r.fn++;
        }
    }
    r.peak_units = c->peak_size;
    r.seconds = monotonic_seconds() - t0;
    kcenter_destroy(c);
    return r;
}

static result_t run_lsh(const double *points, size_t n, size_t dim,
                         size_t table_count, size_t bits,
                         size_t bucket_count, double epsilon,
                         const uint8_t *oracle)
{
    result_t r = {0};
    r.method = "lsh";
    snprintf(r.param_str, sizeof(r.param_str), "T=%zu,b=%zu,B=%zu",
             table_count, bits, bucket_count);
    lsh_cache_t *c = lsh_create(table_count, bits, bucket_count, dim, epsilon);
    if (c == NULL) { r.errors = n; return r; }
    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool novel = lsh_observe(c, points + i * dim);
        if (novel != (oracle[i] != 0U)) {
            r.errors++;
            if (novel && !oracle[i]) r.fp++;
            else r.fn++;
        }
    }
    r.peak_units = c->stored_points;
    r.seconds = monotonic_seconds() - t0;
    lsh_destroy(c);
    return r;
}

static result_t run_exact(const double *points, size_t n, size_t dim,
                           size_t capacity, double epsilon,
                           const uint8_t *oracle)
{
    result_t r = {0};
    r.method = "exact";
    snprintf(r.param_str, sizeof(r.param_str), "k=%zu", capacity);
    exact_cache_t *c = exact_create(capacity, dim, epsilon);
    if (c == NULL) { r.errors = n; return r; }
    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool novel = exact_observe(c, points + i * dim);
        if (novel != (oracle[i] != 0U)) {
            r.errors++;
            if (novel && !oracle[i]) r.fp++;
            else r.fn++;
        }
    }
    r.peak_units = c->peak_count;
    r.seconds = monotonic_seconds() - t0;
    exact_destroy(c);
    return r;
}

static result_t run_futc(const double *points, size_t n, size_t dim,
                          double epsilon, const uint8_t *oracle)
{
    result_t r = {0};
    r.method = "futc";
    snprintf(r.param_str, sizeof(r.param_str), "e=%.4g", epsilon);

    double lo[16], hi[16];
    for (size_t k = 0; k < dim; ++k) { lo[k] = -10.0; hi[k] = 10.0; }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = dim;
    cfg.epsilon = epsilon;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        r.errors = n; return r;
    }
    double t0 = monotonic_seconds();
    size_t peak = 0;
    for (size_t i = 0; i < n; ++i) {
        bool novel = false;
        if (futcache_pack_observe(cache, points + i * dim, &novel)
            != FUTCACHE_OK) { r.errors = n; break; }
        if (novel != (oracle[i] != 0U)) {
            r.errors++;
            if (novel && !oracle[i]) r.fp++;
            else r.fn++;
        }
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache, &stats);
        if (stats.representative_count > peak) {
            peak = stats.representative_count;
        }
    }
    r.peak_units = peak;
    r.seconds = monotonic_seconds() - t0;
    futcache_pack_destroy(cache);
    return r;
}

static void print_row(const result_t *r, size_t n)
{
    double us = r->seconds > 0.0 ? (r->seconds * 1e6) / (double)n : 0.0;
    double err = n > 0 ? (double)r->errors / (double)n : 0.0;
    printf("| %-7s | %-23s | %10zu | %6.4f | %4zu | %4zu | %6.2f |\n",
           r->method, r->param_str, r->peak_units, err,
           r->fp, r->fn, us);
}

int main(void)
{
    enum { N = 5000 };

    workload_t *(*factories[])(size_t) = {
        workload_uniform_2d, workload_grid_2d,
        workload_two_cluster_2d, workload_ring, workload_line,
        workload_uniform_3d, workload_sphere_3d, workload_two_cluster_3d,
    };
    size_t workload_count = sizeof(factories) / sizeof(factories[0]);

    size_t kc_ks[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    size_t kc_count = sizeof(kc_ks) / sizeof(kc_ks[0]);

    /* LSH: 4 tables x 8 bits = 4 * 256 = 1024 buckets per table.
     * Inner geometric tests still use L2 epsilon. */
    struct { size_t T; size_t bits; size_t B; } lsh_params[] = {
        {2, 6, 256}, {4, 8, 512}, {8, 8, 1024}, {4, 12, 4096}
    };
    size_t lsh_count = sizeof(lsh_params) / sizeof(lsh_params[0]);

    /* Includes the per-workload oracle epsilons (0.08, 0.15) so FUTCache
     * can be compared to kcenter/LSH at the SAME epsilon they are run at. */
    double futc_eps[] = {
        0.2, 0.15, 0.1, 0.08, 0.05, 0.025, 0.01, 0.005
    };
    size_t futc_count = sizeof(futc_eps) / sizeof(futc_eps[0]);

    puts("# 2D geometric-baseline comparison: kcenter, LSH, exact, FUTCache");
    puts("# N = 5000 per workload, dim = 2 or 3 per workload, L2 metric");
    puts("# All methods answer the metric-novelty predicate at target epsilon.");
    puts("# FP = false positive (cache said novel, oracle said not)");
    puts("# FN = false negative (cache said not novel, oracle said novel)");
    puts("#");

    for (size_t w = 0; w < workload_count; ++w) {
        workload_t *wl = factories[w](N);
        uint8_t *oracle = malloc(wl->count);
        naive_metric_novelty_2d(wl->points, wl->count,
                                 wl->oracle_epsilon, oracle);
        size_t oracle_novel = 0;
        for (size_t i = 0; i < wl->count; ++i) {
            if (oracle[i]) oracle_novel++;
        }

        printf("## Workload: %s (d=%zu)   oracle_eps=%g   oracle_novel=%zu / %zu\n\n",
               wl->name, wl->dim, wl->oracle_epsilon, oracle_novel, wl->count);

        printf("| method | param                   | peak_units | error  |  FP  |  FN  | us/op |\n");
        printf("|--------|-------------------------|------------|--------|-------|------|-------|\n");

        /* exact at full capacity (oracle-equivalent). */
        result_t rex = run_exact(wl->points, wl->count, wl->dim, N,
                                  wl->oracle_epsilon, oracle);
        print_row(&rex, wl->count);
        printf("| ------ | ----------------------- | ---------- | ------ | ----- | ---- | ------ |\n");

        /* kcenter sweep. */
        for (size_t i = 0; i < kc_count; ++i) {
            result_t r = run_kcenter(wl->points, wl->count, wl->dim,
                                      kc_ks[i], wl->oracle_epsilon, oracle);
            print_row(&r, wl->count);
        }

        printf("| ------ | ----------------------- | ---------- | ------ | ----- | ---- | ------ |\n");

        /* LSH. */
        for (size_t i = 0; i < lsh_count; ++i) {
            result_t r = run_lsh(wl->points, wl->count, wl->dim,
                                  lsh_params[i].T, lsh_params[i].bits,
                                  lsh_params[i].B, wl->oracle_epsilon, oracle);
            print_row(&r, wl->count);
        }

        printf("| ------ | ----------------------- | ---------- | ------ | ----- | ---- | ------ |\n");

        /* FUTCache. */
        for (size_t i = 0; i < futc_count; ++i) {
            result_t r = run_futc(wl->points, wl->count, wl->dim,
                                   futc_eps[i], oracle);
            print_row(&r, wl->count);
        }

        puts("");
        free(oracle);
        workload_free(wl);
    }

    return EXIT_SUCCESS;
}

#define _POSIX_C_SOURCE 200809L

/*
 * cache_comparison_extended.c
 *
 * Extends bench/cache_comparison.c with two geometric-nearest-neighbor
 * baselines that answer the same metric-novelty predicate as FUTCache:
 *
 *   kcenter:  capacity-bounded k-center cache. Maintains at most k
 *             representative centers. Each new point is novel iff its
 *             distance to every existing center exceeds epsilon. On a
 *             novel point and a full cache, the center closest to the
 *             new point is replaced (geometrically aware eviction).
 *             This is the natural competitor for FUTCache: same
 *             predicate, capacity-bounded, no packing bound.
 *
 *   lsh:      locality-sensitive hashing with k hash tables of b bits.
 *             Novelty = "no bucket within any table contains a point
 *             within epsilon." Approximate: there is no certificate.
 *             The point is to show the error/speed tradeoff.
 *
 * Both baselines answer the *metric* predicate. Comparing them head to
 * head against FUTCache on the same workloads isolates the geometric
 * contribution from the predicate-mismatch effect of the original LRU
 * comparison.
 *
 * For each (workload, method, parameter) the bench reports:
 *   - peak_units: maximum resident representatives/centers/buckets
 *   - decision_error: fraction of points where cache novelty disagrees
 *                     with the metric-novelty oracle at target epsilon
 *   - FP, FN: false positive / false negative counts
 *   - us/op: per-observe latency
 *
 * This bench is intentionally a self-contained copy of the workload and
 * oracle machinery in cache_comparison.c so the new baselines can be
 * reviewed independently of the original.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/futcache.h"

/* ============================================================
 * k-center cache (capacity-bounded)
 *
 * Maintains at most k centers in [domain_min, domain_max]. Novelty is
 * tested against the centers with the same predicate as FUTCache.
 * Eviction is geometric: when over capacity, replace the center closest
 * to the new point (the least load-bearing in dense regions). This is
 * the classical Gonzalez k-center greedy without the oracle, applied
 * online.
 *
 * Logical memory = number of centers; allocator bytes not counted.
 * ============================================================ */

typedef struct kcenter_node {
    double key;
    struct kcenter_node *next;  /* linked list of all centers */
} kcenter_node_t;

typedef struct kcenter_cache {
    kcenter_node_t *head;
    size_t capacity;
    size_t size;
    size_t peak_size;
    double domain_min;
    double domain_max;
} kcenter_cache_t;

static kcenter_cache_t *kcenter_create(size_t capacity,
                                        double domain_min,
                                        double domain_max)
{
    kcenter_cache_t *c = calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->capacity = capacity;
    c->domain_min = domain_min;
    c->domain_max = domain_max;
    return c;
}

static void kcenter_destroy(kcenter_cache_t *c)
{
    kcenter_node_t *n = c->head;
    while (n != NULL) {
        kcenter_node_t *next = n->next;
        free(n);
        n = next;
    }
    free(c);
}

static bool kcenter_observe(kcenter_cache_t *c, double key, double epsilon)
{
    /* Test novelty against existing centers. */
    for (kcenter_node_t *n = c->head; n != NULL; n = n->next) {
        double d = key - n->key;
        if (d < 0.0) d = -d;
        if (d <= epsilon) {
            /* Hit: optionally move-to-front is unnecessary since
             * eviction is geometric, not recency-based. */
            return false;
        }
    }

    /* Novel: insert at head. */
    if (c->size >= c->capacity) {
        /* Evict the center closest to the new point (Gonzalez-style
         * online k-center). Walk the list and find min-distance. */
        kcenter_node_t *prev = NULL;
        kcenter_node_t *closest = NULL;
        kcenter_node_t *closest_prev = NULL;
        double min_d = 1e300;
        for (kcenter_node_t *n = c->head; n != NULL; n = n->next) {
            double d = key - n->key;
            if (d < 0.0) d = -d;
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
            free(closest);
            c->size--;
        }
    }

    kcenter_node_t *node = malloc(sizeof(*node));
    if (node == NULL) return false;  /* degenerate */
    node->key = key;
    node->next = c->head;
    c->head = node;
    c->size++;
    if (c->size > c->peak_size) c->peak_size = c->size;
    return true;
}

/* ============================================================
 * LSH cache (1D LSH = grid bucket table)
 *
 * In 1D, LSH with a single hash table is a regular grid with cell width
 * 2*epsilon. Novelty = "the cell and its neighbors contain no prior
 * point." Implemented as a hash table keyed by floor(x / (2*eps)).
 *
 * This is the standard 1D LSH baseline. In higher dimensions the
 * analogous construction is projection onto 1D, which we omit to keep
 * this self-contained.
 * ============================================================ */

typedef struct lsh_node {
    int64_t bucket;
    double key;
    struct lsh_node *next;
} lsh_node_t;

typedef struct lsh_cache {
    lsh_node_t **buckets;
    size_t bucket_count;
    size_t count;
    double epsilon;
    double cell_width;  /* = 2 * epsilon */
} lsh_cache_t;

static uint64_t lsh_bucket_hash(int64_t bucket, size_t bucket_count)
{
    /* Knuth multiplicative hash on signed integer. */
    uint64_t x = (uint64_t)bucket;
    x *= UINT64_C(11400714819323198485);
    return (size_t)(x % (uint64_t)bucket_count);
}

static lsh_cache_t *lsh_create(size_t bucket_count, double epsilon)
{
    lsh_cache_t *c = calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->bucket_count = bucket_count;
    c->buckets = calloc(bucket_count, sizeof(*c->buckets));
    if (c->buckets == NULL) {
        free(c);
        return NULL;
    }
    c->epsilon = epsilon;
    c->cell_width = 2.0 * epsilon;
    return c;
}

static void lsh_destroy(lsh_cache_t *c)
{
    for (size_t i = 0; i < c->bucket_count; ++i) {
        lsh_node_t *n = c->buckets[i];
        while (n != NULL) {
            lsh_node_t *next = n->next;
            free(n);
            n = next;
        }
    }
    free(c->buckets);
    free(c);
}

static bool lsh_observe(lsh_cache_t *c, double key)
{
    int64_t center_bucket = (int64_t)floor(key / c->cell_width);

    /* Probe the cell and its two neighbors. */
    for (int64_t b = center_bucket - 1; b <= center_bucket + 1; ++b) {
        size_t h = lsh_bucket_hash(b, c->bucket_count);
        for (lsh_node_t *n = c->buckets[h]; n != NULL; n = n->next) {
            if (n->bucket == b) {
                double d = key - n->key;
                if (d < 0.0) d = -d;
                if (d <= c->epsilon) return false;
            }
        }
    }

    /* Insert. */
    size_t h = lsh_bucket_hash(center_bucket, c->bucket_count);
    lsh_node_t *node = malloc(sizeof(*node));
    if (node == NULL) return false;
    node->bucket = center_bucket;
    node->key = key;
    node->next = c->buckets[h];
    c->buckets[h] = node;
    c->count++;
    return true;
}

/* ============================================================
 * Workloads (same as bench/cache_comparison.c for direct comparison)
 * ============================================================ */

typedef struct workload {
    const char *name;
    double *points;
    size_t count;
    double oracle_epsilon;
    double domain_min;
    double domain_max;
} workload_t;

static uint64_t rng_state = UINT64_C(0x123456789abcdef0);

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

static double gaussian(double mean, double stddev)
{
    double u1 = uniform_double();
    double u2 = uniform_double();
    if (u1 < 1e-300) u1 = 1e-300;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

static workload_t *workload_reciprocal(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "reciprocal";
    w->count = n;
    w->oracle_epsilon = 0.01;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = 1.0 / (double)(i + 1);
    }
    return w;
}

static workload_t *workload_uniform(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "uniform";
    w->count = n;
    w->oracle_epsilon = 0.01;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = uniform_double();
    }
    return w;
}

static workload_t *workload_cluster(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "three-cluster";
    w->count = n;
    w->oracle_epsilon = 0.05;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    double centers[3] = {0.2, 0.5, 0.8};
    for (size_t i = 0; i < n; ++i) {
        double x = gaussian(centers[i % 3], 0.01);
        if (x < 0.0) x = 0.0;
        if (x > 1.0) x = 1.0;
        w->points[i] = x;
    }
    return w;
}

static workload_t *workload_alternating(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "alternating-extremes";
    w->count = n;
    w->oracle_epsilon = 0.5;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = (i % 2U == 0U) ? 0.01 : 0.99;
    }
    return w;
}

static workload_t *workload_power(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "power-decay";
    w->count = n;
    w->oracle_epsilon = 0.05;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = uniform_double() * uniform_double();
    }
    return w;
}

static void workload_free(workload_t *w)
{
    free(w->points);
    free(w);
}

/* ============================================================
 * Oracle (same as bench/cache_comparison.c)
 * ============================================================ */

static void naive_metric_novelty(const double *points, size_t n,
                                  double epsilon, uint8_t *novel)
{
    for (size_t i = 0; i < n; ++i) {
        novel[i] = 1U;
        for (size_t j = 0; j < i; ++j) {
            double d = points[i] - points[j];
            if (d < 0.0) d = -d;
            if (d <= epsilon) {
                novel[i] = 0U;
                break;
            }
        }
    }
}

typedef struct result {
    const char *method;
    char param_str[32];
    size_t peak_units;
    size_t errors;
    size_t false_positives;
    size_t false_negatives;
    double error_rate;
    double seconds;
} result_t;

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static result_t run_kcenter(const double *points, size_t n,
                             size_t capacity, double epsilon,
                             const uint8_t *oracle_novel,
                             double domain_min, double domain_max)
{
    result_t r;
    memset(&r, 0, sizeof(r));
    r.method = "kcenter";
    snprintf(r.param_str, sizeof(r.param_str), "k=%zu,e=%.4g",
             capacity, epsilon);

    kcenter_cache_t *c = kcenter_create(capacity, domain_min, domain_max);
    if (c == NULL) { r.error_rate = 1.0; return r; }

    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool cache_novel = kcenter_observe(c, points[i], epsilon);
        bool oracle_n = oracle_novel[i] != 0U;
        if (cache_novel != oracle_n) {
            r.errors++;
            if (cache_novel && !oracle_n) r.false_positives++;
            else r.false_negatives++;
        }
    }
    r.peak_units = c->peak_size;
    r.error_rate = (double)r.errors / (double)n;
    r.seconds = monotonic_seconds() - t0;
    kcenter_destroy(c);
    return r;
}

static result_t run_lsh(const double *points, size_t n,
                         size_t bucket_count, double epsilon,
                         const uint8_t *oracle_novel)
{
    result_t r;
    memset(&r, 0, sizeof(r));
    r.method = "lsh";
    snprintf(r.param_str, sizeof(r.param_str), "b=%zu,e=%.4g",
             bucket_count, epsilon);

    lsh_cache_t *c = lsh_create(bucket_count, epsilon);
    if (c == NULL) { r.error_rate = 1.0; return r; }

    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool cache_novel = lsh_observe(c, points[i]);
        bool oracle_n = oracle_novel[i] != 0U;
        if (cache_novel != oracle_n) {
            r.errors++;
            if (cache_novel && !oracle_n) r.false_positives++;
            else r.false_negatives++;
        }
    }
    r.peak_units = c->count;
    r.error_rate = (double)r.errors / (double)n;
    r.seconds = monotonic_seconds() - t0;
    lsh_destroy(c);
    return r;
}

static result_t run_futc(const double *points, size_t n,
                          double epsilon,
                          const uint8_t *oracle_novel,
                          double domain_min, double domain_max)
{
    result_t r;
    memset(&r, 0, sizeof(r));
    r.method = "futc";
    snprintf(r.param_str, sizeof(r.param_str), "e=%.4g", epsilon);

    futcache_config_t cfg;
    futcache_config_init(&cfg);
    cfg.domain_min = domain_min;
    cfg.domain_max = domain_max;
    cfg.epsilon = epsilon;

    futcache_t *cache = NULL;
    if (futcache_create(&cfg, &cache) != FUTCACHE_OK) {
        r.error_rate = 1.0;
        return r;
    }

    size_t peak_intervals = 0;
    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool cache_novel = false;
        if (futcache_observe(cache, points[i], &cache_novel) != FUTCACHE_OK) {
            r.error_rate = 1.0;
            futcache_destroy(cache);
            return r;
        }
        bool oracle_n = oracle_novel[i] != 0U;
        if (cache_novel != oracle_n) {
            r.errors++;
            if (cache_novel && !oracle_n) r.false_positives++;
            else r.false_negatives++;
        }
        futcache_stats_t stats;
        futcache_get_stats(cache, &stats);
        if (stats.interval_count > peak_intervals) {
            peak_intervals = stats.interval_count;
        }
    }
    r.peak_units = peak_intervals;
    r.error_rate = (double)r.errors / (double)n;
    r.seconds = monotonic_seconds() - t0;
    futcache_destroy(cache);
    return r;
}

static void print_row(const result_t *r, size_t n)
{
    double us_per_op = r->seconds > 0.0
        ? (r->seconds * 1e6) / (double)n : 0.0;
    printf("| %-7s | %-15s | %10zu | %6.4f | %4zu | %4zu | %5.2f |\n",
        r->method, r->param_str, r->peak_units, r->error_rate,
        r->false_positives, r->false_negatives, us_per_op);
}

int main(void)
{
    enum { N = 10000 };

    workload_t *(*factories[])(size_t) = {
        workload_reciprocal,
        workload_uniform,
        workload_cluster,
        workload_alternating,
        workload_power,
    };
    size_t workload_count = sizeof(factories) / sizeof(factories[0]);

    /* k-center: capacity sweep covering the same memory range as the
     * packing-bound sweep. */
    size_t kc_ks[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    size_t kc_count = sizeof(kc_ks) / sizeof(kc_ks[0]);

    /* LSH: bucket-count sweep at the workload's oracle epsilon. */
    size_t lsh_bs[] = {64, 256, 1024, 4096, 16384, 65536};
    size_t lsh_count = sizeof(lsh_bs) / sizeof(lsh_bs[0]);

    /* FUTCache: same epsilon grid as bench/cache_comparison.c. */
    double futc_eps[] = {
        0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005
    };
    size_t futc_count = sizeof(futc_eps) / sizeof(futc_eps[0]);

    puts("# Geometric-baseline comparison: k-center, LSH, FUTCache");
    puts("# N = 10000 per workload");
    puts("# All three methods answer the metric-novelty predicate at the");
    puts("# given epsilon. Decision error = fraction of points where cache");
    puts("# novelty disagrees with the metric oracle at target epsilon.");
    puts("# FP = false positive (cache said novel, oracle said not)");
    puts("# FN = false negative (cache said not novel, oracle said novel)");
    puts("#");

    for (size_t w = 0; w < workload_count; ++w) {
        workload_t *wl = factories[w](N);
        uint8_t *oracle = malloc(wl->count);
        naive_metric_novelty(wl->points, wl->count, wl->oracle_epsilon,
                              oracle);
        size_t oracle_novel = 0;
        for (size_t i = 0; i < wl->count; ++i) {
            if (oracle[i] != 0U) oracle_novel++;
        }

        printf("## Workload: %s   oracle_eps=%g   oracle_novel=%zu / %zu\n\n",
               wl->name, wl->oracle_epsilon, oracle_novel, wl->count);

        printf("| method | param           | peak_units | error  |  FP  |  FN  | us/op |\n");
        printf("|--------|-----------------|------------|--------|-------|------|-------|\n");

        /* k-center: sweep at the workload's oracle epsilon. */
        for (size_t i = 0; i < kc_count; ++i) {
            result_t r = run_kcenter(wl->points, wl->count, kc_ks[i],
                                      wl->oracle_epsilon, oracle,
                                      wl->domain_min, wl->domain_max);
            print_row(&r, wl->count);
        }

        printf("| ------ | --------------- | ---------- | ------ | ----- | ---- | ------ |\n");

        /* LSH at the oracle epsilon. */
        for (size_t i = 0; i < lsh_count; ++i) {
            result_t r = run_lsh(wl->points, wl->count, lsh_bs[i],
                                  wl->oracle_epsilon, oracle);
            print_row(&r, wl->count);
        }

        printf("| ------ | --------------- | ---------- | ------ | ----- | ---- | ------ |\n");

        /* FUTCache at the oracle epsilon. */
        for (size_t i = 0; i < futc_count; ++i) {
            result_t r = run_futc(wl->points, wl->count, futc_eps[i],
                                   oracle, wl->domain_min, wl->domain_max);
            print_row(&r, wl->count);
        }

        puts("");
        free(oracle);
        workload_free(wl);
    }

    return EXIT_SUCCESS;
}

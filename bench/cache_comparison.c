#define _POSIX_C_SOURCE 200809L

#include <time.h>

/*
 * cache_comparison.c
 *
 * Compares FUTCache (interval-union) against LRU and an exact set cache
 * on multiple 1D continuous workloads.
 *
 * For each workload:
 *   1. Compute the metric-novelty oracle at a fixed target epsilon.
 *      oracle_novel[i] = 1 iff points[i] is more than epsilon away from
 *      every earlier point in the stream. This is the ground truth.
 *   2. For each (method, parameter) pair, replay the stream:
 *      - Track peak logical occupancy of the cache.
 *      - Track decision_error = fraction of points where the cache's
 *        novelty answer disagrees with the oracle.
 *   3. Emit a Pareto-frontier table (memory vs error) per workload.
 *
 * The benchmark answers the systems question:
 *   "At a given memory budget, which cache predicts the oracle best?"
 *   "At a given error target, which cache uses the least memory?"
 *
 * For exact set caches the parameter is k = capacity.
 * For LRU the parameter is k = capacity.
 * For FUTCache the parameter is epsilon (resolution).
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/futcache.h"

/* ============================================================
 * LRU cache
 * Doubly-linked list + hash table. O(1) observe, O(1) hit test.
 * Logical memory = resident entries; allocator bytes are not included here.
 * ============================================================ */

typedef struct lru_node {
    double key;
    struct lru_node *list_prev;
    struct lru_node *list_next;
    struct lru_node *hash_next;
} lru_node_t;

typedef struct lru_cache {
    lru_node_t *head;        /* most recently used */
    lru_node_t *tail;        /* least recently used */
    lru_node_t **buckets;
    size_t bucket_count;
    size_t capacity;
    size_t size;
    size_t peak_size;
} lru_cache_t;

static uint64_t lru_hash(double key)
{
    union {
        double d;
        uint64_t u;
    } v;
    v.d = key;
    /* mix bits so hash table spreads; Knuth multiplicative hash */
    return v.u * UINT64_C(11400714819323198485);
}

static lru_cache_t *lru_create(size_t capacity)
{
    lru_cache_t *lru = calloc(1, sizeof(*lru));
    if (lru == NULL) return NULL;
    lru->capacity = capacity;
    lru->bucket_count = capacity * 4U + 16U;
    lru->buckets = calloc(lru->bucket_count, sizeof(*lru->buckets));
    if (lru->buckets == NULL) {
        free(lru);
        return NULL;
    }
    return lru;
}

static void lru_unlink(lru_cache_t *lru, lru_node_t *node)
{
    if (node->list_prev != NULL) node->list_prev->list_next = node->list_next;
    else lru->head = node->list_next;
    if (node->list_next != NULL) node->list_next->list_prev = node->list_prev;
    else lru->tail = node->list_prev;

    uint64_t h = lru_hash(node->key);
    size_t b = (size_t)(h % lru->bucket_count);
    if (lru->buckets[b] == node) {
        lru->buckets[b] = node->hash_next;
    } else {
        lru_node_t *prev = lru->buckets[b];
        while (prev != NULL && prev->hash_next != node) {
            prev = prev->hash_next;
        }
        if (prev != NULL) prev->hash_next = node->hash_next;
    }
}

static bool lru_observe(lru_cache_t *lru, double key)
{
    uint64_t h = lru_hash(key);
    size_t b = (size_t)(h % lru->bucket_count);

    for (lru_node_t *n = lru->buckets[b]; n != NULL; n = n->hash_next) {
        if (n->key == key) {
            /* hit: move to head */
            if (n != lru->head) {
                lru_unlink(lru, n);
                n->list_prev = NULL;
                n->list_next = lru->head;
                if (lru->head != NULL) lru->head->list_prev = n;
                lru->head = n;
                if (lru->tail == NULL) lru->tail = n;
                n->hash_next = lru->buckets[b];
                lru->buckets[b] = n;
            }
            return false;
        }
    }

    /* miss: evict if necessary */
    if (lru->size >= lru->capacity && lru->tail != NULL) {
        lru_node_t *victim = lru->tail;
        lru_unlink(lru, victim);
        free(victim);
        lru->size--;
    }

    lru_node_t *node = malloc(sizeof(*node));
    if (node == NULL) return false; /* degenerate */
    node->key = key;
    node->list_prev = NULL;
    node->list_next = lru->head;
    if (lru->head != NULL) lru->head->list_prev = node;
    lru->head = node;
    if (lru->tail == NULL) lru->tail = node;
    node->hash_next = lru->buckets[b];
    lru->buckets[b] = node;
    lru->size++;
    if (lru->size > lru->peak_size) lru->peak_size = lru->size;
    return true;
}

static void lru_destroy(lru_cache_t *lru)
{
    lru_node_t *n = lru->head;
    while (n != NULL) {
        lru_node_t *next = n->list_next;
        free(n);
        n = next;
    }
    free(lru->buckets);
    free(lru);
}

/* ============================================================
 * Exact-set cache
 * Stores every distinct point seen. Capacity-capped.
 * Logical memory = resident entries; allocator bytes are not included here.
 * ============================================================ */

typedef struct exact_cache {
    double *keys;
    size_t count;
    size_t capacity;
    size_t peak_count;
} exact_cache_t;

static exact_cache_t *exact_create(size_t capacity)
{
    exact_cache_t *e = calloc(1, sizeof(*e));
    if (e == NULL) return NULL;
    e->capacity = capacity;
    e->keys = calloc(capacity, sizeof(double));
    if (e->keys == NULL) {
        free(e);
        return NULL;
    }
    return e;
}

static bool exact_observe(exact_cache_t *e, double key)
{
    for (size_t i = 0; i < e->count; ++i) {
        if (e->keys[i] == key) return false;
    }
    if (e->count >= e->capacity) {
        /* saturated: treat subsequent hits as "seen" (not novel) */
        return false;
    }
    e->keys[e->count++] = key;
    if (e->count > e->peak_count) e->peak_count = e->count;
    return true;
}

static void exact_destroy(exact_cache_t *e)
{
    free(e->keys);
    free(e);
}

/* ============================================================
 * Workloads
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
    /* 53-bit precision uniform in [0, 1) */
    return (double)(xorshift64() >> 11) / (double)(UINT64_C(1) << 53);
}

static double gaussian(double mean, double stddev)
{
    /* Box-Muller; consume two uniforms */
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
        /* skewed toward 0; x = u^2 has density 1/(2*sqrt(x)) */
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
 * Oracle
 *
 * naive_metric_novelty:
 *   For each point, novelty is 1 iff its distance to every earlier
 *   point exceeds epsilon. O(n^2). Acceptable for n <= ~50000.
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

/* ============================================================
 * Result of one (method, param) replay
 * ============================================================ */

typedef struct result {
    const char *method;
    char param_str[24];
    size_t final_memory;
    size_t peak_memory;
    size_t errors;
    size_t false_positives;  /* cache said novel, oracle said not */
    size_t false_negatives;  /* cache said not novel, oracle said novel */
    double error_rate;
    size_t novel_count;
} result_t;

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static result_t run_lru(const double *points, size_t n,
                         size_t capacity,
                         const uint8_t *oracle_novel,
                         double *out_seconds)
{
    result_t r;
    memset(&r, 0, sizeof(r));
    r.method = "lru";
    snprintf(r.param_str, sizeof(r.param_str), "k=%zu", capacity);

    lru_cache_t *cache = lru_create(capacity);
    if (cache == NULL) {
        r.error_rate = 1.0;
        return r;
    }

    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool cache_novel = lru_observe(cache, points[i]);
        bool oracle_n = oracle_novel[i] != 0U;
        if (cache_novel != oracle_n) {
            r.errors++;
            if (cache_novel && !oracle_n) r.false_positives++;
            else r.false_negatives++;
        }
        if (cache_novel) r.novel_count++;
    }
    double t1 = monotonic_seconds();

    r.peak_memory = cache->peak_size;
    r.final_memory = cache->size;
    r.error_rate = (double)r.errors / (double)n;
    if (out_seconds != NULL) *out_seconds = t1 - t0;
    lru_destroy(cache);
    return r;
}

static result_t run_exact(const double *points, size_t n,
                           size_t capacity,
                           const uint8_t *oracle_novel,
                           double *out_seconds)
{
    result_t r;
    memset(&r, 0, sizeof(r));
    r.method = "exact";
    snprintf(r.param_str, sizeof(r.param_str), "k=%zu", capacity);

    exact_cache_t *cache = exact_create(capacity);
    if (cache == NULL) {
        r.error_rate = 1.0;
        return r;
    }

    double t0 = monotonic_seconds();
    for (size_t i = 0; i < n; ++i) {
        bool cache_novel = exact_observe(cache, points[i]);
        bool oracle_n = oracle_novel[i] != 0U;
        if (cache_novel != oracle_n) {
            r.errors++;
            if (cache_novel && !oracle_n) r.false_positives++;
            else r.false_negatives++;
        }
        if (cache_novel) r.novel_count++;
    }
    double t1 = monotonic_seconds();

    r.peak_memory = cache->peak_count;
    r.final_memory = cache->count;
    r.error_rate = (double)r.errors / (double)n;
    if (out_seconds != NULL) *out_seconds = t1 - t0;
    exact_destroy(cache);
    return r;
}

static result_t run_futc(const double *points, size_t n,
                          double epsilon,
                          const uint8_t *oracle_novel,
                          double domain_min, double domain_max,
                          double *out_seconds)
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
        if (cache_novel) r.novel_count++;
        futcache_stats_t stats;
        if (futcache_get_stats(cache, &stats) != FUTCACHE_OK) {
            r.error_rate = 1.0;
            futcache_destroy(cache);
            return r;
        }
        if (stats.interval_count > peak_intervals) {
            peak_intervals = stats.interval_count;
        }
    }
    double t1 = monotonic_seconds();

    futcache_stats_t final_stats;
    futcache_get_stats(cache, &final_stats);
    r.peak_memory = peak_intervals;
    r.final_memory = final_stats.interval_count;
    r.error_rate = (double)r.errors / (double)n;
    if (out_seconds != NULL) *out_seconds = t1 - t0;
    futcache_destroy(cache);
    return r;
}

/* ============================================================
 * main
 * ============================================================ */

int main(void)
{
    enum { N = 10000 };

    workload_t *(*workload_factories[])(size_t) = {
        workload_reciprocal,
        workload_uniform,
        workload_cluster,
        workload_alternating,
        workload_power,
    };
    size_t workload_count =
        sizeof(workload_factories) / sizeof(workload_factories[0]);

    size_t lru_ks[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    size_t lru_count = sizeof(lru_ks) / sizeof(lru_ks[0]);

    /* Pick FUTCache epsilons so memory (≈ 1/(2ε) saturating to ~10) is in
     * the same range as LRU k. For domain [0,1] and an absorbing cache,
     * peak interval count is roughly min(N, ceil((b-a)/(2*ε)) + 1). */
    double futc_eps[] = {
        0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005
    };
    size_t futc_count = sizeof(futc_eps) / sizeof(futc_eps[0]);

    puts("# FUTCache vs LRU vs Exact-set: logical memory and decision-error");
    puts("# N = 10000 per workload");
    puts("# decision_error = fraction of points where cache novelty disagrees with metric-novelty oracle at target ε");
    puts("# novel_count   = how many points the cache reported as novel");
    puts("# FP = false positive (cache said novel, oracle said not)");
    puts("# FN = false negative (cache said not novel, oracle said novel)");
    puts("# us/op = microseconds per observe() call (single-thread, N=10000)");
    puts("");

    for (size_t w = 0; w < workload_count; ++w) {
        workload_t *wl = workload_factories[w](N);

        uint8_t *oracle = malloc(wl->count);
        naive_metric_novelty(wl->points, wl->count,
                              wl->oracle_epsilon, oracle);
        size_t oracle_novel_count = 0;
        for (size_t i = 0; i < wl->count; ++i) {
            if (oracle[i] != 0U) oracle_novel_count++;
        }

        printf("## Workload: %s   oracle_eps=%g   oracle_novel=%zu / %zu\n\n",
               wl->name, wl->oracle_epsilon, oracle_novel_count, wl->count);

        printf("| method | param     | peak_units | final_units | error  | novel |  FP  |  FN  | us/op |\n");
        printf("|--------|-----------|------------|-------------|--------|-------|------|------|-------|\n");

        for (size_t i = 0; i < lru_count; ++i) {
            double seconds = 0.0;
            result_t r = run_lru(wl->points, wl->count, lru_ks[i], oracle, &seconds);
            double us_per_op = seconds > 0.0
                ? (seconds * 1e6) / (double)wl->count
                : 0.0;
            printf("| %-7s | %-9s | %10zu | %11zu | %6.4f | %5zu | %4zu | %4zu | %5.2f |\n",
                r.method, r.param_str, r.peak_memory, r.final_memory,
                r.error_rate, r.novel_count, r.false_positives,
                r.false_negatives, us_per_op);
        }
        {
            double seconds = 0.0;
            result_t r = run_exact(wl->points, wl->count, wl->count, oracle, &seconds);
            double us_per_op = seconds > 0.0
                ? (seconds * 1e6) / (double)wl->count
                : 0.0;
            printf("| %-7s | %-9s | %10zu | %11zu | %6.4f | %5zu | %4zu | %4zu | %5.2f |\n",
                r.method, r.param_str, r.peak_memory, r.final_memory,
                r.error_rate, r.novel_count, r.false_positives,
                r.false_negatives, us_per_op);
        }
        printf("| ------ | --------- | ---------- | ----------- | ------ | ----- | ---- | ---- | ------ |\n");
        for (size_t i = 0; i < futc_count; ++i) {
            double seconds = 0.0;
            result_t r = run_futc(wl->points, wl->count, futc_eps[i], oracle,
                                   wl->domain_min, wl->domain_max, &seconds);
            double us_per_op = seconds > 0.0
                ? (seconds * 1e6) / (double)wl->count
                : 0.0;
            printf("| %-7s | %-9s | %10zu | %11zu | %6.4f | %5zu | %4zu | %4zu | %5.2f |\n",
                r.method, r.param_str, r.peak_memory, r.final_memory,
                r.error_rate, r.novel_count, r.false_positives,
                r.false_negatives, us_per_op);
        }

        puts("");
        free(oracle);
        workload_free(wl);
    }

    return EXIT_SUCCESS;
}

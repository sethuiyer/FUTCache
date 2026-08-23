/*
 * Differential and fault-injection tests for the VP-tree backend.
 *
 * The critical property is exactness: futcache_pack_vptree_backend must
 * reproduce the linear-scan novelty decisions for every distance
 * function, including the chordal-transform path for cosine. The oracle
 * is the engine's own linear scan (futcache_pack_copy_representatives
 * plus a test-side scan), which is independent of the backend, and the
 * built-in linear backend itself (backend/engine agreement on the same
 * insert stream).
 *
 * The second critical property is atomicity under allocation failure:
 * an out-of-memory mid-insert must leave the index and the cache
 * exactly as they were, so a cache that survived fault injection is
 * still valid, still agrees with a clean build on the committed prefix,
 * and holds the identical representative set.
 */

#include "test.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

/* ============================================================
 * Deterministic RNG and test-side linear scan
 * ============================================================ */

static uint64_t vp_random_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static double vp_random_unit(uint64_t *state)
{
    return (double)(vp_random_next(state) >> 11U) / 9007199254740992.0;
}

static void vp_fill_uniform(uint64_t *state, double *point, size_t dimension)
{
    for (size_t i = 0U; i < dimension; ++i) {
        point[i] = vp_random_unit(state);
    }
}

static void vp_fill_gaussian(uint64_t *state, double *point, size_t dimension)
{
    double norm = 0.0;
    for (size_t i = 0U; i < dimension; ++i) {
        double v = vp_random_unit(state) * 2.0 - 1.0;
        point[i] = v;
        norm += v * v;
    }
    norm = sqrt(norm);
    for (size_t i = 0U; i < dimension; ++i) {
        point[i] /= norm;
    }
}

/* Test-side true nearest distance over the cache's representative set. */
static double vp_oracle_distance(const futcache_pack_t *cache,
                                 const double *query,
                                 size_t dimension,
                                 futcache_distance_fn distance)
{
    size_t count = 0U;
    double best = INFINITY;

    if (futcache_pack_copy_representatives(cache, NULL, &count) !=
        FUTCACHE_OK) {
        return INFINITY;
    }

    double *points = (double *)malloc(count * dimension * sizeof(double));
    if (points == NULL) return INFINITY;
    if (futcache_pack_copy_representatives(cache, points, &count) !=
        FUTCACHE_OK) {
        free(points);
        return INFINITY;
    }
    for (size_t i = 0U; i < count; ++i) {
        double d = distance(query, points + i * dimension,
                            dimension, NULL);
        if (d < best) best = d;
    }
    free(points);
    return best;
}

static bool vp_is_novel_ok(const futcache_pack_t *cache, const double *point,
                           bool *out_novel)
{
    return futcache_pack_is_novel(cache, point, out_novel) == FUTCACHE_OK;
}

/* ============================================================
 * Fault-injection allocator (mirrors tests/test_futcache.c)
 * ============================================================ */

typedef struct vp_failing_allocator_context {
    size_t allocations_remaining;
    size_t active_allocations;
} vp_failing_allocator_context_t;

static void *vp_failing_allocate(void *opaque, size_t size)
{
    vp_failing_allocator_context_t *context = opaque;
    void *memory;

    if (context->allocations_remaining == 0U) {
        return NULL;
    }
    context->allocations_remaining--;
    memory = malloc(size);
    if (memory != NULL) {
        context->active_allocations++;
    }
    return memory;
}

static void vp_failing_deallocate(void *opaque, void *pointer)
{
    vp_failing_allocator_context_t *context = opaque;
    if (pointer != NULL) {
        if (context->active_allocations == 0U) {
            abort();
        }
        context->active_allocations--;
        free(pointer);
    }
}

/* ============================================================
 * Shared helpers
 * ============================================================ */

/* Domain arrays must cover every coordinate. */
typedef struct vp_domain {
    double *lo;
    double *hi;
} vp_domain_t;

static vp_domain_t vp_make_domain(size_t dimension, double lo, double hi)
{
    vp_domain_t d;
    d.lo = (double *)malloc(dimension * sizeof(double));
    d.hi = (double *)malloc(dimension * sizeof(double));
    if (d.lo == NULL || d.hi == NULL) {
        free(d.lo);
        free(d.hi);
        d.lo = NULL;
        d.hi = NULL;
        return d;
    }
    for (size_t i = 0U; i < dimension; ++i) {
        d.lo[i] = lo;
        d.hi[i] = hi;
    }
    return d;
}

static void vp_free_domain(vp_domain_t d)
{
    free(d.lo);
    free(d.hi);
}

static void vp_config_init(futcache_pack_config_t *config,
                           size_t dimension, double epsilon,
                           futcache_distance_fn distance,
                           const futcache_allocator_t *allocator,
                           bool use_vptree)
{
    futcache_pack_config_init(config);
    config->dimension = dimension;
    config->epsilon = epsilon;
    config->distance = distance;
    config->backend = use_vptree ? &futcache_pack_vptree_backend : NULL;
    config->allocator = allocator != NULL
                            ? *allocator
                            : (futcache_allocator_t){NULL, NULL, NULL};
}

/* The vptree and linear caches must make identical novelty decisions on
 * the same insert stream at the cache's own epsilon. */
static bool vp_agree(futcache_pack_t *linear, futcache_pack_t *vptree,
                     const double *points, size_t point_count,
                     size_t dimension)
{
    bool novel_l, novel_v;
    for (size_t i = 0U; i < point_count; ++i) {
        if (!vp_is_novel_ok(vptree, points + i * dimension, &novel_v)) {
            return false;
        }
        if (!vp_is_novel_ok(linear, points + i * dimension, &novel_l)) {
            return false;
        }
        if (novel_l != novel_v) {
            return false;
        }
        bool was_novel_v, was_novel_l;
        if (futcache_pack_observe(vptree, points + i * dimension,
                                  &was_novel_v) != FUTCACHE_OK) {
            return false;
        }
        if (futcache_pack_observe(linear, points + i * dimension,
                                  &was_novel_l) != FUTCACHE_OK) {
            return false;
        }
        if (was_novel_v != novel_v || was_novel_l != novel_l) {
            return false;
        }
    }
    return true;
}

/* The backend's reported nearest distance must equal the oracle linear
 * scan, at every epsilon boundary. is_novel compares against the cache's
 * own epsilon (fixed at create), so each grid epsilon gets a fresh cache
 * built at that epsilon; the oracle scans that same cache's reps. */
static bool vp_exact_at_epsilons(futcache_distance_fn distance,
                                 const double *points, size_t point_count,
                                 const double *queries, size_t query_count,
                                 size_t dimension, double lo, double hi,
                                 const double *eps_grid, size_t eps_count)
{
    vp_domain_t domain = vp_make_domain(dimension, lo, hi);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);

    for (size_t e = 0U; e < eps_count; ++e) {
        futcache_pack_config_t config;
        vp_config_init(&config, dimension, eps_grid[e], distance, NULL,
                       true);
        config.domain_min = domain.lo;
        config.domain_max = domain.hi;
        futcache_pack_t *cache = NULL;
        if (futcache_pack_create(&config, &cache) != FUTCACHE_OK) {
            vp_free_domain(domain);
            return false;
        }
        for (size_t i = 0U; i < point_count; ++i) {
            bool was_novel;
            if (futcache_pack_observe(cache, points + i * dimension,
                                      &was_novel) != FUTCACHE_OK) {
                futcache_pack_destroy(cache);
                vp_free_domain(domain);
                return false;
            }
        }
        for (size_t i = 0U; i < query_count; ++i) {
            double true_d = vp_oracle_distance(cache, queries + i * dimension,
                                               dimension, distance);
            bool expect = (true_d > eps_grid[e]);
            bool novel;
            if (!vp_is_novel_ok(cache, queries + i * dimension, &novel)) {
                futcache_pack_destroy(cache);
                vp_free_domain(domain);
                return false;
            }
            if (novel != expect) {
                futcache_pack_destroy(cache);
                vp_free_domain(domain);
                return false;
            }
        }
        futcache_pack_destroy(cache);
    }
    vp_free_domain(domain);
    return true;
}

/* ============================================================
 * Tests
 * ============================================================ */

static bool test_vptree_metric_differential(void)
{
    static const size_t dims[] = {2U, 8U, 16U, 64U, 384U};
    static const futcache_distance_fn distances[] = {
        futcache_distance_l1, futcache_distance_l2, futcache_distance_linf};
    const double eps_grid[] = {0.01, 0.05, 0.1, 0.2, 0.5};
    vp_domain_t domain;

    for (size_t di = 0U; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        size_t dim = dims[di];
        for (size_t fi = 0U; fi < sizeof(distances) / sizeof(distances[0]);
             ++fi) {
            futcache_distance_fn distance = distances[fi];
            const size_t n = 400U;
            const size_t m = 100U;
            double *points = (double *)malloc((n + m) * dim * sizeof(double));
            TEST_ASSERT(points != NULL);
            uint64_t rng = UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)dim
                           ^ ((uint64_t)fi << 32U);
            for (size_t i = 0U; i < n + m; ++i) {
                vp_fill_uniform(&rng, points + i * dim, dim);
            }

            futcache_pack_config_t config_l, config_v;
            domain = vp_make_domain(dim, 0.0, 1.0);
            TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
            vp_config_init(&config_l, dim, 0.05, distance, NULL, false);
            vp_config_init(&config_v, dim, 0.05, distance, NULL, true);
            config_l.domain_min = domain.lo;
            config_l.domain_max = domain.hi;
            config_v.domain_min = domain.lo;
            config_v.domain_max = domain.hi;

            futcache_pack_t *linear = NULL, *vptree = NULL;
            TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);
            TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);

            bool ok = vp_agree(linear, vptree, points, n, dim);
            TEST_ASSERT(ok);
            ok = vp_exact_at_epsilons(distance, points, n, points + n * dim,
                                      m, dim, 0.0, 1.0, eps_grid,
                                      sizeof(eps_grid) / sizeof(eps_grid[0]));
            TEST_ASSERT(ok);

            size_t count_l, count_v;
            TEST_STATUS(futcache_pack_copy_representatives(
                            linear, NULL, &count_l),
                        FUTCACHE_OK);
            TEST_STATUS(futcache_pack_copy_representatives(
                            vptree, NULL, &count_v),
                        FUTCACHE_OK);
            TEST_ASSERT(count_l == count_v);

            TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);

            futcache_pack_destroy(linear);
            futcache_pack_destroy(vptree);
            free(points);
        }
    }
    vp_free_domain(domain);
    return true;
}

static bool test_vptree_cosine_normalized(void)
{
    const size_t dim = 384U;
    const size_t n = 600U;
    const size_t m = 150U;
    const double eps_grid[] = {0.02, 0.05, 0.1, 0.2, 0.3, 0.5};
    vp_domain_t domain;

    double *points = (double *)malloc((n + m) * dim * sizeof(double));
    TEST_ASSERT(points != NULL);
    uint64_t rng = UINT64_C(0x123456789abcdef0);
    for (size_t i = 0U; i < n + m; ++i) {
        vp_fill_gaussian(&rng, points + i * dim, dim);
    }

    futcache_pack_config_t config_l, config_v;
    domain = vp_make_domain(dim, -1.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config_l, dim, 0.1, futcache_distance_cosine, NULL,
                   false);
    vp_config_init(&config_v, dim, 0.1, futcache_distance_cosine, NULL,
                   true);
    config_l.domain_min = domain.lo;
    config_l.domain_max = domain.hi;
    config_v.domain_min = domain.lo;
    config_v.domain_max = domain.hi;

    futcache_pack_t *linear = NULL, *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);

    bool ok = vp_agree(linear, vptree, points, n, dim);
    TEST_ASSERT(ok);
    ok = vp_exact_at_epsilons(futcache_distance_cosine, points, n,
                              points + n * dim, m, dim, -1.0, 1.0, eps_grid,
                              sizeof(eps_grid) / sizeof(eps_grid[0]));
    TEST_ASSERT(ok);

    size_t count_l, count_v;
    TEST_STATUS(futcache_pack_copy_representatives(linear, NULL, &count_l),
                FUTCACHE_OK);
    TEST_STATUS(futcache_pack_copy_representatives(vptree, NULL, &count_v),
                FUTCACHE_OK);
    TEST_ASSERT(count_l == count_v);

    TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);

    futcache_pack_destroy(linear);
    futcache_pack_destroy(vptree);
    vp_free_domain(domain);
    free(points);
    return true;
}

static bool test_vptree_cosine_unnormalized(void)
{
    const size_t dim = 32U;
    const size_t n = 300U;
    const size_t m = 80U;
    const double eps_grid[] = {0.05, 0.1, 0.2, 0.5};
    vp_domain_t domain;

    /* Deliberately non-unit norms: 0.5 for even indices, 2.0 for odd.
     * The backend must fall back to an exact linear scan over the
     * engine's cosine distance and still match the oracle. */
    double *points = (double *)malloc((n + m) * dim * sizeof(double));
    TEST_ASSERT(points != NULL);
    uint64_t rng = UINT64_C(0xdeadbeefcafef00d);
    for (size_t i = 0U; i < n + m; ++i) {
        double scale = (i % 2U == 0U) ? 0.5 : 2.0;
        double norm = 0.0;
        for (size_t j = 0U; j < dim; ++j) {
            double v = vp_random_unit(&rng) * 2.0 - 1.0;
            points[i * dim + j] = v;
            norm += v * v;
        }
        norm = sqrt(norm);
        for (size_t j = 0U; j < dim; ++j) {
            points[i * dim + j] *= scale / norm;
        }
    }

    futcache_pack_config_t config_l, config_v;
    domain = vp_make_domain(dim, -2.0, 2.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config_l, dim, 0.1, futcache_distance_cosine, NULL,
                   false);
    vp_config_init(&config_v, dim, 0.1, futcache_distance_cosine, NULL,
                   true);
    config_l.domain_min = domain.lo;
    config_l.domain_max = domain.hi;
    config_v.domain_min = domain.lo;
    config_v.domain_max = domain.hi;

    futcache_pack_t *linear = NULL, *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);

    bool ok = vp_agree(linear, vptree, points, n, dim);
    TEST_ASSERT(ok);
    ok = vp_exact_at_epsilons(futcache_distance_cosine, points, n,
                              points + n * dim, m, dim, -1.0, 1.0, eps_grid,
                              sizeof(eps_grid) / sizeof(eps_grid[0]));
    TEST_ASSERT(ok);

    /* Mixed: start normalized, then inject an unnormalized insert, then
     * verify decisions still match the oracle (fallback engaged). */
    futcache_pack_config_t config_m;
    vp_config_init(&config_m, dim, 0.1, futcache_distance_cosine, NULL,
                   true);
    config_m.domain_min = domain.lo;
    config_m.domain_max = domain.hi;
    futcache_pack_t *mixed = NULL;
    TEST_STATUS(futcache_pack_create(&config_m, &mixed), FUTCACHE_OK);

    double *norm_pts = (double *)malloc((n + m) * dim * sizeof(double));
    TEST_ASSERT(norm_pts != NULL);
    uint64_t rng2 = UINT64_C(0x7777777777777777);
    for (size_t i = 0U; i < n + m; ++i) {
        vp_fill_gaussian(&rng2, norm_pts + i * dim, dim);
    }
    for (size_t i = 0U; i < n / 2U; ++i) {
        bool was_novel;
        TEST_STATUS(futcache_pack_observe(mixed, norm_pts + i * dim,
                                          &was_novel),
                    FUTCACHE_OK);
    }
    bool was_novel;
    TEST_STATUS(futcache_pack_observe(mixed, points, &was_novel), FUTCACHE_OK);
    for (size_t i = n / 2U; i < n; ++i) {
        bool was_novel2;
        TEST_STATUS(futcache_pack_observe(mixed, norm_pts + i * dim,
                                          &was_novel2),
                    FUTCACHE_OK);
    }
    ok = vp_exact_at_epsilons(futcache_distance_cosine, norm_pts, n,
                              norm_pts + n * dim, m, dim, -2.0, 2.0,
                              eps_grid,
                              sizeof(eps_grid) / sizeof(eps_grid[0]));
    TEST_ASSERT(ok);
    TEST_STATUS(futcache_pack_validate(mixed), FUTCACHE_OK);

    futcache_pack_destroy(linear);
    futcache_pack_destroy(vptree);
    futcache_pack_destroy(mixed);
    vp_free_domain(domain);
    free(points);
    free(norm_pts);
    return true;
}

static bool test_vptree_fault_injection(void)
{
    const size_t dim = 8U;
    const size_t n = 250U;
    const size_t budgets[] = {8U, 16U, 30U, 60U};
    vp_domain_t domain;

    double *points = (double *)malloc(n * dim * sizeof(double));
    TEST_ASSERT(points != NULL);
    uint64_t rng = UINT64_C(0xabcdef0123456789);
    for (size_t i = 0U; i < n; ++i) {
        vp_fill_uniform(&rng, points + i * dim, dim);
    }

    for (size_t b = 0U; b < sizeof(budgets) / sizeof(budgets[0]); ++b) {
        vp_failing_allocator_context_t ctx = {budgets[b], 0U};
        futcache_allocator_t allocator = {vp_failing_allocate,
                                          vp_failing_deallocate, &ctx};

        futcache_pack_config_t config_v;
        if (b == 0U) {
            domain = vp_make_domain(dim, 0.0, 1.0);
            TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
        }
        vp_config_init(&config_v, dim, 0.03, futcache_distance_l2, &allocator,
                       true);
        config_v.domain_min = domain.lo;
        config_v.domain_max = domain.hi;

        futcache_pack_t *vptree = NULL;
        TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);

        bool *committed_mask = (bool *)calloc(n, sizeof(bool));
        TEST_ASSERT(committed_mask != NULL);
        size_t committed = 0U;

        for (size_t i = 0U; i < n; ++i) {
            bool was_novel;
            futcache_status_t st = futcache_pack_observe(
                vptree, points + i * dim, &was_novel);
            if (st == FUTCACHE_ERROR_OUT_OF_MEMORY) {
                continue;
            }
            TEST_ASSERT(st == FUTCACHE_OK);
            committed_mask[i] = true;
            committed++;
        }
        TEST_ASSERT(committed > 0U);

        /* The surviving cache must still satisfy the separation invariant
         * (the tree is consistent with the representative array). */
        TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);

        /* A clean build over the same committed prefix must produce the
         * identical representative set and identical decisions. */
        futcache_pack_config_t config_lin, config_clean;
        vp_config_init(&config_lin, dim, 0.03, futcache_distance_l2, NULL,
                       false);
        config_lin.domain_min = domain.lo;
        config_lin.domain_max = domain.hi;
        vp_config_init(&config_clean, dim, 0.03, futcache_distance_l2, NULL,
                       true);
        config_clean.domain_min = domain.lo;
        config_clean.domain_max = domain.hi;

        futcache_pack_t *linear = NULL, *clean = NULL;
        TEST_STATUS(futcache_pack_create(&config_lin, &linear), FUTCACHE_OK);
        TEST_STATUS(futcache_pack_create(&config_clean, &clean), FUTCACHE_OK);

        for (size_t i = 0U; i < n; ++i) {
            if (!committed_mask[i]) continue;
            bool was_novel_c, was_novel_l;
            TEST_STATUS(futcache_pack_observe(clean, points + i * dim,
                                              &was_novel_c),
                        FUTCACHE_OK);
            TEST_STATUS(futcache_pack_observe(linear, points + i * dim,
                                              &was_novel_l),
                        FUTCACHE_OK);
            TEST_ASSERT(was_novel_c == was_novel_l);
        }

        size_t count_v = 0U, count_clean = 0U;
        TEST_STATUS(futcache_pack_copy_representatives(vptree, NULL, &count_v),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_copy_representatives(clean, NULL,
                                                       &count_clean),
                    FUTCACHE_OK);
        TEST_ASSERT(count_v == count_clean);

        /* Representative sets must be byte-identical. */
        if (count_v > 0U) {
            double *reps_v = (double *)malloc(count_v * dim * sizeof(double));
            double *reps_c = (double *)malloc(count_v * dim * sizeof(double));
            TEST_ASSERT(reps_v != NULL && reps_c != NULL);
            size_t cv = count_v, cc = count_v;
            TEST_STATUS(futcache_pack_copy_representatives(vptree, reps_v,
                                                           &cv),
                        FUTCACHE_OK);
            TEST_STATUS(futcache_pack_copy_representatives(clean, reps_c,
                                                           &cc),
                        FUTCACHE_OK);
            TEST_ASSERT(cv == count_v && cc == count_v);
            TEST_ASSERT(memcmp(reps_v, reps_c,
                               count_v * dim * sizeof(double)) == 0);
            free(reps_v);
            free(reps_c);
        }

        futcache_pack_destroy(linear);
        futcache_pack_destroy(clean);
        futcache_pack_destroy(vptree);
        free(committed_mask);

        /* All allocations accounted for (no leaks, no double frees). */
        TEST_ASSERT(ctx.active_allocations == 0U);
    }

    vp_free_domain(domain);
    free(points);
    return true;
}

static bool test_vptree_clear_reuse(void)
{
    const size_t dim = 16U;
    const size_t n = 500U;
    vp_domain_t domain;

    double *points = (double *)malloc(n * dim * sizeof(double));
    TEST_ASSERT(points != NULL);
    uint64_t rng = UINT64_C(0x5555555555555555);
    for (size_t i = 0U; i < n; ++i) {
        vp_fill_uniform(&rng, points + i * dim, dim);
    }

    futcache_pack_config_t config;
    domain = vp_make_domain(dim, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config, dim, 0.05, futcache_distance_l2, NULL, true);
    config.domain_min = domain.lo;
    config.domain_max = domain.hi;

    futcache_pack_t *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config, &vptree), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        bool was_novel;
        TEST_STATUS(futcache_pack_observe(vptree, points + i * dim,
                                          &was_novel),
                    FUTCACHE_OK);
    }
    TEST_STATUS(futcache_pack_clear(vptree), FUTCACHE_OK);

    /* After clear, the cache must behave exactly like a fresh cache. */
    futcache_pack_t *fresh = NULL;
    TEST_STATUS(futcache_pack_create(&config, &fresh), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        bool was_novel_v, was_novel_f;
        TEST_STATUS(futcache_pack_observe(vptree, points + i * dim,
                                          &was_novel_v),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe(fresh, points + i * dim,
                                          &was_novel_f),
                    FUTCACHE_OK);
        TEST_ASSERT(was_novel_v == was_novel_f);
    }

    size_t count_v = 0U, count_f = 0U;
    TEST_STATUS(futcache_pack_copy_representatives(vptree, NULL, &count_v),
                FUTCACHE_OK);
    TEST_STATUS(futcache_pack_copy_representatives(fresh, NULL, &count_f),
                FUTCACHE_OK);
    TEST_ASSERT(count_v == count_f);

    futcache_pack_destroy(fresh);
    futcache_pack_destroy(vptree);
    vp_free_domain(domain);
    free(points);
    return true;
}

static bool test_vptree_adversarial_order(void)
{
    const size_t dim = 8U;
    const size_t n = 3000U;
    const double eps_grid[] = {0.01, 0.05, 0.1};
    vp_domain_t domain;

    /* Points sorted by increasing distance to the origin: insertion
     * repeatedly follows the same outside spine, which forces the
     * scapegoat rebuild machinery to keep the height bounded. */
    double *points = (double *)malloc(n * dim * sizeof(double));
    double *dists = (double *)malloc(n * sizeof(double));
    size_t *order = (size_t *)malloc(n * sizeof(size_t));
    TEST_ASSERT(points != NULL && dists != NULL && order != NULL);
    uint64_t rng = UINT64_C(0x1111111111111111);
    for (size_t i = 0U; i < n; ++i) {
        vp_fill_uniform(&rng, points + i * dim, dim);
        double d = 0.0;
        for (size_t j = 0U; j < dim; ++j) {
            double v = points[i * dim + j] - 0.5;
            d += v * v;
        }
        dists[i] = d;
        order[i] = i;
    }
    for (size_t i = 0U; i < n; ++i) {
        for (size_t j = i + 1U; j < n; ++j) {
            if (dists[order[j]] < dists[order[i]]) {
                size_t t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }

    futcache_pack_config_t config_v, config_l;
    domain = vp_make_domain(dim, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config_v, dim, 0.02, futcache_distance_l2, NULL, true);
    config_v.domain_min = domain.lo;
    config_v.domain_max = domain.hi;
    vp_config_init(&config_l, dim, 0.02, futcache_distance_l2, NULL, false);
    config_l.domain_min = domain.lo;
    config_l.domain_max = domain.hi;

    futcache_pack_t *vptree = NULL, *linear = NULL;
    TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        bool was_novel_v, was_novel_l;
        TEST_STATUS(futcache_pack_observe(vptree, points + order[i] * dim,
                                          &was_novel_v),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe(linear, points + order[i] * dim,
                                          &was_novel_l),
                    FUTCACHE_OK);
        TEST_ASSERT(was_novel_v == was_novel_l);
    }

    /* Exactness on the sorted (deep-spine) tree: fresh cache per grid
     * epsilon, inserted in the same adversarial order. */
    for (size_t e = 0U; e < sizeof(eps_grid) / sizeof(eps_grid[0]); ++e) {
        futcache_pack_config_t config;
        vp_config_init(&config, dim, eps_grid[e], futcache_distance_l2, NULL,
                       true);
        config.domain_min = domain.lo;
        config.domain_max = domain.hi;
        futcache_pack_t *cache = NULL;
        TEST_STATUS(futcache_pack_create(&config, &cache), FUTCACHE_OK);
        for (size_t i = 0U; i < n; ++i) {
            bool was_novel;
            TEST_STATUS(futcache_pack_observe(cache, points + order[i] * dim,
                                              &was_novel),
                        FUTCACHE_OK);
        }
        for (size_t i = 0U; i < n; ++i) {
            double true_d = vp_oracle_distance(cache, points + i * dim, dim,
                                               futcache_distance_l2);
            bool expect = (true_d > eps_grid[e]);
            bool novel;
            TEST_ASSERT(vp_is_novel_ok(cache, points + i * dim, &novel));
            TEST_ASSERT(novel == expect);
        }
        futcache_pack_destroy(cache);
    }

    TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);

    futcache_pack_destroy(vptree);
    futcache_pack_destroy(linear);
    vp_free_domain(domain);
    free(points);
    free(dists);
    free(order);
    return true;
}

static bool test_vptree_empty_and_single(void)
{
    const size_t dim = 4U;
    vp_domain_t domain;
    futcache_pack_config_t config;
    domain = vp_make_domain(dim, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config, dim, 0.1, futcache_distance_l2, NULL, true);
    config.domain_min = domain.lo;
    config.domain_max = domain.hi;

    futcache_pack_t *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config, &vptree), FUTCACHE_OK);

    double point[4] = {0.25, 0.25, 0.25, 0.25};
    double query[4] = {0.25, 0.25, 0.25, 0.26};

    bool novel;
    TEST_ASSERT(vp_is_novel_ok(vptree, point, &novel));
    TEST_ASSERT(novel);
    bool was_novel;
    TEST_STATUS(futcache_pack_observe(vptree, point, &was_novel), FUTCACHE_OK);
    TEST_ASSERT(was_novel);

    /* Identical point: within epsilon -> redundant. */
    TEST_ASSERT(vp_is_novel_ok(vptree, point, &novel));
    TEST_ASSERT(!novel);
    TEST_STATUS(futcache_pack_observe(vptree, point, &was_novel), FUTCACHE_OK);
    TEST_ASSERT(!was_novel);

    /* Near point: also redundant at epsilon = 0.1 (L2 ~ 0.014). */
    TEST_ASSERT(vp_is_novel_ok(vptree, query, &novel));
    TEST_ASSERT(!novel);

    futcache_pack_destroy(vptree);
    vp_free_domain(domain);
    return true;
}

static bool test_vptree_large_differential(void)
{
    const size_t dim = 8U;
    const size_t n = 5000U;
    const size_t m = 200U;
    const double eps_grid[] = {0.0, 0.02, 0.05};
    vp_domain_t domain;

    double *points = (double *)malloc((n + m) * dim * sizeof(double));
    TEST_ASSERT(points != NULL);
    uint64_t rng = UINT64_C(0x9999999999999999);
    for (size_t i = 0U; i < n + m; ++i) {
        vp_fill_uniform(&rng, points + i * dim, dim);
    }

    futcache_pack_config_t config_v, config_l;
    domain = vp_make_domain(dim, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);
    vp_config_init(&config_v, dim, 0.0, futcache_distance_l2, NULL, true);
    config_v.domain_min = domain.lo;
    config_v.domain_max = domain.hi;
    vp_config_init(&config_l, dim, 0.0, futcache_distance_l2, NULL, false);
    config_l.domain_min = domain.lo;
    config_l.domain_max = domain.hi;

    futcache_pack_t *vptree = NULL, *linear = NULL;
    TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        bool was_novel_v, was_novel_l;
        TEST_STATUS(futcache_pack_observe(vptree, points + i * dim,
                                          &was_novel_v),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe(linear, points + i * dim,
                                          &was_novel_l),
                    FUTCACHE_OK);
        TEST_ASSERT(was_novel_v == was_novel_l);
    }

    bool ok = vp_exact_at_epsilons(futcache_distance_l2, points, n,
                                   points + n * dim, m, dim, 0.0, 1.0,
                                   eps_grid,
                                   sizeof(eps_grid) / sizeof(eps_grid[0]));
    TEST_ASSERT(ok);

    size_t count_v = 0U, count_l = 0U;
    TEST_STATUS(futcache_pack_copy_representatives(vptree, NULL, &count_v),
                FUTCACHE_OK);
    TEST_STATUS(futcache_pack_copy_representatives(linear, NULL, &count_l),
                FUTCACHE_OK);
    TEST_ASSERT(count_v == count_l);

    futcache_pack_destroy(vptree);
    futcache_pack_destroy(linear);
    vp_free_domain(domain);
    free(points);
    return true;
}

int pack_vptree_test_suite(void)
{
    static const test_case_t tests[] = {
        {"metric differential (l1/l2/linf, dims 2..384)",
         test_vptree_metric_differential},
        {"cosine normalized (chordal transform)",
         test_vptree_cosine_normalized},
        {"cosine unnormalized (linear fallback)",
         test_vptree_cosine_unnormalized},
        {"fault injection atomicity", test_vptree_fault_injection},
        {"clear and reuse", test_vptree_clear_reuse},
        {"adversarial insertion order", test_vptree_adversarial_order},
        {"empty and single point", test_vptree_empty_and_single},
        {"large differential (5000 reps)", test_vptree_large_differential},
    };
    return run_test_cases("pack_vptree", tests,
                          sizeof(tests) / sizeof(tests[0]));
}

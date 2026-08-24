/*
 * Adversarial stress tests for the packing cache added during the
 * v1.3.0 re-verification pass.
 *
 * The four tests attack the paths that changed most recently:
 *
 *  1. stress_adaptive_vptree_differential: variable-radius observe streams;
 *     the VP-tree backend must reproduce the linear scan exactly for every
 *     novelty decision, final representative set, radius set, lookup
 *     (found/distance/index), and nearest (distance/index) on a query grid.
 *     Covers L1/L2/L_inf/cosine/poincare, zero and huge radii, and mixed
 *     fixed/adaptive usage.
 *
 *     The documented cosine transform is exact only up to the backend's
 *     VP_NORM_TOLERANCE (1e-6): the chordal-distance transform differs from
 *     the raw 1 - dot distance by at most ~1e-6, so cosine comparisons use
 *     that tolerance and any decision disagreement must lie inside the
 *     tolerance band of the union boundary.  Non-cosine metrics compare
 *     bit-exactly.
 *
 *  2. stress_ceiling_eviction_correctness: a hard byte ceiling with
 *     variable radii. The ceiling is shared between representatives and
 *     backend index metadata, so the two backends may legitimately retain
 *     different counts at the same ceiling; each cache must satisfy
 *     validate(), the telemetry identity count + evictions == novel
 *     observations, its own live/peak memory ceiling, and every query must
 *     match an independent linear-scan oracle over that cache's own
 *     surviving representatives.
 *
 *  3. stress_serialize_adaptive_roundtrip: serialize a variable-radius
 *     VP-tree cache (with ceiling and evictions) and restore it; the
 *     restored cache must hold the identical representatives, radii, and
 *     telemetry, and must agree with the original on a continuation stream.
 *
 *  4. stress_concurrent_adaptive_vptree: concurrent writers and readers on
 *     a variable-radius VP-tree cache. The serialize size-query/copy pair
 *     may legitimately race with writers (documented), so readers retry on
 *     FUTCACHE_ERROR_BUFFER_TOO_SMALL. After joining, the cache must
 *     validate and satisfy the telemetry identities; nothing may crash,
 *     deadlock, or corrupt (checked under ASan/UBSan).
 */

#include "test.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

/* Backend cosine-transform tolerance (mirrors VP_NORM_TOLERANCE in
 * src/pack_vptree.c). */
#define STRESS_COSINE_TOLERANCE 1e-6

/* ============================================================
 * Deterministic RNG (xorshift64, same family as the other suites)
 * ============================================================ */

static uint64_t stress_random_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static double stress_random_unit(uint64_t *state)
{
    return (double)(stress_random_next(state) >> 11U) / 9007199254740992.0;
}

/* ============================================================
 * Domain helper (cache does not retain the arrays)
 * ============================================================ */

typedef struct stress_domain {
    double *lo;
    double *hi;
} stress_domain_t;

static stress_domain_t stress_make_domain(size_t dimension,
                                          double lo, double hi)
{
    stress_domain_t d = {NULL, NULL};
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

static void stress_free_domain(stress_domain_t d)
{
    free(d.lo);
    free(d.hi);
}

/* ============================================================
 * Independent oracle scans over copied representatives
 * ============================================================ */

typedef struct stress_reps {
    double *points;
    double *radii;
    size_t count;
    size_t dimension;
} stress_reps_t;

static bool stress_copy_reps(const futcache_pack_t *cache,
                             stress_reps_t *out)
{
    size_t count = 0U;
    if (futcache_pack_copy_representatives(cache, NULL, &count) !=
        FUTCACHE_OK) {
        return false;
    }
    out->count = count;
    if (count == 0U) {
        out->points = NULL;
        out->radii = NULL;
        return true;
    }
    size_t dimension = out->dimension;
    out->points = (double *)malloc(count * dimension * sizeof(double));
    out->radii = (double *)malloc(count * sizeof(double));
    if (out->points == NULL || out->radii == NULL) {
        free(out->points);
        free(out->radii);
        out->points = NULL;
        out->radii = NULL;
        return false;
    }
    size_t n = count;
    if (futcache_pack_copy_representatives(cache, out->points, &n) !=
            FUTCACHE_OK ||
        n != count) {
        free(out->points);
        free(out->radii);
        out->points = NULL;
        out->radii = NULL;
        return false;
    }
    n = count;
    if (futcache_pack_copy_radii(cache, out->radii, &n) != FUTCACHE_OK ||
        n != count) {
        free(out->points);
        free(out->radii);
        out->points = NULL;
        out->radii = NULL;
        return false;
    }
    return true;
}

static void stress_free_reps(stress_reps_t *reps)
{
    free(reps->points);
    free(reps->radii);
    reps->points = NULL;
    reps->radii = NULL;
    reps->count = 0U;
}

/* Oracle: closest containing representative, lowest index on ties. */
static bool stress_oracle_lookup(const stress_reps_t *reps,
                                 const double *query,
                                 futcache_distance_fn distance,
                                 bool *out_found, double *out_distance,
                                 size_t *out_index)
{
    bool found = false;
    double best_distance = INFINITY;
    size_t best_index = SIZE_MAX;
    for (size_t i = 0U; i < reps->count; ++i) {
        double d = distance(query, reps->points + i * reps->dimension,
                            reps->dimension, NULL);
        if (d <= reps->radii[i] &&
            (!found || d < best_distance ||
             (d == best_distance && i < best_index))) {
            found = true;
            best_distance = d;
            best_index = i;
        }
    }
    *out_found = found;
    *out_distance = best_distance;
    *out_index = best_index;
    return true;
}

/* Oracle: nearest representative, lowest index on ties. */
static bool stress_oracle_nearest(const stress_reps_t *reps,
                                  const double *query,
                                  futcache_distance_fn distance,
                                  double *out_distance, size_t *out_index)
{
    double min_d = INFINITY;
    size_t min_i = SIZE_MAX;
    for (size_t i = 0U; i < reps->count; ++i) {
        double d = distance(query, reps->points + i * reps->dimension,
                            reps->dimension, NULL);
        if (d < min_d) {
            min_d = d;
            min_i = i;
        }
    }
    *out_distance = min_d;
    *out_index = min_i;
    return true;
}

/* Signed margin to the union boundary: min_i (d_i - r_i). Positive when the
 * query is strictly outside every ball, negative when covered. */
static double stress_union_margin(const stress_reps_t *reps,
                                  const double *query,
                                  futcache_distance_fn distance)
{
    double margin = INFINITY;
    for (size_t i = 0U; i < reps->count; ++i) {
        double d = distance(query, reps->points + i * reps->dimension,
                            reps->dimension, NULL);
        double m = d - reps->radii[i];
        if (m < margin) margin = m;
    }
    return margin;
}

/* ============================================================
 * Test 1: adaptive-radius VP-tree vs linear differential
 * ============================================================ */

static bool stress_compare_caches(const futcache_pack_t *linear,
                                  const futcache_pack_t *vptree,
                                  const double *queries, size_t query_count,
                                  size_t dimension,
                                  futcache_distance_fn distance,
                                  double tolerance)
{
    stress_reps_t reps_l = {NULL, NULL, 0U, dimension};
    stress_reps_t reps_v = {NULL, NULL, 0U, dimension};

    TEST_ASSERT(stress_copy_reps(linear, &reps_l));
    TEST_ASSERT(stress_copy_reps(vptree, &reps_v));
    TEST_ASSERT(reps_l.count == reps_v.count);
    if (reps_l.count > 0U) {
        TEST_ASSERT(memcmp(reps_l.points, reps_v.points,
                           reps_l.count * dimension * sizeof(double)) == 0);
        TEST_ASSERT(memcmp(reps_l.radii, reps_v.radii,
                           reps_l.count * sizeof(double)) == 0);
    }

    for (size_t i = 0U; i < query_count; ++i) {
        const double *q = queries + i * dimension;
        bool found_l, found_v;
        double dist_l, dist_v;
        size_t idx_l, idx_v;
        TEST_STATUS(futcache_pack_lookup(linear, q, &found_l, &dist_l,
                                         &idx_l),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_lookup(vptree, q, &found_v, &dist_v,
                                         &idx_v),
                    FUTCACHE_OK);
        if (found_l != found_v) {
            /* Allowed only inside the documented transform tolerance band. */
            TEST_ASSERT(tolerance > 0.0);
            TEST_ASSERT(fabs(stress_union_margin(&reps_l, q, distance)) <=
                        tolerance);
        }
        if (found_l) {
            TEST_ASSERT(fabs(dist_l - dist_v) <= tolerance);
            TEST_ASSERT(idx_l == idx_v);
        }

        bool novel_l, novel_v;
        TEST_STATUS(futcache_pack_is_novel(linear, q, &novel_l), FUTCACHE_OK);
        TEST_STATUS(futcache_pack_is_novel(vptree, q, &novel_v), FUTCACHE_OK);
        TEST_ASSERT(novel_l == novel_v || found_l != found_v);
        TEST_ASSERT(novel_l == !found_l);

        double nn_l, nn_v;
        size_t ni_l, ni_v;
        TEST_STATUS(futcache_pack_nearest(linear, q, &nn_l, &ni_l),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_nearest(vptree, q, &nn_v, &ni_v),
                    FUTCACHE_OK);
        TEST_ASSERT(fabs(nn_l - nn_v) <= tolerance);
        TEST_ASSERT(ni_l == ni_v);

        /* Both must match the independent oracle. */
        bool o_found;
        double o_dist;
        size_t o_idx;
        stress_oracle_lookup(&reps_l, q, distance, &o_found, &o_dist,
                             &o_idx);
        TEST_ASSERT(o_found == found_l);
        if (o_found) {
            TEST_ASSERT(o_dist == dist_l);
            TEST_ASSERT(o_idx == idx_l);
        }
        double o_nn;
        size_t o_ni;
        stress_oracle_nearest(&reps_l, q, distance, &o_nn, &o_ni);
        TEST_ASSERT(o_nn == nn_l);
        TEST_ASSERT(o_ni == ni_l);
    }

    stress_free_reps(&reps_l);
    stress_free_reps(&reps_v);
    return true;
}

static bool stress_adaptive_differential_one(
    size_t dimension, futcache_distance_fn distance,
    uint64_t seed, double lo, double hi, bool normalize_points,
    bool poincare_mode, double max_radius)
{
    const size_t n = 500U;
    const size_t m = 250U;
    double tolerance = distance == futcache_distance_cosine
        ? STRESS_COSINE_TOLERANCE : 0.0;
    uint64_t rng = seed;
    stress_domain_t domain = stress_make_domain(dimension, lo, hi);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);

    double *points = (double *)malloc((n + m) * dimension * sizeof(double));
    TEST_ASSERT(points != NULL);
    for (size_t i = 0U; i < n + m; ++i) {
        double norm = 0.0;
        for (size_t d = 0U; d < dimension; ++d) {
            double v = lo + (hi - lo) * stress_random_unit(&rng);
            if (poincare_mode) v *= 0.7;  /* keep norm safely < 1 */
            points[i * dimension + d] = v;
            norm += v * v;
        }
        if (normalize_points) {
            norm = sqrt(norm);
            if (norm > 0.0) {
                for (size_t d = 0U; d < dimension; ++d) {
                    points[i * dimension + d] /= norm;
                }
            }
        }
    }

    futcache_pack_config_t config_l, config_v;
    futcache_pack_config_init(&config_l);
    futcache_pack_config_init(&config_v);
    config_l.dimension = config_v.dimension = dimension;
    config_l.epsilon = config_v.epsilon = 0.05;
    config_l.distance = config_v.distance = distance;
    config_l.domain_min = config_v.domain_min = domain.lo;
    config_l.domain_max = config_v.domain_max = domain.hi;
    config_v.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *linear = NULL, *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config_l, &linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_v, &vptree), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        const double *p = points + i * dimension;
        /* Radius pattern: mostly random, with zero, tiny, and huge values
         * mixed in to exercise degenerate variable-radius states. */
        double radius;
        uint64_t pick = stress_random_next(&rng) % 16U;
        if (pick == 0U) {
            radius = 0.0;
        } else if (pick == 1U) {
            radius = max_radius;   /* covers everything */
        } else if (pick == 2U) {
            radius = 1e-9;         /* near-identity */
        } else {
            radius = stress_random_unit(&rng) * max_radius;
        }
        bool novel_l, novel_v;
        TEST_STATUS(futcache_pack_observe_with_radius(
                        linear, p, radius, &novel_l, NULL, NULL),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe_with_radius(
                        vptree, p, radius, &novel_v, NULL, NULL),
                    FUTCACHE_OK);
        if (novel_l != novel_v) {
            /* Cosine only: the disagreement must sit inside the transform
             * tolerance band of the union boundary. */
            TEST_ASSERT(tolerance > 0.0);
            stress_reps_t reps;
            TEST_ASSERT(stress_copy_reps(linear, &reps));
            TEST_ASSERT(fabs(stress_union_margin(&reps, p, distance)) <=
                        tolerance);
            stress_free_reps(&reps);
        }
        if (i % 7U == 0U) {
            /* Periodically verify intermediate exactness. */
            TEST_ASSERT(stress_compare_caches(linear, vptree,
                                              points + (n + i) % m * dimension,
                                              1U, dimension, distance,
                                              tolerance));
        }
    }

    TEST_STATUS(futcache_pack_validate(linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);
    TEST_ASSERT(stress_compare_caches(linear, vptree, points + n * dimension,
                                      m, dimension, distance, tolerance));

    futcache_pack_destroy(linear);
    futcache_pack_destroy(vptree);
    stress_free_domain(domain);
    free(points);
    return true;
}

static bool test_stress_adaptive_vptree_differential(void)
{
    static const size_t dims[] = {1U, 3U, 16U, 64U};
    static const futcache_distance_fn metrics[] = {
        futcache_distance_linf, futcache_distance_l1,
        futcache_distance_l2};
    uint64_t seed = UINT64_C(0xfeedfacecafebeef);

    for (size_t di = 0U; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        for (size_t mi = 0U; mi < sizeof(metrics) / sizeof(metrics[0]);
             ++mi) {
            TEST_ASSERT(stress_adaptive_differential_one(
                dims[di], metrics[mi], seed ^ (uint64_t)di
                    ^ ((uint64_t)mi << 32U),
                0.0, 1.0, false, false, 0.4));
        }
    }
    /* Cosine: normalized points on [-1, 1]^d. */
    for (size_t di = 0U; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        TEST_ASSERT(stress_adaptive_differential_one(
            dims[di], futcache_distance_cosine,
            seed ^ UINT64_C(0x51a1e) ^ (uint64_t)di, -1.0, 1.0, true, false,
            0.6));
    }
    /* Poincare: norm < 1 enforced by the generator. */
    TEST_ASSERT(stress_adaptive_differential_one(
        2U, futcache_distance_poincare,
        seed ^ UINT64_C(0x70a1ca), -1.0, 1.0, false, true, 1.5));
    return true;
}

/* ============================================================
 * Test 2: byte-ceiling eviction correctness with variable radii
 * ============================================================ */

static bool stress_oracle_check(const futcache_pack_t *cache,
                                const double *queries, size_t query_count,
                                size_t dimension,
                                futcache_distance_fn distance,
                                stress_reps_t *reps)
{
    for (size_t i = 0U; i < query_count; ++i) {
        const double *q = queries + i * dimension;
        bool found;
        double dist;
        size_t idx;
        TEST_STATUS(futcache_pack_lookup(cache, q, &found, &dist, &idx),
                    FUTCACHE_OK);
        bool o_found;
        double o_dist;
        size_t o_idx;
        stress_oracle_lookup(reps, q, distance, &o_found, &o_dist, &o_idx);
        TEST_ASSERT(found == o_found);
        if (found) {
            TEST_ASSERT(dist == o_dist);
            TEST_ASSERT(idx == o_idx);
        }
        double nn;
        size_t ni;
        TEST_STATUS(futcache_pack_nearest(cache, q, &nn, &ni), FUTCACHE_OK);
        double o_nn;
        size_t o_ni;
        stress_oracle_nearest(reps, q, distance, &o_nn, &o_ni);
        TEST_ASSERT(nn == o_nn);
        TEST_ASSERT(ni == o_ni);
    }
    return true;
}

static bool test_stress_ceiling_eviction_correctness(void)
{
    const size_t dimension = 8U;
    const size_t n = 2000U;
    const size_t m = 300U;
    uint64_t rng = UINT64_C(0xdeadbeef12345678);
    stress_domain_t domain = stress_make_domain(dimension, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);

    double *points = (double *)malloc((n + m) * dimension * sizeof(double));
    TEST_ASSERT(points != NULL);
    for (size_t i = 0U; i < n + m; ++i) {
        for (size_t d = 0U; d < dimension; ++d) {
            points[i * dimension + d] = stress_random_unit(&rng);
        }
    }

    /* Measure the empty-cache footprint and one representative's bytes. */
    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = dimension;
    config.epsilon = 0.0;
    config.domain_min = domain.lo;
    config.domain_max = domain.hi;

    futcache_pack_t *probe = NULL;
    TEST_STATUS(futcache_pack_create(&config, &probe), FUTCACHE_OK);
    futcache_pack_stats_t stats;
    TEST_STATUS(futcache_pack_get_stats(probe, &stats), FUTCACHE_OK);
    size_t base_bytes = stats.memory_bytes;
    bool unused;
    TEST_STATUS(futcache_pack_observe(probe, points, &unused), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_get_stats(probe, &stats), FUTCACHE_OK);
    size_t rep_bytes = stats.memory_bytes - base_bytes;
    futcache_pack_destroy(probe);
    TEST_ASSERT(rep_bytes > 0U);

    /* Limit: base + 12 representatives. */
    size_t limit = base_bytes + 12U * rep_bytes;

    futcache_pack_config_t config_lin, config_vp;
    futcache_pack_config_init(&config_lin);
    futcache_pack_config_init(&config_vp);
    config_lin.dimension = config_vp.dimension = dimension;
    config_lin.epsilon = config_vp.epsilon = 0.0;
    config_lin.distance = config_vp.distance = futcache_distance_l2;
    config_lin.domain_min = config_vp.domain_min = domain.lo;
    config_lin.domain_max = config_vp.domain_max = domain.hi;
    config_lin.max_memory_bytes = config_vp.max_memory_bytes = limit;
    config_vp.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *linear = NULL, *vptree = NULL;
    TEST_STATUS(futcache_pack_create(&config_lin, &linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_create(&config_vp, &vptree), FUTCACHE_OK);

    for (size_t i = 0U; i < n; ++i) {
        double radius = stress_random_unit(&rng) * 0.3;
        bool novel_l, novel_v;
        TEST_STATUS(futcache_pack_observe_with_radius(
                        linear, points + i * dimension, radius, &novel_l,
                        NULL, NULL),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe_with_radius(
                        vptree, points + i * dimension, radius, &novel_v,
                        NULL, NULL),
                    FUTCACHE_OK);
        TEST_ASSERT(novel_l == novel_v);
    }

    TEST_STATUS(futcache_pack_validate(linear), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_validate(vptree), FUTCACHE_OK);

    futcache_pack_stats_t stats_l, stats_v;
    TEST_STATUS(futcache_pack_get_stats(linear, &stats_l), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_get_stats(vptree, &stats_v), FUTCACHE_OK);
    TEST_ASSERT(stats_l.novel_observations == stats_v.novel_observations);
    /* The ceiling is shared with backend index metadata: the vptree cache
     * may legitimately retain fewer representatives (and therefore evict
     * more often) than the linear cache, but each must satisfy its own
     * identity and ceiling invariants. */
    TEST_ASSERT(stats_l.representative_count <= 12U);
    TEST_ASSERT(stats_v.representative_count <= 12U);
    TEST_ASSERT(stats_l.representative_count + stats_l.evictions ==
                stats_l.novel_observations);
    TEST_ASSERT(stats_v.representative_count + stats_v.evictions ==
                stats_v.novel_observations);
    TEST_ASSERT(stats_l.memory_bytes <= limit);
    TEST_ASSERT(stats_v.memory_bytes <= limit);
    TEST_ASSERT(stats_l.peak_memory_bytes <= limit);
    TEST_ASSERT(stats_v.peak_memory_bytes <= limit);

    /* Every query must match the independent oracle over that cache's own
     * surviving representatives. */
    stress_reps_t reps_l = {NULL, NULL, 0U, dimension};
    stress_reps_t reps_v = {NULL, NULL, 0U, dimension};
    TEST_ASSERT(stress_copy_reps(linear, &reps_l));
    TEST_ASSERT(stress_copy_reps(vptree, &reps_v));
    TEST_ASSERT(stress_oracle_check(linear, points + n * dimension, m,
                                    dimension, futcache_distance_l2,
                                    &reps_l));
    TEST_ASSERT(stress_oracle_check(vptree, points + n * dimension, m,
                                    dimension, futcache_distance_l2,
                                    &reps_v));
    stress_free_reps(&reps_l);
    stress_free_reps(&reps_v);

    futcache_pack_destroy(linear);
    futcache_pack_destroy(vptree);
    stress_free_domain(domain);
    free(points);
    return true;
}

/* ============================================================
 * Test 3: adaptive-radius serialize roundtrip (vptree + ceiling)
 * ============================================================ */

static bool test_stress_serialize_adaptive_roundtrip(void)
{
    const size_t dimension = 6U;
    const size_t n = 900U;
    const size_t m = 200U;
    uint64_t rng = UINT64_C(0x5eed5eed5eed5eed);
    stress_domain_t domain = stress_make_domain(dimension, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);

    double *points = (double *)malloc((n + m) * dimension * sizeof(double));
    TEST_ASSERT(points != NULL);
    for (size_t i = 0U; i < n + m; ++i) {
        for (size_t d = 0U; d < dimension; ++d) {
            points[i * dimension + d] = stress_random_unit(&rng);
        }
    }

    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = dimension;
    config.epsilon = 0.05;
    config.distance = futcache_distance_l2;
    config.domain_min = domain.lo;
    config.domain_max = domain.hi;
    config.backend = &futcache_pack_vptree_backend;
    config.max_memory_bytes = 64U * 1024U;

    futcache_pack_t *cache = NULL;
    TEST_STATUS(futcache_pack_create(&config, &cache), FUTCACHE_OK);
    for (size_t i = 0U; i < n; ++i) {
        double radius = stress_random_unit(&rng) * 0.25;
        bool was_novel;
        TEST_STATUS(futcache_pack_observe_with_radius(
                        cache, points + i * dimension, radius, &was_novel,
                        NULL, NULL),
                    FUTCACHE_OK);
    }
    TEST_STATUS(futcache_pack_validate(cache), FUTCACHE_OK);

    size_t serialized_size = 0U;
    TEST_STATUS(futcache_pack_serialize(cache, NULL, 0U, &serialized_size),
                FUTCACHE_OK);
    TEST_ASSERT(serialized_size > 0U);
    uint8_t *blob = (uint8_t *)malloc(serialized_size);
    TEST_ASSERT(blob != NULL);
    TEST_STATUS(futcache_pack_serialize(cache, blob, serialized_size,
                                        &serialized_size),
                FUTCACHE_OK);

    futcache_pack_t *restored = NULL;
    TEST_STATUS(futcache_pack_deserialize(blob, serialized_size, NULL,
                                          &restored),
                FUTCACHE_OK);

    /* Identical representative state and telemetry. */
    stress_reps_t reps_a = {NULL, NULL, 0U, dimension};
    stress_reps_t reps_b = {NULL, NULL, 0U, dimension};
    TEST_ASSERT(stress_copy_reps(cache, &reps_a));
    TEST_ASSERT(stress_copy_reps(restored, &reps_b));
    TEST_ASSERT(reps_a.count == reps_b.count);
    TEST_ASSERT(memcmp(reps_a.points, reps_b.points,
                       reps_a.count * dimension * sizeof(double)) == 0);
    TEST_ASSERT(memcmp(reps_a.radii, reps_b.radii,
                       reps_a.count * sizeof(double)) == 0);

    futcache_pack_stats_t stats_a, stats_b;
    TEST_STATUS(futcache_pack_get_stats(cache, &stats_a), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_get_stats(restored, &stats_b), FUTCACHE_OK);
    TEST_ASSERT(stats_a.observations == stats_b.observations);
    TEST_ASSERT(stats_a.novel_observations == stats_b.novel_observations);
    TEST_ASSERT(stats_a.generation == stats_b.generation);
    TEST_ASSERT(stats_a.evictions == stats_b.evictions);
    TEST_ASSERT(stats_a.peak_count == stats_b.peak_count);
    TEST_ASSERT(stats_a.representative_count == stats_b.representative_count);
    TEST_ASSERT(stats_a.memory_limit_bytes == stats_b.memory_limit_bytes);

    /* Identical decisions on the continuation stream. */
    for (size_t i = 0U; i < m; ++i) {
        const double *q = points + (n + i) * dimension;
        bool found_a, found_b;
        double dist_a, dist_b;
        size_t idx_a, idx_b;
        TEST_STATUS(futcache_pack_lookup(cache, q, &found_a, &dist_a, &idx_a),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_lookup(restored, q, &found_b, &dist_b,
                                         &idx_b),
                    FUTCACHE_OK);
        TEST_ASSERT(found_a == found_b);
        if (found_a) {
            TEST_ASSERT(dist_a == dist_b);
            TEST_ASSERT(idx_a == idx_b);
        }
        double radius = stress_random_unit(&rng) * 0.2;
        bool novel_a, novel_b;
        TEST_STATUS(futcache_pack_observe_with_radius(
                        cache, q, radius, &novel_a, NULL, NULL),
                    FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe_with_radius(
                        restored, q, radius, &novel_b, NULL, NULL),
                    FUTCACHE_OK);
        TEST_ASSERT(novel_a == novel_b);
    }
    TEST_STATUS(futcache_pack_validate(restored), FUTCACHE_OK);

    /* Snapshot corruption is still rejected. */
    blob[serialized_size / 2U] ^= 0x5aU;
    futcache_pack_t *bad = NULL;
    TEST_STATUS(futcache_pack_deserialize(blob, serialized_size, NULL, &bad),
                FUTCACHE_ERROR_CORRUPT_DATA);
    TEST_ASSERT(bad == NULL);

    stress_free_reps(&reps_a);
    stress_free_reps(&reps_b);
    futcache_pack_destroy(cache);
    futcache_pack_destroy(restored);
    stress_free_domain(domain);
    free(blob);
    free(points);
    return true;
}

/* ============================================================
 * Test 4: concurrent adaptive-radius VP-tree cache
 * ============================================================ */

#define STRESS_WRITERS 4U
#define STRESS_READERS 2U
#define STRESS_OPS 400U

typedef struct stress_concurrent_ctx {
    futcache_pack_t *cache;
    uint64_t seed;
    size_t dimension;
} stress_concurrent_ctx_t;

static void *stress_writer_main(void *opaque)
{
    stress_concurrent_ctx_t *ctx = (stress_concurrent_ctx_t *)opaque;
    uint64_t rng = ctx->seed;
    double *point = (double *)malloc(ctx->dimension * sizeof(double));
    if (point == NULL) return (void *)(uintptr_t)1U;
    for (size_t i = 0U; i < STRESS_OPS; ++i) {
        double norm = 0.0;
        for (size_t d = 0U; d < ctx->dimension; ++d) {
            double v = stress_random_unit(&rng);
            point[d] = v;
            norm += v * v;
        }
        norm = sqrt(norm);
        if (norm > 0.0) {
            for (size_t d = 0U; d < ctx->dimension; ++d) {
                point[d] /= norm;
            }
        }
        double radius = stress_random_unit(&rng) * 0.5;
        bool was_novel;
        double distance;
        size_t index;
        futcache_status_t st = futcache_pack_observe_with_radius(
            ctx->cache, point, radius, &was_novel, &distance, &index);
        if (st != FUTCACHE_OK) {
            free(point);
            return (void *)(uintptr_t)1U;
        }
    }
    free(point);
    return NULL;
}

static void *stress_reader_main(void *opaque)
{
    stress_concurrent_ctx_t *ctx = (stress_concurrent_ctx_t *)opaque;
    uint64_t rng = ctx->seed;
    double *point = (double *)malloc(ctx->dimension * sizeof(double));
    if (point == NULL) return (void *)(uintptr_t)6U;
    uint8_t *blob = NULL;
    for (size_t i = 0U; i < STRESS_OPS; ++i) {
        for (size_t d = 0U; d < ctx->dimension; ++d) {
            point[d] = stress_random_unit(&rng);
        }
        bool found;
        double distance;
        size_t index;
        if (futcache_pack_lookup(ctx->cache, point, &found, &distance,
                                 &index) != FUTCACHE_OK) {
            free(point);
            free(blob);
            return (void *)(uintptr_t)3U;
        }
        double nn;
        size_t ni;
        if (futcache_pack_nearest(ctx->cache, point, &nn, &ni) !=
            FUTCACHE_OK) {
            free(point);
            free(blob);
            return (void *)(uintptr_t)4U;
        }
        if (i % 8U == 0U) {
            /* The size-query/copy pair may race with writers (documented);
             * retry until the buffer matches the current state. */
            for (;;) {
                size_t required = 0U;
                if (futcache_pack_serialize(ctx->cache, NULL, 0U,
                                            &required) != FUTCACHE_OK) {
                    free(point);
                    free(blob);
                    return (void *)(uintptr_t)5U;
                }
                uint8_t *fresh = (uint8_t *)realloc(blob, required);
                if (fresh == NULL && required != 0U) {
                    free(point);
                    free(blob);
                    return (void *)(uintptr_t)6U;
                }
                blob = fresh;
                size_t wrote = required;
                futcache_status_t st = futcache_pack_serialize(
                    ctx->cache, blob, wrote, &wrote);
                if (st == FUTCACHE_OK) break;
                if (st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
                    free(point);
                    free(blob);
                    return (void *)(uintptr_t)7U;
                }
            }
        }
    }
    free(point);
    free(blob);
    return NULL;
}

static bool test_stress_concurrent_adaptive_vptree(void)
{
    const size_t dimension = 12U;
    stress_domain_t domain = stress_make_domain(dimension, 0.0, 1.0);
    TEST_ASSERT(domain.lo != NULL && domain.hi != NULL);

    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = dimension;
    config.epsilon = 0.05;
    config.distance = futcache_distance_cosine;
    config.domain_min = domain.lo;
    config.domain_max = domain.hi;
    config.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *cache = NULL;
    TEST_STATUS(futcache_pack_create(&config, &cache), FUTCACHE_OK);

    pthread_t writers[STRESS_WRITERS];
    pthread_t readers[STRESS_READERS];
    stress_concurrent_ctx_t wctx[STRESS_WRITERS];
    stress_concurrent_ctx_t rctx[STRESS_READERS];
    for (size_t i = 0U; i < STRESS_WRITERS; ++i) {
        wctx[i].cache = cache;
        wctx[i].seed = UINT64_C(0x1111111111111111) ^ (uint64_t)(i + 1U);
        wctx[i].dimension = dimension;
        TEST_ASSERT(pthread_create(&writers[i], NULL, stress_writer_main,
                                   &wctx[i]) == 0);
    }
    for (size_t i = 0U; i < STRESS_READERS; ++i) {
        rctx[i].cache = cache;
        rctx[i].seed = UINT64_C(0x2222222222222222) ^ (uint64_t)(i + 1U);
        rctx[i].dimension = dimension;
        TEST_ASSERT(pthread_create(&readers[i], NULL, stress_reader_main,
                                   &rctx[i]) == 0);
    }
    int failures = 0;
    void *result = NULL;
    for (size_t i = 0U; i < STRESS_WRITERS; ++i) {
        TEST_ASSERT(pthread_join(writers[i], &result) == 0);
        if (result != NULL) failures++;
    }
    for (size_t i = 0U; i < STRESS_READERS; ++i) {
        TEST_ASSERT(pthread_join(readers[i], &result) == 0);
        if (result != NULL) failures++;
    }
    TEST_ASSERT(failures == 0);

    TEST_STATUS(futcache_pack_validate(cache), FUTCACHE_OK);
    futcache_pack_stats_t stats;
    TEST_STATUS(futcache_pack_get_stats(cache, &stats), FUTCACHE_OK);
    if (stats.novel_observations != UINT64_MAX &&
        stats.evictions != UINT64_MAX) {
        TEST_ASSERT(stats.representative_count + stats.evictions ==
                    stats.novel_observations);
    }
    TEST_ASSERT(stats.representative_count <= stats.peak_count);
    TEST_ASSERT(stats.generation >= stats.observations);

    futcache_pack_destroy(cache);
    stress_free_domain(domain);
    return true;
}

/* ============================================================
 * Suite registration
 * ============================================================ */

int pack_stress_test_suite(void)
{
    static const test_case_t tests[] = {
        {"stress adaptive vptree differential",
         test_stress_adaptive_vptree_differential},
        {"stress ceiling eviction correctness",
         test_stress_ceiling_eviction_correctness},
        {"stress serialize adaptive roundtrip",
         test_stress_serialize_adaptive_roundtrip},
        {"stress concurrent adaptive vptree",
         test_stress_concurrent_adaptive_vptree},
    };
    return run_test_cases("pack_stress", tests,
                          sizeof(tests) / sizeof(tests[0]));
}

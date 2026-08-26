#include "test.h"
#include "futcache/persist_nd.h"
#include "futcache/persist.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * d-D Persistent Packing tests (Design Sketch 01, Phase 3)
 * ============================================================ */

/* Helper: L2 distance for tests. */
static double test_l2(const double *a, const double *b, size_t dim,
                       void *ctx)
{
    (void)ctx;
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

/* Helper: Linf distance for tests. */
static double test_linf(const double *a, const double *b, size_t dim,
                         void *ctx)
{
    (void)ctx;
    double max_d = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

/* Helper: create a 2-D engine with Linf. */
static futcache_persist_nd_t *make_2d_engine(double eps)
{
    double lo[2] = {-1.0, -1.0};
    double hi[2] = { 1.0,  1.0};
    futcache_persist_nd_t *eng = NULL;
    futcache_status_t st = futcache_persist_nd_create(
        2, eps, test_linf, NULL, lo, hi, 0, NULL, &eng);
    TEST_ASSERT(st == FUTCACHE_OK);
    return eng;
}

/* --- 1. Empty engine --- */

static bool test_nd_empty(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    TEST_STATUS(futcache_persist_nd_is_novel_at(eng, (const double[]){0.0, 0.0},
                0.05, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    /* No reps */
    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 0);

    /* Stats */
    futcache_persist_nd_stats_t stats;
    TEST_STATUS(futcache_persist_nd_get_stats(eng, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.rep_count == 0);
    TEST_ASSERT(stats.observations == 0);

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 2. Single observation --- */

static bool test_nd_single(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.0, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    /* 1 rep, nearest_dist = INFINITY (only one rep) */
    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 1);
    TEST_ASSERT(isinf(nd[0]) && nd[0] > 0.0);

    /* Persistence = INFINITY - 0.1 = INFINITY */
    double pers[10];
    size_t pcount = 10;
    TEST_STATUS(futcache_persist_nd_persistences(eng, pers, &pcount),
                FUTCACHE_OK);
    TEST_ASSERT(isinf(pers[0]));

    /* Novelty at different scales */
    TEST_STATUS(futcache_persist_nd_is_novel_at(
        eng, (const double[]){0.0, 0.0}, 0.05, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false); /* same point */

    TEST_STATUS(futcache_persist_nd_is_novel_at(
        eng, (const double[]){0.5, 0.0}, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true); /* 0.5 away > 0.1 */

    TEST_STATUS(futcache_persist_nd_is_novel_at(
        eng, (const double[]){0.5, 0.0}, 0.6, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false); /* 0.5 away < 0.6 */

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 3. Two observations: nearest distances --- */

static bool test_nd_two_obs(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.0, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.5, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true); /* 0.5 > 0.1 */

    /* 2 reps, each has nearest_dist = 0.5 */
    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 2);
    TEST_NEAR(nd[0], 0.5, 1e-12);
    TEST_NEAR(nd[1], 0.5, 1e-12);

    /* Persistence = 0.5 - 0.1 = 0.4 */
    double pers[10];
    size_t pcount = 10;
    TEST_STATUS(futcache_persist_nd_persistences(eng, pers, &pcount),
                FUTCACHE_OK);
    TEST_NEAR(pers[0], 0.4, 1e-12);
    TEST_NEAR(pers[1], 0.4, 1e-12);

    /* Count above tau */
    size_t count = 0;
    TEST_STATUS(futcache_persist_nd_count_above(eng, 0.3, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 2);

    TEST_STATUS(futcache_persist_nd_count_above(eng, 0.5, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 0); /* 0.4 < 0.5 */

    /* Evict lowest: both have same persistence, evict first (index 0) */
    size_t evicted = 99;
    TEST_STATUS(futcache_persist_nd_evict_lowest(eng, &evicted), FUTCACHE_OK);
    TEST_ASSERT(evicted == 0);

    /* Now 1 rep left (the one at 0.5, 0.0) */
    ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 1);

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 4. Redundant observation --- */

static bool test_nd_redundant(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.2);

    bool novel;
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.0, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    /* Observe a point within 0.2 of the existing rep */
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.1, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    /* Still 1 rep */
    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 1);

    /* But observations = 2 */
    futcache_persist_nd_stats_t stats;
    TEST_STATUS(futcache_persist_nd_get_stats(eng, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == 2);
    TEST_ASSERT(stats.rep_count == 1);

    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 5. Three observations: varied persistence --- */

static bool test_nd_three_obs(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.0, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.5, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    /* Third point: close to the first (distance 0.15) but novel (0.15 > 0.1) */
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.15, 0.0},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    /* 3 reps. Nearest distances:
     * Rep 0 (0.0, 0.0): nearest is rep 2 (0.15, 0.0) at distance 0.15
     * Rep 1 (0.5, 0.0): nearest is rep 2 (0.15, 0.0) at distance 0.35
     * Rep 2 (0.15, 0.0): nearest is rep 0 (0.0, 0.0) at distance 0.15
     */
    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 3);
    TEST_NEAR(nd[0], 0.15, 1e-12);
    TEST_NEAR(nd[1], 0.35, 1e-12);
    TEST_NEAR(nd[2], 0.15, 1e-12);

    /* Persistences: nd - 0.1
     * Rep 0: 0.05
     * Rep 1: 0.25
     * Rep 2: 0.05
     */
    double pers[10];
    size_t pcount = 10;
    TEST_STATUS(futcache_persist_nd_persistences(eng, pers, &pcount),
                FUTCACHE_OK);
    TEST_NEAR(pers[0], 0.05, 1e-12);
    TEST_NEAR(pers[1], 0.25, 1e-12);
    TEST_NEAR(pers[2], 0.05, 1e-12);

    /* Evict lowest persistence: rep 0 or rep 2 (both 0.05).
     * The first one found (index 0) is evicted. */
    size_t evicted = 99;
    TEST_STATUS(futcache_persist_nd_evict_lowest(eng, &evicted), FUTCACHE_OK);
    TEST_ASSERT(evicted == 0);

    /* After eviction, recompute nearest distances.
     * Remaining: rep 1 (0.5, 0.0) and old rep 2 (0.15, 0.0) now at index 0.
     * Wait — after evicting index 0, the remaining reps shift down:
     *   Old rep 1 (0.5, 0.0) -> new index 0
     *   Old rep 2 (0.15, 0.0) -> new index 1
     * Their mutual distance is 0.35.
     */
    ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 2);
    TEST_NEAR(nd[0], 0.35, 1e-12);
    TEST_NEAR(nd[1], 0.35, 1e-12);

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 6. Differential: is_novel_at vs brute force (L2, 3-D) --- */

static bool test_nd_differential(void)
{
    double lo[3] = {-1.0, -1.0, -1.0};
    double hi[3] = { 1.0,  1.0,  1.0};
    futcache_persist_nd_t *eng = NULL;
    TEST_STATUS(futcache_persist_nd_create(
        3, 0.1, test_l2, NULL, lo, hi, 0, NULL, &eng), FUTCACHE_OK);

    double obs[][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
        {-0.3, 0.1, 0.2},
        {0.2, -0.3, 0.1},
    };
    size_t n_obs = sizeof(obs) / sizeof(obs[0]);

    bool novel;
    for (size_t i = 0; i < n_obs; ++i) {
        TEST_STATUS(futcache_persist_nd_observe(eng, obs[i], &novel),
                    FUTCACHE_OK);
    }

    /* Differential test: is_novel_at vs brute force */
    double queries[][3] = {
        {0.1, 0.1, 0.1},
        {0.4, 0.0, 0.0},
        {0.0, 0.3, 0.0},
        {-0.5, 0.0, 0.0},
        {0.9, 0.9, 0.9},
    };
    size_t n_queries = sizeof(queries) / sizeof(queries[0]);
    double scales[] = {0.05, 0.1, 0.2, 0.3, 0.5};
    size_t n_scales = sizeof(scales) / sizeof(scales[0]);

    for (size_t qi = 0; qi < n_queries; ++qi) {
        for (size_t si = 0; si < n_scales; ++si) {
            bool persist_novel = false;
            TEST_STATUS(futcache_persist_nd_is_novel_at(
                eng, queries[qi], scales[si], &persist_novel), FUTCACHE_OK);

            bool expected_novel = true;
            for (size_t oi = 0; oi < n_obs; ++oi) {
                double d = test_l2(queries[qi], obs[oi], 3, NULL);
                if (d <= scales[si] + 1e-12) {
                    expected_novel = false;
                    break;
                }
            }

            if (persist_novel != expected_novel) {
                fprintf(stderr, "  ND MISMATCH: q={%.2f,%.2f,%.2f} t=%.2f\n",
                        queries[qi][0], queries[qi][1], queries[qi][2],
                        scales[si]);
                futcache_persist_nd_destroy(eng);
                return false;
            }
        }
    }

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 7. Eviction sequence: repeatedly evict lowest --- */

static bool test_nd_eviction_sequence(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    /* Create 5 well-separated points */
    double pts[][2] = {
        {0.0, 0.0},
        {0.5, 0.0},
        {-0.5, 0.0},
        {0.0, 0.5},
        {0.0, -0.5},
    };
    for (int i = 0; i < 5; ++i) {
        TEST_STATUS(futcache_persist_nd_observe(eng, pts[i], &novel),
                    FUTCACHE_OK);
        TEST_ASSERT(novel == true);
    }

    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 5);

    /* All points are 0.5 apart from their nearest neighbours.
     * Persistence = 0.5 - 0.1 = 0.4 for all.
     * Evict one: the first one found (index 0). */
    size_t evicted = 99;
    TEST_STATUS(futcache_persist_nd_evict_lowest(eng, &evicted), FUTCACHE_OK);
    TEST_ASSERT(evicted == 0);

    /* After eviction, 4 reps remain. Recompute nearest distances.
     * The remaining 4 points:
     *   (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)
     *   Nearest distances:
     *     (0.5, 0.0) -> nearest is (0.0, 0.5) or (0.0, -0.5) at sqrt(0.5)
     *     Wait, let me compute properly.
     *     (0.5, 0.0) to (-0.5, 0.0): 1.0
     *     (0.5, 0.0) to (0.0, 0.5): sqrt(0.25+0.25) = sqrt(0.5) ≈ 0.707
     *     (0.5, 0.0) to (0.0, -0.5): sqrt(0.5) ≈ 0.707
     *     So nearest is 0.707.
     *
     *   Similarly for others by symmetry.
     */
    ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 4);

    /* Evict 3 more to get down to 1 rep. */
    for (int i = 0; i < 3; ++i) {
        TEST_STATUS(futcache_persist_nd_evict_lowest(eng, &evicted),
                    FUTCACHE_OK);
    }

    ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 1);

    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 8. Clear and reuse --- */

static bool test_nd_clear(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.0, 0.0},
                &novel), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.5, 0.0},
                &novel), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_nd_clear(eng), FUTCACHE_OK);

    double nd[10];
    size_t ncount = 10;
    TEST_STATUS(futcache_persist_nd_nearest_distances(eng, nd, &ncount),
                FUTCACHE_OK);
    TEST_ASSERT(ncount == 0);

    /* Can observe again after clear */
    TEST_STATUS(futcache_persist_nd_observe(eng, (const double[]){0.3, 0.3},
                &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 9. Prime birth tracking --- */

static bool test_nd_prime_births(void)
{
    futcache_persist_nd_t *eng = make_2d_engine(0.1);

    bool novel;
    /* Observe 6 well-separated points. Birth indices: 0,1,2,3,4,5.
     * Prime indices: 2, 3, 5 (0 and 1 are not prime, 4 is not prime). */
    double pts[][2] = {
        {0.0, 0.0},
        {0.5, 0.0},
        {-0.5, 0.0},
        {0.0, 0.5},
        {0.0, -0.5},
        {0.5, 0.5},
    };
    for (int i = 0; i < 6; ++i) {
        TEST_STATUS(futcache_persist_nd_observe(eng, pts[i], &novel),
                    FUTCACHE_OK);
        TEST_ASSERT(novel == true);
    }

    futcache_persist_nd_stats_t stats;
    TEST_STATUS(futcache_persist_nd_get_stats(eng, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == 6);
    TEST_ASSERT(stats.rep_count == 6);

    /* Prime birth indices: 2, 3, 5 → 3 reps */
    TEST_ASSERT(stats.prime_birth_count == 3);

    /* Verify birth_prime values match prime table */
    futcache_persist_nd_rep_t reps[10];
    size_t rcount = 10;
    TEST_STATUS(futcache_persist_nd_copy_reps(eng, reps, &rcount), FUTCACHE_OK);
    TEST_ASSERT(rcount == 6);

    for (size_t i = 0; i < 6; ++i) {
        uint64_t expected_prime = futcache_persist_prime_mod(i);
        TEST_ASSERT(reps[i].birth_prime == expected_prime);
        TEST_ASSERT(reps[i].birth_index == i);
    }

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

/* --- 10. Randomized differential (L2, 2-D, 10 points, 50 trials) --- */

static bool test_nd_randomized(void)
{
    double lo[2] = {-1.0, -1.0};
    double hi[2] = { 1.0,  1.0};
    futcache_persist_nd_t *eng = NULL;
    TEST_STATUS(futcache_persist_nd_create(
        2, 0.1, test_l2, NULL, lo, hi, 0, NULL, &eng), FUTCACHE_OK);

    srand(123);
    double obs[10][2];
    bool novel;
    for (int i = 0; i < 10; ++i) {
        obs[i][0] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        obs[i][1] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        TEST_STATUS(futcache_persist_nd_observe(eng, obs[i], &novel),
                    FUTCACHE_OK);
    }

    for (int trial = 0; trial < 50; ++trial) {
        double qx = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double qy = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double qt = ((double)rand() / RAND_MAX) * 0.5;
        const double q[2] = {qx, qy};

        bool persist_novel = false;
        TEST_STATUS(futcache_persist_nd_is_novel_at(eng, q, qt,
                    &persist_novel), FUTCACHE_OK);

        bool expected_novel = true;
        for (int i = 0; i < 10; ++i) {
            double d = test_l2(q, obs[i], 2, NULL);
            if (d <= qt + 1e-12) {
                expected_novel = false;
                break;
            }
        }

        if (persist_novel != expected_novel) {
            fprintf(stderr, "  ND RAND MISMATCH: q={%.4f,%.4f} t=%.4f "
                    "persist=%d expected=%d\n",
                    qx, qy, qt, persist_novel, expected_novel);
            futcache_persist_nd_destroy(eng);
            return false;
        }
    }

    TEST_STATUS(futcache_persist_nd_validate(eng), FUTCACHE_OK);
    futcache_persist_nd_destroy(eng);
    return true;
}

int persist_nd_test_suite(void)
{
    static const test_case_t tests[] = {
        {"empty engine", test_nd_empty},
        {"single observation", test_nd_single},
        {"two observations (nearest dist)", test_nd_two_obs},
        {"redundant observation", test_nd_redundant},
        {"three observations (varied persistence)", test_nd_three_obs},
        {"differential is_novel_at (75 combos, 3-D L2)", test_nd_differential},
        {"eviction sequence", test_nd_eviction_sequence},
        {"clear and reuse", test_nd_clear},
        {"prime birth tracking", test_nd_prime_births},
        {"randomized differential (50 trials, 2-D L2)", test_nd_randomized},
    };
    return run_test_cases("persist_nd", tests,
                          sizeof(tests) / sizeof(tests[0]));
}

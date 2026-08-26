#include "test.h"
#include "futcache/select.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Submodular selection tests (Design Sketch 03)
 * ============================================================ */

/* --- 1. Basic max-coverage: 1D, 3 clusters --- */

static bool test_select_basic_coverage(void)
{
    /* 1D points: cluster A around 0.1 (5 pts), cluster B around 0.5 (5 pts),
     * cluster C around 0.9 (5 pts). eps=0.15.
     * k=3: greedy should pick one from each cluster, covering all 15.
     * k=1: greedy should pick the point in the largest cluster (all same size,
     * so tie-break by lexicographic — lowest value wins). */

    double pts[15];
    int idx = 0;
    for (int i = 0; i < 5; i++) pts[idx++] = 0.1 + i * 0.01;
    for (int i = 0; i < 5; i++) pts[idx++] = 0.5 + i * 0.01;
    for (int i = 0; i < 5; i++) pts[idx++] = 0.9 + i * 0.01;

    double eps = 0.15;

    /* k=3: should cover all 15 points */
    futcache_select_result_t *res = NULL;
    futcache_status_t st = futcache_select_max_coverage(
        pts, 15, 1, eps, 3, futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);

    TEST_ASSERT(res->selected_count == 3);
    TEST_ASSERT(res->total_covered == 15);
    TEST_ASSERT(res->total_points == 15);
    TEST_ASSERT(res->coverage_ratio > 0.99);

    /* Each selected rep should be from a different cluster */
    int clusters_found = 0;
    for (int c = 0; c < 3; c++) {
        for (size_t i = 0; i < res->selected_count; i++) {
            double val = pts[res->selected_indices[i]];
            if (c == 0 && val < 0.3) clusters_found++;
            if (c == 1 && val > 0.3 && val < 0.7) clusters_found++;
            if (c == 2 && val > 0.7) clusters_found++;
        }
    }
    TEST_ASSERT(clusters_found == 3);

    futcache_select_free_result(res);

    /* k=1: covers only ~5 points (one cluster) */
    res = NULL;
    st = futcache_select_max_coverage(pts, 15, 1, eps, 1,
                                       futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_ASSERT(res->selected_count == 1);
    TEST_ASSERT(res->total_covered <= 6); /* at most one cluster + edges */
    TEST_ASSERT(res->total_covered >= 3);
    futcache_select_free_result(res);

    return true;
}

/* --- 2. Approximation ratio on small instance --- */

static bool test_select_approximation_ratio(void)
{
    /* n=8 points in 1D, k=2, eps=0.3.
     * Brute-force optimum should be computable (n <= 20).
     * Verify greedy >= (1-1/e) * opt. */
    double pts[8] = {0.0, 0.1, 0.5, 0.55, 0.9, 0.95, 0.2, 0.7};
    double eps = 0.3;
    size_t k = 2;

    futcache_select_result_t *res = NULL;
    futcache_status_t st = futcache_select_max_coverage(
        pts, 8, 1, eps, k, futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);

    /* Brute force should have been computed (n=8 <= 20) */
    TEST_ASSERT(res->opt_coverage > 0.0);
    TEST_ASSERT(res->approximation_ratio > 0.0);

    /* 1 - 1/e ≈ 0.632 */
    double lower_bound = 0.632 * res->opt_coverage;
    /* Greedy coverage should be at least (1-1/e) * opt, with small numerical
     * tolerance. Actually greedy is usually much better than 1-1/e on small
     * instances, but the theorem guarantees it. */
    TEST_ASSERT((double)res->total_covered >= lower_bound - 0.01);

    /* Greedy should not exceed optimum */
    TEST_ASSERT((double)res->total_covered <= res->opt_coverage + 0.01);

    futcache_select_free_result(res);
    return true;
}

/* --- 3. Determinism: same input, different arrival order --- */

static bool test_select_determinism(void)
{
    /* Same set of points, shuffled. Greedy with lexicographic tie-break
     * should select the same representatives (as a set). */
    double pts_a[6] = {0.1, 0.2, 0.5, 0.6, 0.9, 0.95};
    double pts_b[6] = {0.9, 0.5, 0.1, 0.95, 0.6, 0.2}; /* shuffled */
    double eps = 0.25;
    size_t k = 2;

    futcache_select_result_t *res_a = NULL, *res_b = NULL;
    futcache_status_t st;

    st = futcache_select_max_coverage(pts_a, 6, 1, eps, k,
                                       futcache_distance_l1, NULL, &res_a);
    TEST_STATUS(st, FUTCACHE_OK);
    st = futcache_select_max_coverage(pts_b, 6, 1, eps, k,
                                       futcache_distance_l1, NULL, &res_b);
    TEST_STATUS(st, FUTCACHE_OK);

    /* Same coverage */
    TEST_ASSERT(res_a->total_covered == res_b->total_covered);
    TEST_NEAR(res_a->coverage_ratio, res_b->coverage_ratio, 1e-15);

    /* Same set of selected points (as a set, not sequence) */
    for (size_t i = 0; i < res_a->selected_count; i++) {
        double val_a = pts_a[res_a->selected_indices[i]];
        bool found = false;
        for (size_t j = 0; j < res_b->selected_count; j++) {
            double val_b = pts_b[res_b->selected_indices[j]];
            if (fabs(val_a - val_b) < 1e-15) {
                found = true;
                break;
            }
        }
        TEST_ASSERT(found);
    }

    futcache_select_free_result(res_a);
    futcache_select_free_result(res_b);
    return true;
}

/* --- 4. Evict-worst: evict the most redundant rep --- */

static bool test_select_evict_worst(void)
{
    /* 1D: reps at 0.1, 0.15, 0.5. Points: 5 near 0.1, 3 near 0.15, 5 near 0.5.
     * eps=0.1.
     * Rep 0.15 is within 0.05 of rep 0.1, so its unique coverage is low.
     * Evict-worst should pick rep 0.15 (index 1). */
    double points[13];
    int idx = 0;
    for (int i = 0; i < 5; i++) points[idx++] = 0.08 + i * 0.01;  /* near 0.1 */
    for (int i = 0; i < 3; i++) points[idx++] = 0.13 + i * 0.01;  /* near 0.15 */
    for (int i = 0; i < 5; i++) points[idx++] = 0.48 + i * 0.01;  /* near 0.5 */

    double reps[3] = {0.1, 0.15, 0.5};
    double eps = 0.1;

    size_t evict_idx = 99;
    double marginal_loss = -1.0;
    futcache_status_t st = futcache_select_evict_worst(
        points, 13, reps, 3, 1, eps,
        futcache_distance_l1, NULL, &evict_idx, &marginal_loss);
    TEST_STATUS(st, FUTCACHE_OK);

    /* Reps 0 (val=0.1) and 1 (val=0.15) are within 0.05 of each other.
     * With eps=0.1, every point covered by one is also covered by the other,
     * so both have unique coverage = 0. Rep 2 (val=0.5) has 5 unique points.
     * Tie between index 0 and 1: strict < means first minimum wins → index 0. */
    TEST_ASSERT(evict_idx == 0 || evict_idx == 1);
    TEST_ASSERT(marginal_loss >= 0.0);
    TEST_ASSERT(marginal_loss <= 1.0); /* 0 or 1 unique point at most */

    /* Single rep: evict it, loss = all its points */
    double single_reps[1] = {0.5};
    evict_idx = 99;
    marginal_loss = -1.0;
    st = futcache_select_evict_worst(
        points, 13, single_reps, 1, 1, eps,
        futcache_distance_l1, NULL, &evict_idx, &marginal_loss);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_ASSERT(evict_idx == 0);
    /* Only points near 0.5 are covered: 5 points */
    TEST_ASSERT(marginal_loss >= 3.0);

    return true;
}

/* --- 5. Coverage function --- */

static bool test_select_coverage(void)
{
    /* 1D: 10 points in [0,1], rep at 0.5, eps=0.3.
     * Points within 0.3 of 0.5: [0.2, 0.8] → 6 out of 10. */
    double points[10];
    for (int i = 0; i < 10; i++) points[i] = (double)i / 9.0;

    double reps[1] = {0.5};
    double coverage = -1.0;
    futcache_status_t st = futcache_select_coverage(
        points, 10, reps, 1, 1, 0.3,
        futcache_distance_l1, NULL, &coverage);
    TEST_STATUS(st, FUTCACHE_OK);

    /* Points at 0.2, 0.333, 0.444, 0.555, 0.667, 0.778, 0.8 are within 0.3
     * of 0.5. Let's count: |x - 0.5| <= 0.3 → x in [0.2, 0.8].
     * Points: 0.222(0.2222), 0.333, 0.444, 0.555, 0.667, 0.778, 0.888?
     * 0.888 - 0.5 = 0.388 > 0.3, so not covered.
     * 0.111 - 0.5 = 0.389 > 0.3, not covered.
     * Covered: 0.222, 0.333, 0.444, 0.5, 0.555, 0.667, 0.778 → 7 points.
     * Wait, 0.0 is also a point. |0.0 - 0.5| = 0.5 > 0.3. Not covered.
     * |1.0 - 0.5| = 0.5 > 0.3. Not covered.
     * So 7 out of 10. */
    TEST_NEAR(coverage, 0.6, 1e-12);

    /* No reps → 0 coverage */
    coverage = -1.0;
    st = futcache_select_coverage(points, 10, NULL, 0, 1, 0.3,
                                   futcache_distance_l1, NULL, &coverage);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_NEAR(coverage, 0.0, 1e-15);

    return true;
}

/* --- 6. High-dimensional: 8D clusters --- */

static bool test_select_high_dimensional(void)
{
    /* 8D: 3 clusters of 20 points each. k=3, eps=0.5.
     * Greedy should pick one from each cluster. */
    size_t dim = 8;
    size_t per_cluster = 20;
    size_t n = 3 * per_cluster;

    double *pts = (double *)malloc(n * dim * sizeof(double));
    TEST_ASSERT(pts != NULL);

    double centers[3][8] = {
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
        {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
        {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}
    };

    unsigned int seed = 42;
    for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < per_cluster; i++) {
            for (size_t d = 0; d < dim; d++) {
                double noise = ((double)rand_r(&seed) / RAND_MAX - 0.5) * 0.1;
                pts[(c * per_cluster + i) * dim + d] = centers[c][d] + noise;
            }
        }
    }

    double eps = 0.5;
    size_t k = 3;

    futcache_select_result_t *res = NULL;
    futcache_status_t st = futcache_select_max_coverage(
        pts, n, dim, eps, k, futcache_distance_l2, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);

    TEST_ASSERT(res->selected_count == 3);
    /* All 60 points should be covered (clusters are well-separated) */
    TEST_ASSERT(res->total_covered == n);
    TEST_NEAR(res->coverage_ratio, 1.0, 1e-12);

    free(pts);
    futcache_select_free_result(res);
    return true;
}

/* --- 7. Edge cases --- */

static bool test_select_edge_cases(void)
{
    /* k=1, n=1 */
    double pts[1] = {0.5};
    futcache_select_result_t *res = NULL;
    futcache_status_t st = futcache_select_max_coverage(
        pts, 1, 1, 0.1, 1, futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_ASSERT(res->selected_count == 1);
    TEST_ASSERT(res->total_covered == 1);
    futcache_select_free_result(res);

    /* k=n: select all */
    double pts2[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    res = NULL;
    st = futcache_select_max_coverage(pts2, 5, 1, 0.1, 5,
                                       futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_ASSERT(res->selected_count == 5);
    TEST_ASSERT(res->total_covered == 5);
    futcache_select_free_result(res);

    /* Invalid: k > n */
    res = NULL;
    st = futcache_select_max_coverage(pts2, 5, 1, 0.1, 6,
                                       futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_ASSERT(res == NULL);

    /* Invalid: k = 0 */
    st = futcache_select_max_coverage(pts2, 5, 1, 0.1, 0,
                                       futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_ERROR_INVALID_ARGUMENT);

    /* NULL points */
    st = futcache_select_max_coverage(NULL, 5, 1, 0.1, 1,
                                       futcache_distance_l1, NULL, &res);
    TEST_STATUS(st, FUTCACHE_ERROR_INVALID_ARGUMENT);

    /* evict_worst with 0 reps */
    size_t evict_idx = 99;
    double loss = -1.0;
    st = futcache_select_evict_worst(pts2, 5, pts2, 0, 1, 0.1,
                                      futcache_distance_l1, NULL,
                                      &evict_idx, &loss);
    TEST_STATUS(st, FUTCACHE_ERROR_OUT_OF_RANGE);

    return true;
}

/* --- 8. Streaming swap vs FIFO --- */

static bool test_select_streaming_swap(void)
{
    /* Simulate streaming: feed points, when budget exceeded, evict the
     * worst rep (lowest marginal coverage) vs FIFO. Compare final coverage.
     *
     * 3 clusters in 1D, budget k=3, 30 points total.
     * FIFO evicts in arrival order; swap evicts the least-valuable rep. */
    double pts[30];
    int idx = 0;
    for (int cycle = 0; cycle < 10; cycle++) {
        for (int c = 0; c < 3; c++) {
            pts[idx++] = 0.1 + c * 0.4 + cycle * 0.001;
        }
    }

    double eps = 0.15;
    size_t budget = 3;

    /* --- FIFO strategy --- */
    double fifo_reps[30];
    size_t fifo_count = 0;
    size_t fifo_evictions = 0;
    for (size_t i = 0; i < 30; i++) {
        /* Check if point is already covered */
        bool covered = false;
        for (size_t j = 0; j < fifo_count; j++) {
            if (fabs(pts[i] - fifo_reps[j]) <= eps) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            if (fifo_count < budget) {
                fifo_reps[fifo_count++] = pts[i];
            } else {
                /* Evict oldest (index 0) */
                for (size_t j = 0; j < budget - 1; j++) {
                    fifo_reps[j] = fifo_reps[j + 1];
                }
                fifo_reps[budget - 1] = pts[i];
                fifo_evictions++;
            }
        }
    }

    double fifo_cov;
    TEST_STATUS(futcache_select_coverage(pts, 30, fifo_reps, budget,
                                          1, eps, futcache_distance_l1,
                                          NULL, &fifo_cov), FUTCACHE_OK);

    /* --- Swap strategy --- */
    double swap_reps[30];
    size_t swap_count = 0;
    size_t swap_evictions = 0;
    for (size_t i = 0; i < 30; i++) {
        bool covered = false;
        for (size_t j = 0; j < swap_count; j++) {
            if (fabs(pts[i] - swap_reps[j]) <= eps) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            if (swap_count < budget) {
                swap_reps[swap_count++] = pts[i];
            } else {
                /* Evict the rep with lowest marginal coverage */
                size_t evict_idx;
                double marginal_loss;
                TEST_STATUS(futcache_select_evict_worst(
                    pts, i + 1, swap_reps, swap_count, 1, eps,
                    futcache_distance_l1, NULL, &evict_idx,
                    &marginal_loss), FUTCACHE_OK);
                for (size_t j = evict_idx; j < swap_count - 1; j++) {
                    swap_reps[j] = swap_reps[j + 1];
                }
                swap_reps[swap_count - 1] = pts[i];
                swap_evictions++;
            }
        }
    }

    double swap_cov;
    TEST_STATUS(futcache_select_coverage(pts, 30, swap_reps, budget,
                                          1, eps, futcache_distance_l1,
                                          NULL, &swap_cov), FUTCACHE_OK);

    /* Swap should cover at least as much as FIFO */
    TEST_ASSERT(swap_cov >= fifo_cov - 1e-12);

    printf("    [streaming] fifo_cov=%.3f swap_cov=%.3f "
           "(fifo_evictions=%zu swap_evictions=%zu)\n",
           fifo_cov, swap_cov, fifo_evictions, swap_evictions);

    return true;
}

int select_test_suite(void)
{
    static const test_case_t tests[] = {
        {"basic max-coverage (1D, 3 clusters)", test_select_basic_coverage},
        {"approximation ratio vs brute-force opt", test_select_approximation_ratio},
        {"determinism under permutation", test_select_determinism},
        {"evict-worst selects most redundant", test_select_evict_worst},
        {"coverage function", test_select_coverage},
        {"high-dimensional 8D clusters", test_select_high_dimensional},
        {"edge cases", test_select_edge_cases},
        {"streaming swap vs FIFO", test_select_streaming_swap},
    };
    return run_test_cases("select", tests, sizeof(tests) / sizeof(tests[0]));
}

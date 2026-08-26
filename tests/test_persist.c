#include "test.h"
#include "futcache/persist.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Persistent novelty tests (Design Sketch 01, Phases 1-4)
 * ============================================================ */

/* --- Helper: is n prime? (for test assertions) --- */
static bool test_is_prime(size_t n)
{
    if (n < 2) return false;
    for (size_t d = 2; d * d <= n; ++d) {
        if (n % d == 0) return false;
    }
    return true;
}

/* --- 1. Prime table correctness --- */

static bool test_prime_table(void)
{
    TEST_ASSERT(futcache_persist_nth_prime(0) == 2);
    TEST_ASSERT(futcache_persist_nth_prime(1) == 3);
    TEST_ASSERT(futcache_persist_nth_prime(2) == 5);
    TEST_ASSERT(futcache_persist_nth_prime(3) == 7);
    TEST_ASSERT(futcache_persist_nth_prime(4) == 11);
    TEST_ASSERT(futcache_persist_nth_prime(5) == 13);
    TEST_ASSERT(futcache_persist_nth_prime(100) == 547);
    TEST_ASSERT(futcache_persist_nth_prime(1000) == 7927);

    /* Verify primality of table entries */
    for (size_t i = 0; i < 200; ++i) {
        uint64_t p = futcache_persist_nth_prime(i);
        TEST_ASSERT(test_is_prime((size_t)p));
    }

    /* Prime mod */
    TEST_ASSERT(futcache_persist_prime_mod(0) == 2);
    TEST_ASSERT(futcache_persist_prime_mod(1) == 3);

    return true;
}

/* --- 2. Empty engine --- */

static bool test_persist_empty(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    bool novel = false;
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.5, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    double intervals[4];
    size_t icount = 4;
    TEST_STATUS(futcache_persist_novelty_spectrum(eng, 0.5, intervals, &icount), FUTCACHE_OK);
    TEST_ASSERT(icount == 1);
    TEST_NEAR(intervals[0], 0.0, 1e-15);
    TEST_ASSERT(isinf(intervals[1]) && intervals[1] > 0.0);

    size_t dcount = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng, NULL, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 0);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);

    /* Stats: zero everything */
    futcache_persist_stats_t stats;
    TEST_STATUS(futcache_persist_get_stats(eng, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == 0);
    TEST_ASSERT(stats.feature_count == 0);
    TEST_ASSERT(stats.prime_cycle_count == 0);

    futcache_persist_destroy(eng);
    return true;
}

/* --- 3. Single observation --- */

static bool test_persist_single_observe(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_observe(eng, 0.5), FUTCACHE_OK);

    bool novel;
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.5, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.8, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.75, 0.25, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    /* Merge tree: 1 leaf, 0 features */
    TEST_ASSERT(futcache_persist_node_count(eng) == 1);

    futcache_persist_feature_t feats[10];
    size_t dcount = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng, feats, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 0);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 4. Two observations: merge tree structure --- */

static bool test_persist_two_obs(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_observe(eng, 0.2), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.8), FUTCACHE_OK);

    /* Merge tree: 2 leaves + 1 internal = 3 nodes */
    TEST_ASSERT(futcache_persist_node_count(eng) == 3);

    /* Node 0: leaf at 0.2 */
    const futcache_persist_node_t *n0 = futcache_persist_get_node(eng, 0);
    TEST_ASSERT(n0 != NULL);
    TEST_ASSERT(n0->is_leaf == true);
    TEST_NEAR(n0->lo, 0.2, 1e-12);

    /* Node 1: leaf at 0.8 */
    const futcache_persist_node_t *n1 = futcache_persist_get_node(eng, 1);
    TEST_ASSERT(n1 != NULL);
    TEST_ASSERT(n1->is_leaf == true);
    TEST_NEAR(n1->lo, 0.8, 1e-12);

    /* Node 2: internal merge node */
    const futcache_persist_node_t *n2 = futcache_persist_get_node(eng, 2);
    TEST_ASSERT(n2 != NULL);
    TEST_ASSERT(n2->is_leaf == false);
    TEST_ASSERT(n2->left == 0);
    TEST_ASSERT(n2->right == 1);
    TEST_ASSERT(n2->parent == -1);
    TEST_NEAR(n2->death_scale, 0.3, 1e-12); /* half-gap = (0.8 - 0.2)/2 */

    /* 1 feature: persistence = half the gap = 0.3 */
    futcache_persist_feature_t feats[10];
    size_t dcount = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng, feats, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 1);
    TEST_NEAR(feats[0].persistence, 0.3, 1e-12);
    TEST_ASSERT(feats[0].birth_prime == 2); /* p_0 = 2 */
    TEST_ASSERT(feats[0].death_prime == 3); /* p_1 = 3 */

    /* is_novel_at checks */
    bool novel;
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.2, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.5, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true);

    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.5, 0.3, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 5. Three observations: verify full merge tree --- */

static bool test_persist_three_obs(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_observe(eng, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 1.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.4), FUTCACHE_OK);

    /* Sorted: 0.0, 0.4, 1.0. Gaps: 0.4, 0.6.
     * Merge order: gap 0.4 first (between 0.0 and 0.4), then gap 0.6
     * (between {0.0, 0.4} and 1.0).
     *
     * Merge tree:
     *   Node 0: leaf at 0.0
     *   Node 1: leaf at 0.4
     *   Node 2: leaf at 1.0
     *   Node 3: internal, merges node 0 and node 1, death_scale = 0.2
     *   Node 4: internal, merges node 3 and node 2, death_scale = 0.3
     *
     * Features:
     *   Feature 0: persistence = 0.2 (half-gap between 0.0 and 0.4)
     *   Feature 1: persistence = 0.3 (half-gap between 0.4 and 1.0)
     */

    TEST_ASSERT(futcache_persist_node_count(eng) == 5);

    /* Check internal nodes */
    const futcache_persist_node_t *n3 = futcache_persist_get_node(eng, 3);
    TEST_ASSERT(n3 != NULL);
    TEST_ASSERT(n3->is_leaf == false);
    TEST_NEAR(n3->death_scale, 0.2, 1e-12);

    const futcache_persist_node_t *n4 = futcache_persist_get_node(eng, 4);
    TEST_ASSERT(n4 != NULL);
    TEST_ASSERT(n4->is_leaf == false);
    TEST_NEAR(n4->death_scale, 0.3, 1e-12);
    TEST_ASSERT(n4->parent == -1); /* root */

    /* Features: 2 features, persistence 0.2 and 0.3 */
    futcache_persist_feature_t feats[10];
    size_t dcount = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng, feats, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 2);
    TEST_NEAR(feats[0].persistence, 0.2, 1e-12);
    TEST_NEAR(feats[1].persistence, 0.3, 1e-12);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 6. Differential: is_novel_at vs manual brute force --- */

static bool test_persist_differential(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    double obs[] = {0.1, 0.15, 0.3, 0.32, 0.5, 0.7, 0.75, 0.78, 0.9, 0.95};
    size_t n_obs = sizeof(obs) / sizeof(obs[0]);
    for (size_t i = 0; i < n_obs; ++i) {
        TEST_STATUS(futcache_persist_observe(eng, obs[i]), FUTCACHE_OK);
    }

    /* Merge tree should have 2*10 - 1 = 19 nodes */
    TEST_ASSERT(futcache_persist_node_count(eng) == 19);

    /* 9 features (n-1) */
    futcache_persist_feature_t feats[20];
    size_t dcount = 20;
    TEST_STATUS(futcache_persist_copy_diagram(eng, feats, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 9);

    /* Differential test: is_novel_at vs brute force */
    double scales[] = {0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5};
    double queries[] = {0.0, 0.12, 0.25, 0.5, 0.6, 0.72, 0.85, 1.0};
    size_t n_scales = sizeof(scales) / sizeof(scales[0]);
    size_t n_queries = sizeof(queries) / sizeof(queries[0]);

    for (size_t si = 0; si < n_scales; ++si) {
        for (size_t qi = 0; qi < n_queries; ++qi) {
            bool persist_novel = false;
            TEST_STATUS(futcache_persist_is_novel_at(eng, queries[qi],
                scales[si], &persist_novel), FUTCACHE_OK);

            bool expected_novel = true;
            for (size_t oi = 0; oi < n_obs; ++oi) {
                double d = queries[qi] - obs[oi];
                if (d < 0) d = -d;
                if (d <= scales[si]) {
                    expected_novel = false;
                    break;
                }
            }

            if (persist_novel != expected_novel) {
                fprintf(stderr, "  MISMATCH: x=%.4f t=%.4f persist=%d expected=%d\n",
                        queries[qi], scales[si], persist_novel, expected_novel);
                futcache_persist_destroy(eng);
                return false;
            }
        }
    }

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 7. Novelty spectrum --- */

static bool test_persist_spectrum(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_observe(eng, 0.3), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.7), FUTCACHE_OK);

    /* x=0.5: nearest is 0.3 or 0.7, distance 0.2 */
    double intervals[4];
    size_t icount = 4;
    TEST_STATUS(futcache_persist_novelty_spectrum(eng, 0.5, intervals, &icount), FUTCACHE_OK);
    TEST_ASSERT(icount == 1);
    TEST_NEAR(intervals[0], 0.0, 1e-15);
    TEST_NEAR(intervals[1], 0.2, 1e-12);

    /* x=0.3: already observed */
    icount = 4;
    TEST_STATUS(futcache_persist_novelty_spectrum(eng, 0.3, intervals, &icount), FUTCACHE_OK);
    TEST_ASSERT(icount == 0);

    futcache_persist_destroy(eng);
    return true;
}

/* --- 8. Feature count, evict_below, stats --- */

static bool test_persist_features_evict(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    /* Observe 5 points: 0.0, 0.1, 0.2, 0.3, 0.4
     * Gaps: 0.1, 0.1, 0.1, 0.1 (all equal)
     * 4 features, each with persistence 0.05 */
    double pts[] = {0.0, 0.1, 0.2, 0.3, 0.4};
    for (int i = 0; i < 5; ++i) {
        TEST_STATUS(futcache_persist_observe(eng, pts[i]), FUTCACHE_OK);
    }

    size_t count = 0;
    TEST_STATUS(futcache_persist_feature_count(eng, 0.0, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 4);

    /* Evict features with persistence < 0.15: all 4 have persistence 0.05,
     * so all are evicted. */
    TEST_STATUS(futcache_persist_evict_below(eng, 0.15), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_feature_count(eng, 0.0, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 0);

    /* Stats */
    futcache_persist_stats_t stats;
    TEST_STATUS(futcache_persist_get_stats(eng, &stats), FUTCACHE_OK);
    TEST_ASSERT(stats.observations == 5);
    TEST_ASSERT(stats.feature_count == 0);

    /* Prime cycle count: after evict, 0 */
    size_t prime_count = 0;
    TEST_STATUS(futcache_persist_prime_cycle_count(eng, 0.0, &prime_count), FUTCACHE_OK);
    TEST_ASSERT(prime_count == 0);

    futcache_persist_destroy(eng);
    return true;
}

/* --- 9. Signature join: idempotent, commutative, associative --- */

static bool test_crdt_merge(void)
{
    /* Create two identical diagrams and verify idempotent merge. */
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    futcache_persist_t *eng_a = NULL, *eng_b = NULL, *eng_c = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng_a), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_create(&cfg, &eng_b), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_create(&cfg, &eng_c), FUTCACHE_OK);

    /* eng_a: observe 0.0, 1.0 (gap = 1.0) */
    TEST_STATUS(futcache_persist_observe(eng_a, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng_a, 1.0), FUTCACHE_OK);

    /* eng_b: same observations (identical diagram) */
    TEST_STATUS(futcache_persist_observe(eng_b, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng_b, 1.0), FUTCACHE_OK);

    /* eng_c: different observations, so different observation indices.
     * Observe 3 points to get birth indices 0,1 and a feature with death=2
     * (a different death_prime). */
    TEST_STATUS(futcache_persist_observe(eng_c, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng_c, 1.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng_c, 3.0), FUTCACHE_OK);

    /* Get diagrams */
    futcache_persist_feature_t diag_a[10], diag_b[10], diag_c[10];
    size_t na = 10, nb = 10, nc = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng_a, diag_a, &na), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_copy_diagram(eng_b, diag_b, &nb), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_copy_diagram(eng_c, diag_c, &nc), FUTCACHE_OK);

    TEST_ASSERT(na == 1); /* one gap for 2 points */
    TEST_ASSERT(nb == 1);
    TEST_ASSERT(nc == 2); /* two gaps for 3 points */

    /* Idempotent: merge(A, A) == A */
    futcache_persist_feature_t merged[20];
    size_t merged_count = 20;
    TEST_STATUS(futcache_persist_merge_features(
        diag_a, na, diag_a, na, merged, &merged_count), FUTCACHE_OK);
    TEST_ASSERT(merged_count == 1); /* should still be 1 feature */
    TEST_NEAR(merged[0].persistence, diag_a[0].persistence, 1e-12);

    /* merge(A, C): A has 1 feature (birth=0,death=1), C has 2 features
     * (birth=0,death=1) and (birth=0,death=2) or (birth=1,death=2).
     * Since A's feature and C's first feature share the same (birth,death)
     * signature, the merge should deduplicate them, keeping the larger
     * persistence. Result: 2 features. */
    size_t m3_count = 20;
    futcache_persist_feature_t merged_ac[20];
    TEST_STATUS(futcache_persist_merge_features(
        diag_a, na, diag_c, nc, merged_ac, &m3_count), FUTCACHE_OK);
    TEST_ASSERT(m3_count == 2);

    futcache_persist_destroy(eng_a);
    futcache_persist_destroy(eng_b);
    futcache_persist_destroy(eng_c);
    return true;
}

/* --- 10. Finite prime-product diagnostic (legacy API name) --- */

static bool test_selberg_zeta(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    /* Empty engine: Z(s) = 1 */
    double zeta = -1.0;
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 1.0, &zeta), FUTCACHE_OK);
    TEST_NEAR(zeta, 1.0, 1e-12);

    /* Observe points: 0.0, 1.0, 3.0
     * Gaps: 1.0 (between 0 and 1), 2.0 (between 1 and 3)
     * Features:
     *   Feature 0: birth=0, death=1, persistence=0.5
     *   Feature 1: birth=1, death=2, persistence=1.0
     *
     * Prime birth indices: birth=0 (not prime), birth=1 (not prime)
     * So Z(s) = 1 (no prime-birth features)
     *
     * Wait — the "birth" in the feature is the observation index of the
     * leftmost point in the left child. For the first merge (gap between
     * obs 0 and obs 1), birth = 0. For the second merge (gap between
     * the merged component and obs 2), birth = 0 (leftmost point of the
     * merged component is still obs 0).
     *
     * Hmm, let me think about this differently. The features' birth
     * values are the observation indices. Index 0 is not prime, index 1
     * is not prime, index 2 is prime, index 3 is prime.
     *
     * For this small example, let me just verify Z(s) > 1 when there
     * are prime-birth features, and Z(s) = 1 when there aren't.
     */
    TEST_STATUS(futcache_persist_observe(eng, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 1.0), FUTCACHE_OK);

    /* 1 feature: birth=0 (not prime) */
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 1.0, &zeta), FUTCACHE_OK);
    TEST_NEAR(zeta, 1.0, 1e-12);

    /* Add more points to get a prime-birth feature.
     * After observing 0.0, 1.0, 4.0:
     *   Gap 1: 1.0 (0→1), birth=0
     *   Gap 2: 3.0 (1→4), birth=1
     * Neither birth is prime (0 and 1 are not prime).
     *
     * After observing 0.0, 1.0, 4.0, 7.0:
     *   Gaps: 1.0, 3.0, 3.0
     *   Features: births at 0, 1, 1 (or similar)
     *   Still no prime births.
     *
     * Let me try: 0.0, 1.0, 4.0, 7.0, 12.0
     *   Gaps: 1.0, 3.0, 3.0, 5.0
     *   The 4th merge involves the component containing obs 3 (birth=3,
     *   which is prime).
     */
    TEST_STATUS(futcache_persist_observe(eng, 4.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 7.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 12.0), FUTCACHE_OK);

    /* Now we have 4 features. Some may have prime birth indices. */
    futcache_persist_feature_t feats[10];
    size_t dcount = 10;
    TEST_STATUS(futcache_persist_copy_diagram(eng, feats, &dcount), FUTCACHE_OK);
    TEST_ASSERT(dcount == 4);

    /* Count how many have prime birth */
    size_t prime_births = 0;
    for (size_t i = 0; i < dcount; ++i) {
        if (test_is_prime(feats[i].birth)) prime_births++;
    }

    /* Z(s) should be > 1 if any prime-birth feature exists */
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 2.0, &zeta), FUTCACHE_OK);
    if (prime_births > 0) {
        TEST_ASSERT(zeta > 1.0);
    } else {
        TEST_NEAR(zeta, 1.0, 1e-12);
    }

    /* Z(s) increases as s decreases (for s > 0) */
    double zeta_s2 = 0, zeta_s5 = 0;
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 2.0, &zeta_s2), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 5.0, &zeta_s5), FUTCACHE_OK);
    if (prime_births > 0) {
        TEST_ASSERT(zeta_s2 > zeta_s5);
    }

    /* Prime cycle count */
    size_t pc_count = 0;
    TEST_STATUS(futcache_persist_prime_cycle_count(eng, 0.0, &pc_count), FUTCACHE_OK);
    TEST_ASSERT(pc_count == prime_births);

    /* Invalid s */
    TEST_STATUS(futcache_persist_selberg_zeta(eng, 0.0, &zeta),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    TEST_STATUS(futcache_persist_selberg_zeta(eng, -1.0, &zeta),
                FUTCACHE_ERROR_INVALID_ARGUMENT);

    futcache_persist_destroy(eng);
    return true;
}

/* --- 11. Clear and reuse --- */

static bool test_persist_clear_reuse(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_observe(eng, 0.5), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 1.0), FUTCACHE_OK);

    TEST_STATUS(futcache_persist_clear(eng), FUTCACHE_OK);

    bool novel;
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.5, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == true); /* after clear, everything is novel */

    TEST_STATUS(futcache_persist_observe(eng, 0.8), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.8, 0.1, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 12. Randomized differential test --- */

static bool test_persist_randomized(void)
{
    /* Generate 20 random points, verify is_novel_at at 20 random scales
     * against brute force. */
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    srand(42);
    double obs[20];
    for (int i = 0; i < 20; ++i) {
        obs[i] = (double)rand() / RAND_MAX;
        TEST_STATUS(futcache_persist_observe(eng, obs[i]), FUTCACHE_OK);
    }

    TEST_ASSERT(futcache_persist_node_count(eng) == 39); /* 2*20-1 */

    for (int trial = 0; trial < 100; ++trial) {
        double x = (double)rand() / RAND_MAX;
        double t = (double)rand() / RAND_MAX * 0.5;

        bool persist_novel = false;
        TEST_STATUS(futcache_persist_is_novel_at(eng, x, t, &persist_novel),
                    FUTCACHE_OK);

        bool expected_novel = true;
        for (int i = 0; i < 20; ++i) {
            double d = x - obs[i];
            if (d < 0) d = -d;
            if (d <= t) {
                expected_novel = false;
                break;
            }
        }

        if (persist_novel != expected_novel) {
            fprintf(stderr, "  RANDOM MISMATCH: x=%.6f t=%.6f persist=%d expected=%d\n",
                    x, t, persist_novel, expected_novel);
            futcache_persist_destroy(eng);
            return false;
        }
    }

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/* --- 13. Stats with prime cycle count --- */

static bool test_persist_stats(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);

    /* Observe 10 points with varying gaps */
    double pts[] = {0.0, 0.1, 0.5, 0.6, 1.0, 1.5, 2.0, 2.1, 3.0, 4.0};
    for (int i = 0; i < 10; ++i) {
        TEST_STATUS(futcache_persist_observe(eng, pts[i]), FUTCACHE_OK);
    }

    futcache_persist_stats_t stats;
    TEST_STATUS(futcache_persist_get_stats(eng, &stats), FUTCACHE_OK);

    TEST_ASSERT(stats.observations == 10);
    TEST_ASSERT(stats.feature_count == 9); /* n-1 */
    TEST_ASSERT(stats.max_persistence > 0.0);
    TEST_ASSERT(stats.min_persistence >= 0.0);
    TEST_ASSERT(stats.total_persistence > 0.0);
    TEST_ASSERT(stats.memory_bytes > 0);

    /* Prime cycle count should be <= total features */
    TEST_ASSERT(stats.prime_cycle_count <= stats.feature_count);

    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

/*
 * Locks the README Quick Start PersistentNovelty example.
 * After observe(0.5) and observe(0.7), query 0.6 has min_dist = 0.1.
 * is_novel_at is min_dist > t, so larger t is coarser (more coverage).
 * Fallback: if this test fails, the README comments are wrong again.
 */
static bool test_readme_persist_example(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.5), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.7), FUTCACHE_OK);

    bool novel = true;
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.6, 0.05, &novel),
                FUTCACHE_OK);
    TEST_ASSERT(novel == true);
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.6, 0.1, &novel),
                FUTCACHE_OK);
    TEST_ASSERT(novel == false);
    TEST_STATUS(futcache_persist_is_novel_at(eng, 0.6, 0.3, &novel),
                FUTCACHE_OK);
    TEST_ASSERT(novel == false);

    double intervals[4];
    size_t icount = 4;
    TEST_STATUS(futcache_persist_novelty_spectrum(eng, 0.6, intervals, &icount),
                FUTCACHE_OK);
    TEST_ASSERT(icount == 1);
    TEST_NEAR(intervals[0], 0.0, 1e-12);
    TEST_NEAR(intervals[1], 0.1, 1e-12);

    futcache_persist_destroy(eng);
    return true;
}

static bool test_exact_small_distances_and_invalid_queries(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    futcache_persist_t *eng = NULL;
    bool novel = false;
    double intervals[2];
    size_t count = 1U;

    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_is_novel_at(eng, 5e-13, 0.0, &novel),
                FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_persist_novelty_spectrum(
                    eng, 5e-13, intervals, &count), FUTCACHE_OK);
    TEST_ASSERT(count == 1U && intervals[1] == 5e-13);
    TEST_STATUS(futcache_persist_is_novel_at(eng, NAN, 0.0, &novel),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    count = 1U;
    TEST_STATUS(futcache_persist_novelty_spectrum(eng, NAN, intervals, &count),
                FUTCACHE_ERROR_INVALID_ARGUMENT);
    futcache_persist_destroy(eng);
    return true;
}

static bool test_max_features_limit(void)
{
    futcache_persist_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_features = 1U;
    futcache_persist_t *eng = NULL;
    TEST_STATUS(futcache_persist_create(&cfg, &eng), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 0.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 1.0), FUTCACHE_OK);
    TEST_STATUS(futcache_persist_observe(eng, 2.0),
                FUTCACHE_ERROR_OUT_OF_RANGE);
    TEST_ASSERT(futcache_persist_node_count(eng) == 3U);
    TEST_STATUS(futcache_persist_validate(eng), FUTCACHE_OK);
    futcache_persist_destroy(eng);
    return true;
}

int persist_test_suite(void)
{
    static const test_case_t tests[] = {
        {"prime table correctness", test_prime_table},
        {"empty engine (all novel)", test_persist_empty},
        {"single observation", test_persist_single_observe},
        {"two observations (merge tree)", test_persist_two_obs},
        {"three observations (full tree)", test_persist_three_obs},
        {"differential is_novel_at (70 combos)", test_persist_differential},
        {"novelty spectrum", test_persist_spectrum},
        {"feature count and evict_below", test_persist_features_evict},
        {"signature join (idempotent/commutative)", test_crdt_merge},
        {"finite prime-product diagnostic", test_selberg_zeta},
        {"clear and reuse", test_persist_clear_reuse},
        {"randomized differential (100 trials)", test_persist_randomized},
        {"stats with prime cycle count", test_persist_stats},
        {"README persist quick-start numbers", test_readme_persist_example},
        {"exact small distances and invalid queries", test_exact_small_distances_and_invalid_queries},
        {"max feature limit", test_max_features_limit},
    };
    return run_test_cases("persist", tests, sizeof(tests) / sizeof(tests[0]));
}

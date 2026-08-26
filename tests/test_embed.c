#include "test.h"
#include "futcache/embed.h"
#include "futcache/crdt.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Embedding tests (Design Sketch 04 — Learned Metric Layer)
 * ============================================================ */

/* 2D L2 distance (matches futcache_distance_l2 signature). */
static double test_dist_l2(const double *a, const double *b,
                            size_t dim, void *ctx)
{
    (void)ctx;
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

/* --- 1. Basic embedding correctness --- */

static bool test_embed_point_correctness(void)
{
    /* 1D domain [0,1], anchors at 0.0 and 1.0 (2 anchors), L1 distance.
     * phi(x) = (|x-0|, |x-1|) = (x, 1-x) for x in [0,1]. */
    double anchors[2] = {0.0, 1.0};
    double lo[1] = {0.0}, hi[1] = {1.0};

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 1;
    cfg.anchor_count = 2;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    /* phi(0.3) = (0.3, 0.7) */
    double pt[1] = {0.3};
    double out[2];
    TEST_STATUS(futcache_embed_point(emb, pt, out), FUTCACHE_OK);
    TEST_NEAR(out[0], 0.3, 1e-15);
    TEST_NEAR(out[1], 0.7, 1e-15);

    /* phi(0.9) = (0.9, 0.1) */
    pt[0] = 0.9;
    TEST_STATUS(futcache_embed_point(emb, pt, out), FUTCACHE_OK);
    TEST_NEAR(out[0], 0.9, 1e-15);
    TEST_NEAR(out[1], 0.1, 1e-15);

    /* phi(0.0) = (0.0, 1.0) */
    pt[0] = 0.0;
    TEST_STATUS(futcache_embed_point(emb, pt, out), FUTCACHE_OK);
    TEST_NEAR(out[0], 0.0, 1e-15);
    TEST_NEAR(out[1], 1.0, 1e-15);

    /* Accessors */
    TEST_ASSERT(futcache_embed_anchor_count(emb) == 2);
    TEST_ASSERT(futcache_embed_original_dimension(emb) == 1);

    /* Covering radius for anchors {0, 1} in [0,1] under L1:
     * sup_x min(x, 1-x) = 0.5 (worst point is x=0.5) */
    TEST_ASSERT(futcache_embed_covering_radius(emb) > 0.4);
    TEST_ASSERT(futcache_embed_covering_radius(emb) <= 0.51);

    futcache_embed_destroy(emb);
    return true;
}

/* --- 2. 1-Lipschitz property --- */

static bool test_embed_1_lipschitz(void)
{
    /* For any x, y: |phi(x)_i - phi(y)_i| <= d(x, y) for all i.
     * Equivalently: ||phi(x) - phi(y)||_inf <= d(x, y). */
    double anchors[8] = {0.1, 0.1, 0.4, 0.4, 0.7, 0.7, 0.9, 0.9};
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 2;
    cfg.anchor_count = 4;
    cfg.anchors = anchors;
    cfg.distance = test_dist_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    /* Test 100 random pairs */
    unsigned int seed = 12345;
    for (int trial = 0; trial < 100; trial++) {
        double x[2], y[2];
        x[0] = (double)rand_r(&seed) / RAND_MAX;
        x[1] = (double)rand_r(&seed) / RAND_MAX;
        y[0] = (double)rand_r(&seed) / RAND_MAX;
        y[1] = (double)rand_r(&seed) / RAND_MAX;

        double px[4], py[4];
        TEST_STATUS(futcache_embed_point(emb, x, px), FUTCACHE_OK);
        TEST_STATUS(futcache_embed_point(emb, y, py), FUTCACHE_OK);

        /* L_inf of embedded difference */
        double emb_linf = 0.0;
        for (int i = 0; i < 4; i++) {
            double d = fabs(px[i] - py[i]);
            if (d > emb_linf) emb_linf = d;
        }

        /* Original L2 distance */
        double orig = test_dist_l2(x, y, 2, NULL);

        /* 1-Lipschitz: emb_linf <= orig + numerical noise */
        TEST_ASSERT(emb_linf <= orig + 1e-12);
    }

    futcache_embed_destroy(emb);
    return true;
}

/* --- 3. Additive distortion bound |d(x,y) - ||phi(x)-phi(y)||_inf| <= 2*delta --- */

static bool test_embed_distortion_bound(void)
{
    /* Dense grid of anchors in 1D [0,1]: anchors at 0, 0.25, 0.5, 0.75, 1.0
     * Covering radius delta = 0.125 (max distance from any point to nearest anchor)
     * So |d(x,y) - ||phi(x)-phi(y)||_inf| <= 2*0.125 = 0.25 */
    double anchors[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    double lo[1] = {0.0}, hi[1] = {1.0};

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 1;
    cfg.anchor_count = 5;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    double delta = futcache_embed_covering_radius(emb);
    TEST_ASSERT(delta > 0.0);
    TEST_ASSERT(delta <= 0.126); /* should be ~0.125 */

    /* Test 200 random pairs */
    unsigned int seed = 67890;
    for (int trial = 0; trial < 200; trial++) {
        double x[1] = {(double)rand_r(&seed) / RAND_MAX};
        double y[1] = {(double)rand_r(&seed) / RAND_MAX};

        double px[5], py[5];
        TEST_STATUS(futcache_embed_point(emb, x, px), FUTCACHE_OK);
        TEST_STATUS(futcache_embed_point(emb, y, py), FUTCACHE_OK);

        double emb_linf = 0.0;
        for (int i = 0; i < 5; i++) {
            double d = fabs(px[i] - py[i]);
            if (d > emb_linf) emb_linf = d;
        }

        double orig = fabs(x[0] - y[0]); /* L1 in 1D */
        double gap = fabs(orig - emb_linf);

        /* |d(x,y) - ||phi(x)-phi(y)||_inf| <= 2*delta + numerical */
        TEST_ASSERT(gap <= 2.0 * delta + 1e-12);
    }

    futcache_embed_destroy(emb);
    return true;
}

/* --- 4. Adjusted epsilon --- */

static bool test_embed_adjusted_epsilon(void)
{
    double anchors[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    double lo[1] = {0.0}, hi[1] = {1.0};

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 1;
    cfg.anchor_count = 5;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    /* epsilon_original = 0.5, delta ~ 0.125
     * adjusted = 0.5 - 2*0.125 = 0.25 */
    double adjusted;
    TEST_STATUS(futcache_embed_adjusted_epsilon(emb, 0.5, &adjusted),
                FUTCACHE_OK);
    TEST_ASSERT(adjusted > 0.24);
    TEST_ASSERT(adjusted < 0.26);

    /* epsilon_original too small: 0.2 < 2*delta ~ 0.25 -> OUT_OF_RANGE */
    TEST_STATUS(futcache_embed_adjusted_epsilon(emb, 0.2, &adjusted),
                FUTCACHE_ERROR_OUT_OF_RANGE);

    futcache_embed_destroy(emb);
    return true;
}

/* --- 5. Embed + VP-tree end-to-end --- */

static bool test_embed_pack_e2e(void)
{
    /* 2D L2 space, grid of 4 anchors, VP-tree in embedded space.
     * Points that are close in original space should be close in embedded
     * space and vice versa (within distortion 2*delta). */
    /* 9 anchors in a 3x3 grid — dense enough that covering radius is small
     * so epsilon=0.3 survives the 2*delta adjustment. */
    double anchors[18] = {
        0.0, 0.0, 0.5, 0.0, 1.0, 0.0,
        0.0, 0.5, 0.5, 0.5, 1.0, 0.5,
        0.0, 1.0, 0.5, 1.0, 1.0, 1.0
    };
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};

    /* Embed a few points */
    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 2;
    cfg.anchor_count = 9;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    size_t m = futcache_embed_anchor_count(emb);
    TEST_ASSERT(m == 9);

    double delta = futcache_embed_covering_radius(emb);
    (void)delta;

    /* Original-space epsilon: 0.8 (must exceed 2*delta ≈ 0.71 for 3x3 grid) */
    double eps_orig = 0.8;
    double eps_emb;
    TEST_STATUS(futcache_embed_adjusted_epsilon(emb, eps_orig, &eps_emb),
                FUTCACHE_OK);
    TEST_ASSERT(eps_emb > 0.0);

    /* Create pack cache in embedded space with VP-tree */
    double *emb_lo = (double *)calloc(m, sizeof(double));
    double *emb_hi = (double *)malloc(m * sizeof(double));
    TEST_ASSERT(emb_lo != NULL && emb_hi != NULL);
    for (size_t i = 0; i < m; i++) {
        emb_lo[i] = 0.0;
        emb_hi[i] = 1.5; /* max possible L2 distance in unit square ~ 1.414 */
    }

    futcache_pack_config_t pcfg;
    futcache_pack_config_init(&pcfg);
    pcfg.dimension = m;
    pcfg.epsilon = eps_emb;
    pcfg.distance = futcache_embed_distance;
    pcfg.distance_context = emb;
    pcfg.domain_min = emb_lo;
    pcfg.domain_max = emb_hi;
    pcfg.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *cache = NULL;
    TEST_STATUS(futcache_pack_create(&pcfg, &cache), FUTCACHE_OK);

    /* Insert 3 well-separated points in original space */
    double pts[3][2] = {{0.1, 0.1}, {0.5, 0.5}, {0.9, 0.9}};
    bool novel;
    for (int i = 0; i < 3; i++) {
        double emb_pt[9];
        TEST_STATUS(futcache_embed_point(emb, pts[i], emb_pt), FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe(cache, emb_pt, &novel), FUTCACHE_OK);
        TEST_ASSERT(novel);
    }

    /* Query a point near (0.1, 0.1) — should NOT be novel */
    double query[2] = {0.12, 0.08};
    double query_emb[9];
    TEST_STATUS(futcache_embed_point(emb, query, query_emb), FUTCACHE_OK);
    bool is_novel;
    TEST_STATUS(futcache_pack_is_novel(cache, query_emb, &is_novel),
                FUTCACHE_OK);
    TEST_ASSERT(!is_novel);

    /* Query a far point — should be novel */
    query[0] = 0.5; query[1] = 0.9;
    TEST_STATUS(futcache_embed_point(emb, query, query_emb), FUTCACHE_OK);
    TEST_STATUS(futcache_pack_is_novel(cache, query_emb, &is_novel),
                FUTCACHE_OK);
    TEST_ASSERT(is_novel);

    free(emb_lo);
    free(emb_hi);
    futcache_pack_destroy(cache);
    futcache_embed_destroy(emb);
    return true;
}

/* --- 6. Anchor generation integration --- */

static bool test_embed_anchor_generation(void)
{
    /* Use futcache_crdt_generate_safe_anchors to build anchors,
     * then wrap in an embedding. */
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};
    double *anchors = (double *)malloc(10000 * 2 * sizeof(double));
    TEST_ASSERT(anchors != NULL);

    size_t anchor_count = 0;
    double covering_radius = 0.0;
    futcache_status_t st = futcache_crdt_generate_safe_anchors(
        2, 0.3, lo, hi, futcache_distance_l2, NULL,
        FUTCACHE_CRDT_ANCHOR_GRID, 10000, 1000,
        anchors, &anchor_count, &covering_radius);
    TEST_STATUS(st, FUTCACHE_OK);
    TEST_ASSERT(anchor_count >= 4);
    TEST_ASSERT(covering_radius > 0.0);

    /* Create embedding from generated anchors */
    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 2;
    cfg.anchor_count = anchor_count;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);
    TEST_ASSERT(futcache_embed_anchor_count(emb) == anchor_count);

    /* Verify distortion bound on a sample */
    double delta = futcache_embed_covering_radius(emb);
    double x[2] = {0.3, 0.7}, y[2] = {0.6, 0.2};
    double px[10000], py[10000];
    TEST_STATUS(futcache_embed_point(emb, x, px), FUTCACHE_OK);
    TEST_STATUS(futcache_embed_point(emb, y, py), FUTCACHE_OK);

    double emb_linf = 0.0;
    for (size_t i = 0; i < anchor_count; i++) {
        double d = fabs(px[i] - py[i]);
        if (d > emb_linf) emb_linf = d;
    }
    double orig = test_dist_l2(x, y, 2, NULL);
    double gap = fabs(orig - emb_linf);
    TEST_ASSERT(gap <= 2.0 * delta + 1e-12);

    futcache_embed_destroy(emb);
    free(anchors);
    return true;
}

/* --- 7. Validation of embedded pack cache --- */

static bool test_embed_pack_validate(void)
{
    double anchors[3] = {0.2, 0.5, 0.8};
    double lo[1] = {0.0}, hi[1] = {1.0};

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = 1;
    cfg.anchor_count = 3;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);

    size_t m = futcache_embed_anchor_count(emb);
    double eps_orig = 0.4;
    double eps_emb;
    TEST_STATUS(futcache_embed_adjusted_epsilon(emb, eps_orig, &eps_emb),
                FUTCACHE_OK);

    double *emb_lo = (double *)calloc(m, sizeof(double));
    double *emb_hi = (double *)calloc(m, sizeof(double));
    for (size_t i = 0; i < m; i++) {
        emb_lo[i] = 0.0;
        emb_hi[i] = 1.0;
    }

    futcache_pack_config_t pcfg;
    futcache_pack_config_init(&pcfg);
    pcfg.dimension = m;
    pcfg.epsilon = eps_emb;
    pcfg.distance = futcache_embed_distance;
    pcfg.distance_context = emb;
    pcfg.domain_min = emb_lo;
    pcfg.domain_max = emb_hi;

    futcache_pack_t *cache = NULL;
    TEST_STATUS(futcache_pack_create(&pcfg, &cache), FUTCACHE_OK);

    /* Insert points and validate */
    double pts[] = {0.1, 0.5, 0.9};
    bool novel;
    for (int i = 0; i < 3; i++) {
        double emb_pt[3];
        TEST_STATUS(futcache_embed_point(emb, &pts[i], emb_pt), FUTCACHE_OK);
        TEST_STATUS(futcache_pack_observe(cache, emb_pt, &novel), FUTCACHE_OK);
    }

    TEST_STATUS(futcache_pack_validate(cache), FUTCACHE_OK);

    free(emb_lo);
    free(emb_hi);
    futcache_pack_destroy(cache);
    futcache_embed_destroy(emb);
    return true;
}

/* --- 8. High-dimensional embedding (384D) --- */

static bool test_embed_high_dimensional(void)
{
    /* 384D L2 space (typical sentence embedding dimension),
     * 100 Halton anchors. Verify embedding works and distortion bound holds. */
    size_t dim = 384;
    size_t n_anchors = 100;

    double lo[384], hi[384];
    for (int i = 0; i < 384; i++) { lo[i] = -1.0; hi[i] = 1.0; }

    double *anchors = (double *)malloc(n_anchors * dim * sizeof(double));
    TEST_ASSERT(anchors != NULL);
    TEST_STATUS(futcache_crdt_generate_halton_anchors(
        dim, lo, hi, n_anchors, anchors), FUTCACHE_OK);

    futcache_embed_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = dim;
    cfg.anchor_count = n_anchors;
    cfg.anchors = anchors;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_embed_t *emb = NULL;
    TEST_STATUS(futcache_embed_create(&cfg, &emb), FUTCACHE_OK);
    TEST_ASSERT(futcache_embed_anchor_count(emb) == n_anchors);
    TEST_ASSERT(futcache_embed_original_dimension(emb) == dim);

    double delta = futcache_embed_covering_radius(emb);
    TEST_ASSERT(delta > 0.0);

    /* Embed a point */
    double pt[384], emb_pt[100];
    for (int i = 0; i < 384; i++) pt[i] = 0.0;
    TEST_STATUS(futcache_embed_point(emb, pt, emb_pt), FUTCACHE_OK);

    /* All embedded coordinates should be finite and >= 0 */
    for (int i = 0; i < 100; i++) {
        TEST_ASSERT(isfinite(emb_pt[i]));
        TEST_ASSERT(emb_pt[i] >= 0.0);
    }

    /* Distortion check: two points close in L2 should be close in embedding */
    double p1[384], p2[384];
    for (int i = 0; i < 384; i++) {
        p1[i] = 0.0;
        p2[i] = 0.001; /* tiny perturbation */
    }
    double e1[100], e2[100];
    TEST_STATUS(futcache_embed_point(emb, p1, e1), FUTCACHE_OK);
    TEST_STATUS(futcache_embed_point(emb, p2, e2), FUTCACHE_OK);

    double emb_linf = 0.0;
    for (int i = 0; i < 100; i++) {
        double d = fabs(e1[i] - e2[i]);
        if (d > emb_linf) emb_linf = d;
    }
    /* 1-Lipschitz: emb_linf <= ||p1-p2||_2 = sqrt(384)*0.001 ~ 0.0196 */
    double orig = sqrt(384.0) * 0.001;
    TEST_ASSERT(emb_linf <= orig + 1e-9);

    futcache_embed_destroy(emb);
    free(anchors);
    return true;
}

int embed_test_suite(void)
{
    static const test_case_t tests[] = {
        {"embedding correctness (1D L1)", test_embed_point_correctness},
        {"1-Lipschitz property (2D L2)", test_embed_1_lipschitz},
        {"additive distortion bound (1D dense grid)",
            test_embed_distortion_bound},
        {"adjusted epsilon", test_embed_adjusted_epsilon},
        {"end-to-end embed + VP-tree pack", test_embed_pack_e2e},
        {"anchor generation integration", test_embed_anchor_generation},
        {"embedded pack validation", test_embed_pack_validate},
        {"high-dimensional 384D embedding", test_embed_high_dimensional},
    };
    return run_test_cases("embed", tests, sizeof(tests) / sizeof(tests[0]));
}

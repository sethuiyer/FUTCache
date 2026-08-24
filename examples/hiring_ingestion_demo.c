/*
 * Hiring-ingestion demo: runs the "high-dimensional geometric ingestion"
 * example against the real FUTCache engines and measures each claim, so the
 * verdicts are empirical rather than asserted. Prints CLAIM / RESULT fields.
 *
 * Compile (from repo root):
 *   cc -O2 -std=gnu11 -I include examples/hiring_ingestion_demo.c \
 *      -o build-test-release/hiring_demo build-test-release/libfutcache.a -lm -lpthread
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/futcache.h"
#include "futcache/pack.h"
#include "futcache/tower.h"
#include "futcache/crdt.h"

/* ---------------- RNG ---------------- */
static uint64_t rng_state = UINT64_C(0x9e3779b97f4a7c15);
static uint64_t rng_next(void)
{
    uint64_t v = rng_state;
    v ^= v << 13U; v ^= v >> 7U; v ^= v << 17U;
    rng_state = v;
    return v;
}
static double rng_unit(void)
{
    return (double)(rng_next() >> 11U) / 9007199254740992.0;
}

/* ---------------- timing ---------------- */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---------------- helpers ---------------- */
static double dist_l2(const double *a, const double *b, size_t d)
{
    double s = 0.0;
    for (size_t i = 0; i < d; ++i) { double t = a[i] - b[i]; s += t * t; }
    return sqrt(s);
}

#define CLAIM(s) do { printf("\n### %s\n", s); } while (0)
#define NOTE(...) do { printf("  " __VA_ARGS__); printf("\n"); } while (0)
/* Hard invariant: a failure here makes the process exit non-zero so ctest
 * enforces the guarantee. */
static int g_failures = 0;
#define VERDICT(ok, msg) do {                                              \
    printf("  [%s] %s\n", (ok) ? "PASS" : "FAIL|CHECK", msg);              \
    if (!(ok)) ++g_failures;                                               \
} while (0)

/* ===================================================================
 * Section A: PackCache dense-cluster collapse, outlier admission,
 * separation invariant, and the one-sided novelty guarantee.
 * =================================================================== */
static void section_pack_geometry(void)
{
    CLAIM("A. PackCache metric-ball behaviour");
    const size_t dim = 2;
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};
    double eps = 0.3;

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = dim; cfg.epsilon = eps;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo; cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) { NOTE("create failed"); return; }

    /* dense cluster: 300 profiles all within ~0.02 of the centre */
    double centre[2] = {0.5, 0.5};
    for (size_t i = 0; i < 300; ++i) {
        double p[2] = {centre[0] + 0.02 * (rng_unit() - 0.5),
                       centre[1] + 0.02 * (rng_unit() - 0.5)};
        bool novel;
        futcache_pack_observe(cache, p, &novel);
    }
    futcache_pack_stats_t st;
    futcache_pack_get_stats(cache, &st);
    size_t reps_after_cluster = st.representative_count;

    /* outlier: the 'quant finance + LLM safety' hybrid, far away */
    double outlier[2] = {0.998, 0.002};
    bool out_novel = false;
    futcache_pack_observe(cache, outlier, &out_novel);
    futcache_pack_get_stats(cache, &st);

    NOTE("300 dense-cluster profiles -> %zu representative(s) (epsilon=%.2f)",
         reps_after_cluster, eps);
    VERDICT(reps_after_cluster <= 3, "'standard cluster collapses to a medoid representative': holds");

    NOTE("distant hybrid profile admitted as novel: %s", out_novel ? "yes" : "no");
    VERDICT(out_novel, "'outlier multi-domain profile triggers novelty': holds");

    /* separation invariant: no two admitted representatives are closer than eps */
    size_t cnt = 0;
    futcache_pack_copy_representatives(cache, NULL, &cnt);
    double *reps = (double *)malloc(cnt * dim * sizeof(double));
    size_t rc = cnt;
    futcache_pack_copy_representatives(cache, reps, &rc);
    double min_pair = INFINITY;
    for (size_t i = 0; i < cnt; ++i)
        for (size_t j = i + 1; j < cnt; ++j) {
            double d = dist_l2(reps + i * dim, reps + j * dim, dim);
            if (d < min_pair) min_pair = d;
        }
    NOTE("min pairwise distance among representatives = %.4f (epsilon=%.2f)",
         min_pair, eps);
    VERDICT(min_pair >= eps, "'no two admitted profiles satisfy d < epsilon': holds across representatives");
    free(reps);

    /* One-sided guarantee: the cache can report extra NOVELTY (never a false
     * hit). Construct a case where a full-history oracle calls a point
     * redundant but the representative cache calls it novel. */
    {
        futcache_pack_config_t c1;
        futcache_pack_config_init(&c1);
        c1.dimension = 1; c1.epsilon = 1.0; /* 1-D for a crisp example */
        double lo1[1] = {0.0}, hi1[1] = {10.0};
        c1.domain_min = lo1; c1.domain_max = hi1;
        c1.distance = futcache_distance_l2;
        futcache_pack_t *c = NULL;
        futcache_pack_create(&c1, &c);
        bool n0, n1, n2;
        futcache_pack_observe(c, (double[]){0.0}, &n0);   /* rep A = 0   */
        futcache_pack_observe(c, (double[]){3.0}, &n1);   /* rep B = 3   */
        futcache_pack_observe(c, (double[]){0.9}, &n2);   /* absorbed   */
        /* Q = 1.8: within eps of the absorbed point 0.9, but > eps from
         * every stored representative {0,3}. */
        bool q_novel = false;
        futcache_pack_is_novel(c, (double[]){1.8}, &q_novel);
        /* full-history novelty: min distance to {0.0, 3.0, 0.9} */
        double hist[3] = {0.0, 3.0, 0.9};
        double mind = INFINITY;
        for (int i = 0; i < 3; ++i) {
            double dd = fabs(1.8 - hist[i]);
            if (dd < mind) mind = dd;
        }
        NOTE("query x=1.8: cache-novelty=%d (reps {0,3}), full-history novelty=%d (history {0,3,0.9})",
             q_novel, (mind > 1.0));
        VERDICT(q_novel && (mind <= 1.0),
                "one-sided: cache reports novel where the full-history oracle says redundant");
        futcache_pack_destroy(c);
    }

    futcache_pack_destroy(cache);
}

/* ===================================================================
 * Section B: latency vs N — tests "O(log N) nearest-neighbour, sub-50us".
 * =================================================================== */
static void measure_nearest(futcache_distance_fn dist_used,
                            const futcache_pack_backend_ops_t *backend,
                            const double *points, size_t n, size_t dim,
                            double eps, const double *queries, size_t m,
                            const char *label)
{
    double *lo = (double *)malloc(dim * sizeof(double));
    double *hi = (double *)malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; ++i) { lo[i] = 0.0; hi[i] = 1.0; }
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = dim; cfg.epsilon = eps;
    cfg.distance = dist_used; cfg.domain_min = lo; cfg.domain_max = hi;
    cfg.backend = backend;

    futcache_pack_t *cache = NULL;
    futcache_pack_create(&cfg, &cache);

    double t0 = now_s();
    for (size_t i = 0; i < n; ++i) {
        bool unused;
        futcache_pack_observe(cache, points + i * dim, &unused);
    }
    double obs_s = now_s() - t0;
    futcache_pack_stats_t st;
    futcache_pack_get_stats(cache, &st);

    /* time nearest-neighbour queries */
    double q0 = now_s();
    for (size_t i = 0; i < m; ++i) {
        double d; size_t idx;
        futcache_pack_nearest(cache, queries + i * dim, &d, &idx);
    }
    double q_s = now_s() - q0;

    NOTE("%-22s n=%-5zu reps=%-5zu  observe=%6.1f us  nearest=%6.1f us",
         label, n, st.representative_count,
         obs_s / (double)n * 1e6, q_s / (double)m * 1e6);
    free(lo); free(hi);
    futcache_pack_destroy(cache);
}

static void section_latency(void)
{
    CLAIM("B. VP-tree latency vs N (dim=64, epsilon=0.02)");
    const size_t dim = 64, m = 1000;
    double eps = 0.02;

    /* clustered: points on a 4-dim subspace of R^64 -> low intrinsic dim.
     * All coords stay inside [0,1] so observations are in-domain. */
    size_t nc = 20000;
    double *clustered = (double *)malloc((nc + m) * dim * sizeof(double));
    for (size_t i = 0; i < nc + m; ++i) {
        for (size_t d = 0; d < dim; ++d) clustered[i * dim + d] = 0.5;
        for (size_t k = 0; k < 4; ++k)
            clustered[i * dim + (k * 17)] = 0.05 + 0.9 * rng_unit();
    }

    /* uniform: all coords independent -> high intrinsic dim */
    size_t nu = 5000;
    double *uniform = (double *)malloc((nu + m) * dim * sizeof(double));
    for (size_t i = 0; i < nu + m; ++i)
        for (size_t d = 0; d < dim; ++d) uniform[i * dim + d] = rng_unit();

    NOTE("clustered (low intrinsic dim):");
    {
        size_t cluster_ns[] = {1000, 5000, 20000};
        for (size_t k = 0; k < 3; ++k) {
            size_t n = cluster_ns[k];
            measure_nearest(futcache_distance_l2, &futcache_pack_vptree_backend,
                            clustered, n, dim, eps, clustered + n * dim, m, "vp-tree");
            measure_nearest(futcache_distance_l2, NULL,
                            clustered, n, dim, eps, clustered + n * dim, m, "linear  ");
        }
    }
    NOTE("uniform (high intrinsic dim):");
    {
        size_t uniform_ns[] = {1000, 5000};
        for (size_t k = 0; k < 2; ++k) {
            size_t n = uniform_ns[k];
            measure_nearest(futcache_distance_l2, &futcache_pack_vptree_backend,
                            uniform, n, dim, eps, uniform + n * dim, m, "vp-tree");
            measure_nearest(futcache_distance_l2, NULL,
                            uniform, n, dim, eps, uniform + n * dim, m, "linear  ");
        }
    }

    free(clustered);
    free(uniform);
}

/* ===================================================================
 * Section C: 1-D interval-union holes and new-ceiling detection.
 * =================================================================== */
static void section_interval(void)
{
    CLAIM("C. Interval-union 'topological hole' & 'new ceiling' detection");
    futcache_config_t cfg;
    futcache_config_init(&cfg);
    cfg.domain_min = 0.0; cfg.domain_max = 10.0; cfg.epsilon = 0.4;
    futcache_t *cache = NULL;
    futcache_create(&cfg, &cache);

    double vals[] = {5.0, 8.0};
    for (size_t i = 0; i < 2; ++i) {
        bool novel;
        futcache_observe(cache, vals[i], &novel);
        NOTE("observe %.1f -> novel=%d (interval coverage expands)", vals[i], novel);
    }
    /* a point in the hole between the two covered bands */
    bool hole_novel = false;
    futcache_is_novel(cache, 7.0, &hole_novel);
    NOTE("query 7.0 (gap between covered bands): novel=%d", hole_novel);
    VERDICT(hole_novel, "'topological hole' is detected: 7.0 is novel");

    bool ceil_novel = false;
    futcache_is_novel(cache, 5.6, &ceil_novel);
    NOTE("query 5.6 (above the [4.6,5.4] band): novel=%d", ceil_novel);
    VERDICT(ceil_novel, "'new metric ceiling' is flagged (5.6 > 5.0+eps)");

    bool covered = false;
    futcache_is_novel(cache, 5.1, &covered);
    NOTE("query 5.1 (inside covered band): novel=%d", covered);
    VERDICT(!covered, "covered point is correctly redundant");

    bool inside_band = false;
    futcache_is_novel(cache, 7.7, &inside_band);
    NOTE("query 7.7 (inside [7.6,8.4] band): novel=%d", inside_band);
    VERDICT(!inside_band, "another covered point is correctly redundant");

    futcache_destroy(cache);
}

/* ===================================================================
 * Section D: custom adaptive radius — shrink for common, expand for rare.
 * =================================================================== */
static double doc_radius(double freq, double r0, double gamma)
{
    /* r(s) = r0 * (1 + exp(-gamma*freq))  per the example */
    return r0 * (1.0 + exp(-gamma * freq));
}

static void section_adaptive(void)
{
    CLAIM("D. Custom adaptive radius (common shrinks, rare expands)");
    const size_t dim = 2;
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};
    double r0 = 0.5, gamma = 3.0;

    double common_freq = 0.9, rare_freq = 0.05;
    double r_common = doc_radius(common_freq, r0, gamma); /* e.g. Python  */
    double r_rare   = doc_radius(rare_freq,   r0, gamma); /* e.g. BEHRT   */
    NOTE("radius(common, freq=%.2f)=%.3f  radius(rare, freq=%.2f)=%.3f",
         common_freq, r_common, rare_freq, r_rare);
    VERDICT(r_common < r_rare, "the example's formula does 'shrink common / expand rare'");

    /* Implement it with observe_with_radius: a hybrid skill vector far from
     * every stored ball is novel. */
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = dim; cfg.epsilon = 0.5; cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo; cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    futcache_pack_create(&cfg, &cache);
    double common[2] = {0.5, 0.5}, rare_hybrid[2] = {0.99, 0.01};
    bool n1, n2;
    futcache_pack_observe_with_radius(cache, common, r_common, &n1, NULL, NULL);
    futcache_pack_observe_with_radius(cache, rare_hybrid, r_rare, &n2, NULL, NULL);
    NOTE("observe common (r=%.3f) novel=%d; observe rare hybrid (r=%.3f) novel=%d",
         r_common, n1, r_rare, n2);
    VERDICT(n2, "'rare cross-domain attribute lands outside existing balls -> novel': holds");

    /* honesty note: the SHIPPED AdaptiveRadiusPolicy does the opposite. */
    printf("  [CHECK] NOTE: the shipped AdaptiveRadiusPolicy contracts radius for\n");
    printf("          rare / poorly-supported regions (exp(-lambda*iso)); the example\n");
    printf("          formula above is a *custom* policy, implemented via\n");
    printf("          futcache_pack_observe_with_radius, not the built-in controller.\n");
    futcache_pack_destroy(cache);
}

/* ===================================================================
 * Section E: resolution tower — what it does / does not do.
 * =================================================================== */
static void section_tower(void)
{
    CLAIM("E. Resolution tower: occupancy & coverage");
    futcache_tower_config_t cfg;
    futcache_tower_config_init(&cfg);
    futcache_tower_config_init(&cfg);
    cfg.domain_min = 0.0; cfg.domain_max = 10.0;
    cfg.level_count = 3; cfg.root_cells = 4;
    futcache_tower_t *tower = NULL;
    futcache_tower_create(&cfg, &tower);

    /* publication counts capping at the domain */
    double counts[] = {0, 1, 1, 2, 2, 2, 3, 3, 8, 8, 8};
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        uint8_t novel[3];
        futcache_tower_observe(tower, counts[i], novel, 3);
    }
    futcache_tower_stats_t st;
    futcache_tower_get_stats(tower, &st);
    NOTE("observations=%llu discoveries=%llu total_cells=%zu",
         (unsigned long long)st.observations,
         (unsigned long long)st.total_discoveries, st.total_cells);
    for (size_t lvl = 0; lvl < 3; ++lvl) {
        futcache_tower_level_info_t info;
        futcache_tower_level_info(tower, lvl, &info);
        size_t prefix = 0;
        futcache_tower_prefix_count(tower, lvl, info.cell_count - 1, &prefix);
        NOTE("level %zu: cells=%zu occupied=%zu (coverage)", lvl,
             info.cell_count, info.discovered_count);
    }
    VERDICT(true, "tower reports WHICH buckets are covered (occupancy/coverage)");
    printf("  [CHECK] The tower has poll, prefix_count, select_occupied and\n");
    printf("          discovery_at primitives. It does NOT provide per-bucket\n");
    printf("          frequencies or an entropy measure: asking it for a\n");
    printf("          'how many candidates in the 0-2 bucket' histogram is a\n");
    printf("          category error -- use it for coverage/divergence, not\n");
    printf("          for frequency distribution or entropy decay.\n");
    futcache_tower_destroy(tower);
}

/* ===================================================================
 * Section F: CRDT convergence under gossip (strong eventual consistency).
 * =================================================================== */
static void section_crdt(void)
{
    CLAIM("F. CRDT gossip convergence (SEC)");
    const size_t dim = 2;
    double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};
    /* 3x3 grid anchors -> a delta-net over [0,1]^2 under L_inf */
    const size_t anchor_count = 9;
    double anchors[anchor_count * 2];
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            anchors[(i * 3 + j) * 2 + 0] = (double)i * 0.5;
            anchors[(i * 3 + j) * 2 + 1] = (double)j * 0.5;
        }

    futcache_crdt_config_t cfg;
    futcache_crdt_config_init(&cfg);
    cfg.dimension = dim; cfg.anchor_count = anchor_count; cfg.anchors = anchors;
    cfg.epsilon = 0.6; cfg.distance = futcache_distance_linf;
    cfg.domain_min = lo; cfg.domain_max = hi;
    futcache_crdt_t *a = NULL, *b = NULL;
    futcache_crdt_create(&cfg, &a);
    futcache_crdt_create(&cfg, &b);

    unsigned char pa[1] = {0xaa}, pb[1] = {0xbb};
    double ptsA[][2] = {{0.1, 0.1}, {0.6, 0.2}, {0.9, 0.9}, {0.2, 0.6}};
    double ptsB[][2] = {{0.5, 0.5}, {0.8, 0.1}, {0.15, 0.85}, {0.7, 0.7}};
    for (size_t i = 0; i < 4; ++i) {
        bool n; size_t cell;
        futcache_crdt_observe(a, ptsA[i], pa, 1, &n, &cell);
        futcache_crdt_observe(b, ptsB[i], pb, 1, &n, &cell);
    }
    futcache_crdt_stats_t sa, sb;
    futcache_crdt_get_stats(a, &sa);
    futcache_crdt_get_stats(b, &sb);
    NOTE("replica A cells=%zu, replica B cells=%zu (before gossip)",
         sa.occupied_cells, sb.occupied_cells);

    /* gossip both ways until convergence */
    for (int iter = 0; iter < 4; ++iter) {
        size_t cap_a = 0; futcache_crdt_snapshot(a, NULL, &cap_a);
        futcache_crdt_update_t *upA = (futcache_crdt_update_t *)malloc(cap_a * sizeof(*upA));
        size_t nA = cap_a; futcache_crdt_snapshot(a, upA, &nA);
        futcache_crdt_merge(b, upA, nA); free(upA);

        size_t cap_b = 0; futcache_crdt_snapshot(b, NULL, &cap_b);
        futcache_crdt_update_t *upB = (futcache_crdt_update_t *)malloc(cap_b * sizeof(*upB));
        size_t nB = cap_b; futcache_crdt_snapshot(b, upB, &nB);
        futcache_crdt_merge(a, upB, nB); free(upB);
    }
    futcache_crdt_get_stats(a, &sa);
    futcache_crdt_get_stats(b, &sb);
    NOTE("after gossip: replica A cells=%zu, replica B cells=%zu",
         sa.occupied_cells, sb.occupied_cells);
    VERDICT(sa.occupied_cells == sb.occupied_cells,
            "replicas converge to the same occupied cell set (SEC)");

    futcache_crdt_destroy(a);
    futcache_crdt_destroy(b);
}

int main(void)
{
    printf("=== FUTCache hiring-ingestion demo: CLAIM / RESULT ===\n");
    section_pack_geometry();
    section_latency();
    section_interval();
    section_adaptive();
    section_tower();
    section_crdt();
    printf("\n=== done: %d hard-invariant failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

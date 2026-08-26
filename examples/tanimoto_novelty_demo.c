/* tanimoto_novelty_demo.c — validate the drug-discovery novelty-screening pitch.
 *
 * Checks, with real runs:
 *   1. Tanimoto distance is a METRIC (numeric triangle-inequality check).
 *   2. The packing cache's novelty gate on synthetic 2048-bit Morgan-style
 *      fingerprints: does epsilon=0.15 catch "near-duplicate" derivatives
 *      (1-3 bit flips of a parent scaffold) as redundant while admitting truly
 *      novel scaffolds?
 *   3. The VP-tree vs linear-scan latency at the resulting representative set
 *      -- the pitch claims "2ms linear -> ~200us VP-tree", which we test.
 *
 * Build (from repo root):
 *   cc -O2 -std=gnu11 -I include examples/tanimoto_novelty_demo.c \
 *      -o build-test-release/tanimoto_demo build-test-release/libfutcache.a -lm -lpthread
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/pack.h"

enum { DIM = 2048, SCAFFOLDS = 200, DERIVS = 20, NOVELS = 200 };
/* total = SCAFFOLDS*(1+DERIVS) + NOVELS */
#define TOTAL (SCAFFOLDS * (1 + DERIVS) + NOVELS)

static uint64_t rng_state = UINT64_C(0x5eed55aa12345678);
static uint64_t rng_next(void)
{
    uint64_t v = rng_state;
    v ^= v << 13U; v ^= v >> 7U; v ^= v << 17U;
    rng_state = v;
    return v;
}
static uint32_t rng_u32(void) { return (uint32_t)(rng_next() & 0xffffffffU); }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Tanimoto distance = 1 - |A & B| / |A | B|  (a genuine metric on bit-sets). */
static double tanimoto(const double *a, const double *b, size_t d, void *ctx)
{
    (void)ctx;
    double inter = 0.0, uni = 0.0;
    for (size_t i = 0; i < d; ++i) {
        bool x = a[i] > 0.5, y = b[i] > 0.5;
        if (x && y) inter += 1.0;
        if (x || y) uni += 1.0;
    }
    return uni > 0.0 ? 1.0 - inter / uni : 0.0;
}

static void set_random_bits(double *fp, int nbits)
{
    for (int b = 0; b < nbits; ++b) fp[rng_u32() % DIM] = 1.0;
}
static void flip_few(double *fp, int flips)
{
    for (int b = 0; b < flips; ++b) fp[rng_u32() % DIM] = fp[rng_u32() % DIM] > 0.5 ? 0.0 : 1.0;
}

static double dist(const double *a, const double *b) { return tanimoto(a, b, DIM, NULL); }

int main(void)
{
    double *lo = (double *)malloc(DIM * sizeof(double));
    double *hi = (double *)malloc(DIM * sizeof(double));
    for (size_t i = 0; i < DIM; ++i) { lo[i] = 0.0; hi[i] = 1.0; }

    /* --- check 1: is tanimoto a metric (triangle inequality on randoms)? --- */
    int tri_fail = 0;
    for (int t = 0; t < 2000; ++t) {
        double a[DIM] = {0}, b[DIM] = {0}, c[DIM] = {0};
        set_random_bits(a, 40); set_random_bits(b, 40); set_random_bits(c, 40);
        if (dist(a, c) > dist(a, b) + dist(b, c) + 1e-9) tri_fail++;
    }
    printf("1) Tanimoto triangle-inequality    : %s (%d/2000 violations)\n",
           tri_fail == 0 ? "PASS (is a metric)" : "FAIL", tri_fail);

    /* --- build the stream: scaffolds + derivatives + some truly novel --- */
    double *pts = (double *)calloc(TOTAL, DIM * sizeof(double));
    size_t n = 0;
    double *parents = (double *)calloc(SCAFFOLDS, DIM * sizeof(double));
    for (int s = 0; s < SCAFFOLDS; ++s) {
        double *p = parents + (size_t)s * DIM;
        set_random_bits(p, 40);
        memcpy(pts + (size_t)n++ * DIM, p, DIM * sizeof(double));
        for (int d = 0; d < DERIVS; ++d) {          /* near-duplicate */
            memcpy(pts + (size_t)n * DIM, p, DIM * sizeof(double));
            flip_few(pts + (size_t)n * DIM, 1 + (rng_u32() % 3)); /* 1-3 flips */
            n++;
        }
    }
    for (int v = 0; v < NOVELS; ++v) {              /* genuinely different */
        set_random_bits(pts + (size_t)n++ * DIM, 40);
    }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = DIM;
    cfg.epsilon = 0.15;
    cfg.distance = tanimoto;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    cfg.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *vp = NULL, *lin = NULL;
    futcache_pack_create(&cfg, &vp);
    cfg.backend = NULL;
    futcache_pack_create(&cfg, &lin);

    double t0 = now_s();
    size_t novel_cnt = 0, red_cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        bool was_novel; futcache_pack_observe(vp, pts + i * DIM, &was_novel);
        if (was_novel) novel_cnt++; else red_cnt++;
    }
    double obs_s = now_s() - t0;

    futcache_pack_stats_t st;
    futcache_pack_get_stats(vp, &st);

    printf("2) Novelty gate (epsilon=0.15)        : %zu novel / %zu redundant "
           "across %zu candidates\n", novel_cnt, red_cnt, n);
    printf("   representatives retained: %zu   dedup ratio: %.1fx\n",
           st.representative_count, (double)n / (double)st.representative_count);

    /* --- check 3: VP-tree vs linear latency at the rep set --- */
    t0 = now_s();
    for (size_t i = 0; i < n; ++i) {
        bool unused; futcache_pack_observe(lin, pts + i * DIM, &unused);
    }
    double lin_obs_s = now_s() - t0;

    /* query latency on a fresh batch of near-duplicates + novels */
    double *qq = (double *)calloc(500, DIM * sizeof(double));
    for (int i = 0; i < 500; ++i) {
        memcpy(qq + i * DIM, parents + ((size_t)i % SCAFFOLDS) * DIM, DIM * sizeof(double));
        flip_few(qq + i * DIM, 1 + (rng_u32() % 3));
    }
    t0 = now_s();
    for (int i = 0; i < 500; ++i) { bool u; futcache_pack_is_novel(vp, qq + i * DIM, &u); }
    double vp_q = (now_s() - t0) / 500.0;
    t0 = now_s();
    for (int i = 0; i < 500; ++i) { bool u; futcache_pack_is_novel(lin, qq + i * DIM, &u); }
    double lin_q = (now_s() - t0) / 500.0;

    printf("3) observe (all candidate)           : vp-tree %.1f ms   linear %.1f ms\n",
           obs_s * 1e3, lin_obs_s * 1e3);
    printf("   novelty query (500 near-dup)      : vp-tree %.0f us   linear %.0f us\n",
           vp_q * 1e6, lin_q * 1e6);

    futcache_pack_destroy(vp);
    futcache_pack_destroy(lin);
    free(pts); free(parents); free(qq); free(lo); free(hi);
    return 0;
}

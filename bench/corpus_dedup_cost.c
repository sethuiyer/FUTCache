/*
 * Corpus-dedup cost benchmark: measures the honest economics of using the
 * packing cache to de-duplicate a vector corpus BEFORE embedding/indexing.
 *
 * The point is evidence, not marketing. It reports, as a function of epsilon,
 * the actual reduction ratio on a REALISTICALLY STRUCTURED embedding set
 * (topic clusters + long tail -- the shape most corpora have), and the
 * downstream cost that deduplication would save. It also runs the same
 * measurement on a pure-uniform set so the reader sees how the ratio
 * degrades when the data has no clusters at all.
 *
 * Usage:
 *   futcache_corpus_dedup                    # synthetic structured data
 *   futcache_corpus_dedup embeddings.bin 384 # a real corpus (N*384 float64 LE)
 *   futcache_corpus_dedup --uniform          # uniform-sphere baseline only
 *
 * Cost model (labelled, clearly an assumption, override with --cost X):
 *   - embedding tokens/doc        = 1000
 *   - embed price                 = $0.20 / 1M tokens
 *   - vector storage              = 4 bytes/coord (float32)
 * The dedup saves the embedding/index/ingest cost of the near-duplicate
 * vectors that collapse into a representative.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

static uint64_t rng_state = UINT64_C(0x243f6a8885a308d3);
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
static void normalize(double *v, size_t d)
{
    double n = 0.0;
    for (size_t i = 0; i < d; ++i) n += v[i] * v[i];
    n = sqrt(n);
    if (n > 0.0) for (size_t i = 0; i < d; ++i) v[i] /= n;
}
/* ---- cost model (assumptions) ---- */
#define TOKENS_PER_DOC   1000.0
#define EMBED_PRICE_MTok 0.20   /* $ per 1M tokens */
#define BYTES_PER_COORD  4.0    /* float32 */

static double g_cost_per_mtok = EMBED_PRICE_MTok;
static double cost_per_doc(void)
{
    return TOKENS_PER_DOC * (g_cost_per_mtok / 1e6);
}

/* Build synthetic structured corpus: topic clusters (tight) + long tail. */
static double *gen_structured(size_t n, size_t dim, double tail_frac,
                              size_t *out_clusters)
{
    double *pts = (double *)malloc(n * dim * sizeof(double));
    if (pts == NULL) return NULL;
    const size_t n_clusters = 40;
    *out_clusters = n_clusters;
    size_t idx = 0;
    size_t cluster_points = (size_t)((double)n * (1.0 - tail_frac) / (double)n_clusters);
    for (size_t c = 0; c < n_clusters; ++c) {
        /* a random unit base vector for this topic */
        double *base = (double *)malloc(dim * sizeof(double));
        if (base == NULL) { free(pts); return NULL; }
        for (size_t i = 0; i < dim; ++i) base[i] = rng_unit() * 2.0 - 1.0;
        normalize(base, dim);
        /* paraphrase spread: small noise so members are near-duplicates */
        for (size_t k = 0; k < cluster_points; ++k) {
            double *p = pts + idx * dim;
            for (size_t i = 0; i < dim; ++i) p[i] = base[i];
            for (size_t i = 0; i < dim; ++i)
                p[i] += 0.03 * (rng_unit() * 2.0 - 1.0);
            normalize(p, dim);
            ++idx;
            if (idx >= n) break;
        }
        free(base);
        if (idx >= n) break;
    }
    /* long tail: genuinely distinct points, barely deduplicate */
    while (idx < n) {
        double *p = pts + idx * dim;
        for (size_t i = 0; i < dim; ++i) p[i] = rng_unit() * 2.0 - 1.0;
        normalize(p, dim);
        ++idx;
    }
    return pts;
}

/* Pure-uniform corpus: maximal packing, worst case for dedup. */
static double *gen_uniform(size_t n, size_t dim)
{
    double *pts = (double *)malloc(n * dim * sizeof(double));
    if (pts == NULL) return NULL;
    for (size_t i = 0; i < n; ++i) {
        double *p = pts + i * dim;
        for (size_t d = 0; d < dim; ++d) p[d] = rng_unit() * 2.0 - 1.0;
        normalize(p, dim);
    }
    return pts;
}

/* Read N*dim little-endian float64 from a file. */
static double *read_bin(const char *path, size_t *out_n, size_t dim)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);
    if (bytes <= 0) { fclose(f); return NULL; }
    size_t n = (size_t)bytes / (dim * sizeof(double));
    double *pts = (double *)malloc(n * dim * sizeof(double));
    if (pts == NULL) { fclose(f); return NULL; }
    size_t got = fread(pts, sizeof(double), n * dim, f);
    fclose(f);
    if (got != n * dim) { free(pts); return NULL; }
    *out_n = n;
    return pts;
}

/* Run dedup on `pts` and report economics for a sweep of epsilon. */
static void run_economics(const double *pts, size_t n, size_t dim,
                          const char *label, size_t clusters, double tail_frac,
                          size_t scale_n)
{
    double *lo = (double *)malloc(dim * sizeof(double));
    double *hi = (double *)malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; ++i) { lo[i] = -1.0; hi[i] = 1.0; }

    const double eps_grid[] = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40};
    const size_t eps_n = sizeof(eps_grid) / sizeof(eps_grid[0]);

    printf("\n--- %s (n=%zu, dim=%zu) ---\n", label, n, dim);
    if (clusters)
        printf("    structure: %zu topic clusters + %.0f%% long tail\n",
               clusters, tail_frac * 100.0);
    else
        printf("    structure: no clusters\n");
    printf("    cost model: %zu tok/doc * $%.2f/1M tok + %.0f B/coord\n",
           (size_t)TOKENS_PER_DOC, g_cost_per_mtok, BYTES_PER_COORD);
    printf("    epsilon  | reps   | novel  | dedup_ratio | %%saved | $ saved (this corpus) | $ saved (scaled to %zu docs)\n",
           scale_n);
    printf("    ---------|--------|--------|-------------|--------|----------------------|----------------------------\n");

    for (size_t e = 0; e < eps_n; ++e) {
        double eps = eps_grid[e];
        futcache_pack_config_t cfg;
        futcache_pack_config_init(&cfg);
        cfg.dimension = dim;
        cfg.epsilon = eps;
        cfg.distance = futcache_distance_cosine;
        cfg.domain_min = lo;
        cfg.domain_max = hi;
        cfg.backend = &futcache_pack_vptree_backend;
        futcache_pack_t *cache = NULL;
        if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) continue;

        for (size_t i = 0; i < n; ++i) {
            bool unused;
            futcache_pack_observe(cache, pts + i * dim, &unused);
        }
        futcache_pack_stats_t st;
        futcache_pack_get_stats(cache, &st);
        double ratio = st.representative_count > 0
            ? (double)n / (double)st.representative_count : 0.0;
        double pct = 100.0 * (double)(n - st.representative_count) / (double)n;
        double dollars = (double)(n - st.representative_count) * cost_per_doc();
        double scaled = (double)scale_n * (pct / 100.0) * cost_per_doc();
        printf("    %-7.2f | %-6zu | %-6zu | %11.2f | %5.1f%% | $%-20.2f | $%.2f\n",
               eps, st.representative_count, st.novel_observations,
               ratio, pct, dollars, scaled);
        futcache_pack_destroy(cache);
    }
    free(lo);
    free(hi);
}

int main(int argc, char **argv)
{
    size_t dim = 128;      /* default synthetic corpus size (fast, exact scan) */
    size_t n = 4000;
    size_t scale_n = 100000;   /* normalize the cost column to this corpus size */
    bool uniform_only = false;
    const char *file = NULL;
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--uniform") == 0) uniform_only = true;
        else if (strcmp(argv[a], "--dim") == 0 && a + 1 < argc)
            dim = (size_t)strtoull(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--n") == 0 && a + 1 < argc)
            n = (size_t)strtoull(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--scale") == 0 && a + 1 < argc)
            scale_n = (size_t)strtoull(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--cost") == 0 && a + 1 < argc)
            g_cost_per_mtok = atof(argv[++a]);
        else if (file == NULL) file = argv[a];
    }

    printf("=== FUTCache corpus-dedup cost benchmark ===\n");

    if (file != NULL) {
        size_t nfile = 0;
        double *pts = read_bin(file, &nfile, dim);
        if (pts == NULL) { fprintf(stderr, "could not read %s\n", file); return 1; }
        run_economics(pts, nfile, dim, "real corpus", 0, 0.0, scale_n);
        free(pts);
    } else if (uniform_only) {
        double *pts = gen_uniform(n, dim);
        if (pts == NULL) return 1;
        run_economics(pts, n, dim, "uniform corpus (worst case)", 0, 0.0, scale_n);
        free(pts);
    } else {
        size_t clusters = 0;
        double *pts = gen_structured(n, dim, 0.25, &clusters);
        if (pts == NULL) return 1;
        run_economics(pts, n, dim, "balanced corpus (clusters + long tail)",
                      clusters, 0.25, scale_n);
        free(pts);

        pts = gen_structured(n, dim, 0.03, &clusters);
        if (pts == NULL) return 1;
        run_economics(pts, n, dim, "tight-cluster corpus (almost no tail)",
                      clusters, 0.03, scale_n);
        free(pts);

        pts = gen_uniform(n, dim);
        if (pts == NULL) return 1;
        run_economics(pts, n, dim, "uniform corpus (worst case)", 0, 0.0, scale_n);
        free(pts);
    }
    printf("\nNOTES:\n");
    printf("  * dedup_ratio = corpus / representatives (how much content collapses).\n");
    printf("  * $ saved = embedding/index/ingest cost of the vectors that collapse\n");
    printf("    into a representative, at %zu docs (scaled column = %zu docs).\n",
           n, scale_n);
    printf("  * The ratio is DISTRIBUTION-dependent: it is bounded by the clustered\n");
    printf("    fraction, because a uniform long tail in high dimension is genuinely\n");
    printf("    far apart and does not collapse at any reasonable epsilon.\n");
    printf("  * The pack cache is one-sided (never suppresses novelty, may keep a few\n");
    printf("    extra non-duplicates), so '%%saved' is a LOWER bound, not a promise.\n");
    return 0;
}

#define _POSIX_C_SOURCE 200809L

/*
 * bekko_semantic_cache.c
 *
 * Replays pre-computed sentence embeddings through futcache_pack and
 * reports representative count, semantic-reuse correctness, and per-call
 * latency at multiple resolutions and epsilon thresholds.
 *
 * Input binary format (little-endian, written by scripts/bekko_generate.py):
 *
 *   header (32 bytes):
 *     uint32_t magic            ('F','U','T','E' = 0x45545546 LE)
 *     uint32_t version          (currently 1)
 *     uint32_t count            (number of embedding records)
 *     uint32_t dim              (embedding dimension, 64/128/256/384)
 *     uint32_t label_count      (number of semantic-group labels, >= 1)
 *     uint32_t reserved[3]      (zero)
 *
 *   then `count` records, each:
 *     uint32_t label            (semantic group, 0..label_count-1)
 *     uint32_t lang             (BCP-47 packed, e.g. 'en' = 0x656e)
 *     float    norm             (precomputed L2 norm of the vector)
 *     double   coords[dim]      (the embedding, expected L2-normalized)
 *
 * The Python generator emits the format; this binary reads it.
 *
 * For each (dim truncation, epsilon) the program reports:
 *
 *   - representative_count: how many semantic groups were remembered
 *   - novel_count:          how many inputs the cache reported as novel
 *   - correct_reuse:        fraction of redundant inputs whose label was
 *                           already represented (true semantic hits)
 *   - false_reuse:          fraction of novel inputs whose label was
 *                           already represented (cache hit but no true
 *                           neighbour — under-counts in low-d)
 *   - microseconds/op:      per-call latency, observe + stats
 *
 * Output is a markdown table.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/pack.h"

#define BEKKO_MAGIC UINT32_C(0x45545546)  /* 'FUTE' */
#define BEKKO_VERSION UINT32_C(1)

typedef struct embedding_record {
    uint32_t label;
    uint32_t lang;
    float norm;
    double *coords;
} embedding_record_t;

typedef struct dataset {
    uint32_t count;
    uint32_t dim;
    uint32_t label_count;
    embedding_record_t *records;
} dataset_t;

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static bool read_u32(FILE *fp, uint32_t *out)
{
    uint8_t buf[4];
    if (fread(buf, 1, 4, fp) != 4) return false;
    *out = (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
    return true;
}

static bool read_f32(FILE *fp, float *out)
{
    uint8_t buf[4];
    if (fread(buf, 1, 4, fp) != 4) return false;
    uint32_t u = (uint32_t)buf[0]
               | ((uint32_t)buf[1] << 8)
               | ((uint32_t)buf[2] << 16)
               | ((uint32_t)buf[3] << 24);
    memcpy(out, &u, sizeof(float));
    return true;
}

static bool read_f64(FILE *fp, double *out)
{
    uint8_t buf[8];
    if (fread(buf, 1, 8, fp) != 8) return false;
    uint64_t u = 0;
    for (size_t i = 0; i < 8; ++i) {
        u |= ((uint64_t)buf[i]) << (i * 8);
    }
    memcpy(out, &u, sizeof(double));
    return true;
}

static dataset_t *load_dataset(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic, version, count, dim, label_count;
    uint32_t reserved[3];

    if (!read_u32(fp, &magic) || magic != BEKKO_MAGIC) {
        fprintf(stderr, "bad magic in %s\n", path);
        fclose(fp);
        return NULL;
    }
    if (!read_u32(fp, &version) || version != BEKKO_VERSION) {
        fprintf(stderr, "unsupported version %u\n", version);
        fclose(fp);
        return NULL;
    }
    if (!read_u32(fp, &count) || !read_u32(fp, &dim)
        || !read_u32(fp, &label_count)) {
        fprintf(stderr, "header truncated\n");
        fclose(fp);
        return NULL;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (!read_u32(fp, &reserved[i])) {
            fprintf(stderr, "header truncated\n");
            fclose(fp);
            return NULL;
        }
    }

    dataset_t *ds = calloc(1, sizeof(*ds));
    ds->count = count;
    ds->dim = dim;
    ds->label_count = label_count;
    ds->records = calloc(count, sizeof(embedding_record_t));

    for (uint32_t i = 0; i < count; ++i) {
        if (!read_u32(fp, &ds->records[i].label)
            || !read_u32(fp, &ds->records[i].lang)
            || !read_f32(fp, &ds->records[i].norm)) {
            fprintf(stderr, "record %u header truncated\n", i);
            fclose(fp);
            return NULL;
        }
        ds->records[i].coords = calloc(dim, sizeof(double));
        for (uint32_t k = 0; k < dim; ++k) {
            if (!read_f64(fp, &ds->records[i].coords[k])) {
                fprintf(stderr, "record %u coords truncated\n", i);
                fclose(fp);
                return NULL;
            }
        }
    }

    fclose(fp);
    return ds;
}

static void free_dataset(dataset_t *ds)
{
    if (ds == NULL) return;
    for (uint32_t i = 0; i < ds->count; ++i) {
        free(ds->records[i].coords);
    }
    free(ds->records);
    free(ds);
}

typedef struct run_stats {
    size_t representative_count;
    size_t novel_count;
    size_t redundant_count;
    /* Confusion against the semantic-group label:
     *   correct_reuse   = redundant input whose label already represented
     *   incorrect_reuse = novel input whose label already represented
     *                       (cache hit, true hit missed)
     *   missed_reuse    = redundant input whose label NOT represented
     *                       (cache said redundant for an unseen group)
     *   correct_novel   = novel input whose label NOT represented */
    size_t correct_reuse;
    size_t incorrect_reuse;
    size_t missed_reuse;
    size_t correct_novel;
    double seconds;
} run_stats_t;

static run_stats_t run_pack(
    const dataset_t *ds,
    double epsilon,
    uint32_t truncate_dim)
{
    run_stats_t r = {0};

    /* Each record is independently truncated to the first `truncate_dim`
     * coordinates. The cache's domain is the truncated dim. */
    double lo[384], hi[384];
    for (uint32_t i = 0; i < truncate_dim; ++i) {
        lo[i] = -1.0;
        hi[i] = 1.0;
    }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = truncate_dim;
    cfg.epsilon = epsilon;
    cfg.distance = futcache_distance_cosine;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        fprintf(stderr, "cache create failed (truncate_dim=%u, eps=%g)\n",
                truncate_dim, epsilon);
        return r;
    }

    /* Track which labels have ever been represented. */
    uint8_t *label_seen = calloc(ds->label_count, 1);

    double t0 = monotonic_seconds();
    for (uint32_t i = 0; i < ds->count; ++i) {
        const embedding_record_t *rec = &ds->records[i];
        bool cache_novel = false;
        if (futcache_pack_observe(cache, rec->coords, &cache_novel)
            != FUTCACHE_OK) {
            fprintf(stderr, "observe failed at record %u\n", i);
            futcache_pack_destroy(cache);
            free(label_seen);
            return r;
        }

        bool label_known = label_seen[rec->label] != 0;
        if (cache_novel) {
            r.novel_count++;
            if (label_known) {
                r.incorrect_reuse++;
            } else {
                r.correct_novel++;
                label_seen[rec->label] = 1;
            }
        } else {
            r.redundant_count++;
            if (label_known) {
                r.correct_reuse++;
            } else {
                r.missed_reuse++;
                label_seen[rec->label] = 1;
            }
        }
    }
    double t1 = monotonic_seconds();

    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    r.representative_count = stats.representative_count;
    r.seconds = t1 - t0;

    futcache_pack_destroy(cache);
    free(label_seen);
    return r;
}

static void print_table_header(void)
{
    puts("| truncate | epsilon | reps | novel | reuse_rate | "
         "reuse_precision | correct_reuse | incorrect_reuse | "
         "missed_reuse | us/op |");
    puts("|----------|---------|------|-------|------------|"
         "-----------------|---------------|-----------------|"
         "--------------|-------|");
}

static void print_row(uint32_t truncate_dim, double epsilon,
                      const run_stats_t *r, uint32_t count)
{
    double us_per_op = r->seconds > 0.0
        ? (r->seconds * 1e6) / (double)count : 0.0;
    double correct = (double)r->correct_reuse / (double)count;
    double incorrect = (double)r->incorrect_reuse / (double)count;
    double missed = (double)r->missed_reuse / (double)count;
    size_t reuse_count = r->correct_reuse + r->missed_reuse;
    double reuse_rate = (double)reuse_count / (double)count;
    double reuse_precision = reuse_count > 0
        ? (double)r->correct_reuse / (double)reuse_count
        : 0.0;
    printf("|   %3u    | %.3f   | %4zu | %5zu |   %.4f    |     "
           "%.4f      |    %.4f     |    %.4f      |   "
           "%.4f     | %5.2f |\n",
        truncate_dim, epsilon, r->representative_count, r->novel_count,
        reuse_rate, reuse_precision, correct, incorrect, missed, us_per_op);
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s <embeddings.bin>\n"
        "  reads a Bekko embedding binary and sweeps dim/epsilon\n",
        prog);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    dataset_t *ds = load_dataset(argv[1]);
    if (ds == NULL) return EXIT_FAILURE;

    printf("# Bekko semantic cache experiment\n");
    printf("# records = %u, source dim = %u, label_count = %u\n\n",
           ds->count, ds->dim, ds->label_count);

    /* Sweep truncation dimension and epsilon. */
    const uint32_t truncs[] = {64, 128, 256, 384};
    const double epsilons[] = {
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
        0.70, 0.80, 0.90
    };

    for (size_t t = 0; t < sizeof(truncs) / sizeof(truncs[0]); ++t) {
        uint32_t td = truncs[t];
        if (td > ds->dim) continue;  /* can't truncate upward */
        printf("## truncate_dim = %u\n\n", td);
        print_table_header();
        for (size_t e = 0; e < sizeof(epsilons) / sizeof(epsilons[0]); ++e) {
            run_stats_t r = run_pack(ds, epsilons[e], td);
            print_row(td, epsilons[e], &r, ds->count);
            if (r.representative_count == 0U) break;
        }
        puts("");
    }

    puts("# Column legend");
    puts("#  reps:            representative count at end of stream");
    puts("#  novel:           inputs the cache reported as novel");
    puts("#  correct_reuse:   P(redundant AND label already represented)");
    puts("#  incorrect_reuse: P(novel AND label already represented)");
    puts("#                    cache hit, but a true semantic neighbour");
    puts("#                    was missed (packing approximation at");
    puts("#                    low dim or large epsilon)");
    puts("#  missed_reuse:    P(redundant AND label NOT yet represented)");
    puts("#                    cache hit, true novelty not seen");
    puts("#  us/op:           per-call latency, end-to-end observe");

    free_dataset(ds);
    return EXIT_SUCCESS;
}

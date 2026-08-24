/* soc_novelty_demo.c — stream a log, inject an anomaly, does FUTCache see it?
 *
 * Validates the cyber/SOC use case concretely:
 *   - each log event is encoded as a 512-bit behavioral fingerprint (bits =
 *     "behavior predicates": syscall/registry/network/file patterns);
 *   - FUTCache pack cache with a custom HAMMING distance (a genuine metric,
 *     the "deeply novel metric" framing);
 *   - the stream is mostly NORMAL traffic (a few recurring event types, with
 *     jitter) plus ONE injected ANOMALY (a novel behavioral signature);
 *   - we check that normal traffic collapses to a few representatives
 *     (redundant = "known"), and the ANOMALY is flagged NOVEL the moment it
 *     appears -- and, by the one-sided guarantee, is NEVER suppressed.
 *
 * Build (from repo root):
 *   cc -O2 -std=gnu11 -I include examples/soc_novelty_demo.c \
 *      -o build-test-release/soc_demo build-test-release/libfutcache.a -lm -lpthread
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

enum { DIM = 512, NORMAL_TYPES = 4, STREAM_LEN = 3000, ANOMALY_AT = 1500 };

static uint64_t rng_state = UINT64_C(0x9e3779b97f4a7c15);
static uint64_t rng_next(void)
{
    uint64_t v = rng_state;
    v ^= v << 13U; v ^= v >> 7U; v ^= v << 17U;
    rng_state = v;
    return v;
}
static uint32_t rng_u32(void) { return (uint32_t)(rng_next() & 0xffffffffU); }

/* Hamming distance on behavior bit-vectors (a genuine metric). */
static double hamming(const double *a, const double *b, size_t d, void *ctx)
{
    (void)ctx;
    double s = 0.0;
    for (size_t i = 0; i < d; ++i)
        if ((a[i] > 0.5) != (b[i] > 0.5)) s += 1.0;
    return s;
}

static void set_bits(double *v, const uint32_t *bits, size_t n)
{
    for (size_t i = 0; i < n; ++i) v[bits[i]] = 1.0;
}

/* Behavioral signatures: base bit-sets per event type. */
static const uint32_t http_get[] = {1,5,9,13,17,21,25,33,37,41,45,49};
static const uint32_t db_query[] = {60,62,64,66,68,70,72,74};
static const uint32_t auth_ok[]  = {80,82,84,86,88,90};
static const uint32_t hrtbeat[]  = {100,102,104,106};
static const uint32_t anomaly[]  = {150,152,154,156,158,160,162,5,13,164,166};
/* ^ anomaly overlaps http_get on bits {5,13} but adds a distinctive C2/behavior
 *   pattern -- novel combination, not a pure duplicate. */

static void fill_normal(double *v, int type)
{
    memset(v, 0, DIM * sizeof(double));
    if (type == 0) set_bits(v, http_get, sizeof(http_get)/sizeof(http_get[0]));
    else if (type == 1) set_bits(v, db_query, sizeof(db_query)/sizeof(db_query[0]));
    else if (type == 2) set_bits(v, auth_ok, sizeof(auth_ok)/sizeof(auth_ok[0]));
    else set_bits(v, hrtbeat, sizeof(hrtbeat)/sizeof(hrtbeat[0]));
    /* jitter: 0-1 random bit (small noise on otherwise-recurring behavior) */
    for (int j = 0; j < (int)(rng_u32() % 2); ++j) v[rng_u32() % DIM] = 1.0;
}

static void fill_anomaly(double *v)
{
    memset(v, 0, DIM * sizeof(double));
    set_bits(v, anomaly, sizeof(anomaly)/sizeof(anomaly[0]));
    for (int j = 0; j < (int)(rng_u32() % 2); ++j) v[rng_u32() % DIM] = 1.0;
}

static const char *tag_for(int type) {
    static const char *t[NORMAL_TYPES] = {"http_get","db_query","auth_ok","heartbeat"};
    return t[type];
}

int main(void)
{
    double *lo = (double *)malloc(DIM * sizeof(double));
    double *hi = (double *)malloc(DIM * sizeof(double));
    for (size_t i = 0; i < DIM; ++i) { lo[i] = 0.0; hi[i] = 1.0; }

    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = DIM;
    cfg.epsilon = 3.0;              /* 3 behavior-bit differences = same-ish */
    cfg.distance = hamming;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    /* use the exact VP-tree (Hamming is a metric, so pruning is sound) */
    cfg.backend = &futcache_pack_vptree_backend;

    futcache_pack_t *cache = NULL;
    futcache_pack_create(&cfg, &cache);

    double fp[DIM];
    int novel_count = 0, redundant_count = 0;
    bool anomaly_seen_novel = false;
    int anomaly_event_index = -1;
    printf("=== streaming SOC log (Hamming behavior novelty, eps=%.0f bits) ===\n", cfg.epsilon);
    printf("  %-10s %-10s %s\n", "event#", "type", "decision");
    for (int i = 0; i < STREAM_LEN; ++i) {
        int type;
        if (i == ANOMALY_AT) {
            fill_anomaly(fp);
            type = -1; /* anomaly */
        } else {
            type = (int)(rng_u32() % NORMAL_TYPES);
            fill_normal(fp, type);
        }
        bool was_novel = false;
        futcache_pack_observe(cache, fp, &was_novel);
        if (was_novel) novel_count++; else redundant_count++;
        if (i == ANOMALY_AT) {
            anomaly_seen_novel = was_novel;
            anomaly_event_index = i;
            printf("  %-10d %-10s %s  <-- ANOMALY\n", i,
                   type < 0 ? "ANOMALY" : tag_for(type),
                   was_novel ? "NOVEL (detected)" : "REDUNDANT");
        } else if (i % 400 == 0) {
            printf("  %-10d %-10s %s\n", i, tag_for(type),
                   was_novel ? "novel" : "redundant");
        }
    }

    futcache_pack_stats_t st;
    futcache_pack_get_stats(cache, &st);

    printf("\n=== summary ===\n");
    printf("  events streamed: %d  (normal + 1 anomaly at #%d)\n",
           STREAM_LEN, anomaly_event_index);
    printf("  novelty: %d novel / %d redundant\n", novel_count, redundant_count);
    printf("  representatives retained: %zu\n", st.representative_count);
    printf("  normal-traffic compression: %d normal events -> %zu reps\n",
           STREAM_LEN, st.representative_count);

    printf("\n--- THE CHECK ---\n");
    printf("  anomaly (#%d) flagged NOVEL: %s\n", anomaly_event_index,
           anomaly_seen_novel ? "YES" : "NO");
    printf("  one-sided guarantee (never suppress a novel behavior): %s\n",
           anomaly_seen_novel ? "HOLDS" : "VIOLATED");
    printf("  expected reps ~=%d (4 normal types + anomaly) + jitter reps\n",
           NORMAL_TYPES + 1);
    if (!anomaly_seen_novel)
        printf("  ... but FUTCache suppressed the anomaly -- the gate is NOT safe\n");

    futcache_pack_destroy(cache);
    free(lo); free(hi);
    return anomaly_seen_novel ? 0 : 1;
}

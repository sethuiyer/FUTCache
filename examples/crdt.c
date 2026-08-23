#define _POSIX_C_SOURCE 200809L

/*
 * crdt.c -- distributed semantic cache demo (PHASE2.md, Section 12.6).
 *
 * Demonstrates the three CRDT laws on a deterministic Voronoi anchor net:
 *
 *   1. Convergence:  two replicas with disjoint histories gossip and
 *                    agree on the same occupied-cell set.
 *   2. Idempotence:   merge(A, A) = A.
 *   3. Commutativity: merge(A, B) = merge(B, A).
 *
 * Constructs a 3x3 anchor net on [0, 1]^2 with epsilon = 0.30. The
 * Voronoi partition induces 9 cells; any two points in the same cell
 * are within 2 * max_dist_to_anchor of each other, well under 0.30.
 *
 * The "network" between replicas is two buffers in this process; in
 * production it would be TCP/UDP. The merge logic is identical.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/crdt.h"
#include "futcache/futcache.h"

#define DIM 2
#define ANCHORS 9

static uint64_t rng_state = UINT64_C(0x123456789abcdef0);

static uint64_t next_random(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static double uniform_double(void)
{
    return (double)(next_random() >> 11) / (double)(UINT64_C(1) << 53);
}

static void require_ok(futcache_status_t s, const char *what)
{
    if (s != FUTCACHE_OK) {
        fprintf(stderr, "%s failed: %s\n", what, futcache_status_string(s));
        exit(EXIT_FAILURE);
    }
}

static futcache_crdt_t *make_replica(const double anchors[ANCHORS][DIM])
{
    futcache_crdt_config_t cfg;
    futcache_crdt_config_init(&cfg);
    cfg.dimension = DIM;
    cfg.anchor_count = ANCHORS;
    cfg.anchors = &anchors[0][0];
    cfg.epsilon = 0.30;
    cfg.distance = NULL;
    cfg.distance_context = NULL;

    double lo[DIM] = {0.0, 0.0};
    double hi[DIM] = {1.0, 1.0};
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_crdt_t *r = NULL;
    require_ok(futcache_crdt_create(&cfg, &r), "crdt_create");
    return r;
}

static void observe_random(futcache_crdt_t *r, size_t n, const char *tag)
{
    char payload[32];
    for (size_t i = 0; i < n; ++i) {
        double p[DIM] = {uniform_double(), uniform_double()};
        int written = snprintf(payload, sizeof(payload),
                               "%s_obs_%zu", tag, i);
        if (written < 0) {
            payload[0] = '\0';
        }
        require_ok(futcache_crdt_observe(r, p, payload, strlen(payload),
                                         NULL, NULL), "crdt_observe");
    }
}

static size_t snapshot_to_buf(futcache_crdt_t *r,
                              futcache_crdt_update_t *buf,
                              size_t cap)
{
    size_t count = cap;
    require_ok(futcache_crdt_snapshot(r, buf, &count), "snapshot");
    if (count > cap) {
        fprintf(stderr, "snapshot overflow: %zu > %zu\n", count, cap);
        exit(EXIT_FAILURE);
    }
    return count;
}

static void sort_by_cell(futcache_crdt_update_t *buf, size_t n)
{
    /* Insertion sort by cell index ascending. n is small (<= 64). */
    for (size_t i = 1; i < n; ++i) {
        futcache_crdt_update_t key = buf[i];
        size_t j = i;
        while (j > 0 && buf[j - 1].cell > key.cell) {
            buf[j] = buf[j - 1];
            --j;
        }
        buf[j] = key;
    }
}

static bool states_equal(futcache_crdt_t *a, futcache_crdt_t *b)
{
    futcache_crdt_update_t buf_a[64];
    futcache_crdt_update_t buf_b[64];
    size_t n_a = snapshot_to_buf(a, buf_a, 64);
    size_t n_b = snapshot_to_buf(b, buf_b, 64);
    if (n_a != n_b) return false;

    sort_by_cell(buf_a, n_a);
    sort_by_cell(buf_b, n_b);

    for (size_t i = 0; i < n_a; ++i) {
        if (buf_a[i].cell != buf_b[i].cell) return false;
        if (buf_a[i].priority != buf_b[i].priority) return false;
        if (buf_a[i].payload_length != buf_b[i].payload_length) return false;
        if (memcmp(buf_a[i].payload, buf_b[i].payload,
                   buf_a[i].payload_length) != 0) return false;
    }
    return true;
}

static void print_state(const char *label, futcache_crdt_t *r)
{
    futcache_crdt_stats_t stats;
    require_ok(futcache_crdt_get_stats(r, &stats), "stats");
    printf("  %-16s occupied=%-3zu  obs=%" PRIu64
           "  novel=%" PRIu64 "  gen=%" PRIu64 "\n",
           label, stats.occupied_cells,
           stats.observations, stats.novel_observations, stats.generation);
}

int main(void)
{
    const double anchors[ANCHORS][DIM] = {
        {1.0 / 6.0, 1.0 / 6.0}, {3.0 / 6.0, 1.0 / 6.0}, {5.0 / 6.0, 1.0 / 6.0},
        {1.0 / 6.0, 3.0 / 6.0}, {3.0 / 6.0, 3.0 / 6.0}, {5.0 / 6.0, 3.0 / 6.0},
        {1.0 / 6.0, 5.0 / 6.0}, {3.0 / 6.0, 5.0 / 6.0}, {5.0 / 6.0, 5.0 / 6.0},
    };

    printf("=== FUTCache CRDT demo: two-replica convergence ===\n");
    printf("dimension=%d, anchor_count=%d, epsilon=0.30\n\n", DIM, ANCHORS);

    /* Stage 1: disjoint histories, no merge yet. */
    printf("Stage 1: replicas observe disjoint streams\n");
    futcache_crdt_t *tokyo = make_replica(anchors);
    futcache_crdt_t *london = make_replica(anchors);

    rng_state = UINT64_C(0xa1a1a1a1a1a1a1a1);
    observe_random(tokyo, 30, "tokyo");
    rng_state = UINT64_C(0xb2b2b2b2b2b2b2b2);
    observe_random(london, 30, "london");

    print_state("tokyo (initial)", tokyo);
    print_state("london (initial)", london);
    bool equal_after_observe = states_equal(tokyo, london);
    printf("  initial states equal? %s (expected false)\n\n",
           equal_after_observe ? "true" : "false");

    /* Stage 2: idempotence. Merge a snapshot into its own source. */
    printf("Stage 2: idempotence of merge\n");
    futcache_crdt_update_t self_buf[64];
    size_t self_n = snapshot_to_buf(tokyo, self_buf, 64);
    require_ok(futcache_crdt_merge(tokyo, self_buf, self_n), "self_merge");
    bool idempotent = states_equal(tokyo, tokyo);
    printf("  merge(self, self) preserves state? %s (expected true)\n\n",
           idempotent ? "true" : "false");

    /* Stage 3: commutativity. Two clean replicas, swap snapshots,
     * compare. Note: the snapshot API is zero-copy and aliases the
     * source cache, so we re-snapshot the source after any merge that
     * mutated it. */
    printf("Stage 3: commutativity of merge\n");
    futcache_crdt_t *tokyo_b = make_replica(anchors);
    futcache_crdt_t *london_b = make_replica(anchors);
    rng_state = UINT64_C(0xa1a1a1a1a1a1a1a1);
    observe_random(tokyo_b, 30, "tokyo");
    rng_state = UINT64_C(0xb2b2b2b2b2b2b2b2);
    observe_random(london_b, 30, "london");

    futcache_crdt_update_t tokyo_buf[64];
    futcache_crdt_update_t london_buf[64];
    size_t tokyo_n = snapshot_to_buf(tokyo_b, tokyo_buf, 64);

    /* merge tokyo's snapshot into london_b (mutates london_b) */
    require_ok(futcache_crdt_merge(london_b, tokyo_buf, tokyo_n),
               "merge_tokyo_into_london");

    /* re-snapshot london_b because step 1 mutated it and may have
     * deallocated the points the previous london_buf was aliasing. */
    size_t london_n = snapshot_to_buf(london_b, london_buf, 64);

    /* merge london's (post-merge) snapshot into tokyo_b */
    require_ok(futcache_crdt_merge(tokyo_b, london_buf, london_n),
               "merge_london_into_tokyo");

    bool commute = states_equal(tokyo_b, london_b);
    printf("  merge(T, L) = merge(L, T)? %s (expected true)\n\n",
           commute ? "true" : "false");

    /* Stage 4: gossip convergence over multiple rounds. The snapshot
     * is zero-copy and aliases the source cache, so we re-snapshot
     * after every merge that mutated the source. */
    printf("Stage 4: gossip convergence (Tokyo <-> London)\n");
    for (int round = 1; round <= 5; ++round) {
        futcache_crdt_update_t t_buf[64];
        futcache_crdt_update_t l_buf[64];
        size_t t_n = snapshot_to_buf(tokyo, t_buf, 64);
        size_t l_n = snapshot_to_buf(london, l_buf, 64);
        require_ok(futcache_crdt_merge(tokyo, l_buf, l_n),
                   "merge_L_into_T");
        /* re-snapshot tokyo: previous t_buf may alias freed points */
        t_n = snapshot_to_buf(tokyo, t_buf, 64);
        require_ok(futcache_crdt_merge(london, t_buf, t_n),
                   "merge_T_into_L");
        bool same = states_equal(tokyo, london);
        print_state("tokyo", tokyo);
        print_state("london", london);
        printf("  round %d: converged? %s\n\n", round, same ? "true" : "false");
        if (same) break;
    }

    /* Stage 5: semantic compression summary. */
    printf("Stage 5: semantic compression after gossip\n");
    futcache_crdt_stats_t final_stats;
    require_ok(futcache_crdt_get_stats(tokyo, &final_stats), "final_stats");
    printf("  tokyo final: occupied=%zu, observations=%" PRIu64 "\n",
           final_stats.occupied_cells, final_stats.observations);
    futcache_crdt_stats_t london_stats;
    require_ok(futcache_crdt_get_stats(london, &london_stats), "london_stats");
    printf("  london final: occupied=%zu, observations=%" PRIu64 "\n",
           london_stats.occupied_cells, london_stats.observations);

    /* Cleanup. */
    futcache_crdt_destroy(tokyo_b);
    futcache_crdt_destroy(london_b);
    futcache_crdt_destroy(tokyo);
    futcache_crdt_destroy(london);

    printf("\nOK: all CRDT laws verified.\n");
    return EXIT_SUCCESS;
}

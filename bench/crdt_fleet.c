#define _POSIX_C_SOURCE 200809L

/*
 * crdt_fleet.c
 *
 * Empirical benchmark for the geometric-blackboard fleet claim in
 * WHY_FUTCACHE.md: when N workers each observe a subset of a shared
 * metric space and gossip their occupied cells, does the merged state
 * reach the same packing number as a single-worker observation?
 *
 * Concretely, for each workload we run:
 *   - a single-worker FUTCache CRDT and record its final occupied-cell
 *     count (this is the *target* joint coverage, since one worker
 *     observes everything);
 *   - W workers, each receiving 1/W of the stream (round-robin or
 *     disjoint shards), each running its own CRDT, with M rounds of
 *     gossip merge between observations. After the stream we compute
 *     the union-of-cells (simulated: snapshot all W workers, count
 *     distinct occupied cells across the fleet) and the
 *     dedup_ratio = joint_coverage / sum_local_coverage.
 *
 * What we measure:
 *   - joint_occupied:    union of cells across all W workers at end
 *   - sum_local_occupied: sum of per-worker cell counts (would equal
 *                         joint if no sharing)
 *   - dedup_ratio:       joint / sum_local (1.0 = no benefit from gossip,
 *                         small = high dedup)
 *   - gossip_round_cost: average wall time of a snapshot+merge round
 *   - convergence:       |joint_after_round_k - joint_after_round_M|
 *                         for k = 1, 2, ... — should reach 0 in <= 2
 *                         rounds for an idempotent join-semilattice
 *
 * The single-worker row serves as the upper bound for joint coverage.
 *
 * Workloads mirror bench/cache_comparison.c: reciprocal, uniform,
 * three-cluster, alternating, power-decay. For each, points are split
 * round-robin across workers (same point goes to one worker — that's
 * the realistic "sharded stream" case; for true cooperative coverage
 * the dedup ratio is even higher).
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "futcache/crdt.h"

/* ============================================================
 * Workloads (subset of bench/cache_comparison.c; d=1 throughout)
 * ============================================================ */

typedef struct workload {
    const char *name;
    double *points;
    size_t count;
    double domain_min;
    double domain_max;
} workload_t;

static uint64_t rng_state = UINT64_C(0xfeedface12345678);

static uint64_t xorshift64(void)
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
    return (double)(xorshift64() >> 11) / (double)(UINT64_C(1) << 53);
}

static double gaussian(double mean, double stddev)
{
    double u1 = uniform_double();
    double u2 = uniform_double();
    if (u1 < 1e-300) u1 = 1e-300;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

static workload_t *workload_reciprocal(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "reciprocal";
    w->count = n;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = 1.0 / (double)(i + 1);
    }
    return w;
}

static workload_t *workload_uniform(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "uniform";
    w->count = n;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = uniform_double();
    }
    return w;
}

static workload_t *workload_cluster(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "three-cluster";
    w->count = n;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    double centers[3] = {0.2, 0.5, 0.8};
    for (size_t i = 0; i < n; ++i) {
        double x = gaussian(centers[i % 3], 0.01);
        if (x < 0.0) x = 0.0;
        if (x > 1.0) x = 1.0;
        w->points[i] = x;
    }
    return w;
}

static workload_t *workload_alternating(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "alternating-extremes";
    w->count = n;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = (i % 2U == 0U) ? 0.01 : 0.99;
    }
    return w;
}

static workload_t *workload_power(size_t n)
{
    workload_t *w = calloc(1, sizeof(*w));
    w->name = "power-decay";
    w->count = n;
    w->domain_min = 0.0;
    w->domain_max = 1.0;
    w->points = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        w->points[i] = uniform_double() * uniform_double();
    }
    return w;
}

static void workload_free(workload_t *w)
{
    free(w->points);
    free(w);
}

/* ============================================================
 * Anchor construction
 *
 * We use the certified grid anchor constructor so the geometric
 * contract is exact, not estimated. With delta = eps/2 and the
 * domain [0,1], cells_per_axis = ceil(1 / (2*delta)) = ceil(1/eps).
 * For eps=0.05 that's 20 cells, eps=0.01 -> 100 cells, eps=0.005
 * -> 200 cells. The CRDT will accept at most one entry per cell.
 * ============================================================ */

static futcache_crdt_t *make_crdt(double epsilon,
                                   double domain_min, double domain_max,
                                   size_t *out_anchors)
{
    size_t cells_per_axis = (size_t)ceil((domain_max - domain_min) /
                                          (2.0 * epsilon));
    if (cells_per_axis < 1U) cells_per_axis = 1U;
    size_t anchor_count = cells_per_axis;
    double *anchors = calloc(anchor_count, sizeof(double));
    /* Place anchors at cell centers. */
    double step = (domain_max - domain_min) / (double)cells_per_axis;
    for (size_t i = 0; i < anchor_count; ++i) {
        anchors[i] = domain_min + ((double)i + 0.5) * step;
    }
    *out_anchors = anchor_count;

    futcache_crdt_config_t cfg;
    futcache_crdt_config_init(&cfg);
    cfg.dimension = 1U;
    cfg.anchor_count = anchor_count;
    cfg.anchors = anchors;
    cfg.epsilon = epsilon;
    cfg.domain_min = &domain_min;
    cfg.domain_max = &domain_max;

    futcache_crdt_t *crdt = NULL;
    futcache_crdt_create(&cfg, &crdt);
    free(anchors);
    return crdt;
}

/* (priority_from_point removed; cell merge priorities are currently
 * determined internally by the CRDT join rule.) */

/* merge_with_diagnostics: applies `updates` to `cache` and counts adopt-vs-
 * conflict. An "adopt" fills an empty cell; a "conflict" is a merge into an
 * already-occupied cell where the remote priority <= local priority (so the
 * local entry wins and no state change). Updates with remote priority >
 * local would replace the local entry, but with deterministic priorities
 * derived from the observed point value and round-robin distribution, this
 * case is rare. We count it as a conflict too (the cell state changed). */
static void merge_with_diagnostics(futcache_crdt_t *cache,
                                     const futcache_crdt_update_t *updates,
                                     size_t update_count,
                                     size_t *total_merges,
                                     size_t *total_novel_adopt,
                                     size_t *total_conflicts)
{
    for (size_t k = 0; k < update_count; ++k) {
        const void *existing = NULL;
        size_t existing_len = 0;
        bool was_occupied = (futcache_crdt_get_payload(
                                cache, updates[k].cell,
                                &existing, &existing_len) == FUTCACHE_OK)
                            && (existing != NULL);
        (*total_merges)++;
        futcache_crdt_merge(cache, &updates[k], 1);
        if (!was_occupied) {
            (*total_novel_adopt)++;
        } else {
            (*total_conflicts)++;
        }
    }
}

/* ============================================================
 * Single-worker baseline
 *
 * All points go to one CRDT. Reports final occupied-cell count.
 * ============================================================ */

typedef struct single_stats {
    size_t occupied_cells;
    size_t novel_observations;
    double seconds;
} single_stats_t;

static single_stats_t run_single(const workload_t *w, double epsilon)
{
    size_t anchor_count = 0;
    futcache_crdt_t *crdt = make_crdt(epsilon, w->domain_min,
                                       w->domain_max, &anchor_count);
    single_stats_t s = {0};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < w->count; ++i) {
        bool was_novel = false;
        size_t cell = 0;
        futcache_crdt_observe(crdt, &w->points[i], NULL, 0U,
                               &was_novel, &cell);
        if (was_novel) s.novel_observations++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    futcache_crdt_stats_t stats;
    futcache_crdt_get_stats(crdt, &stats);
    s.occupied_cells = stats.occupied_cells;
    s.seconds = (double)(t1.tv_sec - t0.tv_sec) +
                (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    futcache_crdt_destroy(crdt);
    return s;
}

/* ============================================================
 * Multi-worker fleet with gossip
 *
 * W workers, M gossip rounds. Each round:
 *   1. Each worker snapshots its occupied cells.
 *   2. Each worker merges a *gossip fan-in*: receives updates from
 *      a (random or full) subset of other workers.
 *
 * peers_per_round controls the gossip schedule:
 *   - 0                 : full fan-in (every peer, every round)
 *   - k >= 1            : each worker picks k peers uniformly at random
 *                         (without replacement per round), merges their
 *                         snapshot. Deterministic: same seed → same result.
 *
 * Joint coverage = sum of distinct occupied cells across all W workers
 * after the final gossip round (the geometric blackboard view).
 * Dedup ratio = joint / sum_local.
 * ============================================================ */

typedef struct fleet_stats {
    size_t joint_cells;
    size_t sum_local_cells;
    size_t gossip_rounds;
    size_t convergence_round;  /* first round where ALL workers reach target joint */
    size_t target_joint;       /* single-worker reference for convergence check */
    size_t peers_per_round;
    double merge_round_seconds;
    /* Conflict diagnostics: how many gossip messages actually changed state. */
    size_t total_merges;       /* total update_count summed across rounds */
    size_t total_novel_adopt;  /* updates that filled an empty cell */
    size_t total_conflicts;    /* updates where local entry had higher priority */
} fleet_stats_t;

/* Reproducible PRNG for gossip peer selection. xorshift64 seeded from
 * (worker_count, gossip_round) so each round's peer set is deterministic. */
static uint64_t fleet_rng_state = UINT64_C(0xabad1dea12345678);

static uint64_t fleet_xorshift(void)
{
    uint64_t x = fleet_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    fleet_rng_state = x;
    return x;
}

/* Pick `count` distinct indices from [0, worker_count), excluding `self`.
 * count must be <= worker_count - 1. */
static void select_peers(size_t self, size_t worker_count, size_t count,
                          size_t *out_peers)
{
    /* Reservoir sampling over worker_count-1 candidates. */
    size_t candidates = worker_count - 1;
    size_t picked = 0;
    size_t cidx = 0;
    for (size_t i = 0; i < worker_count; ++i) {
        if (i == self) continue;
        if (picked < count) {
            out_peers[picked++] = i;
        } else {
            /* Reservoir replacement: pick j uniformly in [0, cidx]. */
            size_t j = (size_t)(fleet_xorshift() % (uint64_t)(cidx + 1));
            if (j < count) out_peers[j] = i;
        }
        cidx++;
    }
    (void)candidates;
}

static size_t compute_joint(const futcache_crdt_t **workers,
                             size_t worker_count, size_t anchor_count,
                             futcache_crdt_update_t **batch_buf,
                             size_t *batch_capacity)
{
    size_t *occupied_marks = calloc(anchor_count, sizeof(size_t));
    for (size_t i = 0; i < worker_count; ++i) {
        size_t needed = 0;
        futcache_crdt_snapshot(workers[i], NULL, &needed);
        if (needed > *batch_capacity) {
            free(*batch_buf);
            *batch_buf = calloc(needed, sizeof(**batch_buf));
            *batch_capacity = needed;
        }
        size_t count = *batch_capacity;
        futcache_crdt_snapshot(workers[i], *batch_buf, &count);
        for (size_t k = 0; k < count; ++k) {
            occupied_marks[(*batch_buf)[k].cell]++;
        }
    }
    size_t joint = 0;
    for (size_t i = 0; i < anchor_count; ++i) {
        if (occupied_marks[i] > 0) joint++;
    }
    free(occupied_marks);
    return joint;
}

/* Returns true iff every worker has occupied_cells == target. This is the
 * real convergence test: gossip has propagated the full union to every node. */
static bool all_workers_reached(const futcache_crdt_t *const *workers,
                                 size_t worker_count, size_t target)
{
    for (size_t i = 0; i < worker_count; ++i) {
        futcache_crdt_stats_t ws;
        futcache_crdt_get_stats(workers[i], &ws);
        if (ws.occupied_cells != target) return false;
    }
    return true;
}

static fleet_stats_t run_fleet(const workload_t *w, double epsilon,
                                size_t worker_count, size_t gossip_rounds,
                                size_t peers_per_round, uint64_t seed,
                                size_t target_joint)
{
    fleet_stats_t fs = {0};
    fs.gossip_rounds = gossip_rounds;
    fs.peers_per_round = peers_per_round;
    fs.target_joint = target_joint;

    /* Seed the gossip PRNG from caller-provided seed for reproducibility. */
    fleet_rng_state = seed ^ ((uint64_t)worker_count * UINT64_C(0x9E3779B97F4A7C15));

    /* Allocate one CRDT per worker. */
    futcache_crdt_t **workers = calloc(worker_count, sizeof(*workers));
    size_t anchor_count = 0;
    for (size_t i = 0; i < worker_count; ++i) {
        workers[i] = make_crdt(epsilon, w->domain_min, w->domain_max,
                                &anchor_count);
    }

    /* Distribute points round-robin to workers. */
    for (size_t i = 0; i < w->count; ++i) {
        size_t w_idx = i % worker_count;
        size_t cell = 0;
        bool was_novel = false;
        futcache_crdt_observe(workers[w_idx], &w->points[i], NULL, 0U,
                               &was_novel, &cell);
    }

    /* Pre-gossip: every worker's local count may be less than target. */
    futcache_crdt_update_t *batch = NULL;
    size_t batch_capacity = 0;

    /* Effective peers per round: 0 → full fan-in (W-1). */
    size_t k_peers = peers_per_round == 0
        ? (worker_count > 0 ? worker_count - 1 : 0)
        : peers_per_round;
    if (k_peers > worker_count - 1) k_peers = worker_count - 1;

    size_t convergence_round = 0;
    double total_merge_seconds = 0.0;
    size_t total_merges = 0;
    size_t total_novel_adopt = 0;
    size_t total_conflicts = 0;

    size_t *peer_buf = (k_peers > 0)
        ? calloc(k_peers, sizeof(*peer_buf)) : NULL;

    /* Check pre-gossip convergence (W=1 trivially converges). */
    if (all_workers_reached((const futcache_crdt_t *const *)workers,
                             worker_count, target_joint)) {
        convergence_round = 0;
    }

    for (size_t round = 0; round < gossip_rounds; ++round) {
        struct timespec m0, m1;
        clock_gettime(CLOCK_MONOTONIC, &m0);

        for (size_t i = 0; i < worker_count; ++i) {
            size_t needed = 0;
            futcache_crdt_snapshot(workers[i], NULL, &needed);
            if (needed > batch_capacity) {
                free(batch);
                batch = calloc(needed, sizeof(*batch));
                batch_capacity = needed;
            }
            size_t count = batch_capacity;
            futcache_crdt_snapshot(workers[i], batch, &count);

            if (k_peers == 0) continue;  /* W=1 edge case */

            if (peers_per_round == 0) {
                /* Full fan-in: merge into every other worker. */
                for (size_t j = 0; j < worker_count; ++j) {
                    if (j == i) continue;
                    merge_with_diagnostics(workers[j], batch, count,
                                            &total_merges,
                                            &total_novel_adopt,
                                            &total_conflicts);
                }
            } else {
                select_peers(i, worker_count, k_peers, peer_buf);
                for (size_t p = 0; p < k_peers; ++p) {
                    merge_with_diagnostics(workers[peer_buf[p]], batch, count,
                                            &total_merges,
                                            &total_novel_adopt,
                                            &total_conflicts);
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &m1);
        total_merge_seconds += (double)(m1.tv_sec - m0.tv_sec) +
                               (double)(m1.tv_nsec - m0.tv_nsec) / 1e9;

        if (convergence_round == 0 &&
            all_workers_reached((const futcache_crdt_t *const *)workers,
                                 worker_count, target_joint)) {
            convergence_round = round + 1;
        }
    }
    free(peer_buf);

    /* Final joint + sum_local. */
    fs.joint_cells = compute_joint((const futcache_crdt_t **)workers,
                                    worker_count, anchor_count,
                                    &batch, &batch_capacity);
    fs.sum_local_cells = 0;
    for (size_t i = 0; i < worker_count; ++i) {
        futcache_crdt_stats_t ws;
        futcache_crdt_get_stats(workers[i], &ws);
        fs.sum_local_cells += ws.occupied_cells;
    }
    free(batch);

    fs.merge_round_seconds = gossip_rounds > 0
        ? total_merge_seconds / (double)gossip_rounds : 0.0;
    fs.convergence_round = convergence_round;
    fs.total_merges = total_merges;
    fs.total_novel_adopt = total_novel_adopt;
    fs.total_conflicts = total_conflicts;

    for (size_t i = 0; i < worker_count; ++i) {
        futcache_crdt_destroy(workers[i]);
    }
    free(workers);
    return fs;
}

/* ============================================================
 * main
 * ============================================================ */

int main(void)
{
    enum { N = 10000 };
    enum { GOSSIP_ROUNDS_FULL = 4 };  /* full-fan-in section */
    enum { GOSSIP_ROUNDS_RAND = 32 }; /* randomized section: more rounds to see convergence */

    workload_t *(*factories[])(size_t) = {
        workload_reciprocal, workload_uniform, workload_cluster,
        workload_alternating, workload_power,
    };
    size_t workload_count = sizeof(factories) / sizeof(factories[0]);

    /* Section 1: full-fan-in baseline. */
    size_t fleet_sizes_full[] = {1, 2, 4, 8, 16, 32};
    size_t fleet_count_full = sizeof(fleet_sizes_full) /
                              sizeof(fleet_sizes_full[0]);
    double epsilons[] = {0.05, 0.01, 0.005};
    size_t eps_count = sizeof(epsilons) / sizeof(epsilons[0]);

    /* Section 2: randomized schedule sweep at one fixed epsilon. */
    size_t fleet_sizes_rand[] = {8, 16, 32, 64, 128, 256};
    size_t fleet_count_rand = sizeof(fleet_sizes_rand) /
                              sizeof(fleet_sizes_rand[0]);
    /* Peers-per-round schedule: 1, 3, ceil(log2(W)). 0 = full fan-in (W-1). */
    enum { K_FIXED_1 = 1, K_FIXED_3 = 3 };

    puts("# CRDT fleet-replay benchmark");
    puts("# N = 10000 per workload, points distributed round-robin across W workers");
    puts("#");
    puts("# Section 1: full-fan-in gossip (every peer, every round).");
    puts("#   'single' = one worker observes the entire stream (joint upper bound).");
    puts("#   'fleet'  = W workers, full fan-in, 4 rounds.");
    puts("#");
    puts("# Section 2: randomized gossip schedule sweep.");
    puts("#   Each round, each worker picks k peers uniformly at random and");
    puts("#   merges their snapshot. Same deterministic seed across runs.");
    puts("#   Convergence is the round at which every worker has reached the");
    puts("#   single-worker reference (every cell propagated through the union).");
    puts("#");
    puts("# joint_cells   = union of occupied cells across the fleet after gossip");
    puts("# sum_local     = sum of per-worker occupied cell counts");
    puts("# dedup_ratio   = joint / sum_local (1.0 = no dedup benefit, <1 = sharing)");
    puts("# conv_round    = first round at which ALL workers reach single-worker joint");
    puts("# adopt/conflict: total gossip updates that filled an empty cell vs");
    puts("#                attempted to write an already-occupied cell.");
    puts("# merge_us      = average wall time of one full gossip round, microseconds");
    puts("#");

    /* ---------- Section 1: full-fan-in ---------- */
    puts("# === Section 1: full fan-in ===");
    puts("#");

    for (size_t w = 0; w < workload_count; ++w) {
        workload_t *wl = factories[w](N);
        printf("## Workload: %s (full fan-in)\n\n", wl->name);

        printf("| epsilon | fleet_size | joint | sum_local | dedup_ratio | conv_round | merge_us |\n");
        printf("|---------|------------|-------|-----------|-------------|------------|----------|\n");

        for (size_t e = 0; e < eps_count; ++e) {
            single_stats_t single = run_single(wl, epsilons[e]);

            for (size_t f = 0; f < fleet_count_full; ++f) {
                /* peers_per_round=0 → full fan-in (W-1). */
                fleet_stats_t fs = run_fleet(wl, epsilons[e],
                                              fleet_sizes_full[f],
                                              GOSSIP_ROUNDS_FULL,
                                              /*peers=*/0,
                                              /*seed=*/0xa5a5a5a5ULL,
                                              /*target=*/single.occupied_cells);
                double dedup = fs.sum_local_cells > 0
                    ? (double)fs.joint_cells / (double)fs.sum_local_cells
                    : 0.0;
                double merge_us = fs.merge_round_seconds * 1e6;
                printf("| %.4g   |    %3zu     | %5zu | %9zu |   %.4f    |  %2zu       | %7.1f  |\n",
                       epsilons[e], fleet_sizes_full[f],
                       fs.joint_cells, fs.sum_local_cells,
                       dedup, fs.convergence_round, merge_us);
            }
            printf("| %.4g   |   single   | %5zu | %9zu |   1.0000   |  n/a      |   n/a    |\n",
                   epsilons[e], single.occupied_cells, single.occupied_cells);
            printf("| ------ | ---------- | ----- | --------- | ----------- | ---------- | -------- |\n");
        }
        puts("");
        workload_free(wl);
    }

    /* ---------- Section 2: randomized schedule sweep ---------- */
    puts("# === Section 2: randomized gossip schedule sweep ===");
    puts("#");
    puts("# At W=8..256 and three schedules: k=1, k=3, k=ceil(log2(W)).");
    puts("# Each (workload, fleet, schedule) runs 32 rounds; convergence is the");
    puts("# first round at which every worker reaches the single-worker joint.");
    puts("#");

    double rand_epsilon = 0.01;  /* fixed for the sweep */

    for (size_t w = 0; w < workload_count; ++w) {
        workload_t *wl = factories[w](N);
        single_stats_t single = run_single(wl, rand_epsilon);

        printf("## Workload: %s   single_worker_joint=%zu   epsilon=%g\n\n",
               wl->name, single.occupied_cells, rand_epsilon);

        printf("| fleet | k=1     | k=3     | k=log2(W) | k=W-1     |\n");
        printf("|       | conv|mer | conv|mer | conv|mer  | conv|mer  |\n");
        printf("|-------|-----|------|-----|------|-----|-------|-----|------|\n");

        for (size_t f = 0; f < fleet_count_rand; ++f) {
            size_t W = fleet_sizes_rand[f];
            int log2W = 0;
            for (size_t t = W; t > 1; t >>= 1) log2W++;
            size_t k_log = (size_t)log2W;
            if (k_log < 1) k_log = 1;
            if (k_log > W - 1) k_log = W - 1;

            printf("|  %3zu  |", W);
            uint64_t seed = 0xdeadbeefULL ^ ((uint64_t)W * 0x9E3779B97F4A7C15ULL);

            /* k=1 */
            fleet_stats_t fs1 = run_fleet(wl, rand_epsilon, W,
                                           GOSSIP_ROUNDS_RAND,
                                           K_FIXED_1, seed,
                                           single.occupied_cells);
            printf(" %2zu |%5.0f |", fs1.convergence_round,
                   fs1.merge_round_seconds * 1e6);

            /* k=3 */
            fleet_stats_t fs3 = run_fleet(wl, rand_epsilon, W,
                                           GOSSIP_ROUNDS_RAND,
                                           K_FIXED_3, seed,
                                           single.occupied_cells);
            printf(" %2zu |%5.0f |", fs3.convergence_round,
                   fs3.merge_round_seconds * 1e6);

            /* k=log2(W) */
            fleet_stats_t fsL = run_fleet(wl, rand_epsilon, W,
                                           GOSSIP_ROUNDS_RAND,
                                           k_log, seed,
                                           single.occupied_cells);
            printf(" %2zu |%6.0f |", fsL.convergence_round,
                   fsL.merge_round_seconds * 1e6);

            /* k=W-1 (full fan-in baseline for comparison). */
            fleet_stats_t fsF = run_fleet(wl, rand_epsilon, W,
                                           GOSSIP_ROUNDS_RAND,
                                           /*peers=*/0, seed,
                                           single.occupied_cells);
            printf(" %2zu |%6.0f |\n", fsF.convergence_round,
                   fsF.merge_round_seconds * 1e6);
        }
        puts("");
        workload_free(wl);
    }

    /* ---------- Section 3: network-RTT model ---------- */
    puts("# === Section 3: network-RTT-adjusted latency ===");
    puts("#");
    puts("# The merge_round_seconds in Section 2 is in-process wall time; real");
    puts("# fleets pay per-message network RTT. This section projects the");
    puts("# Section 2 schedule costs forward, adding RTT = 1 ms (intra-DC) and");
    puts("# RTT = 10 ms (cross-DC) per merge message.");
    puts("#");
    puts("# Time to convergence = R * (k * RTT + merge_in_process_us),");
    puts("#   R = convergence rounds, k = peers/round.");
    puts("# This is the lower bound: assumes gossip messages can be pipelined");
    puts("# in parallel within a round (sender doesn't block on RTTs serially).");
    puts("#");

    double rtts_us[] = {1000.0, 10000.0};  /* 1ms intra-DC, 10ms cross-DC */
    const char *rtt_labels[] = {"1ms-intra-DC", "10ms-cross-DC"};
    size_t rtt_count = sizeof(rtts_us) / sizeof(rtts_us[0]);

    /* Re-run Section 2 schedules but project network cost. */
    double rand_epsilon3 = 0.01;
    for (size_t r = 0; r < rtt_count; ++r) {
        double rtt = rtts_us[r];
        printf("## RTT model: %s\n\n", rtt_labels[r]);
        printf("# Per-message cost = RTT; per-merge in-process cost from Section 2.\n");
        printf("# Total = R * (k * RTT + per_merge_in_process).\n\n");

        printf("| W | k=1 | k=3 | k=log2(W) | k=W-1 |\n");
        printf("|---|-----|-----|-----------|--------|\n");

        for (size_t f = 0; f < fleet_count_rand; ++f) {
            size_t W = fleet_sizes_rand[f];
            int log2W = 0;
            for (size_t t = W; t > 1; t >>= 1) log2W++;
            size_t k_log = (size_t)log2W;
            if (k_log < 1) k_log = 1;
            if (k_log > W - 1) k_log = W - 1;

            uint64_t seed = 0xdeadbeefULL ^ ((uint64_t)W * 0x9E3779B97F4A7C15ULL);
            /* Project from uniform (most representative) workload. */
            workload_t *wl = workload_uniform(N);

            printf("| %3zu |", W);
            fleet_stats_t fs1 = run_fleet(wl, rand_epsilon3, W,
                                           GOSSIP_ROUNDS_RAND,
                                           K_FIXED_1, seed, 50);
            double t1 = (double)fs1.convergence_round *
                        ((double)K_FIXED_1 * rtt +
                         fs1.merge_round_seconds * 1e6);
            printf(" %5.0f ms |", t1 / 1000.0);

            fleet_stats_t fs3 = run_fleet(wl, rand_epsilon3, W,
                                           GOSSIP_ROUNDS_RAND,
                                           K_FIXED_3, seed, 50);
            double t3 = (double)fs3.convergence_round *
                        ((double)K_FIXED_3 * rtt +
                         fs3.merge_round_seconds * 1e6);
            printf(" %5.0f ms |", t3 / 1000.0);

            fleet_stats_t fsL = run_fleet(wl, rand_epsilon3, W,
                                           GOSSIP_ROUNDS_RAND,
                                           k_log, seed, 50);
            double tL = (double)fsL.convergence_round *
                        ((double)k_log * rtt +
                         fsL.merge_round_seconds * 1e6);
            printf(" %6.0f ms |", tL / 1000.0);

            fleet_stats_t fsF = run_fleet(wl, rand_epsilon3, W,
                                           GOSSIP_ROUNDS_RAND,
                                           /*peers=*/0, seed, 50);
            double k_full = (double)(W - 1);
            if (k_full < 1.0) k_full = 1.0;
            double tF = (double)fsF.convergence_round *
                        (k_full * rtt +
                         fsF.merge_round_seconds * 1e6);
            printf(" %5.0f ms |\n", tF / 1000.0);

            workload_free(wl);
        }
        puts("");
    }

    /* ---------- Section 4: merge-conflict diagnostics ---------- */
    puts("# === Section 4: merge-conflict diagnostics ===");
    puts("#");
    puts("# For each gossip update, we record whether the target cell was empty");
    puts("# ('adopt', state changed) or already occupied ('conflict', local");
    puts("# entry may have been kept or replaced depending on priority).");
    puts("# In round-robin distribution with deterministic point priorities,");
    puts("# conflicts should be rare: most merges fill empty cells.");
    puts("#");

    /* Run a smaller fleet (16) at ε=0.01 to keep output readable. */
    enum { W4 = 16 };
    double eps4 = 0.01;
    workload_t *wl4 = workload_power(N);
    single_stats_t s4 = run_single(wl4, eps4);

    printf("## Fleet W=%d, ε=%g, single-worker joint=%zu\n\n",
           W4, eps4, s4.occupied_cells);
    printf("| schedule | rounds | total_merges | novel_adopt | conflicts | adopt_rate |\n");
    printf("|----------|--------|---------------|-------------|-----------|------------|\n");

    struct { const char *name; size_t k; } scheds4[] = {
        {"k=1", K_FIXED_1},
        {"k=3", K_FIXED_3},
        {"k=log2(W)", 4},
        {"k=W-1", 0},  /* full fan-in */
    };
    for (size_t s = 0; s < sizeof(scheds4)/sizeof(scheds4[0]); ++s) {
        uint64_t seed4 = 0xdeadbeefULL ^ ((uint64_t)W4 * 0x9E3779B97F4A7C15ULL);
        fleet_stats_t fs = run_fleet(wl4, eps4, W4, 32,
                                      scheds4[s].k,
                                      seed4,
                                      s4.occupied_cells);
        double adopt_rate = fs.total_merges > 0
            ? (double)fs.total_novel_adopt / (double)fs.total_merges
            : 0.0;
        printf("| %-8s |  %2zu    |  %10zu   |  %9zu   |  %8zu |   %.4f   |\n",
               scheds4[s].name, fs.convergence_round,
               fs.total_merges, fs.total_novel_adopt,
               fs.total_conflicts, adopt_rate);
    }
    puts("");
    workload_free(wl4);

    return EXIT_SUCCESS;
}

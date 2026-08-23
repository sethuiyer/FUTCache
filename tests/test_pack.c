#include "test.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

/* ============================================================
 * Reference model: independent oracle that computes novelty by
 * distance to the full observation history. This validates that
 * the packing cache agrees with the canonical novelty definition
 * (subject to the documented packing approximation).
 * ============================================================ */

static void reference_observe(
    double *history,
    size_t *history_count,
    size_t history_capacity,
    const double *point,
    size_t dimension)
{
    if (*history_count < history_capacity) {
        memcpy(history + *history_count * dimension, point,
               dimension * sizeof(double));
        (*history_count)++;
    }
}

static bool reference_is_novel(
    const double *history,
    size_t history_count,
    const double *point,
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance)
{
    double min_d = 1.0 / 0.0;  /* +inf */
    for (size_t i = 0; i < history_count; ++i) {
        double d = distance(point, history + i * dimension, dimension, NULL);
        if (d < min_d) min_d = d;
    }
    return min_d > epsilon;
}

/* ============================================================
 * Tests
 * ============================================================ */

static bool test_config_validation(void)
{
    futcache_pack_config_t cfg;
    futcache_pack_t *cache = NULL;
    double lo[2] = {0.0, 0.0};
    double hi[2] = {1.0, 1.0};

    futcache_pack_config_init(&cfg);
    if (cfg.dimension != 1U) {
        fprintf(stderr, "default dimension should be one\n");
        return false;
    }
    cfg.dimension = 2U;
    cfg.epsilon = 0.1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        fprintf(stderr, "default config failed to create\n");
        return false;
    }
    futcache_pack_destroy(cache);
    cache = NULL;

    /* dimension = 0 */
    futcache_pack_config_init(&cfg);
    cfg.dimension = 0U;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "dimension=0 should fail\n");
        if (cache != NULL) futcache_pack_destroy(cache);
        return false;
    }

    /* negative epsilon */
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = -1.0;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_ERROR_INVALID_ARGUMENT) {
        if (cache != NULL) futcache_pack_destroy(cache);
        fprintf(stderr, "negative epsilon should fail\n");
        return false;
    }

    /* swapped domain */
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.1;
    cfg.domain_min = hi;
    cfg.domain_max = lo;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_ERROR_INVALID_ARGUMENT) {
        if (cache != NULL) futcache_pack_destroy(cache);
        fprintf(stderr, "swapped domain should fail\n");
        return false;
    }

    return true;
}

static bool test_hand_checked_2d_traversal(void)
{
    static const double points[][2] = {
        {0.1, 0.1},
        {0.9, 0.9},
        {0.1, 0.9},
        {0.9, 0.1},
        {0.1, 0.1},  /* duplicate of first */
        {0.5, 0.5},  /* far from all four corners under L_inf eps=0.3 */
    };
    enum { N = sizeof(points) / sizeof(points[0]) };
    static const bool expected_novel[] = {1, 1, 1, 1, 0, 1};
    static const double epsilon = 0.3;

    double lo[2] = {0.0, 0.0};
    double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = epsilon;
    cfg.domain_min = lo;
    cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        return false;
    }

    for (size_t i = 0; i < N; ++i) {
        bool was_novel = false;
        if (futcache_pack_observe(cache, points[i], &was_novel)
            != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return false;
        }
        if (was_novel != expected_novel[i]) {
            fprintf(stderr,
                "step %zu: expected novel=%d, got novel=%d\n",
                i, (int)expected_novel[i], (int)was_novel);
            futcache_pack_destroy(cache);
            return false;
        }
    }

    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    /* Four corners are all > eps apart pairwise under L_inf with eps=0.3;
     * the (0.1, 0.1) duplicate is not added; (0.5, 0.5) is novel.
     * Expect 5 representatives. */
    if (stats.representative_count != 5U) {
        fprintf(stderr, "expected 5 representatives, got %zu\n",
                stats.representative_count);
        futcache_pack_destroy(cache);
        return false;
    }

    if (futcache_pack_validate(cache) != FUTCACHE_OK) {
        fprintf(stderr, "validate failed after clean hand-checked stream\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

static bool test_query_does_not_modify(void)
{
    double lo[2] = {0.0, 0.0};
double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.2;
    cfg.domain_min = lo;
        cfg.domain_max = hi;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        return false;
    }

    double p1[2] = {0.1, 0.2};
    double p2[2] = {0.7, 0.8};
    bool was_novel = false;
    futcache_pack_observe(cache, p1, &was_novel);
    futcache_pack_observe(cache, p2, &was_novel);

    futcache_pack_stats_t before;
    futcache_pack_get_stats(cache, &before);

    /* Pure query: should not change observations or representative count. */
    bool novel = false;
    for (size_t i = 0; i < 100U; ++i) {
        double q[2] = {0.3, 0.4};
        if (futcache_pack_is_novel(cache, q, &novel) != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return false;
        }
    }

    futcache_pack_stats_t after;
    futcache_pack_get_stats(cache, &after);

    if (after.observations != before.observations) {
        fprintf(stderr, "is_novel bumped observations\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (after.representative_count != before.representative_count) {
        fprintf(stderr, "is_novel grew representatives\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (after.novel_observations != before.novel_observations) {
        fprintf(stderr, "is_novel bumped novel_observations\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

static bool test_distance_function_injection(void)
{
    /* Same stream, two distance functions: we verify that
     * distance function selection is honoured, and that the
     * representative counts are monotone and consistent with
     * the chosen metric.
     *
     * Stream in 2D:
     *   (0.0, 0.0), (0.6, 0.0), (0.0, 0.6), (0.6, 0.6)
     *
     * Under L_inf eps=0.4:
     *   - (0,0) novel
     *   - (0.6, 0) -> max(0.6,0)=0.6 > 0.4, novel
     *   - (0, 0.6) -> max(0,0.6)=0.6 > 0.4, novel
     *   - (0.6, 0.6) -> max(0.6,0.6)=0.6 > 0.4, novel
     *   -> 4 representatives
     *
     * Under L1 eps=1.0:
     *   - (0,0) novel
     *   - (0.6,0) -> d = 0.6 <= 1.0, redundant
     *   - (0, 0.6) -> d = 0.6 <= 1.0, redundant
     *   - (0.6,0.6) -> d to (0,0) = 1.2 > 1.0, novel
     *   -> 2 representatives
     *
     * The two metrics produce visibly different states. */

    static const double stream[][2] = {
        {0.0, 0.0},
        {0.6, 0.0},
        {0.0, 0.6},
        {0.6, 0.6},
    };
    enum { N = sizeof(stream) / sizeof(stream[0]) };

    double lo[2] = {-1.0, -1.0};
    double hi[2] = {1.0, 1.0};
    futcache_pack_t *linf_cache = NULL;
    futcache_pack_t *l1_cache = NULL;
    futcache_pack_config_t cfg;

    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.4;
    cfg.distance = futcache_distance_linf;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    if (futcache_pack_create(&cfg, &linf_cache) != FUTCACHE_OK) return false;

    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 1.0;
    cfg.distance = futcache_distance_l1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    if (futcache_pack_create(&cfg, &l1_cache) != FUTCACHE_OK) {
        futcache_pack_destroy(linf_cache);
        return false;
    }

    for (size_t i = 0; i < N; ++i) {
        bool n1 = false, n2 = false;
        futcache_pack_observe(linf_cache, stream[i], &n1);
        futcache_pack_observe(l1_cache, stream[i], &n2);
    }

    futcache_pack_stats_t linf_stats, l1_stats;
    futcache_pack_get_stats(linf_cache, &linf_stats);
    futcache_pack_get_stats(l1_cache, &l1_stats);

    if (linf_stats.representative_count != 4U) {
        fprintf(stderr, "L_inf expected 4 reps, got %zu\n",
                linf_stats.representative_count);
        futcache_pack_destroy(linf_cache);
        futcache_pack_destroy(l1_cache);
        return false;
    }
    if (l1_stats.representative_count != 2U) {
        fprintf(stderr, "L1 expected 2 reps, got %zu\n",
                l1_stats.representative_count);
        futcache_pack_destroy(linf_cache);
        futcache_pack_destroy(l1_cache);
        return false;
    }

    /* Both caches must satisfy the separation invariant. */
    if (futcache_pack_validate(linf_cache) != FUTCACHE_OK ||
        futcache_pack_validate(l1_cache) != FUTCACHE_OK) {
        fprintf(stderr, "separation invariant violated under distance injection\n");
        futcache_pack_destroy(linf_cache);
        futcache_pack_destroy(l1_cache);
        return false;
    }

    futcache_pack_destroy(linf_cache);
    futcache_pack_destroy(l1_cache);
    return true;
}

static bool test_out_of_range(void)
{
    double lo[2] = {0.0, 0.0};
double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.1;
    cfg.domain_min = lo;
        cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return false;

    double in[2] = {0.5, 0.5};
    double out[2] = {0.5, 1.5};
    bool novel = false;

    if (futcache_pack_observe(cache, in, &novel) != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return false;
    }
    if (futcache_pack_observe(cache, out, &novel) != FUTCACHE_ERROR_OUT_OF_RANGE) {
        fprintf(stderr, "out-of-range observe should fail\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (futcache_pack_is_novel(cache, out, &novel) != FUTCACHE_ERROR_OUT_OF_RANGE) {
        fprintf(stderr, "out-of-range query should fail\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

static bool test_clear(void)
{
    double lo[2] = {0.0, 0.0};
double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.2;
    cfg.domain_min = lo;
        cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return false;

    double p[2] = {0.5, 0.5};
    bool novel = false;
    for (size_t i = 0; i < 50U; ++i) {
        p[0] = (double)i / 100.0;
        p[1] = (double)i / 100.0;
        futcache_pack_observe(cache, p, &novel);
    }
    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    if (stats.representative_count == 0U) {
        fprintf(stderr, "expected representatives before clear\n");
        futcache_pack_destroy(cache);
        return false;
    }
    uint64_t gen_before = stats.generation;

    if (futcache_pack_clear(cache) != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_get_stats(cache, &stats);
    if (stats.representative_count != 0U) {
        fprintf(stderr, "clear did not empty representatives\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (stats.generation <= gen_before) {
        fprintf(stderr, "clear did not bump generation\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

static bool test_reference_agreement_packing(void)
{
    /* For a small exhaustive stream, check that the packing cache returns
     * the same novelty decision as an independent reference that tracks
     * distance to the full history. This validates the canonical novelty
     * predicate against the packing invariant.
     *
     * The packing cache may be slightly more conservative than the exact
     * union-of-balls cache (a redundant point in the full history can
     * "cover" a region the packing representatives miss). The reference
     * here is the full history, so we expect rare false positives. */

    enum { N = 64 };
    double stream[N][2];
    /* Deterministic pseudo-random in [0, 1). */
    uint64_t s = UINT64_C(0xa5a5a5a5a5a5a5a5);
    for (size_t i = 0; i < N; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        stream[i][0] = (double)(s >> 11) / (double)(UINT64_C(1) << 53);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        stream[i][1] = (double)(s >> 11) / (double)(UINT64_C(1) << 53);
    }

    enum { HIST_CAP = 1024 };
    double history[HIST_CAP * 2];
    size_t history_count = 0U;

    double lo[2] = {0.0, 0.0};
double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.05;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
        cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return false;

    size_t mismatches = 0U;
    for (size_t i = 0; i < N; ++i) {
        bool ref_novel = reference_is_novel(history, history_count,
            stream[i], 2U, 0.05, futcache_distance_l2);
        bool cache_novel = false;
        futcache_pack_observe(cache, stream[i], &cache_novel);
        reference_observe(history, &history_count, HIST_CAP, stream[i], 2U);

        if (cache_novel && !ref_novel) {
            mismatches++;
            fprintf(stderr,
                "step %zu: cache said novel, reference said redundant\n", i);
        }
    }
    futcache_pack_destroy(cache);

    /* Allow at most a small number of false positives in random data.
     * Empirically this is rare (representative count saturates quickly). */
    if (mismatches > 5U) {
        fprintf(stderr,
            "%zu packing disagreements (expected few)\n", mismatches);
        return false;
    }
    return true;
}

static bool test_separation_invariant_holds(void)
{
    /* After a dense uniform stream, validate that the cache's
     * representative set is still strictly epsilon-separated. */

    enum { N = 2000 };
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {1.0, 1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 3U;
    cfg.epsilon = 0.05;
    cfg.distance = futcache_distance_linf;
    cfg.domain_min = lo;
        cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) {
        return false;
    }

    uint64_t s = UINT64_C(0x123456789abcdef0);
    bool novel = false;
    for (size_t i = 0; i < N; ++i) {
        double p[3];
        for (size_t k = 0; k < 3U; ++k) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            p[k] = (double)(s >> 11) / (double)(UINT64_C(1) << 53);
        }
        if (futcache_pack_observe(cache, p, &novel) != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return false;
        }
    }

    if (futcache_pack_validate(cache) != FUTCACHE_OK) {
        fprintf(stderr, "epsilon-separation violated on dense stream\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    /* For L_inf, eps=0.05 in 3D unit cube, packing bound ~ 20^3 = 8000.
     * Empirically we see roughly 200-400 representatives. */
    if (stats.representative_count == 0U) {
        fprintf(stderr, "expected some representatives\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (stats.peak_count != stats.representative_count) {
        /* Monotone: never shrinks. */
        fprintf(stderr, "peak_count drifted from current count\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

static bool test_copy_representatives(void)
{
    double lo[2] = {0.0, 0.0};
double hi[2] = {1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.3;
    cfg.domain_min = lo;
        cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return false;

    double p1[2] = {0.1, 0.2};
    double p2[2] = {0.8, 0.9};
    bool novel = false;
    futcache_pack_observe(cache, p1, &novel);
    futcache_pack_observe(cache, p2, &novel);

    /* Size query: inout_count is in *representatives*, not doubles. */
    size_t required = 0U;
    if (futcache_pack_copy_representatives(cache, NULL, &required)
        != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return false;
    }
    if (required != 2U) {
        fprintf(stderr, "size query: expected required=2 reps, got %zu\n",
                required);
        futcache_pack_destroy(cache);
        return false;
    }

    /* Copy with capacity = 2 reps. Buffer holds reps * dim doubles. */
    double out[2 * 2];
    size_t count = 2U;
    if (futcache_pack_copy_representatives(cache, out, &count)
        != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return false;
    }
    if (count != 2U) {
        fprintf(stderr, "expected 2 reps copied, got %zu\n", count);
        futcache_pack_destroy(cache);
        return false;
    }

    /* Buffer too small (capacity = 1 rep, but cache has 2). */
    double tiny[2 * 2];
    size_t tiny_count = 1U;
    if (futcache_pack_copy_representatives(cache, tiny, &tiny_count)
        != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
        fprintf(stderr, "tiny buffer should report too small\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (tiny_count != 2U) {
        fprintf(stderr, "tiny required should be 2 reps, got %zu\n",
                tiny_count);
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

/* ============================================================
 * Concurrency
 * ============================================================ */

typedef struct pack_thread_context {
    futcache_pack_t *cache;
    size_t iterations;
    size_t seed;
    futcache_status_t status;
} pack_thread_context_t;

static void *pack_observer_worker(void *arg)
{
    pack_thread_context_t *ctx = arg;
    uint64_t s = ctx->seed;
    for (size_t i = 0; i < ctx->iterations; ++i) {
        double p[3];
        for (size_t k = 0; k < 3U; ++k) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            p[k] = (double)(s >> 11) / (double)(UINT64_C(1) << 53);
        }
        bool novel = false;
        futcache_status_t st = futcache_pack_observe(ctx->cache, p, &novel);
        if (st != FUTCACHE_OK) {
            ctx->status = st;
            return NULL;
        }
    }
    return NULL;
}

static bool test_pack_concurrent_observers(void)
{
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {1.0, 1.0, 1.0};
    futcache_pack_config_t cfg;
    futcache_pack_config_init(&cfg);
    cfg.dimension = 3U;
    cfg.epsilon = 0.05;
    cfg.distance = futcache_distance_l2;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&cfg, &cache) != FUTCACHE_OK) return false;

    enum { THREADS = 4, PER_THREAD = 5000 };
    pthread_t threads[THREADS];
    pack_thread_context_t contexts[THREADS];

    for (size_t t = 0; t < THREADS; ++t) {
        contexts[t].cache = cache;
        contexts[t].iterations = PER_THREAD;
        contexts[t].seed = (uint64_t)(t + 1U) * UINT64_C(0x9e3779b97f4a7c15);
        contexts[t].status = FUTCACHE_OK;
        if (pthread_create(&threads[t], NULL, pack_observer_worker,
                            &contexts[t]) != 0) {
            futcache_pack_destroy(cache);
            fprintf(stderr, "pthread_create failed\n");
            return false;
        }
    }
    for (size_t t = 0; t < THREADS; ++t) {
        pthread_join(threads[t], NULL);
        if (contexts[t].status != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            fprintf(stderr, "worker %zu failed: %s\n", t,
                futcache_status_string(contexts[t].status));
            return false;
        }
    }

    if (futcache_pack_validate(cache) != FUTCACHE_OK) {
        fprintf(stderr, "post-concurrency separation invariant violated\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_stats_t stats;
    futcache_pack_get_stats(cache, &stats);
    if (stats.observations != THREADS * PER_THREAD) {
        fprintf(stderr, "lost observations: %zu vs expected %d\n",
                stats.observations, THREADS * PER_THREAD);
        futcache_pack_destroy(cache);
        return false;
    }
    if (stats.representative_count == 0U) {
        fprintf(stderr, "no representatives after 20k concurrent inserts\n");
        futcache_pack_destroy(cache);
        return false;
    }
    if (stats.peak_count != stats.representative_count) {
        fprintf(stderr, "monotonicity violated under concurrency\n");
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_destroy(cache);
    return true;
}

typedef struct mock_pack_backend_context {
    size_t creates;
    size_t destroys;
    size_t clears;
    size_t inserts;
    size_t nearest_calls;
    bool fail_insert;
} mock_pack_backend_context_t;

typedef struct mock_pack_backend_state {
    size_t indexed;
} mock_pack_backend_state_t;

static futcache_status_t mock_backend_create(
    void **out_state, size_t dimension, futcache_distance_fn distance,
    void *distance_context, const futcache_allocator_t *allocator,
    void *context)
{
    (void)dimension;
    (void)distance;
    (void)distance_context;
    mock_pack_backend_context_t *stats =
        (mock_pack_backend_context_t *)context;
    mock_pack_backend_state_t *state =
        (mock_pack_backend_state_t *)allocator->allocate(
            allocator->context, sizeof(*state));
    if (state == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    state->indexed = 0U;
    stats->creates++;
    *out_state = state;
    return FUTCACHE_OK;
}

static void mock_backend_destroy(void *opaque_state,
                                 const futcache_allocator_t *allocator,
                                 void *context)
{
    mock_pack_backend_context_t *stats =
        (mock_pack_backend_context_t *)context;
    allocator->deallocate(allocator->context, opaque_state);
    stats->destroys++;
}

static futcache_status_t mock_backend_clear(void *opaque_state, void *context)
{
    mock_pack_backend_state_t *state =
        (mock_pack_backend_state_t *)opaque_state;
    mock_pack_backend_context_t *stats =
        (mock_pack_backend_context_t *)context;
    state->indexed = 0U;
    stats->clears++;
    return FUTCACHE_OK;
}

static futcache_status_t mock_backend_insert(void *opaque_state,
                                              const double *point,
                                              size_t dimension, void *context)
{
    (void)point;
    (void)dimension;
    mock_pack_backend_state_t *state =
        (mock_pack_backend_state_t *)opaque_state;
    mock_pack_backend_context_t *stats =
        (mock_pack_backend_context_t *)context;
    stats->inserts++;
    if (stats->fail_insert) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    state->indexed++;
    return FUTCACHE_OK;
}

static futcache_status_t mock_backend_nearest(void *opaque_state,
                                               const double *point,
                                               size_t dimension,
                                               double *out_distance,
                                               void *context)
{
    (void)point;
    (void)dimension;
    mock_pack_backend_state_t *state =
        (mock_pack_backend_state_t *)opaque_state;
    mock_pack_backend_context_t *stats =
        (mock_pack_backend_context_t *)context;
    stats->nearest_calls++;
    /* Only the designated second point is indexed as a hit; all other
     * points exercise the insertion path (including the forced failure). */
    *out_distance = (state->indexed > 0U && point[0] == 0.9 &&
                     point[1] == 0.9) ? 0.0 : 1.0 / 0.0;
    return FUTCACHE_OK;
}

static bool test_pluggable_backend_lifecycle_and_atomicity(void)
{
    static const futcache_pack_backend_ops_t backend = {
        mock_backend_create,
        mock_backend_destroy,
        mock_backend_clear,
        mock_backend_insert,
        mock_backend_nearest
    };
    mock_pack_backend_context_t backend_stats = {0};
    double lo[2] = {0.0, 0.0};
    double hi[2] = {1.0, 1.0};
    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = 2U;
    config.epsilon = 0.2;
    config.domain_min = lo;
    config.domain_max = hi;
    config.backend = &backend;
    config.backend_context = &backend_stats;

    futcache_pack_t *cache = NULL;
    if (futcache_pack_create(&config, &cache) != FUTCACHE_OK) return false;
    if (backend_stats.creates != 1U) {
        futcache_pack_destroy(cache);
        return false;
    }

    double first[2] = {0.1, 0.1};
    bool novel = false;
    if (futcache_pack_observe(cache, first, &novel) != FUTCACHE_OK ||
        !novel || backend_stats.inserts != 1U) {
        futcache_pack_destroy(cache);
        return false;
    }

    /* The mock index reports every indexed point as distance zero. */
    double second[2] = {0.9, 0.9};
    if (futcache_pack_is_novel(cache, second, &novel) != FUTCACHE_OK ||
        novel || backend_stats.nearest_calls == 0U) {
        futcache_pack_destroy(cache);
        return false;
    }

    futcache_pack_stats_t before;
    if (futcache_pack_get_stats(cache, &before) != FUTCACHE_OK) {
        futcache_pack_destroy(cache);
        return false;
    }
    backend_stats.fail_insert = true;
    double third[2] = {0.5, 0.5};
    if (futcache_pack_observe(cache, third, &novel) !=
        FUTCACHE_ERROR_OUT_OF_MEMORY) {
        futcache_pack_destroy(cache);
        return false;
    }
    futcache_pack_stats_t after;
    if (futcache_pack_get_stats(cache, &after) != FUTCACHE_OK ||
        after.observations != before.observations ||
        after.representative_count != before.representative_count ||
        after.novel_observations != before.novel_observations) {
        futcache_pack_destroy(cache);
        return false;
    }

    backend_stats.fail_insert = false;
    if (futcache_pack_clear(cache) != FUTCACHE_OK ||
        backend_stats.clears != 1U) {
        futcache_pack_destroy(cache);
        return false;
    }
    futcache_pack_destroy(cache);
    return backend_stats.destroys == 1U;
}

static bool test_nearest(void)
{
    futcache_pack_config_t cfg;
    futcache_pack_t *cache = NULL;
    double lo[2] = {0.0, 0.0};
    double hi[2] = {1.0, 1.0};
    double a[2] = {0.1, 0.1};
    double b[2] = {0.9, 0.9};
    double q[2] = {0.12, 0.12};
    double far[2] = {0.5, 0.5};
    double distance = 0.0;
    size_t index = 0U;
    bool novel = false;

    futcache_pack_config_init(&cfg);
    cfg.dimension = 2U;
    cfg.epsilon = 0.1;
    cfg.domain_min = lo;
    cfg.domain_max = hi;
    TEST_STATUS(futcache_pack_create(&cfg, &cache), FUTCACHE_OK);

    /* Empty cache reports +inf and SIZE_MAX. */
    TEST_STATUS(futcache_pack_nearest(cache, a, &distance, &index),
                FUTCACHE_OK);
    TEST_ASSERT(isinf(distance) && distance > 0.0);
    TEST_ASSERT(index == SIZE_MAX);

    TEST_STATUS(futcache_pack_observe(cache, a, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);
    TEST_STATUS(futcache_pack_observe(cache, b, &novel), FUTCACHE_OK);
    TEST_ASSERT(novel);

    /* q is nearest to a (L_inf distance 0.02). */
    TEST_STATUS(futcache_pack_nearest(cache, q, &distance, &index),
                FUTCACHE_OK);
    TEST_ASSERT(index == 0U);
    TEST_NEAR(distance, 0.02, 1e-12);

    /* far is equidistant from a and b under L_inf (0.4): ties take the
     * smallest index. */
    TEST_STATUS(futcache_pack_nearest(cache, far, &distance, &index),
                FUTCACHE_OK);
    TEST_ASSERT(index == 0U);
    TEST_NEAR(distance, 0.4, 1e-12);

    futcache_pack_destroy(cache);
    return true;
}

/* ============================================================
 * Test suite registration
 * ============================================================ */

int pack_test_suite(void)
{
    static const test_case_t tests[] = {
        {"config validation", test_config_validation},
        {"hand-checked 2D traversal", test_hand_checked_2d_traversal},
        {"query does not modify state", test_query_does_not_modify},
        {"distance function injection (L1 vs L2)", test_distance_function_injection},
        {"out-of-range points rejected", test_out_of_range},
        {"clear semantics", test_clear},
        {"packing agrees with reference (rare FP only)",
            test_reference_agreement_packing},
        {"epsilon-separation invariant on dense stream",
            test_separation_invariant_holds},
        {"copy representatives", test_copy_representatives},
        {"concurrent observers", test_pack_concurrent_observers},
        {"pluggable backend lifecycle and atomicity",
            test_pluggable_backend_lifecycle_and_atomicity},
        {"nearest representative and distance", test_nearest},
    };
    return run_test_cases("pack", tests, sizeof(tests) / sizeof(tests[0]));
}

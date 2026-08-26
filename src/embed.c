#define _POSIX_C_SOURCE 200809L

#include "futcache/embed.h"
#include "futcache/futcache.h"
#include "futcache/pack.h"
#include "futcache/crdt.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Default allocator (matches pack.c / crdt.c pattern)
 * ============================================================ */

static void *default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

/* ============================================================
 * Embedding object
 * ============================================================ */

struct futcache_embed {
    size_t dimension;       /* original space dimension d */
    size_t anchor_count;    /* number of anchors m (embedded dimension) */
    double *anchors;        /* [anchor_count * dimension], owned copy */
    futcache_distance_fn distance;   /* metric on original space */
    void *distance_context;
    double covering_radius; /* estimated delta (lower bound) */
    bool covering_radius_certified;

    futcache_allocator_t owner_allocator;
    futcache_allocator_t allocator;
};

static bool normalize_allocator(
    const futcache_allocator_t *requested,
    futcache_allocator_t *normalized)
{
    if (normalized == NULL) {
        return false;
    }
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = default_allocate;
        normalized->deallocate = default_deallocate;
        normalized->context = NULL;
        return true;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL) {
        return false;
    }
    normalized->allocate = requested->allocate;
    normalized->deallocate = requested->deallocate;
    normalized->context = requested->context;
    return true;
}

/* ============================================================
 * futcache_embed_create / destroy
 * ============================================================ */

futcache_status_t futcache_embed_create(
    const futcache_embed_config_t *config,
    futcache_embed_t **out_embed)
{
    if (config == NULL || out_embed == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_embed = NULL;

    if (config->dimension == 0U || config->anchor_count == 0U) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (config->anchors == NULL || config->domain_min == NULL ||
        config->domain_max == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->covering_radius_upper_bound) ||
        config->covering_radius_upper_bound < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    for (size_t coordinate = 0U; coordinate < config->dimension;
         ++coordinate) {
        if (!isfinite(config->domain_min[coordinate]) ||
            !isfinite(config->domain_max[coordinate]) ||
            config->domain_max[coordinate] <= config->domain_min[coordinate]) {
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
        }
    }
    size_t anchor_values;
    if (config->anchor_count > SIZE_MAX / config->dimension) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    anchor_values = config->anchor_count * config->dimension;
    for (size_t value = 0U; value < anchor_values; ++value) {
        if (!isfinite(config->anchors[value])) {
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
        }
    }
    /* Overflow check: anchor_count * dimension must fit in size_t. */
    if (anchor_values >
        (SIZE_MAX - sizeof(futcache_embed_t)) / sizeof(double)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_allocator_t allocator;
    if (!normalize_allocator(&config->allocator, &allocator)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    size_t anchor_bytes = anchor_values * sizeof(double);
    if (anchor_bytes > SIZE_MAX - sizeof(futcache_embed_t)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_embed_t *embed = (futcache_embed_t *)allocator.allocate(
        allocator.context, sizeof(futcache_embed_t) + anchor_bytes);
    if (embed == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    embed->dimension = config->dimension;
    embed->anchor_count = config->anchor_count;
    embed->anchors = (double *)(embed + 1);
    memcpy(embed->anchors, config->anchors, anchor_bytes);
    embed->distance = config->distance != NULL
        ? config->distance : futcache_distance_linf;
    embed->distance_context = config->distance_context;
    embed->covering_radius = 0.0;
    embed->covering_radius_certified = false;
    embed->owner_allocator = allocator;
    embed->allocator = allocator;

    /* Estimate covering radius by sampling. Use the same probe strategy as
     * the CRDT engine: deterministic quasi-random Halton probe points.
     * This gives a lower bound on the true covering radius. */
    {
        size_t probe_count = config->anchor_count > SIZE_MAX / 10U
            ? SIZE_MAX : config->anchor_count * 10U;
        if (probe_count < 1000U) probe_count = 1000U;
        if (probe_count > 100000U) probe_count = 100000U;

        futcache_status_t st = futcache_crdt_estimate_covering_radius(
            embed->anchors,
            embed->anchor_count,
            embed->dimension,
            config->domain_min,
            config->domain_max,
            embed->distance,
            embed->distance_context,
            probe_count,
            &embed->covering_radius);
        if (st != FUTCACHE_OK) {
            allocator.deallocate(allocator.context, embed);
            return st;
        }
        if (isfinite(config->covering_radius_upper_bound) &&
            config->covering_radius_upper_bound > 0.0) {
            if (config->covering_radius_upper_bound < embed->covering_radius) {
                allocator.deallocate(allocator.context, embed);
                return FUTCACHE_ERROR_INVALID_ARGUMENT;
            }
            embed->covering_radius = config->covering_radius_upper_bound;
            embed->covering_radius_certified = true;
        }
    }

    *out_embed = embed;
    return FUTCACHE_OK;
}

void futcache_embed_destroy(futcache_embed_t *embed)
{
    if (embed == NULL) return;
    embed->allocator.deallocate(embed->allocator.context, embed);
}

/* ============================================================
 * futcache_embed_point
 * ============================================================ */

futcache_status_t futcache_embed_point(
    const futcache_embed_t *embed,
    const double *point,
    double *out_embedded)
{
    if (embed == NULL || point == NULL || out_embedded == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    for (size_t coordinate = 0U; coordinate < embed->dimension; ++coordinate) {
        if (!isfinite(point[coordinate])) {
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
    }
    for (size_t i = 0; i < embed->anchor_count; ++i) {
        const double *anchor = embed->anchors + i * embed->dimension;
        double coordinate = embed->distance(point, anchor,
                                             embed->dimension,
                                             embed->distance_context);
        if (!isfinite(coordinate) || coordinate < 0.0) {
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
        out_embedded[i] = coordinate;
    }
    return FUTCACHE_OK;
}

/* ============================================================
 * Accessors
 * ============================================================ */

double futcache_embed_covering_radius(const futcache_embed_t *embed)
{
    if (embed == NULL) return 0.0;
    return embed->covering_radius;
}

size_t futcache_embed_anchor_count(const futcache_embed_t *embed)
{
    if (embed == NULL) return 0;
    return embed->anchor_count;
}

size_t futcache_embed_original_dimension(const futcache_embed_t *embed)
{
    if (embed == NULL) return 0;
    return embed->dimension;
}

futcache_status_t futcache_embed_adjusted_epsilon(
    const futcache_embed_t *embed,
    double epsilon_original,
    double *out_epsilon_embedded)
{
    if (embed == NULL || out_epsilon_embedded == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(epsilon_original) || epsilon_original < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!embed->covering_radius_certified) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    double delta = embed->covering_radius;
    double adjusted = epsilon_original - 2.0 * delta;
    if (adjusted <= 0.0) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    *out_epsilon_embedded = adjusted;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_embed_distance — L_inf distance in embedded space
 * ============================================================ */

double futcache_embed_distance(const double *a, const double *b,
                                size_t dimension, void *context)
{
    (void)context;
    double max_d = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        double diff = a[i] - b[i];
        if (diff < 0.0) diff = -diff;
        if (diff > max_d) max_d = diff;
    }
    return max_d;
}

/* ============================================================
 * futcache_embed_pack_create — convenience wrapper
 * ============================================================ */

futcache_status_t futcache_embed_pack_create(
    size_t original_dimension,
    double epsilon_original,
    futcache_distance_fn original_distance,
    void *original_distance_context,
    const double *domain_min,
    const double *domain_max,
    double target_radius,
    futcache_crdt_anchor_strategy_t strategy,
    size_t max_anchors,
    size_t probe_count,
    const futcache_allocator_t *allocator,
    futcache_embed_t **out_embed,
    futcache_pack_t **out_cache)
{
    if (out_embed == NULL || out_cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_embed = NULL;
    *out_cache = NULL;
    if (original_dimension == 0U || max_anchors == 0U ||
        domain_min == NULL || domain_max == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(epsilon_original) || epsilon_original <= 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(target_radius) || target_radius <= 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (strategy != FUTCACHE_CRDT_ANCHOR_GRID) {
        /* A sampled Halton radius is a lower bound, so it cannot certify the
         * distortion adjustment performed by this convenience constructor. */
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    /* Step 1: Generate anchors with covering radius <= target_radius. */
    double *anchors = NULL;
    size_t anchor_count = 0;
    double covering_radius = 0.0;
    if (max_anchors > SIZE_MAX / original_dimension ||
        max_anchors * original_dimension > SIZE_MAX / sizeof(double)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    size_t alloc_bytes = max_anchors * original_dimension * sizeof(double);
    anchors = (double *)malloc(alloc_bytes);
    if (anchors == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    futcache_status_t st = futcache_crdt_generate_safe_anchors(
        original_dimension,
        2.0 * target_radius, /* epsilon for generate_safe_anchors */
        domain_min,
        domain_max,
        original_distance,
        original_distance_context,
        strategy,
        max_anchors,
        probe_count,
        anchors,
        &anchor_count,
        &covering_radius);
    if (st != FUTCACHE_OK) {
        free(anchors);
        return st;
    }

    /* Step 2: Create the embedding. */
    futcache_embed_config_t econfig;
    memset(&econfig, 0, sizeof(econfig));
    econfig.dimension = original_dimension;
    econfig.anchor_count = anchor_count;
    econfig.anchors = anchors;
    econfig.distance = original_distance != NULL
        ? original_distance : futcache_distance_linf;
    econfig.distance_context = original_distance_context;
    econfig.domain_min = domain_min;
    econfig.domain_max = domain_max;
    econfig.covering_radius_upper_bound = covering_radius;
    if (allocator != NULL) econfig.allocator = *allocator;

    futcache_embed_t *embed = NULL;
    st = futcache_embed_create(&econfig, &embed);
    if (st != FUTCACHE_OK) {
        free(anchors);
        return st;
    }

    free(anchors);
    anchors = NULL;

    /* Step 3: Adjust epsilon for distortion. */
    double epsilon_embedded;
    st = futcache_embed_adjusted_epsilon(embed, epsilon_original,
                                          &epsilon_embedded);
    if (st != FUTCACHE_OK) {
        futcache_embed_destroy(embed);
        return st;
    }

    /* Step 4: Create the pack cache in the embedded space. */
    /* Embedded domain bounds: each coordinate is a distance in [0, max_dist].
     * The max distance between any two points in the domain is bounded by
     * the diagonal. For L_inf it's ||domain_max - domain_min||_inf.
     * For other metrics we use a conservative upper bound. */
    double *emb_domain_min = (double *)calloc(anchor_count, sizeof(double));
    double *emb_domain_max = (double *)malloc(anchor_count * sizeof(double));
    if (emb_domain_min == NULL || emb_domain_max == NULL) {
        free(emb_domain_min);
        free(emb_domain_max);
        futcache_embed_destroy(embed);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    futcache_distance_fn distance = original_distance != NULL
        ? original_distance : futcache_distance_linf;
    /* Compute a certified coordinate bound for the built-in norm metrics.
     * For other metrics DBL_MAX is conservative and avoids rejecting a valid
     * embedded point on the basis of an unproved corner heuristic. */
    for (size_t i = 0; i < anchor_count; ++i) {
        emb_domain_min[i] = 0.0;
        const double *anchor = (const double *)(embed->anchors +
                                                i * original_dimension);
        double max_dist = 0.0;
        if (distance == futcache_distance_l1 ||
            distance == futcache_distance_l2 ||
            distance == futcache_distance_linf) {
            double accumulator = 0.0;
            for (size_t coordinate = 0U; coordinate < original_dimension;
                 ++coordinate) {
                double dlo = fabs(anchor[coordinate] - domain_min[coordinate]);
                double dhi = fabs(anchor[coordinate] - domain_max[coordinate]);
                double far = dlo > dhi ? dlo : dhi;
                if (distance == futcache_distance_l1) accumulator += far;
                else if (distance == futcache_distance_l2) accumulator += far * far;
                else if (far > accumulator) accumulator = far;
            }
            max_dist = distance == futcache_distance_l2
                ? sqrt(accumulator) : accumulator;
        } else {
            max_dist = DBL_MAX;
        }
        emb_domain_max[i] = max_dist;
    }

    futcache_pack_config_t pconfig;
    futcache_pack_config_init(&pconfig);
    pconfig.dimension = anchor_count;
    pconfig.epsilon = epsilon_embedded;
    pconfig.distance = futcache_embed_distance;
    pconfig.distance_context = embed;
    pconfig.domain_min = emb_domain_min;
    pconfig.domain_max = emb_domain_max;
    pconfig.backend = &futcache_pack_vptree_backend;
    if (allocator != NULL) pconfig.allocator = *allocator;

    futcache_pack_t *cache = NULL;
    st = futcache_pack_create(&pconfig, &cache);
    if (st != FUTCACHE_OK) {
        free(emb_domain_min);
        free(emb_domain_max);
        futcache_embed_destroy(embed);
        return st;
    }

    free(emb_domain_min);
    free(emb_domain_max);

    *out_embed = embed;
    *out_cache = cache;
    return FUTCACHE_OK;
}

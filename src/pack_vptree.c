/*
 * Exact scapegoat VP-tree for fixed- and variable-radius packing caches.
 *
 * Each node borrows its immutable coordinate vector from the owning cache;
 * it never duplicates the d-dimensional embedding. The cache clears this
 * derived index before it recycles a representative, so the pointer lifetime
 * is explicit and safe. Index memory is therefore O(n), not O(n*d).
 *
 * Besides exact nearest-neighbour search, every node records two bounds:
 *
 *   centre_radius       max d(vantage, centre) in the subtree
 *   max_accept_radius   max acceptance radius in the subtree
 *
 * A subtree cannot contain a matching ball when
 *
 *   d(query, vantage) - centre_radius > max_accept_radius.
 *
 * This makes adaptive lookup exact even when the nearest centre owns a ball
 * too small to contain the query but a farther centre owns a larger one.
 */

#include "futcache/pack.h"
#include "pack_vptree_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define VP_ALPHA (1.0 / log2(4.0 / 3.0))
#define VP_NORM_TOLERANCE (1e-6)

typedef struct vp_item {
    const double *coordinates;
    double acceptance_radius;
    size_t index;
} vp_item_t;

typedef struct vp_node {
    double mu;
    double centre_radius;
    double acceptance_radius;
    double acceptance_radius_internal;
    double max_acceptance_radius_internal;
    size_t size;
    size_t height;
    size_t index;
    struct vp_node *inside;
    struct vp_node *outside;
    const double *coordinates;
} vp_node_t;

typedef struct vp_tree_state {
    vp_node_t *root;
    size_t count;
    size_t dimension;
    futcache_distance_fn distance;
    futcache_distance_fn internal;
    void *distance_context;
    futcache_allocator_t allocator;
    bool cosine_mode;
    bool degenerate;
} vp_tree_state_t;

static double internal_dist(const vp_tree_state_t *state,
                            const double *a, const double *b)
{
    return state->internal(a, b, state->dimension, state->distance_context);
}

static double radius_to_internal(const vp_tree_state_t *state, double radius)
{
    return state->cosine_mode ? sqrt(2.0 * radius) : radius;
}

static double distance_to_external(const vp_tree_state_t *state,
                                   double distance)
{
    return state->cosine_mode ? (distance * distance) / 2.0 : distance;
}

static vp_node_t *node_create(const futcache_allocator_t *allocator,
                              const vp_tree_state_t *state,
                              const vp_item_t *item)
{
    vp_node_t *node = (vp_node_t *)allocator->allocate(
        allocator->context, sizeof(*node));
    if (node == NULL) return NULL;
    node->mu = 0.0;
    node->centre_radius = 0.0;
    node->acceptance_radius = item->acceptance_radius;
    node->acceptance_radius_internal = radius_to_internal(
        state, item->acceptance_radius);
    node->max_acceptance_radius_internal =
        node->acceptance_radius_internal;
    node->size = 1U;
    node->height = 1U;
    node->index = item->index;
    node->inside = NULL;
    node->outside = NULL;
    node->coordinates = item->coordinates;
    return node;
}

static void subtree_free(const futcache_allocator_t *allocator,
                         vp_node_t *node)
{
    if (node == NULL) return;
    subtree_free(allocator, node->inside);
    subtree_free(allocator, node->outside);
    allocator->deallocate(allocator->context, node);
}

static void swap_item(vp_item_t *left, vp_item_t *right)
{
    vp_item_t temporary = *left;
    *left = *right;
    *right = temporary;
}

static void swap_item_distance(vp_item_t *items, double *distances,
                               size_t left, size_t right)
{
    if (left == right) return;
    swap_item(&items[left], &items[right]);
    double temporary = distances[left];
    distances[left] = distances[right];
    distances[right] = temporary;
}

static size_t quickselect(vp_item_t *items, double *distances,
                          size_t lo, size_t hi, size_t target)
{
    while (lo < hi) {
        /* Three-way partitioning makes an equidistant metric linear here;
         * Lomuto partitioning would peel off one equal value per pass and
         * turn a rebuild into O(n^2). */
        double pivot = distances[lo + (hi - lo) / 2U];
        size_t below = lo;
        size_t current = lo;
        size_t above = hi;
        while (current <= above) {
            if (distances[current] < pivot) {
                swap_item_distance(items, distances, current, below);
                ++current;
                ++below;
            } else if (distances[current] > pivot) {
                swap_item_distance(items, distances, current, above);
                --above;
            } else {
                ++current;
            }
        }
        if (target < below) {
            hi = below - 1U;
        } else if (target > above) {
            lo = above + 1U;
        } else {
            return target;
        }
    }
    return lo;
}

/* Build uses one caller-owned distance scratch vector for the whole
 * recursion. Parent distances are dead once its partition is complete, so
 * the same memory can be reused by each child. */
static vp_node_t *build(vp_item_t *items, size_t count,
                        const vp_tree_state_t *state,
                        const futcache_allocator_t *allocator,
                        double *distance_scratch)
{
    if (count == 0U) return NULL;
    if (count == 1U) return node_create(allocator, state, &items[0]);

    size_t vantage = 0U;
    double farthest = -1.0;
    for (size_t index = 1U; index < count; ++index) {
        double distance = internal_dist(
            state, items[0].coordinates, items[index].coordinates);
        if (distance > farthest) {
            farthest = distance;
            vantage = index;
        }
    }
    swap_item(&items[0], &items[vantage]);

    distance_scratch[0] = 0.0;
    double centre_radius = 0.0;
    for (size_t index = 1U; index < count; ++index) {
        double distance = internal_dist(
            state, items[0].coordinates, items[index].coordinates);
        distance_scratch[index] = distance;
        if (distance > centre_radius) centre_radius = distance;
    }

    size_t median_target = 1U + (count - 2U) / 2U;
    size_t median = quickselect(items, distance_scratch, 1U, count - 1U,
                                median_target);
    double mu = distance_scratch[median];

    /* quickselect already leaves values below/above the median on the two
     * sides. Split by rank rather than moving every value equal to mu inside;
     * this keeps equidistant data balanced. Search correctness uses subtree
     * metric-ball bounds, so equal-to-mu points may safely occur on either
     * side. */
    size_t partition = median + 1U;

    vp_node_t *node = node_create(allocator, state, &items[0]);
    if (node == NULL) return NULL;
    node->mu = mu;
    node->centre_radius = centre_radius;
    node->inside = build(items + 1U, partition - 1U, state, allocator,
                         distance_scratch);
    if (partition > 1U && node->inside == NULL) {
        subtree_free(allocator, node);
        return NULL;
    }
    node->outside = build(items + partition, count - partition, state,
                          allocator, distance_scratch);
    if (count > partition && node->outside == NULL) {
        subtree_free(allocator, node);
        return NULL;
    }
    node->size = count;
    {
        size_t inside_height = node->inside != NULL
            ? node->inside->height : 0U;
        size_t outside_height = node->outside != NULL
            ? node->outside->height : 0U;
        node->height = 1U + (inside_height > outside_height
                            ? inside_height : outside_height);
    }
    if (node->inside != NULL &&
        node->inside->max_acceptance_radius_internal >
            node->max_acceptance_radius_internal) {
        node->max_acceptance_radius_internal =
            node->inside->max_acceptance_radius_internal;
    }
    if (node->outside != NULL &&
        node->outside->max_acceptance_radius_internal >
            node->max_acceptance_radius_internal) {
        node->max_acceptance_radius_internal =
            node->outside->max_acceptance_radius_internal;
    }
    return node;
}

static void collect(const vp_node_t *node, vp_item_t *items, size_t *index)
{
    if (node == NULL) return;
    items[*index].coordinates = node->coordinates;
    items[*index].acceptance_radius = node->acceptance_radius;
    items[*index].index = node->index;
    ++(*index);
    collect(node->inside, items, index);
    collect(node->outside, items, index);
}

static futcache_status_t rebuild_with_insert(vp_node_t **slot,
                                             const vp_item_t *item,
                                             vp_tree_state_t *state)
{
    const futcache_allocator_t *allocator = &state->allocator;
    size_t new_count = (*slot)->size + 1U;
    if (new_count > SIZE_MAX / sizeof(vp_item_t) ||
        new_count > SIZE_MAX / sizeof(double)) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    vp_item_t *items = (vp_item_t *)allocator->allocate(
        allocator->context, new_count * sizeof(*items));
    if (items == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    double *distance_scratch = (double *)allocator->allocate(
        allocator->context, new_count * sizeof(*distance_scratch));
    if (distance_scratch == NULL) {
        allocator->deallocate(allocator->context, items);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    size_t index = 0U;
    collect(*slot, items, &index);
    items[index++] = *item;
    if (index != new_count) {
        allocator->deallocate(allocator->context, distance_scratch);
        allocator->deallocate(allocator->context, items);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    vp_node_t *replacement = build(items, new_count, state, allocator,
                                   distance_scratch);
    allocator->deallocate(allocator->context, distance_scratch);
    allocator->deallocate(allocator->context, items);
    if (replacement == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    subtree_free(allocator, *slot);
    *slot = replacement;
    return FUTCACHE_OK;
}

static futcache_status_t vp_insert(vp_node_t **slot,
                                   const vp_item_t *item,
                                   vp_tree_state_t *state)
{
    vp_node_t *node = *slot;
    if (node == NULL) {
        vp_node_t *leaf = node_create(&state->allocator, state, item);
        if (leaf == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        *slot = leaf;
        return FUTCACHE_OK;
    }

    size_t new_size = node->size + 1U;
    if ((double)node->height > VP_ALPHA * log2((double)new_size)) {
        return rebuild_with_insert(slot, item, state);
    }

    double distance = internal_dist(state, item->coordinates,
                                    node->coordinates);
    vp_node_t **child = distance <= node->mu
        ? &node->inside : &node->outside;
    futcache_status_t status = vp_insert(child, item, state);
    if (status != FUTCACHE_OK) return status;

    node->size++;
    if (distance > node->centre_radius) node->centre_radius = distance;
    double internal_radius = radius_to_internal(
        state, item->acceptance_radius);
    if (internal_radius > node->max_acceptance_radius_internal) {
        node->max_acceptance_radius_internal = internal_radius;
    }
    {
        size_t inside_height = node->inside != NULL
            ? node->inside->height : 0U;
        size_t outside_height = node->outside != NULL
            ? node->outside->height : 0U;
        node->height = 1U + (inside_height > outside_height
                            ? inside_height : outside_height);
    }
    return FUTCACHE_OK;
}

static void nearest_search(const vp_node_t *node, const double *query,
                           const vp_tree_state_t *state, double *best)
{
    if (node == NULL) return;
    double distance = internal_dist(state, query, node->coordinates);
    if (distance < *best) *best = distance;
    if (distance - node->centre_radius >= *best) return;

    if (distance < node->mu) {
        nearest_search(node->inside, query, state, best);
        if (distance + *best > node->mu) {
            nearest_search(node->outside, query, state, best);
        }
    } else {
        nearest_search(node->outside, query, state, best);
        if (distance - *best < node->mu) {
            nearest_search(node->inside, query, state, best);
        }
    }
}

static void linear_nearest(const vp_node_t *node, const double *query,
                           const vp_tree_state_t *state, double *best)
{
    if (node == NULL) return;
    double distance = state->distance(
        query, node->coordinates, state->dimension,
        state->distance_context);
    if (distance < *best) *best = distance;
    linear_nearest(node->inside, query, state, best);
    linear_nearest(node->outside, query, state, best);
}

typedef struct vp_match {
    bool found;
    double distance_internal;
    double distance_external;
    size_t index;
} vp_match_t;

static bool match_is_better(const vp_match_t *match,
                            double distance_external, size_t index)
{
    return !match->found || distance_external < match->distance_external ||
        (distance_external == match->distance_external &&
         index < match->index);
}

static void nearest_index_search(const vp_node_t *node, const double *query,
                                 const vp_tree_state_t *state,
                                 vp_match_t *match)
{
    if (node == NULL) return;
    double distance = internal_dist(state, query, node->coordinates);
    double lower_bound = distance - node->centre_radius;
    if (lower_bound < 0.0) lower_bound = 0.0;
    if (match->found && lower_bound > match->distance_internal) return;

    double external = distance_to_external(state, distance);
    if (match_is_better(match, external, node->index)) {
        match->found = true;
        match->distance_internal = distance;
        match->distance_external = external;
        match->index = node->index;
    }
    if (distance < node->mu) {
        nearest_index_search(node->inside, query, state, match);
        nearest_index_search(node->outside, query, state, match);
    } else {
        nearest_index_search(node->outside, query, state, match);
        nearest_index_search(node->inside, query, state, match);
    }
}

static void linear_nearest_index(const vp_node_t *node, const double *query,
                                 const vp_tree_state_t *state,
                                 vp_match_t *match)
{
    if (node == NULL) return;
    double distance = state->distance(
        query, node->coordinates, state->dimension,
        state->distance_context);
    if (match_is_better(match, distance, node->index)) {
        match->found = true;
        match->distance_external = distance;
        match->distance_internal = distance;
        match->index = node->index;
    }
    linear_nearest_index(node->inside, query, state, match);
    linear_nearest_index(node->outside, query, state, match);
}

static void covering_search(const vp_node_t *node, const double *query,
                            const vp_tree_state_t *state, vp_match_t *match)
{
    if (node == NULL) return;
    double distance = internal_dist(state, query, node->coordinates);
    double lower_bound = distance - node->centre_radius;
    if (lower_bound < 0.0) lower_bound = 0.0;
    if (lower_bound > node->max_acceptance_radius_internal ||
        (match->found && lower_bound > match->distance_internal)) {
        return;
    }

    if (distance <= node->acceptance_radius_internal) {
        double external = distance_to_external(state, distance);
        if (match_is_better(match, external, node->index)) {
            match->found = true;
            match->distance_internal = distance;
            match->distance_external = external;
            match->index = node->index;
        }
    }

    if (distance < node->mu) {
        covering_search(node->inside, query, state, match);
        covering_search(node->outside, query, state, match);
    } else {
        covering_search(node->outside, query, state, match);
        covering_search(node->inside, query, state, match);
    }
}

static void linear_covering(const vp_node_t *node, const double *query,
                            const vp_tree_state_t *state, vp_match_t *match)
{
    if (node == NULL) return;
    double distance = state->distance(
        query, node->coordinates, state->dimension,
        state->distance_context);
    if (distance <= node->acceptance_radius &&
        match_is_better(match, distance, node->index)) {
        match->found = true;
        match->distance_external = distance;
        match->distance_internal = distance;
        match->index = node->index;
    }
    linear_covering(node->inside, query, state, match);
    linear_covering(node->outside, query, state, match);
}

static bool is_unit_norm(const double *point, size_t dimension)
{
    double squared_norm = 0.0;
    for (size_t index = 0U; index < dimension; ++index) {
        squared_norm += point[index] * point[index];
    }
    return fabs(squared_norm - 1.0) <= VP_NORM_TOLERANCE;
}

static futcache_status_t vptree_create(
    void **out_state, size_t dimension, futcache_distance_fn distance,
    void *distance_context, const futcache_allocator_t *allocator,
    void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)allocator->allocate(
        allocator->context, sizeof(*state));
    if (state == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    state->root = NULL;
    state->count = 0U;
    state->dimension = dimension;
    state->distance = distance != NULL ? distance : futcache_distance_linf;
    state->distance_context = distance_context;
    state->allocator = *allocator;
    state->cosine_mode = state->distance == futcache_distance_cosine;
    state->internal = state->cosine_mode
        ? futcache_distance_l2 : state->distance;
    state->degenerate = false;
    *out_state = state;
    return FUTCACHE_OK;
}

static void vptree_destroy(void *opaque_state,
                           const futcache_allocator_t *allocator,
                           void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    subtree_free(allocator, state->root);
    allocator->deallocate(allocator->context, state);
}

static futcache_status_t vptree_clear(void *opaque_state, void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    subtree_free(&state->allocator, state->root);
    state->root = NULL;
    state->count = 0U;
    state->degenerate = false;
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_vptree_insert_ball_internal(
    void *opaque_state,
    const double *point,
    size_t dimension,
    double radius,
    size_t index,
    void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension || !isfinite(radius) || radius < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    vp_item_t item = {point, radius, index};
    futcache_status_t status = vp_insert(&state->root, &item, state);
    if (status == FUTCACHE_OK) {
        state->count++;
        if (state->cosine_mode && !is_unit_norm(point, dimension)) {
            state->degenerate = true;
        }
    }
    return status;
}

static futcache_status_t vptree_insert(void *opaque_state,
                                       const double *point,
                                       size_t dimension, void *context)
{
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    return futcache_pack_vptree_insert_ball_internal(
        opaque_state, point, dimension, 0.0, state->count, context);
}

static futcache_status_t vptree_nearest(void *opaque_state,
                                        const double *point,
                                        size_t dimension,
                                        double *out_distance,
                                        void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension || out_distance == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (state->root == NULL) {
        *out_distance = INFINITY;
        return FUTCACHE_OK;
    }
    if (state->cosine_mode &&
        (state->degenerate || !is_unit_norm(point, dimension))) {
        double best = INFINITY;
        linear_nearest(state->root, point, state, &best);
        *out_distance = best;
        return FUTCACHE_OK;
    }
    double best = INFINITY;
    nearest_search(state->root, point, state, &best);
    *out_distance = distance_to_external(state, best);
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_vptree_covering_internal(
    void *opaque_state,
    const double *point,
    size_t dimension,
    bool *out_found,
    double *out_distance,
    size_t *out_index,
    void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension || out_found == NULL ||
        out_distance == NULL || out_index == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    vp_match_t match = {false, INFINITY, INFINITY, SIZE_MAX};
    if (state->root != NULL) {
        if (state->cosine_mode &&
            (state->degenerate || !is_unit_norm(point, dimension))) {
            linear_covering(state->root, point, state, &match);
        } else {
            covering_search(state->root, point, state, &match);
        }
    }
    *out_found = match.found;
    *out_distance = match.distance_external;
    *out_index = match.index;
    return FUTCACHE_OK;
}

futcache_status_t futcache_pack_vptree_nearest_index_internal(
    void *opaque_state,
    const double *point,
    size_t dimension,
    double *out_distance,
    size_t *out_index,
    void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension || out_distance == NULL ||
        out_index == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    vp_match_t match = {false, INFINITY, INFINITY, SIZE_MAX};
    if (state->root != NULL) {
        if (state->cosine_mode &&
            (state->degenerate || !is_unit_norm(point, dimension))) {
            linear_nearest_index(state->root, point, state, &match);
        } else {
            nearest_index_search(state->root, point, state, &match);
        }
    }
    *out_distance = match.distance_external;
    *out_index = match.index;
    return FUTCACHE_OK;
}

const futcache_pack_backend_ops_t futcache_pack_vptree_backend = {
    vptree_create,
    vptree_destroy,
    vptree_clear,
    vptree_insert,
    vptree_nearest
};

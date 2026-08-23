/*
 * Scapegoat VP-tree nearest-neighbour backend for the packing cache.
 *
 * Design
 * ------
 *
 * A vantage-point tree stores one representative per node as the
 * "vantage point" together with a split radius mu. The inside subtree
 * holds every point at distance <= mu from the vantage; the outside
 * subtree holds the rest. Because the points are static once inserted,
 * the invariants dist(vp, p) <= mu (inside) and dist(vp, p) > mu
 * (outside) hold forever, and the triangle inequality gives exact
 * pruning bounds for nearest-neighbour search:
 *
 *   inside points:  dist(q, p) >= |dist(q, vp) - dist(vp, p)| >= |d - mu|
 *   outside points: dist(q, p) >= dist(vp, p) - d > mu - d
 *
 * Unlike a k-d tree there are no axis-aligned splits, so the structure
 * is valid for arbitrary metric spaces (L1, L2, L_inf, custom metrics)
 * and its search cost tracks the intrinsic dimensionality of the data
 * (the doubling constant), not the ambient dimension.
 *
 * The tree is a scapegoat tree: inserts are amortized O(log n) with a
 * hard height bound (height <= alpha * log2(size), alpha = 1/log2(4/3)),
 * so adversarial insertion orders cannot degenerate the tree into a
 * chain. Rebuilds are atomic: the subtree is collected, a fresh tree is
 * built, and only then is the old subtree detached and freed. An
 * allocation failure anywhere leaves the tree exactly as it was.
 *
 * Cosine
 * ------
 *
 * 1 - dot(a, b) is not a metric (no triangle inequality), so the
 * pruning bounds above do not apply to it directly. But for
 * L2-normalized inputs the chordal distance |a - b| satisfies
 *
 *   |a - b|^2 = 2 - 2 dot(a, b)   =>   1 - dot = |a - b|^2 / 2,
 *
 * and the chordal distance is a metric (Euclidean distance on the unit
 * sphere). The backend therefore indexes cosine workloads with the
 * chordal metric and converts the returned distance back, which is
 * exact. To keep the one-sided guarantee when the normalization
 * contract is violated (inputs that are not unit-norm), the backend
 * detects the violation at insert time and per query and falls back to
 * an exact linear scan using the engine's cosine distance for those
 * points: a degenerate input can cause extra LLM calls (false novelty)
 * but can never turn a genuinely novel point into a cache hit.
 */

#include "futcache/pack.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Node: variable-size struct with inline coordinates, mirroring
 * pack_representative_t in pack.c (one allocation per node).
 * ============================================================ */

typedef struct vp_node {
    double mu;                 /* inside subtree: dist(vp, p) <= mu */
    double radius;             /* ball bound: all subtree pts within
                                  radius of the vantage (center) */
    size_t size;               /* points in this subtree (incl. self) */
    size_t height;             /* subtree height (leaf = 1) */
    struct vp_node *inside;
    struct vp_node *outside;
    size_t dimension;
    double coordinates[];      /* [dimension] */
} vp_node_t;

/* Scapegoat balance: height <= alpha * log2(size). */
#define VP_ALPHA (1.0 / log2(4.0 / 3.0))

/* Normalization tolerance for the cosine contract. */
#define VP_NORM_TOLERANCE (1e-6)

typedef struct vp_tree_state {
    vp_node_t *root;
    size_t count;
    size_t dimension;
    futcache_distance_fn distance;   /* engine distance (may be cosine) */
    futcache_distance_fn internal;   /* chordal (l2) when cosine_mode */
    void *distance_context;
    futcache_allocator_t allocator;
    bool cosine_mode;
    bool degenerate;                 /* non-unit-norm input seen */
} vp_tree_state_t;

/* ============================================================
 * Node helpers
 * ============================================================ */

static vp_node_t *node_create(const futcache_allocator_t *allocator,
                              size_t dimension, const double *point)
{
    vp_node_t *node = (vp_node_t *)allocator->allocate(
        allocator->context, sizeof(vp_node_t) + dimension * sizeof(double));
    if (node == NULL) return NULL;
    node->mu = 0.0;
    node->radius = 0.0;
    node->size = 1U;
    node->height = 1U;
    node->inside = NULL;
    node->outside = NULL;
    node->dimension = dimension;
    memcpy(node->coordinates, point, dimension * sizeof(double));
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

static double internal_dist(const vp_tree_state_t *state,
                            const double *a, const double *b)
{
    return state->internal(a, b, state->dimension, state->distance_context);
}

/* ============================================================
 * Balanced build from an array of point pointers.
 *
 * Vantage point = the point farthest from the first point
 * (deterministic, spreads well); mu = median distance to the vantage.
 * Recurses until singletons. On allocation failure the partially
 * built tree is freed and NULL is returned; the caller's `pts` array
 * is owned by the caller.
 * ============================================================ */

static size_t quickselect(const double **pts, double *dists,
                          size_t lo, size_t hi, size_t k)
{
    for (;;) {
        if (lo >= hi) return lo;
        double pivot = dists[hi];
        size_t i = lo;
        for (size_t j = lo; j < hi; ++j) {
            if (dists[j] <= pivot) {
                const double *tp = pts[j]; pts[j] = pts[i]; pts[i] = tp;
                double td = dists[j]; dists[j] = dists[i]; dists[i] = td;
                ++i;
            }
        }
        {
            const double *tp = pts[hi]; pts[hi] = pts[i]; pts[i] = tp;
            double td = dists[hi]; dists[hi] = dists[i]; dists[i] = td;
        }
        if (i == k) return i;
        if (i < k) {
            lo = i + 1U;
        } else {
            hi = i - 1U;   /* i >= lo >= 1 in all call sites */
        }
    }
}

static vp_node_t *build(const double **pts, size_t n,
                        const vp_tree_state_t *state,
                        const futcache_allocator_t *allocator)
{
    if (n == 0U) return NULL;
    if (n == 1U) {
        return node_create(allocator, state->dimension, pts[0]);
    }

    /* Vantage: farthest point from pts[0]. */
    size_t v = 0U;
    double best_d = -1.0;
    for (size_t i = 1U; i < n; ++i) {
        double d = internal_dist(state, pts[0], pts[i]);
        if (d > best_d) {
            best_d = d;
            v = i;
        }
    }
    {
        const double *tp = pts[0]; pts[0] = pts[v]; pts[v] = tp;
    }

    double *dists = (double *)allocator->allocate(
        allocator->context, n * sizeof(double));
    if (dists == NULL) return NULL;
    dists[0] = 0.0;
    for (size_t i = 1U; i < n; ++i) {
        dists[i] = internal_dist(state, pts[0], pts[i]);
    }

    /* Median of dists[1..n-1]; permutes pts and dists in parallel. */
    size_t med = quickselect(pts, dists, 1U, n - 1U, (n - 1U) / 2U);
    double mu = dists[med];

    /* Partition [1..n-1] into inside (d <= mu) / outside (d > mu). */
    size_t k = 1U;
    for (size_t i = 1U; i < n; ++i) {
        if (dists[i] <= mu) {
            const double *tp = pts[i]; pts[i] = pts[k]; pts[k] = tp;
            double td = dists[i]; dists[i] = dists[k]; dists[k] = td;
            ++k;
        }
    }

    vp_node_t *node = node_create(allocator, state->dimension, pts[0]);
    if (node == NULL) {
        allocator->deallocate(allocator->context, dists);
        return NULL;
    }
    node->mu = mu;
    node->inside = build(pts + 1U, k - 1U, state, allocator);
    if (k > 1U && node->inside == NULL) {
        allocator->deallocate(allocator->context, dists);
        subtree_free(allocator, node);
        return NULL;
    }
    node->outside = build(pts + k, n - k, state, allocator);
    if (n > k && node->outside == NULL) {
        allocator->deallocate(allocator->context, dists);
        subtree_free(allocator, node);
        return NULL;
    }
    /* Ball bound: every point in this subtree is within `radius` of the
     * vantage. Bottom-up: child points are within child_radius of the
     * child vantage, which is dist(vp, child_vp) + child_radius away. */
    node->radius = mu;
    if (node->inside != NULL) {
        double r = internal_dist(state, node->coordinates,
                                 node->inside->coordinates)
                   + node->inside->radius;
        if (r > node->radius) node->radius = r;
    }
    if (node->outside != NULL) {
        double r = internal_dist(state, node->coordinates,
                                 node->outside->coordinates)
                   + node->outside->radius;
        if (r > node->radius) node->radius = r;
    }
    node->size = n;
    {
        size_t h_in = node->inside != NULL ? node->inside->height : 0U;
        size_t h_out = node->outside != NULL ? node->outside->height : 0U;
        node->height = 1U + (h_in > h_out ? h_in : h_out);
    }
    allocator->deallocate(allocator->context, dists);
    return node;
}

/* ============================================================
 * Atomic scapegoat insert.
 *
 * The scapegoat check runs BEFORE any mutation: when a node on the
 * insertion path would exceed the height bound, the subtree rooted
 * there (plus the new point) is collected, rebuilt, and swapped in.
 * All allocations happen before the swap, so a failure leaves the
 * tree untouched. The common (no-rebuild) path allocates a single
 * leaf and mutates sizes/heights along the way.
 * ============================================================ */

static void collect(const vp_node_t *node, const double **out, size_t *idx)
{
    if (node == NULL) return;
    out[(*idx)++] = node->coordinates;
    collect(node->inside, out, idx);
    collect(node->outside, out, idx);
}

static futcache_status_t vp_insert(vp_node_t **slot, const double *point,
                                   vp_tree_state_t *state)
{
    const futcache_allocator_t *a = &state->allocator;
    vp_node_t *node = *slot;

    if (node == NULL) {
        vp_node_t *leaf = node_create(a, state->dimension, point);
        if (leaf == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        *slot = leaf;
        return FUTCACHE_OK;
    }

    /* Would this subtree exceed the height bound after the insert? */
    size_t new_size = node->size + 1U;
    if ((double)node->height > VP_ALPHA * log2((double)new_size)) {
        const double **pts = (const double **)a->allocate(
            a->context, new_size * sizeof(const double *));
        if (pts == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        size_t idx = 0U;
        collect(node, pts, &idx);
        pts[idx++] = point;
        if (idx != new_size) {
            a->deallocate(a->context, pts);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        vp_node_t *rebuilt = build(pts, new_size, state, a);
        a->deallocate(a->context, pts);
        if (rebuilt == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        subtree_free(a, node);
        *slot = rebuilt;
        return FUTCACHE_OK;
    }

    double d = internal_dist(state, point, node->coordinates);
    vp_node_t **child = (d <= node->mu) ? &node->inside : &node->outside;
    futcache_status_t st = vp_insert(child, point, state);
    if (st != FUTCACHE_OK) return st;
    node->size++;
    if (d > node->radius) node->radius = d;
    {
        size_t h_in = node->inside != NULL ? node->inside->height : 0U;
        size_t h_out = node->outside != NULL ? node->outside->height : 0U;
        node->height = 1U + (h_in > h_out ? h_in : h_out);
    }
    return FUTCACHE_OK;
}

/* ============================================================
 * Exact nearest-neighbour search with triangle-inequality pruning.
 * ============================================================ */

static void nn_search(const vp_node_t *node, const double *query,
                      const vp_tree_state_t *state, double *best)
{
    if (node == NULL) return;

    double d = internal_dist(state, query, node->coordinates);
    if (d < *best) *best = d;

    /* Ball prune: every point in this subtree is within `radius` of the
     * vantage, so nothing here can beat `best` when d - radius >= best. */
    if (d - node->radius >= *best) return;

    if (d < node->mu) {
        nn_search(node->inside, query, state, best);
        if (d + *best > node->mu) {
            nn_search(node->outside, query, state, best);
        }
    } else {
        nn_search(node->outside, query, state, best);
        if (d - *best < node->mu) {
            nn_search(node->inside, query, state, best);
        }
    }
}

/* Exact fallback for degenerate cosine inputs: walk every point using
 * the engine's own distance function. Recursive like nn_search; depth
 * is bounded by the scapegoat height bound. */
static void linear_scan_walk(const vp_node_t *node, const double *query,
                             const vp_tree_state_t *state, double *best)
{
    if (node == NULL) return;
    double d = state->distance(query, node->coordinates, state->dimension,
                               state->distance_context);
    if (d < *best) *best = d;
    linear_scan_walk(node->inside, query, state, best);
    linear_scan_walk(node->outside, query, state, best);
}

static double linear_scan_all(const vp_tree_state_t *state,
                              const double *query)
{
    double best = INFINITY;
    linear_scan_walk(state->root, query, state, &best);
    return best;
}

/* ============================================================
 * Cosine contract: unit-norm check.
 * ============================================================ */

static bool is_unit_norm(const double *point, size_t dimension)
{
    double sum = 0.0;
    for (size_t i = 0U; i < dimension; ++i) {
        sum += point[i] * point[i];
    }
    return fabs(sum - 1.0) <= VP_NORM_TOLERANCE;
}

/* ============================================================
 * Backend ops
 * ============================================================ */

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
    state->cosine_mode = (state->distance == futcache_distance_cosine);
    /* For cosine, index with the chordal (L2) metric, which is a true
     * metric on the unit sphere and satisfies 1 - dot = d^2 / 2. */
    state->internal = state->cosine_mode ? futcache_distance_l2
                                         : state->distance;
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

static futcache_status_t vptree_insert(void *opaque_state,
                                       const double *point,
                                       size_t dimension, void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (state->cosine_mode && !state->degenerate &&
        !is_unit_norm(point, dimension)) {
        state->degenerate = true;   /* fall back to the linear scan */
    }
    futcache_status_t st = vp_insert(&state->root, point, state);
    if (st == FUTCACHE_OK) state->count++;
    return st;
}

static futcache_status_t vptree_nearest(void *opaque_state,
                                        const double *point,
                                        size_t dimension,
                                        double *out_distance,
                                        void *context)
{
    (void)context;
    vp_tree_state_t *state = (vp_tree_state_t *)opaque_state;
    if (dimension != state->dimension) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (state->count == 0U || state->root == NULL) {
        *out_distance = INFINITY;
        return FUTCACHE_OK;
    }

    if (state->cosine_mode) {
        /* The chordal -> cosine conversion is only valid for unit-norm
         * inputs; otherwise scan exactly with the engine distance. */
        if (state->degenerate ||
            !is_unit_norm(point, dimension)) {
            *out_distance = linear_scan_all(state, point);
            return FUTCACHE_OK;
        }
        double best = INFINITY;
        nn_search(state->root, point, state, &best);
        *out_distance = (best * best) / 2.0;   /* 1 - dot = d^2 / 2 */
        return FUTCACHE_OK;
    }

    double best = INFINITY;
    nn_search(state->root, point, state, &best);
    *out_distance = best;
    return FUTCACHE_OK;
}

const futcache_pack_backend_ops_t futcache_pack_vptree_backend = {
    vptree_create,
    vptree_destroy,
    vptree_clear,
    vptree_insert,
    vptree_nearest
};

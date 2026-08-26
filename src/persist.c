#define _POSIX_C_SOURCE 200809L

#include "futcache/persist.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Default allocator
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

static bool normalize_allocator(
    const futcache_allocator_t *requested, futcache_allocator_t *normalized)
{
    if (normalized == NULL) return false;
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = default_allocate;
        normalized->deallocate = default_deallocate;
        normalized->context = NULL;
        return true;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL)
        return false;
    normalized->allocate = requested->allocate;
    normalized->deallocate = requested->deallocate;
    normalized->context = requested->context;
    return true;
}

/* ============================================================
 * Prime table
 * ============================================================ */

#define PERSIST_PRIME_TABLE_SIZE 10000
static uint64_t prime_table_[PERSIST_PRIME_TABLE_SIZE];
static bool prime_table_init_ = false;

static uint64_t next_prime(uint64_t n)
{
    if (n < 2) return 2;
    uint64_t c = n + 1;
    if (c <= 2) return 2;
    if ((c & 1U) == 0U) c++;
    for (;;) {
        bool ok = true;
        for (uint64_t d = 3; d * d <= c; d += 2) {
            if (c % d == 0) { ok = false; break; }
        }
        if (ok && c % 2 != 0) return c;
        if (ok && c == 2) return 2;
        c += 2;
    }
}

static void init_primes(void)
{
    if (prime_table_init_) return;
    uint64_t p = 0;
    for (size_t i = 0; i < PERSIST_PRIME_TABLE_SIZE; ++i) {
        p = next_prime(p);
        prime_table_[i] = p;
    }
    prime_table_init_ = true;
}

uint64_t futcache_persist_nth_prime(size_t i)
{
    init_primes();
    if (i < PERSIST_PRIME_TABLE_SIZE) return prime_table_[i];
    uint64_t p = prime_table_[PERSIST_PRIME_TABLE_SIZE - 1];
    for (size_t j = PERSIST_PRIME_TABLE_SIZE; j <= i; ++j) p = next_prime(p);
    return p;
}

uint64_t futcache_persist_prime_mod(size_t i)
{
    uint64_t p = futcache_persist_nth_prime(i);
    return p % FUTCACHE_PERSIST_PRIME_MODULUS;
}

/* Is sorted-point index i a "prime index"?
 * Index 0 -> p_0 = 2 (prime), 1 -> p_1 = 3 (prime), etc.
 * Every index maps to a prime, so every index is "prime".
 *
 * BUT for the Selberg zeta, we want to distinguish "prime cycles"
 * from "composite cycles". The distinction is: a feature is a
 * "prime cycle" if its persistence (gap width) is one of the
 * "fundamental" gaps — i.e., it's the first time this scale of
 * gap appears. In the merge tree, this corresponds to the feature
 * being at the "outermost" level of the dendrogram.
 *
 * For simplicity, we define: a feature is a "prime cycle" if its
 * birth sorted-point index (0-based) is a prime number.
 * Index 0 -> not prime (0 is not prime)
 * Index 1 -> not prime (1 is not prime)
 * Index 2 -> prime
 * Index 3 -> prime
 * Index 4 -> not prime
 * Index 5 -> prime
 * ...
 */

static bool is_prime_uint(uint64_t n)
{
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
    uint64_t i = 5;
    while (i * i <= n) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
        i += 6;
    }
    return true;
}

/* ============================================================
 * Engine internals
 * ============================================================ */

struct futcache_persist {
    /* Observed points (unsorted) */
    double *points;
    size_t point_count;
    size_t point_capacity;

    /* Merge tree: built via sort + gap-merge.
     * Layout: nodes [0..n-1] are leaves (sorted points).
     *         nodes [n..2n-2] are internal merge nodes.
     * Total: 2n - 1 nodes for n > 0, 0 for n == 0. */
    futcache_persist_node_t *nodes;
    size_t node_count;
    size_t node_capacity;

    /* Persistence features (barcodes): n-1 features for n > 1.
     * Feature i corresponds to internal node n-1+i. */
    futcache_persist_feature_t *features;
    size_t feature_count;
    size_t feature_capacity;

    /* Cached: nearest-point data for is_novel_at. */
    double *sorted_points;
    size_t sorted_count;

    uint64_t observations;
    size_t max_features;

    futcache_allocator_t allocator;
};

/* ============================================================
 * Merge tree construction
 *
 * For n points x_1 <= x_2 <= ... <= x_n (sorted), the single-linkage
 * merge tree is built as follows:
 *
 * 1. Start with n leaves, each representing a single point.
 * 2. Compute n-1 gaps: gap_i = x_{i+1} - x_i for i = 0..n-2.
 * 3. Process gaps in order of increasing width (left to right in
 *    the case of ties — this gives a canonical tree).
 * 4. When merging adjacent components A and B at gap g:
 *    - Create an internal node with death_scale = g.
 *    - The merged component's lo = A.lo, hi = B.hi.
 *    - A and B become children of the new node.
 *    - The merged component is now available for further merging.
 *
 * This is the standard 1-D single-linkage / merge tree algorithm.
 * The resulting tree is a full binary tree with n leaves.
 * ============================================================ */

/* Simple merge sort for the gap indices (sort by gap width). */
typedef struct {
    double gap;
    size_t left_idx;  /* left component index */
    size_t right_idx; /* right component index */
} gap_entry_t;

static int gap_compare(const void *a, const void *b)
{
    const gap_entry_t *ga = (const gap_entry_t *)a;
    const gap_entry_t *gb = (const gap_entry_t *)b;
    if (ga->gap < gb->gap) return -1;
    if (ga->gap > gb->gap) return 1;
    /* Tie-break by position: left gap before right gap. */
    if (ga->left_idx < gb->left_idx) return -1;
    if (ga->left_idx > gb->left_idx) return 1;
    return 0;
}

/* Build the merge tree and features from the sorted points.
 *
 * This implements the classic 1-D persistent homology algorithm:
 * the merge tree is equivalent to the "reduced merge tree" of the
 * distance function, and the persistence diagram consists of the
 * (0, gap_i) pairs for each gap, where gap_i is the width of the
 * i-th merge.
 *
 * For the persistence diagram in 1-D:
 *   - Each of the n-1 gaps produces one persistent feature.
 *   - Feature i has birth = 0 (the component appears at t=0) and
 *     death = gap_i / 2 (radius-t balls meet halfway across the gap).
 *   - The "birth_obs" is the index of the left point in the gap.
 *   - The "death_obs" is the index of the right point in the gap.
 *
 * The merge tree structure:
 *   - Leaves are the sorted points.
 *   - Internal nodes are created in order of increasing gap.
 *   - Each internal node merges the two components adjacent to the gap.
 */
static futcache_status_t rebuild_merge_tree_inplace(futcache_persist_t *engine)
{
    size_t n = engine->point_count;
    if (n == 0) {
        engine->node_count = 0;
        engine->feature_count = 0;
        return FUTCACHE_OK;
    }

    /* Sort points (for the sorted array and merge tree). */
    if (engine->sorted_count < n) {
        size_t new_cap = n < 64 ? 64 : n * 2;
        double *new_sp = (double *)engine->allocator.allocate(
            engine->allocator.context, new_cap * sizeof(double));
        if (new_sp == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        if (engine->sorted_count > 0) {
            memcpy(new_sp, engine->sorted_points,
                   engine->sorted_count * sizeof(double));
        }
        engine->allocator.deallocate(engine->allocator.context,
                                     engine->sorted_points);
        engine->sorted_points = new_sp;
        engine->sorted_count = new_cap; /* capacity, not count */
    }

    /* Copy and sort points. */
    for (size_t i = 0; i < n; ++i) {
        engine->sorted_points[i] = engine->points[i];
    }
    /* Simple insertion sort (n is small in practice; for large n,
     * a proper merge sort would be better). */
    for (size_t i = 1; i < n; ++i) {
        double key = engine->sorted_points[i];
        size_t j = i;
        while (j > 0 && engine->sorted_points[j - 1] > key) {
            engine->sorted_points[j] = engine->sorted_points[j - 1];
            j--;
        }
        engine->sorted_points[j] = key;
    }

    /* Build merge tree.
     *
     * For n points sorted as s[0] <= s[1] <= ... <= s[n-1],
     * there are n-1 gaps: gap[i] = s[i+1] - s[i] for i = 0..n-2.
     *
     * The merge tree has 2n-1 nodes:
     *   - Leaves: nodes[0..n-1], one per point.
     *   - Internal: nodes[n..2n-2], one per merge.
     *
     * The merge order is by increasing gap width. We use a
     * "component" array to track which components are adjacent.
     *
     * For the persistence diagram, we don't actually need the full
     * merge tree structure — we just need the (birth, death) pairs.
     * In 1-D, the persistence diagram is simply:
     *   - For each gap i (sorted by width): (0, gap_width_i / 2)
     *   - The last component (root) has death = INFINITY.
     *
     * But for the merge tree structure (needed for validate and
     * future d-D extension), we build the full tree.
     */

    if (n == 1) {
        /* Single point: one leaf, no merges. */
        if (engine->node_count < 1) {
            size_t new_cap = 16;
            futcache_persist_node_t *new_nodes =
                (futcache_persist_node_t *)engine->allocator.allocate(
                    engine->allocator.context, new_cap * sizeof(futcache_persist_node_t));
            if (new_nodes == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
            engine->allocator.deallocate(engine->allocator.context, engine->nodes);
            engine->nodes = new_nodes;
            engine->node_capacity = new_cap;
        }
        futcache_persist_node_t *node = &engine->nodes[0];
        memset(node, 0, sizeof(*node));
        node->lo = engine->sorted_points[0];
        node->hi = engine->sorted_points[0];
        node->death_scale = INFINITY;
        node->birth_obs = 0;
        node->death_obs = SIZE_MAX;
        node->parent = -1;
        node->left = -1;
        node->right = -1;
        node->is_leaf = true;
        engine->node_count = 1;
        engine->feature_count = 0;
        return FUTCACHE_OK;
    }

    /* Need 2n-1 nodes. */
    size_t needed_nodes = 2 * n - 1;
    if (engine->node_count < needed_nodes) {
        size_t new_cap = needed_nodes < 64 ? 64 : needed_nodes * 2;
        futcache_persist_node_t *new_nodes =
            (futcache_persist_node_t *)engine->allocator.allocate(
                engine->allocator.context, new_cap * sizeof(futcache_persist_node_t));
        if (new_nodes == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        engine->allocator.deallocate(engine->allocator.context, engine->nodes);
        engine->nodes = new_nodes;
        engine->node_capacity = new_cap;
    }

    /* Initialize leaves. */
    for (size_t i = 0; i < n; ++i) {
        futcache_persist_node_t *node = &engine->nodes[i];
        memset(node, 0, sizeof(*node));
        node->lo = engine->sorted_points[i];
        node->hi = engine->sorted_points[i];
        node->death_scale = INFINITY;
        node->birth_obs = i; /* index in sorted order */
        node->death_obs = SIZE_MAX;
        node->parent = -1;
        node->left = -1;
        node->right = -1;
        node->is_leaf = true;
    }

    /* Compute gaps. */
    size_t n_gaps = n - 1;
    gap_entry_t *gaps = (gap_entry_t *)malloc(n_gaps * sizeof(gap_entry_t));
    if (gaps == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;

    for (size_t i = 0; i < n_gaps; ++i) {
        gaps[i].gap = engine->sorted_points[i + 1] - engine->sorted_points[i];
        gaps[i].left_idx = i;
        gaps[i].right_idx = i + 1;
    }

    /* Sort gaps by width (ascending). Ties broken by position. */
    qsort(gaps, n_gaps, sizeof(gap_entry_t), gap_compare);

    /* Build internal nodes via union-find on the sorted points.
     *
     * We maintain a "component" array: comp[i] is the node index of
     * the component containing sorted point i. Initially, comp[i] = i
     * (each point is its own component).
     *
     * For each gap (in order of increasing width), we merge the two
     * components on either side of the gap. The new component gets
     * an internal node with death_scale = half the gap width.
     */
    int32_t *comp = (int32_t *)malloc(n * sizeof(int32_t));
    if (comp == NULL) { free(gaps); return FUTCACHE_ERROR_OUT_OF_MEMORY; }

    for (size_t i = 0; i < n; ++i) comp[i] = (int32_t)i;

    /* Track the leftmost and rightmost sorted-point index for each
     * component. Components are indexed by node index (0..2n-2). */
    int32_t *comp_left = (int32_t *)malloc((2 * n) * sizeof(int32_t));
    int32_t *comp_right = (int32_t *)malloc((2 * n) * sizeof(int32_t));
    if (comp_left == NULL || comp_right == NULL) {
        free(gaps); free(comp); free(comp_left); free(comp_right);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < n; ++i) {
        comp_left[i] = (int32_t)i;
        comp_right[i] = (int32_t)i;
    }

    size_t next_node = n; /* internal nodes start at index n */

    for (size_t g = 0; g < n_gaps; ++g) {
        /* Find the components on either side of gap g.
         * The gap is between sorted_points[gaps[g].left_idx] and
         * sorted_points[gaps[g].right_idx].
         * We need to find the component containing the left point and
         * the component containing the right point. */
        size_t li = gaps[g].left_idx;
        size_t ri = gaps[g].right_idx;
        int32_t left_comp = comp[li];
        int32_t right_comp = comp[ri];

        if (left_comp == right_comp) continue; /* already merged */

        /* Create internal node. */
        futcache_persist_node_t *merge = &engine->nodes[next_node];
        memset(merge, 0, sizeof(*merge));

        futcache_persist_node_t *lc = &engine->nodes[left_comp];
        futcache_persist_node_t *rc = &engine->nodes[right_comp];

        merge->lo = lc->lo;
        merge->hi = rc->hi;
        merge->death_scale = INFINITY; /* will be updated when this merges */
        merge->birth_obs = lc->birth_obs;
        merge->death_obs = SIZE_MAX;
        merge->parent = -1;
        merge->left = left_comp;
        merge->right = right_comp;
        merge->is_leaf = false;

        lc->parent = (int32_t)next_node;
        rc->parent = (int32_t)next_node;

        /* Update component tracking. */
        int32_t new_comp = (int32_t)next_node;
        for (size_t i = 0; i < n; ++i) {
            if (comp[i] == left_comp || comp[i] == right_comp) {
                comp[i] = new_comp;
            }
        }
        comp_left[new_comp] = comp_left[left_comp];
        comp_right[new_comp] = comp_right[right_comp];

        next_node++;
    }

    /* The root is the last internal node (if n > 1).
     * Its death_scale stays INFINITY (the final component). */

    engine->node_count = next_node;

    /* Build persistence features.
     *
     * In 1-D, the persistence diagram has n-1 finite features (one
     * per internal node) and 1 infinite feature (the root).
     *
     * For internal node k (node index n + k, where k = 0..n-2):
     *   - birth = 0 (the component exists from t=0)
     *   - death = half the gap width at which the radius-t balls meet
     *   - persistence = death - birth = gap width / 2
     *
     * The gap width for the k-th merge (in sorted order) is
     * gaps[k].gap after sorting.
     *
     * The birth_obs is the leftmost sorted-point index in the
     * left child component. The death_obs is the leftmost sorted-point
     * index in the right child component.
     *
     * For prime tagging: birth_prime = p_birth_obs mod M,
     * death_prime = p_death_obs mod M.
     */
    if (n > 1) {
        size_t n_features = n - 1;
        if (engine->feature_count < n_features) {
            size_t new_cap = n_features < 64 ? 64 : n_features * 2;
            futcache_persist_feature_t *new_feats =
                (futcache_persist_feature_t *)engine->allocator.allocate(
                    engine->allocator.context,
                    new_cap * sizeof(futcache_persist_feature_t));
            if (new_feats == NULL) {
                free(gaps); free(comp); free(comp_left); free(comp_right);
                return FUTCACHE_ERROR_OUT_OF_MEMORY;
            }
            engine->allocator.deallocate(engine->allocator.context,
                                         engine->features);
            engine->features = new_feats;
            engine->feature_capacity = new_cap;
        }

        for (size_t k = 0; k < n_features; ++k) {
            size_t node_idx = n + k;
            futcache_persist_node_t *node = &engine->nodes[node_idx];
            futcache_persist_feature_t *f = &engine->features[k];

            /* Find the gap width: it's the gap between the leftmost
             * point of the left child and the rightmost point of the
             * right child... no, it's the gap at the merge point.
             *
             * Actually, the death_scale of the internal node is the
             * filtration scale at which the merge happened. We stored this
             * during the merge. But we set death_scale = INFINITY for
             * all internal nodes initially. Let me fix this: the
             * death_scale should be set to half the gap width during the
             * merge. */

            /* For now, compute the persistence from the gap array.
             * The k-th merge (in sorted order) corresponds to gaps[k]. */
            f->birth = (size_t)node->left; /* left child node index */
            f->death = (size_t)node->right; /* right child node index */

            /* The actual observation indices: use the leftmost point
             * of each child component. */
            futcache_persist_node_t *lc = &engine->nodes[node->left];
            futcache_persist_node_t *rc = &engine->nodes[node->right];
            f->birth = lc->birth_obs;
            f->death = rc->birth_obs;

            f->birth_prime = futcache_persist_prime_mod(f->birth);
            f->death_prime = futcache_persist_prime_mod(f->death);

            f->birth_value = lc->lo;
            f->death_value = (lc->hi + rc->lo) / 2.0; /* midpoint of gap */

            /* Radius-t balls around adjacent components meet at the gap
             * midpoint, so the filtration death scale is half the gap. */
            f->persistence = (rc->lo - lc->hi) / 2.0;

            /* Also update the node's death_scale. */
            node->death_scale = f->persistence;
        }

        engine->feature_count = n_features;
    } else {
        engine->feature_count = 0;
    }

    free(gaps);
    free(comp);
    free(comp_left);
    free(comp_right);
    return FUTCACHE_OK;
}

/* Build all derived state off to the side, then publish it in one commit.
 * This keeps observe failure-atomic even when a custom allocator fails at any
 * tree/diagram allocation site. */
static futcache_status_t rebuild_merge_tree(futcache_persist_t *engine)
{
    futcache_persist_t next = *engine;
    next.nodes = NULL;
    next.node_count = 0U;
    next.node_capacity = 0U;
    next.features = NULL;
    next.feature_count = 0U;
    next.feature_capacity = 0U;
    next.sorted_points = NULL;
    next.sorted_count = 0U;

    futcache_status_t status = rebuild_merge_tree_inplace(&next);
    if (status != FUTCACHE_OK) {
        engine->allocator.deallocate(engine->allocator.context, next.nodes);
        engine->allocator.deallocate(engine->allocator.context, next.features);
        engine->allocator.deallocate(engine->allocator.context,
                                     next.sorted_points);
        return status;
    }

    engine->allocator.deallocate(engine->allocator.context, engine->nodes);
    engine->allocator.deallocate(engine->allocator.context, engine->features);
    engine->allocator.deallocate(engine->allocator.context,
                                 engine->sorted_points);
    engine->nodes = next.nodes;
    engine->node_count = next.node_count;
    engine->node_capacity = next.node_capacity;
    engine->features = next.features;
    engine->feature_count = next.feature_count;
    engine->feature_capacity = next.feature_capacity;
    engine->sorted_points = next.sorted_points;
    engine->sorted_count = next.sorted_count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_create / destroy
 * ============================================================ */

futcache_status_t futcache_persist_create(
    const futcache_persist_config_t *config, futcache_persist_t **out_engine)
{
    if (config == NULL || out_engine == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *out_engine = NULL;

    futcache_allocator_t allocator;
    if (!normalize_allocator(&config->allocator, &allocator))
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    futcache_persist_t *engine =
        (futcache_persist_t *)allocator.allocate(allocator.context,
                                                 sizeof(futcache_persist_t));
    if (engine == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;

    memset(engine, 0, sizeof(*engine));
    engine->allocator = allocator;
    engine->max_features = config->max_features;

    *out_engine = engine;
    return FUTCACHE_OK;
}

void futcache_persist_destroy(futcache_persist_t *engine)
{
    if (engine == NULL) return;
    futcache_allocator_t *a = &engine->allocator;
    a->deallocate(a->context, engine->points);
    a->deallocate(a->context, engine->nodes);
    a->deallocate(a->context, engine->features);
    a->deallocate(a->context, engine->sorted_points);
    a->deallocate(a->context, engine);
}

/* ============================================================
 * futcache_persist_observe
 * ============================================================ */

futcache_status_t futcache_persist_observe(futcache_persist_t *engine, double x)
{
    if (engine == NULL || !isfinite(x))
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    if (engine->max_features != 0U &&
        engine->point_count > engine->max_features) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    /* Grow the points array. */
    if (engine->point_count >= engine->point_capacity) {
        size_t new_cap = engine->point_capacity == 0 ? 16 : engine->point_capacity * 2;
        double *new_pts = (double *)engine->allocator.allocate(
            engine->allocator.context, new_cap * sizeof(double));
        if (new_pts == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
        if (engine->point_count > 0)
            memcpy(new_pts, engine->points, engine->point_count * sizeof(double));
        engine->allocator.deallocate(engine->allocator.context, engine->points);
        engine->points = new_pts;
        engine->point_capacity = new_cap;
    }

    engine->points[engine->point_count++] = x;
    futcache_status_t status = rebuild_merge_tree(engine);
    if (status != FUTCACHE_OK) {
        engine->point_count--;
        return status;
    }
    engine->observations++;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_is_novel_at
 * ============================================================ */

futcache_status_t futcache_persist_is_novel_at(
    const futcache_persist_t *engine, double x, double t, bool *out_is_novel)
{
    if (engine == NULL || out_is_novel == NULL || !isfinite(x) ||
        !isfinite(t) || t < 0.0)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *out_is_novel = false;

    if (engine->point_count == 0) {
        *out_is_novel = true;
        return FUTCACHE_OK;
    }

    /* Binary search on sorted_points for the nearest point. */
    size_t n = engine->point_count;
    const double *sp = engine->sorted_points;

    /* Find the insertion point. */
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (sp[mid] <= x) lo = mid + 1;
        else hi = mid;
    }

    /* Check lo-1 and lo. */
    double min_dist = INFINITY;
    if (lo > 0) {
        double d = x - sp[lo - 1];
        if (d < 0) d = -d;
        if (d < min_dist) min_dist = d;
    }
    if (lo < n) {
        double d = sp[lo] - x;
        if (d < 0) d = -d;
        if (d < min_dist) min_dist = d;
    }

    *out_is_novel = (min_dist > t);
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_novelty_spectrum
 * ============================================================ */

futcache_status_t futcache_persist_novelty_spectrum(
    const futcache_persist_t *engine, double x, double *out_intervals,
    size_t *inout_count)
{
    if (engine == NULL || inout_count == NULL || !isfinite(x))
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t required = 1;
    if (*inout_count < required) {
        *inout_count = required;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out_intervals == NULL) {
        *inout_count = 0;
        return FUTCACHE_OK;
    }

    if (engine->point_count == 0) {
        out_intervals[0] = 0.0;
        out_intervals[1] = INFINITY;
        *inout_count = 1;
        return FUTCACHE_OK;
    }

    /* Find nearest point via binary search. */
    size_t n = engine->point_count;
    const double *sp = engine->sorted_points;
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (sp[mid] <= x) lo = mid + 1;
        else hi = mid;
    }

    double min_dist = INFINITY;
    if (lo > 0) {
        double d = x - sp[lo - 1];
        if (d < 0) d = -d;
        if (d < min_dist) min_dist = d;
    }
    if (lo < n) {
        double d = sp[lo] - x;
        if (d < 0) d = -d;
        if (d < min_dist) min_dist = d;
    }

    if (min_dist == 0.0) {
        *inout_count = 0;
    } else {
        out_intervals[0] = 0.0;
        out_intervals[1] = min_dist;
        *inout_count = 1;
    }
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_copy_diagram
 * ============================================================ */

futcache_status_t futcache_persist_copy_diagram(
    const futcache_persist_t *engine,
    futcache_persist_feature_t *out_features, size_t *inout_count)
{
    if (engine == NULL || inout_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (*inout_count < engine->feature_count) {
        *inout_count = engine->feature_count;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out_features != NULL && engine->feature_count > 0)
        memcpy(out_features, engine->features,
               engine->feature_count * sizeof(futcache_persist_feature_t));
    *inout_count = engine->feature_count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_node_count / get_node
 * ============================================================ */

size_t futcache_persist_node_count(const futcache_persist_t *engine)
{
    return engine == NULL ? 0 : engine->node_count;
}

const futcache_persist_node_t *futcache_persist_get_node(
    const futcache_persist_t *engine, size_t index)
{
    if (engine == NULL || index >= engine->node_count) return NULL;
    return &engine->nodes[index];
}

/* ============================================================
 * futcache_persist_feature_count
 * ============================================================ */

futcache_status_t futcache_persist_feature_count(
    const futcache_persist_t *engine, double tau, size_t *out_count)
{
    if (engine == NULL || out_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t count = 0;
    for (size_t i = 0; i < engine->feature_count; ++i) {
        if (engine->features[i].persistence >= tau) count++;
    }
    *out_count = count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_evict_below
 * ============================================================ */

futcache_status_t futcache_persist_evict_below(futcache_persist_t *engine, double tau)
{
    if (engine == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t write = 0;
    for (size_t i = 0; i < engine->feature_count; ++i) {
        if (engine->features[i].persistence >= tau) {
            if (write != i)
                engine->features[write] = engine->features[i];
            write++;
        }
    }
    engine->feature_count = write;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_clear
 * ============================================================ */

futcache_status_t futcache_persist_clear(futcache_persist_t *engine)
{
    if (engine == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    engine->point_count = 0;
    engine->node_count = 0;
    engine->feature_count = 0;
    engine->observations = 0;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_validate
 * ============================================================ */

futcache_status_t futcache_persist_validate(const futcache_persist_t *engine)
{
    if (engine == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t n = engine->point_count;

    /* Node count: 0 for n=0, 1 for n=1, 2n-1 for n>1. */
    size_t expected_nodes;
    if (n == 0) expected_nodes = 0;
    else if (n == 1) expected_nodes = 1;
    else expected_nodes = 2 * n - 1;
    if (engine->node_count != expected_nodes)
        return FUTCACHE_ERROR_CORRUPT_DATA;

    /* Feature count: 0 for n<=1, n-1 for n>1. */
    size_t expected_feats = (n > 1) ? n - 1 : 0;
    if (engine->feature_count != expected_feats)
        return FUTCACHE_ERROR_CORRUPT_DATA;

    /* Check leaves: nodes[0..n-1] should be leaves. */
    for (size_t i = 0; i < n && i < engine->node_count; ++i) {
        const futcache_persist_node_t *node = &engine->nodes[i];
        if (!node->is_leaf) return FUTCACHE_ERROR_CORRUPT_DATA;
        if (node->hi < node->lo) return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* Check internal nodes. */
    for (size_t i = n; i < engine->node_count; ++i) {
        const futcache_persist_node_t *node = &engine->nodes[i];
        if (node->is_leaf) return FUTCACHE_ERROR_CORRUPT_DATA;
        if (node->left < 0 || node->right < 0)
            return FUTCACHE_ERROR_CORRUPT_DATA;
        if (node->left >= (int32_t)engine->node_count ||
            node->right >= (int32_t)engine->node_count)
            return FUTCACHE_ERROR_CORRUPT_DATA;
        if (node->hi < node->lo) return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    /* Check features: persistence >= 0. */
    for (size_t i = 0; i < engine->feature_count; ++i) {
        if (engine->features[i].persistence < 0.0)
            return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_merge_features (deterministic signature join)
 * ============================================================ */

static int feature_compare(const void *a, const void *b)
{
    const futcache_persist_feature_t *fa = (const futcache_persist_feature_t *)a;
    const futcache_persist_feature_t *fb = (const futcache_persist_feature_t *)b;
    if (fa->birth_prime < fb->birth_prime) return -1;
    if (fa->birth_prime > fb->birth_prime) return 1;
    if (fa->death_prime < fb->death_prime) return -1;
    if (fa->death_prime > fb->death_prime) return 1;
    if (fa->birth < fb->birth) return -1;
    if (fa->birth > fb->birth) return 1;
    if (fa->death < fb->death) return -1;
    if (fa->death > fb->death) return 1;
    if (fa->persistence < fb->persistence) return -1;
    if (fa->persistence > fb->persistence) return 1;
    return 0;
}

futcache_status_t futcache_persist_merge_features(
    const futcache_persist_feature_t *a, size_t a_count,
    const futcache_persist_feature_t *b, size_t b_count,
    futcache_persist_feature_t *out, size_t *inout_out_count)
{
    if (inout_out_count == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if ((a == NULL && a_count > 0) || (b == NULL && b_count > 0))
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    /* Worst case: all features from both inputs are unique. */
    size_t max_out = a_count + b_count;

    if (*inout_out_count < max_out) {
        *inout_out_count = max_out;
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (out == NULL) {
        *inout_out_count = 0;
        return FUTCACHE_OK;
    }

    /* Merge: collect all features, sort, dedup by signature.
     * For same signature, keep the one with larger persistence. */
    size_t total = a_count + b_count;
    if (total == 0) {
        *inout_out_count = 0;
        return FUTCACHE_OK;
    }

    /* Copy all features to out (temporarily). */
    size_t i = 0;
    for (size_t j = 0; j < a_count; ++j) out[i++] = a[j];
    for (size_t j = 0; j < b_count; ++j) out[i++] = b[j];

    /* Sort by signature first so even modular-collision groups are
     * contiguous, then use full fields for deterministic output. */
    qsort(out, total, sizeof(futcache_persist_feature_t), feature_compare);

    /* Dedup: walk through sorted array, keep one feature per signature.
     * For same signature, the last one (highest persistence) wins. */
    size_t write = 0;
    for (size_t j = 0; j < total; ++j) {
        if (write == 0) {
            out[write++] = out[j];
        } else {
            bool same_signature =
                out[write - 1].birth_prime == out[j].birth_prime &&
                out[write - 1].death_prime == out[j].death_prime;
            if (!same_signature) {
                out[write++] = out[j];
            } else {
                /* Same signature: keep the one with larger persistence. */
                if (out[j].persistence > out[write - 1].persistence) {
                    out[write - 1] = out[j];
                }
            }
        }
    }

    *inout_out_count = write;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_selberg_zeta
 * ============================================================ */

futcache_status_t futcache_persist_selberg_zeta(
    const futcache_persist_t *engine, double s, double *out_zeta)
{
    if (engine == NULL || out_zeta == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (!isfinite(s) || s <= 0.0)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    double zeta = 1.0;
    for (size_t i = 0; i < engine->feature_count; ++i) {
        const futcache_persist_feature_t *f = &engine->features[i];

        /* Only "prime" features: birth index is a prime number. */
        if (!is_prime_uint(f->birth)) continue;

        double p = f->persistence;
        if (p <= 0.0) continue;

        /* Use N = 1 + persistence so every finite factor has N > 1.
         * Using persistence directly makes the advertised positive product
         * singular at p=1 and negative for 0<p<1. */
        double term = pow(1.0 + p, -s);
        double denom = 1.0 - term;
        if (fabs(denom) < 1e-300) {
            zeta = INFINITY;
            break;
        }
        zeta /= denom;
    }

    *out_zeta = zeta;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_prime_cycle_count
 * ============================================================ */

futcache_status_t futcache_persist_prime_cycle_count(
    const futcache_persist_t *engine, double tau, size_t *out_count)
{
    if (engine == NULL || out_count == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    size_t count = 0;
    for (size_t i = 0; i < engine->feature_count; ++i) {
        const futcache_persist_feature_t *f = &engine->features[i];
        if (is_prime_uint(f->birth) && f->persistence >= tau) {
            count++;
        }
    }
    *out_count = count;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_persist_get_stats
 * ============================================================ */

futcache_status_t futcache_persist_get_stats(
    const futcache_persist_t *engine, futcache_persist_stats_t *out_stats)
{
    if (engine == NULL || out_stats == NULL)
        return FUTCACHE_ERROR_INVALID_ARGUMENT;

    out_stats->observations = engine->observations;
    out_stats->feature_count = engine->feature_count;
    out_stats->alive_feature_count = 0;
    out_stats->prime_cycle_count = 0;
    out_stats->max_persistence = 0.0;
    out_stats->min_persistence = INFINITY;
    out_stats->total_persistence = 0.0;

    size_t prime_count = 0;
    double max_p = 0.0, min_p = INFINITY, total_p = 0.0;

    for (size_t i = 0; i < engine->feature_count; ++i) {
        double p = engine->features[i].persistence;
        if (p > max_p) max_p = p;
        if (p < min_p) min_p = p;
        total_p += p;
        if (is_prime_uint(engine->features[i].birth)) prime_count++;
    }

    out_stats->max_persistence = (engine->feature_count > 0) ? max_p : 0.0;
    out_stats->min_persistence = (engine->feature_count > 0) ? min_p : 0.0;
    out_stats->total_persistence = total_p;
    out_stats->prime_cycle_count = prime_count;

    out_stats->memory_bytes = sizeof(futcache_persist_t) +
                              engine->point_count * sizeof(double) +
                              engine->node_count * sizeof(futcache_persist_node_t) +
                              engine->feature_count * sizeof(futcache_persist_feature_t);
    return FUTCACHE_OK;
}

#ifndef FUTCACHE_SELECT_H
#define FUTCACHE_SELECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"
#include "futcache/pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_select: submodular representative selection (Design Sketch 03).
 *
 * Mathematical foundation.
 *
 *   Given an observed point set X = {x_1, ..., x_n} in a metric space
 *   (X, d) and a cardinality budget k, the coverage function is:
 *
 *       f(S) = |{ x in X : min_{s in S} d(x, s) <= eps }|    for S ⊆ X
 *
 *   f is monotone and submodular (diminishing returns). The lazy greedy
 *   algorithm — at each step add the candidate with the largest marginal
 *   gain f(S ∪ {x}) - f(S) — achieves:
 *
 *       f(R_greedy) >= (1 - 1/e) * f(R_opt)
 *
 *   (Nemhauser, Wolsey, Fisher 1978). This is tight for submodular
 *   maximisation under a cardinality constraint.
 *
 *   Connection to metric entropy: k = P(K, eps) (the packing number) is
 *   the minimal number of reps needed for full coverage f(S) = n. The
 *   greedy tells you WHICH k to pick, not just how many.
 *
 *   Determinism / CRDT compatibility: ties in marginal gain are broken by
 *   lexicographic order of representative coordinates. This makes the
 *   selected set a deterministic function of the input multiset,
 *   independent of arrival order.
 *
 *   Streaming swap (Chen, Mirrokni, Nanongkai, Wang 2018): when |R| > k,
 *   evict the rep with the lowest marginal loss f(R) - f(R \ {r}). This
 *   retains the 1 - 1/e guarantee in the streaming setting.
 */

/*
 * Result of a max-coverage selection.
 */
typedef struct futcache_select_result {
    /* Indices (into the input point array) of the selected representatives,
     * in the order they were greedily selected. Length = selected_count. */
    size_t *selected_indices;
    size_t selected_count;

    /* Coverage statistics */
    size_t total_covered;   /* f(R_greedy): number of points within eps of at least one selected rep */
    size_t total_points;    /* n: total number of input points */
    double coverage_ratio;  /* f(R_greedy) / n */

    /* Marginal gain at each greedy step (for diagnostics). Length = selected_count. */
    double *marginal_gains;

    /* If a brute-force optimum was computed (n <= brute_force_limit),
     * the optimal coverage. Otherwise -1. */
    double opt_coverage;
    double approximation_ratio; /* f_greedy / f_opt, or -1 if opt not computed */

    /* Caller must free selected_indices, marginal_gains, and the result
     * struct itself with futcache_select_free_result(). */
} futcache_select_result_t;

/*
 * Maximum input size for brute-force optimal comparison.
 * For n > this value, opt_coverage = -1 and approximation_ratio = -1.
 */
#define FUTCACHE_SELECT_BRUTE_FORCE_LIMIT 20

/*
 * Computes the max-coverage representative selection.
 *
 * Given n points and a cardinality budget k, selects k representatives
 * that maximize the number of points within distance epsilon of at least
 * one selected representative.
 *
 * Algorithm: Lazy greedy (Nemhauser-Wolsey-Fisher).
 *   - At each of k steps, scan all n candidates, compute marginal gain
 *     (points newly covered by adding this candidate), pick the argmax.
 *   - Ties broken by lexicographic order of point coordinates (deterministic).
 *   - Complexity: O(k * n^2 * d) worst case. For n <= 1000, k <= 100,
 *     this is fast in C.
 *
 * If n <= FUTCACHE_SELECT_BRUTE_FORCE_LIMIT, also computes the exact
 * optimum via enumeration for the approximation ratio.
 *
 * Parameters:
 *   points        - [n * dimension] array of observed points
 *   n             - number of observed points (>= 1)
 *   dimension     - dimensionality of each point (>= 1)
 *   epsilon       - coverage radius (a point is "covered" if within this
 *                   distance of a selected representative)
 *   k             - cardinality budget (1 <= k <= n)
 *   distance      - metric distance function (NULL = L_inf)
 *   context       - opaque pointer passed to distance
 *
 * Output:
 *   out_result    - heap-allocated result; caller frees with
 *                   futcache_select_free_result()
 *
 * Returns FUTCACHE_OK on success.
 */
FUTCACHE_API futcache_status_t futcache_select_max_coverage(
    const double *points,
    size_t n,
    size_t dimension,
    double epsilon,
    size_t k,
    futcache_distance_fn distance,
    void *context,
    futcache_select_result_t **out_result);

/*
 * Frees a select result returned by futcache_select_max_coverage().
 */
FUTCACHE_API void futcache_select_free_result(futcache_select_result_t *result);

/*
 * Streaming swap eviction: given the current representative set R (of size
 * |R|) and all observed points X (of size n), finds the representative r*
 * in R with the lowest marginal coverage loss:
 *
 *   marginal_loss(r) = |{ x in X : d(x, r) <= eps AND for all s in R\{r}:
 *                        d(x, s) > eps }|
 *
 * i.e., the number of points that ONLY r covers (no other rep is within
 * eps). Evicting r* causes the smallest loss in covered points.
 *
 * This is the swap heuristic from Chen et al. 2018 for submodular
 * maximisation under a cardinality constraint in the streaming setting.
 *
 * Parameters:
 *   points        - [n * dimension] all observed points
 *   n             - number of observed points
 *   reps          - [rep_count * dimension] current representatives
 *   rep_count     - number of current representatives (>= 1)
 *   dimension     - dimensionality
 *   epsilon       - coverage radius
 *   distance      - metric distance function (NULL = L_inf)
 *   context       - opaque pointer passed to distance
 *   out_evict_index - receives the index (into reps) of the rep to evict
 *   out_marginal_loss - receives the marginal coverage loss of evicting
 *                        that rep (how many points lose coverage)
 *
 * Returns FUTCACHE_OK on success, FUTCACHE_ERROR_OUT_OF_RANGE if
 * rep_count == 0.
 */
FUTCACHE_API futcache_status_t futcache_select_evict_worst(
    const double *points,
    size_t n,
    const double *reps,
    size_t rep_count,
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *context,
    size_t *out_evict_index,
    double *out_marginal_loss);

/*
 * Computes the coverage of a given representative set R over observed
 * points X: the fraction of points within epsilon of at least one rep.
 *
 *   coverage = |{ x in X : min_{r in R} d(x, r) <= eps }| / n
 *
 * Returns 0.0 if R is empty.
 */
FUTCACHE_API futcache_status_t futcache_select_coverage(
    const double *points,
    size_t n,
    const double *reps,
    size_t rep_count,
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *context,
    double *out_coverage);

#ifdef __cplusplus
}
#endif

#endif /* FUTCACHE_SELECT_H */

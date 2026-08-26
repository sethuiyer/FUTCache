#define _POSIX_C_SOURCE 200809L

#include "futcache/select.h"
#include "futcache/futcache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Default distance: L_inf
 * ============================================================ */

static double default_distance_linf(const double *a, const double *b,
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

/* Returns the distance function to use (L_inf if NULL). */
static futcache_distance_fn get_distance_fn(futcache_distance_fn distance)
{
    return distance ? distance : default_distance_linf;
}

/* ============================================================
 * Helper: compute coverage f(S) for a candidate set S over points X.
 * Returns the number of points in X within epsilon of at least one
 * element of S.
 * ============================================================ */

static size_t compute_coverage_count(
    const double *points,
    size_t n,
    const size_t *selected,
    size_t selected_count,
    size_t dimension,
    double epsilon,
    futcache_distance_fn dist_fn,
    void *context)
{
    if (selected_count == 0U) return 0U;

    size_t covered = 0U;
    for (size_t i = 0; i < n; ++i) {
        const double *pt = points + i * dimension;
        bool is_covered = false;
        for (size_t s = 0; s < selected_count; ++s) {
            const double *rep = points + selected[s] * dimension;
            double d = dist_fn(pt, rep, dimension, context);
            if (d <= epsilon) {
                is_covered = true;
                break;
            }
        }
        if (is_covered) covered++;
    }
    return covered;
}

/*
 * Lexicographic tie-break: returns true if point_a < point_b
 * (lexicographic over coordinates), false otherwise.
 * Used to make greedy selection deterministic and order-independent.
 */
static bool lexicographic_less(
    const double *a,
    const double *b,
    size_t dimension)
{
    for (size_t i = 0; i < dimension; ++i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false; /* equal */
}

/* ============================================================
 * futcache_select_max_coverage
 * ============================================================ */

futcache_status_t futcache_select_max_coverage(
    const double *points,
    size_t n,
    size_t dimension,
    double epsilon,
    size_t k,
    futcache_distance_fn distance,
    void *context,
    futcache_select_result_t **out_result)
{
    if (points == NULL || out_result == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_result = NULL;

    if (n == 0U || dimension == 0U) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (k == 0U || k > n) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(epsilon) || epsilon < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_distance_fn dist_fn = get_distance_fn(distance);

    /* Allocation: selected indices (k), marginal gains (k). */
    size_t *selected = (size_t *)malloc(k * sizeof(size_t));
    double *marginal_gains = (double *)malloc(k * sizeof(double));
    futcache_select_result_t *result =
        (futcache_select_result_t *)malloc(sizeof(futcache_select_result_t));

    if (selected == NULL || marginal_gains == NULL || result == NULL) {
        free(selected);
        free(marginal_gains);
        free(result);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    memset(selected, 0, k * sizeof(size_t));
    memset(marginal_gains, 0, k * sizeof(double));

    /* Track which points are already covered by the current selection. */
    /* covered[i] = true if point i is within eps of at least one selected rep. */
    bool *covered = (bool *)malloc(n * sizeof(bool));
    if (covered == NULL) {
        free(selected);
        free(marginal_gains);
        free(result);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memset(covered, 0, n * sizeof(bool));

    size_t selected_count = 0U;
    size_t current_covered = 0U;

    /* Lazy greedy: at each step, find the candidate with the largest
     * marginal gain. Tie-break by lexicographic order. */
    for (size_t step = 0; step < k; ++step) {
        double best_gain = -1.0;
        size_t best_idx = n; /* sentinel: no candidate yet */

        for (size_t cand = 0; cand < n; ++cand) {
            /* Skip if this candidate is already selected. */
            bool already_selected = false;
            for (size_t s = 0; s < selected_count; ++s) {
                if (selected[s] == cand) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;

            /* Compute marginal gain: how many new points does adding
             * candidate `cand` cover? */
            double gain = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (covered[i]) continue; /* already covered by current S */
                const double *pt_i = points + i * dimension;
                const double *pt_cand = points + cand * dimension;
                double d = dist_fn(pt_i, pt_cand, dimension, context);
                if (d <= epsilon) {
                    gain += 1.0;
                }
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_idx = cand;
            } else if (gain == best_gain && best_idx != n) {
                /* Tie-break: lexicographic order for determinism. */
                if (lexicographic_less(points + cand * dimension,
                                       points + best_idx * dimension,
                                       dimension)) {
                    best_idx = cand;
                }
            }
        }

        if (best_idx == n) {
            /* No more candidates with positive gain; break early. */
            break;
        }

        /* Add best_idx to the selection. */
        selected[selected_count] = best_idx;
        marginal_gains[selected_count] = best_gain;
        selected_count++;

        /* Update covered[] for all points within eps of best_idx. */
        const double *pt_best = points + best_idx * dimension;
        for (size_t i = 0; i < n; ++i) {
            if (!covered[i]) {
                const double *pt_i = points + i * dimension;
                double d = dist_fn(pt_i, pt_best, dimension, context);
                if (d <= epsilon) {
                    covered[i] = true;
                    current_covered++;
                }
            }
        }
    }

    /* Compute final coverage. */
    size_t total_covered = 0U;
    for (size_t i = 0; i < n; ++i) {
        if (covered[i]) total_covered++;
    }

    /* Brute-force optimum for small n. */
    double opt_coverage = -1.0;
    double approx_ratio = -1.0;
    if (n <= FUTCACHE_SELECT_BRUTE_FORCE_LIMIT && k <= n) {
        /* Try all C(n, k) subsets — only feasible for very small n.
         * For n=20, k=10, C(20,10)=184756, which is fine.
         * For larger k or n, skip. */
        if (n <= 20 && k <= 20) {
            /* Bitmask enumeration for n <= 16 (practical limit). */
            if (n <= 16) {
                size_t *brute_selected = (size_t *)malloc(k * sizeof(size_t));
                if (brute_selected != NULL) {
                    size_t best_opt = 0U;
                    size_t total_combos = (size_t)1 << n;
                    for (size_t mask = 0; mask < total_combos; ++mask) {
                        /* Count bits */
                        size_t bit_count = 0;
                        for (size_t b = 0; b < n; ++b) {
                            if (mask & (1U << b)) bit_count++;
                        }
                        if (bit_count != k) continue;

                        /* Build selected array */
                        size_t bs_count = 0;
                        for (size_t b = 0; b < n; ++b) {
                            if (mask & (1U << b)) {
                                brute_selected[bs_count++] = b;
                            }
                        }

                        /* Compute coverage */
                        size_t cov = compute_coverage_count(
                            points, n, brute_selected, bs_count,
                            dimension, epsilon, dist_fn, context);
                        if (cov > best_opt) best_opt = cov;
                    }
                    opt_coverage = (double)best_opt;
                }
                free(brute_selected);
            }
        }
        if (opt_coverage > 0.0) {
            approx_ratio = (double)total_covered / opt_coverage;
        }
    }

    free(covered);

    /* Fill result. */
    result->selected_indices = selected;
    result->selected_count = selected_count;
    result->total_covered = total_covered;
    result->total_points = n;
    result->coverage_ratio = (n > 0) ? (double)total_covered / (double)n : 0.0;
    result->marginal_gains = marginal_gains;
    result->opt_coverage = opt_coverage;
    result->approximation_ratio = approx_ratio;

    *out_result = result;
    return FUTCACHE_OK;
}

void futcache_select_free_result(futcache_select_result_t *result)
{
    if (result == NULL) return;
    free(result->selected_indices);
    free(result->marginal_gains);
    free(result);
}

/* ============================================================
 * futcache_select_evict_worst (streaming swap)
 * ============================================================ */

futcache_status_t futcache_select_evict_worst(
    const double *points,
    size_t n,
    const double *reps,
    size_t rep_count,
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *context,
    size_t *out_evict_index,
    double *out_marginal_loss)
{
    if (points == NULL || reps == NULL || out_evict_index == NULL ||
        out_marginal_loss == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_evict_index = 0;
    *out_marginal_loss = 0.0;

    if (rep_count == 0U) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    if (n == 0U || dimension == 0U) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_distance_fn dist_fn = get_distance_fn(distance);

    double best_loss = INFINITY;
    size_t best_evict = 0U;

    /* For each rep, compute how many points it uniquely covers
     * (points within eps of this rep but NOT within eps of any other rep). */
    for (size_t r = 0; r < rep_count; ++r) {
        double loss = 0.0;
        const double *rep_r = reps + r * dimension;

        for (size_t i = 0; i < n; ++i) {
            const double *pt_i = points + i * dimension;

            /* Is point i within eps of rep r? */
            double d_to_r = dist_fn(pt_i, rep_r, dimension, context);
            if (d_to_r > epsilon) continue;

            /* Is point i within eps of any OTHER rep? */
            bool covered_by_other = false;
            for (size_t s = 0; s < rep_count; ++s) {
                if (s == r) continue;
                const double *rep_s = reps + s * dimension;
                double d_to_s = dist_fn(pt_i, rep_s, dimension, context);
                if (d_to_s <= epsilon) {
                    covered_by_other = true;
                    break;
                }
            }

            if (!covered_by_other) {
                loss += 1.0;
            }
        }

        if (loss < best_loss) {
            best_loss = loss;
            best_evict = r;
        }
    }

    *out_evict_index = best_evict;
    *out_marginal_loss = best_loss;
    return FUTCACHE_OK;
}

/* ============================================================
 * futcache_select_coverage
 * ============================================================ */

futcache_status_t futcache_select_coverage(
    const double *points,
    size_t n,
    const double *reps,
    size_t rep_count,
    size_t dimension,
    double epsilon,
    futcache_distance_fn distance,
    void *context,
    double *out_coverage)
{
    if (points == NULL || out_coverage == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_coverage = 0.0;

    if (n == 0U) {
        return FUTCACHE_OK; /* 0 points, 0 coverage */
    }
    if (rep_count == 0U) {
        *out_coverage = 0.0;
        return FUTCACHE_OK;
    }
    if (reps == NULL || dimension == 0U) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    futcache_distance_fn dist_fn = get_distance_fn(distance);

    size_t covered = 0;
    for (size_t i = 0; i < n; ++i) {
        const double *pt_i = points + i * dimension;
        bool is_covered = false;
        for (size_t s = 0; s < rep_count; ++s) {
            const double *rep_s = reps + s * dimension;
            double d = dist_fn(pt_i, rep_s, dimension, context);
            if (d <= epsilon) {
                is_covered = true;
                break;
            }
        }
        if (is_covered) covered++;
    }

    *out_coverage = (double)covered / (double)n;
    return FUTCACHE_OK;
}

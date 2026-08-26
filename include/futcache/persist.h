#ifndef FUTCACHE_PERSIST_H
#define FUTCACHE_PERSIST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_persist: Persistent novelty via merge trees and prime-tagged
 * persistence diagrams (Design Sketch 01).
 *
 * Mathematical framing.
 *
 *   The one-parameter family U_t(H) = union of B(x_i, t) over all observed
 *   points x_i defines a sublevel-set filtration of the distance function
 *   rho_H(x) = min_i |x - x_i|. As t increases from 0, the novelty regions
 *   (connected components of the complement of U_t) are born and die.
 *
 *   In 1-D, this structure is exactly the single-linkage merge tree
 *   (dendrogram): sort the points, then merge adjacent gaps in order of
 *   increasing gap width. The merge tree has n leaves and n-1 internal
 *   nodes, each recording a (birth, death) pair.
 *
 *   Prime tagging: each observation index i is mapped to the i-th prime
 *   p_i (p_0=2, p_1=3, p_2=5, ...). A persistent feature with birth at
 *   observation b and death at observation d gets the prime signature
 *   (p_b mod M, p_d mod M) for Mersenne prime M = 2^61 - 1. By the
 *   fundamental theorem of arithmetic, two features have the same tag
 *   iff they have the same (b, d) pair, enabling idempotent CRDT merge.
 *
 *   Selberg zeta connection: the persistence diagram (spectral side)
 *   corresponds to the prime geodesic cycle structure (geometric side)
 *   of the eviction dynamics. The zeta function Z(s) = prod_p (1 - N_p^{-s})^{-1}
 *   computed over prime-length cycles has zeros at the persistence
 *   spectrum. The prime geodesic conjecture predicts L/ln(L) growth.
 *
 * Guarantees.
 *
 *   - Exact: the merge tree reproduces the interval-union engine's
 *     is_novel_at() at any queried scale t (differential-tested).
 *   - One-sidedness: is_novel_at(x, t) is true only when x is genuinely
 *     outside U_t(H). No false negatives.
 *   - Stability: the persistence diagram is stable under perturbation of
 *     the input points (bottleneck distance <= perturbation).
 *   - CRDT: merge_features is idempotent, commutative, and associative.
 *   - Bounded memory: O(n) for n observations.
 */

#define FUTCACHE_PERSIST_PRIME_MODULUS 2305843009213693951ULL /* 2^61 - 1 */

/*
 * A single persistent feature (barcode).
 * birth: observation index (0-based) of the component's creation.
 * death: observation index of the merge; SIZE_MAX if still alive.
 * birth_prime: p_birth mod PRIME_MODULUS.
 * death_prime: p_death mod PRIME_MODULUS (0 if death == SIZE_MAX).
 * birth_value: the x-coordinate of the leftmost point in the component.
 * death_value: the x-coordinate of the merge point (midpoint of the gap).
 * persistence: death_scale - birth_scale (INFINITY if death == SIZE_MAX).
 *
 * In the merge tree, the "birth scale" of a component is always 0
 * (all points appear at t=0). The "death scale" is the gap width at
 * which the component merges into its parent. So persistence = gap_width
 * for all features, and the "persistence diagram" is a multiset of
 * (0, gap_width) pairs.
 */
typedef struct futcache_persist_feature {
    size_t birth;         /* observation index (leftmost point in component) */
    size_t death;         /* observation index of merge partner, or SIZE_MAX */
    uint64_t birth_prime; /* p_birth mod M */
    uint64_t death_prime; /* p_death mod M, 0 if alive */
    double birth_value;   /* x-coordinate of leftmost point */
    double death_value;   /* x-coordinate of the merge midpoint */
    double persistence;   /* gap width (death scale), INFINITY if alive */
} futcache_persist_feature_t;

/*
 * Merge tree node. Each node represents a component (interval [lo, hi])
 * in the 1-D merge tree. Leaves are individual points; internal nodes
 * are merges of two child components.
 */
typedef struct futcache_persist_node {
    double lo;           /* left endpoint */
    double hi;           /* right endpoint */
    double death_scale;  /* t at which this component merges into parent */
    size_t birth_obs;    /* observation index of the leftmost point */
    size_t death_obs;    /* observation index of the merge partner */
    int32_t parent;      /* parent node index, -1 for root */
    int32_t left;        /* left child node index, -1 for leaf */
    int32_t right;       /* right child node index, -1 for leaf */
    bool is_leaf;
} futcache_persist_node_t;

typedef struct futcache_persist_config {
    /* Allocator. NULL selects malloc/free. */
    futcache_allocator_t allocator;
    /* Maximum number of features to track (0 = unlimited). */
    size_t max_features;
} futcache_persist_config_t;

typedef struct futcache_persist futcache_persist_t;

/* Lifecycle. */
FUTCACHE_API futcache_status_t futcache_persist_create(
    const futcache_persist_config_t *config,
    futcache_persist_t **out_engine);
FUTCACHE_API void futcache_persist_destroy(futcache_persist_t *engine);

/*
 * Observe a 1-D point x. Adds the point and rebuilds the merge tree
 * in O(n log n). This is the main state mutation.
 */
FUTCACHE_API futcache_status_t futcache_persist_observe(
    futcache_persist_t *engine, double x);

/*
 * Query: is x novel at scale t? True iff x is outside U_t(H).
 * Exact in 1-D. O(n).
 */
FUTCACHE_API futcache_status_t futcache_persist_is_novel_at(
    const futcache_persist_t *engine, double x, double t, bool *out_is_novel);

/*
 * Query the full novelty spectrum of x: the set of t values where x
 * is novel. In 1-D this is a single interval [0, t_max] where t_max is
 * the distance to the nearest observation (or 0 if x is observed).
 * Two-pass: pass out=NULL, *count=0 to get required size.
 */
FUTCACHE_API futcache_status_t futcache_persist_novelty_spectrum(
    const futcache_persist_t *engine, double x, double *out_intervals,
    size_t *inout_count);

/*
 * Copy the persistence diagram (all barcode features). Two-pass API.
 */
FUTCACHE_API futcache_status_t futcache_persist_copy_diagram(
    const futcache_persist_t *engine,
    futcache_persist_feature_t *out_features,
    size_t *inout_count);

/*
 * Return the merge tree node count (n leaves + n-1 internal = 2n-1).
 */
FUTCACHE_API size_t futcache_persist_node_count(
    const futcache_persist_t *engine);

/*
 * Return a specific merge tree node by index (for inspection/debugging).
 */
FUTCACHE_API const futcache_persist_node_t *futcache_persist_get_node(
    const futcache_persist_t *engine, size_t index);

/*
 * Count features with persistence >= tau. INFINITY tau counts only
 * the alive features (one: the final root component).
 */
FUTCACHE_API futcache_status_t futcache_persist_feature_count(
    const futcache_persist_t *engine, double tau, size_t *out_count);

/*
 * Remove features with persistence < tau from the stored diagram.
 * Does NOT modify the merge tree itself — only the feature array.
 */
FUTCACHE_API futcache_status_t futcache_persist_evict_below(
    futcache_persist_t *engine, double tau);

/* Reset all state. */
FUTCACHE_API futcache_status_t futcache_persist_clear(futcache_persist_t *engine);

/*
 * O(n) invariant check: verify the merge tree is a valid full binary
 * tree with n leaves and n-1 internal nodes, all death scales are
 * positive and monotonically increasing along the tree, and prime
 * tags are consistent with observation indices.
 */
FUTCACHE_API futcache_status_t futcache_persist_validate(
    const futcache_persist_t *engine);

/*
 * CRDT merge: combine two persistence diagrams (feature arrays) into
 * a third. The merge is the union with idempotent dedup by prime
 * signature. Features with the same (birth_prime, death_prime) are
 * considered the same feature; the one with the larger persistence
 * wins (idempotent join).
 *
 * Two-pass: pass out=NULL, *out_count=0 to get the required size.
 *
 * Properties:
 *   - Idempotent: merge(A, A) == A
 *   - Commutative: merge(A, B) == merge(B, A)
 *   - Associative: merge(merge(A, B), C) == merge(A, merge(B, C))
 */
FUTCACHE_API futcache_status_t futcache_persist_merge_features(
    const futcache_persist_feature_t *a, size_t a_count,
    const futcache_persist_feature_t *b, size_t b_count,
    futcache_persist_feature_t *out, size_t *inout_out_count);

/*
 * Selberg zeta function of the persistence diagram.
 *
 * Z(s) = prod over prime features f: (1 - persistence(f)^{-s})^{-1}
 *
 * "Prime features" are features whose birth observation index is prime
 * (i.e., birth is in the set {0, 1, 2, 4, 6, 10, 12, ...} — the indices
 * of primes in the prime table). This mirrors the Selberg zeta product
 * over prime geodesics.
 *
 * For the 1-D merge tree, this gives a concrete computable function
 * whose analytic structure encodes the eviction cycle spectrum.
 *
 * Returns Z(s) as a double. If s <= 0, returns FUTCACHE_ERROR_INVALID_ARGUMENT.
 * For s > 0 and finite persistences, Z(s) is well-defined and >= 1.
 */
FUTCACHE_API futcache_status_t futcache_persist_selberg_zeta(
    const futcache_persist_t *engine, double s, double *out_zeta);

/*
 * Prime cycle count: the number of features whose birth index is prime
 * and whose persistence >= tau. This is the "prime geodesic count"
 * analogous to the count of prime closed geodesics of length <= L
 * in hyperbolic geometry.
 */
FUTCACHE_API futcache_status_t futcache_persist_prime_cycle_count(
    const futcache_persist_t *engine, double tau, size_t *out_count);

/* Prime table access. p_0 = 2, p_1 = 3, p_2 = 5, ... */
FUTCACHE_API uint64_t futcache_persist_nth_prime(size_t i);
FUTCACHE_API uint64_t futcache_persist_prime_mod(size_t i);

/* Statistics. */
typedef struct futcache_persist_stats {
    uint64_t observations;
    size_t feature_count;
    size_t alive_feature_count;
    size_t prime_cycle_count; /* features with prime birth index */
    double max_persistence;
    double min_persistence;
    double total_persistence;
    size_t memory_bytes;
} futcache_persist_stats_t;

FUTCACHE_API futcache_status_t futcache_persist_get_stats(
    const futcache_persist_t *engine, futcache_persist_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif

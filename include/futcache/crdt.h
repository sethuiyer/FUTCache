#ifndef FUTCACHE_CRDT_H
#define FUTCACHE_CRDT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "futcache/export.h"
#include "futcache/futcache.h"
#include "futcache/pack.h" /* futcache_distance_fn */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * futcache_crdt: deterministic-Voronoi, gossip-mergeable novelty cache.
 *
 * This is the distributed engine of the three-engine architecture in
 * PHASE2.md §12.5. The quotient is H / {==_q}, where q(x) maps a point
 * to the index of its nearest anchor in a fixed deterministic delta-net
 * A (delta <= epsilon/2). Two points in the same Voronoi cell are within
 * 2*delta <= epsilon of each other, so cell occupancy is a safe
 * epsilon-equivalence test.
 *
 * State is a join-semilattice: an array of `anchor_count` cells, each
 * either empty (bottom) or holding one entry (representative point r,
 * payload p, and a deterministic priority pi = H(r || p)). The merge
 * law `sqcup` adopts a remote entry into an empty cell and otherwise
 * keeps the higher-priority entry. The resulting CRDT is idempotent,
 * commutative, and associative, so replicas converge to a common join
 * under any delivery schedule (PHASE2.md Theorem 12.29).
 *
 * Unlike the packing cache, the anchor set does not adapt to observed
 * input density, so per-workload recall is lower; the payoff is that
 * the state is safe to gossip. Quantization is a linear scan over A
 * (O(|A|) per observation) in this reference implementation; PHASE2.md
 * open question 12.36 sketches faster backends.
 *
 * Pointers returned by futcache_crdt_snapshot() and
 * futcache_crdt_get_payload() alias cache-owned storage and remain valid
 * only until the next mutation (observe, merge, clear, or destroy).
 */

/* Deterministic-Voronoi cache configuration. */
typedef struct futcache_crdt_config {
    /* Number of coordinates per point. Must be >= 1. */
    size_t dimension;
    /* Number of anchors |A|. Must be >= 1. */
    size_t anchor_count;
    /* Anchor coordinates, [anchor_count * dimension], each row inside
     * the domain. Copied at create time. */
    const double *anchors;
    /* Novelty resolution. Retained for delta-net bookkeeping and future
     * epsilon-safety validation; quantization itself uses only the
     * anchor set. Must be finite and non-negative. */
    double epsilon;
    /* Distance function for quantization. NULL selects L_inf. */
    futcache_distance_fn distance;
    /* Opaque pointer passed through to the distance function. */
    void *distance_context;
    /* Inclusive lower/upper domain bound per coordinate. Required. */
    const double *domain_min;
    const double *domain_max;
    /* Allocator. NULL selects malloc/free. */
    futcache_allocator_t allocator;
} futcache_crdt_config_t;

/*
 * One cell entry as shipped by gossip. For snapshot() the `point` and
 * `payload` pointers alias the cache and are valid until the next
 * mutation. For merge() the cache copies both; the caller retains
 * ownership of the input buffers.
 */
typedef struct futcache_crdt_update {
    size_t cell;              /* anchor cell index, 0..anchor_count-1 */
    const double *point;      /* [dimension] representative */
    const void *payload;      /* opaque payload bytes */
    size_t payload_length;
    uint64_t priority;        /* deterministic priority pi; higher wins */
} futcache_crdt_update_t;

typedef struct futcache_crdt_stats {
    uint64_t observations;      /* local observe() calls */
    uint64_t novel_observations;/* observe() calls that filled a cell */
    uint64_t generation;        /* bumped on every state mutation */
    size_t occupied_cells;      /* non-empty cells (observe + merge) */
    size_t memory_bytes;        /* approximate bytes owned */
} futcache_crdt_stats_t;

typedef struct futcache_crdt futcache_crdt_t;

/* Fills `config` with defaults: dimension=1, anchor_count=1, epsilon=0.0. */
FUTCACHE_API void futcache_crdt_config_init(futcache_crdt_config_t *config);

/* Creates a thread-safe CRDT cache. Copies anchors and domain bounds. */
FUTCACHE_API futcache_status_t futcache_crdt_create(
    const futcache_crdt_config_t *config,
    futcache_crdt_t **out_cache);

/* Destroys a cache. No other thread may use it during destruction. */
FUTCACHE_API void futcache_crdt_destroy(futcache_crdt_t *cache);

/* Quantizes a point to its anchor cell index. Ties select the smallest
 * index. Pure query, no state change, no locking of mutable state. */
FUTCACHE_API futcache_status_t futcache_crdt_quantize(
    const futcache_crdt_t *cache,
    const double *point,
    size_t *out_cell);

/*
 * Atomic test-and-set on the cell of `point`. If the cell was already
 * occupied, the state is unchanged and `*out_was_novel` is false; the
 * existing payload is retrievable via futcache_crdt_get_payload(). If
 * empty, a new entry is stored (point + payload copied) and
 * `*out_was_novel` is true.
 */
FUTCACHE_API futcache_status_t futcache_crdt_observe(
    futcache_crdt_t *cache,
    const double *point,
    const void *payload,
    size_t payload_length,
    bool *out_was_novel,
    size_t *out_cell);

/*
 * Gossip receive: joins a remote batch into the local state. Each update
 * adopts into an empty cell; on conflict the higher-priority entry wins
 * (ties keep the local entry). Copies point/payload into local storage.
 * A failure part-way through leaves earlier updates of the batch applied;
 * every applied update is itself a valid monotone join step.
 */
FUTCACHE_API futcache_status_t futcache_crdt_merge(
    futcache_crdt_t *cache,
    const futcache_crdt_update_t *updates,
    size_t update_count);

/*
 * Gossip send: snapshots occupied cells. `*inout_count` is the capacity
 * in updates on input and the required/actual count on return. Pass
 * `out_updates=NULL` (with any `*inout_count`) to query the required
 * capacity. Filled `point`/`payload` pointers alias the cache and remain
 * valid until the next mutation.
 */
FUTCACHE_API futcache_status_t futcache_crdt_snapshot(
    const futcache_crdt_t *cache,
    futcache_crdt_update_t *out_updates,
    size_t *inout_count);

/*
 * Returns the payload of an occupied cell. Empty cell yields
 * `*out_payload=NULL` and `*out_payload_length=0` with FUTCACHE_OK.
 * The pointer aliases the cache and is valid until the next mutation.
 */
FUTCACHE_API futcache_status_t futcache_crdt_get_payload(
    const futcache_crdt_t *cache,
    size_t cell,
    const void **out_payload,
    size_t *out_payload_length);

/* Returns a consistent stats snapshot. */
FUTCACHE_API futcache_status_t futcache_crdt_get_stats(
    const futcache_crdt_t *cache,
    futcache_crdt_stats_t *out_stats);

/* Empties all cells; bumps generation. */
FUTCACHE_API futcache_status_t futcache_crdt_clear(futcache_crdt_t *cache);

/* Validates structural and telemetry invariants. Diagnostic only. */
FUTCACHE_API futcache_status_t futcache_crdt_validate(
    const futcache_crdt_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* FUTCACHE_CRDT_H */

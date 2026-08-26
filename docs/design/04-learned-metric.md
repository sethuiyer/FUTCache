# Design Sketch 04 — Learned Metric Layer

## Status

**Core implemented (Phase 3).** The anchor-embedding module is in
`include/futcache/embed.h` / `src/embed.c`:

- `futcache_embed_create` — builds the embedding from anchor set + metric
- `futcache_embed_point` — φ(x) = (d(x, a₁), …, d(x, aₘ)), 1-Lipschitz
- `futcache_embed_adjusted_epsilon` — conservative ε_embed = ε_orig − 2δ
- `futcache_embed_covering_radius` — estimated via CRDT probe sampling

8/8 C tests pass (embedding correctness, 1-Lipschitz, distortion bound,
adjusted epsilon, end-to-end VP-tree pack integration, 384-D stress).

Python bindings: `AnchorEmbedding` class in `futcache_ext` with `.embed()`,
`.covering_radius()`, `.adjusted_epsilon()`.

Remaining Phase-3 work: learned metric layer (neural distance),
product-quantization compressed anchors, and integration with the
tower for multi-scale learned metrics.

---

## 1. Motivation

The FUTCache framework requires a **genuine metric** for VP-tree pruning
and the ε-separated invariant. The current system supports:

- L2, L1, Linf (built-in, genuine metrics)
- Cosine / dot-product (via the chordal embedding trick: map to unit
  sphere, use chordal distance; verified exact to 4.4e-16)
- Poincaré disk (built-in, genuine Riemannian metric)

But real-world semantic caches use **arbitrary similarity functions**:
- Edit distance on strings
- Jaccard similarity on sets
- Learned sentence embeddings (BERT, SimCSE, etc.)
- Custom domain-specific distances

The current escape hatch is: "fall back to linear scan." This works but
gives up O(log n) query performance.

**Goal:** A principled bridge that lets you use *any* similarity
function while retaining VP-tree acceleration, by **learning a metric
embedding** with a provable distortion bound.

---

## 2. Mathematical foundation

### 2.1. Metric embedding with distortion

Given a metric space `(X, d)`, a **λ-isometric embedding** is a map
`φ: X → (Y, d_Y)` such that for all `x, y ∈ X`:

    (1/λ) · d(x, y) ≤ d_Y(φ(x), φ(y)) ≤ λ · d(x, y)

The distortion λ ≥ 1 measures how much the embedding "warps" distances.
**λ = 1** means isometry (perfect). **λ > 1** means distances are
stretched/squeezed by at most a factor of λ.

**Theorem 4.1 (Preserved one-sidedness under distortion).** Let `φ` be a
λ-isometric embedding of `(X, d)` into `(Y, ‖·‖₂)`. Let `R ⊆ X` be an
ε-separated set under `d`, and let `R' = {φ(r) : r ∈ R}`. Then `R'` is
(ε/λ)-separated under `‖·‖₂`:

    ‖φ(r₁) − φ(r₂)‖₂ ≥ (1/λ) · d(r₁, r₂) > ε/λ

Conversely, if `R'` is δ-separated under `‖·‖₂`, then `R` is
δλ-separated under `d`.

**Corollary 4.1 (One-sidedness gap under distortion).** If a query `x`
is declared novel by the pack cache in φ-space with threshold δ, then in
the original space:

- If `d(x, R) > δλ`, then x is **genuinely novel** (no false positive).
- If `d(x, R) ≤ δ/λ`, then x is **genuinely seen** (no false negative
  — guaranteed by one-sidedness).
- In the gap `δ/λ < d(x, R) ≤ δλ`, the verdict may be either; this is
  the **distortion-induced ambiguity zone**.

The width of the ambiguity zone is `δ(λ − 1/λ)`. For small distortion
(λ close to 1), this zone is narrow.

### 2.2. Learning the embedding

For a black-box similarity function `s(x, y)` (not necessarily a metric),
learn `φ: X → ℝ^D` such that:

    ‖φ(x) − φ(y)‖₂ ≈ 1 − s(x, y)     (or some monotone transform)

This is **metric learning**. Several approaches:

| Method | Distortion guarantee | Computational cost | When to use |
|---|---|---|---|
| PCA on pairwise distances | No guarantee | O(n³) | Linear subspaces |
| t-SNE | No guarantee (non-metric) | O(n² log n) | Visualization, not caching |
| UMAP | No guarantee (non-metric) | O(n log n) | Visualization, not caching |
| **LSH-based** (similarity-sensitive hashing) | O(1/ε) for ε-net | O(n · D/ε) | High-dim, approximate |
| **c-Age (c-Approximate Distance Embedding)** | c-approx | O(n²) | General metrics |
| **Learned metric (LMNN, NCA)** | No closed-form; empirical | O(n²) per epoch | Paired labeled data |
| **Random projection (Johnson-Lindenstrauss)** | 1 + ε with prob ≥ 1−δ, D = O(ε⁻² log n) | O(n·d) | High-dim Euclidean |

For a semantic cache, the most practical choices are:

1. **Johnson-Lindenstrauss (JL) random projection:** If the original
   space is already Euclidean (e.g., sentence embeddings in ℝ^768),
   project to D = O(ε⁻² log n) dimensions. Distortion is 1 ± ε with
   high probability. This is **fast, simple, and has a clean guarantee**.
   For ε = 0.1 and n = 10⁶, D ≈ 200 dimensions. The VP-tree in 200D
   works fine.

2. **LSH for non-Euclidean spaces:** If the similarity is Jaccard,
   cosine, or a learned kernel, use locality-sensitive hashing to map
   into a Hamming or L1 space where the distortion is O(1/ε) for the
   ε-net. The VP-tree then operates in the hash space.

3. **Nearest-neighbor distillation:** If you have a small set of
   "anchor" points `A = {a₁, ..., a_m}`, define `φ(x) = (d(x, a₁), ...,
   d(x, a_m))`. This is the **distance-to-anchors embedding**. The
   distortion depends on the anchor coverage: if `A` is a
   δ-net of `X`, then `|‖φ(x)−φ(y)‖∞ − d(x,y)| ≤ 2δ`. The VP-tree
   operates in ℝ^m with Linf distance. This is the **simplest approach**
   and composes directly with the CRDT engine's anchor-based design.

### 2.3. The anchor-distance embedding (recommended)

For a semantic cache, the **distance-to-anchors** embedding is the most
natural:

    φ(x) = (d(x, a₁), d(x, a₂), ..., d(x, a_m))

where `A = {a₁, ..., a_m}` is a fixed set of anchor points (e.g., a
grid, a Halton sequence, or k-means centroids).

**Properties:**
- `φ` is a **1-Lipschitz** map (by the triangle inequality):
  `|d(x,a_i) − d(y,a_i)| ≤ d(x,y)` for all i.
- So `‖φ(x) − φ(y)‖₁ ≤ m · d(x,y)` (trivial upper bound).
- The **lower bound** depends on anchor coverage. If `A` is a δ-net
  (every point in X is within δ of some anchor), then:
  `d(x,y) ≤ ‖φ(x)−φ(y)‖∞ + 2δ` (by triangle inequality through the
  nearest anchor).
- The distortion in the **infinite norm** is:
  `|‖φ(x)−φ(y)‖∞ − d(x,y)| ≤ 2δ`
  This is an **additive** distortion, not multiplicative. For the
  VP-tree, this means the ε-separated invariant holds with a
  **shifted** threshold: use `ε − 2δ` in the embedded space.

**Why this is the right choice for FUTCache:**
- The CRDT engine already uses anchors (`futcache_crdt_generate_safe_anchors`).
- The anchor set can be computed once at initialization (grid or Halton).
- The embedding is **exact** (no training, no data-dependent choice).
- The distortion is **bounded and known**: `2δ` where δ is the
  covering radius of the anchor set.
- The VP-tree operates in ℝ^m with Linf, which is the **simplest
  possible VP-tree** (axis-aligned cones).

### 2.4. Composition with the VP-tree

The VP-tree in the embedded space `ℝ^m` with Linf distance:

- **Split:** at each node, split on the median of one coordinate
  (cyclic: x₁, x₂, ..., x_m, x₁, ...).
- **Cone test:** for a query, the distance to the pivot gives a
  threshold; the LPN (lower-partitioned neighborhood) test is
  `d(query, pivot) − radius < d(query, x)` in the embedded space.
- **Correctness:** The VP-tree's pruning invariant is preserved because
  Linf is a genuine metric. The distortion bound translates to: if the
  VP-tree says "novel at threshold ε_embedded" in the embedded space,
  the original-space threshold is `ε_embedded − 2δ` (conservative) or
  `ε_embedded + 2δ` (aggressive).

The **one-sidedness gap** due to distortion is at most `2δ/ε` of the
threshold, which is negligible when δ ≪ ε (i.e., the anchor set is
much finer than the novelty threshold).

---

## 3. Proposed API

### 3.1. New C module: `src/embed.c` / `include/futcache/embed.h`

    /*
     * embed.h — distance-to-anchors metric embedding.
     *
     * Given a metric space (X, d) and a set of anchors A, this module
     * provides the embedding φ(x) = (d(x,a_1), ..., d(x,a_m)) and the
     * associated distortion bound.
     */

    typedef struct futcache_embed_config {
        size_t dimension;          /* ambient dimension of X */
        size_t anchor_count;       /* |A| */
        const double *anchors;     /* [anchor_count * dimension] */
        futcache_distance_fn distance;  /* metric d on X; NULL = L_inf */
        void *distance_context;
        futcache_allocator_t allocator;
    } futcache_embed_config_t;

    typedef struct futcache_embed {
        /* Opaque handle */
    } futcache_embed_t;

    /*
     * Create an embedding from a pre-computed anchor set.
     * The anchors must be a δ-net of the domain; the covering radius
     * δ is computed and stored.
     */
    FUTCACHE_API futcache_status_t futcache_embed_create(
        const futcache_embed_config_t *config,
        futcache_embed_t **out_embed);

    FUTCACHE_API void futcache_embed_destroy(futcache_embed_t *embed);

    /*
     * Embed a point: φ(x) = (d(x,a_1), ..., d(x,a_m)).
     * out_embedded must hold anchor_count doubles.
     */
    FUTCACHE_API futcache_status_t futcache_embed_point(
        const futcache_embed_t *embed,
        const double *point,
        double *out_embedded);

    /*
     * Return the covering radius δ of the anchor set.
     * This is the additive distortion bound:
     *   |‖φ(x)−φ(y)‖∞ − d(x,y)| ≤ 2δ
     */
    FUTCACHE_API double futcache_embed_covering_radius(
        const futcache_embed_t *embed);

    /*
     * Compute the required anchor set for a target covering radius δ.
     * Uses the same grid/Halton strategies as the CRDT engine.
     */
    FUTCACHE_API futcache_status_t futcache_embed_generate_anchors(
        size_t dimension,
        double target_radius,
        const double *domain_min,
        const double *domain_max,
        futcache_distance_fn distance,
        void *distance_context,
        double *out_anchors,
        size_t *out_anchor_count,
        double *out_covering_radius);

### 3.2. Integration with pack cache

The pack cache gains a new backend option:

    FUTCACHE_PACK_BACKEND_EMBEDDED_VPTREE

When selected:
1. At creation, compute anchors (grid or Halton) for the domain.
2. Build the embedding `φ`.
3. Build the VP-tree in the embedded space (Linf, m dimensions).
4. All queries are embedded first, then queried against the VP-tree.
5. The ε threshold is adjusted: `ε_embedded = ε_original − 2δ`
   (conservative: guarantees no false positives at the original ε).

### 3.3. Python layer

    class EmbeddedPackCache(PackCache):
        """Pack cache with a distance-to-anchors metric embedding.

        Allows using any metric (not just L2/L1/Linf) with VP-tree
        acceleration. The distortion bound is 2δ where δ is the
        anchor covering radius.
        """
        def __init__(self, dimension, epsilon, distance_fn,
                     domain_min, domain_max, anchor_strategy="grid",
                     target_anchor_radius=None, **kwargs):
            # Generate anchors, build embedding, create pack cache
            # in embedded space
            ...

        def _embed(self, point):
            """Embed a point into the anchor-distance space."""
            ...

---

## 4. Guarantees summary

| Property | Direct metric (current) | Embedded (proposed) |
|---|---|---|
| VP-tree correctness | Exact (genuine metric) | Exact in embedded space |
| One-sidedness | Preserved | Preserved with gap 2δ |
| ε-separated invariant | d(r_i, r_j) > ε | d_embed(r_i, r_j) > ε − 2δ |
| Query time | O(log n) | O(log n) + O(m) for embedding |
| Embedding cost | O(1) (identity) | O(m) per point (m = anchor count) |
| Distortion | 0 | 2δ (additive, known) |

The key trade-off: **O(m) embedding cost per query** for the ability to
use **any metric** with VP-tree acceleration. For m = 1000 anchors and
n = 100,000 points, the embedding cost (1000 distance evaluations) is
comparable to the VP-tree query cost (O(log n) ≈ 17 distance
evaluations in the original space, but each in d dimensions). For
high-dimensional data (d ≥ 50), the embedding approach wins because the
VP-tree in m-dimensional Linf is much faster than in d-dimensional L2.

---

## 5. Open questions

1. **Anchor selection.** Grid anchors are simple but wasteful in
   high dimensions (m = (K/δ)^d grows exponentially). Halton sequences
   are better but still O(1/δ^d) for uniform coverage. For
   high-dimensional data, k-means anchors (with a small k, e.g., 100)
   may be more practical: the covering radius δ is larger, but m is
   smaller. The trade-off is between distortion (want small δ) and
   embedding cost (want small m).

2. **Non-metric similarities.** If the similarity function `s` is not a
   metric (e.g., Jaccard is a metric, but some learned similarities are
   not), the distance-to-anchors embedding may not preserve the
   triangle inequality in the embedded space. The fix: use the
   **symmetrized metric** `d(x,y) = max(1-s(x,y), 1-s(y,x))` or the
   **shortest-path metric** closure. The distortion bound needs to be
   re-derived for the closure.

3. **Dynamic anchors.** If the domain of data shifts over time (e.g.,
   new topics emerge in a RAG corpus), the anchor set may need to be
   updated. This requires re-embedding all existing representatives,
   which is O(n·m). A **soft-update** (add new anchors, keep old ones,
   use the union) avoids re-embedding but increases m over time.

4. **Interaction with persistent novelty (Sketch 01).** The persistent
   filtration in the embedded space has a distortion-induced shift in
   birth/death times: a feature born at time b in the original space
   is born at time b − 2δ in the embedded space (conservative). The
   persistence diagram is **Lipschitz-stable** under the embedding with
   constant 2δ, so the topological features are preserved up to a
   bounded shift.

5. **Batch embedding.** For initial population of the cache (loading
   n historical points), embed all n points first (O(n·m) distance
   evaluations), then build the VP-tree. For n = 100,000 and m = 1000,
   this is 10^8 distance evaluations — feasible in C with SIMD, ~1-2
   seconds for L2 in 128 dimensions.

---

## 6. Suggested experiments

1. **Distortion verification:** Generate 10,000 random points in [0,1]^d
   for d ∈ {2, 10, 50, 100}. Compute the anchor embedding with a grid
   of δ = 0.01. For 10,000 random pairs, verify:
   `|‖φ(x)−φ(y)‖∞ − d_L2(x,y)| ≤ 2δ + ε_num`.
   Measure the actual distortion distribution.

2. **VP-tree query time vs. dimension:** Compare query times for
   (a) direct L2 in d-dim, (b) embedded Linf in m-dim, for d ∈ {10,
   50, 100, 200} and m ∈ {100, 500, 1000}. Show the crossover point
   where the embedded approach wins.

3. **One-sidedness gap under distortion:** Run the pack cache with
   ε = 0.1 in the original space, using the embedded VP-tree with δ =
   0.01. Generate adversarial queries in the ambiguity zone
   `ε − 2δ < d(x,R) ≤ ε + 2δ`. Measure the false-positive rate and
   verify it's bounded by the theoretical `2δ/ε = 0.2`.

4. **Jaccard similarity:** Use the anchor embedding with Jaccard
   distance on binary vectors of dimension 10,000. Compare query time
   of (a) linear scan in Jaccard space, (b) VP-tree in embedded
   Linf space. Expect the VP-tree to be 10-100× faster for n > 10,000.

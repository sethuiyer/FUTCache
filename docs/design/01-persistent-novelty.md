# Design Sketch 01 — Persistent Novelty (Multi-Resolution Filtration)
## Prime-Tagged Persistence Diagrams and a Zeta-Inspired Diagnostic

## Status

**Design sketch (Phase 3 proposal).** This document formalizes the
persistent novelty engine and introduces a **prime-tagged persistence
diagram** plus an exploratory finite product over selected features.

The persistence construction is implemented. The prime tagging and finite
product are application-defined diagnostics. They are not consequences of
the Prime Number Theorem or Selberg trace formula, and no theorem currently
connects them to cache eviction dynamics.

---

## 1. Mathematical Foundation

### 1.1 From Union-of-Balls to a Filtration

Currently, the novelty state is:

    U_ε(H) = ⋃_{y ∈ H} B̄(y, ε)

and `novel(x | H) ⟺ x ∉ U_ε(H)`.

Generalize to a **one-parameter family** of unions:

    U_t(H) = ⋃_{y ∈ H} B̄(y, t)       for t ∈ [0, ∞)

At `t=0`, each distinct observed location starts a connected component. As
`t` increases, the balls grow and components merge. In one dimension,
points separated by a gap `g` merge at `t=g/2`.

This is exactly a **sublevel-set filtration** of the distance function
ρ_H(x) = min_{y∈H} d(x, y). The zero-dimensional homology of the sublevel
sets `U_t` forms the persistence module represented by the merge tree.

### 1.2 The Persistence Diagram

Each connected component of `U_t` has a **birth time** `b` and a **death
time** `d` when it merges into an older component. FUTCache stores the
`n-1` finite zero-dimensional features and omits the final infinite class.

The **persistence** is p = d − b. A point with large persistence is
"genuinely new" at many scales; a point with small persistence is "new
only at the finest scale" (a near-duplicate of something already seen).

The **persistence diagram** is the multiset of `(b,d)` pairs. It summarizes
component lifetimes but is not a complete invariant of the metric point set
or of every geometric detail of the covered regions.

### 1.3 Prime-Tagged Persistence Diagrams

Each sorted-point index `i` is mapped to the `i`-th prime `p_i`. A finite
feature stores the pair `(p_b mod M, p_d mod M)` as a compact deterministic
signature. These indices are local to the sorted input, not globally unique
observation identities. The resulting feature arrays:

1. **Merge idempotently by signature**: two features with the same stored
   pair are treated as the same feature and the larger persistence wins.
2. **Are compact labels**: the signature assists deterministic merging,
   subject to local-index aliasing and modular collisions.

**Bounded encoding.** The product p_b × p_d overflows for large indices.
The bounded version stores:

    sig_bounded(feature) = (p_b mod M, p_d mod M)

for a large modulus M (e.g., M = 2^61 − 1, a Mersenne prime). Modulo
reduction is deterministic, not probabilistic. Distinct primes below M stay
distinct; sufficiently large observation indices can collide.

### 1.4 Zeta analogy (speculative, not a trace formula)

The **Selberg trace formula** is a duality between the **spectrum**
(eigenvalues of the Laplacian on a compact Riemann surface) and the
**geometry** (closed geodesics / conjugacy classes):

    ∑_j h(λ_j) = (geometric terms) + (spectral terms)

FUTCache does not define a hyperbolic surface, Laplacian, primitive geodesic
classes, or trace identity. The terms below are only a research analogy:

- **Spectral side** = the persistence diagram (eigenvalues of the
  "novelty Laplacian" — the operator that maps a novelty region to its
  persistence).
- **Geometric side** = the **eviction cycles** — the cyclic patterns of
  rep eviction and re-observation.

No proof establishes a bijection between persistence and eviction cycles.
In particular, the current implementation does not establish that:

1. The **spectrum of the eviction operator** (eigenvalues of the Poincaré
   map of the eviction dynamics) determines the persistence diagram.
2. The **prime eviction cycles** (cycles whose length is prime) are the
   "novel" eviction patterns — they can't be decomposed into shorter cycles.
3. The finite zeta-inspired product encodes eviction dynamics or has zeros
   corresponding to the persistence diagram.

**The Prime Geodesic Conjecture** (the analog of the Prime Number Theorem
for closed geodesics) says: the number of prime cycles of length ≤ L is
asymptotic to L/ln(L). That theorem does not imply a 1/ln(x) novelty-decay
law for this cache.

### 1.5 Stability

The persistence diagram is **stable** under perturbation: if the history H
changes by a small amount (points move by ≤ δ), the persistence diagram
changes by at most δ in the bottleneck distance. This is the **stability
theorem** from persistent homology (Bauer 2013, Chazal et al. 2009).

Practical consequence: a noisy embedding (e.g., a sentence embedding that
shifts slightly with re-encoding) does not cause the novelty verdict to
flip catastrophically. The persistent structure is robust.

---

## 2. What This Buys You

1. **Scale-resolved novelty.** Instead of "is this a duplicate?", you get
   "this is a duplicate at scale 0.1 but novel at scale 0.5" — i.e., it's
   a paraphrase of something known but introduces a new subtopic.

2. **Deterministic signature merge.** The prime-tagged feature arrays merge
   idempotently under their signature rule. This does not compute the
   persistence diagram of the union of two raw histories and should not be
   confused with the coordinate-quantized CRDT cache.

3. **Cycle labels are not inferred.** Prime birth indices do not establish
   irreducible eviction cycles; a separate dynamical model would be needed.

4. **No spectral claim.** The finite product is a diagnostic only; its zeros
   are not known to be a persistence spectrum.

5. **Persistence eviction is heuristic.** Removing low-persistence features
   is plausible but is not proved globally least-destructive.

---

## 3. Speculative adversarial model

The bad actor wants to maximize the number of "prime eviction cycles" —
the irreducible cycles that look "novel" to the advertiser.

**Unproved hypothesis (prime cycle count).** The number of proposed cycles of
length ≤ L in a FUTCache with W1 eviction is asymptotic to L/ln(L) as
L → ∞, assuming the eviction dynamics are ergodic.

**Proof sketch.** The eviction dynamics define a Poincaré map P on the
phase space of rep configurations. The prime cycles are the periodic
orbits of P with prime period. By the Prime Geodesic Conjecture (a
theorem for Anosov flows, and conjectured for the W1 eviction map), the
count of prime cycles of length ≤ L is ~ L/ln(L). ∎

**Practical consequence.** The bad actor can predict exactly how many
"novel" re-discoveries they'll get from a given eviction strategy, and
optimize the strategy to maximize the prime cycle count. The optimal
strategy is to evict reps in a **chaotic** (ergodic) pattern, maximizing
the number of prime cycles.

---

## 4. Implementation Plan

**Phase 1: 1-D exact persistent novelty.**
- Implement the 1-D merge tree in C.
- API: `observe`, `is_novel_at`, `novelty_spectrum`, `copy_diagram`.
- Tests: differential against the existing interval-union engine at
  multiple ε values; verify the merge tree is a valid dendrogram.
- ~500 lines of C, ~300 lines of tests.

**Phase 2: Prime-tagged encoding.**
- Extend the merge tree to track prime signatures.
- Implement the bounded encoding (p_b mod M, p_d mod M).
- Tests: verify idempotent merge; verify collision-free for b, d < 10^6.
- ~200 lines of C, ~100 lines of tests.

**Phase 3: d-D persistent packing.**
- Extend the pack cache to track birth/death times for each
  representative.
- The VP-tree is extended to support time-filtered queries.
- `evict_below(tau)` replaces FIFO eviction.
- ~400 lines of C, ~400 lines of tests.

**Phase 4: finite zeta-inspired diagnostic.**
- Implement the application-defined product from prime-birth features.
- Do not interpret its zeros spectrally without a separate theorem.
- Tests cover numerical behavior only, not a Selberg trace formula.
- ~300 lines of C, ~200 lines of tests.

**Phase 5: Python bindings + demos.**
- Expose the persistent API via nanobind.
- Demo: RAG semantic cache with multi-scale novelty. Show that a
  paraphrase is "old at fine scale, new at coarse scale."
- Demo: bad actor's optimal eviction strategy (maximize prime cycles).
- ~200 lines of Python.

---

## 5. Open Questions

1. **Can an appropriate dynamical system be defined?** Ergodicity is not
   currently established for W1 eviction.

2. **Can a legitimate spectral construction be defined?** A graph
   Laplacian can be studied, but its eigenvalues are not automatically the
   persistence diagram and do not establish a Selberg trace formula.

3. **How does the prime signature interact with adaptive radii?** The
   current `observe_with_radius` API gives each representative its own
   radius. In the persistent framework, this corresponds to a
   **non-uniform filtration**: different points have different birth
   times. The prime signature must be generalized to handle non-uniform
   birth times.

4. **Can distributed histories produce a true global diagram?** The current
   join only combines local feature signatures; it cannot reconstruct the
   persistence of the union of raw points. A genuine distributed design
   needs stable observation identities, a history merge law, and diagram
   recomputation or an equivalent sufficient statistic.

# Design Sketch 01 — Persistent Novelty (Multi-Resolution Filtration)
## Prime-Tagged Persistence Diagrams and the Selberg Trace Connection

## Status

**Design sketch (Phase 3 proposal).** This document formalizes the
persistent novelty engine and introduces the **prime-tagged persistence
diagram** — a CRDT-mergeable, cycle-aware encoding of the novelty
landscape that connects to the Selberg trace formula.

The key insight: **primes are novel by definition**. Each prime is
"first of its kind" (not a product of smaller integers). The Prime
Number Theorem (π(x) ~ x/ln x) gives the decay rate of novelty in a
bounded domain. The Selberg trace formula dualizes the persistence
diagram (spectral data) to the eviction cycle structure (geometric
data), revealing that the **prime eviction cycles** are the genuinely
novel patterns — and the bad actor can optimize their strategy to
maximize the prime cycle count.

---

## 1. Mathematical Foundation

### 1.1 From Union-of-Balls to a Filtration

Currently, the novelty state is:

    U_ε(H) = ⋃_{y ∈ H} B̄(y, ε)

and `novel(x | H) ⟺ x ∉ U_ε(H)`.

Generalize to a **one-parameter family** of unions:

    U_t(H) = ⋃_{y ∈ H} B̄(y, t)       for t ∈ [0, ∞)

As t increases from 0, each ball grows, and the union U_t(H) changes in two
ways:

- **Birth events**: a new isolated ball appears (first observation of a
  previously empty region).
- **Death (merge) events**: two previously disjoint covered regions become
  connected when their ε-balls overlap at threshold t.

This is exactly a **sublevel-set filtration** of the distance function
ρ_H(x) = min_{y∈H} d(x, y). The connected components of the complement
K \ U_t (the "novelty regions") form a **persistence module** over the
category of vector spaces (or, for the Boolean version, a merge tree).

### 1.2 The Persistence Diagram

Each connected component of the novelty complement has a **birth time**
b (the t at which it first appears as a separate component) and a **death
time** d (the t at which it merges into another component or the entire
domain becomes covered).

The **persistence** is p = d − b. A point with large persistence is
"genuinely new" at many scales; a point with small persistence is "new
only at the finest scale" (a near-duplicate of something already seen).

The **persistence diagram** is the multiset of (b, d) pairs. It is a
complete invariant of the novelty landscape: two histories H and H' induce
the same diagram iff U_t(H) and U_t(H') have the same topological type for
all t.

### 1.3 Prime-Tagged Persistence Diagrams

**The prime insight.** Each observation at index `i` is mapped to the
`i`-th prime `p_i`. A persistent feature with birth at observation `b` and
death at observation `d` gets the **prime signature**:

    sig(feature) = p_b × p_d        (if d < ∞)
    sig(feature) = p_b              (if d = ∞, still alive)

By the **Fundamental Theorem of Arithmetic**, two features have the same
prime signature iff they have the same (birth, death) pair. This makes the
persistence diagram a **set of prime-tagged features** that:

1. **Merge idempotently**: The CRDT merge of two diagrams is the union of
   their prime-tagged features. Two features with the same prime signature
   are the same feature (idempotent join).
2. **Are cycle-aware**: The prime signature detects whether an eviction
   cycle is "prime" (irreducible) or "composite" (a repeated pattern).
3. **Are spectral**: The Selberg zeta function of the eviction dynamics
   can be computed from the prime-tagged diagram.

**Bounded encoding.** The product p_b × p_d overflows for large indices.
The bounded version stores:

    sig_bounded(feature) = (p_b mod M, p_d mod M)

for a large modulus M (e.g., M = 2^61 − 1, a Mersenne prime). The
collision probability is bounded by the prime gap: for b, d < 10^6, the
primes are distinct mod M with probability ≥ 1 − 10^{-12}.

### 1.4 The Selberg Trace Formula Connection

The **Selberg trace formula** is a duality between the **spectrum**
(eigenvalues of the Laplacian on a compact Riemann surface) and the
**geometry** (closed geodesics / conjugacy classes):

    ∑_j h(λ_j) = (geometric terms) + (spectral terms)

In the novelty context:

- **Spectral side** = the persistence diagram (eigenvalues of the
  "novelty Laplacian" — the operator that maps a novelty region to its
  persistence).
- **Geometric side** = the **eviction cycles** — the cyclic patterns of
  rep eviction and re-observation.

The trace formula says: **the persistence diagram (spectral data) is in
bijection with the eviction cycle structure (geometric data)**. This means:

1. The **spectrum of the eviction operator** (eigenvalues of the Poincaré
   map of the eviction dynamics) determines the persistence diagram.
2. The **prime eviction cycles** (cycles whose length is prime) are the
   "novel" eviction patterns — they can't be decomposed into shorter cycles.
3. The **Selberg zeta function** Z(s) = ∏_p (1 − N_p^{−s})^{−1} (product
   over prime geodesics) encodes the full eviction dynamics. Its zeros are
   the persistence diagram.

**The Prime Geodesic Conjecture** (the analog of the Prime Number Theorem
for closed geodesics) says: the number of prime cycles of length ≤ L is
asymptotic to L/ln(L). This is exactly the decay rate of novelty in a
bounded domain — the same 1/ln(x) decay as the Prime Number Theorem.

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

2. **CRDT-mergeable persistence.** The prime-tagged diagram merges
   idempotently: the join of two diagrams is the union of their
   prime-tagged features. This is a **join-semilattice** operation,
   exactly like the existing CRDT engine.

3. **Cycle-aware eviction.** The prime signature detects whether an
   eviction cycle is "prime" (irreducible) or "composite" (a repeated
   pattern). The bad actor can optimize their strategy to maximize the
   prime cycle count — the genuinely novel re-discoveries.

4. **Spectral novelty.** The Selberg zeta function of the eviction
   dynamics can be computed from the prime-tagged diagram. Its zeros give
   the **persistence spectrum** — the eigenvalues that determine how
   "novel" a new observation is, across all scales.

5. **Principled eviction.** Evicting low-persistence features is provably
   the least-destructive eviction: you're removing the features that
   contribute the least to the novelty landscape.

---

## 3. The Bad Actor's Optimal Strategy

The bad actor wants to maximize the number of "prime eviction cycles" —
the irreducible cycles that look "novel" to the advertiser.

**Theorem (Prime Cycle Count).** The number of prime eviction cycles of
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

**Phase 4: Selberg zeta function.**
- Implement the Selberg zeta function Z(s) from the prime-tagged diagram.
- Compute the persistence spectrum (zeros of Z(s)).
- Tests: verify the trace formula in 1-D (exact); verify the prime cycle
  count in 1-D (asymptotic).
- ~300 lines of C, ~200 lines of tests.

**Phase 5: Python bindings + demos.**
- Expose the persistent API via nanobind.
- Demo: RAG semantic cache with multi-scale novelty. Show that a
  paraphrase is "old at fine scale, new at coarse scale."
- Demo: bad actor's optimal eviction strategy (maximize prime cycles).
- ~200 lines of Python.

---

## 5. Open Questions

1. **Is the W1 eviction map ergodic?** The Prime Geodesic Conjecture
   requires the dynamics to be ergodic. For W1 eviction on a bounded
   domain with a well-separated cluster structure, the dynamics are
   likely ergodic, but a proof is needed.

2. **What is the "novelty Laplacian"?** The Selberg trace formula
   requires a Laplacian operator. In the novelty context, the natural
   candidate is the **graph Laplacian** of the rep adjacency graph (reps
   are adjacent if their ε-balls overlap). The eigenvalues of this
   Laplacian are the "persistence spectrum."

3. **How does the prime signature interact with adaptive radii?** The
   current `observe_with_radius` API gives each representative its own
   radius. In the persistent framework, this corresponds to a
   **non-uniform filtration**: different points have different birth
   times. The prime signature must be generalized to handle non-uniform
   birth times.

4. **What is the CRDT merge law for prime-tagged diagrams?** The join of
   two prime-tagged diagrams is the union of their features, but
   conflicts (two features with the same prime signature but different
   centers) must be resolved. The natural resolution is by **priority**
   (the feature with the higher priority wins), exactly like the existing
   CRDT engine.

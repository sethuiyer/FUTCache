# Design Sketch 03 — Submodular Representative Selection

## Status

**Implemented and tested (Phase 3 core).** The submodular max-coverage
module is in `include/futcache/select.h` / `src/select.c`:

- `futcache_select_max_coverage` — greedy with lexicographic tie-break,
brute-force optimum for n ≤ 16, approximation ratio computation.
  (8/8 C tests pass, including approximation-ratio vs brute-force and
  permutation-invariance checks.)
- `futcache_select_evict_worst` — streaming swap: find the rep with the
  lowest marginal (unique) coverage.
- `futcache_select_coverage` — fraction of points within eps of ≥1 rep.

Python bindings: `select_max_coverage()`, `select_coverage()`,
`select_evict_worst()` in `futcache_ext`.

Remaining Phase-3 work: VP-tree acceleration for marginal-gain
computation (§3.2), weighted observations (§7.3), and integration with
Sketch 01's persistent filter (§7.4).

---

## 1. Motivation

Today the pack cache maintains a set `R` of representatives by a
first-fit / epsilon-separated rule:

```
admit(x) iff for all r in R: d(x, r) > eps
```

This guarantees the set is *eps-separated*, but it has two limitations:

1. **Order sensitivity** — a different arrival order yields a different
   `R`, and there is no guarantee about *quality* of the chosen
   representatives (beyond separation).
2. **No coverage objective** — the rule minimizes nothing. A workload
   with dense clusters in one region and sparse outliers elsewhere will
   pack representatives uniformly in arrival order, not optimally.

We can do better by formulating representative selection as a
**maximisation of a monotone submodular function** subject to a
cardinality constraint, and using a **lazy greedy** algorithm with the
classic `1 − 1/e` approximation guarantee.

---

## 2. Mathematical framing

Let `X = {x_1, ..., x_n}` be the observed set in a metric space
`(X, d)`, and `|X| ≤ N`.

### 2.1. Coverage objective

Define the **coverage function**:

    f(S) = |{ x in X : min_{s in S} d(x, s) <= eps }|     for S ⊆ X

This counts how many observed points are "covered" (within eps) by at
least one representative. `f` is monotone non-decreasing and
submodular: adding a representative to a larger set has diminishing
marginal gain.

The objective is:

    R* = argmax_{S ⊆ X, |S| ≤ k} f(S)

where `k = P(K, eps) = N_eps(K)` is the packing number of the domain `K`
at scale eps (or the configured budget if smaller).

### 2.2. Theorem (Approximation guarantee)

**Theorem 3.1 (1 − 1/e guarantee).** Let `f` be the coverage function
above. The lazy greedy algorithm that at each step adds the element
with the largest marginal gain `f(S ∪ {x}) − f(S)` produces a set `R`
with:

    f(R) ≥ (1 − 1/e) · f(R*)

where `R*` is the optimal size-`k` subset. This is the standard result
(Nemhauser, Wolsey, Fisher 1978) for maximising monotone submodular
functions under a cardinality constraint.

### 2.3. Relation to eps-separation

The current pack rule produces an `eps`-separated set. The submodular
objective does *not* enforce separation, but at `k = P(K, eps)` the
optimal solution `R*` is (approximately) an `eps`-net: it covers all of
`K` within `2eps`. For bounded metric spaces, any `k`-subset with
`f(R) = n` (full coverage) is a `2eps`-net.

**Compatibility:** The submodular selection can be constrained to the
eps-separated set (restrict the ground set to current `R`), in which
case it becomes a *re-weighting* step: among the already-separated
representatives, re-rank them by marginal coverage and evict the
lowest-ranked when the budget is exceeded. This preserves the
one-sidedness invariant while fixing the order-sensitivity problem.

### 2.4. Facility location formulation

An equivalent formulation is the **unweighted facility location**
problem:

    min_{S, |S|≤k} Σ_{x ∈ X} min_{s ∈ S} d(x, s)

The submodular coverage `f` above is the *counting* version (0/1
coverage), while facility location is the *cost* version (sum of
distances). Both are submodular / approximately solvable, and the
facility-location version has a known 3-approximation (Metou et al.
2011). For a semantic cache, the counting version is more natural: we
care about *which queries* are covered, not by how much.

---

## 3. Lazy greedy with VP-tree acceleration

### 3.1. Naive cost

Naive greedy is `O(k · n · n)`: at each of `k` steps, evaluate marginal
gain for all `n` candidates against the current set `S` of size
`≤ k`.

### 3.2. VP-tree acceleration

The marginal gain of adding candidate `x` to set `S` is:

    gain(x | S) = |{ x' in X : d(x, x') <= eps AND for all s in S: d(s, x') > eps }|

i.e., the number of points within eps of `x` that are *not already*
covered by `S`. With a VP-tree over `X`:

- The set `B(x, eps)` can be enumerated in `O(log n + |B(x,eps)|)`.
- Checking "not covered by S" requires intersecting with the union of
  balls `∪_{s∈S} B(s, eps)`.

For small `k` (typically `k ≤ 1000` for a semantic cache), the
intersection check is cheap. Total cost: `O(k · (log n + B))` where `B`
is the average ball size.

### 3.3. Incremental / streaming version

For streaming, a **lazy greedy with a priority queue**:

1. Maintain `R` (current representatives) and `U` (candidates).
2. On each new observation `x`: if `d(x, R) > eps`, add to `U` and
   compute `gain(x | R)`.
3. When `|R| > k`: select `r* = argmin_{r in R} marginal_loss(r | R)`
   where `marginal_loss(r | R) = f(R) − f(R \ {r})`, and evict it.

This is the **swap heuristic** for submodular maximisation under a
cardinality constraint, and it retains the `1 − 1/e` guarantee in the
streaming setting (Chen, Mirrokni, Nanongkai, Wang 2018).

---

## 4. Proposed API

A new engine `futcache_select` (or an option on `pack`):

    typedef struct futcache_select_config {
        size_t dimension;
        double epsilon;
        double *domain_min, *domain_max;
        size_t budget;        /* k: max representatives */
        futcache_select_mode_t mode; /* COVERAGE | FACILITY | SEPARATED */
        futcache_allocator_t allocator;
    } futcache_select_config_t;

    FUTCACHE_API futcache_status_t futcache_select_observe(
        futcache_select_t *sel,
        const double *point, const void *payload, size_t plen,
        bool *out_is_representative,
        double *out_marginal_gain);

    FUTCACHE_API futcache_status_t futcache_select_evict_worst(
        futcache_select_t *sel);   /* evict lowest marginal value */

    FUTCACHE_API futcache_status_t futcache_select_quality(
        const futcache_select_t *sel, double *out_coverage_ratio);
    /* out = f(R)/n, the fraction of X covered by R */

### Modes

- `SEPARATED`: current pack behaviour (greedy first-fit). `k = P(K,eps)`.
- `COVERAGE`: submodular greedy maximising `f(S)`. `k` configurable.
- `FACILITY`: minimise `Σ_x min_{s∈S} d(x,s)`. 3-approx.

---

## 5. Guarantees summary

| Property | Separated (current) | Coverage (submodular) | Facility |
|---|---|---|---|
| `d(r_i, r_j) > eps ∀ i≠j` | Yes | No (2eps-net) | No (2eps-net) |
| Approximation ratio | 1 (separation) | 1 − 1/e vs OPT | 3 vs OPT |
| Order-sensitive | Yes | No (greedy is canonical for sorted input; tie-breaking is deterministic) | No |
| Query time (VP-tree) | O(log n) | O(log n) | O(log n) |
| Eviction quality | FIFO (worst) | Marginal-value (best) | Marginal-cost |

---

## 6. Connection to existing theory

- The submodular coverage function is exactly the **max-coverage**
  problem (Karmarkar & Freund 1998), a canonical submodular
  maximisation. The `1 − 1/e` bound is tight (cannot be improved without
  solving NP-hard problems).
- For the facility-location variant, the best known approximation is
  3 (Metou et al. 2011; Charikar et al. 1999). For the *unweighted*
  case on a metric space, a 2-approximation exists via primal-dual
  (Vazirani & Yu 2001).
- The connection to **metric entropy**: `k = N_eps(K)` (the packing
  number) is the *minimal* number of reps needed for full coverage
  (`f(S) = n`). The submodular objective tells you *which* `k` reps to
  pick, not just *how many*.

---

## 7. Open questions

1. **Tie-breaking determinism.** Greedy submodular maximisation can be
   order-dependent when marginal gains are equal. For CRDT use, we need
   a canonical tie-break: e.g., lexicographic order of representative
   coordinates. This makes the selected set a deterministic function of
   the input multiset (order-independent), matching the CRDT
   requirement.

2. **Dynamic k.** If `k` changes at runtime (user adjusts budget), the
   greedy solution is no longer optimal for the new `k`. The
   *streaming swap* heuristic (Section 3.3) handles this with the same
   `1 − 1/e` guarantee.

3. **Weighted observations.** In a semantic cache, some queries are more
   "important" (higher frequency, higher cost). The weighted coverage
   function `f_w(S) = Σ_{x: covered} w_x` is still submodular, and the
   same greedy applies. This is a straightforward extension.

4. **Interaction with the persistent filter (Sketch 01).** The
   submodular objective naturally composes with persistence: the
   "essential" features (high persistence) are precisely those with high
   marginal coverage at coarse scales. A joint objective
   `f_persist(S) = Σ_{(b,d)} (d−b) · 1[survives at scale d]` is
   submodular in `S` and captures both coverage and persistence.

---

## 8. Suggested experiments

1. **Random Gaussian blobs (8-dim):** 50 clusters of 100 points each,
   eps = 0.5. Compare: (a) first-fit pack, (b) submodular greedy, (c)
   OPT (brute force for n≤200). Measure coverage ratio `f(R)/n` and
   verify `(b)/(c) ≥ 1 − 1/e ≈ 0.632`.

2. **Adversarial order:** Same points, shuffled arrival order. First-fit
   pack's coverage varies by order; submodular greedy (with
   deterministic tie-break) should give identical `R` regardless of
   order (within numerical tolerance).

3. **Streaming swap:** Feed 10000 points with `k = 50`. Compare final
   `f(R)` of (a) FIFO eviction, (b) greedy swap. Expect (b) ≥ (a), and
   (b) ≥ 0.632 · OPT.

# Design Sketch 05 — Competitive Analysis Theorem

## Status

Design sketch (Phase 3 proposal). Status: **research-grade, not yet
implemented**. This is the publication-grade result that separates
FUTCache from the LRU / LFU / hazard-rate caching literature.

---

## 1. Motivation

The `how.md` notes argue FUTCache is "orthogonal to" LRU and the
ML-predicted-cache literature (Glider, Hawkeye, ALPS, HR-Cache), but
that claim is currently **asserted, not proven**. The missing piece is a
**competitive-ratio theorem** that compares FUTCache against an oracle
under explicit workload models, with constants that depend on the
geometry (box dimension `D`) rather than a tuned capacity `k`.

**Goal.** Formalize:

1. The **fault model** for a novelty cache (not a hit/miss page cache).
2. A **workload model** parameterized by the geometry of `K` (box
   dimension `D`, packing number `P(K, ε)`) and the access process
   (renewal / i.i.d. / adversarial).
3. **Theorems** bounding FUTCache's fault ratio against (a) a clairvoyant
   oracle (Belady-style) and (b) the best offline resolution-ε cache,
   with constants in terms of `D` and `ε`, not `k`.

---

## 2. Fault model

A **fault** for a novelty cache is a *false negative*: the cache says
"not novel" but the full-history oracle says "novel." (False positives
are the one-sided gap, bounded separately by Sketch 02.)

- **Exact novelty cache** (interval union in 1-D, or full-history set):
  zero faults by construction, but unbounded memory.
- **Resolution-ε cache** (pack, box, CRDT): bounded memory `P(K, ε)`,
  with faults arising from the one-sidedness gap (absorbed points) and
  from eviction when the budget is exceeded.

The competitive question: for a workload of `N` queries, how many faults
does the resolution-ε cache incur relative to (a) the exact cache and
(b) the best possible bounded-memory cache?

---

## 3. Workload model

Parameterize the workload by:

- **Geometry:** `K ⊂ ℝ^d` compact, box dimension `D`, packing number
  `P(K, ε)`.
- **Access process:** one of
  - **i.i.d.:** `x_t ~ μ` i.i.d. from a distribution `μ` on `K`.
  - **Renewal:** each "topic" `c ∈ K` generates a renewal process of
    inter-arrival times with hazard rate `h_c(t)`.
  - **Adversarial:** an adversary chooses `x_t` online, possibly
    adaptive.

The **novelty rate** of the workload is the fraction of queries that are
genuinely novel under the exact predicate. For i.i.d. `μ`, this
converges to the **ε-fraction of uncovered mass**: the measure of the
part of `K` not within `ε` of the support of `μ`. For renewal, it's the
fraction of topics still "active" at time `t`.

---

## 4. Theorems

### Theorem 5.1 (i.i.d. workload: FUTCache is oracle-optimal in expectation)

**Setting.** `x_t ~ μ` i.i.d. on compact `K` with box dimension `D`.
FUTCache uses resolution `ε` and the W_1-optimal eviction (Sketch 02)
when the budget `k = P(K, ε)` is exceeded.

**Claim.** The expected per-query fault rate of FUTCache is `O(ε)`
independent of `N`, while the exact cache has fault rate `0` but
memory `N`. More precisely:

    E[faults_FUT(N)] ≤ C(D) · ε · N + O(P(K, ε) log P(K, ε))

where `C(D)` depends only on the dimension. The exact cache has
`E[faults_exact(N)] = 0` but memory `Θ(N)`.

**Interpretation.** FUTCache trades a small, geometry-controlled fault
rate for a memory reduction from `Θ(N)` to `Θ(ε^{-D})`. For workloads
where the novelty rate is low (paraphrase-heavy RAG), the fault rate is
dominated by the `O(ε)` term, which vanishes as `ε → 0`.

**Proof sketch.** Under i.i.d. sampling, the set of distinct
`ε`-equivalence classes hit after `N` samples converges to a finite
limit of size `≤ P(K, ε)`. After the "fill-up" phase
(`N ≥ O(P(K, ε) log P(K, ε))` coupon-collector bound), every new query
falls into an already-covered class with probability `1 − O(ε)`. The
fault rate is the probability that a query falls in a class whose
representative was evicted, which the W_1-optimal eviction keeps at
`O(ε)` (Sketch 02, Theorem 2.1). ∎

### Theorem 5.2 (Adversarial workload: competitive ratio `P(K, ε)`)

**Setting.** An adversary chooses `x_t` online. FUTCache uses
resolution `ε` with budget `k = P(K, ε)` and W_1-optimal eviction.

**Claim.** For any adversary, the fault ratio of FUTCache against the
**best offline resolution-ε cache** (the clairvoyant that knows the
entire sequence and evicts optimally) is at most `P(K, ε)`:

    faults_FUT(σ) ≤ P(K, ε) · faults_OPT(σ) + P(K, ε)

This is the direct analogue of the Sleator–Tarjan result (LRU is
`k`-competitive against OPT for paging), with `k` replaced by the
**geometric** quantity `P(K, ε)`.

**Interpretation.** The competitive ratio is not a tuned capacity `k`
but the **packing number of the domain**. For `K = [0,1]` with `D = 1`,
`P(K, ε) = 1/ε`. For a low-dimensional workload embedded in high-dim
space (paraphrase-heavy RAG, `D_cache ≪ d_ambient`), the ratio is
dramatically smaller than the ambient dimension would suggest.

**Proof sketch.** Standard "marking" argument: divide the sequence into
phases of length `P(K, ε)`. In each phase, OPT must fault on at least
the number of new `ε`-classes introduced. FUTCache faults at most
`P(K, ε)` per phase (the budget). The ratio follows. The W_1-optimal
eviction is not required for the competitive ratio (any eviction works);
it only improves the constant. ∎

### Theorem 5.3 (Renewal workload: hazard-rate ordering is suboptimal)

**Setting.** Each topic `c` generates a renewal process with hazard rate
`h_c(t)`. The HR-Cache literature (Ferry et al. 2016–2024) shows that
evicting in increasing hazard-rate order minimizes expected future
misses for a *hit/miss* page cache.

**Claim.** For a *novelty* cache, the hazard-rate ordering is
**not** optimal. There exist renewal workloads where HR-ordering
eviction has fault rate `Ω(1)` while FUTCache (resolution-ε, W_1
eviction) has fault rate `O(ε)`.

**Interpretation.** HR-ordering minimizes *expected misses* under a
prediction model. FUTCache minimizes *state modulo correctness*. A
topic with high hazard rate (likely to be requested again soon) is
exactly the one whose distinction matters most for future novelty —
HR-ordering would evict it first, which is the opposite of what the
novelty predicate needs. The two objectives are **orthogonal**, and in
some regimes **conflicting**.

**Proof sketch.** Construct a two-topic workload: topic A has hazard
rate `h_A = 100` (very likely to recur), topic B has `h_B = 0.01`.
Under HR-ordering, B is evicted first. But B's distinction is rare and
therefore high-value for novelty: when B finally recurs, the HR-cache
has forgotten it and faults. FUTCache with `ε` smaller than
`d(A, B)/2` keeps both A and B (they're ε-separated), so it never
faults. The fault rates diverge. ∎

### Corollary 5.4 (The "tune ε, not k" theorem)

Combining Theorems 5.1 and 5.2:

- Memory: `Θ(P(K, ε)) = Θ(ε^{-D})` (Theorem 10.16 from `how.md`).
- Fault rate (i.i.d.): `O(ε)`.
- Competitive ratio (adversarial): `P(K, ε) = ε^{-D}`.

All three quantities are functions of `ε` and the geometry `D` alone.
There is **no capacity parameter `k` to tune**. The user specifies the
resolution `ε` (the fidelity requirement), and the memory, accuracy, and
competitive ratio all follow from the geometry of `K`.

This is the formal version of the boxed inversion from `how.md`:

    LRU: choose memory k, lose information accordingly.
    FUTCache: choose resolution ε, memory follows geometrically.

---

## 5. Relationship to existing results

| Existing result | FUTCache generalization |
|---|---|
| Belady's MIN (1966) | Offline-optimal eviction for hit/miss. FUTCache's oracle (Thm 5.2) is the novelty-analogue: best offline resolution-ε cache. |
| Sleator–Tarjan (1985): LRU is k-competitive | Thm 5.2: FUTCache is `P(K,ε)`-competitive, with `P(K,ε)` geometric rather than tuned. |
| Mattson's stack property (1970) | The tower's `q_{j+1,j}` projection is the spatial analogue of the LRU stack. |
| Denning's working set (1968) | `Nov_W(x, t) = 1[x ∉ V(H_{t−W:t})]` is the working-set predicate; safe TTL needs counts (existing `how.md` §5). |
| HR-Cache (Ferry et al. 2016–2024) | Thm 5.3: HR-ordering is suboptimal for novelty; the objectives are orthogonal. |
| Glider / Hawkeye / ALPS (learned caches) | These predict the future to make eviction choices. FUTCache defines the minimal state *without* future knowledge. Complementary, not competing. |

---

## 6. What this buys you

1. **A theorem, not an assertion.** The "FUTCache is orthogonal to
   LRU" claim in `how.md` becomes a provable result (Thm 5.3) with an
   explicit counterexample.

2. **A concrete competitive ratio.** `P(K, ε) = ε^{-D}` is a number the
   user can compute from their workload's geometry. For a paraphrase-heavy
   RAG corpus with `D_cache = 1.2` and `ε = 0.1`, the competitive ratio
   is `~16`, not the ambient `d = 768`.

3. **A design principle with teeth.** "Tune ε, not k" is no longer a
   slogan — it's Corollary 5.4, with the memory/accuracy/ratio all
   expressed in terms of `ε` and `D`.

4. **A bridge to the ML-cache literature.** Thm 5.3 gives a precise
   sense in which FUTCache complements (rather than competes with)
   Glider/Hawkeye/ALPS/HR-Cache: they optimize expected misses via
   prediction; FUTCache optimizes state via correctness. The two can be
   **combined**: use a learned predictor to choose `ε` per query, then
   let FUTCache handle the state.

---

## 7. Open questions

1. **Tightness of Thm 5.2.** The `P(K, ε)` competitive ratio is an upper
   bound. Is it tight? For `K = [0,1]` with an adversarial sequence that
   cycles through all `1/ε` ε-separated points, FUTCache faults on every
   query after the fill-up phase, and OPT (with the same budget) also
   faults on every query. The ratio is 1 in that case. Need a workload
   where OPT does strictly better than FUTCache by a factor of
   `P(K, ε)`. The marking argument suggests the bound is tight up to a
   constant, but a clean lower-bound construction is needed.

2. **W_1 eviction in the competitive analysis.** Thm 5.2 holds for *any*
   eviction policy. The W_1-optimal eviction (Sketch 02) improves the
   constant but doesn't change the ratio. Can a geometry-aware eviction
   (e.g., one that uses the box-dimension structure of `K`) improve the
   ratio below `P(K, ε)`? This is an open question.

3. **Non-stationary workloads.** The i.i.d. and renewal models assume
   stationarity. Real RAG workloads are non-stationary (topics drift).
   The persistent novelty framework (Sketch 01) is the natural
   extension: the fault rate becomes a function of the persistence
   diagram's drift over time.

4. **Combination with learned predictors.** Can a learned model
   (Glider-style) predict the per-query ε, and FUTCache use that to
   adapt its resolution? The competitive analysis would need to be
   extended to a **two-level** game: the predictor chooses ε_t, then
   FUTCache resolves the fault at ε_t. This is a genuine research
   direction.

5. **Empirical validation.** The theorems are clean but abstract. The
   empirical program (Section 8) is needed to confirm the constants and
   to identify the regimes where the theory matches practice.

---

## 8. Suggested experiments

1. **i.i.d. fault rate vs. ε.** Generate i.i.d. samples from a
   Gaussian-mixture distribution in [0,1]^d with known box dimension
   D. Run FUTCache (pack, W_1 eviction) at ε ∈ {0.05, 0.1, 0.2, 0.5}.
   Measure the empirical fault rate and verify it's O(ε) as predicted
   by Thm 5.1.

2. **Adversarial competitive ratio.** Construct an adversarial sequence
   that cycles through ε-separated points in [0,1]^d. Measure the fault
   ratio of FUTCache vs. a Belady-style oracle (which knows the
   sequence). Verify the ratio is ≤ P(K, ε) as predicted by Thm 5.2.

3. **HR-ordering vs. FUTCache.** Implement the two-topic workload from
   Thm 5.3's proof sketch. Compare fault rates of HR-ordering eviction
   vs. FUTCache (W_1 eviction). Verify the divergence predicted by
   Thm 5.3.

4. **Dimension vs. competitive ratio.** Fix ε = 0.1, vary the box
   dimension D of the workload (by embedding in higher-dimensional
   spaces with lower intrinsic dimension). Measure the empirical
   competitive ratio and verify it scales as ε^{-D}, not as
   ε^{-d_ambient}.

5. **Combination with a learned ε-predictor.** Train a small model to
   predict ε_t per query (e.g., from query length, embedding norm, or
   topic). Measure the fault rate of FUTCache with the predicted ε_t vs.
   a fixed ε. This is the empirical precursor to the two-level
   competitive analysis in Open Question 4.

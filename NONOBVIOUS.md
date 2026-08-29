# What is actually non-obvious in FUTCache

This document is a deliberately honest inventory of the ideas in this
project that are *not* immediately reproducible by a competent reader.
It is **not** a moat claim. Non-obvious and defensible are different
things; the last section makes that distinction explicit.

The core of FUTCache — a first-fit `ε`-separated representative set that
answers "is this point new?" — is **not** non-obvious. It is the classic
online greedy packing / Gonzalez k-center construction. At matched `ε` it
is bit-for-bit identical to a ~100-line k-center, and slower. That is the
honest baseline, and everything below exists *around* it, not instead of it.

---

## The thesis

> Novelty is the missing primitive between ontology and embeddings.

Ontologies give exact symbolic guardrails; embeddings give continuous
similarity. Neither says "have we already been to this region?" FUTCache's
claim is that **visited-set novelty** is a first-class primitive that
belongs between them. That framing is non-obvious and is the intellectual
spine of the project — even though the underlying packing is not.

---

## The non-obvious contributions

### 1. Future-equivalence is the right state abstraction

The observation that two histories with the same `ε`-ball union agree on
every future novelty query — and therefore the raw stream can be discarded
and only the covered region retained — is the cleanest idea in the project.

- **Why it is non-obvious:** Most caches keep *items* and worry about which
  to evict. FUTCache keeps a *quotient of histories* (the Myhill–Nerode
  future-equivalence class) and never stores the items at all.
- **Its limit:** It is a re-framing of known geometry (packing numbers,
  covering sets), not a new theorem. The lower-bound argument is correct
  but standard.

### 2. The one-sided guarantee

`lookup(x) = d(x, R) > ε` where `R ⊂ H` is a *subset* of the true history.
This can over-report novelty near net gaps but **can never call a genuinely
new point old**.

- **Why it is non-obvious:** Almost every approximate-membership structure
  (Bloom filters, LSH, nearest-neighbor caches) trades both directions of
  error. Building the structure so that the false-negative rate is
  *provably zero* is a real design choice with a proof, and it is what makes
  the cache safe to use as a "don't miss rare events" primitive.
- **Its limit:** The cost is extra memory relative to a symmetric
  approximator, and the guarantee is only as meaningful as the supplied
  metric.

### 3. Deterministic Voronoi quantization for CRDT merge

The CRDT engine maps points to anchor cells so that `q(x) = q(y) ⟹
d(x,y) ≤ ε`, then merges replicas with a join-semilattice. Replicas
converge without coordination, and convergence is idempotent.

- **Why it is non-obvious:** CRDTs usually merge *ordered* state (counters,
  registers). Mapping a *metric* visited-set into a commutative,
  associative, idempotent join is a non-trivial reduction. The empirical
  result that `k = ⌈log₂ W⌉` gossip converges in 2 rounds at W=256 —
  12–31× faster than full fan-in at equal coverage — is a genuine finding.
- **Its limit:** The merge itself is just set union over quantized cells.
  The conflict-resolution path (priority-based) is confirmed by the bench
  to be *dormant* under round-robin workloads — useful defensively, not the
  source of the speedup.

### 4. Persistent novelty as a filtration, not a threshold

`PersistentNovelty` / `persist_nd` store a single-linkage merge tree and
report novelty at *every* scale `t`, exposing a persistence diagram over
the whole filtration rather than one fixed `ε`.

- **Why it is non-obvious:** The single-knob cache answers one question at
  one resolution; the persistent version answers the *family* of questions
  and lets you evict by persistence (the lifetime of a feature), not by
  recency. That is a different abstraction and not present in k-center.
- **Its limit:** It is `O(n·d)` per operation and is not locked; it is a
  side module, not the workhorse.

### 5. Adversarial geometry: the E1 pulse attack and its mitigation

`docs/EXPLOIT.md` shows that a *deterministic* eviction rule (W1, evict the
closest pair) leaks enough geometry through the novelty oracle to be
exploited: an attacker probes the boundary, plants decoys, force-evicts a
load-bearing representative, and re-bills the same point as novel.

- **Why it is non-obvious:** Correctness and security are usually treated as
  separate. Here the cache is *provably correct* (one-sidedness holds) and
  yet the attacker controls *when* it fires. The resurgent-flagging
  mitigation (remember evicted points; flag "resurgent" instead of "novel")
  is a non-obvious defensive pattern for this class of structure.
- **Its limit:** The attack is latent on the default FIFO path (W1 is
  opt-in). The mitigation is validated on 4 synthetic workloads, not
  production traffic.

### 6. The negative results stated plainly

The project documents several places where the approach *does not* work:

- **MDL is not semantic safety.** Geometric compressibility does not certify
  semantic interchangeability (`docs/mdl-semantic-negative.md`).
- **The Bekko margin is negative.** No single cosine threshold cleanly
  separates topics; the cache works by insertion dynamics, not a clean
  margin.
- **KDD Cup '99: AUC ≈ 1-NN.** The edge is operational, not detection.
- **Embedding collapse is the embedder's geometry, not a cache leak.**

- **Why it is non-obvious:** Negative results are expensive to obtain and
  almost never survive a marketing pass. Their presence is a signal of
  intellectual honesty, and they are the most trustworthy material here.

### 7. The CDCL/SAT geometric-recurrence signal (unverified here)

The README claims metric clause geometry predicts CDCL backjump depth with
`p ≤ 0.007` over matched-permutation nulls, and that a greedy `ε`-net
compacts clause databases 91.8–98.1%. If those numbers survive independent
reproduction, this is the single most valuable claim in the project.

- **Why it is non-obvious:** "Proof search has exploitable geometric
  recurrence" is a real, falsifiable, non-obvious hypothesis with a
  statistical design (matched-pair permutations) behind it.
- **Its limit:** This review did not verify it. Treat as an open hypothesis,
  not a receipt.

---

## What is *not* non-obvious

- **First-fit packing / k-center.** Textbook.
- **"Fewer representatives than LSH at the same accuracy."** The 1D win is
  real but small; the 3D "9×/10×" claim was an `ε`-mismatch artifact and
  collapses to identity at matched `ε`.
- **VP-tree indexing.** Off-the-shelf.
- **Serialization, CRC32, sanitizers, 134 tests.** Table stakes, not edges.
- **"Zero false positives."** Only holds in the 1D interval-union regime;
  the d-D pack reports false positives by design (one-sidedness).

---

## Non-obvious is not a moat

Every item above is **symmetric**: a competent reader can reproduce it once
it is written down. That is the whole point of this file.

A moat requires an **asymmetry** — something a competitor cannot match
cheaply even with full knowledge. The candidates here, in descending order
of plausibility:

1. **Distribution** — being the default visited-set embedded in a platform
   or framework. Currently absent.
2. **Data/network effects** — a fleet that *shares* the geometric blackboard
   gets better with more workers. Real in principle, absent at this stage.
3. **Trust** — the only cheap moat a small project can build. The inflated
   comparative benchmarks actively erode it; rewriting them to the matched-ε
   truth is the highest-leverage fix available.

The ideas in this file are the *seed*. They become a moat only when one of
the asymmetries above forms around them.

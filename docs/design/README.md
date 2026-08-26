# FUTCache — General-Purpose Design Sketches (Phase 3)

Five design sketches that extend FUTCache from a single-resolution
semantic cache to a **general-purpose, multi-scale, provably optimal**
novelty engine. Each is independent; each composes with the others.

| # | File | Core idea | Status |
|---|---|---|---|
| 01 | [01-persistent-novelty.md](01-persistent-novelty.md) | **Prime-tagged** persistence diagram — novelty as a function of scale, with Selberg trace formula connection to eviction cycles | Design |
| 02 | [02-wasserstein-eviction.md](02-wasserstein-eviction.md) | Replace FIFO eviction with **W₁-optimal** (nearest-neighbor-minimum) eviction; quantify the one-sided gap | Design |
| 03 | [03-submodular-selection.md](03-submodular-selection.md) | Replace order-sensitive ε-separated insertion with **submodular coverage** maximization (1−1/e guarantee) | Design |
| 04 | [04-learned-metric.md](04-learned-metric.md) | Bridge to **arbitrary similarity** via distance-to-anchors embedding with bounded distortion | Design |
| 05 | [05-competitive-theorem.md](05-competitive-theorem.md) | **Competitive-ratio theorems** vs. LRU/HR-cache; "tune ε, not k" becomes a theorem | Design |

## How they compose

```
                    ┌─────────────────────────────┐
                    │   04: Learned Metric Layer   │  (any similarity → metric)
                    │   φ(x) = (d(x,a₁),...,a_m)  │
                    └──────────────┬──────────────┘
                                   │ feeds
                    ┌──────────────▼──────────────┐
                    │   03: Submodular Selection   │  (which reps to keep)
                    │   max coverage, 1−1/e       │
                    └──────────────┬──────────────┘
                                   │ evicts via
                    ┌──────────────▼──────────────┐
                    │   02: W₁-Optimal Eviction    │  (which rep to drop)
                    │   nearest-neighbor minimum   │
                    └──────────────┬──────────────┘
                                   │ observed over
                    ┌──────────────▼──────────────┐
                    │   01: Persistent Novelty     │  (novelty at all scales)
                    │   persistence diagram        │
                    └──────────────┬──────────────┘
                                   │ bounded by
                    ┌──────────────▼──────────────┐
                    │   05: Competitive Theorem    │  (tune ε, not k)
                    │   ratio = P(K,ε) = ε^−D     │
                    └─────────────────────────────┘
```

- **04** is the input layer: makes the framework work with any
  similarity, not just L2/L1/Linf/cosine/Poincaré.
- **03** is the selection layer: picks the best representatives, not
  just the first-seen.
- **02** is the eviction layer: drops the least load-bearing rep, not
  the oldest.
- **01** is the output layer: answers novelty at every scale, not just
  one.
- **05** is the guarantee layer: proves the whole system is
  geometrically optimal, not just heuristically reasonable.

## Shared invariants (preserved across all five)

1. **One-sidedness.** No false negatives: the cache never suppresses
   true novelty. (01 preserves it per-scale; 02/03 quantify the
   false-positive gap; 04 bounds it by distortion 2δ.)
2. **Bounded memory.** `|R| ≤ P(K, ε)` at every scale. (01: bounded by
   the packing number at the queried scale; 02/03: explicit budget `k`.)
3. **Determinism.** Tie-breaking is canonical (lexicographic on
   coordinates), so the state is a function of the input multiset, not
   the arrival order. (Required for the CRDT engine; 03 makes it
   explicit.)
4. **VP-tree query time.** All layers preserve O(log n) expected query
   time. (04 adds O(m) embedding cost, but the VP-tree query itself is
   unchanged.)

## Suggested build order

1. **02** (W₁ eviction) — smallest change, biggest immediate win.
   ~350 lines C, reuses the VP-tree.
2. **03** (submodular selection) — composes with 02; fixes order-
   sensitivity. ~600 lines C.
3. **01** (persistent novelty) — the conceptual unlock; requires new
   data structures (merge tree). ~1200 lines C.
4. **04** (learned metric) — the generality unlock; reuses the CRDT
   anchor machinery. ~400 lines C.
5. **05** (competitive theorem) — the paper; no new code, just the
   theorems + the empirical program.

Total: ~2600 lines of C across the five sketches, plus theorems and
tests. Each is independently shippable; the full stack is the
"general purpose" FUTCache.

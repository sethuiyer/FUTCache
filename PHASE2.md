# FUTCache Phase 2: Distributed Semantic Caching as a Geometric CRDT

> *"Don't replicate requests. Replicate semantic discoveries."*

This document formalizes the distributed evolution of FUTCache. The
single-node theorems of `formal.md` describe the **local** cache
dimension, the Myhill–Nerode lower bound, and the β = αD Heaps-law
factorization. Phase 2 establishes a complementary object: a
**distributed semantic cache** whose replicas converge without
coordination, whose state remains geometrically bounded, and whose
merge law comes from a deterministic Voronoi quotient rather than
from a packing heuristic.

The headline is:

$$\boxed{\text{A fixed geometric quotient of the metric space is a
state-based CRDT.}}$$

Concretely: every replica holds a subset of a deterministic
$\delta$-net. Replicas gossip subsets. Set union is commutative,
associative, and idempotent. Therefore the merge is
coordination-free. And the $\delta \leq \epsilon / 2$ choice guarantees
that the quotient refines metric similarity without ever introducing
false reuse.

The text is organized as follows.

1. The obstruction: why greedy packing is **not** a CRDT.
2. The construction: deterministic anchors, Voronoi partition,
   quantization map, per-cell semilattice.
3. The convergence theorem.
4. The memory theorem.
5. The three-engine architecture (interval, pack, crdt).
6. The gossip protocol.
7. The product structure for symmetric workloads.
8. Open questions and a sketch of `futcache/crdt.h`.

---

## 12.1 The obstruction: why distributed PackCache is not a CRDT

The single-node packing cache of §10.11.2 (and `pack.h`) is exact for
novelty on any metric space. Naïvely, one might hope that two
replicas running the same cache independently and then merging their
representative sets would converge to the same state. They will not.

### Definition 12.1 (greedy distributed merge)

Let $R_A, R_B \subseteq K$ be the representative sets held by
replicas $A$ and $B$ after each has observed some local history. The
**greedy distributed merge** is

$$R_A \sqcup R_B \;=\; \mathcal{P}^\star(R_A \cup R_B),$$

where $\mathcal{P}^\star$ denotes any maximal $\epsilon$-separated subset
of the union, computed greedily by repeated farthest-point insertion.

### Proposition 12.2 (greedy distributed merge is not a CRDT)

The merge $\sqcup$ of Definition 12.1 fails associativity on compact
metric spaces. There exist finite point sets $R_A, R_B, R_C \subseteq K$
and a threshold $\epsilon$ such that

$$(R_A \sqcup R_B) \sqcup R_C \;\neq\; R_A \sqcup (R_B \sqcup R_C).$$

#### Proof

Take $K = \mathbb{R}$ with $d(x, y) = |x - y|$ and $\epsilon = 1$.
Let

$$R_A = \{0\},\qquad R_B = \{0.9\},\qquad R_C = \{1.8\}.$$

Each set is $\{0\}$-separated within itself. Now

- $R_A \sqcup R_B$: $d(0, 0.9) = 0.9 < 1$, so the greedy merge returns
  $\{0\}$.
- $R_B \sqcup R_C$: $d(0.9, 1.8) = 0.9 < 1$, so the greedy merge returns
  $\{0.9\}$.
- $R_A \sqcup (R_B \sqcup R_C) = \{0\} \sqcup \{0.9\} = \{0\}$.

Versus

- $R_A \sqcup R_B = \{0\}$ as above.
- $(R_A \sqcup R_B) \sqcup R_C = \{0\} \sqcup \{1.8\}$: $d(0, 1.8) = 1.8 > 1$,
  greedy farthest-point accepts $1.8$, returns $\{0, 1.8\}$.
- $R_A \sqcup (R_B \sqcup R_C) = \{0\} \sqcup \{0.9\} = \{0\}$.

So $\{0, 1.8\} \neq \{0\}$. $\square$

### Corollary 12.3 (greedy merge is not even commutative in general)

There exist $R_A, R_B$ for which $R_A \sqcup R_B \neq R_B \sqcup R_A$.

#### Proof

Same construction as Proposition 12.2 but with the observation order
swapped. The greedy farthest-point heuristic breaks ties by index, so
$R_A$ processing $\{0, 0.9, 1.8\}$ in that order yields $\{0, 1.8\}$,
whereas processing $\{1.8, 0.9, 0\}$ yields $\{1.8, 0\}$. $\square$

### Remark 12.4 (the deeper reason)

The obstruction is not the greedy heuristic. It is that
**metric similarity is not transitive.** The relation

$$x \sim_\epsilon y \;\iff\; d(x, y) \leq \epsilon$$

fails the transitivity axiom of equivalence relations: there are
$x, y, z$ with $x \sim_\epsilon y$ and $y \sim_\epsilon z$ but
$x \not\sim_\epsilon z$. A CRDT state space requires a join-semilattice,
which in turn requires a partial order, which in turn requires
reflexivity and transitivity on the underlying equality.

> **Lesson.** Distributed FUTCache cannot replicate the *packing*
> state directly. It must instead replicate a quotient state whose
> equality is genuinely transitive.

---

## 12.2 The construction

We replace the order-dependent greedy packing by a **fixed
geometric quotient** of the metric space. The quotient is the
Voronoi partition induced by a deterministic $\delta$-net, with
$\delta$ small enough that two points in the same Voronoi cell are
guaranteed to be within $\epsilon$ of each other.

### Definition 12.5 (deterministic $\delta$-net)

A **deterministic $\delta$-net** of a metric space $(K, d)$ is a
finite set $A = \{a_1, \dots, a_m\} \subseteq K$ such that:

1. **$\delta$-separated:** for all $i \neq j$, $d(a_i, a_j) > \delta$.
2. **$\delta$-covering:** for every $x \in K$, there exists $i$ with
   $d(x, a_i) \leq \delta$.
3. **Deterministic:** $A$ is a function of $(K, d, \delta)$ alone,
   not of the construction history.

For compact $(K, d)$ such a set exists for every $\delta > 0$, and its
cardinality $m = m(\delta)$ is bounded by the packing number
$P(K, \delta)$.

### Definition 12.6 (Voronoi partition)

Given a deterministic $\delta$-net $A$, the **Voronoi partition**
$\{V_i\}_{i = 1}^m$ is the disjoint covering of $K$ defined by

$$V_i \;=\; \bigl\{\, x \in K : d(x, a_i) \leq d(x, a_j)\ \forall j \neq i \,\bigr\},$$

with deterministic tie-breaking when $x$ lies on a boundary
(e.g., smallest index).

### Definition 12.7 (quantization map)

The **quantization map** is

$$q(x) \;=\; \arg\min_{i}\; d(x, a_i).$$

It assigns every $x \in K$ to a unique cell index $i \in \{1, \dots, m\}$.

### Definition 12.8 (cell equivalence)

Define the relation $\equiv_q \subseteq K \times K$ by

$$x \equiv_q y \;\iff\; q(x) = q(y).$$

This is the cell-membership relation. By construction,
$\equiv_q$ is reflexive, symmetric, and transitive. It is an
**equivalence relation on $K$**.

### Definition 12.9 (resolution requirement)

We say the construction is **$\epsilon$-safe** if

$$\delta \;\leq\; \frac{\epsilon}{2}.$$

This is the condition that makes the cell-membership equivalence refine
metric similarity.

### Theorem 12.10 (Voronoi Semantic CRDT Convergence)

Let $(K, d)$ be a metric space, $A = \{a_1, \dots, a_m\}$ a
deterministic $\delta$-net with $\delta \leq \epsilon / 2$, $\{V_i\}$
its Voronoi partition, and $q$ the quantization map. Then:

1. **$\epsilon$-refinement:** $q(x) = q(y) \implies d(x, y) \leq \epsilon$.
2. **One-sided safety:** the converse does not hold in general; if
   $q(x) \neq q(y)$, it is possible that $d(x, y) \leq \epsilon$ (extra
   misses) but never that $d(x, y) > \epsilon$.
3. **Bounded state:** $|A| \leq P(K, \epsilon / 2)$.

#### Proof

**Part 1.** Suppose $q(x) = q(y) = i$. By Definition 12.7, $a_i$ is
the nearest anchor to both $x$ and $y$. By Definition 12.5,
$d(x, a_i) \leq \delta$ and $d(y, a_i) \leq \delta$. By the triangle
inequality,

$$d(x, y) \;\leq\; d(x, a_i) + d(a_i, y) \;\leq\; 2\delta \;\leq\; \epsilon.$$

**Part 2.** The "extra misses" claim is straightforward. If
$q(x) = j \neq i = q(y)$, both $x$ and $y$ are within $\delta$ of their
respective anchors. If $a_j$ happens to lie within $\epsilon$ of $a_i$
across the boundary, points near that boundary can have
$d(x, y) \leq \epsilon$ despite $q(x) \neq q(y)$. The "never extra
reuse" direction follows from Part 1 applied contrapositively:
$d(x, y) > \epsilon \implies q(x) \neq q(y)$.

**Part 3.** The anchors form a $\delta$-packing (Definition 12.5(1))
with $\delta = \epsilon / 2$. By definition of the packing number,
the cardinality of any $\delta$-separated subset of $K$ is at most
$P(K, \delta) = P(K, \epsilon / 2)$. $\square$

### Remark 12.11 (interpretation)

The construction trades **recall** for **algebra**. Two close
points that happen to lie on opposite sides of a Voronoi boundary
are reported as distinct. The cache becomes more conservative than
the metric predicate alone. But the algebra is exact: the cell
identity relation is genuinely transitive, and replicas that
exchange occupied cells converge to the same state regardless of
network timing.

This is precisely the one-sided error profile we want for a
semantic cache: never merge unrelated queries, occasionally
re-evaluate queries that are similar but happened to land in
different cells.

---

## 12.3 The state space and CRDT laws

The deterministic $\delta$-net and the per-cell join structure
together give the distributed cache state.

### Definition 12.12 (per-cell state)

For each cell $i$, let $\mathcal{E}_i$ denote the set of admissible
entries for that cell. An entry is a tuple $(r, p, \pi)$ where:

- $r \in V_i$ is a representative point that triggered the cell's
  discovery;
- $p$ is an opaque payload (e.g., an LLM response or retrieval
  result);
- $\pi$ is a deterministic priority used to break ties.

Admissible states are $E_i = \{\bot\} \cup \mathcal{E}_i$, where $\bot$
denotes "cell not yet observed."

### Definition 12.13 (per-cell priority order)

Give each candidate entry a globally deterministic priority via a
fixed total order. A canonical choice is

$$\pi(r, p) \;=\; \mathcal{H}\!\bigl(\, r \,\|\, p \,\bigr)$$

where $\mathcal{H}$ is a cryptographic hash and $\|$ is concatenation;
any deterministic function $\pi: K \times \text{payload} \to \mathbb{N}$
with a strict total order suffices.

### Definition 12.14 (per-cell join)

For two entries $e = (r, p, \pi)$ and $e' = (r', p', \pi')$ both
admissible for cell $i$, define the per-cell join

$$e \sqcup_i e' \;=\;
\begin{cases}
e & \text{if } \pi > \pi', \\
e' & \text{if } \pi < \pi', \\
e & \text{if } \pi = \pi' \text{ and } e = e'.
\end{cases}$$

The third case is the tie-breaker; in practice cryptographic hashes
make ties negligible.

### Proposition 12.15 (per-cell join-semilattice)

For each cell $i$, $(E_i, \sqcup_i)$ is a join-semilattice. That is,
$\sqcup_i$ is commutative, associative, idempotent, and $\bot$ is its
least element.

#### Proof

Restrict attention to entries with priorities comparable under the
fixed total order on $\pi$. On this subset, $\sqcup_i$ reduces to
$\max$ on $\pi$, with payload chosen by the higher-priority entry.
Maximum over a total order is commutative, associative, and
idempotent. The element $\bot$ is the identity (least element) since
$\bot \sqcup_i e = e$ for any $e$. The full $E_i$ including ties
inherits these properties by the tie-breaking rule. $\square$

### Definition 12.16 (distributed cache state)

The distributed cache state is the product

$$\mathcal{S} \;=\; \prod_{i = 1}^{m} E_i.$$

A state $S \in \mathcal{S}$ assigns an entry (or $\bot$) to each cell.

### Definition 12.17 (state merge)

For two states $S, T \in \mathcal{S}$, define the **pointwise merge**

$$(S \sqcup T)_i \;=\; S_i \sqcup_i T_i.$$

### Theorem 12.18 (state join-semilattice)

$(\mathcal{S}, \sqcup)$ is a join-semilattice. The bottom element is
$\bot_{\mathcal{S}} = (\bot, \dots, \bot)$.

#### Proof

A product of join-semilattices is a join-semilattice: commutativity,
associativity, and idempotence hold pointwise, hence globally. The
bottom is the product of bottoms. $\square$

### Corollary 12.19 (state-based CRDT)

The pair $(\mathcal{S}, \sqcup)$ is a **CvRDT** in the sense of
Shapiro et al.: a join-semilattice state space with a commutative,
associative, idempotent merge.

#### Proof

Theorem 12.18. $\square$

### Corollary 12.20 (gossip delivery semantics)

Replicas that exchange states via gossip converge to a common
limit $\bigsqcup_r S_r$ regardless of network reordering, message
duplication, or transient disconnection. The limit is the join of
the replicas' current states at the moment of observation.

#### Proof

The merge is commutative, associative, and idempotent (Theorem
12.18). Any delivery order yields the same join. Idempotence absorbs
duplicates. Disconnection simply delays delivery; once messages are
finally exchanged, the join is unchanged. $\square$

---

## 12.4 The memory theorem

The deterministic anchor set bounds both the per-replica state size
and the network bandwidth per gossip round.

### Theorem 12.21 (state size bound)

For a deterministic $\epsilon$-safe construction on a compact
metric space $(K, d)$,

$$|\mathcal{S}|_{\max} \;=\; m \;=\; |A| \;\leq\; P\!\left(K, \frac{\epsilon}{2}\right).$$

In particular, the distributed cache state contains at most
$P(K, \epsilon / 2)$ non-$\bot$ cells.

#### Proof

The non-$\bot$ cells form a subset of the anchor set $A$ (a state can
have non-$\bot$ only at indices in $\{1, \dots, m\}$). The anchor set
is $\delta$-separated with $\delta = \epsilon / 2$, hence
$|A| \leq P(K, \delta)$ by definition of the packing number. $\square$

### Corollary 12.22 (network bandwidth)

Each gossip message carries at most $P(K, \epsilon / 2)$ dirty-cell
updates plus per-cell entry metadata. The amortized per-message size
is independent of the request stream length.

### Theorem 12.23 (geometric scaling)

Suppose $(K, d)$ has box-counting dimension $D$. Then

$$|\mathcal{S}|_{\max} \;=\; P(K, \epsilon / 2) \;=\; \left(\frac{2}{\epsilon}\right)^{D + o(1)}.$$

Equivalently,

$$\log |\mathcal{S}|_{\max} \;=\; D \cdot \log(1 / \epsilon) + O(D) + o(\log(1 / \epsilon)).$$

The asymptotic cache dimension is therefore

$$\limsup_{\epsilon \to 0} \frac{\log |\mathcal{S}|_{\max}}{\log(1 / \epsilon)} \;=\; D.$$

#### Proof

The packing number of a metric space of box dimension $D$ scales as
$P(K, \epsilon) = (1/\epsilon)^{D + o(1)}$. Substituting
$\epsilon \mapsto \epsilon / 2$ and expanding,

$$P(K, \epsilon / 2) = (2/\epsilon)^{D + o(1)} = 2^{D + o(1)} \cdot \epsilon^{-(D + o(1))}.$$

Taking $\log$,

$$\log P(K, \epsilon / 2) = (D + o(1)) \log(2/\epsilon).$$

The $o(1)$ term, when multiplied by $\log(2/\epsilon)$, becomes
$o(\log(1/\epsilon))$ in the standard asymptotic convention, which
is dominated by $D \log(1/\epsilon)$. The factor $2^D$ enters the
$O(D)$ term, not the scaling exponent. $\square$

### Remark 12.24 (the price of being a CRDT)

Comparing with `formal.md` §10.11 Theorem 10.16, the single-node
packing cache attains $|\mathcal{S}| = P(K, \epsilon)$ exactly, while
the distributed quotient attains $|\mathcal{S}| \leq P(K, \epsilon / 2)$.
The factor $2^D$ is the asymptotic cost of converting a fuzzy metric
relation into a transitive equivalence. The **exponent** $D$ is
unchanged: both constructions have the same cache dimension.

This is the central asymptotic message: **distributing FUTCache
preserves the geometric scaling law.** We pay a constant-factor
boundary fragmentation cost, not a dimensional one.

---

## 12.5 The three-engine architecture

The two constructions so far — exact interval-union (§10.11) and
greedy packing (§10.11.2) — were local. The Voronoi semilattice of
§12.2–§12.4 completes a three-way taxonomy that resolves an
otherwise forced trade-off.

### Proposition 12.25 (the three engines)

| Engine | File | Quotient | Merge | Suitable for |
|---|---|---|---|---|
| **Interval** | `futcache.h` | $H / {\sim_{\text{fut}}}$ (exact on $\mathbb{R}$) | $\cup$ of disjoint intervals | 1-D exact novelty, sensor fusion, financial ticks |
| **Pack** | `pack.h` | $H / {\sim_{\epsilon}}^{\text{greedy}}$ | repacking on union (single node) | single-node RAG, embeddings, semantic dedup |
| **CRDT** | `crdt.h` (Phase 2) | $H / {\equiv_q}$ (deterministic Voronoi) | $\sqcup$ on cell entries | distributed multi-region, gossip-replicated state |

#### Proof of completeness

The interval engine is the special case $D = 1$, $L_\infty$ metric,
exact quotient. The pack engine is the local (single-node, no merge
law needed) version of the same principle. The CRDT engine is the
distributed version of the same principle on arbitrary finite-d
metric spaces. The three engines are not three independent designs;
they are three projections of one construction at different points in
the design space (1-D vs. d-D, exact vs. adaptive, local vs.
distributed). $\square$

### Remark 12.26 (what each engine maximizes)

- **Interval** maximizes **exactness**: zero false-positive reuse.
- **Pack** maximizes **adaptive recall**: maximal reuse on whatever
  the input distribution happens to be.
- **CRDT** maximizes **composability**: mergeability under any
  delivery schedule.

The CRDT engine is strictly less recall-efficient than Pack on any
single workload, because the deterministic anchor set cannot
adapt to observed input density. But it is the only one of the
three that is safe to gossip.

---

## 12.6 The gossip protocol

The CRDT engine ships with a reference gossip protocol. The
protocol is intentionally minimal: it assumes neither reliable
delivery nor total order, and does not require a leader or a
consensus algorithm.

### Algorithm 12.27 (cell ingress)

On a node observing a query $x$:

```
1.  i = q(x)                              # quantize to anchor cell
2.  if cell_i is occupied locally:
3.      return entries[i].payload         # semantic HIT, no RPC
4.  else:
5.      response = call_llm(x)            # miss path
6.      π = hash(x || response)            # deterministic priority
7.      entries[i] = (x, response, π)
8.      mark dirty[i]                     # flag for gossip
```

### Algorithm 12.28 (gossip merge)

On receipt of a remote batch of dirty cells $B$:

```
1.  for each (i, entry) in B:
2.      if cell_i is not occupied locally:
3.          entries[i] = entry             # adopt remote discovery
4.      else:
5.          if entry.π > local.entries[i].π:
6.              entries[i] = entry         # higher-priority wins
7.          else:
8.              # keep local entry; lower-priority loses
9.      mark dirty[i] = false              # we have the latest
```

### Theorem 12.29 (gossip convergence)

Two replicas that exchange gossip messages for arbitrarily long,
under any combination of message delay, loss, duplication, and
reordering, eventually reach a common state equal to the join of
their initial states.

#### Proof

By Corollary 12.20. Each gossip round is a state join. The merge is
idempotent, so repeated delivery of the same update is a no-op.
Commutativity and associativity make delivery order irrelevant.
Loss simply delays convergence; once any message is delivered, it is
absorbed into the join and persists across subsequent merges. $\square$

### Theorem 12.30 (gossip termination)

If gossip runs at fixed intervals and the dirty set is bounded, each
node converges within $O(\log N)$ rounds of accepting $N$ new cells.

#### Proof

A dirty cell is cleared once a gossip round delivers it to a peer
that already has it, or that adopts it. With $N$ total cells and a
fixed gossip interval, the dirty set size per round is bounded by
the new observations per interval. Standard epidemic-gossip bounds
apply. $\square$

### Remark 12.31 (network assumption)

The protocol assumes **eventual delivery**, not reliable delivery.
In a real deployment this is achieved by anti-entropy: every
$\Delta$ seconds, each node exchanges its full dirty set with a
random peer. Crashed nodes that come back online re-gossip their
state and are absorbed into the cluster join. The algebra is
sufficient for correctness; the network only needs to deliver
messages eventually.

---

## 12.7 Product with cyclic workflows

For workloads with cyclic symmetry — e.g., a periodic agent whose
state space is the orbit of a cyclic group action — the semantic
quotient of §12.2 can be composed with the orbit quotient.

### Definition 12.32 (cyclic workflow)

Let $C_n$ act on $K$ by isometries $\rho_0, \rho_1, \dots, \rho_{n - 1}$.
Two queries $x, y$ are **trace-equivalent** if $y = \rho_k(x)$ for some
$k$. The trace quotient is $K / C_n$.

### Theorem 12.33 (product quotient)

The combined state space is the product of two semilattices:

$$\mathcal{S}_{\text{global}} \;=\; \mathcal{S}_{\text{semantic}} \;\times\; \mathcal{S}_{\text{trace}}.$$

The product inherits the join-semilattice structure pointwise. The
total state size is bounded by

$$|\mathcal{S}_{\text{global}}|_{\max} \;\leq\; P\!\left(K, \frac{\epsilon}{2}\right) \cdot n.$$

#### Proof

Direct product of semilattices is a semilattice. The semantic
component contributes its own anchor set; the trace component
contributes one slot per orbit representative. Independence of the
two constructions gives the additive bound. $\square$

### Remark 12.34 (deadlock: the open question)

The semantics of merging a semantic HIT with a trace shift is
non-trivial. If cell $i$ in semantic space contains the LLM response
for query $x$, and a trace shift moves the same query to query $\rho_k
(x)$, the cell identifier is unchanged but the canonical priority
of the merged representative may differ. The conservative resolution
is to keep the higher-priority entry per cell; this is what the
algebra gives for free. A more aggressive resolution could carry
both payloads and let the application pick at query time. This is an
open design choice.

---

## 12.8 Open questions and the `crdt.h` skeleton

The construction in §12.2–§12.4 is mathematically complete. Three
implementation questions remain.

### Open question 12.35 (anchor construction)

How should the deterministic $\delta$-net $A$ be computed in
production? Options:
- **Static net:** sample $K$ offline, hash the result to get a
  reproducible anchor set. Sufficient for bounded domains.
- **Self-organizing net:** run a packing algorithm on a calibration
  sample and freeze its output as the global anchor set.
- **Hierarchical net:** combine multiple scales, à la the tower
  cache of §10.11. Each scale contributes a level of the partition.

The hierarchical option composes with the existing tower code in
`tower.h`. Each tower level becomes a refinement of the partition.

### Open question 12.36 (quantization complexity)

$q(x)$ is currently defined as a linear scan over $A$ for $|A| = m$
queries. For $m = P(K, \epsilon / 2)$ this is acceptable for small
metric spaces and small $\epsilon$. For high-d embedding spaces, the
anchor count grows as $2^D \epsilon^{-D}$ and linear scan becomes
prohibitive. Practical quantization uses:
- k-d trees or cover trees for $D \leq 20$;
- HNSW or IVF-PQ for high-d;
- learned hashes for the embedding-specific case.

The API should hide the quantization backend behind a single
`futcache_crdt_quantize(x)` entry point so that backends can be
swapped.

### Open question 12.37 (payload handling)

The reference protocol (Algorithm 12.28) ships the entire entry
$(r, p, \pi)$ on gossip. For large payloads (LLM responses can be
kilobytes) this is wasteful. Three options:
- **Pointer-only:** gossip only the anchor index and the priority;
  fetch payloads on demand via a side channel.
- **Blob storage:** gossip only changed blobs, identified by hash.
- **Tiered:** gossip only cell *summaries*; gossip full payloads
  lazily.

The choice depends on workload. The algebra is the same in all
cases.

### Skeleton: `include/futcache/crdt.h`

The intended public API:

```c
typedef struct futcache_crdt_config {
    size_t dimension;
    size_t anchor_count;            /* |A| */
    const double *anchors;         /* [anchor_count * dimension] */
    double epsilon;
    futcache_distance_fn distance;
    void *distance_context;
    const double *domain_min;
    const double *domain_max;
    futcache_allocator_t allocator;
} futcache_crdt_config_t;

typedef struct futcache_crdt_t futcache_crdt_t;

FUTCACHE_API futcache_status_t futcache_crdt_create(
    const futcache_crdt_config_t *config,
    futcache_crdt_t **out_cache);

FUTCACHE_API void futcache_crdt_destroy(futcache_crdt_t *cache);

/* Quantize a point to its anchor cell index. */
FUTCACHE_API futcache_status_t futcache_crdt_quantize(
    const futcache_crdt_t *cache,
    const double *point,
    size_t *out_cell);

/* Atomic test-and-set on a cell. Returns whether the cell was
 * already occupied, and the existing or new entry's payload. */
FUTCACHE_API futcache_status_t futcache_crdt_observe(
    futcache_crdt_t *cache,
    const double *point,
    const void *payload,
    size_t payload_length,
    bool *out_was_novel,
    size_t *out_cell);

/* Gossip: merge a remote batch of (cell, payload, priority). */
FUTCACHE_API futcache_status_t futcache_crdt_merge(
    futcache_crdt_t *cache,
    const futcache_crdt_update_t *updates,
    size_t update_count);

/* Snapshot for outgoing gossip. */
FUTCACHE_API futcache_status_t futcache_crdt_snapshot(
    const futcache_crdt_t *cache,
    futcache_crdt_update_t *out_updates,
    size_t *inout_count);

FUTCACHE_API futcache_status_t futcache_crdt_validate(
    const futcache_crdt_t *cache);
```

The three engines share a common algebraic backbone:

$$\boxed{
\begin{aligned}
&\textbf{Interval: } H / {\sim_{\text{fut}}} \text{ over } \mathbb{R}, \text{ merged by } \cup. \\
&\textbf{Pack: } H / {\sim_{\epsilon}}^{\text{greedy}} \text{ over any } (K, d), \text{ local only}. \\
&\textbf{CRDT: } H / {\equiv_q} \text{ over any } (K, d), \text{ merged by } \sqcup.
\end{aligned}}$$

Three quotients, three merge laws, one principle:
**the future-equivalence state is the smallest sufficient state for
the chosen observable.**

---

### Epilogue

> "Distributed systems engineers ship CRDTs. Database engineers ship
> consensus protocols. FUTCache ships a metric space and lets the
> geometry do the rest."

The interval-union cache of v1.0 was an exact novelty oracle.
The packing cache of v1.1 was a Voronoi seed set on arbitrary
metrics. Phase 2 turns that seed set into a fixed geometric quotient
that composes across replicas without coordination.

The headline remains:

$$\boxed{\text{Specify fidelity } \epsilon. \text{ Distributed state follows geometry.}}$$

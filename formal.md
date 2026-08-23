Yep. This wants to read like a theorem section, not a systems pitch with equations sprinkled on top. 🔥

## 10.11 Cache Complexity: Resolution Before Capacity

Classical capacity-bounded caching begins with an exogenous memory budget (k) and asks which distinctions should be discarded when that budget is exceeded. The construction developed above reverses this order. One first specifies the observational resolution at which distinctions must be preserved; the geometry of the distinguishable state space then determines the memory requirement.

[
\boxed{\text{LRU: choose memory, lose information accordingly.}}
]

[
\boxed{\text{FUTCache: choose information resolution, memory follows geometrically.}}
]

This inversion turns cache capacity from a tuning parameter into a derived geometric quantity.

### Definition 10.15 (Resolution complexity)

Let ((K,d)) be a compact metric space, let

[
\epsilon_j=2^{-j},
]

and let (P(K,\epsilon_j)) denote the packing number of (K) at resolution (\epsilon_j). For a traversal

[
L=(x_0,x_1,\ldots),
]

let (D_j(L)) be its first-discovery word at level (j), and define

[
M_j(L):=|D_j(L)|.
]

Thus (M_j(L)) is the number of distinguishable cells discovered by (L) at resolution (2^{-j}).

Define the **cache dimension** of (L) relative to the tower (P) by

[
\boxed{
D_{\mathrm{cache}}(L;P)
:=
\limsup_{j\to\infty}
\frac{\log M_j(L)}
{j\log 2}.
}
]

This quantity measures the asymptotic exponent governing the growth of future-relevant distinctions as observational resolution increases.

---

### Theorem 10.16 (Geometric cache-growth law)

Suppose the box dimension of (K) exists and equals (D):

[
D
=

\lim_{j\to\infty}
\frac{\log P(K,2^{-j})}
{j\log 2}.
]

Then

[
\boxed{
P(K,2^{-j})
===========

2^{jD+o(j)}.
}
]

Consequently, every traversal satisfies

[
M_j(L)
\le
2^{jD+o(j)}.
]

If, moreover, (L) discovers a nonvanishing fraction of the available level-(j) cells, i.e. there exists (c>0) such that for all sufficiently large (j),

[
M_j(L)\ge c,P(K,2^{-j}),
]

then

[
\boxed{
M_j(L)=2^{jD+o(j)}
}
]

and hence

[
\boxed{
D_{\mathrm{cache}}(L;P)=D.
}
]

#### Proof

By definition of box dimension,

[
\frac{\log P(K,2^{-j})}{j\log2}
===============================

D+o(1).
]

Multiplying by (j\log2),

[
\log P(K,2^{-j})
================

jD\log2+o(j),
]

and exponentiating gives

[
P(K,2^{-j})
===========

2^{jD+o(j)}.
]

Since a traversal cannot discover more level-(j) cells than exist,

[
M_j(L)\le P(K,2^{-j}),
]

which proves the upper bound.

Under the nonvanishing-coverage hypothesis,

[
cP(K,2^{-j})
\le M_j(L)
\le P(K,2^{-j}).
]

Taking logarithms,

[
\log c+\log P(K,2^{-j})
\le
\log M_j(L)
\le
\log P(K,2^{-j}).
]

Because (\log c=O(1)=o(j)),

[
\log M_j(L)
===========

jD\log2+o(j),
]

and therefore

[
M_j(L)=2^{jD+o(j)}.
\qquad\square
]

The exponent is forced by geometry. No fitted power law is required. Box dimension determines the exponent; stronger regularity assumptions such as Minkowski measurability or suitable Ahlfors regularity are required only if one wishes to strengthen this statement to an asymptotic law with a fixed multiplicative constant.

---

### Corollary 10.17 (Cost of one additional bit of resolution)

Under the hypotheses of Theorem 10.16,

[
\log_2 M_j
==========

jD+o(j).
]

Hence increasing the resolution depth from (j) to (j+1), corresponding to

[
\epsilon_j\longmapsto\frac{\epsilon_j}{2},
]

has asymptotic memory multiplier

[
\boxed{
\frac{M_{j+1}}{M_j}
\approx 2^D
}
]

whenever the lower-order fluctuations are sufficiently regular.

Thus (D) is the marginal exponent relating fidelity to memory. In a (D)-dimensional distinguishability geometry, each additional bit of linear resolution costs approximately a factor (2^D) in cache state.

This yields the design principle

[
\boxed{\text{Specify fidelity, not capacity. The space tells you the cache size.}}
]

---

### Example 10.18 (Reciprocal traversal)

Consider

[
K=
\left{0,1,\frac12,\frac13,\ldots\right}
\subset[0,1],
]

with traversal

[
L=\left(1,\frac12,\frac13,\ldots\right).
]

At dyadic resolution (N=2^j), the occupied cells are determined, up to endpoint conventions, by the distinct values

[
\left\lfloor\frac{N}{n}\right\rfloor.
]

Let (M_j) denote their number.

For (n\le\sqrt N), there are at most (\sqrt N) possible contributions. For (n>\sqrt N),

[
\left\lfloor\frac{N}{n}\right\rfloor
<
\sqrt N,
]

so there are at most another (\sqrt N+O(1)) possible values. Hence

[
M_j=O(\sqrt N).
]

Conversely, all integers (q) in a range of order (\sqrt N) occur as values of (\lfloor N/n\rfloor), yielding

[
M_j=\Omega(\sqrt N).
]

Therefore

[
M_j=\Theta(\sqrt N)
===================

\Theta(2^{j/2}),
]

and

[
\boxed{
D_{\mathrm{cache}}=\frac12.
}
]

This example separates three notions of dimension:

[
\boxed{
\dim_{\mathrm{ambient}}=1,
\qquad
\dim_H K=0,
\qquad
D_{\mathrm{cache}}=\frac12.
}
]

The exact point set is countably infinite, so an exact cache that stores every discovered point grows without bound. At resolution (2^{-j}), however, the number of future-relevant distinctions grows only as

[
M_j=\Theta(2^{j/2}).
]

Consequently, doubling linear resolution asymptotically multiplies the required number of distinguishable states by

[
2^{1/2}=\sqrt2.
]

The effective memory complexity is therefore neither the ambient dimension nor the Hausdorff dimension of the underlying countable set. It is the dimension induced by the interaction between the traversal and the observational tower.

---

### Definition 10.19 (Future equivalence)

Let (\mathcal H) denote the set of finite histories and let

[
Out(H;w)
]

denote the observable output produced after history (H) under a finite continuation (w). Define

[
H\sim_{\mathrm{fut}}H'
\iff
\forall w,\qquad
Out(H;w)=Out(H';w).
]

The quotient

[
\boxed{
M_{\mathrm{fut}}
================

\mathcal H/!\sim_{\mathrm{fut}}
}
]

is the canonical future-sufficient state space for the observable (Out).

This separates two questions that are conflated in ordinary cache design:

[
\boxed{
\text{What information must survive?}
}
]

and

[
\boxed{
\text{How should that information be represented?}
}
]

FUTCache addresses the first question before applying an implementation-specific representation to the resulting quotient.

---

### Theorem 10.20 (FUT lower bound)

Let (A) be any deterministic online algorithm that exactly computes (Out) for every finite history and every continuation. If (A) has (S) possible internal states, then

[
\boxed{
S
\ge
|\mathcal H/!\sim_{\mathrm{fut}}|.
}
]

Equivalently, any exact implementation using (b) bits of memory must satisfy

[
\boxed{
b
\ge
\left\lceil
\log_2
|\mathcal H/!\sim_{\mathrm{fut}}|
\right\rceil.
}
]

#### Proof

Suppose, toward contradiction, that two histories

[
H\not\sim_{\mathrm{fut}}H'
]

lead (A) to the same internal state.

Since the histories are not future-equivalent, there exists a continuation (w) for which

[
Out(H;w)\ne Out(H';w).
]

But after (H) and (H'), algorithm (A) occupies the same internal state. Being deterministic, when subsequently presented with the same continuation (w), it must undergo the same sequence of state transitions and produce the same outputs.

This contradicts

[
Out(H;w)\ne Out(H';w).
]

Hence distinct future-equivalence classes require distinct machine states, proving

[
S\ge|\mathcal H/!\sim_{\mathrm{fut}}|.
\qquad\square
]

This is the cache analogue of the Myhill--Nerode distinguishability argument and of the distinguishing-set technique used in streaming lower bounds.

---

### Corollary 10.21 (Optimality for exact novelty)

Let (K) be a finite alphabet with

[
|K|=m,
]

and consider exact novelty

[
Nov(x\mid H)
============

\mathbf 1[x\notin V(H)].
]

Then

[
H\sim_{\mathrm{fut}}H'
\iff
V(H)=V(H').
]

Every subset of (K) is realizable as (V(H)). Therefore

[
|\mathcal H/!\sim_{\mathrm{fut}}|
=================================

2^m.
]

By Theorem 10.20,

[
b\ge\log_2 2^m=m.
]

Hence every deterministic exact novelty cache requires at least

[
\boxed{m\text{ bits}}
]

in the worst case.

A membership bitset uses exactly (m) bits and therefore attains the lower bound:

[
\boxed{
b_{\min}=m.
}
]

The bitset is consequently not merely a convenient representation of exact novelty. It is information-theoretically optimal among deterministic exact implementations.

Auxiliary structures such as Fenwick trees may support rank and select queries efficiently, but they augment the representation for additional observables; they do not reduce the (m)-bit lower bound required for exact membership novelty itself.

---

### Theorem 10.22 (Minimal state for metric novelty)

Fix (\epsilon>0) and define

[
U_\epsilon(H)
:=
\bigcup_{x\in V(H)}
\overline B(x,\epsilon).
]

Metric novelty is

[
Nov_\epsilon(y\mid H)
=====================

\mathbf1[
d(y,V(H))>\epsilon
]
=

\mathbf1[
y\notin U_\epsilon(H)
].
]

Assume the admissible query domain separates unequal coverage sets. Then

[
\boxed{
H\sim_{\mathrm{fut}}H'
\iff
U_\epsilon(H)=U_\epsilon(H').
}
]

Thus the future-sufficient state for exact metric novelty is the induced coverage set:

[
\boxed{
M_{\mathrm{fut}}(H)
\cong
U_\epsilon(H).
}
]

#### Proof

If

[
U_\epsilon(H)=U_\epsilon(H'),
]

then every future query (y) has identical novelty value under the two histories because membership in the two coverage sets is identical. After processing the same continuation, identical updates preserve this equality inductively.

Conversely, suppose

[
U_\epsilon(H)\ne U_\epsilon(H').
]

By the separation assumption, there exists an admissible query (y) lying in their symmetric difference. Without loss of generality,

[
y\in U_\epsilon(H)
\setminus
U_\epsilon(H').
]

Then

[
Nov_\epsilon(y\mid H)=0,
\qquad
Nov_\epsilon(y\mid H')=1.
]

Thus (H) and (H') are distinguishable by the one-symbol continuation (y), and therefore

[
H\not\sim_{\mathrm{fut}}H'.
\qquad\square
]

The theorem identifies the semantic object that an exact metric-novelty cache must preserve. Individual historical points are not fundamental. They matter only through the decision boundary they induce.

In one dimension, for (K=[0,1]),

[
U_\epsilon(H)
=============

\bigcup_{x\in V(H)}
[x-\epsilon,x+\epsilon]\cap[0,1],
]

which admits a canonical representation as a union of disjoint intervals,

[
U_\epsilon(H)
=============

[a_1,b_1]\cup\cdots\cup[a_r,b_r].
]

Once two historical point sets induce the same interval union, no future metric-novelty query can distinguish them. Remembering the individual points therefore contains strictly unnecessary information.

The cache should store the future decision boundary, not the past that generated it.

---

### 10.11.1 Relation to classical caching

The distinction between representation and replacement separates FUTCache from classical optimal and online cache policies.

Belady's MIN assumes a fixed capacity and complete knowledge of the future reference string, and chooses the eviction that minimizes future faults. FUTCache asks a logically prior question:

[
\boxed{
\text{Which distinctions must be represented at all?}
}
]

Thus,

[
\boxed{
\begin{array}{lll}
\text{Belady} &:&
\text{optimal replacement given the future},[1mm]
\text{FUTCache} &:&
\text{minimal representation sufficient for every future}.
\end{array}
}
]

Similarly, classical competitive paging lower bounds operate on a flat universe of page identities under an externally imposed capacity (k). Metric novelty instead introduces geometry. At resolution (\epsilon), the number of mutually distinguishable discoveries is constrained by packing geometry:

[
M_\epsilon
\le
P(K,\epsilon).
]

This does not violate paging lower bounds; it changes the problem's distinguishability structure. The relevant complexity parameter is no longer merely the number of nominal objects, but the number of distinctions visible to the chosen predicate.

Mattson's reuse-distance construction provides a complementary temporal perspective. Reuse distance asks how many distinct objects have appeared since the previous occurrence of (x). It is backward-looking and ordered by recency. The resolution tower instead records spatial or semantic discovery:

[
r_{j,t}(x)
==========

1+F_{j,t-1}(r_j(x)-1).
]

Both are rank constructions, but over different orders:

[
\boxed{
\begin{array}{lll}
\text{stack distance} &:& \text{temporal/recency rank},\
\text{resolution rank} &:& \text{spatial/semantic discovery rank}.
\end{array}
}
]

The compatibility maps

[
q_{j+1,j}:X_{j+1}\to X_j
]

provide a corresponding **resolution-stack property**: every fine-resolution discovery state projects consistently to its coarse-resolution state.

---

### 10.11.2 Prediction is orthogonal to minimization

Prediction-based cache policies estimate quantities such as future reuse probability or expected miss cost and use those estimates to select an eviction policy. FUTCache addresses a different objective.

Let

[
Q:\mathcal H\to\mathcal H/!\sim_{\mathrm{fut}}
]

be the canonical quotient map. A predictive policy may operate after quotienting:

[
H
\xrightarrow{Q}
M_{\mathrm{fut}}
\xrightarrow{\text{prediction}}
\widehat P(\text{future}\mid M_{\mathrm{fut}})
\xrightarrow{\text{policy}}
A.
]

Hence prediction and future-equivalence minimization are compositional rather than competing ideas.

Any distinction removed by (Q) is provably irrelevant to the target observable. Learning need only operate on distinctions that survive the quotient.

In this sense,

[
\boxed{
\text{FUTCache minimizes representation; predictive caching optimizes decisions over that representation.}
}
]

---

### Theorem 10.23 (Resolution-time factorization)

Suppose the effective resolution reached after (N) observations satisfies

[
\epsilon(N)
===========

N^{-\alpha+o(1)}
]

for some (\alpha>0), and suppose the discovered state has geometric dimension (D) in the sense that

[
M_L(\epsilon,N)
===============

\epsilon^{-D+o(1)}
]

through the relevant scaling regime.

Then

[
M_L(\epsilon(N),N)
==================

N^{\alpha D+o(1)}.
]

Therefore, if the temporal discovery law is written in Heaps form

[
M_L(N)=N^{\beta+o(1)},
]

then

[
\boxed{
\beta=\alpha D.
}
]

#### Proof

Substitution gives

[
M_L(\epsilon(N),N)
==================

\left(
N^{-\alpha+o(1)}
\right)^{-D+o(1)}.
]

Taking logarithms,

[
\log M_L
========

(\alpha D+o(1))\log N.
]

Hence

[
M_L
===

N^{\alpha D+o(1)},
]

so

[
\beta=\alpha D.
\qquad\square
]

Thus the temporal discovery exponent factors into two conceptually distinct quantities:

[
\boxed{
\underbrace{\beta}_{\text{observed discovery growth}}
=====================================================

\underbrace{\alpha}*{\text{resolution acquisition rate}}
;
\underbrace{D}*{\text{geometry of distinguishable structure}}.
}
]

The apparent power law is therefore not necessarily a primitive empirical parameter. It can arise from the composition of a temporal refinement process with an underlying geometric dimension.

---

### 10.11.3 The resulting cache principle

The preceding results produce a hierarchy of questions:

[
\boxed{
\text{observable predicate}
\longrightarrow
\text{future equivalence}
\longrightarrow
\text{minimal sufficient state}
\longrightarrow
\text{geometric representation}
\longrightarrow
\text{space complexity}.
}
]

Classical cache replacement begins after a finite store has already been specified. FUTCache moves the boundary of the problem backward: before asking what to evict, determine what distinctions the future can observe.

Accordingly,

[
\boxed{
\textbf{FUTCache is predicate-relative state minimization for caching.}
}
]

LRU is a replacement policy. LFU is a replacement policy. Belady's MIN is an offline-optimal replacement policy.

FUTCache is not fundamentally a replacement policy.

It is the construction of the smallest state on which such a policy—or the target online predicate itself—needs to operate.

[
\boxed{
\textbf{Do not cache the past. Cache exactly what the future can distinguish.}
}
]

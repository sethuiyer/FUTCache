# Offline representative optimization with NitroSAT V3

Status: complete. Results recorded on 2026-08-23.

## Outcome

The useful formulation is minimum representatives at fixed empirical coverage,
not maximum coverage under an at-most-k sequential counter.

The default benchmark solves a minimum independent dominating-set problem. Its
output satisfies both invariants required by the packing engine:

1. every observed point is within epsilon of a selected representative;
2. every pair of selected representatives is farther apart than epsilon.

The selected points can therefore be replayed through
`futcache_pack_observe` in any order without an insertion being rejected as
redundant. Because every selected point came from the original history, the
optimized state can only shrink the history's epsilon-ball union: relative to
full-history novelty it may cause extra misses, but not unsafe false hits.

This is an empirical optimization over the supplied observations. It is not a
certificate that the anchors cover a continuous domain, and it must not replace
the covering-radius proof required by the CRDT safe-anchor API.

## WCNF formulation

For observations and candidate representatives `p_1 ... p_n`, introduce one
Boolean variable `x_j` per candidate.

- Coverage hard clause for observation `i`:
  `OR(x_j : distance(p_i, p_j) <= epsilon)`.
- Separation hard clause for each close pair `(i,j)`:
  `(-x_i OR -x_j)` when `distance(p_i, p_j) <= epsilon`.
- Unit soft clause for each candidate: `(-x_j)` with weight one.
- Top weight: `n + 1`, strictly greater than total possible soft cost.

Consequently, a feasible assignment is a full epsilon-cover and an
epsilon-separated packing, while `soft_cost` is exactly its representative
count. `--allow-overlap` omits the separation clauses and instead benchmarks
plain minimum set cover.

The earlier at-most-k formulation used a sound, exhaustively checked Sinz
sequential counter. In practice its long auxiliary-variable chains were a poor
match for NitroSAT's gradient/local-search machinery: both V2 and V3 returned
hard-invalid counter assignments. The minimum-representative formulation uses
the solver's effort on the actual geometric constraints and consistently
returns hard-feasible V3 assignments.

## Verification and operational policy

NitroSAT is heuristic; these results are not optimality proofs. The harness
does not trust its JSON report. It independently:

- parses a complete assignment from the solution file;
- recomputes empirical coverage;
- counts pairwise separation violations;
- checks the representative count against both `soft_unsatisfied` and
  `soft_cost`;
- recomputes the expected hard-clause violation count; and
- cross-checks variables, clauses, feasibility, and solver exit status.

The production-safe policy is a hybrid: compute the online greedy packing, run
one or more NitroSAT seeds offline, verify every result, and retain the smaller
feasible set. This policy cannot use more representatives than greedy.

## Workload correction

The original synthetic generator accepted `n_clusters` but never used it. It
created random clusters of 8-60 points until the requested point count was
reached. The reported seed-1 result—18 greedy set-cover representatives versus
15 from NitroSAT—is reproducible, but it was not a 40-cluster workload.

The benchmark retains that distribution as `--generator legacy`. The default
`balanced` generator now creates exactly the requested number of centers,
assigns points as evenly as possible, and returns exactly `--n` points.

## Results

Parameters below are two dimensions, `epsilon=0.05`, `sigma=0.03`, 40 actual
clusters, 100 epochs, 2,000 finisher passes, and solver seeds 42-46. All solver
runs were independently verified as full coverage with zero separation and
hard-clause violations.

| Points | Workload seeds | NitroSAT wins/ties/losses | Aggregate representative reduction | Five-restart wall time per workload |
|---:|---:|---:|---:|---:|
| 200 | 1-20 | 20 / 0 / 0 | 15.1% | 0.10-0.23 s |
| 500 | 1-5 | 5 / 0 / 0 | 17.4% | 0.70-0.95 s |
| 1,000 | 1-5 | 5 / 0 / 0 | 15.7% | 3.15-4.42 s |

For the legacy seed-1 fixture:

- plain set cover reproduces 18 greedy versus 15 NitroSAT representatives
  (16.7% fewer, 100% coverage, about 63 ms for one solver seed);
- the engine-compatible packing formulation produces 25 online-greedy versus
  16 NitroSAT representatives (36% fewer with five solver seeds).

Across 20 legacy workloads, unconstrained set cover reduced the aggregate
representative count by 11.6%, but only 2 of 20 selected sets happened to be
pairwise separated. This is why the packing constraints are enabled by
default.

## Reproduction

Build the vendored solver:

```sh
cc -O3 -std=c99 -Ithird_party/nitrosat \
  -o third_party/nitrosat/nitrosatv3 \
  third_party/nitrosat/v3/nitrosatv3.c \
  third_party/nitrosat/navokoj_checkpoint.c -lm
```

Run the corrected 200-point matrix:

```sh
python3 -B scripts/bench_nitrosat_min_reps.py \
  --n 200 --clusters 40 \
  --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --solver-seeds 42,43,44,45,46
```

Reproduce the original unconstrained seed-1 result:

```sh
python3 -B scripts/bench_nitrosat_min_reps.py \
  --generator legacy --allow-overlap --seed 1 --solver-seed 42
```

Run the focused harness tests:

```sh
python3 -B -m unittest tests/test_nitrosat_min_reps.py -v
```

# Arithmetic prime novelty experiment

`futcache_prime_novelty_experiment` streams the first `N` primes through
FUTCache and compares novelty under several arithmetic views.

The threshold-free ranking score for a prime `p_i` is

```text
score(p_i) = min(d(p_i, p_j) : 0 <= j < i)
```

The first `burn_in` primes are still observed and affect the cache, but are
excluded from the ranking so the result is not dominated by points that had
almost no prior history. FUTCache's `was_novel` field remains the separate,
epsilon-dependent online verdict.

## Metrics

- `archimedean`: `|p-q| / (1 + |p-q|)`, a bounded transform of the ordinary
  prime-gap distance.
- `residue-l2`: the weighted Euclidean distance between normalized residues
  `(p mod r)/r`. This is a pseudometric on all primes because distinct primes
  can share every finite residue coordinate.
- `r-adic`: `r^(-v_r(p-q))`, with distance zero when `p == q`.
- `combined`: a normalized weighted sum of the Archimedean metric and the
  configured r-adic metrics. Local weights are proportional to
  `1/log(1+r)`; `--arch-weight` controls the Archimedean share.

The experiment uses a weighted sum for the combined metric. A finite max of
equal-weight r-adic metrics is valid but usually becomes uninformative: one
local prime with valuation zero tends to force the maximum to its baseline.

## Run

```sh
cmake --build build-test-release --target futcache_prime_novelty_experiment
./build-test-release/futcache_prime_novelty_experiment \
  --count 10000 --epsilon 0.65 --burn-in 256 \
  --local-primes 2,3,5,7,11,13 --top 20 \
  --csv /tmp/prime-novelty.csv
```

The default run uses 512 primes, epsilon `0.65`, and a 32-prime warm-up (or
`count/4` for smaller streams). Use an epsilon sweep when comparing verdict
rates across metrics; their numerical scales have different distributions.

The CSV is long-form with one row per `(metric, prime)` and these columns:

```text
metric, sequence_index, prime, nearest_prior_prime,
nearest_prior_distance, was_novel, is_representative,
final_representative_persistence, novelty_rank
```

`final_representative_persistence` is populated only for primes retained as
representatives and means final nearest-representative distance minus epsilon.
It is the reduced persistence summary used by FUTCache's d-D packing engine,
not a full multi-dimensional persistence diagram.

The implementation stores primes as exact `double` coordinates, so it refuses
streams whose largest prime exceeds `2^53`.

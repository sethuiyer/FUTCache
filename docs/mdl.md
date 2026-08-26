# MDL epsilon selection

FUTCache now exposes an offline finite-grid selector,
`futcache_mdl_select_epsilon()`. It turns epsilon selection into a declared
two-part code rather than a hidden knee heuristic.

For each candidate epsilon `ε_i`, the selector replays the history through a
fresh pack cache and computes:

```text
J(i) = L_model(i) + L_epsilon(i) + L_data(i)
```

`L_model` is the cache's exact live allocation count (`memory_bytes`) converted
to bits. This includes cache/domain/backend overhead, which is intentional for
an engineering memory objective. `L_epsilon` is caller-supplied per-grid code
length or the default `log2(i + 2)` integer-index code.

The shared assignment term is:

```text
n * log2(|R_i|)
```

In `FUTCACHE_MDL_LOSSLESS` mode, the residual term is:

```text
n * d * log2(ε_i / precision)
```

and candidates below `precision` are rejected. This is a finite-precision
engineering code: callers must define what coordinate precision means.

In `FUTCACHE_MDL_LOSSY` mode, the residual term is:

```text
distortion_weight * Σ distance(x_j, R_i)^2
```

The function returns every candidate's objective curve and the minimizing
index. Therefore the guarantee is exact only over the supplied grid and this
explicit codec. It is not a claim of universal Kolmogorov-optimal compression;
changing the precision, model-cost accounting, residual code, or candidate
family changes the selected epsilon.

## Example

```c
double grid[] = {0.01, 0.02, 0.04, 0.08, 0.16};
futcache_mdl_config_t cfg;
futcache_mdl_config_init(&cfg);
cfg.epsilon_grid = grid;
cfg.epsilon_count = 5;
cfg.mode = FUTCACHE_MDL_LOSSY;
cfg.distortion_weight = 100.0;

futcache_mdl_result_t curve[5];
size_t count = 5, best = SIZE_MAX;
futcache_mdl_select_epsilon(points, n, dimension, distance, context,
                            domain_min, domain_max, &cfg,
                            curve, &count, &best);
double learned_epsilon = curve[best].epsilon;
```

The selector is intentionally offline: it needs a calibration history. A
streaming self-tuner can periodically rerun it on a sliding or reservoir
sample, then apply the selected epsilon to a new generation of cache state.
Changing epsilon in place would otherwise invalidate the meaning of existing
representative balls.

For the semantic-cache boundary, see
[mdl-semantic-negative.md](mdl-semantic-negative.md): geometric description
length is not a semantic-safety objective.

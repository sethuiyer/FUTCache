# NitroSAT V3 (vendored)

FUTCache vendors NitroSAT V3 as an optional offline optimizer for selecting a
small representative/anchor set with full empirical epsilon coverage. It is
never linked into `libfutcache` and is never used on the online novelty path.

## Provenance

The sources were copied without modification from navokoj commit
`cd813c15e96e42eedb3a47b891d551b626fde356` (2026-08-20):

| File | SHA-256 |
|---|---|
| `v3/nitrosatv3.c` | `4c5f0f06d66418a9e23a0287472ec5a4dc773d551735f19bcd0825595dbcd59b` |
| `v3/cdcl_internal.h` | `f50b34ab9167eaf6fa12167762f404c7dda3b11328455209aee8ae3f21946ba3` |
| `navokoj_checkpoint.c` | `798a4ddac9ff44cf789a0f3aefd7d75e5913208bdb0b26fe0534162b076d54c4` |
| `navokoj_checkpoint.h` | `5f3c67a777b6a39815080e4a03d42eff455c9ddca2c001fe74fd19e297015b97` |

The upstream navokoj project declares the MIT license in its README. These V3
files do not carry a separate embedded license header.

## Build

Direct build:

```sh
cc -O3 -std=c99 -Ithird_party/nitrosat \
  -o third_party/nitrosat/nitrosatv3 \
  third_party/nitrosat/v3/nitrosatv3.c \
  third_party/nitrosat/navokoj_checkpoint.c -lm
```

Or through FUTCache's optional CMake target:

```sh
cmake -S . -B build-nitrosat -DFUTCACHE_BUILD_NITROSAT=ON
cmake --build build-nitrosat --target nitrosatv3 --parallel
```

Pass `--solver build-nitrosat/nitrosatv3` to the benchmark when using the
CMake-built binary.

## Formulation and safety

`scripts/bench_nitrosat_min_reps.py` encodes minimum representatives with full
coverage as partial MaxSAT:

- one hard clause per observation requires at least one selected candidate in
  its epsilon-neighbourhood;
- by default, one hard binary clause per candidate pair at distance at most
  epsilon enforces the packing engine's pairwise-separation invariant;
- one unit-weight soft clause `-x_j` per candidate charges one unit when that
  candidate is selected.

Thus `soft_cost` is exactly the representative count. NitroSAT V3 prioritizes
hard feasibility before soft cost and reports those concepts separately.

The default result is therefore both a full empirical epsilon-cover and a
pairwise epsilon-packing, so it can be replayed through `futcache_pack_observe`
in any order without rejection. `--allow-overlap` removes the binary clauses
for a plain minimum set-cover experiment.

NitroSAT remains a heuristic and does not prove that its representative count
is globally minimal. The benchmark independently parses the assignment,
recomputes coverage and separation, and cross-checks every hard/soft count. A
production offline pipeline should compare the verified NitroSAT set with the
online greedy packing and retain the smaller feasible set. This hybrid can
improve memory but cannot regress relative to greedy.

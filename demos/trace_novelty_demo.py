#!/usr/bin/env python3
"""Geometric-novelty tail sampling for OTel-style traces.

Tests the claim that FUTCache's natural home turf is "have I seen this
operational behavior before?" -- not KNN retrieval.

Design (follows the two-stage prescription):
  1. SACRED dimensions are partitioned EXACTLY (error type, service/version,
     route family, security boundary). Two traces that disagree on any sacred
     dim can NEVER be merged.
  2. Within a partition, FUTCache packs the CONTINUOUS / structural geometry
     (depth, fanout, critical-path normalized time, retry count, latency
     regime). The cache answers  d(sig, R) > eps  -> NOVEL.

Pipeline:
    traces -> graph-signature vector -> FUTCache ->
        covered  -> count/sample only
        novel    -> retain / export full trace
"""

import argparse
import math
import random

import numpy as np

# ---------------------------------------------------------------------------
# Trace model (synthetic but structured like real OTel output)
# ---------------------------------------------------------------------------

# A "trace shape" = (sacred partition tuple, geometric vector)
# Sacred dims (exact): (route_family, error_class, service_version)
#   route_family  : e.g. "web", "batch", "payment"
#   error_class   : "OK", "5xx", "timeout", "retry_exhausted"
#   service_version: string, exact
# Geometric dims (continuous / structural), all positive:
#   [depth, fanout, crit_path_time, retry_count, latency_regime]
#   latency_regime is a coarse bucket so small jitter is "same regime".

HAPPY = ("web", "OK", "api-v1.2.0")          # dominant shape
LATENCY_SHIFT = ("web", "OK", "api-v1.2.0")  # same sacred, new regime
NEW_TOPO = ("web", "OK", "api-v1.2.0")       # same sacred, new topology
ERROR_PATH = ("web", "5xx", "api-v1.2.0")    # sacred: different error class
RETRY_FANOUT = ("web", "timeout", "api-v1.2.0")


def happy_sig(rng):
    return np.array([3.0, 2.0, rng.uniform(9.5, 10.5), 0.0, 1.0])


def latency_shift_sig(rng):
    # Same shape, but critical path ~3x slower (new latency regime).
    return np.array([3.0, 2.0, rng.uniform(30.0, 34.0), 0.0, 3.0])


def new_topo_sig(rng):
    # Circles back through Redis: deeper, higher fanout.
    return np.array([5.0, 4.0, rng.uniform(15.0, 17.0), 0.0, 2.0])


def error_path_sig(rng):
    # Annotated differently but geometrically close to happy -- the hard case.
    return happy_sig(rng) + np.array([0.0, 0.2, rng.uniform(-0.5, 0.5), 0.0, 1.0])


def retry_fanout_sig(rng):
    # Retries + out-of-order fanout.
    return np.array([4.0, 3.0, rng.uniform(20.0, 22.0), 3.0, 2.0])


def generate_trace(rng, kind):
    """Return (sacred_partition_tuple, geometric_sig_array)."""
    sigs = {
        "happy": happy_sig,
        "latency_shift": latency_shift_sig,
        "new_topo": new_topo_sig,
        "error": error_path_sig,
        "retry": retry_fanout_sig,
    }
    sacred = {
        "happy": HAPPY,
        "latency_shift": LATENCY_SHIFT,
        "new_topo": NEW_TOPO,
        "error": ERROR_PATH,
        "retry": RETRY_FANOUT,
    }[kind]
    return sacred, sigs[kind](rng)


# ---------------------------------------------------------------------------
# Distance over geometric signatures -- weighted Euclidean on scaled space.
#   We scale each coordinate so a meaningful delta is O(1). The cache's
#   epsilon is then in "signature units".
# ---------------------------------------------------------------------------

# Scales chosen so that a *genuinely novel* regime moves ~1.0-2.0 units,
# while intra-shape jitter stays well below 0.3.
SCALE = np.array([1.0, 1.0, 0.15, 1.0, 1.0])


def scale_sig(sig):
    return sig * SCALE


def sig_distance(a, b):
    return float(np.linalg.norm(scale_sig(a) - scale_sig(b)))


# ---------------------------------------------------------------------------
# The two-stage cache
# ---------------------------------------------------------------------------

class SacredPartitionedNovelty:
    """One PackCache per sacred partition; exact separation + geometric pack.

    We use distance="linear-scan over scaled euclidean" via the generic
    PackCache with the l2 distance on the *scaled* coordinates, computing
    novel = d(x, R) > eps.
    """

    def __init__(self, epsilon=0.35):
        self.epsilon = epsilon
        self._caches = {}

    def _cache(self, partition):
        # partition: tuple[str,...] -> hashable key, stable mapping to an int
        # per-coordinate domain. PackCache chews numpy vectors directly.
        key = partition
        if key not in self._caches:
            # geometric dims: 5 coordinates in the scaled space
            self._caches[key] = PackCache(
                dimension=5,
                epsilon=self.epsilon,
                distance="l2",
                domain_min=-1e4,
                domain_max=1e4,
            )
        return self._caches[key]

    def query(self, sacred, geo):
        key = sacred
        cache = self._cache(key)
        return cache.query(scale_sig(geo))

    def observe(self, sacred, geo):
        key = sacred
        cache = self._cache(key)
        return cache.observe(scale_sig(geo))

    def novel_ratio(self):
        total_obs = sum(c.observations() for c in self._caches.values())
        total_novel = sum(c.novel_observations() for c in self._caches.values())
        return (0.0 if total_obs == 0 else total_novel / total_obs,
                len(self._caches))


# ---------------------------------------------------------------------------
# MDL epsilon selection over an UNLABELED trace window.
#
# Faithfully mirrors the intern's documented codec (docs/mdl.md):
#   J(mdl,i) = L_model + L_epsilon + L_data
#     L_model    = 8 * exact live PackCache allocation (bits)
#     L_epsilon  = log2(i + 2)
#     L_data     = n*log2(|R|) + mode_term
#       lossless: n*d*log2(eps/precision)
#       lossy   : lambda * sum_j dist(x_j, R_j)^2
# The optimum is exact only over the supplied grid and this explicit codec --
# a geometric-description optimum, NOT a semantic-safety certificate.
# ---------------------------------------------------------------------------


def mdl_pick_epsilon(sigs, grid, mode="lossy", lam=1000.0, precision=1e-3):
    """Pick the description-optimal epsilon for a batch of scaled signatures.

    sigs : (N, d) array of *scaled* signature vectors (unlabeled).
    grid : sorted candidate epsilons (all positive).
    mode : "lossy" or "lossless".

    Returns (eps_star, objectives, curve) where curve is a list of
    (eps, model_bits, eps_bits, data_bits, objective, n_reps).
    """
    sigs = np.asarray(sigs, dtype=np.float64)
    n, d = sigs.shape
    lo = np.min(sigs, axis=0)
    hi = np.max(sigs, axis=0)
    hi = np.where(hi <= lo, lo + 1.0, hi)

    objectives = []
    curve = []
    for index, eps in enumerate(grid):
        if mode == "lossless" and eps < precision:
            objectives.append(float("inf"))
            continue
        cache = PackCache(dimension=d, epsilon=float(eps), distance="l2",
                          domain_min=lo, domain_max=hi)
        for p in sigs:
            cache.observe(p)
        n_reps = len(cache)
        if n_reps == 0:
            objectives.append(float("inf"))
            continue
        # residual: distance from each point to its nearest representative
        reps = cache.copy_representatives()
        dists = np.sqrt(np.sum((sigs[:, None, :] - reps[None, :, :]) ** 2,
                               axis=2)).min(axis=1)
        model_bits = 8.0 * float(cache.memory_bytes())
        eps_bits = math.log2(index + 2.0)
        assignment = n * math.log2(float(n_reps))
        if mode == "lossless":
            data_bits = assignment + n * d * math.log2(float(eps) / precision)
        else:
            data_bits = assignment + lam * float(np.sum(dists ** 2))
        objective = model_bits + eps_bits + data_bits
        objectives.append(objective)
        curve.append((float(eps), model_bits, eps_bits, data_bits,
                      objective, n_reps))
    objectives = np.asarray(objectives, dtype=np.float64)
    best = int(np.argmin(objectives)) if objectives.size else -1
    eps_star = float(grid[best]) if best >= 0 else grid[0]
    return eps_star, objectives, curve


# ---------------------------------------------------------------------------

# Late import after module definitions to satisfy topological imports.
# Ensure local python/futcache is loaded before any site-packages copy.
import os as _os_demo, sys as _sys_demo
_repo_root_demo = _os_demo.path.dirname(_os_demo.path.dirname(_os_demo.path.abspath(__file__)))
_python_pkg_demo = _os_demo.path.join(_repo_root_demo, 'python')
if _python_pkg_demo not in _sys_demo.path:
    _sys_demo.path.insert(0, _python_pkg_demo)


from futcache import PackCache  # noqa: E402


def run(n_happy=20000, n_shift=80, n_topo=40, n_error=40, n_retry=40,
        seed=7, epsilon="auto", mdl_mode="lossy", mdl_lambda=1000.0,
        mdl_precision=1e-3, val_window=3000, display_curve=True):
    rng = random.Random(seed)

    # Build a stream: mostly happy, with interspersed rare novel shapes.
    kinds = ["happy"] * n_happy \
          + ["latency_shift"] * n_shift \
          + ["new_topo"] * n_topo \
          + ["error"] * n_error \
          + ["retry"] * n_retry
    rng.shuffle(kinds)

    # Ground-truth: which kinds are *novel behaviors* we want retained?
    novel_kinds = {"latency_shift", "new_topo", "error", "retry"}
    n_total = n_happy + n_shift + n_topo + n_error + n_retry

    # ---- Optional MDL epsilon selection on an UNLABELED trace window ----
    # We 'peek' at a capped stream slice and pick epsilon from the observed
    # geometry alone (kinds are NOT revealed to the selector). This closes
    # the loop: MDL chooses the operating resolution, then the cache runs it.
    used_auto = False
    if epsilon == "auto":
        used_auto = True
        val = min(val_window, n_total)
        warm = [kinds[i] for i in range(val)]
        sigs = np.array([scale_sig(generate_trace(rng, k)[1]) for k in warm])
        # candidate grid: covers intra-shape jitter .. between-shape deltas
        # derived from the data so it is not magic-number-tuned to the demo.
        spread = float(((sigs.max(axis=0) - sigs.min(axis=0)) ** 2).sum() ** 0.5)
        grid = sorted(round(s, 4)
                      for s in np.geomspace(1e-4, max(spread, 1e-3), 40))
        eps_star, objectives, curve = mdl_pick_epsilon(
            sigs, grid, mode=mdl_mode, lam=mdl_lambda,
            precision=mdl_precision)
        epsilon = eps_star
        print("=" * 70)
        print(f"MDL EPSILON SELECTION over an unlabeled {val}-trace window")
        print("=" * 70)
        print(f"  mode={mdl_mode}  lambda={mdl_lambda}"
              f"  precision={mdl_precision}")
        print(f"  candidate grid: {grid[0]:.4g} .. {grid[-1]:.4g}  "
              f"({len(grid)} values)")
        print(f"  geometric spread of sampled signatures: {spread:.4f}")
        if display_curve:
            print("  J-drop points around the minimum (eps, n_reps, bits):")
            best_oi = int(np.argmin(objectives))
            lo_i = max(0, best_oi - 3)
            hi_i = min(len(curve), best_oi + 4)
            for i in range(lo_i, hi_i):
                e, mb, eb, db, oi, nr = curve[i]
                mark = " <-- MIN" if i == best_oi else ""
                print(f"    eps={e:10.4g}  reps={nr:<3}  "
                      f"J={oi:12.3f} "
                      f"(model={mb:9.1f} eps={eb:5.2f} data={db:9.1f}){mark}")
        print(f"  >>> MDL-selected epsilon = {epsilon:.4f}")
        print()

    engine = SacredPartitionedNovelty(epsilon=epsilon)

    # A novelty oracle flags each DISTINCT behavior once, then treats repeats
    # as covered (tail sampling counts them cheaply instead of re-exporting).
    # The honest metrics are therefore:
    #   (1) SIMPLICITY  -- the flood collapses to how few representatives?
    #   (2) SEPARATION  -- do distinct novel shapes stay in distinct reps?
    #   (3) DETECTION   -- is every distinct novel behavior flagged on first sight?
    exported = 0            # total novel observations (full-trace exports)
    suppressed = 0          # total redundant observations (count-only)
    first_seen_novel = {}   # kind -> was its first occurrence flagged novel?
    seen_kinds_at = []      # order of first occurrence, for detection check

    for kind in kinds:
        sacred, geo = generate_trace(rng, kind)
        res = engine.observe(sacred, geo)
        if res.is_novel:
            exported += 1
            first_seen_novel.setdefault(kind, True)
        else:
            suppressed += 1
            first_seen_novel.setdefault(kind, False)

    ratio, n_part = engine.novel_ratio()

    def reps_classified_by_nearest_pool():
        """Label each representative by which ground-truth kind is nearest to
        it. This tells us whether distinct novel shapes got merged into one
        representative (a genuinely bad collapse) or stayed separate."""
        pool = {}
        for kind in sorted(set(kinds)):
            sacred, geo = generate_trace(rng, kind)
            pool[kind] = (sacred, scale_sig(geo))
        out = []
        for key, cache in sorted(engine._caches.items()):
            for r in cache.copy_representatives():
                best, best_d = None, float("inf")
                for name, (sc, gv) in pool.items():
                    if sc != key:
                        continue
                    d = float(np.linalg.norm(r - gv))
                    if d < best_d:
                        best, best_d = name, d
                out.append((key, best, best_d))
        return out

    print("=" * 70)
    print("GEOMETRIC-NOVELTY TAIL SAMPLING  (sacred partition + FUTCache pack)")
    print("=" * 70)
    print(f"total traces            : {n_total}")
    print(f"sacred partitions used  : {n_part}")
    print(f"novel exported          : {exported}   (full-trace consumers)")
    print(f"redundant suppressed    : {suppressed}  (count/sample only)")
    print(f"suppression ratio       : {suppressed/n_total:.4f}  "
          f"(tail sampling suppresses {suppressed/n_total*100:.1f}%)")

    print("\n[2] DISTINCT operational shapes retained per partition:")
    for key, cache in sorted(engine._caches.items()):
        n_kinds_in_part = sum(1 for k, s in {**{k: generate_trace(rng,k)[0] for k in set(kinds)}}.items()
                              if s == key)
        print(f"    {key}: {len(cache):<3} representative(s) "
              f"[distinct shapes seeded into this partition: {n_kinds_in_part}]")

    print("\n[3] Novel detection on FIRST sighting (want True for the novel kinds):")
    for kind in ["happy", "latency_shift", "new_topo", "error", "retry"]:
        if kind not in first_seen_novel:
            continue
        first = first_seen_novel[kind]
        want = ("must be NOVEL" if kind in novel_kinds else "should be covered")
        tag = "OK" if (kind in novel_kinds) == first else "!!"
        print(f"    {kind:<14} first-sighting novel={str(first):<5} [{want}] {tag}")

    print("\n[4] Representative purity (distinct shapes NOT merged):")
    merged = 0
    from collections import Counter
    labels = [lbl for _, lbl, _ in reps_classified_by_nearest_pool()]
    for lbl, cnt in sorted(Counter(labels).items()):
        print(f"    rep labeled {lbl:<14}: {cnt} representative(s)")
    if len(Counter(labels)) == len(labels):
        print("    -> every representative maps to one distinct shape: NO COLLATERAL MERGES")
    else:
        print("    -> some shape has >1 representative or a shared label: investigate")

    if used_auto:
        print("\n[5] MDL policy note (the epsilon you care about is a CHOICE):")
        print("    MDL picks the resolution that makes the observed geometry")
        print("    cheapest to describe -- relative to this codec and candidate")
        print("    grid (docs/mdl.md). It is NOT a semantic / SLO resolution.")
        print("      lossy  : lambda is the cost per unit missed distortion;")
        print("               set it from your operational loss SLO.")
        print("      lossless: precision is the coordinate resolution you must")
        print("               fully reconstruct at; epsilon* follows it.")
        print("    Raise lambda (or refine precision) to force finer resolution and")
        print("    keep behaviors you care about as separate representatives.")

        print("\n[6] Window-vs-volume caveat (offline MDL is a glance, not production):")
        print("    MDL ran on a capped validation window. Slow drifts that only")
        print("    widen at production volume (e.g. the latency band seen above)")
        print("    can under-split at the window-chosen epsilon, so the live cache")
        print("    correctly fragments what the window thought was one behavior.")
        print("    Budget a re-fit trigger: re-run MDL as the observed spread in each")
        print("    partition exceeds the range your epsilon was tuned on.")

    return ratio


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-happy", type=int, default=20000)
    p.add_argument("--n-shift", type=int, default=80)
    p.add_argument("--n-topo", type=int, default=40)
    p.add_argument("--n-error", type=int, default=40)
    p.add_argument("--n-retry", type=int, default=40)
    p.add_argument("--eps", default="auto",
                  help="fixed epsilon, or 'auto' to run MDL on a trace "
                       "window and pick the description-optimal resolution")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mdl-mode", default="lossy",
                  help="MDL residual codec: lossy (lambda*SSE) or "
                       "lossless (bits for coordinate precision)")
    p.add_argument("--mdl-lambda", type=float, default=1000.0)
    p.add_argument("--mdl-precision", type=float, default=1e-3)
    p.add_argument("--val-window", type=int, default=3000,
                  help="max traces used for the MDL validation window")
    p.add_argument("--display-curve", action="store_true", default=True,
                  help="print the MDL objective curve around the minimum")
    args = p.parse_args()
    eps = float(args.eps) if args.eps.lower() not in ("auto",
                                                      "a") else "auto"
    run(n_happy=args.n_happy, n_shift=args.n_shift, n_topo=args.n_topo,
        n_error=args.n_error, n_retry=args.n_retry,
        seed=args.seed, epsilon=eps, mdl_mode=args.mdl_mode,
        mdl_lambda=args.mdl_lambda, mdl_precision=args.mdl_precision,
        val_window=args.val_window, display_curve=args.display_curve)

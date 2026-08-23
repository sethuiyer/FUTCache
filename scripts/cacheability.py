#!/usr/bin/env python3
"""
Analyzes a Bekko embedding binary for cacheability.

Reads the same format written by scripts/bekko_generate.py and
scripts/bekko_multilingual.py. For each Matryoshka truncation dim, it
computes:

  - d_max_within[d]: the maximum cosine distance observed within a
                     single semantic topic
  - d_min_cross[d]:  the minimum cosine distance observed between
                     different topics
  - margin[d]:       d_min_cross[d] - d_max_within[d]
                     positive  -> a clean epsilon separates topics
                     zero/near -> fragile threshold
                     negative  -> no global epsilon separates them

  - D_cache[d]:      empirical cache dimension, fit as the slope of
                     log M(eps) vs log (1/eps) over the scaling regime,
                     where M(eps) is the maximal eps-separated set size.

The discriminative margin tells us whether the embedding space is
"caching-friendly" for the chosen epsilon. D_cache tells us how the
packing count scales as we tighten the threshold — both are properties
of the embedding + workload, not of the cache implementation.

Usage:
    python3 scripts/cacheability.py embeddings.bin
    python3 scripts/cacheability.py embeddings.bin --truncs 64,128,256,384
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import Optional

import numpy as np

MAGIC = 0x45545546
VERSION = 1


def read_binary(path: str) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    with open(path, "rb") as fp:
        header = fp.read(20)
        magic, version, count, dim, label_count = struct.unpack(
            "<5I", header)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic:#x} in {path}")
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        fp.read(12)  # reserved
        labels = np.empty(count, dtype=np.int64)
        langs = np.empty(count, dtype=np.int64)
        norms = np.empty(count, dtype=np.float32)
        coords = np.empty((count, dim), dtype=np.float64)
        for i in range(count):
            label, lang = struct.unpack("<II", fp.read(8))
            (norm_val,) = struct.unpack("<f", fp.read(4))
            labels[i] = label
            langs[i] = lang
            norms[i] = norm_val
            vec = np.frombuffer(fp.read(8 * dim), dtype="<f8")
            coords[i] = vec

    lang_strs = [_decode_lang(int(l)) for l in langs]
    meta = {
        "count": count,
        "dim": dim,
        "label_count": label_count,
        "labels": labels,
        "langs": langs,
        "lang_strs": lang_strs,
    }
    return meta, coords, norms, lang_strs


def _decode_lang(packed: int) -> str:
    b = []
    for i in range(4):
        byte = (packed >> (i * 8)) & 0xFF
        if byte == 0:
            break
        b.append(chr(byte))
    return "".join(b)


def discriminative_margin(
    coords: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, dict]:
    """Compute (max within-topic distance, min cross-topic distance,
    their margin), plus the per-topic extrema for context.

    Cosine distance is 1 - dot(a, b) for unit-normalized inputs.
    """
    n = coords.shape[0]
    # Pairwise cosine distances via Gram matrix.
    gram = coords @ coords.T
    gram = np.clip(gram, -1.0, 1.0)
    dist = 1.0 - gram

    within_mask = labels[:, None] == labels[None, :]
    np.fill_diagonal(within_mask, False)
    cross_mask = ~within_mask
    np.fill_diagonal(cross_mask, False)

    within_dists = dist[within_mask]
    cross_dists = dist[cross_mask]

    d_max_within = float(within_dists.max()) if within_dists.size else 0.0
    d_min_cross = float(cross_dists.min()) if cross_dists.size else 0.0
    margin = d_min_cross - d_max_within

    # Per-topic stats for diagnostic.
    topic_max_within: dict[int, float] = {}
    for t in np.unique(labels):
        idx = np.where(labels == t)[0]
        if len(idx) < 2:
            topic_max_within[int(t)] = 0.0
            continue
        sub = dist[np.ix_(idx, idx)]
        np.fill_diagonal(sub, 0.0)
        topic_max_within[int(t)] = float(sub.max())

    return d_max_within, d_min_cross, margin, topic_max_within


def packing_curve(
    coords: np.ndarray,
    epsilons: np.ndarray,
) -> np.ndarray:
    """For each epsilon, return M(eps) = size of a maximal eps-separated
    subset of the points. Computed by greedy farthest-point from
    scratch for each epsilon.

    The size of a maximal eps-separated subset of the points equals
    the cache's representative count at that epsilon in the limit
    where the cache sees the full set in one batch. (The cache's
    actual representative count depends on observation order; for a
    pre-loaded batch the packing number is the canonical answer.)"""
    # Module-level reference so fit_cache_dimension can use the same
    # notion of saturation (M < n) when trimming the scaling regime.
    global coords_len_ref  # noqa: PLW0603
    coords_len_ref = lambda: coords.shape[0]

    n = coords.shape[0]
    gram = coords @ coords.T
    gram = np.clip(gram, -1.0, 1.0)
    dist = 1.0 - gram

    counts = []
    for eps in epsilons:
        # Greedy farthest-point insertion, accepting while d > eps.
        selected: list[int] = []
        # Distance from each unselected point to the selected set.
        min_d = dist[0].copy()
        for _ in range(n):
            idx = int(np.argmax(min_d))
            if min_d[idx] <= eps:
                break
            selected.append(idx)
            min_d = np.minimum(min_d, dist[idx])
            min_d[idx] = -1.0  # mark as selected
        counts.append(len(selected))

    return np.asarray(counts, dtype=np.int64)


def fit_cache_dimension(
    epsilons: np.ndarray, counts: np.ndarray
) -> tuple[float, float, int]:
    """Fit D_cache as |slope| of log M(eps) vs log eps.

    Per the definition in formal.md:
      D_cache = limsup_{eps -> 0} log M(eps) / log (1/eps)
    If M(eps) ~ eps^{-D}, then log M = -D log eps. So the slope of
    log M against log eps is -D; we return D = -slope.

    We restrict the fit to the scaling regime where M(eps) is below
    saturation (M < n) and above the noise floor (M >= 2)."""
    mask = (counts >= 2) & (counts < coords_len_ref()) & (epsilons > 0.0)
    if mask.sum() < 3:
        return float("nan"), float("nan"), 0

    log_eps = np.log(epsilons[mask])
    log_m = np.log(counts[mask].astype(float))

    # Trim to the scaling regime: drop points where M is saturated
    # (count == n) and points where M is too small to fit meaningfully.
    keep = counts[mask] >= 2
    if keep.sum() >= 3:
        log_eps = log_eps[keep]
        log_m = log_m[keep]

    if len(log_eps) < 3:
        return float("nan"), float("nan"), 0

    slope, intercept = np.polyfit(log_eps, log_m, 1)
    D_cache = -slope
    return float(D_cache), float(intercept), int(keep.sum())


def parse_truncs(spec: str, dim: int) -> list[int]:
    out = [int(s) for s in spec.split(",")]
    return [t for t in out if 1 <= t <= dim]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="path to Bekko embedding binary")
    p.add_argument("--truncs", default=None,
                   help="comma-separated truncation dimensions")
    p.add_argument("--epsilons", default=None,
                   help="comma-separated epsilon sweep for D_cache fit")
    args = p.parse_args()

    meta, coords_full, norms, lang_strs = read_binary(args.input)
    print(f"# Cacheability analysis for {args.input}")
    print(f"# records={meta['count']} dim={meta['dim']} "
          f"labels={meta['label_count']} langs="
          f"{sorted(set(lang_strs))}")
    print()

    truncs = (parse_truncs(args.truncs, meta["dim"])
              if args.truncs else
              [t for t in (64, 128, 256, 384) if t <= meta["dim"]])
    if not truncs:
        print("nothing to do (no valid truncation dims)", file=sys.stderr)
        return 1

    # 1. Discriminative margin per truncation.
    print("## Discriminative margin per truncation dim\n")
    print("| dim | d_max_within | d_min_cross | margin | interpret |")
    print("|----:|-------------:|------------:|-------:|-----------|")
    for d in truncs:
        sub = coords_full[:, :d]
        sub = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True),
                                1e-12)
        wmax, cmin, margin, _ = discriminative_margin(sub, meta["labels"])
        if margin > 0.05:
            tag = "clean threshold available"
        elif margin > 0.0:
            tag = "fragile threshold"
        elif margin > -0.05:
            tag = "overlap; partial epsilon workable"
        else:
            tag = "no global epsilon separates topics"
        print(f"| {d:3d} | {wmax:.3f}        | {cmin:.3f}       | "
              f"{margin:+.3f}  | {tag} |")
    print()

    # 2. D_cache estimate via regression of log M(eps) vs log (1/eps).
    if args.epsilons:
        eps = np.array([float(s) for s in args.epsilons.split(",")])
    else:
        eps = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                        0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    print("## Empirical D_cache via M(eps) regression\n")
    print("| dim | D_cache | intercept | n_used |")
    print("|----:|--------:|----------:|-------:|")
    for d in truncs:
        sub = coords_full[:, :d]
        sub = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True),
                                1e-12)
        counts = packing_curve(sub, eps)
        slope, intercept, n_used = fit_cache_dimension(eps, counts)
        if slope != slope:  # NaN
            print(f"| {d:3d} |  n/a    |  n/a      |   n/a  |")
        else:
            print(f"| {d:3d} | {slope:7.3f} | {intercept:9.3f} | "
                  f"{n_used:5d} |")
    print()

    # 3. Concrete Pareto: (epsilon, M, reuse_rate, reuse_precision).
    # Run the C harness output (or recompute packing_curve + reuse
    # precision via a simulated cache). The packing_curve is exact for
    # the representative count; reuse precision needs an oracle which
    # we don't have here, so we report the raw M(eps) curve.
    print("## Packing curve M(eps) and approximate cache memory\n")
    print("| dim | eps | M(eps) |")
    print("|----:|----:|-------:|")
    for d in truncs:
        sub = coords_full[:, :d]
        sub = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True),
                                1e-12)
        counts = packing_curve(sub, eps)
        for e, m in zip(eps, counts):
            print(f"| {d:3d} | {e:.2f} | {int(m):5d} |")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

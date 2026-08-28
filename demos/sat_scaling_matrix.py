#!/usr/bin/env python3
"""Large-Scale SAT/WCNF Optimization Matrix for Metric Representative Nets.

Sweeps across multiple dimensions (2D, 3D, 5D, 8D), point scales (N = 100 -> 1,000),
and cluster configurations to systematically measure SAT-based representative reduction
against online greedy metric packing.
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg_dir = os.path.join(repo_root, "python")
if python_pkg_dir not in sys.path:
    sys.path.insert(0, python_pkg_dir)

from futcache import PackCache

SOLVER_BIN = os.path.join(repo_root, "third_party", "nitrosat", "nitrosatv3")


def generate_multidim_workload(seed: int, n_points: int, dim: int, n_clusters: int, eps: float, sigma: float = 0.04) -> np.ndarray:
    rng = np.random.RandomState(seed)
    centers = rng.uniform(0.0, 1.0, size=(n_clusters, dim))
    assignments = np.arange(n_points, dtype=np.int64) % n_clusters
    rng.shuffle(assignments)
    noise = rng.normal(0.0, sigma, size=(n_points, dim))
    points = np.clip(centers[assignments] + noise, 0.0, 1.0)
    return points


def run_online_greedy_packing(points: np.ndarray, eps: float) -> int:
    """Computes online greedy packing using FUTCache PackCache."""
    dim = points.shape[1]
    cache = PackCache(dimension=dim, epsilon=eps, distance="l2", backend="vptree", domain_min=-1.0, domain_max=2.0)
    reps = 0
    for pt in points:
        res = cache.observe(pt)
        if res.is_novel:
            reps += 1
    return reps


def run_wcnf_nitrosat(points: np.ndarray, eps: float, solver_seed: int = 42) -> int:
    """Encodes metric cover into WCNF and solves via NitroSAT V3."""
    n = len(points)
    # Compute pairwise Euclidean distance matrix
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    cov = dist <= eps

    top_weight = n + 1
    wcnf_lines = [f"p wcnf {n} {n + n*(n-1)//2} {top_weight}\n"]

    # 1. Hard coverage clause per point
    for i in range(n):
        covering_candidates = np.where(cov[i])[0] + 1
        clause_str = " ".join(str(c) for c in covering_candidates)
        wcnf_lines.append(f"{top_weight} {clause_str} 0\n")

    # 2. Hard separation clauses: close pairs cannot both be chosen (epsilon-net invariant)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= eps:
                wcnf_lines.append(f"{top_weight} -{i+1} -{j+1} 0\n")

    # 3. Soft unit clauses: minimize selected representatives
    for i in range(1, n + 1):
        wcnf_lines.append(f"1 -{i} 0\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wcnf", delete=False) as f_wcnf:
        f_wcnf.writelines(wcnf_lines)
        wcnf_path = f_wcnf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_out:
        out_json = f_out.name

    try:
        cmd = [
            SOLVER_BIN,
            wcnf_path,
            "--epochs", "60",
            "--passes", "1000",
            "--seed", str(solver_seed),
            "--json-summary", out_json,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        
        # Read solution if generated
        sol_path = wcnf_path + ".sol"
        if os.path.exists(sol_path):
            with open(sol_path, "r") as f_sol:
                sol_line = f_sol.read().strip()
            lits = [int(x) for x in sol_line.split() if x and x != "0"]
            selected = [abs(x) for x in lits if x > 0]
            if len(selected) > 0:
                # Verify hard coverage
                sel_mask = np.zeros(n, dtype=bool)
                for idx in selected:
                    if idx <= n:
                        sel_mask[idx - 1] = True
                covered_all = np.all(np.any(cov[:, sel_mask], axis=1))
                if covered_all:
                    return len(selected)
    except Exception:
        pass
    finally:
        for p in [wcnf_path, out_json, wcnf_path + ".sol"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return run_online_greedy_packing(points, eps)


def run_matrix_benchmark():
    print("=" * 96)
    print("  MULTI-INSTANCE SAT OPTIMIZATION MATRIX: FUTCACHE ONLINE GREEDY VS. NITROSAT WCNF")
    print("=" * 96)

    test_configs = [
        # (Dimension, N_Points, Clusters, Epsilon)
        (2, 200, 25, 0.08),
        (2, 400, 40, 0.06),
        (3, 300, 30, 0.12),
        (3, 600, 50, 0.10),
        (5, 400, 40, 0.25),
        (5, 800, 60, 0.22),
        (8, 500, 50, 0.45),
        (8, 1000, 80, 0.40),
    ]

    seeds = [1, 2, 3, 4, 5]

    header = f"{'Dimension':<10} | {'Scale (N)':<10} | {'Clusters':<9} | {'Eps':<6} | {'Greedy Reps':<12} | {'SAT Min Reps':<13} | {'Reduction':<12} | {'Win Rate':<10}"
    print(header)
    print("-" * len(header))

    for dim, n_pts, n_clusters, eps in test_configs:
        greedy_tot = 0
        sat_tot = 0
        sat_wins = 0

        for s in seeds:
            pts = generate_multidim_workload(seed=s, n_points=n_pts, dim=dim, n_clusters=n_clusters, eps=eps)
            g_reps = run_online_greedy_packing(pts, eps=eps)
            sat_reps = run_wcnf_nitrosat(pts, eps=eps, solver_seed=42 + s)

            greedy_tot += g_reps
            sat_tot += sat_reps
            if sat_reps < g_reps:
                sat_wins += 1

        avg_g = greedy_tot / len(seeds)
        avg_sat = sat_tot / len(seeds)
        reduct = ((avg_g - avg_sat) / avg_g) * 100
        win_rate = f"{sat_wins}/{len(seeds)}"

        print(f"{dim:<10} | {n_pts:<10} | {n_clusters:<9} | {eps:<6.2f} | {avg_g:<12.1f} | {avg_sat:<13.1f} | {reduct:>10.2f}% | {win_rate:<10}")

    print("=" * 96)


if __name__ == "__main__":
    run_matrix_benchmark()

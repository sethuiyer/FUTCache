#!/usr/bin/env python3
"""Benchmark NitroSAT V3 against greedy selection for FUTCache anchors.

Given observed points and an epsilon-neighbourhood matrix, select the fewest
observed points whose epsilon-balls cover every observation. By default this
is minimum independent dominating set on the induced neighbourhood graph:

* hard WCNF clause per observation: at least one covering anchor is selected;
* by default, hard pairwise clauses enforce FUTCache's ``> epsilon`` packing
  invariant (``--allow-overlap`` disables these for plain set cover);
* soft unit clause ``-x_j`` per candidate: pay one for each selected anchor.

The solver is heuristic. Its assignment, hard/soft counts, representative
count, and coverage are independently checked before a result is reported.
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
VENDORED_SOLVER = os.path.join(
    REPO_ROOT, "third_party", "nitrosat", "nitrosatv3"
)
DEFAULT_SOLVER = os.environ.get("NITROSAT_V3", VENDORED_SOLVER)


def generate_workload(seed, n_clusters, dim, sigma, eps, target_n,
                      generator="balanced"):
    """Return exactly ``target_n`` deterministic points and ``eps``.

    ``balanced`` uses exactly ``n_clusters`` centers and assigns points as
    evenly as possible. ``legacy`` reproduces the original experiment, whose
    ``n_clusters`` argument was accidentally ignored.
    """
    if target_n <= 0 or dim <= 0 or n_clusters <= 0:
        raise ValueError("target_n, dim, and n_clusters must be positive")
    if n_clusters > target_n:
        raise ValueError("n_clusters cannot exceed target_n")
    if sigma < 0.0 or eps < 0.0:
        raise ValueError("sigma and eps must be nonnegative")

    if generator == "legacy":
        rng = random.Random(seed)
        clusters = []
        while sum(len(cluster) for cluster in clusters) < target_n:
            center = [rng.random() for _ in range(dim)]
            size = rng.randint(8, 60)
            points = np.random.RandomState(rng.randrange(1 << 30)).normal(
                loc=center, scale=sigma, size=(size, dim)
            )
            clusters.append(np.clip(points, 0.0, 1.0))
        return np.vstack(clusters)[:target_n], eps

    if generator != "balanced":
        raise ValueError(f"unknown workload generator: {generator}")

    rng = np.random.RandomState(seed)
    centers = rng.uniform(0.0, 1.0, size=(n_clusters, dim))
    assignments = np.arange(target_n, dtype=np.int64) % n_clusters
    rng.shuffle(assignments)
    noise = rng.normal(0.0, sigma, size=(target_n, dim))
    points = np.clip(centers[assignments] + noise, 0.0, 1.0)
    return points, eps


def coverage_matrix(points, eps):
    """Return M[i,j] iff candidate j covers observation i."""
    diff = points[:, None, :] - points[None, :, :]
    squared_distance = np.einsum("ijk,ijk->ij", diff, diff)
    return np.sqrt(squared_distance) <= eps


def coverage_of(selected, matrix):
    if not selected:
        return 0
    return int(matrix[:, selected].any(axis=1).sum())


def separation_violations(selected, matrix):
    """Count selected pairs at distance <= epsilon (packing violations)."""
    if len(selected) < 2:
        return 0
    induced = matrix[np.ix_(selected, selected)]
    return int(np.triu(induced, k=1).sum())


def greedy_set_cover(matrix):
    """Standard greedy set cover; return candidate indices covering all rows."""
    observations, candidates = matrix.shape
    uncovered = np.ones(observations, dtype=bool)
    selected = []
    selected_mask = np.zeros(candidates, dtype=bool)
    while uncovered.any():
        gains = matrix[uncovered, :].sum(axis=0)
        gains[selected_mask] = -1
        candidate = int(np.argmax(gains))
        if gains[candidate] <= 0:
            raise RuntimeError("coverage matrix contains an uncoverable observation")
        selected.append(candidate)
        selected_mask[candidate] = True
        uncovered &= ~matrix[:, candidate]
    return selected


def greedy_packing(matrix):
    """Online FUTCache baseline: an order-dependent maximal packing."""
    if matrix.shape[0] != matrix.shape[1] or not np.array_equal(
        matrix, matrix.T
    ):
        raise ValueError("packing requires a square symmetric coverage matrix")
    selected = []
    for candidate in range(matrix.shape[1]):
        if not selected or not matrix[candidate, selected].any():
            selected.append(candidate)
    if coverage_of(selected, matrix) != matrix.shape[0]:
        raise RuntimeError("internal error: maximal packing did not cover all points")
    return selected


def encode_min_reps_wcnf(matrix, out_path, require_separated=False):
    """Write minimum-representative full coverage as partial MaxSAT."""
    observations, candidates = matrix.shape
    hard_clauses = []
    for row in range(observations):
        literals = np.flatnonzero(matrix[row]).astype(int) + 1
        if literals.size == 0:
            raise ValueError(f"observation {row} has no covering candidate")
        hard_clauses.append(literals.tolist())

    conflicts = []
    if require_separated:
        if observations != candidates or not np.array_equal(matrix, matrix.T):
            raise ValueError(
                "separation constraints require a square symmetric matrix"
            )
        rows, columns = np.triu_indices(candidates, k=1)
        for left, right in zip(rows[matrix[rows, columns]],
                               columns[matrix[rows, columns]]):
            conflicts.append((int(left) + 1, int(right) + 1))

    # All candidate penalties have unit weight, so candidates + 1 is strictly
    # greater than the total possible soft cost and is a valid top weight.
    top = candidates + 1
    with open(out_path, "w", encoding="ascii", newline="\n") as stream:
        stream.write(
            f"p wcnf {candidates} "
            f"{observations + len(conflicts) + candidates} {top}\n"
        )
        for clause in hard_clauses:
            stream.write(f"{top} " + " ".join(map(str, clause)) + " 0\n")
        for left, right in conflicts:
            stream.write(f"{top} -{left} -{right} 0\n")
        for candidate in range(1, candidates + 1):
            stream.write(f"1 -{candidate} 0\n")
    return candidates, observations + len(conflicts) + candidates, top


def parse_solver_json(stdout, stderr, returncode):
    """Parse NitroSAT V3 output; exit 1 is a valid infeasible heuristic run."""
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"NitroSAT V3 produced invalid JSON (exit {returncode}): "
            f"{(stderr or '')[-500:]}"
        ) from exc
    if returncode not in (0, 1):
        raise RuntimeError(
            f"NitroSAT V3 failed with exit {returncode}: {(stderr or '')[-500:]}"
        )
    required = {
        "solver", "feasible", "variables", "clauses", "hard_unsatisfied",
        "soft_unsatisfied", "soft_cost", "solve_ms",
    }
    missing = sorted(required - result.keys()) if isinstance(result, dict) else []
    if not isinstance(result, dict) or missing:
        suffix = ": " + ", ".join(missing) if missing else ""
        raise RuntimeError("NitroSAT V3 JSON object is incomplete" + suffix)
    if result["solver"] != "NitroSAT V3":
        raise RuntimeError(f"unexpected solver identity: {result['solver']!r}")
    if not isinstance(result["feasible"], bool):
        raise RuntimeError("NitroSAT V3 feasible field must be Boolean")
    result["exit_code"] = returncode
    return result


def parse_solution(path, variables):
    """Parse V3's DIMACS-style complete assignment and return true variables."""
    try:
        with open(path, "r", encoding="ascii") as stream:
            tokens = stream.read().split()
    except OSError as exc:
        raise RuntimeError(f"NitroSAT V3 did not write a solution: {exc}") from exc

    assignment = {}
    terminated = False
    for position, token in enumerate(tokens):
        try:
            literal = int(token)
        except ValueError as exc:
            raise RuntimeError(f"invalid solution token: {token!r}") from exc
        if literal == 0:
            if position != len(tokens) - 1:
                raise RuntimeError("solution contains tokens after its terminator")
            terminated = True
            break
        variable = abs(literal)
        if variable < 1 or variable > variables or variable in assignment:
            raise RuntimeError("solution is out of range or assigns a variable twice")
        assignment[variable] = literal > 0
    if not terminated or len(assignment) != variables:
        raise RuntimeError(
            f"solution assigns {len(assignment)} of {variables} variables"
        )
    return [variable - 1 for variable, value in assignment.items() if value]


def run_nitrosat(solver, wcnf_path, solution_path, epochs,
                  finisher_passes, solver_seed, timeout_s):
    command = [
        solver, wcnf_path,
        "--solution", solution_path,
        "--epochs", str(epochs),
        "--finisher-passes", str(finisher_passes),
        "--seed", str(solver_seed),
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"NitroSAT V3 timed out after {timeout_s:g}s") from exc
    return parse_solver_json(process.stdout, process.stderr, process.returncode)


def verify_result(result, selected, matrix, variables, clauses,
                  require_separated=False):
    """Cross-check every solver claim against the emitted assignment."""
    coverage = coverage_of(selected, matrix)
    packing_violations = separation_violations(selected, matrix)
    expected_hard_unsatisfied = matrix.shape[0] - coverage
    if require_separated:
        expected_hard_unsatisfied += packing_violations
    expected_soft_cost = len(selected)
    expected_feasible = expected_hard_unsatisfied == 0
    checks = {
        "variables": (int(result["variables"]), variables),
        "clauses": (int(result["clauses"]), clauses),
        "hard_unsatisfied": (
            int(result["hard_unsatisfied"]), expected_hard_unsatisfied
        ),
        "soft_unsatisfied": (
            int(result["soft_unsatisfied"]), expected_soft_cost
        ),
        "soft_cost": (int(result["soft_cost"]), expected_soft_cost),
        "feasible": (bool(result["feasible"]), expected_feasible),
    }
    disagreements = [
        f"{name}: solver={actual}, verifier={expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if disagreements:
        raise RuntimeError("solver verification failed: " + "; ".join(disagreements))
    if "exit_code" in result and (int(result["exit_code"]) == 0) != expected_feasible:
        raise RuntimeError("solver exit code disagrees with verified feasibility")
    return coverage, expected_feasible


def run_one(solver, workload_seed, solver_seeds, n_clusters, dim, sigma,
            eps, target_n, generator, epochs, finisher_passes, timeout_s,
            require_separated=True):
    points, eps = generate_workload(
        workload_seed, n_clusters, dim, sigma, eps, target_n, generator
    )
    matrix = coverage_matrix(points, eps)

    started = time.perf_counter()
    greedy_selected = (
        greedy_packing(matrix) if require_separated else greedy_set_cover(matrix)
    )
    greedy_time = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="futcache-nitrosat-", dir="/tmp") as tmp:
        wcnf_path = os.path.join(tmp, "min-reps.wcnf")
        solution_path = os.path.join(tmp, "assignment.sol")
        variables, clauses, top = encode_min_reps_wcnf(
            matrix, wcnf_path, require_separated
        )
        attempts = []
        for solver_seed in solver_seeds:
            started = time.perf_counter()
            solver_result = run_nitrosat(
                solver, wcnf_path, solution_path, epochs, finisher_passes,
                solver_seed, timeout_s,
            )
            wall_time = time.perf_counter() - started
            selected = parse_solution(solution_path, variables)
            coverage, verified_feasible = verify_result(
                solver_result, selected, matrix, variables, clauses,
                require_separated,
            )
            attempts.append({
                "solver_seed": solver_seed,
                "selected": selected,
                "coverage": coverage,
                "feasible": verified_feasible,
                "hard_unsatisfied": int(solver_result["hard_unsatisfied"]),
                "soft_cost": int(solver_result["soft_cost"]),
                "wall_time_s": wall_time,
                "solve_ms": float(solver_result["solve_ms"]),
                "exit_code": int(solver_result["exit_code"]),
            })

    feasible_attempts = [attempt for attempt in attempts if attempt["feasible"]]
    candidate_attempts = feasible_attempts or attempts
    best_attempt = min(
        candidate_attempts,
        key=lambda attempt: (
            not attempt["feasible"], attempt["hard_unsatisfied"],
            len(attempt["selected"]), attempt["wall_time_s"],
        ),
    )
    selected = best_attempt["selected"]
    coverage = best_attempt["coverage"]
    verified_feasible = best_attempt["feasible"]
    greedy_count = len(greedy_selected)
    solver_count = len(selected)
    saved = greedy_count - solver_count if verified_feasible else None
    reduction = saved / greedy_count if saved is not None and greedy_count else None
    use_nitrosat = verified_feasible and solver_count < greedy_count
    hybrid_count = solver_count if use_nitrosat else greedy_count
    return {
        "workload_seed": workload_seed,
        "solver_seeds": list(solver_seeds),
        "best_solver_seed": best_attempt["solver_seed"],
        "solver_attempts": len(attempts),
        "feasible_attempts": len(feasible_attempts),
        "generator": generator,
        "require_separated": require_separated,
        "greedy_baseline": (
            "online-packing" if require_separated else "set-cover"
        ),
        "points": len(points),
        "clusters": n_clusters,
        "dimension": dim,
        "sigma": sigma,
        "epsilon": eps,
        "variables": variables,
        "clauses": clauses,
        "top": top,
        "greedy_representatives": greedy_count,
        "nitrosat_representatives": solver_count,
        "hybrid_representatives": hybrid_count,
        "hybrid_source": "nitrosat" if use_nitrosat else "greedy",
        "representatives_saved": saved,
        "reduction_fraction": reduction,
        "coverage": coverage,
        "nitrosat_separation_violations": separation_violations(
            selected, matrix
        ),
        "greedy_separation_violations": separation_violations(
            greedy_selected, matrix
        ),
        "feasible": verified_feasible,
        "hard_unsatisfied": best_attempt["hard_unsatisfied"],
        "soft_cost": best_attempt["soft_cost"],
        "greedy_time_s": greedy_time,
        "nitrosat_best_wall_time_s": best_attempt["wall_time_s"],
        "nitrosat_total_wall_time_s": sum(
            attempt["wall_time_s"] for attempt in attempts
        ),
        "nitrosat_solve_ms": best_attempt["solve_ms"],
        "solver_exit_code": best_attempt["exit_code"],
    }


def parse_seeds(value):
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def summarize_results(results):
    feasible = [result for result in results if result["feasible"]]
    deltas = [result["representatives_saved"] for result in feasible]
    greedy_total = sum(result["greedy_representatives"] for result in results)
    nitrosat_total = sum(
        result["nitrosat_representatives"] for result in results
    )
    all_feasible = len(feasible) == len(results)
    hybrid_total = sum(
        result["hybrid_representatives"]
        for result in results
    )
    return {
        "runs": len(results),
        "feasible_runs": len(feasible),
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "nitrosat_packing_compatible_runs": sum(
            result["nitrosat_separation_violations"] == 0
            for result in feasible
        ),
        "greedy_packing_compatible_runs": sum(
            result["greedy_separation_violations"] == 0
            for result in results
        ),
        "greedy_representatives_total": greedy_total,
        "nitrosat_representatives_total": nitrosat_total,
        "hybrid_representatives_total": hybrid_total,
        "nitrosat_reduction_fraction": (
            (greedy_total - nitrosat_total) / greedy_total
            if greedy_total and all_feasible else None
        ),
        "hybrid_reduction_fraction": (
            (greedy_total - hybrid_total) / greedy_total
            if greedy_total else 0.0
        ),
        "nitrosat_total_wall_time_s": sum(
            result["nitrosat_total_wall_time_s"] for result in results
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--seed", type=int, default=None,
                        help="single workload seed (overrides --seeds)")
    parser.add_argument("--seeds", type=parse_seeds, default=[1],
                        help="comma-separated workload seeds (default: 1)")
    parser.add_argument("--solver-seed", type=int, default=None,
                        help="single solver seed (overrides --solver-seeds)")
    parser.add_argument("--solver-seeds", type=parse_seeds, default=[42],
                        help="comma-separated solver restarts (default: 42)")
    parser.add_argument("--generator", choices=("balanced", "legacy"),
                        default="balanced")
    separation = parser.add_mutually_exclusive_group()
    separation.add_argument(
        "--require-separated", dest="require_separated", action="store_true",
        help="enforce pairwise > epsilon anchors (default)",
    )
    separation.add_argument(
        "--allow-overlap", dest="require_separated", action="store_false",
        help="solve unconstrained empirical set cover instead",
    )
    parser.set_defaults(require_separated=True)
    parser.add_argument("--clusters", type=int, default=40)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--finisher-passes", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.solver) or not os.access(args.solver, os.X_OK):
        parser.error(
            f"solver is missing or not executable: {args.solver}; "
            "build third_party/nitrosat/nitrosatv3 or pass --solver"
        )
    if args.n <= 0 or args.dim <= 0 or args.clusters <= 0:
        parser.error("--n, --dim, and --clusters must be positive")
    if args.clusters > args.n:
        parser.error("--clusters cannot exceed --n")
    if args.sigma < 0.0 or args.eps < 0.0:
        parser.error("--sigma and --eps must be nonnegative")
    if args.epochs <= 0 or args.finisher_passes <= 0 or args.timeout <= 0:
        parser.error("--epochs, --finisher-passes, and --timeout must be positive")

    seeds = [args.seed] if args.seed is not None else args.seeds
    solver_seeds = (
        [args.solver_seed] if args.solver_seed is not None else args.solver_seeds
    )
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("workload seeds must be between 0 and 2^32-1")
    if any(seed < 0 or seed >= 2**64 for seed in solver_seeds):
        parser.error("solver seeds must be between 0 and 2^64-1")
    results = [
        run_one(
            args.solver, seed, solver_seeds, args.clusters, args.dim,
            args.sigma, args.eps, args.n, args.generator, args.epochs,
            args.finisher_passes, args.timeout, args.require_separated,
        )
        for seed in seeds
    ]
    summary = summarize_results(results)

    if args.json:
        output = results[0] if len(results) == 1 else {
            "runs": results, "summary": summary,
        }
        print(json.dumps(output, sort_keys=True))
        return

    print("FUTCache minimum-representative benchmark (hard = full coverage)")
    print(
        f"  generator={args.generator} points={args.n} clusters={args.clusters} "
        f"dim={args.dim} eps={args.eps} "
        f"separated={args.require_separated} "
        f"baseline={results[0]['greedy_baseline']}"
    )
    print()
    print(" seed  greedy  nitrosat  saved  reduction  coverage  feasible  wall")
    for result in results:
        saved = "-" if result["representatives_saved"] is None else str(
            result["representatives_saved"]
        )
        reduction = "-" if result["reduction_fraction"] is None else (
            f"{result['reduction_fraction']:.1%}"
        )
        print(
            f" {result['workload_seed']:>4}  "
            f"{result['greedy_representatives']:>6}  "
            f"{result['nitrosat_representatives']:>8}  "
            f"{saved:>5}  {reduction:>9}  "
            f"{result['coverage']:>4}/{result['points']:<4}  "
            f"{str(result['feasible']):>8}  "
            f"{result['nitrosat_total_wall_time_s']:.3f}s"
        )
    if len(results) > 1:
        print()
        raw_reduction = summary["nitrosat_reduction_fraction"]
        raw_text = "n/a" if raw_reduction is None else f"{raw_reduction:.1%}"
        print(
            f" summary: wins/ties/losses={summary['wins']}/"
            f"{summary['ties']}/{summary['losses']}, "
            f"raw reduction={raw_text}, "
            f"verified hybrid reduction={summary['hybrid_reduction_fraction']:.1%}, "
            f"packing-compatible NitroSAT runs="
            f"{summary['nitrosat_packing_compatible_runs']}/"
            f"{summary['feasible_runs']}"
        )


if __name__ == "__main__":
    main()

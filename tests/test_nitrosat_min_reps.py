import json
import os
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import bench_nitrosat_min_reps as bench  # noqa: E402


class WorkloadTests(unittest.TestCase):
    def test_balanced_generator_is_exact_and_deterministic(self):
        first, eps = bench.generate_workload(7, 5, 2, 0.03, 0.05, 23)
        second, _ = bench.generate_workload(7, 5, 2, 0.03, 0.05, 23)
        self.assertEqual(first.shape, (23, 2))
        self.assertEqual(eps, 0.05)
        np.testing.assert_array_equal(first, second)

    def test_cluster_count_changes_balanced_workload(self):
        few, _ = bench.generate_workload(7, 2, 2, 0.03, 0.05, 20)
        many, _ = bench.generate_workload(7, 5, 2, 0.03, 0.05, 20)
        self.assertFalse(np.array_equal(few, many))

    def test_legacy_generator_still_returns_exact_target(self):
        points, _ = bench.generate_workload(
            1, 10, 3, 0.03, 0.05, 37, generator="legacy"
        )
        self.assertEqual(points.shape, (37, 3))


class FormulationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ])

    def test_greedy_set_cover(self):
        selected = bench.greedy_set_cover(self.matrix)
        self.assertEqual(len(selected), 2)
        self.assertEqual(bench.coverage_of(selected, self.matrix), 4)

    def test_separation_violations(self):
        symmetric = np.array([
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ])
        self.assertEqual(bench.separation_violations([0, 1], symmetric), 1)
        self.assertEqual(bench.separation_violations([0, 2], symmetric), 0)

    def test_separated_formulation_adds_pairwise_hard_clause(self):
        symmetric = np.array([
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            path = stream.name
        try:
            variables, clauses, top = bench.encode_min_reps_wcnf(
                symmetric, path, require_separated=True
            )
            with open(path, "r", encoding="ascii") as stream:
                lines = stream.read().splitlines()
        finally:
            os.unlink(path)
        self.assertEqual((variables, clauses, top), (3, 7, 4))
        self.assertIn("4 -1 -2 0", lines)
        self.assertEqual(len(bench.greedy_packing(symmetric)), 2)

    def test_min_reps_wcnf(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            path = stream.name
        try:
            variables, clauses, top = bench.encode_min_reps_wcnf(
                self.matrix, path
            )
            with open(path, "r", encoding="ascii") as stream:
                lines = stream.read().splitlines()
        finally:
            os.unlink(path)
        self.assertEqual((variables, clauses, top), (3, 7, 4))
        self.assertEqual(lines[0], "p wcnf 3 7 4")
        self.assertEqual(lines[1:5], [
            "4 1 0", "4 1 2 0", "4 2 3 0", "4 3 0",
        ])
        self.assertEqual(lines[5:], ["1 -1 0", "1 -2 0", "1 -3 0"])

    def test_exact_result_verification(self):
        result = {
            "variables": 3,
            "clauses": 7,
            "hard_unsatisfied": 0,
            "soft_unsatisfied": 2,
            "soft_cost": 2,
            "feasible": True,
        }
        coverage, feasible = bench.verify_result(
            result, [0, 2], self.matrix, 3, 7
        )
        self.assertEqual(coverage, 4)
        self.assertTrue(feasible)

    def test_verifier_rejects_false_solver_claim(self):
        result = {
            "variables": 3,
            "clauses": 7,
            "hard_unsatisfied": 0,
            "soft_unsatisfied": 1,
            "soft_cost": 1,
            "feasible": True,
        }
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            bench.verify_result(result, [0], self.matrix, 3, 7)


class SolverProtocolTests(unittest.TestCase):
    def test_partial_exit_json_is_parseable(self):
        payload = {
            "solver": "NitroSAT V3",
            "feasible": False,
            "variables": 3,
            "clauses": 7,
            "hard_unsatisfied": 1,
            "soft_unsatisfied": 2,
            "soft_cost": 2,
            "solve_ms": 1.25,
        }
        parsed = bench.parse_solver_json(json.dumps(payload), "", 1)
        self.assertEqual(parsed["exit_code"], 1)

    def test_solution_parser(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            stream.write("1 -2 3 0\n")
            path = stream.name
        try:
            self.assertEqual(bench.parse_solution(path, 3), [0, 2])
        finally:
            os.unlink(path)

    def test_solution_parser_rejects_incomplete_assignment(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            stream.write("1 -2 0\n")
            path = stream.name
        try:
            with self.assertRaisesRegex(RuntimeError, "assigns 2 of 3"):
                bench.parse_solution(path, 3)
        finally:
            os.unlink(path)

    @unittest.skipUnless(
        os.path.isfile(bench.DEFAULT_SOLVER) and
        os.access(bench.DEFAULT_SOLVER, os.X_OK),
        "vendored NitroSAT V3 binary is not built",
    )
    def test_legacy_seed_one_integration(self):
        result = bench.run_one(
            bench.DEFAULT_SOLVER, 1, [42], 40, 2, 0.03, 0.05, 200,
            "legacy", 100, 2000, 30.0, False,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["coverage"], 200)
        self.assertEqual(result["greedy_representatives"], 18)
        self.assertEqual(result["nitrosat_representatives"], 15)
        self.assertEqual(result["hard_unsatisfied"], 0)
        self.assertEqual(result["soft_cost"], 15)


if __name__ == "__main__":
    unittest.main()

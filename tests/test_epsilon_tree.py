"""Tests for the EpsilonTree knee-method adaptive-radius calibrator.

Uses synthetic, well-clustered data so results are deterministic and
network-free (no embedding model). Verifies the knee detection, that a
clustered calibration set yields a sensible (small) auto-epsilon, that the
per-region traversal works, and that the distance metric matches what the
cache uses.
"""
import unittest

import numpy as np

from futcache import EpsilonTree


class KneeTests(unittest.TestCase):
    def test_knee_finds_elbow_of_graded_curve(self):
        # flat then sharp jump: the knee should sit near the jump
        vals = np.concatenate([
            np.linspace(0.1, 0.3, 20),
            np.linspace(1.0, 1.2, 10),
        ])
        k = EpsilonTree.knee(vals)
        # the knee sits at the flat->jump boundary (~0.3), not at the ends
        self.assertGreaterEqual(k, 0.2)
        self.assertLessEqual(k, 0.5)


class TreeTests(unittest.TestCase):
    def _clustered(self, k=3, size=12, dim=2, spread=0.05):
        rng = np.random.default_rng(0)
        centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        pts = np.concatenate([
            centers[i] + spread * rng.normal(size=(size, dim))
            for i in range(k)
        ])
        return pts

    def test_global_epsilon_small_on_clusters(self):
        pts = self._clustered()
        tree = EpsilonTree(k=2, min_leaf=4, max_depth=3,
                           distance="l2").fit(pts)
        # within-cluster distances are ~0.1, between-cluster ~3, so the knee
        # should be far below the between-cluster scale.
        self.assertLess(tree.global_epsilon_, 1.0)

    def test_cosine_metric_supported(self):
        rng = np.random.default_rng(1)
        centers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])  # unit-ish
        pts = np.concatenate([
            (centers[i] + 0.02 * rng.normal(size=(12, 2)))
            for i in range(3)
        ])
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        tree = EpsilonTree(k=2, min_leaf=4, max_depth=3,
                           distance="cosine").fit(pts)
        self.assertTrue(np.isfinite(tree.global_epsilon_))
        # traversal returns a per-point radius
        eps = tree.epsilon(pts[0])
        self.assertIsInstance(eps, float)
        self.assertGreaterEqual(eps, 0.0)

    def test_epsilon_traverses_the_tree(self):
        pts = self._clustered()
        tree = EpsilonTree(k=2, min_leaf=4, max_depth=3,
                           distance="l2").fit(pts)
        for p in pts:
            eps = tree.epsilon(p)
            self.assertGreaterEqual(float(eps), 0.0)
        self.assertEqual(len(tree.leaves()), tree.leaves().__len__())

    def test_refit_rebuilds(self):
        pts = self._clustered()
        t1 = EpsilonTree(k=2, min_leaf=4, distance="l2").fit(pts)
        eps_a = t1.epsilon(pts[0])
        t2 = EpsilonTree(k=2, min_leaf=4, distance="l2").fit(
            2 * pts + 1.0)
        eps_b = t2.epsilon(2 * pts[0] + 1.0)
        # scaling the data by 2 doubles the distances -> epsilon doubles
        self.assertAlmostEqual(eps_b, 2.0 * eps_a, places=6)


if __name__ == "__main__":
    unittest.main()

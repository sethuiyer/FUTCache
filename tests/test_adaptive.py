import math
import unittest

import numpy as np

from futcache import (
    AdaptiveRadiusController,
    AdaptiveRadiusPolicy,
    CompactIsolationForest,
    PackCache,
    halton_sequence,
    halton_trials,
    poincare_distance,
    poincare_embed,
)


class AdaptiveUtilityTests(unittest.TestCase):
    def test_halton_uses_prime_bases(self):
        sequence = halton_sequence(3, 2)
        np.testing.assert_allclose(sequence, [
            [0.5, 1.0 / 3.0],
            [0.25, 2.0 / 3.0],
            [0.75, 1.0 / 9.0],
        ])
        trials = list(halton_trials({
            "epsilon_0": (0.0, 1.0),
            "gamma": (0.0, 4.0),
            "lambda": (0.0, 6.0),
        }, 4))
        self.assertEqual(len(trials), 4)
        self.assertEqual(tuple(trials[0]), ("epsilon_0", "gamma", "lambda"))

    def test_poincare_projection_and_policy(self):
        points = poincare_embed(
            [[1.0, 0.0], [0.0, 2.0]], [0.25, 0.75], max_norm=0.8)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), [0.2, 0.6])
        self.assertAlmostEqual(
            poincare_distance([0.0, 0.0], [0.5, 0.0]), math.log(3.0))

        policy = AdaptiveRadiusPolicy(
            base_radius=0.8,
            gamma=1.0,
            isolation_weight=math.log(2.0),
            margin_safety=0.5,
        )
        radii = policy.radii(points, [0.0, 1.0])
        expected = [0.8 * (1.0 - 0.2**2),
                    0.8 * (1.0 - 0.6**2) * 0.5]
        np.testing.assert_allclose(radii, expected)
        capped = policy.radii(points, [0.0, 0.0],
                              incompatible_distances=[0.2, 0.4])
        np.testing.assert_allclose(capped, [0.1, 0.2])

    def test_compact_isolation_forest(self):
        rng = np.random.default_rng(7)
        cluster = rng.normal(0.0, 0.05, size=(512, 6))
        forest = CompactIsolationForest(
            n_estimators=48, max_samples=128, random_state=11).fit(cluster)
        familiar = forest.score_samples(cluster[:32])
        outlier = forest.score_samples(np.full(6, 4.0))
        self.assertGreater(outlier, float(np.median(familiar)))
        self.assertLessEqual(
            forest.memory_bytes,
            48 * (2 * 128 - 1) * 5 * np.dtype(np.int32).itemsize,
        )

        policy = AdaptiveRadiusPolicy(base_radius=0.5, gamma=0.0,
                                      isolation_weight=2.0)
        controller = AdaptiveRadiusController(policy, forest)
        familiar_radius = controller.radius(np.zeros(6))
        outlier_radius = controller.radius(np.full(6, 0.3))
        self.assertLess(outlier_radius, familiar_radius)


class AdaptiveBindingTests(unittest.TestCase):
    def test_exact_variable_ball_lookup(self):
        cache = PackCache(
            dimension=1,
            epsilon=0.1,
            distance="l2",
            domain_min=-1.0,
            domain_max=2.0,
            backend="vptree",
        )
        first = cache.observe([0.0], payload=b"narrow", radius=0.05)
        second = cache.observe([0.3], payload=b"broad", radius=0.4)
        self.assertTrue(first.inserted)
        self.assertTrue(second.inserted)
        hit = cache.query([0.08])
        self.assertFalse(hit.is_novel)
        self.assertEqual(hit.representative_id, 1)
        self.assertAlmostEqual(hit.distance, 0.22)
        self.assertEqual(cache.get_payload(hit.representative_id), b"broad")
        np.testing.assert_allclose(cache.copy_radii(), [0.05, 0.4])

    def test_poincare_binding_rejects_boundary(self):
        cache = PackCache(
            dimension=2,
            epsilon=0.2,
            distance="poincare",
            backend="vptree",
        )
        self.assertTrue(cache.observe([0.0, 0.0]).is_novel)
        with self.assertRaises(ValueError):
            cache.query([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

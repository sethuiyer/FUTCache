"""Adaptive resolution for semantic caches.

The engine owns exact variable-radius ball lookup. This module supplies the
calibration layer proposed for Adaptive FUTCache:

    epsilon(x) = epsilon_0 * (1 - ||z(x)||^2)^gamma
                 * exp(-lambda * isolation_score(x))

An optional nearest-incompatible margin can cap the result. Prime-base Halton
trials explore the small parameter space without the memory and clustering of
a Cartesian grid. ``CompactIsolationForest`` retains only flat, float32/int32
tree arrays; it does not keep the fitted embedding matrix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Final

import numpy as np


_EULER_GAMMA: Final = 0.5772156649015329


def _as_matrix(points, *, name: str = "points") -> tuple[np.ndarray, bool]:
    array = np.asarray(points, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n, d) or (d,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array), was_vector


def poincare_embed(
    directions,
    specificity,
    *,
    min_norm: float = 0.0,
    max_norm: float = 0.95,
):
    """Place Euclidean directions inside the Poincare ball.

    ``specificity`` is a scalar or one value per row in ``[0, 1]``. Generic
    concepts map near ``min_norm``; specialised concepts map near
    ``max_norm``. This is a deterministic adapter for an externally estimated
    hierarchy signal, not a replacement for learning hyperbolic embeddings.
    Zero direction vectors remain at the origin.
    """
    if not (math.isfinite(min_norm) and math.isfinite(max_norm) and
            0.0 <= min_norm <= max_norm < 1.0):
        raise ValueError("norm bounds must satisfy 0 <= min_norm <= max_norm < 1")
    matrix, was_vector = _as_matrix(directions, name="directions")
    values = np.asarray(specificity, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(matrix.shape[0], float(values), dtype=np.float64)
    if values.shape != (matrix.shape[0],):
        raise ValueError("specificity must be scalar or have one value per row")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("specificity values must lie in [0, 1]")

    norms = np.linalg.norm(matrix, axis=1)
    unit = np.zeros_like(matrix)
    nonzero = norms > 0.0
    unit[nonzero] = matrix[nonzero] / norms[nonzero, None]
    radial = min_norm + (max_norm - min_norm) * values
    embedded = unit * radial[:, None]
    return embedded[0] if was_vector else embedded


def poincare_distance(left, right) -> float:
    """Stable curvature-minus-one Poincare-ball distance."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or a.size == 0:
        raise ValueError("left and right must be equal-length vectors")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("points must contain only finite values")
    norm_a_squared = float(a @ a)
    norm_b_squared = float(b @ b)
    if norm_a_squared >= 1.0 or norm_b_squared >= 1.0:
        raise ValueError("Poincare points must have norm strictly below 1")
    denominator = math.sqrt(
        (1.0 - norm_a_squared) * (1.0 - norm_b_squared))
    return 2.0 * math.asinh(float(np.linalg.norm(a - b)) / denominator)


@dataclass(frozen=True, slots=True)
class AdaptiveRadiusPolicy:
    """Poincare- and isolation-calibrated representative radius."""

    base_radius: float
    gamma: float = 1.0
    isolation_weight: float = 1.0
    min_radius: float = 0.0
    max_radius: float | None = None
    margin_safety: float = 0.5

    def __post_init__(self) -> None:
        values = {
            "base_radius": self.base_radius,
            "gamma": self.gamma,
            "isolation_weight": self.isolation_weight,
            "min_radius": self.min_radius,
            "margin_safety": self.margin_safety,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_radius is not None:
            if not math.isfinite(self.max_radius) or self.max_radius < self.min_radius:
                raise ValueError("max_radius must be finite and >= min_radius")
        if self.margin_safety > 1.0:
            raise ValueError("margin_safety must lie in [0, 1]")

    def radii(
        self,
        poincare_points,
        anomaly_scores=0.0,
        *,
        incompatible_distances=None,
    ) -> np.ndarray | float:
        """Compute radii without allocating pairwise distance matrices."""
        points, was_vector = _as_matrix(
            poincare_points, name="poincare_points")
        squared_norms = np.einsum("ij,ij->i", points, points)
        if np.any(squared_norms >= 1.0):
            raise ValueError("Poincare points must have norm strictly below 1")

        scores = np.asarray(anomaly_scores, dtype=np.float64)
        if scores.ndim == 0:
            scores = np.full(points.shape[0], float(scores), dtype=np.float64)
        if scores.shape != (points.shape[0],) or not np.all(np.isfinite(scores)):
            raise ValueError("anomaly_scores must be scalar or one finite value per row")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("anomaly_scores must lie in [0, 1]")

        radial_factor = np.power(
            np.maximum(0.0, 1.0 - squared_norms), self.gamma)
        radii = self.base_radius * radial_factor
        radii *= np.exp(-self.isolation_weight * scores)

        upper = self.max_radius if self.max_radius is not None else np.inf
        radii = np.clip(radii, self.min_radius, upper)

        if incompatible_distances is not None:
            margins = np.asarray(incompatible_distances, dtype=np.float64)
            if margins.ndim == 0:
                margins = np.full(points.shape[0], float(margins), dtype=np.float64)
            if (margins.shape != (points.shape[0],) or
                    not np.all(np.isfinite(margins)) or np.any(margins < 0.0)):
                raise ValueError(
                    "incompatible_distances must be scalar or one finite "
                    "non-negative value per row")
            radii = np.minimum(radii, self.margin_safety * margins)
        return float(radii[0]) if was_vector else radii


def _average_unsuccessful_path_length(sample_count):
    count = np.asarray(sample_count, dtype=np.float64)
    result = np.zeros_like(count)
    result[count == 2.0] = 1.0
    mask = count > 2.0
    selected = count[mask]
    result[mask] = (
        2.0 * (np.log(selected - 1.0) + _EULER_GAMMA)
        - 2.0 * (selected - 1.0) / selected
    )
    return result


@dataclass(frozen=True, slots=True)
class _IsolationTree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf_adjustment: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return sum(array.nbytes for array in (
            self.feature, self.threshold, self.left, self.right,
            self.leaf_adjustment))


class CompactIsolationForest:
    """Small dependency-free Isolation Forest for threshold calibration.

    Trees use flat int32/float32 arrays and retain no training vectors. The
    default 64 trees with 256-sample subsampling generally consume well under
    one MiB, independent of the full calibration-set size.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 64,
        max_samples: int = 256,
        max_feature_trials: int = 8,
        random_state: int | None = 0,
    ) -> None:
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")
        if not isinstance(max_feature_trials, int) or max_feature_trials <= 0:
            raise ValueError("max_feature_trials must be a positive integer")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_feature_trials = max_feature_trials
        self.random_state = random_state
        self._trees: tuple[_IsolationTree, ...] = ()
        self._normalizer = 0.0
        self.n_features_in_: int | None = None

    def fit(self, points) -> "CompactIsolationForest":
        matrix, _ = _as_matrix(points)
        if matrix.shape[0] < 2:
            raise ValueError("at least two calibration points are required")
        rng = np.random.default_rng(self.random_state)
        sample_count = min(self.max_samples, matrix.shape[0])
        max_depth = math.ceil(math.log2(sample_count))
        trees: list[_IsolationTree] = []

        for _ in range(self.n_estimators):
            if sample_count == matrix.shape[0]:
                indices = np.arange(sample_count, dtype=np.int64)
            else:
                indices = rng.choice(
                    matrix.shape[0], size=sample_count,
                    replace=False, shuffle=False)
            feature: list[int] = []
            threshold: list[float] = []
            left: list[int] = []
            right: list[int] = []
            leaf_size: list[int] = []

            def build_node(rows: np.ndarray, depth: int) -> int:
                node = len(feature)
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                leaf_size.append(int(rows.size))
                if depth >= max_depth or rows.size <= 1:
                    return node

                split_feature = -1
                low = high = 0.0
                for _trial in range(self.max_feature_trials):
                    candidate = int(rng.integers(matrix.shape[1]))
                    values = matrix[rows, candidate]
                    candidate_low = float(np.min(values))
                    candidate_high = float(np.max(values))
                    if candidate_low < candidate_high:
                        split_feature = candidate
                        low, high = candidate_low, candidate_high
                        break
                if split_feature < 0:
                    return node

                split = float(rng.uniform(low, high))
                values = matrix[rows, split_feature]
                goes_left = values < split
                if not np.any(goes_left) or np.all(goes_left):
                    return node
                feature[node] = split_feature
                threshold[node] = split
                leaf_size[node] = 0
                left[node] = build_node(rows[goes_left], depth + 1)
                right[node] = build_node(rows[~goes_left], depth + 1)
                return node

            build_node(indices, 0)
            sizes = np.asarray(leaf_size, dtype=np.int32)
            adjustment = _average_unsuccessful_path_length(sizes).astype(
                np.float32)
            trees.append(_IsolationTree(
                feature=np.asarray(feature, dtype=np.int32),
                threshold=np.asarray(threshold, dtype=np.float32),
                left=np.asarray(left, dtype=np.int32),
                right=np.asarray(right, dtype=np.int32),
                leaf_adjustment=adjustment,
            ))

        self._trees = tuple(trees)
        self._normalizer = float(_average_unsuccessful_path_length(
            np.asarray([sample_count], dtype=np.int32))[0])
        self.n_features_in_ = matrix.shape[1]
        return self

    def _check_fitted_points(self, points) -> tuple[np.ndarray, bool]:
        if not self._trees or self.n_features_in_ is None:
            raise RuntimeError("CompactIsolationForest must be fitted first")
        matrix, was_vector = _as_matrix(points)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"points have {matrix.shape[1]} features; expected "
                f"{self.n_features_in_}")
        return matrix, was_vector

    def score_samples(self, points) -> np.ndarray | float:
        """Return Isolation Forest anomaly scores in ``(0, 1]``."""
        matrix, was_vector = self._check_fitted_points(points)
        path_sum = np.zeros(matrix.shape[0], dtype=np.float64)
        for tree in self._trees:
            nodes = np.zeros(matrix.shape[0], dtype=np.int32)
            depths = np.zeros(matrix.shape[0], dtype=np.float64)
            active = np.ones(matrix.shape[0], dtype=bool)
            while np.any(active):
                rows = np.flatnonzero(active)
                current = nodes[rows]
                leaf = tree.feature[current] < 0
                if np.any(leaf):
                    leaf_rows = rows[leaf]
                    leaf_nodes = current[leaf]
                    depths[leaf_rows] += tree.leaf_adjustment[leaf_nodes]
                    active[leaf_rows] = False
                branch_rows = rows[~leaf]
                if branch_rows.size:
                    branch_nodes = current[~leaf]
                    branch_features = tree.feature[branch_nodes]
                    branch_values = matrix[branch_rows, branch_features]
                    choose_left = branch_values < tree.threshold[branch_nodes]
                    nodes[branch_rows] = np.where(
                        choose_left,
                        tree.left[branch_nodes],
                        tree.right[branch_nodes],
                    )
                    depths[branch_rows] += 1.0
            path_sum += depths
        mean_path = path_sum / len(self._trees)
        scores = np.exp2(-mean_path / self._normalizer)
        return float(scores[0]) if was_vector else scores

    @property
    def memory_bytes(self) -> int:
        """Exact bytes occupied by fitted NumPy tree buffers."""
        return sum(tree.memory_bytes for tree in self._trees)


@dataclass(slots=True)
class AdaptiveRadiusController:
    """Combine a radius policy with an optional fitted isolation model."""

    policy: AdaptiveRadiusPolicy
    isolation_forest: CompactIsolationForest | None = None

    def radii(self, poincare_points, *, incompatible_distances=None):
        scores = (0.0 if self.isolation_forest is None else
                  self.isolation_forest.score_samples(poincare_points))
        return self.policy.radii(
            poincare_points, scores,
            incompatible_distances=incompatible_distances)

    def radius(self, poincare_point, *, incompatible_distance=None) -> float:
        return float(self.radii(
            poincare_point,
            incompatible_distances=incompatible_distance))


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        limit = math.isqrt(candidate)
        if all(candidate % prime for prime in primes if prime <= limit):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _radical_inverse(index: int, base: int) -> float:
    inverse = 1.0 / base
    factor = inverse
    value = 0.0
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor *= inverse
    return value


def halton_sequence(
    size: int,
    dimension: int,
    *,
    start_index: int = 1,
) -> np.ndarray:
    """Return a prime-base low-discrepancy sequence in ``[0, 1)^d``."""
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(start_index, int) or start_index <= 0:
        raise ValueError("start_index must be a positive integer")
    bases = _first_primes(dimension)
    sequence = np.empty((size, dimension), dtype=np.float64)
    for row, index in enumerate(range(start_index, start_index + size)):
        for column, base in enumerate(bases):
            sequence[row, column] = _radical_inverse(index, base)
    return sequence


def halton_trials(
    bounds: Mapping[str, Sequence[float]],
    count: int,
    *,
    start_index: int = 1,
) -> Iterator[dict[str, float]]:
    """Yield prime-base parameter trials without constructing a grid."""
    if not bounds:
        raise ValueError("bounds must not be empty")
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not isinstance(start_index, int) or start_index <= 0:
        raise ValueError("start_index must be a positive integer")
    names = tuple(bounds)
    ranges: list[tuple[float, float]] = []
    for name in names:
        pair = tuple(bounds[name])
        if len(pair) != 2:
            raise ValueError(f"bounds[{name!r}] must contain (low, high)")
        low, high = float(pair[0]), float(pair[1])
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError(f"invalid bounds for {name!r}")
        ranges.append((low, high))
    bases = _first_primes(len(names))
    for index in range(start_index, start_index + count):
        yield {
            name: low + _radical_inverse(index, base) * (high - low)
            for name, base, (low, high) in zip(names, bases, ranges)
        }


__all__ = [
    "AdaptiveRadiusController",
    "AdaptiveRadiusPolicy",
    "CompactIsolationForest",
    "halton_sequence",
    "halton_trials",
    "poincare_distance",
    "poincare_embed",
]

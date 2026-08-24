"""Density-aware adaptive epsilon via a region tree + knee method.

The engine's single global ``epsilon`` is a blunt instrument: a threshold
that "just works" everywhere is either too loose in dense regions or too
tight in sparse ones. This module learns an ``epsilon`` that *varies by
region*:

  1. Build a binary space-partition tree (recursive median split) over a
     calibration set of points (the reference density).
  2. At each leaf, estimate a local ``epsilon`` as the **knee** of the
     k-th-nearest-neighbour distance curve of the points in that leaf.
     The knee of a k-distance plot is the canonical density-based clustering
     threshold (the DBSCAN heuristic): distances drop smoothly within a
     cluster and then the curve turns up sharply at the between-cluster
     boundary, and the knee sits at that turn.
  3. A query traverses the tree to its leaf and uses that leaf's knee-based
     radius, passed to ``futcache.observe_with_radius``.

The knee is detected by the standard "max distance from the chord" elbow
method. This is a calibrator, not a substitute for the engine's exact
decision: ``observe_with_radius`` still performs exact variable-ball
stabbing on the radii you hand it.

Usage:

    tree = EpsilonTree(k=3, min_leaf=6, max_depth=4).fit(ref_points)
    eps = tree.epsilon(query)          # a region-specific radius
    cpu  = PackCache(dim, 0.0, ...)    # base radius 0; adaptive per query
    res = cpu.observe(query, radius=eps)
"""

from __future__ import annotations

import math

import numpy as np


class EpsilonTree:
    """Region-adaptive epsilon learned from a calibration point set.

    Parameters
    ----------
    k:
        Order of the nearest-neighbour distance used for the k-distance
        curve (the DBSCAN-style epsilon heuristic).
    min_leaf:
        Stop splitting a node once it has at most this many points; such a
        node becomes a leaf with its own knee-based epsilon (or the global
        epsilon if it is too small for a stable knee).
    max_depth:
        Maximum binary-split depth (bounds tree size / over-fitting on the
        calibration set).
    """

    def __init__(self, k: int = 3, min_leaf: int = 6, max_depth: int = 4,
                 distance: str = "l2"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if min_leaf < 3:
            raise ValueError("min_leaf must be >= 3")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if distance not in ("l2", "l1", "linf", "cosine"):
            raise ValueError("distance must be l2, l1, linf, or cosine")
        self.k = k
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.distance = distance
        self.ref_ = None
        self.root_ = None
        self.global_epsilon_ = None
        self.knee_curve_ = None

    # ------------------------------------------------------------ distance
    def _dist_matrix(self, pts):
        """Pairwise distance matrix in the cache's metric units."""
        n = pts.shape[0]
        if self.distance == "cosine":
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            nrm = pts / norms
            sim = nrm @ nrm.T
            sim = np.clip(sim, -1.0, 1.0)
            return 1.0 - sim
        diff = pts[:, None, :] - pts[None, :, :]
        if self.distance == "l2":
            return np.sqrt(np.sum(diff * diff, axis=2))
        if self.distance == "l1":
            return np.sum(np.abs(diff), axis=2)
        return np.max(np.abs(diff), axis=2)   # linf

    # ------------------------------------------------------------ knee
    @staticmethod
    def knee(values):
        """Knee (elbow) index and value of a sorted ascending curve.

        Uses the max-distance-from-the-chord method on the normalised
        ``(index, value)`` curve, restricted to the interior so it does not
        return an endpoint.
        """
        values = np.asarray(values, dtype=np.float64)
        n = values.size
        if n < 3:
            return float(values[0]) if n else 0.0
        x = np.linspace(0.0, 1.0, n)
        span = values[-1] - values[0]
        if span < 1e-12:                 # degenerate: flat curve, no elbow
            return float(values[0])
        y = (values - values[0]) / span
        d = np.abs(y - x)                      # distance to the chord y = x
        d[0] = d[-1] = -np.inf
        i = int(np.argmax(d))
        return float(values[i])

    # ------------------------------------------------------------ k-dist
    def _k_distance_curve(self, pts):
        """Sorted k-th-nearest-neighbour distances within ``pts`` (cache units)."""
        n = pts.shape[0]
        k = self.k
        if n <= k:
            k = max(n - 1, 1)
        if n < 2:
            return None
        d = self._dist_matrix(pts)
        np.fill_diagonal(d, np.inf)
        kth = np.partition(d, k - 1, axis=1)[:, k - 1]
        return np.sort(kth)

    def _leaf_epsilon(self, idx):
        pts = self.ref_[idx]
        curve = self._k_distance_curve(pts)
        if curve is None or curve.size < 4:
            return self.global_epsilon_
        return self.knee(curve)

    # ------------------------------------------------------------ tree
    def _build(self, idx, depth):
        if len(idx) <= self.min_leaf or depth >= self.max_depth:
            return ("leaf", idx, self._leaf_epsilon(idx))
        sub = self.ref_[idx]
        spread = sub.max(axis=0) - sub.min(axis=0)
        dim = int(np.argmax(spread))
        order = idx[np.argsort(sub[:, dim])]
        mid = len(order) // 2
        split = float(self.ref_[order[mid], dim])
        left = self._build(order[:mid], depth + 1)
        right = self._build(order[mid:], depth + 1)
        return ("node", dim, split, left, right)

    def fit(self, ref_points):
        """Learn the region epsilons from ``ref_points`` (N, d)."""
        self.ref_ = np.asarray(ref_points, dtype=np.float64)
        if self.ref_.ndim != 2 or self.ref_.shape[0] < 2:
            raise ValueError("ref_points must be a (N, d) array with N >= 2")
        # global epsilon = knee of the global k-distance curve
        curve = self._k_distance_curve(self.ref_)
        self.knee_curve_ = curve
        self.global_epsilon_ = self.knee(curve) if curve is not None else 0.0
        idx = np.arange(self.ref_.shape[0])
        self.root_ = self._build(idx, 0)
        return self

    # ------------------------------------------------------------ query
    def epsilon(self, point):
        """Return the knee-based radius for ``point`` (traverse to leaf)."""
        if self.root_ is None:
            raise RuntimeError("EpsilonTree not fitted")
        p = np.asarray(point, dtype=np.float64)
        node = self.root_
        while node[0] == "node":
            _, dim, split, left, right = node
            node = left if p[dim] <= split else right
        # node is ("leaf", idx, eps)
        return node[2]

    # ------------------------------------------------------------ stats
    def leaves(self):
        """Yield (level, epsilon, point_count) for each leaf."""
        out = []

        def walk(node, level):
            if node[0] == "leaf":
                out.append((level, node[2], len(node[1])))
            else:
                walk(node[3], level + 1)
                walk(node[4], level + 1)
        if self.root_ is not None:
            walk(self.root_, 0)
        return out

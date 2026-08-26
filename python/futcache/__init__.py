"""FUTCache Python bindings.

Public API:

    from futcache import PackCache, NoveltyResult

The C cache lives in ``futcache_ext``. This module re-exports the
classes so callers do not need to know the split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .adaptive import (
    AdaptiveRadiusController,
    AdaptiveRadiusPolicy,
    CompactIsolationForest,
    halton_sequence,
    halton_trials,
    poincare_distance,
    poincare_embed,
)
from .epsilon_tree import EpsilonTree
from .futcache_ext import (
    AnchorEmbedding as _AnchorEmbeddingRaw,
    NoveltyResult as _NoveltyResultRaw,
    PackCache as _PackCacheRaw,
    PersistentNovelty as _PersistentNoveltyRaw,
    PersistentNoveltyND as _PersistentNoveltyNDRaw,
    merge_persistence_diagrams as _merge_persistence_diagrams,
    nth_prime as _nth_prime,
    select_max_coverage as _select_max_coverage,
    select_coverage as _select_coverage,
    select_evict_worst as _select_evict_worst,
)


@dataclass(frozen=True)
class NoveltyResult:
    """Outcome of a PackCache query or observe.

    Attributes:
        representative_id: slot index of the matched or new
            representative. It is -1 only for a novel non-mutating query.
            On a
            semantic HIT this is the index to pass to ``get_payload()``.
        is_novel: True when the point lies outside every existing
            representative's acceptance radius.
        distance: distance to the closest containing representative on a
            HIT, the nearest centre on a non-mutating miss, and 0.0 for a
            novel ``observe()`` (the point became its own representative).
        inserted: True when ``observe()`` added a new representative.
    """

    representative_id: int
    is_novel: bool
    distance: float
    inserted: bool


def _wrap(raw: _NoveltyResultRaw) -> NoveltyResult:
    if raw is None:
        raise RuntimeError("internal: empty NoveltyResult from extension")
    return NoveltyResult(
        representative_id=raw.representative_id,
        is_novel=raw.is_novel,
        distance=raw.distance,
        inserted=raw.inserted,
    )


class PackCache:
    """Voronoi packing novelty cache for arbitrary metric spaces.

    This wraps ``futcache_pack`` from the C library. Novelty semantics
    match the underlying C cache exactly. Ordinary ``observe()`` calls use
    ``epsilon`` for every representative; passing an adaptive ``radius``
    gives a newly inserted representative its own acceptance ball.
    The cache is exact for membership in the stored representative balls.
    Fixed-radius representative count is bounded by ``P(K, epsilon)``;
    adaptive radii retain a geometric packing bound when they have a positive
    floor, while ``max_memory_bytes`` is the unconditional physical bound.

    Payloads (LLM responses, retrieval results, etc.) are stored in a
    Python dict keyed by representative slot index. The C cache itself
    manages only novelty state. With a memory limit, ids may shift after a
    novel observation triggers FIFO eviction; the wrapper shifts payloads in
    the same operation, so treat an id as valid only until the next mutation.

    Typical RAG-cache use:

        cache = PackCache(dimension=384, epsilon=0.55,
                          distance="cosine",
                          domain_min=-1.0, domain_max=1.0)
        res = cache.observe(embedding, payload=llm_response)
        if not res.is_novel:
            return cache.get_payload(res.representative_id)

    Args:
        dimension: number of coordinates per point.
        epsilon: novelty resolution in the chosen distance units.
        distance: one of ``"linf"``, ``"l1"``, ``"l2"``, ``"cosine"``,
            ``"poincare"``.
            Default ``"linf"`` is the natural extension of the 1D
            interval-union semantics.
        domain_min, domain_max: per-coordinate bounds. Scalars are
            broadcast to a length-``dimension`` vector. Default
            ``-1.0`` / ``1.0`` for every coordinate. Inputs must lie
            inside the inclusive bounds.
        backend: nearest-neighbour index. ``"linear"`` (default) scans
            every representative (O(|R|) per observation); ``"vptree"``
            uses an exact scapegoat VP-tree with triangle-inequality
            pruning (logarithmic inserts, exact queries) — identical
            novelty semantics, faster at large representative counts.
        max_memory_bytes: hard bound for all live native allocations owned by
            this cache. Zero (default) is unlimited. At the bound, the oldest
            representative and its payload are evicted and its native
            allocation is recycled without exceeding the ceiling.
    """

    def __init__(self,
                 dimension: int,
                 epsilon: float,
                 distance: str = "linf",
                 domain_min=None,
                 domain_max=None,
                 backend: str = "linear",
                 max_memory_bytes: int = 0,
                 max_entries: int = 0,
                 ttl: float = 0.0) -> None:
        if not isinstance(max_memory_bytes, int) or max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be a non-negative integer")
        if not isinstance(max_entries, int) or max_entries < 0:
            raise ValueError("max_entries must be a non-negative integer")
        ttl = 0.0 if ttl is None else float(ttl)
        if not math.isfinite(ttl) or ttl < 0.0:
            raise ValueError("ttl must be a finite non-negative number of seconds")
        lo, hi = _broadcast_domain(dimension, domain_min, domain_max)
        self._dimension = dimension
        self._impl = _PackCacheRaw(
            dimension=dimension,
            epsilon=epsilon,
            distance=distance,
            domain_min=lo,
            domain_max=hi,
            backend=backend,
            max_memory_bytes=max_memory_bytes,
            max_entries=max_entries,
            ttl=ttl,
        )

    def query(self, point) -> NoveltyResult:
        """Return novelty of ``point`` without mutating the cache."""
        arr = _as_ndarray(point, self._dimension)
        try:
            return _wrap(self._impl.query(arr))
        except IndexError as e:
            raise ValueError(str(e)) from None

    def observe(self, point, payload=None, *, radius=None) -> NoveltyResult:
        """Atomic query + update.

        If ``point`` is novel, it is added as a new representative and
        ``payload`` (when provided) is attached to that representative.
        If ``point`` is redundant, the state is unchanged and the
        existing payload for the matched representative is left as-is.

        ``radius=None`` uses the cache's fixed ``epsilon``. A finite
        non-negative radius enables adaptive resolution for a newly inserted
        representative; existing representatives always retain their own
        radii.
        """
        arr = _as_ndarray(point, self._dimension)
        if payload is not None and not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes, bytearray, or None")
        raw_payload = bytes(payload) if payload is not None else None
        try:
            if radius is None:
                raw = self._impl.observe(arr, raw_payload)
            else:
                radius = float(radius)
                raw = self._impl.observe_with_radius(
                    arr, radius, raw_payload)
            return _wrap(raw)
        except IndexError as e:
            raise ValueError(str(e)) from None

    def get_or_compute(self, point, compute, *, radius=None):
        """Drop-in semantic answer cache.

        Returns ``(answer, res)`` where ``answer`` is a non-None payload
        (bytes). If the query is novel, or its cached payload was evicted
        (LRU) or expired (TTL), ``compute(point)`` is called and its result
        stored for that representative; otherwise the cached payload is
        returned without calling ``compute``. This is the primitive that
        lets you skip an LLM/expensive call on a semantically-redundant
        query.

        ``compute`` must return ``bytes``, ``bytearray``, or ``str`` (str is
        UTF-8 encoded). ``res`` is a ``NoveltyResult``; note that a cache
        *miss through expiry* still reports ``is_novel=False`` because the
        geometry is redundant even though the answer is recomputed.
        """
        res = self.observe(point, payload=None, radius=radius)
        slot = res.representative_id
        if slot < 0:
            raise RuntimeError("observe returned no representative slot")
        answer = self.get_payload(slot)
        if answer is None:
            value = compute(point)
            if value is None:
                raise ValueError("compute() returned None")
            if isinstance(value, str):
                value = value.encode()
            answer = bytes(value)
            self.set_payload(slot, answer)
        return answer, res

    def payload_count(self) -> int:
        """Number of payloads currently stored (excluding evicted/expired)."""
        return self._impl.payload_count()

    def purge(self) -> int:
        """Drop all expired payloads; returns the number removed."""
        return self._impl.purge()

    def get_payload(self, representative_id: int):
        """Return the stored payload for ``representative_id``, or None."""
        if not isinstance(representative_id, int):
            raise TypeError("representative_id must be an int")
        return self._impl.get_payload(representative_id)

    def set_payload(self, representative_id: int, payload) -> None:
        """Attach or replace the payload for ``representative_id``.

        Pass ``None`` to drop a payload.
        """
        if not isinstance(representative_id, int):
            raise TypeError("representative_id must be an int")
        if payload is not None and not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes, bytearray, or None")
        raw_payload = bytes(payload) if payload is not None else None
        self._impl.set_payload(representative_id, raw_payload)

    def __len__(self) -> int:
        return self._impl.__len__()

    def __repr__(self) -> str:
        return (
            f"PackCache(representatives={len(self)}, "
            f"peak={self.peak_count()}, "
            f"observations={self.observations()}, "
            f"novel={self.novel_observations()}, "
            f"evictions={self.evictions()}, "
            f"memory_bytes={self.memory_bytes()})"
        )

    @property
    def representative_count(self) -> int:
        return len(self)

    def peak_count(self) -> int:
        return self._impl.peak_count()

    def memory_bytes(self) -> int:
        return self._impl.memory_bytes()

    def observations(self) -> int:
        return self._impl.observations()

    def novel_observations(self) -> int:
        return self._impl.novel_observations()

    def evictions(self) -> int:
        return self._impl.evictions()

    def peak_memory_bytes(self) -> int:
        return self._impl.peak_memory_bytes()

    def memory_limit_bytes(self) -> int:
        return self._impl.memory_limit_bytes()

    def copy_representatives(self):
        """Return a (N, dimension) numpy ndarray of current reps."""
        import numpy as np
        rows = self._impl.copy_representatives()
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def copy_radii(self):
        """Return representative radii in the same slot order as vectors."""
        import numpy as np
        return np.asarray(self._impl.copy_radii(), dtype=np.float64)

    def clear(self) -> None:
        """Empty the representative set and drop all payloads."""
        self._impl.clear()

    def evict_w1(self) -> int:
        """W1-optimal eviction: remove the rep with the smallest
        nearest-neighbour distance (the "most crowded" rep).

        Returns the slot index that was evicted.
        Payloads with ids above the evicted slot shift down by one.
        """
        return self._impl.evict_w1()

    def rep_importance(self):
        """Per-rep nearest-neighbour distance (lower = more redundant).

        Returns a numpy array of shape (N,). A lower value means the
        rep is closer to its nearest neighbour and thus a better
        eviction candidate under the W1 policy.
        """
        import numpy as np
        return np.asarray(self._impl.rep_importance(), dtype=np.float64)

    @staticmethod
    def version() -> str:
        return (f"{_PackCacheRaw.version_major()}."
                f"{_PackCacheRaw.version_minor()}."
                f"{_PackCacheRaw.version_patch()}")


def _broadcast_domain(dimension: int, lo, hi):
    """Return (lo_array, hi_array) as numpy ndarrays of length dimension.

    Scalars broadcast to the full length. Mismatched lengths raise.
    Domain is required to be finite with min < max per coordinate.
    """
    import numpy as np
    if dimension <= 0:
        raise ValueError("dimension must be >= 1")

    def _expand(name: str, value, default: float):
        if value is None:
            return np.full(dimension, default, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(dimension, float(arr), dtype=np.float64)
        if arr.shape != (dimension,):
            raise ValueError(
                f"{name} must be a scalar or length-{dimension} array, "
                f"got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr

    lo_arr = _expand("domain_min", lo, -1.0)
    hi_arr = _expand("domain_max", hi, 1.0)
    if np.any(hi_arr <= lo_arr):
        raise ValueError("domain_max must be strictly greater than domain_min per coordinate")
    return lo_arr, hi_arr


def _as_ndarray(point, dimension: int):
    """Coerce ``point`` to a contiguous, writable 1-D ndarray of length dimension."""
    import numpy as np
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise ValueError(
            f"point must be a 1-D array of length {dimension}, got shape {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


__all__ = [
    "AdaptiveRadiusController",
    "AdaptiveRadiusPolicy",
    "AnchorEmbedding",
    "CompactIsolationForest",
    "EpsilonTree",
    "NoveltyResult",
    "PackCache",
    "PersistentNovelty",
    "PersistentNoveltyND",
    "halton_sequence",
    "halton_trials",
    "merge_persistence_diagrams",
    "nth_prime",
    "poincare_distance",
    "poincare_embed",
    "select_coverage",
    "select_evict_worst",
    "select_max_coverage",
    "__version__",
]
__version__ = PackCache.version()


def nth_prime(i: int) -> int:
    """Return the i-th prime (0-indexed). p_0=2, p_1=3, p_2=5, ..."""
    return _nth_prime(i)


def merge_persistence_diagrams(diagram_a: list, diagram_b: list) -> list:
    """CRDT merge: union of two persistence diagrams (idempotent, commutative)."""
    return _merge_persistence_diagrams(diagram_a, diagram_b)


class PersistentNovelty:
    """1-D persistent novelty engine (Design Sketch 01).

    Maintains a merge tree (single-linkage dendrogram) over observed 1-D
    points. Supports scale-resolved novelty queries, prime-tagged
    persistence diagrams, CRDT merge, and the Selberg zeta function.

    Typical use:

        eng = PersistentNovelty()
        eng.observe(0.5)
        eng.observe(0.8)
        eng.is_novel_at(0.55, 0.1)   # False (within 0.1 of 0.5)
        eng.novelty_spectrum(0.6)     # [0.0, 0.1]  (novel up to t=0.1)
        eng.copy_diagram()            # list of feature dicts
        eng.selberg_zeta(2.0)         # Selberg zeta at s=2
    """

    def __init__(self):
        self._impl = _PersistentNoveltyRaw()

    def observe(self, x: float) -> None:
        """Add a 1-D observation point."""
        self._impl.observe(x)

    def is_novel_at(self, x: float, t: float) -> bool:
        """True iff x is outside U_t(H) — novel at scale t."""
        return self._impl.is_novel_at(x, t)

    def novelty_spectrum(self, x: float) -> list:
        """Return [0, t_max] if x is novel, or [] if x is already observed."""
        raw = self._impl.novelty_spectrum(x)
        if not raw:
            return []
        # raw = [0.0, t_max] pairs; return as list of (lo, hi) tuples
        return list(zip(raw[0::2], raw[1::2]))

    def copy_diagram(self) -> list:
        """Return the prime-tagged persistence diagram as a list of dicts.

        Each dict has: birth, death, birth_prime, death_prime,
        birth_value, death_value, persistence.
        """
        raw = self._impl.copy_diagram()
        keys = ["birth", "death", "birth_prime", "death_prime",
                 "birth_value", "death_value", "persistence"]
        return [dict(zip(keys, row)) for row in raw]

    def merge(self, other: "PersistentNovelty") -> list:
        """CRDT merge: idempotent, commutative union of two diagrams.

        Returns list of dicts with same structure as copy_diagram().
        """
        raw = self._impl.merge(other._impl)
        keys = ["birth", "death", "birth_prime", "death_prime",
                 "birth_value", "death_value", "persistence"]
        return [dict(zip(keys, row)) for row in raw]

    def selberg_zeta(self, s: float) -> float:
        """Selberg zeta function over prime-birth features."""
        return self._impl.selberg_zeta(s)

    def prime_cycle_count(self, tau: float = 0.0) -> int:
        """Count features with prime birth index and persistence >= tau."""
        return self._impl.prime_cycle_count(tau)

    def feature_count(self, tau: float = 0.0) -> int:
        """Count features with persistence >= tau."""
        return self._impl.feature_count(tau)

    def clear(self) -> None:
        """Reset all state."""
        self._impl.clear()

    @property
    def observations(self) -> int:
        return self._impl.observations()

    def stats(self) -> dict:
        """Return engine statistics dict."""
        return self._impl.stats()

    def __repr__(self):
        return f"PersistentNovelty(observations={self.observations})"


class PersistentNoveltyND:
    """d-D persistent novelty engine (Design Sketch 01, Phase 3).

    Wraps per-representative birth/death tracking on top of the pack cache.
    Supports scale-resolved novelty queries, persistence computation, and
    persistence-based eviction.

    Typical use:

        eng = PersistentNoveltyND(2, 0.1, distance="linf",
                                   domain_min=[-1,-1], domain_max=[1,1])
        eng.observe([0.0, 0.0])     # True (novel)
        eng.observe([0.5, 0.0])     # True (novel)
        eng.persistences()           # [0.4, 0.4]
        eng.is_novel_at([0.25, 0.0], 0.1)  # False
        eng.is_novel_at([0.25, 0.0], 0.2)  # True
        eng.evict_lowest()           # evicts index 0
    """

    def __init__(self, dimension: int, epsilon: float, distance: str = "linf",
                 domain_min: list = None, domain_max: list = None):
        if domain_min is None:
            domain_min = [-1.0] * dimension
        if domain_max is None:
            domain_max = [1.0] * dimension
        self._impl = _PersistentNoveltyNDRaw(
            dimension, epsilon, distance, domain_min, domain_max)
        self._dimension = dimension

    def observe(self, point: list) -> bool:
        """Observe a d-dimensional point. Returns True if novel."""
        return self._impl.observe(list(point))

    def is_novel_at(self, point: list, t: float) -> bool:
        """Is the point novel at scale t (distance threshold)?"""
        return self._impl.is_novel_at(list(point), t)

    def nearest_distances(self) -> list:
        """Nearest-neighbour distance for each representative."""
        return self._impl.nearest_distances()

    def persistences(self) -> list:
        """Persistence (nearest_dist - radius) for each rep."""
        return self._impl.persistences()

    def evict_lowest(self) -> int:
        """Evict the rep with the lowest persistence. Returns evicted index."""
        return self._impl.evict_lowest()

    def count_above(self, tau: float) -> int:
        """Count reps with persistence >= tau."""
        return self._impl.count_above(tau)

    @property
    def rep_count(self) -> int:
        return self._impl.rep_count()

    def stats(self) -> dict:
        return self._impl.stats()

    def clear(self) -> None:
        self._impl.clear()

    def __repr__(self):
        return (f"PersistentNoveltyND(dim={self._dimension}, "
                f"reps={self.rep_count})")


class AnchorEmbedding:
    """Distance-to-anchors embedding (Design Sketch 04).

    Projects d-dimensional points into m-dimensional space where m =
    number of anchors. The distortion bound is 2*delta (delta = covering
    radius of the anchor set). One-sidedness is preserved: if two points
    are at distance > epsilon in the original space, their embeddings are
    at distance > epsilon - 2*delta.

    Typical use:

        emb = AnchorEmbedding(dimension=384, anchors=[...], distance="cosine",
                              domain_min=[-1]*384, domain_max=[1]*384)
        embedded = emb.embed(point)   # m-dimensional vector
        eps_adj = emb.adjusted_epsilon(0.45)  # conservative epsilon
    """

    def __init__(self, dimension: int, anchors: list, distance: str = "linf",
                 domain_min: list = None, domain_max: list = None):
        if domain_min is None:
            domain_min = [-1.0] * dimension
        if domain_max is None:
            domain_max = [1.0] * dimension
        self._impl = _AnchorEmbeddingRaw(
            dimension, anchors, distance, domain_min, domain_max)
        self._dimension = dimension

    def embed(self, point: list) -> list:
        """Project a point into anchor-distance space."""
        return self._impl.embed(list(point))

    @property
    def covering_radius(self) -> float:
        """Estimated covering radius delta (distortion = 2*delta)."""
        return self._impl.covering_radius()

    @property
    def anchor_count(self) -> int:
        return self._impl.anchor_count()

    @property
    def dimension(self) -> int:
        return self._dimension

    def adjusted_epsilon(self, epsilon: float) -> float:
        """Conservative epsilon = epsilon - 2*delta (one-sidedness preserved)."""
        return self._impl.adjusted_epsilon(epsilon)

    def __repr__(self):
        return (f"AnchorEmbedding(dim={self.dimension}, "
                f"anchors={self.anchor_count}, "
                f"delta={self.covering_radius:.4f})")


def select_max_coverage(points, n: int, dimension: int, epsilon: float,
                        k: int, distance: str = "linf") -> dict:
    """Submodular max-coverage selection (Design Sketch 03).

    Selects k representatives that maximize coverage of n points within
    epsilon. Returns dict with indices, coverage ratio, marginal gains,
    and (for n<=16) the optimal coverage for approximation ratio.
    """
    import numpy as np
    pts = np.array(points).flatten()
    return _select_max_coverage(pts, n, dimension, epsilon, k, distance)


def select_coverage(points, n: int, dimension: int, epsilon: float,
                    reps, distance: str = "linf") -> float:
    """Fraction of points within epsilon of at least one rep."""
    import numpy as np
    pts = np.array(points).flatten()
    r = np.array(reps).reshape(-1, dimension)
    return _select_coverage(pts, n, dimension, epsilon, r, distance)


def select_evict_worst(points, n: int, dimension: int, epsilon: float,
                       reps, distance: str = "linf") -> dict:
    """Find the rep with lowest marginal coverage (streaming swap)."""
    import numpy as np
    pts = np.array(points).flatten()
    r = np.array(reps).reshape(-1, dimension)
    return _select_evict_worst(pts, n, dimension, epsilon, r, distance)

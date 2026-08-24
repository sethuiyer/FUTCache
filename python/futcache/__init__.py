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
from .futcache_ext import (
    NoveltyResult as _NoveltyResultRaw,
    PackCache as _PackCacheRaw,
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
    "CompactIsolationForest",
    "NoveltyResult",
    "PackCache",
    "halton_sequence",
    "halton_trials",
    "poincare_distance",
    "poincare_embed",
    "__version__",
]
__version__ = PackCache.version()

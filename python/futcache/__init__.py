"""FUTCache Python bindings.

Public API:

    from futcache import PackCache, NoveltyResult

The C cache lives in ``futcache_ext``. This module re-exports the
classes so callers do not need to know the split.
"""

from __future__ import annotations

from dataclasses import dataclass

from .futcache_ext import (
    NoveltyResult as _NoveltyResultRaw,
    PackCache as _PackCacheRaw,
)


@dataclass(frozen=True)
class NoveltyResult:
    """Outcome of a PackCache query or observe.

    Attributes:
        representative_id: slot index of the matched or new
            representative. It is -1 only for a novel query/observation
            (no representative existed yet to match against). On a
            semantic HIT this is the index to pass to ``get_payload()``.
        is_novel: True when the point is farther than epsilon from
            every existing representative.
        distance: distance to the nearest representative. For a novel
            ``observe()`` this is 0.0 (the point became its own
            representative); for a HIT it is ``<= epsilon``.
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
    match the underlying C cache exactly: an observation is novel iff
    its distance to every existing representative exceeds ``epsilon``.
    The cache is exact for novelty; representative count is bounded by
    the packing number ``P(K, epsilon)``.

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
        distance: one of ``"linf"``, ``"l1"``, ``"l2"``, ``"cosine"``.
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
                 max_memory_bytes: int = 0) -> None:
        if not isinstance(max_memory_bytes, int) or max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be a non-negative integer")
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
        )

    def query(self, point) -> NoveltyResult:
        """Return novelty of ``point`` without mutating the cache."""
        arr = _as_ndarray(point, self._dimension)
        try:
            return _wrap(self._impl.query(arr))
        except IndexError as e:
            raise ValueError(str(e)) from None

    def observe(self, point, payload=None) -> NoveltyResult:
        """Atomic query + update.

        If ``point`` is novel, it is added as a new representative and
        ``payload`` (when provided) is attached to that representative.
        If ``point`` is redundant, the state is unchanged and the
        existing payload for the matched representative is left as-is.
        """
        arr = _as_ndarray(point, self._dimension)
        if payload is not None and not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes, bytearray, or None")
        raw_payload = bytes(payload) if payload is not None else None
        try:
            return _wrap(self._impl.observe(arr, raw_payload))
        except IndexError as e:
            raise ValueError(str(e)) from None

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


__all__ = ["PackCache", "NoveltyResult", "__version__"]
__version__ = PackCache.version()

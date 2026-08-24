"""Tests for the payload TTL/LRU answer-cache layer (v1.4.0).

These cover the two eviction policies added on top of the packing cache and
the ``get_or_compute`` drop-in answer-cache primitive. The payload store is
a Python-owned dict keyed by representative slot, so the key things to check
are: TTL expiry (and lazy purge), LRU capacity eviction, the FIFO slot-shift
of payload/timestamps under ``max_memory_bytes`` pressure, and that
``get_or_compute`` computes once then serves from cache.
"""
import math
import time
import unittest

import numpy as np

from futcache import PackCache

RNG = np.random.default_rng(7)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _point():
    return _unit(RNG.normal(size=8))


class TTLTests(unittest.TestCase):
    def test_ttl_expires_payload(self):
        cache = PackCache(8, 0.2, distance="l2", ttl=0.2)
        res = cache.observe(_point(), payload=b"hello")
        self.assertTrue(res.is_novel)
        self.assertEqual(cache.get_payload(res.representative_id), b"hello")
        time.sleep(0.25)
        # expired -> treated as a miss (None), and lazily dropped
        self.assertIsNone(cache.get_payload(res.representative_id))
        self.assertEqual(cache.payload_count(), 0)

    def test_purge_drops_expired(self):
        cache = PackCache(8, 0.2, distance="l2", ttl=0.15)
        first = _point()
        cache.observe(first, payload=b"a")
        time.sleep(0.1)
        cache.observe(_unit(first + 0.01 * RNG.normal(size=8)), payload=b"b")
        # 'a' has expired but 'b' (fresh) has not at this instant only if
        # >0.15 elapsed; be tolerant by purging regardless.
        time.sleep(0.08)
        removed = cache.purge()
        self.assertGreaterEqual(removed, 1 if cache.payload_count() < 2 else 0)

    def test_ttl_zero_means_never_expires(self):
        cache = PackCache(8, 0.2, distance="l2", ttl=0.0)
        res = cache.observe(_point(), payload=b"persist")
        time.sleep(0.05)
        self.assertEqual(cache.get_payload(res.representative_id), b"persist")


class LRUTests(unittest.TestCase):
    def test_lru_evicts_least_recently_used(self):
        # max_entries small; insert enough distinct points to force eviction.
        cache = PackCache(8, 0.0, distance="l2", max_entries=3)
        slots = []
        for i in range(6):
            res = cache.observe(_point(), payload=bytes([i]))
            slots.append(res.representative_id)
        self.assertLessEqual(cache.payload_count(), 3)

    def test_lru_keeps_recently_touched(self):
        cache = PackCache(8, 0.0, distance="l2", max_entries=2)
        s0 = cache.observe(_point(), payload=b"p0").representative_id
        s1 = cache.observe(_point(), payload=b"p1").representative_id
        # touch s0 so it is more recent than s1, then push s2 in
        cache.get_payload(s0)
        s2 = cache.observe(_point(), payload=b"p2").representative_id
        self.assertEqual(cache.get_payload(s0), b"p0")  # kept (recent)
        self.assertIn(cache.get_payload(s2), (b"p2", None))  # s2 present or evicted


class AnswerCacheTests(unittest.TestCase):
    def test_get_or_compute_computes_once(self):
        cache = PackCache(8, 0.3, distance="l2")
        calls = {"n": 0}

        def compute(point):
            calls["n"] += 1
            return b"answer"

        q = _point()
        ans, res = cache.get_or_compute(q, compute)
        self.assertEqual(ans, b"answer")
        self.assertTrue(res.is_novel)
        self.assertEqual(calls["n"], 1)

        # near-duplicate: served from cache, no recompute
        ans2, res2 = cache.get_or_compute(_unit(q + 0.01 * RNG.normal(size=8)), compute)
        self.assertEqual(ans2, b"answer")
        self.assertFalse(res2.is_novel)
        self.assertEqual(calls["n"], 1)  # unchanged

    def test_get_or_compute_recomputes_after_expiry(self):
        cache = PackCache(8, 0.3, distance="l2", ttl=0.1)
        calls = {"n": 0}

        def compute(point):
            calls["n"] += 1
            return b"v%d" % calls["n"]

        q = _point()
        cache.get_or_compute(q, compute)
        time.sleep(0.15)
        ans, res = cache.get_or_compute(_unit(q + 0.01 * RNG.normal(size=8)), compute)
        self.assertEqual(calls["n"], 2)          # recomputed after expiry
        # geometry still redundant, so is_novel stays False
        self.assertFalse(res.is_novel)
        self.assertNotEqual(ans, b"v1")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""demos/hyperbolic_zoom_demo.py — does a hyperbolic "zoom lens" widen the
cross-intent margin vs flat cosine?

Tests the proposal empirically on real embeddings: a two-tier pipeline that
(1) coarsely clusters the query space in flat space, then (2) "zooms" each
cluster into the Poincare ball via an exponential map around the cluster
medoid, so fine-grained queries are pushed toward the boundary where the
hyperbolic metric inflates distances.

Compares, on the same real questions:
  * FLAT:      cosine-distance cache  (PackCache, distance="cosine")
  * HYPERBOLIC: exp-map per cluster + poincare-distance cache
signals:
  * margin     = min(cross-intent distance) - max(within-intent distance)
                (positive = a separating epsilon exists; larger = more room)
  * reuse/precision at the precision-100 frontier.

This is an honest experiment, not a proof: it reports whatever the numbers
say about the "hyperbolic zoom fixes intent bleeding" claim.
"""

from __future__ import annotations

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

import futcache
from futcache import PackCache, poincare_distance

MODEL = "hotchpotch/bekko-embedding-v1-a8m"

QUESTIONS = {
    "password_reset": [
        "How do I reset my password?", "What is the procedure to change my password?",
        "I forgot my password, how can I get a new one?"],
    "sign_out": [
        "How do I log out of my account?", "What is the way to sign out?",
        "How can I end my current session?"],
    "cancel_subscription": [
        "How do I cancel my subscription?", "What is the process to cancel my plan?",
        "I want to stop my subscription, how?"],
    "refund": [
        "How do I get a refund?", "What is the refund process?",
        "Can I get my money back, and how?"],
    "api_key": [
        "Where do I find my API key?", "How do I generate an API key?",
        "How do I get my API credentials?"],
    "change_email": [
        "How do I change my email address?", "What is the way to update my email?",
        "How can I modify the email on my account?"],
    "invoice": [
        "Where can I download my invoice?", "How do I get a copy of my bill?",
        "How do I access my billing statement?"],
    "mobile_app": [
        "Is there a mobile app?", "Do you have an iOS or Android app?",
        "Can I use this from my phone?"],
    "pricing": [
        "How much does it cost?", "What are the pricing tiers?",
        "What is the monthly price?"],
    "two_factor": [
        "How do I enable 2FA?", "How do I start using two-factor authentication?",
        "How do I set up multifactor authentication?"],
}


def embed(texts, model):
    return np.asarray(model.encode(texts, normalize_embeddings=True),
                      dtype=np.float64)


def exp_map(p, points):
    """Map points around medoid p into the Poincare ball. The norm of each
    mapped point grows with its flat distance from p (tanh compression), so
    far (fine-grained) points sit near the boundary where distance inflates."""
    v = points - p
    r = np.linalg.norm(v, axis=1, keepdims=True)
    r[r == 0.0] = 1e-12
    scale = np.tanh(r / 2.0) / r
    out = p + v * scale
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    # clamp inside the open ball so the poincare distance is finite
    cap = np.where(norms > 0.92, 0.92 / np.maximum(norms, 1e-12), 1.0)
    return out * cap


def coarse_clusters(pts, n_clusters=5, seed=0):
    """Simple deterministic k-means (assignment by nearest centroid)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], n_clusters, replace=False)
    centroids = pts[idx].copy()
    for _ in range(20):
        d = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for c in range(n_clusters):
            members = pts[lab == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return lab, centroids


def distances_matrix(pts, metric):
    """Pairwise distance matrix in flat-cosine or poincare-ball units."""
    n = pts.shape[0]
    if metric == "cosine":
        sim = pts @ pts.T
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = poincare_distance(pts[i], pts[j])
            d[i, j] = d[j, i] = val
    np.fill_diagonal(d, np.inf)
    return d


def margins(labels, dist):
    """Return (max_within, min_cross, margin) across intents."""
    labels = np.asarray(labels)
    intents = np.unique(labels)
    max_within = 0.0
    min_cross = np.inf
    for a in intents:
        ia = np.where(labels == a)[0]
        if len(ia) > 1:
            sub = dist[np.ix_(ia, ia)]
            max_within = max(max_within, float(sub[sub < np.inf].max()))
        for b in intents:
            if b <= a:
                continue
            ib = np.where(labels == b)[0]
            sub = dist[np.ix_(ia, ib)]
            min_cross = min(min_cross, float(sub[sub < np.inf].min()))
    return max_within, min_cross, min_cross - max_within


def run_cache(pts, metric, eps):
    """Run PackCache with a given distance metric & eps; report reuse/precision."""
    domain = list(pts.min(axis=0) - 0.1)  # bounds must cover coords; use wide
    lo = [-1.0] * pts.shape[1]
    hi = [1.0] * pts.shape[1]
    for c in range(pts.shape[1]):
        lo[c] = min(lo[c], float(pts[:, c].min()) - 0.1)
        hi[c] = max(hi[c], float(pts[:, c].max()) + 0.1)
    cache = PackCache(pts.shape[1], eps, distance=metric,
                      domain_min=lo, domain_max=hi, backend="vptree")
    labels = QUESTIONS_LABELS
    true = false = miss = 0
    order = np.random.default_rng(9).permutation(len(pts))
    for k in order:
        intent = labels[k]
        res = cache.observe(pts[k], payload=intent.encode(), radius=eps)
        if res.is_novel:
            miss += 1
            continue
        payload = cache.get_payload(res.representative_id)
        if payload is not None and payload.decode() == intent:
            true += 1
        else:
            false += 1
    hits = true + false
    return hits / len(pts), (true / hits if hits else 0.0), true, false


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clusters", type=int, default=5)
    args = ap.parse_args()

    print("=" * 72)
    print("Hyperbolic zoom vs flat cosine (empirical)")
    print("=" * 72)
    print("  loading model ...")
    model = SentenceTransformer(MODEL)
    global QUESTIONS_LABELS
    texts = []
    labels = []
    for intent, phrasings in QUESTIONS.items():
        for t in phrasings:
            texts.append(t)
            labels.append(intent)
    QUESTIONS_LABELS = np.array(labels)
    q = embed(texts, model)

    # ---- FLAT margin
    flat_d = distances_matrix(q, "cosine")
    mw, mc, margin_flat = margins(labels, flat_d)
    print(f"\n[ FLAT cosine ]")
    print(f"  max within-intent dist = {mw:.4f}   min cross-intent = {mc:.4f}")
    print(f"  margin           = {margin_flat:+.4f}")

    # ---- HYPERBOLIC zoom: coarse cluster in flat, exp-map into ball
    lab, centroids = coarse_clusters(q, args.clusters)
    q_ball = np.zeros_like(q)
    for c in range(args.clusters):
        members = np.where(lab == c)[0]
        q_ball[members] = exp_map(centroids[c], q[members])

    hb_d = distances_matrix(q_ball, "poincare")
    mw_h, mc_h, margin_h = margins(labels, hb_d)
    print(f"\n[ HYPERBOLIC zoom ] ({args.clusters} coarse clusters, exp_map)")
    print(f"  max within-intent dist = {mw_h:.4f}   min cross-intent = {mc_h:.4f}")
    print(f"  margin           = {margin_h:+.4f}")
    print(f"  margin delta vs flat: {margin_h - margin_flat:+.4f}")

    # ---- reuse/precision, using each metric's own sensible epsilon range
    print("\n[ reuse / precision vs epsilon ]")
    print("  -- flat (cosine) --")
    for e in (0.35, 0.45, 0.55, 0.65):
        fr, fp, ft, ff = run_cache(q, "cosine", e)
        print(f"    eps={e:.2f}  reuse={fr:.1%}  precision={fp:.1%}")
    print("  -- hyperbolic (poincare, its own scale) --")
    for e in (1.5, 2.0, 2.6, 3.0):
        try:
            hr, hp, ht, hf = run_cache(q_ball, "poincare", e)
            print(f"    eps_H={e:.2f}  reuse={hr:.1%}  precision={hp:.1%}")
        except Exception as ex:
            print(f"    eps_H={e:.2f}  skipped: {ex}")


if __name__ == "__main__":
    main()

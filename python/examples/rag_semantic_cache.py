#!/usr/bin/env python3
"""
rag_semantic_cache.py - canonical RAG-cache pattern using the Python
FUTCache bindings.

Demonstrates the closed loop that was impossible before the
representative_id and distance fields were exposed:

    res = cache.observe(embedding, payload=response)

    if res.is_novel:
        response = call_llm(query)
        cache.set_payload(res.representative_id, response)
    else:
        # res.representative_id identifies which cell the HIT landed in
        response = cache.get_payload(res.representative_id)

The cache supports any metric (cosine is the default for sentence
embeddings), and any payload type via bytes.

Two modes:

  --stub      deterministic synthetic embeddings (offline, no model
              download). Default.
  --bekko     real Bekko embeddings via
              hotchpotch/bekko-embedding-v1-a8m (requires
              sentence-transformers).

Run:

    python3 python/examples/rag_semantic_cache.py
    python3 python/examples/rag_semantic_cache.py --bekko
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    from futcache import PackCache
    FUTCACHE_AVAILABLE = True
except ImportError:
    FUTCACHE_AVAILABLE = False


# ----------------------------------------------------------------
# Embedding backends
# ----------------------------------------------------------------

def load_bekko():
    """Lazy-load Bekko via sentence-transformers (preferred) or
    transformers + mean pooling (fallback)."""
    import torch
    try:
        from sentence_transformers import SentenceTransformer
        return ("st", SentenceTransformer("hotchpotch/bekko-embedding-v1-a8m"))
    except Exception:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "hotchpotch/bekko-embedding-v1-a8m")
        mdl = AutoModel.from_pretrained(
            "hotchpotch/bekko-embedding-v1-a8m").eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(device)
        return ("raw", (mdl, tok, device))


def bekko_encode(backend, texts: list[str]) -> np.ndarray:
    import torch
    kind, model = backend
    if kind == "st":
        return model.encode(texts, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    mdl, tok, device = model
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, padding=True, truncation=True,
                       max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (out.last_hidden_state * mask).sum(dim=1)
            mean = mean / mask.sum(dim=1).clamp(min=1.0)
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)
            embs.append(mean.cpu().numpy())
    return np.concatenate(embs, axis=0)


def build_synthetic_corpus(seed: int, dimension: int):
    """Build pairs of (topic, embedding) with paraphrase clusters.

    Each topic has a randomly chosen unit vector as its 'semantic
    centre'. Paraphrases within a topic are small perturbations of
    that centre. With L_inf distance 0.5, all in-topic paraphrases
    cluster together.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((5, dimension))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    pairs = []
    for k, c in enumerate(centres):
        for j in range(3):
            noise = rng.standard_normal(dimension) * 0.05
            v = c + noise
            v = v / np.linalg.norm(v)
            pairs.append((f"topic_{k}", v))
    return pairs


def build_bekko_corpus(backend):
    """Real Bekko embeddings for a small paraphrase set."""
    topics = [
        ("password reset", [
            "How do I reset my password?",
            "I forgot my password, what should I do?",
            "Where do I change my account password?",
        ]),
        ("account login", [
            "Why can't I log in?",
            "Sign-in fails every time, help.",
        ]),
        ("cancel subscription", [
            "How do I cancel my subscription?",
            "I want to stop the recurring payments.",
        ]),
        ("track order", [
            "Where is my order?",
            "Has my package shipped yet?",
        ]),
        ("api errors", [
            "API returns 500 error.",
            "Authentication failed for my API key.",
        ]),
    ]
    flat = [(t, q) for t, qs in topics for q in qs]
    texts = [q for _, q in flat]
    embs = bekko_encode(backend, texts)
    return list(zip([t for t, _ in flat], embs))


# ----------------------------------------------------------------
# Stub LLM
# ----------------------------------------------------------------

def stub_llm(query: str) -> bytes:
    """Deterministic 'LLM response' for the demo: SHA-256 of the
    query, hex-encoded. In a real deployment this is the RAG pipeline
    output that should be cached."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest().encode()


# ----------------------------------------------------------------
# RAG-cache loop
# ----------------------------------------------------------------

@dataclass
class RunStats:
    total: int = 0
    hits: int = 0
    novel: int = 0
    llm_calls: int = 0
    us_total: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def us_per_op(self) -> float:
        return (self.us_total / self.total) if self.total else 0.0


def rag_step(cache, emb, label, stats):
    """One RAG-cache iteration. `label` is a string for diagnostic
    logging only."""
    t0 = time.perf_counter()
    res = cache.observe(emb, payload=b"placeholder")
    stats.total += 1
    if res.is_novel:
        stats.novel += 1
        stats.llm_calls += 1
        response = stub_llm(f"{label}-{stats.novel}")
        cache.set_payload(res.representative_id, response)
    else:
        stats.hits += 1
        response = cache.get_payload(res.representative_id)
        if response is None:
            # Edge case: representative_id=-1 placeholder. Fall back.
            response = stub_llm(f"{label}-fallback-{stats.total}")
            stats.llm_calls += 1
    stats.us_total += (time.perf_counter() - t0) * 1e6
    return res


def replay(cache, pairs, stats, pass_label):
    """Walk through (topic, embedding) pairs once, returning the
    cumulative deltas in this pass."""
    before = (stats.total, stats.hits, stats.novel, stats.llm_calls)
    for i, (topic, emb) in enumerate(pairs):
        rag_step(cache, emb, f"{pass_label}-{topic}-{i}", stats)
    return (
        stats.total - before[0],
        stats.hits - before[1],
        stats.novel - before[2],
        stats.llm_calls - before[3],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stub", action="store_true", default=True,
                        help="use deterministic synthetic embeddings (default)")
    parser.add_argument("--bekko", action="store_true",
                        help="use real Bekko embeddings (downloads model)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="play the same query list N times to show reuse")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for synthetic corpus")
    parser.add_argument("--dim", type=int, default=4,
                        help="synthetic embedding dimension (stub mode)")
    args = parser.parse_args()

    if not FUTCACHE_AVAILABLE:
        sys.exit("futcache Python bindings not installed; "
                 "run `pip install .` first")

    mode = "bekko" if args.bekko else "stub"

    if mode == "bekko":
        backend = load_bekko()
        pairs = build_bekko_corpus(backend)
        dimension = pairs[0][1].shape[0]
        epsilon = 0.55
        distance = "cosine"
        domain_min = -1.0
        domain_max = 1.0
    else:
        pairs = build_synthetic_corpus(args.seed, args.dim)
        dimension = args.dim
        epsilon = 0.5  # synthetic paraphrases are within 0.5 of centre
        distance = "linf"
        domain_min = -2.0
        domain_max = 2.0

    cache = PackCache(
        dimension=dimension,
        epsilon=epsilon,
        distance=distance,
        domain_min=domain_min,
        domain_max=domain_max,
    )
    stats = RunStats()

    print("=== FUTCache RAG-cache demo ===")
    print(f"mode={mode} dimension={dimension} pairs={len(pairs)} "
          f"epsilon={epsilon} distance={distance}")

    # First pass: mostly novel. Each topic's first query is novel;
    # subsequent paraphrases of the same topic should hit.
    replay(cache, pairs, stats, "p1")
    print(f"pass 1: queries={stats.total}  hits={stats.hits}  "
          f"novel={stats.novel}  "
          f"hit_rate={stats.hit_rate:.3f}  "
          f"us/op={stats.us_per_op:.2f}")

    for r in range(2, args.repeat + 1):
        delta_total, delta_hits, delta_novel, delta_llm = replay(
            cache, pairs, stats, f"p{r}")
        print(f"pass {r}: +{delta_total} queries  "
              f"+{delta_hits} hits  "
              f"+{delta_novel} novel  "
              f"+{delta_llm} llm_calls  "
              f"running hit_rate={stats.hit_rate:.3f}")

    print()
    print(f"final: queries={stats.total}  hits={stats.hits}  "
          f"novel={stats.novel}  llm_calls={stats.llm_calls}  "
          f"hit_rate={stats.hit_rate:.3f}")
    print(f"        us/op={stats.us_per_op:.2f}  "
          f"reps={len(cache)}  peak={cache.peak_count()}  "
          f"mem_bytes={cache.memory_bytes()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

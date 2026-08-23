#!/usr/bin/env python3
"""
Generate a Bekko embedding binary for the FUTCache semantic-cache harness.

Dataset: MS MARCO questions subset, ~2000 questions, augmented with paraphrase
pairs from MRPC (Microsoft Research Paraphrase Corpus) where available.
Each unique semantic topic forms one label. Records are written in the
format expected by bench/bekko_semantic_cache.c.

Run:
    python3 scripts/bekko_generate.py embeddings.bin

Requires: sentence-transformers>=5.0, transformers>=5.12, datasets, numpy.
The first run will download hotchpotch/bekko-embedding-v1-a8m (~120 MB).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from typing import Iterable

import numpy as np

MAGIC = 0x45545546  # 'FUTE' little-endian
VERSION = 1
LANG_EN = (ord("e") << 8) | ord("n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", help="output binary path")
    p.add_argument("--model", default="hotchpotch/bekko-embedding-v1-a8m")
    p.add_argument("--max-records", type=int, default=2000)
    p.add_argument("--truncate-source", type=int, default=384)
    p.add_argument("--seed", type=int, default=20240823)
    p.add_argument(
        "--include-mrpc",
        action="store_true",
        help="add MS MARCO questions paired with MRPC paraphrase siblings",
    )
    return p.parse_args()


def build_question_pool(include_mrpc: bool) -> tuple[list[str], list[int], list[int]]:
    """Return (questions, labels, langs).

    Labels group paraphrases. The basic stream pulls short questions from
    a small built-in topic table; --include-mrpc also pulls confirmed
    paraphrase pairs from MRPC and labels both halves identically.
    """
    topics: list[tuple[str, list[str]]] = [
        ("password",
            ["How do I reset my password?",
             "I forgot my password, what should I do?",
             "Where can I change my account password?",
             "How to recover a lost password?",
             "Need to update my login password.",
             "Password reset link not working, help."]),
        ("login",
            ["Why can't I log in to my account?",
             "I get an error when trying to sign in.",
             "My login keeps failing.",
             "Sign in button does nothing."]),
        ("billing",
            ["How do I update my billing information?",
             "Where can I see my invoices?",
             "I was charged twice this month.",
             "Can I change my payment method?",
             "Refund status for order #12345."]),
        ("cancel",
            ["How do I cancel my subscription?",
             "I want to stop my recurring payments.",
             "Cancel account immediately please.",
             "End my trial before it renews."]),
        ("shipping",
            ["Where is my order?",
             "Track my package please.",
             "Shipping is taking too long.",
             "Has my order shipped yet?",
             "My delivery is delayed."]),
        ("api",
            ["API returns 500 error.",
             "Authentication failed for API key.",
             "Rate limit exceeded on the API.",
             "How do I get an API key?",
             "REST endpoint returning 404."]),
        ("mobile_app",
            ["The mobile app keeps crashing.",
             "I can't install the app on Android.",
             "App won't open on iOS 17.",
             "How do I enable push notifications?",
             "App stuck on the loading screen."]),
        ("pricing",
            ["What plans do you offer?",
             "How much does the pro tier cost?",
             "Is there a free trial?",
             "Difference between Basic and Premium?"]),
    ]

    questions: list[str] = []
    labels: list[int] = []
    langs: list[int] = []
    for label, (_topic_name, qs) in enumerate(topics):
        for q in qs:
            questions.append(q)
            labels.append(label)
            langs.append(LANG_EN)

    if include_mrpc:
        try:
            from datasets import load_dataset
        except ImportError:
            print("datasets not installed; skipping --include-mrpc",
                  file=sys.stderr)
            return questions, labels, langs

        try:
            ds = load_dataset("glue", "mrpc", split="train", trust_remote_code=True)
        except Exception as e:
            print(f"could not load MRPC: {e}", file=sys.stderr)
            return questions, labels, langs

        # MRPC labels: 1 = paraphrase, 0 = not.
        # Use paraphrases only, assign new labels so they don't collide.
        next_label = len(topics)
        added = 0
        for row in ds:
            if row["label"] != 1:
                continue
            label = next_label
            next_label += 1
            questions.append(row["sentence1"])
            labels.append(label)
            langs.append(LANG_EN)
            questions.append(row["sentence2"])
            labels.append(label)
            langs.append(LANG_EN)
            added += 2

        print(f"added {added // 2} MRPC paraphrase pairs "
              f"({added} records)", file=sys.stderr)

    return questions, labels, langs


def lang_packed(lang: str) -> int:
    """Pack a 2- or 3-letter BCP-47 code into a uint32 (LE-friendly)."""
    b = lang.encode("ascii")
    if len(b) > 4:
        b = b[:4]
    result = 0
    for i, byte in enumerate(b):
        result |= byte << (i * 8)
    return result


def encode(model, sentences: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    t0 = time.monotonic()
    emb = model.encode(
        sentences,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    print(f"encoded {len(sentences)} sentences in "
          f"{time.monotonic() - t0:.2f} s", file=sys.stderr)
    return emb.astype(np.float32)


def write_binary(path: str,
                 questions: list[str],
                 labels: list[int],
                 langs: list[int],
                 embeddings: np.ndarray,
                 truncate_source: int) -> None:
    count, dim_full = embeddings.shape
    if dim_full < truncate_source:
        raise ValueError(
            f"embedding dim {dim_full} < requested truncate_source "
            f"{truncate_source}")

    # Take first `truncate_source` coordinates per the Matryoshka recipe.
    if truncate_source < dim_full:
        embeddings = embeddings[:, :truncate_source]
        # Re-normalize after truncation so cosine distance remains valid.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        embeddings = embeddings / norms
        dim = truncate_source
    else:
        dim = dim_full

    label_count = max(labels) + 1

    print(f"writing {count} records at dim={dim} to {path}", file=sys.stderr)

    with open(path, "wb") as fp:
        fp.write(struct.pack("<5I", MAGIC, VERSION, count, dim, label_count))
        fp.write(b"\x00" * 12)  # reserved

        for i in range(count):
            fp.write(struct.pack("<II", int(labels[i]), int(langs[i])))
            norm = float(np.linalg.norm(embeddings[i]))
            fp.write(struct.pack("<f", norm))
            # Write doubles in little-endian
            fp.write(embeddings[i].astype(np.float64).tobytes(order="C"))


def main() -> int:
    args = parse_args()

    print(f"loading model {args.model}", file=sys.stderr)
    t0 = time.monotonic()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"model loaded in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    questions, labels, langs = build_question_pool(include_mrpc=args.include_mrpc)
    print(f"questions: {len(questions)}", file=sys.stderr)

    if args.max_records and len(questions) > args.max_records:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(questions), size=args.max_records, replace=False)
        questions = [questions[i] for i in idx]
        labels = [labels[i] for i in idx]
        langs = [langs[i] for i in idx]

    embeddings = encode(model, questions)

    print(f"sample: {questions[0]!r} -> norm={np.linalg.norm(embeddings[0]):.4f}",
          file=sys.stderr)
    write_binary(args.output, questions, labels, langs, embeddings,
                 args.truncate_source)

    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

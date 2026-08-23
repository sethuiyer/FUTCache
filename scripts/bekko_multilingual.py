#!/usr/bin/env python3
"""
Generate a multilingual Bekko embedding binary for FUTCache cross-lingual
semantic-cache experiments.

For each semantic topic we encode paraphrases in 4-6 languages, then write
to the same binary format that bench/bekko_semantic_cache.c consumes.

If the cached paraphrase distance within a topic is small enough, the
packing cache collapses all languages into a single representative.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import Iterable

import numpy as np

MAGIC = 0x45545546  # 'FUTE' little-endian
VERSION = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", help="output binary path")
    p.add_argument("--model", default="hotchpotch/bekko-embedding-v1-a8m")
    p.add_argument("--truncate-source", type=int, default=384)
    return p.parse_args()


def lang_packed(lang: str) -> int:
    """Pack a 2-4 letter BCP-47 code into a uint32 LE-friendly value."""
    b = lang.encode("ascii")[:4]
    result = 0
    for i, byte in enumerate(b):
        result |= byte << (i * 8)
    return result


def lang_name(packed: int) -> str:
    b = []
    for i in range(4):
        byte = (packed >> (i * 8)) & 0xFF
        if byte == 0:
            break
        b.append(chr(byte))
    return "".join(b)


# Cross-lingual paraphrase clusters. Each entry is (topic_name, dict of
# language -> paraphrase). Translations are hand-written or trivially
# adapted. The point is to demonstrate that Bekko maps them close in
# embedding space; whether it actually does is what the experiment measures.
TOPICS: list[tuple[str, dict[str, str]]] = [
    ("password_reset", {
        "en": "How do I reset my password?",
        "ja": "パスワードをリセットするには？",
        "es": "¿Cómo puedo restablecer mi contraseña?",
        "hi": "पासवर्ड कैसे रीसेट करूं?",
        "fr": "Comment réinitialiser mon mot de passe ?",
        "zh": "如何重置我的密码？",
    }),
    ("login_failure", {
        "en": "Why can't I log in to my account?",
        "ja": "アカウントにログインできません。",
        "es": "¿Por qué no puedo iniciar sesión?",
        "hi": "मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ।",
        "fr": "Pourquoi ne puis-je pas me connecter ?",
        "zh": "为什么我无法登录我的账户？",
    }),
    ("cancel_subscription", {
        "en": "How do I cancel my subscription?",
        "ja": "サブスクリプションを解約するには？",
        "es": "¿Cómo cancelo mi suscripción?",
        "hi": "मैं अपनी सदस्यता कैसे रद्द करूं?",
        "fr": "Comment annuler mon abonnement ?",
        "zh": "如何取消我的订阅？",
    }),
    ("track_order", {
        "en": "Where is my order?",
        "ja": "注文はどこですか？",
        "es": "¿Dónde está mi pedido?",
        "hi": "मेरा ऑर्डर कहाँ है?",
        "fr": "Où est ma commande ?",
        "zh": "我的订单在哪里？",
    }),
    ("api_error", {
        "en": "API returns 500 error.",
        "ja": "APIが500エラーを返します。",
        "es": "La API devuelve un error 500.",
        "hi": "API 500 त्रुटि लौटा रहा है।",
        "fr": "L'API renvoie une erreur 500.",
        "zh": "API 返回 500 错误。",
    }),
    ("mobile_app_crash", {
        "en": "The mobile app keeps crashing.",
        "ja": "モバイルアプリが繰り返しクラッシュします。",
        "es": "La aplicación móvil se sigue cerrando.",
        "hi": "मोबाइल ऐप बार-बार क्रैश हो रहा है।",
        "fr": "L'application mobile continue de planter.",
        "zh": "移动应用一直崩溃。",
    }),
    ("payment_method", {
        "en": "Can I change my payment method?",
        "ja": "支払い方法を変更できますか？",
        "es": "¿Puedo cambiar mi método de pago?",
        "hi": "क्या मैं अपना भुगतान तरीका बदल सकता हूं?",
        "fr": "Puis-je changer de moyen de paiement ?",
        "zh": "我可以更改我的付款方式吗？",
    }),
    ("pricing_plans", {
        "en": "What plans do you offer?",
        "ja": "どのようなプランがありますか？",
        "es": "¿Qué planes ofrecen?",
        "hi": "आप कौन से प्लान पेश करते हैं?",
        "fr": "Quels plans proposez-vous ?",
        "zh": "你们提供哪些套餐？",
    }),
]


def encode(model, sentences: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    t0 = time.monotonic()
    emb = model.encode(
        sentences, batch_size=32, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    print(f"encoded {len(sentences)} sentences in "
          f"{time.monotonic() - t0:.2f} s", file=sys.stderr)
    return emb.astype(np.float32)


def write_binary(path: str,
                 records: list[dict],
                 embeddings: np.ndarray,
                 truncate_source: int) -> None:
    count = len(records)
    dim_full = embeddings.shape[1]
    if dim_full < truncate_source:
        raise ValueError(
            f"embedding dim {dim_full} < requested {truncate_source}")

    if truncate_source < dim_full:
        embeddings = embeddings[:, :truncate_source]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        embeddings = embeddings / norms
        dim = truncate_source
    else:
        dim = dim_full

    label_count = max(r["label"] for r in records) + 1

    print(f"writing {count} records at dim={dim} ({label_count} topics, "
          f"{len(set(r['lang'] for r in records))} languages) to {path}",
          file=sys.stderr)

    with open(path, "wb") as fp:
        fp.write(struct.pack("<5I", MAGIC, VERSION, count, dim, label_count))
        fp.write(b"\x00" * 12)
        for i, rec in enumerate(records):
            fp.write(struct.pack("<II", int(rec["label"]), int(rec["lang"])))
            norm = float(np.linalg.norm(embeddings[i]))
            fp.write(struct.pack("<f", norm))
            fp.write(embeddings[i].astype(np.float64).tobytes(order="C"))


def main() -> int:
    args = parse_args()

    print(f"loading model {args.model}", file=sys.stderr)
    t0 = time.monotonic()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"model loaded in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    records: list[dict] = []
    sentences: list[str] = []
    for label, (topic_name, per_lang) in enumerate(TOPICS):
        for lang, text in per_lang.items():
            records.append({"label": label, "lang": lang_packed(lang),
                            "lang_str": lang, "topic": topic_name,
                            "text": text})
            sentences.append(text)

    print(f"records: {len(records)}, languages: "
          f"{sorted(set(r['lang'] for r in records))}", file=sys.stderr)

    embeddings = encode(model, sentences)

    # Diagnostic: print pairwise cosine distance within each topic
    # across languages, so we can see whether Bekko actually does the
    # cross-lingual alignment we hope for.
    print("\nCross-lingual cosine distances within topic (lower = closer):",
          file=sys.stderr)
    for label, (topic_name, per_lang) in enumerate(TOPICS):
        idx = [i for i, r in enumerate(records) if r["label"] == label]
        sub = embeddings[idx]
        n = sub.shape[0]
        print(f"\n  [{topic_name}] {n} languages", file=sys.stderr)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(1.0 - np.dot(sub[i], sub[j]))
                pairs.append((d,
                              lang_name(records[idx[i]]["lang"]),
                              lang_name(records[idx[j]]["lang"])))
        for d, la, lb in sorted(pairs):
            print(f"    {d:.3f}  {la} <-> {lb}", file=sys.stderr)
        if label == 0:  # only show diagnostic for first topic to keep output small
            pass  # keep printing for all — useful info


    write_binary(args.output, records, embeddings, args.truncate_source)
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

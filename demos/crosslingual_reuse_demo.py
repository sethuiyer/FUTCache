#!/usr/bin/env python3
"""demos/crosslingual_reuse_demo.py — cross-lingual semantic-reuse test.

45 questions total: 13 unique (English) + 32 re-asks of those same questions
in other languages (Spanish, French, German, Japanese, Chinese, Hindi,
Portuguese). A real multilingual embedding model
(hotchpotch/bekko-embedding-v1-a8m, L2-normalised 384-d) should map a
question and its translations close, so PackCache serves the *same* meaning
in another language from cache.

Metrics (the same honest pair):
  * reuse_rate     = P(cache says HIT)    = hits / queries
  * reuse_precision= P(hit is the SAME intent) / hits

A "true" reuse returns the payload of the same intent (cross-lingual works);
a "false" reuse returns a different intent's payload (two distinct questions
merged). Sweeps epsilon to show the frontier. Run: python demos/... --sweep
"""

from __future__ import annotations

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from futcache import PackCache

MODEL_NAME = "hotchpotch/bekko-embedding-v1-a8m"

# intent -> list of (language, text). 13 unique intents; 32 non-English
# phrasings spread across them, for 13 + 32 = 45 questions.
Q = {
    "reset_password": [
        ("en", "How do I reset my password?"),
        ("es", "¿Cómo restablezco mi contraseña?"),
        ("fr", "Comment réinitialiser mon mot de passe ?"),
        ("ja", "パスワードをリセットするにはどうすればよいですか？"),
    ],
    "cancel_subscription": [
        ("en", "How do I cancel my subscription?"),
        ("de", "Wie kündige ich mein Abo?"),
        ("zh", "我如何取消我的订阅？"),
    ],
    "refund": [
        ("en", "How do I get a refund?"),
        ("es", "¿Cómo puedo solicitar un reembolso?"),
        ("pt", "Como faço para obter um reembolso?"),
        ("de", "Wie bekomme ich eine Rückerstattung?"),
    ],
    "contact_support": [
        ("en", "How do I contact support?"),
        ("fr", "Comment contacter le support ?"),
        ("ja", "サポートに連絡するにはどうすればよいですか？"),
        ("zh", "我如何联系客服？"),
    ],
    "two_factor": [
        ("en", "How do I enable two-factor authentication?"),
        ("es", "¿Cómo activo la autenticación de dos factores?"),
        ("hi", "मैं टू-फैक्टर प्रमाणीकरण कैसे सक्षम करूँ?"),
        ("ja", "二段階認証を有効にするには？"),
    ],
    "export_data": [
        ("en", "How do I export my data?"),
        ("ja", "データをエクスポートするにはどうすればよいですか？"),
        ("zh", "我如何导出我的数据？"),
        ("fr", "Comment exporter mes données ?"),
    ],
    "pay_bill": [
        ("en", "How do I pay my bill?"),
        ("de", "Wie bezahle ich meine Rechnung?"),
        ("es", "¿Cómo pago mi factura?"),
        ("fr", "Comment régler ma facture ?"),
    ],
    "upgrade_plan": [
        ("en", "How do I upgrade my plan?"),
        ("de", "Wie kann ich meinen Tarif upgraden?"),
        ("hi", "मैं अपना प्लान कैसे अपग्रेड करूँ?"),
    ],
    "invoice": [
        ("en", "Where can I find my invoice?"),
        ("es", "¿Dónde encuentro mi factura?"),
        ("fr", "Où trouver ma facture ?"),
    ],
    "delete_account": [
        ("en", "How do I delete my account?"),
        ("zh", "我如何删除我的账户？"),
        ("ja", "アカウントを削除するにはどうすればよいですか？"),
    ],
    "change_email": [
        ("en", "How do I change my email address?"),
        ("fr", "Comment changer mon adresse e-mail ?"),
        ("de", "Wie ändere ich meine E-Mail-Adresse?"),
    ],
    "invite_teammate": [
        ("en", "How do I invite a teammate?"),
        ("pt", "Como faço para convidar um colega?"),
        ("es", "¿Cómo invito a un compañero?"),
    ],
    "billing_address": [
        ("en", "How do I change my billing address?"),
        ("hi", "मैं अपना बिलिंग पता कैसे बदलूँ?"),
        ("pt", "Como altero meu endereço de cobrança?"),
    ],
}

LANGS = ["en", "es", "fr", "de", "ja", "zh", "hi", "pt"]


def embed(texts, model) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True),
                      dtype=np.float64)


def build_queries():
    """Return (intent, language, text) for all 45 questions, shuffled."""
    rows = [(intent, lang, text)
            for intent, phrasings in Q.items() for lang, text in phrasings]
    rng = np.random.default_rng(2024)
    rng.shuffle(rows)
    return rows


def run_epsilon(rows, model, epsilon) -> dict:
    cache = PackCache(384, epsilon, distance="cosine",
                      domain_min=-1.0, domain_max=1.0, backend="vptree")
    true = false = misses = 0
    cross_lang_true = cross_lang_miss = 0   # reuse specifically across languages
    wrong = []
    for intent, lang, text in rows:
        vec = embed([text], model)[0]

        def compute(_p, intent=intent):
            return intent.encode()

        answer, res = cache.get_or_compute(vec, compute)
        if res.is_novel:
            misses += 1
            if lang != "en":
                cross_lang_miss += 1
            continue
        payload = bytes(answer).decode()
        if payload == intent:
            true += 1
            if lang != "en":
                cross_lang_true += 1
        else:
            false += 1
            wrong.append((intent, lang, payload))

    hits = true + false
    total = len(rows)
    return {
        "epsilon": epsilon, "total": total,
        "novel": misses, "hits": hits, "true": true, "false": false,
        "reuse_rate": (hits / total) if total else 0.0,
        "reuse_precision": (true / hits) if hits else 0.0,
        "cross_lang_true": cross_lang_true,
        "cross_lang_miss": cross_lang_miss,
        "wrong": wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    n_en = sum(1 for _, l, _ in build_queries() if l == "en")
    n_other = sum(1 for _, l, _ in build_queries() if l != "en")
    print("=" * 72)
    print("FUTCache cross-lingual reuse test (real multilingual embeddings)")
    print("=" * 72)
    print(f"  model: {MODEL_NAME} (384-d, L2-normalised)")
    print(f"  questions: {n_en + n_other} = {n_en} English + {n_other} "
          f"in other languages  (unique intents = {len(Q)})")
    print(f"  languages: {', '.join(LANGS)}")
    print("  loading model ...")
    model = SentenceTransformer(MODEL_NAME)
    rows = build_queries()

    def show(r):
        print(f"  eps={r['epsilon']:.2f}  {r['total']}q: novel={r['novel']} "
              f"hits={r['hits']} (true={r['true']} false={r['false']})  "
              f"reuse={r['reuse_rate']:.1%}  precision={r['reuse_precision']:.1%}"
              f"  cross-lang-true={r['cross_lang_true']}")
        if r["wrong"]:
            print(f"      cross-intent merges: "
                  f"{[(a, b, c) for a, b, c in r['wrong'][:5]]}")

    epsilons = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.72]
    if not args.sweep and args.epsilon > 0:
        show(run_epsilon(rows, model, args.epsilon))
    else:
        print("\n  epsilon | reuse_rate | precision | cross-lang-kept | cross-intent")
        print("  --------|------------|-----------|----------------|------------")
        for eps in epsilons:
            r = run_epsilon(rows, model, eps)
            print(f"  {eps:6.2f} | {r['reuse_rate']:8.1%} | "
                  f"{r['reuse_precision']:8.1%} | {r['cross_lang_true']:12} | "
                  f"{r['false']}")
            if r["wrong"]:
                print(f"        merges: {[(a, b) for a, b, c in r['wrong'][:3]]}")
        print("\n  'cross-lang-kept' = non-English re-asks served from cache of the\n"
              "  SAME intent (true cross-lingual reuse).")
        print("  Sweet spot = highest epsilon with precision at 100%.")


if __name__ == "__main__":
    main()

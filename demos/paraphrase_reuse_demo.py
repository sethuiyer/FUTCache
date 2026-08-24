#!/usr/bin/env python3
"""demos/paraphrase_reuse_demo.py — real-text semantic-reuse test.

Tests PackCache on a concrete, human-judgeable workload: a 45-sentence
support corpus and 10 questions, each with alternate ways to ask the same
thing. Uses a real embedding model (hotchpotch/bekko-embedding-v1-a8m, 384-d,
L2-normalised here) so paraphrase-vs-distinct distances are real, not
synthetic. Reports the honest metrics:

  * reuse_rate   = P(cache says HIT)  = hits / queries
  * reuse_precision = P(reuse is CORRECT | cache says HIT)
                   = true_reuse / (true_reuse + false_reuse)

A "true" reuse returns the payload of the SAME intent; a "false" reuse
returns a different intent's payload (an epsilon too large merged two
intents). The boundary margin between paraphrase distance and cross-intent
distance determines where the cache is useful.

Run:  python demos/paraphrase_reuse_demo.py [--epsilon 0.45]
"""

from __future__ import annotations

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from futcache import PackCache

MODEL_NAME = "hotchpotch/bekko-embedding-v1-a8m"

# 45-sentence support corpus (the background knowledge). Phase 1 ingests it.
CORPUS = [
    "You can reset your password from the account settings page.",
    "Password resets require a confirmation link sent to your email.",
    "Choose a password with at least eight characters and a number.",
    "A password reset link expires after sixty minutes.",
    "Contact support if the reset email does not arrive.",
    "To sign out, open the user menu and select logout.",
    "Logging out clears your local session on the device.",
    "You stay signed in on trusted devices by default.",
    "Ending a session on a shared computer is recommended.",
    "Subscriptions are managed in the billing section.",
    "You can cancel a subscription at the end of the billing cycle.",
    "Cancelling stops renewals but keeps access until the period ends.",
    "A confirmation email is sent when a subscription is cancelled.",
    "Downgrades take effect on the next billing date.",
    "Refunds are issued within five to ten business days.",
    "Eligible refunds are processed back to your original payment method.",
    "Contact billing to request a refund for a recent charge.",
    "Pro-rated refunds apply to annual plans.",
    "API keys are generated under the developer settings tab.",
    "Keep your API key secret; do not commit it to a repository.",
    "You can create multiple API keys for different environments.",
    "Rotate an API key if you suspect it has leaked.",
    "Change your email address from the profile settings.",
    "A verification email is sent to the new address.",
    "Updating your email does not affect your account data.",
    "Invoices are available for download in the billing history.",
    "Each invoice includes a line-by-line summary of charges.",
    "Receipts are emailed at the end of each billing cycle.",
    "A mobile app is available for iOS and Android.",
    "The mobile app supports the same features as the web version.",
    "Push notifications are configurable in the mobile settings.",
    "There is a free tier with a monthly usage limit.",
    "Paid plans unlock higher usage caps and priority support.",
    "You are billed monthly, with an option to pay annually.",
    "Two-factor authentication adds a second step at sign in.",
    "Enable 2FA from the security settings on your account.",
    "Authenticator apps generate the verification code.",
    "Backup codes let you sign in if your device is lost.",
    "You can invite teammates from the organization settings.",
    "Team roles control who can change billing details.",
    "Audit logs record changes made to your workspace.",
    "Data is encrypted at rest and in transit.",
    "You can export your data in a CSV file.",
    "The help center has step-by-step guides for common tasks.",
    "Session length and timeout are configurable by admins.",
]

# 10 questions, each with alternate ways to ask the same thing.
# key = canonical intent; value = [main phrasing, alt 1, alt 2]
QUESTIONS = {
    "password_reset": [
        "How do I reset my password?",
        "What is the procedure to change my password?",
        "I forgot my password, how can I get a new one?",
    ],
    "sign_out": [
        "How do I log out of my account?",
        "What is the way to sign out?",
        "How can I end my current session?",
    ],
    "cancel_subscription": [
        "How do I cancel my subscription?",
        "What is the process to cancel my plan?",
        "I want to stop my subscription, how?",
    ],
    "refund": [
        "How do I get a refund?",
        "What is the refund process?",
        "Can I get my money back, and how?",
    ],
    "api_key": [
        "Where do I find my API key?",
        "How do I generate an API key?",
        "How do I get my API credentials?",
    ],
    "change_email": [
        "How do I change my email address?",
        "What is the way to update my email?",
        "How can I modify the email on my account?",
    ],
    "invoice": [
        "Where can I download my invoice?",
        "How do I get a copy of my bill?",
        "How do I access my billing statement?",
    ],
    "mobile_app": [
        "Is there a mobile app?",
        "Do you have an iOS or Android app?",
        "Can I use this from my phone?",
    ],
    "pricing": [
        "How much does it cost?",
        "What are the pricing tiers?",
        "What is the monthly price?",
    ],
    "two_factor": [
        "How do I enable 2FA?",
        "What is the process to turn on two-factor authentication?",
        "How do I set up multifactor authentication?",
    ],
}


def embed(texts, model) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float64)


def build_queries():
    """Flatten QUESTIONS into (intent, phrasing) in a realistic mixed order."""
    pairs = [(intent, phrasing)
             for intent, phrasings in QUESTIONS.items()
             for phrasing in phrasings]
    rng = np.random.default_rng(1234)
    rng.shuffle(pairs)   # interleave intents like real traffic
    return pairs


def run_epsilon(queries, model, epsilon, dim=384) -> dict:
    cache = PackCache(dim, epsilon, distance="cosine",
                      domain_min=-1.0, domain_max=1.0, backend="vptree")
    true_reuse = false_reuse = misses = 0
    wrong = []
    for intent, phrasing in queries:
        vec = embed([phrasing], model)[0]

        def compute(_p, intent=intent):
            # stand-in for "answer": the intent label (or a synthetic answer)
            return intent.encode()

        answer, res = cache.get_or_compute(vec, compute)
        if res.is_novel:
            misses += 1
            continue
        # served from cache: is the returned payload the SAME intent?
        payload = bytes(answer).decode()
        if payload == intent:
            true_reuse += 1
        else:
            false_reuse += 1
            wrong.append((intent, payload))

    hits = true_reuse + false_reuse
    total = len(queries)
    reuse_rate = hits / total if total else 0.0
    reuse_precision = (true_reuse / hits) if hits else 0.0
    return {
        "epsilon": epsilon, "total": total, "misses": misses,
        "hits": hits, "true_reuse": true_reuse, "false_reuse": false_reuse,
        "reuse_rate": reuse_rate, "reuse_precision": reuse_precision,
        "wrong": wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epsilon", type=float, default=0.45,
                    help="run a single epsilon (sweep if omitted)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep epsilon and print the frontier")
    args = ap.parse_args()

    print("=" * 72)
    print("FUTCache semantic-reuse test (real embeddings)")
    print("=" * 72)
    print(f"  model: {args.model} (384-d, L2-normalised)")
    print(f"  corpus: {len(CORPUS)} sentences   questions: "
          f"{len(QUESTIONS)} intents x 3 phrasings")
    print("  loading model ...")
    model = SentenceTransformer(args.model)

    def run_single(eps):
        q = build_queries()
        r = run_epsilon(q, model, eps)
        print(f"\n  epsilon={eps:.2f}  queries={r['total']}  "
              f"hits={r['hits']}  misses={r['misses']}")
        print(f"    true reuse={r['true_reuse']}  false reuse={r['false_reuse']}")
        print(f"    reuse_rate={r['reuse_rate']:.1%}  "
              f"reuse_precision={r['reuse_precision']:.1%}")
        if r["wrong"]:
            print(f"    cross-intent merges: {[(a, b) for a, b in r['wrong'][:4]]}")
        return r

    if args.epsilon > 0 and not args.sweep:
        run_single(args.epsilon)
        print("\n  (run with --sweep to see the epsilon frontier)")
    else:
        epsilons = [0.25, 0.35, 0.42, 0.45, 0.5, 0.55, 0.6, 0.7]
        print(f"\n[ Phase 1 ] ingest {len(CORPUS)}-sentence corpus")
        cache = PackCache(384, 0.45, distance="cosine",
                          domain_min=-1.0, domain_max=1.0, backend="vptree")
        cor = embed(CORPUS, model)
        for v in cor:
            cache.observe(v)
        print(f"  corpus -> {len(cache)} representative(s) "
              f"({len(CORPUS)/max(len(cache),1):.1f}x dedup within corpus)")

        print("\n[ Phase 2 ] question reuse frontier")
        print("  epsilon | reuse_rate | reuse_precision | float(cross-merge)")
        print("  --------|------------|-----------------|-----------------")
        for eps in epsilons:
            r = run_single(eps)
            print(f"  {eps:6.2f} | {r['reuse_rate']:8.1%} | "
                  f"{r['reuse_precision']:13.1%} | {r['false_reuse']}")
        print("\n  The useful regime is where reuse_precision stays high while")
        print("  reuse_rate rises. Above it, two intents start merging (false reuse).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""demos/semantic_assistant_integration.py — wire PackCache into a real app.

A realistic end-to-end integration: a support assistant that answers user
questions via a (mock) LLM over a 45-sentence knowledge base, with a
`PackCache` semantic cache in front of it. A repeat or rephrased question is
served from cache (bypassing the LLM) via `get_or_compute`, with TTL and an
LRU cap.

Measures (honest, on real embeddings):
  * cache hit rate and LLM calls avoided
  * cost saved (input+output tokens at a stated price)
  * latency: cached vs cold
  * precision: P(a served answer belongs to the SAME intent) -- a cross-intent
    merge (epsilon too large) serves the WRONG intent's answer

Run:  python demos/semantic_assistant_integration.py [--epsilon 0.45]
                                                       [--repeats 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from futcache import PackCache

MODEL = "hotchpotch/bekko-embedding-v1-a8m"
COST_PER_1M_OUTPUT = 15.00
COST_PER_1M_INPUT = 3.00
AVG_IN = 200
AVG_OUT = 120
LLM_LATENCY_S = 0.20

CORPUS = [
    "You can reset your password from the account settings page.",
    "Password resets require a confirmation link sent to your email.",
    "Choose a password with at least eight characters and a number.",
    "A password reset link expires after sixty minutes.",
    "To sign out, open the user menu and select logout.",
    "Logging out clears your local session on the device.",
    "Subscriptions are managed in the billing section.",
    "You can cancel a subscription at the end of the billing cycle.",
    "Cancelling stops renewals but keeps access until the period ends.",
    "Refunds are issued within five to ten business days.",
    "Eligible refunds are processed back to your original payment method.",
    "API keys are generated under the developer settings tab.",
    "Keep your API key secret; do not commit it to a repository.",
    "Change your email address from the profile settings.",
    "A verification email is sent to the new address.",
    "Invoices are available for download in the billing history.",
    "Each invoice includes a line-by-line summary of charges.",
    "A mobile app is available for iOS and Android.",
    "The mobile app supports the same features as the web version.",
    "There is a free tier with a monthly usage limit.",
    "Paid plans unlock higher usage caps and priority support.",
    "Two-factor authentication adds a second step at sign in.",
    "Enable 2FA from the security settings on your account.",
    "Authenticator apps generate the verification code.",
    "You can invite teammates from the organization settings.",
    "Audit logs record changes made to your workspace.",
    "Data is encrypted at rest and in transit.",
    "You can export your data in a CSV file.",
    "The help center has step-by-step guides for common tasks.",
]

# intent -> (canonical question, list of rephrasings)
QUESTIONS = {
    "password_reset": ["How do I reset my password?",
                       "What is the procedure to change my password?",
                       "I forgot my password, how can I get a new one?"],
    "sign_out": ["How do I log out of my account?",
                 "What is the way to sign out?",
                 "How can I end my session?"],
    "cancel_subscription": ["How do I cancel my subscription?",
                            "What is the process to cancel my plan?",
                            "I want to stop my subscription, how?"],
    "refund": ["How do I get a refund?",
               "What is the refund process?",
               "Can I get my money back, and how?"],
    "api_key": ["Where do I find my API key?",
                "How do I generate an API key?",
                "How do I get my API credentials?"],
    "change_email": ["How do I change my email address?",
                     "What is the way to update my email?",
                     "How can I modify the email on my account?"],
    "invoice": ["Where can I download my invoice?",
                "How do I get a copy of my bill?",
                "How do I access my billing statement?"],
    "mobile_app": ["Is there a mobile app?",
                   "Do you have an iOS or Android app?",
                   "Can I use this from my phone?"],
    "pricing": ["How much does it cost?",
                "What are the pricing tiers?",
                "What is the monthly price?"],
    "two_factor": ["How do I enable 2FA?",
                   "How do I start two-factor authentication?",
                   "How do I set up multifactor authentication?"],
}


class SupportAssistant:
    """A PackCache-backed semantic answer cache in front of an LLM."""

    def __init__(self, model, dim=384, eps=0.45, ttl=0.0, max_entries=0):
        self.model = model
        self.eps = eps
        self.corpus_vecs = model.encode(CORPUS, normalize_embeddings=True)
        self.cache = PackCache(dim, eps, distance="cosine",
                               domain_min=-1.0, domain_max=1.0,
                               backend="vptree", ttl=ttl,
                               max_entries=max_entries)

    def _answer(self, question, intent):
        """The 'LLM': retrieve the best knowledge sentence and answer."""
        time.sleep(LLM_LATENCY_S)  # simulated network + generation latency
        qv = self.model.encode([question], normalize_embeddings=True)[0]
        sim = self.corpus_vecs @ qv
        best = int(np.argmax(sim))
        # embed the intent so a served answer can be attributed to its intent
        return f"<intent:{intent}> {CORPUS[best]}"

    def ask(self, question, intent):
        """Return (answer, was_cached). Uses get_or_compute so a repeat or
        rephrase is served from cache and skips the LLM."""
        qv = self.model.encode([question], normalize_embeddings=True)[0]

        def compute(_p, intent=intent):
            return self._answer(question, intent).encode()

        answer, res = self.cache.get_or_compute(qv, compute, radius=self.eps)
        return answer.decode(), (not res.is_novel), res


def decode_intent(answer):
    return answer.split(">")[0].replace("<intent:", "")


def evaluate(assistant, stream):
    """Run the stream and return a confusion matrix at the query level.

    Ground truth: a query is NOVEL if its intent has not been seen yet in the
    stream (must call the LLM), REDUNDANT if it's a repeat/rephrase (should
    reuse). The cache decides HIT (served) or MISS (LLM called).

      TP = HIT and served the correct intent   (correct reuse)
      FP = HIT but served the WRONG intent     (cross-intent merge)
      FN = MISS but the intent was seen        (missed reuse -> wasted LLM)
      TN = MISS and genuinely novel            (correct new-intent LLM call)
    """
    seen = set()
    tp = fp = fn = tn = 0
    for intent, question in stream:
        first = intent not in seen
        seen.add(intent)
        answer, was_cached, _ = assistant.ask(question, intent)
        served = decode_intent(answer)
        if was_cached and served == intent:
            tp += 1
        elif was_cached:
            fp += 1
        elif first:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize(c):
    total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    reuse = c["tp"] + c["fp"]
    precision = c["tp"] / reuse if reuse else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "total": total, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        "tn": c["tn"], "reuse_rate": reuse / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epsilon", type=float, default=0.45)
    ap.add_argument("--ttl", type=float, default=0.0)
    ap.add_argument("--max-entries", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=4,
                    help="times to repeat each question stream")
    args = ap.parse_args()

    print("=" * 72)
    print("PackCache integration: semantic support assistant")
    print("=" * 72)
    print(f"  model: {MODEL} (384-d, L2-normalised)")
    print(f"  knowledge base: {len(CORPUS)} sentences; "
          f"{len(QUESTIONS)} intents x 3 phrasings")
    print(f"  epsilon={args.epsilon}  ttl={args.ttl}s  "
          f"max_entries={args.max_entries}  repeats={args.repeats}")
    print("  loading model ...")
    model = SentenceTransformer(MODEL)
    assistant = SupportAssistant(model, eps=args.epsilon, ttl=args.ttl,
                                 max_entries=args.max_entries)

    cost_per_call = (AVG_IN * COST_PER_1M_INPUT +
                     AVG_OUT * COST_PER_1M_OUTPUT) / 1e6

    # Build the stream: shuffle phrasings, repeat the whole thing.
    phrasings = [(intent, q) for intent, qs in QUESTIONS.items()
                 for q in qs]
    rng = np.random.default_rng(11)
    order = rng.permutation(len(phrasings))
    stream = [phrasings[i] for i in order]
    stream = stream * args.repeats
    n = len(stream)

    # ---- single epsilon: full confusion matrix
    def build(eps):
        return SupportAssistant(model, eps=eps, ttl=args.ttl,
                                max_entries=args.max_entries)

    # measure latency/cost at the requested epsilon
    ass = build(args.epsilon)
    c = evaluate(ass, stream)
    s = summarize(c)
    naive_cost = n * cost_per_call
    cached_cost = (c["tn"] + c["fn"]) * cost_per_call
    saved = naive_cost - cached_cost

    print("\n[ Confusion matrix @ epsilon=%.2f ]" % args.epsilon)
    print(f"  TN (correct novel, LLM called) = {c['tn']}")
    print(f"  TP (correct reuse, served)     = {c['tp']}")
    print(f"  FN (missed reuse, wasted LLM)  = {c['fn']}")
    print(f"  FP (WRONG intent served)       = {c['fp']}   <-- false positives")
    print(f"\n  reuse_rate  = {s['reuse_rate']:.1%}")
    print(f"  precision   = {s['precision']:.1%}   (TP/(TP+FP))")
    print(f"  recall      = {s['recall']:.1%}   (TP/(TP+FN))")
    print(f"  F1          = {s['f1']:.3f}")
    print(f"  cost: naive=${naive_cost:.2f}  with cache=${cached_cost:.2f}  "
          f"saved=${saved:.2f} ({saved/naive_cost:.1%})")

    # ---- epsilon frontier (shows how false positives rise)
    print("\n[ epsilon frontier: FP rises as epsilon grows ]")
    print("  eps   | reuse | precision | recall | F1    | FP | FN | TN")
    print("  ------|-------|-----------|--------|-------|----|----|----")
    for eps in (0.35, 0.45, 0.55, 0.65, 0.72):
        s2 = summarize(evaluate(build(eps), stream))
        print(f"  {eps:.2f} | {s2['reuse_rate']:5.1%} | "
              f"{s2['precision']:8.1%} | {s2['recall']:6.1%} | "
              f"{s2['f1']:.3f} | {s2['fp']:2d} | {s2['fn']:2d} | {s2['tn']:2d}")

    print("\n[ Example transcript ]")
    for intent, question in stream[:6]:
        answer, was_cached, _ = ass.ask(question, intent)
        tag = "CACHE" if was_cached else "LLM  "
        print(f"  [{tag}] {question[:52]:54s} -> {answer.split('> ')[-1][:46]}")
    print("\n  FP = a cache hit that served the WRONG intent's answer (epsilon too")
    print("      large merged two intents); FN = a rephrase that missed the cache")
    print("      and wasted an LLM call (epsilon too small). Both are real costs.")


if __name__ == "__main__":
    main()
